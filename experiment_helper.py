import argparse
import os
from typing import List

import wandb
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from rec_sys.protomf_dataset import get_protorecdataset_dataloader
from rec_sys.tester import Tester
from rec_sys.trainer import Trainer
from utilities.consts import NEG_VAL, OPTIMIZING_METRIC, SEED_LIST, SINGLE_SEED, NUM_SAMPLES, WANDB_API_KEY, \
    PROJECT_NAME, DATA_PATH, NUM_WORKERS, CPU_PER_TRIAL, GPU_PER_TRIAL, MAX_PATIENCE, \
    MIN_DELTA_DEFAULTS, MIN_DELTA_FALLBACK
from utilities.utils import reproducible, generate_id


def _resolve(resource_cfg, key, default):
    """Get a value from resource_cfg dict, falling back to a default."""
    if resource_cfg:
        return resource_cfg.get(key, default)
    return default


def load_data(conf: argparse.Namespace, is_train: bool = True, num_workers: int = NUM_WORKERS):
    if is_train:
        train_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='train',
            n_neg=conf.neg_train,
            neg_strategy=conf.train_neg_strategy,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=5 if num_workers > 0 else None
        )

        val_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='val',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=num_workers
        )

        return {'train_loader': train_loader, 'val_loader': val_loader}
    else:

        test_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='test',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=num_workers
        )

        return {'test_loader': test_loader}


def start_training(config):
    config = argparse.Namespace(**config)
    print(config)

    # Initialize W&B directly in the trial process for live metrics + console capture.
    # W&B metadata is passed through the config dict (not env vars) so each trial
    # gets the correct identity even when multiple combos run in the same process.
    wandb_project = getattr(config, '_wandb_project', None)
    if wandb_project:
        try:
            from ray.tune import get_context as get_tune_context
            trial_id = (get_tune_context().get_trial_name() or 'trial').split('_')[-1]
        except (ImportError, AttributeError):
            from ray.train import get_context
            trial_id = (get_context().get_trial_name() or 'trial').split('_')[-1]
        model_tag = getattr(config, '_wandb_model', '')
        dataset_tag = getattr(config, '_wandb_dataset', '')
        wandb_group = getattr(config, '_wandb_group', '')
        wandb_tags = getattr(config, '_wandb_tags', [])
        seed = config.seed

        # Flatten key nested hyperparams so they appear as sortable W&B columns
        ft = vars(config).get('ft_ext_param', {})
        item_ft = ft.get('item_ft_ext_param', {})
        user_ft = ft.get('user_ft_ext_param', {})
        flat_config = {
            k: v for k, v in vars(config).items() if not k.startswith('_wandb_')
        }
        flat_config.update({
            'embedding_dim': ft.get('embedding_dim'),
            'item_n_prototypes': item_ft.get('n_prototypes'),
            'item_sim_proto_weight': item_ft.get('sim_proto_weight'),
            'item_sim_batch_weight': item_ft.get('sim_batch_weight'),
            'user_n_prototypes': user_ft.get('n_prototypes'),
            'user_sim_proto_weight': user_ft.get('sim_proto_weight'),
            'user_sim_batch_weight': user_ft.get('sim_batch_weight'),
        })

        wandb.init(
            project=wandb_project,
            group=wandb_group,
            tags=wandb_tags,
            name=f"{model_tag}_{dataset_tag}_s{seed}_{trial_id}",
            job_type='train/val',
            config=flat_config,
            settings=wandb.Settings(console='auto'),
            reinit=True,
        )

    num_workers = getattr(config, '_num_workers', NUM_WORKERS)
    data_loaders_dict = load_data(config, num_workers=num_workers)

    reproducible(config.seed)

    trainer = Trainer(data_loaders_dict['train_loader'], data_loaders_dict['val_loader'], config)

    try:
        trainer.run()
    finally:
        wandb.finish()


def start_testing(config, model_load_path: str):
    config = argparse.Namespace(**config)
    print(config)

    num_workers = getattr(config, '_num_workers', NUM_WORKERS)
    data_loaders_dict = load_data(config, is_train=False, num_workers=num_workers)

    reproducible(config.seed)

    tester = Tester(data_loaders_dict['test_loader'], config, model_load_path)

    metric_values = tester.test()
    return metric_values


def start_hyper(conf: dict, model: str, dataset: str, seed: int = SINGLE_SEED,
                resource_cfg: dict = None):
    """
    Run hyperparameter optimization for a single model/dataset/seed combo.

    resource_cfg keys (all optional, fall back to consts.py defaults):
        gpu_per_trial, cpu_per_trial, num_samples, num_workers,
        grace_period, patience, min_delta, optimizing_metric, n_epochs, asha
    """
    # Resolve execution-level parameters from resource_cfg or global defaults
    gpu_per_trial = _resolve(resource_cfg, 'gpu_per_trial', GPU_PER_TRIAL)
    cpu_per_trial = _resolve(resource_cfg, 'cpu_per_trial', CPU_PER_TRIAL)
    num_samples_default = _resolve(resource_cfg, 'num_samples', NUM_SAMPLES)
    num_workers = _resolve(resource_cfg, 'num_workers', NUM_WORKERS)
    grace_period = _resolve(resource_cfg, 'grace_period', 4)
    patience = _resolve(resource_cfg, 'patience', MAX_PATIENCE)
    optimizing_metric = _resolve(resource_cfg, 'optimizing_metric', OPTIMIZING_METRIC)
    # min_delta default is metric-aware: fall back to per-metric table, then global fallback
    min_delta = _resolve(resource_cfg, 'min_delta', None)
    if min_delta is None:
        min_delta = MIN_DELTA_DEFAULTS.get(optimizing_metric, MIN_DELTA_FALLBACK)
    n_epochs = _resolve(resource_cfg, 'n_epochs', None)
    asha = _resolve(resource_cfg, 'asha', True)
    extra_wandb_tags = _resolve(resource_cfg, 'wandb_tags', [])

    print('Starting Hyperparameter Optimization')
    print(f'Seed is {seed}')
    print(f'Resources: gpu_per_trial={gpu_per_trial}, cpu_per_trial={cpu_per_trial}, '
          f'num_workers={num_workers}, grace_period={grace_period}, patience={patience}, '
          f'min_delta={min_delta}, asha={asha}')

    # Initialize Ray with scratch temp dir on HPC (avoids /tmp quota issues)
    import ray
    if not ray.is_initialized():
        ray_tmpdir = os.environ.get('RAY_TMPDIR', None)
        if ray_tmpdir:
            try:
                ray.init(_temp_dir=ray_tmpdir)
            except TypeError:
                # Ray < 2.0 doesn't support _temp_dir
                ray.init()
        else:
            ray.init()

    # Search Algorithm — skip if num_samples=1 (debug/fixed config with no search space)
    num_samples = conf.pop('num_samples', num_samples_default)
    search_alg = HyperOptSearch(random_state_seed=seed) if num_samples > 1 else None

    scheduler = ASHAScheduler(grace_period=grace_period) if asha else None

    # Hostname
    import platform
    host_name = platform.node()[:2]

    # Dataset
    data_path = DATA_PATH
    conf['data_path'] = os.path.join(data_path, dataset)

    # Seed
    conf['seed'] = seed

    # Inject execution-level overrides into config (passed through to trials)
    if n_epochs is not None:
        conf['n_epochs'] = n_epochs
    conf['_num_workers'] = num_workers
    conf['_max_patience'] = patience
    conf['_min_delta'] = min_delta
    conf['_optimizing_metric'] = optimizing_metric

    # W&B metadata — passed through config dict so each trial gets correct identity,
    # even when run_combo.py chains multiple combos in the same process.
    # Keys prefixed with _wandb_ to avoid polluting hyperparameter space.
    group_name = f'{model}_{dataset}_{host_name}_{seed}'
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    conf['_wandb_project'] = PROJECT_NAME
    conf['_wandb_group'] = f'{model}_{dataset}'
    conf['_wandb_tags'] = [model, dataset, f'seed:{seed}', 'replication'] + extra_wandb_tags
    conf['_wandb_model'] = model
    conf['_wandb_dataset'] = dataset

    tune.register_trainable(group_name, start_training)
    tune_kwargs = dict(
        config=conf,
        name=generate_id(prefix=group_name),
        resources_per_trial={'gpu': gpu_per_trial, 'cpu': cpu_per_trial},
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=num_samples,
        metric=optimizing_metric,
        mode='max',
        trial_dirname_creator=lambda trial: trial.trial_id,
    )
    # Direct Ray results to scratch on HPC
    ray_results_dir = os.environ.get('RAY_RESULTS_DIR', None)
    if ray_results_dir:
        tune_kwargs['storage_path'] = ray_results_dir
    analysis = tune.run(group_name, **tune_kwargs)
    metric_name = optimizing_metric
    best_trial = analysis.get_best_trial(metric_name, 'max', scope='all')
    best_trial_config = best_trial.config
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric_name, 'max')
    best_trial_checkpoint = os.path.join(best_checkpoint.to_directory(), 'best_model.pth')

    # Extract best validation metrics (at the checkpoint epoch, not last epoch)
    val_metrics = {}
    try:
        trial_df = analysis.trial_dataframes[best_trial.logdir]
        best_idx = trial_df[metric_name].idxmax()
        best_row = trial_df.loc[best_idx]
        val_metrics = {k: float(best_row[k]) for k in best_row.index
                       if k.startswith(('hit_ratio@', 'ndcg@', 'val_loss'))}
    except Exception:
        val_metrics = {k: v for k, v in best_trial.last_result.items()
                       if isinstance(v, (int, float)) and k.startswith(('hit_ratio@', 'ndcg@', 'val_loss'))}

    # Trial stats
    num_completed = len([t for t in analysis.trials if t.status == 'TERMINATED'])
    num_total = len(analysis.trials)

    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT_NAME, group=f'{model}_{dataset}', config=best_trial_config,
               name=f'{model}_{dataset}_s{seed}_test', force=True,
               job_type='test', tags=[model, dataset, f'seed:{seed}', 'replication'] + extra_wandb_tags)
    test_metrics = start_testing(best_trial_config, best_trial_checkpoint)
    wandb.finish()

    return {
        'test_metrics': test_metrics,
        'val_metrics': val_metrics,
        'best_config': best_trial_config,
        'best_checkpoint_path': best_trial_checkpoint,
        'num_samples': num_samples,
        'num_completed': num_completed,
        'num_total': num_total,
        'run_name': tune_kwargs['name'],
        'model': model,
        'dataset': dataset,
        'seed': seed,
    }


def start_multiple_hyper(conf: dict, model: str, dataset: str, seed_list: List = SEED_LIST):
    print('Starting Multi-Hyperparameter Optimization')
    print('seed_list is ', seed_list)
    results_list = []
    mean_values = dict()

    for seed in seed_list:
        results_list.append(start_hyper(conf, model, dataset, seed))

    test_metrics_list = [r['test_metrics'] for r in results_list]
    for key in test_metrics_list[0].keys():
        _sum = 0
        for metric_values in test_metrics_list:
            _sum += metric_values[key]
        _mean = _sum / len(test_metrics_list)

        mean_values[key] = _mean

    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT_NAME, group=f'{model}_{dataset}', name=f'{model}_{dataset}_aggr',
               force=True, job_type='aggregate', tags=[model, dataset, 'replication'])
    wandb.log(mean_values)
    wandb.finish()
