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
    PROJECT_NAME, DATA_PATH, NUM_WORKERS, CPU_PER_TRIAL, GPU_PER_TRIAL
from utilities.utils import reproducible, generate_id


def load_data(conf: argparse.Namespace, is_train: bool = True):
    if is_train:
        train_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='train',
            n_neg=conf.neg_train,
            neg_strategy=conf.train_neg_strategy,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            prefetch_factor=5 if NUM_WORKERS > 0 else None
        )

        val_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='val',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=NUM_WORKERS
        )

        return {'train_loader': train_loader, 'val_loader': val_loader}
    else:

        test_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='test',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=NUM_WORKERS
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
        from ray.train import get_context
        trial_id = (get_context().get_trial_name() or 'trial').split('_')[-1]  # short hash
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

    data_loaders_dict = load_data(config)

    reproducible(config.seed)

    trainer = Trainer(data_loaders_dict['train_loader'], data_loaders_dict['val_loader'], config)

    try:
        trainer.run()
    finally:
        wandb.finish()


def start_testing(config, model_load_path: str):
    config = argparse.Namespace(**config)
    print(config)

    data_loaders_dict = load_data(config, is_train=False)

    reproducible(config.seed)

    tester = Tester(data_loaders_dict['test_loader'], config, model_load_path)

    metric_values = tester.test()
    return metric_values


def start_hyper(conf: dict, model: str, dataset: str, seed: int = SINGLE_SEED):
    print('Starting Hyperparameter Optimization')
    print(f'Seed is {seed}')

    # Search Algorithm — skip if num_samples=1 (debug/fixed config with no search space)
    num_samples = conf.pop('num_samples', NUM_SAMPLES)
    search_alg = HyperOptSearch(random_state_seed=seed) if num_samples > 1 else None

    scheduler = ASHAScheduler(grace_period=4)

    # Hostname
    import platform
    host_name = platform.node()[:2]

    # Dataset
    data_path = DATA_PATH
    conf['data_path'] = os.path.join(data_path, dataset)

    # Seed
    conf['seed'] = seed

    # W&B metadata — passed through config dict so each trial gets correct identity,
    # even when run_combo.py chains multiple combos in the same process.
    # Keys prefixed with _wandb_ to avoid polluting hyperparameter space.
    group_name = f'{model}_{dataset}_{host_name}_{seed}'
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    conf['_wandb_project'] = PROJECT_NAME
    conf['_wandb_group'] = f'{model}_{dataset}'
    conf['_wandb_tags'] = [model, dataset, f'seed:{seed}', 'replication']
    conf['_wandb_model'] = model
    conf['_wandb_dataset'] = dataset

    tune.register_trainable(group_name, start_training)
    analysis = tune.run(
        group_name,
        config=conf,
        name=generate_id(prefix=group_name),
        resources_per_trial={'gpu': GPU_PER_TRIAL, 'cpu': CPU_PER_TRIAL},
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=num_samples,
        metric=OPTIMIZING_METRIC,
        mode='max',
        trial_dirname_creator=lambda trial: trial.trial_id,
    )
    metric_name = OPTIMIZING_METRIC
    best_trial = analysis.get_best_trial(metric_name, 'max', scope='all')
    best_trial_config = best_trial.config
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric_name, 'max')
    best_trial_checkpoint = os.path.join(best_checkpoint.to_directory(), 'best_model.pth')

    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT_NAME, group=f'{model}_{dataset}', config=best_trial_config,
               name=f'{model}_{dataset}_s{seed}_test', force=True,
               job_type='test', tags=[model, dataset, f'seed:{seed}', 'replication'])
    metric_values = start_testing(best_trial_config, best_trial_checkpoint)
    wandb.finish()
    return metric_values


def start_multiple_hyper(conf: dict, model: str, dataset: str, seed_list: List = SEED_LIST):
    print('Starting Multi-Hyperparameter Optimization')
    print('seed_list is ', seed_list)
    metric_values_list = []
    mean_values = dict()

    for seed in seed_list:
        metric_values_list.append(start_hyper(conf, model, dataset, seed))

    for key in metric_values_list[0].keys():
        _sum = 0
        for metric_values in metric_values_list:
            _sum += metric_values[key]
        _mean = _sum / len(metric_values_list)

        mean_values[key] = _mean

    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT_NAME, group=f'{model}_{dataset}', name=f'{model}_{dataset}_aggr',
               force=True, job_type='aggregate', tags=[model, dataset, 'replication'])
    wandb.log(mean_values)
    wandb.finish()
