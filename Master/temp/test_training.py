"""
Bottom-up Windows training debug script.
Tests 3 progressive layers to isolate where training hangs.

Usage:
    python Master/temp/test_training.py --level 1 --dataset ml-1m   # Pure PyTorch
    python Master/temp/test_training.py --level 2 --dataset ml-1m   # + Ray Tune
    python Master/temp/test_training.py --level 3 --dataset ml-1m   # + Ray Tune + W&B
"""

import argparse
import os
import sys
import time

# Add repo root to path so imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)


# Fixed config — no tune.*() objects. Mirrors debug_hyper_params but with plain values.
def get_fixed_config(dataset):
    import torch
    data_path = os.path.join(REPO_ROOT, 'data', dataset)
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 3,
        'neg_train': 5,
        'train_neg_strategy': 'uniform',
        'eval_neg_strategy': 'uniform',
        'loss_func_name': 'bce',
        'loss_func_aggr': 'mean',
        'batch_size': 128,
        'val_batch_size': 256,
        'seed': 42,
        'data_path': data_path,
        'optim_param': {
            'optim': 'adam',
            'wd': 1e-3,
            'lr': 1e-3,
        },
        'rec_sys_param': {'use_bias': 0},
        'ft_ext_param': {
            'ft_type': 'prototypes_double_tie',
            'embedding_dim': 8,
            'item_ft_ext_param': {
                'ft_type': 'prototypes_double_tie',
                'sim_proto_weight': 1.0,
                'sim_batch_weight': 1.0,
                'use_weight_matrix': False,
                'n_prototypes': 3,
                'cosine_type': 'shifted',
                'reg_proto_type': 'max',
                'reg_batch_type': 'max',
            },
            'user_ft_ext_param': {
                'ft_type': 'prototypes_double_tie',
                'sim_proto_weight': 1.0,
                'sim_batch_weight': 1.0,
                'use_weight_matrix': False,
                'n_prototypes': 3,
                'cosine_type': 'shifted',
                'reg_proto_type': 'max',
                'reg_batch_type': 'max',
            },
        },
    }


def run_level_1(dataset):
    """Level 1: Pure PyTorch — no Ray, no W&B."""
    print('=' * 60)
    print('LEVEL 1: Pure PyTorch (no Ray, no W&B)')
    print('=' * 60)

    from rec_sys.protomf_dataset import get_protorecdataset_dataloader
    from rec_sys.trainer import Trainer
    from utilities.consts import NEG_VAL
    from utilities.utils import reproducible

    config = get_fixed_config(dataset)
    conf = argparse.Namespace(**config)

    reproducible(conf.seed)

    print(f'\nDevice: {conf.device}')
    print(f'Dataset: {dataset}')
    print(f'Loading data...')

    t0 = time.time()

    train_loader = get_protorecdataset_dataloader(
        data_path=conf.data_path,
        split_set='train',
        n_neg=conf.neg_train,
        neg_strategy=conf.train_neg_strategy,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = get_protorecdataset_dataloader(
        data_path=conf.data_path,
        split_set='val',
        n_neg=NEG_VAL,
        neg_strategy=conf.eval_neg_strategy,
        batch_size=conf.val_batch_size,
        num_workers=0,
    )

    print(f'Data loaded in {time.time() - t0:.1f}s')
    print(f'Train: {len(train_loader.dataset)} interactions, Val: {len(val_loader.dataset)} interactions')

    print('\nBuilding trainer (use_ray=False)...')
    trainer = Trainer(train_loader, val_loader, conf, use_ray=False)

    print('\nStarting training...')
    t0 = time.time()
    trainer.run()
    elapsed = time.time() - t0

    print(f'\nLEVEL 1 PASSED — {conf.n_epochs} epochs in {elapsed:.1f}s')
    return True


def run_level_2(dataset):
    """Level 2: Ray Tune (no W&B)."""
    print('=' * 60)
    print('LEVEL 2: Ray Tune (no W&B)')
    print('=' * 60)

    from ray import tune
    from utilities.utils import reproducible

    config = get_fixed_config(dataset)

    def train_fn(trial_config):
        import argparse
        from rec_sys.protomf_dataset import get_protorecdataset_dataloader
        from rec_sys.trainer import Trainer
        from utilities.consts import NEG_VAL
        from utilities.utils import reproducible

        conf = argparse.Namespace(**trial_config)
        reproducible(conf.seed)

        train_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='train',
            n_neg=conf.neg_train,
            neg_strategy=conf.train_neg_strategy,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='val',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=0,
        )

        trainer = Trainer(train_loader, val_loader, conf, use_ray=True)
        trainer.run()

    print(f'\nDevice: {config["device"]}')
    print(f'Dataset: {dataset}')
    print('\nStarting Ray Tune with num_samples=1, no callbacks...')

    t0 = time.time()
    analysis = tune.run(
        train_fn,
        config=config,
        num_samples=1,
        resources_per_trial={'gpu': 0.5 if config['device'] == 'cuda' else 0, 'cpu': 1},
        callbacks=[],  # No W&B
        metric='hit_ratio@10',
        mode='max',
        verbose=2,
    )
    elapsed = time.time() - t0

    best_trial = analysis.best_trial
    print(f'\nBest trial metric: {analysis.best_result}')
    print(f'\nLEVEL 2 PASSED — Ray Tune completed in {elapsed:.1f}s')
    return True


def run_level_3(dataset):
    """Level 3: Ray Tune + W&B."""
    print('=' * 60)
    print('LEVEL 3: Ray Tune + W&B')
    print('=' * 60)

    from ray import tune
    from ray.air.integrations.wandb import WandbLoggerCallback
    from utilities.consts import WANDB_API_KEY, PROJECT_NAME

    config = get_fixed_config(dataset)

    def train_fn(trial_config):
        import argparse
        import wandb
        from rec_sys.protomf_dataset import get_protorecdataset_dataloader
        from rec_sys.trainer import Trainer
        from utilities.consts import NEG_VAL
        from utilities.utils import reproducible

        conf = argparse.Namespace(**trial_config)
        reproducible(conf.seed)

        train_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='train',
            n_neg=conf.neg_train,
            neg_strategy=conf.train_neg_strategy,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='val',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=0,
        )

        trainer = Trainer(train_loader, val_loader, conf, use_ray=True)
        trainer.run()
        wandb.finish()

    callback = WandbLoggerCallback(
        project=PROJECT_NAME,
        log_config=True,
        api_key=WANDB_API_KEY,
        reinit=True,
        force=True,
        job_type='debug_test',
        tags=['debug', 'level3', dataset],
    )

    print(f'\nDevice: {config["device"]}')
    print(f'Dataset: {dataset}')
    print('\nStarting Ray Tune with W&B callback...')

    t0 = time.time()
    analysis = tune.run(
        train_fn,
        config=config,
        num_samples=1,
        resources_per_trial={'gpu': 0.5 if config['device'] == 'cuda' else 0, 'cpu': 1},
        callbacks=[callback],
        metric='hit_ratio@10',
        mode='max',
        verbose=2,
    )
    elapsed = time.time() - t0

    print(f'\nBest trial metric: {analysis.best_result}')
    print(f'\nLEVEL 3 PASSED — Ray Tune + W&B completed in {elapsed:.1f}s')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bottom-up Windows training debug')
    parser.add_argument('--level', type=int, required=True, choices=[1, 2, 3],
                        help='Test level: 1=PyTorch only, 2=+Ray, 3=+W&B')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset to use (default: ml-1m)')
    args = parser.parse_args()

    print(f'\n{"=" * 60}')
    print(f'Windows Training Debug — Level {args.level}')
    print(f'{"=" * 60}\n')

    runners = {1: run_level_1, 2: run_level_2, 3: run_level_3}

    try:
        success = runners[args.level](args.dataset)
        if success:
            print(f'\n*** LEVEL {args.level} PASSED ***')
    except Exception as e:
        print(f'\n*** LEVEL {args.level} FAILED ***')
        print(f'Error: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
