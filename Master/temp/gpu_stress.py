"""
Single-trial GPU stress test: user_item_proto with maxed-out hyperparameters.
Runs 3 epochs to measure peak VRAM usage. Watch with monitor.ps1.
Usage: conda activate protomf && python Master/temp/gpu_stress.py
"""
import os
import sys

# Same env vars as start.py
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

from experiment_helper import start_hyper

# Worst-case config: user_item_proto with all values at search space maximums
stress_config = {
    'device': 'cuda',
    'n_epochs': 3,
    'num_samples': 1,
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256,
    'rec_sys_param': {'use_bias': 0},
    'neg_train': 50,
    'train_neg_strategy': 'popular',
    'loss_func_name': 'bce',
    'loss_func_aggr': 'mean',
    'batch_size': 512,
    'optim_param': {
        'optim': 'adam',
        'wd': 1e-3,
        'lr': 1e-3,
    },
    'ft_ext_param': {
        "ft_type": "prototypes_double_tie",
        'embedding_dim': 100,
        'item_ft_ext_param': {
            "ft_type": "prototypes_double_tie",
            'sim_proto_weight': 1.0,
            'sim_batch_weight': 1.0,
            'use_weight_matrix': False,
            'n_prototypes': 100,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
        },
        'user_ft_ext_param': {
            "ft_type": "prototypes_double_tie",
            'sim_proto_weight': 1.0,
            'sim_batch_weight': 1.0,
            'use_weight_matrix': False,
            'n_prototypes': 100,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
        },
    },
}

print("=" * 60)
print("GPU STRESS TEST — user_item_proto, maxed hyperparameters")
print("embedding_dim=100, n_prototypes=100, batch_size=512, neg_train=50")
print("Watch GPU with: pwsh Master\\scripts\\monitor.ps1")
print("=" * 60)

start_hyper(stress_config, 'gpu_stress', 'amazon2014')
