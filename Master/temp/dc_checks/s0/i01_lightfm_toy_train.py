# i01 — lightfm end-to-end toy train: Trainer(use_ray=False) + Tester over a synthetic
# dataset directory, both configs (tags / tags+ids). Exercises the real training path:
# inject_feature_ids in _build_model, negative sampling, the NEW Evaluator counted-rows
# assert (val/test rows == n_users), checkpointing, and strict Tester load.
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from _harness import make_check

from rec_sys.protomf_dataset import get_protorecdataset_dataloader
from rec_sys.tester import Tester
from rec_sys.trainer import Trainer
from utilities.utils import reproducible

check, fails = make_check()

tmp = tempfile.mkdtemp(prefix='s0_i01_')

# --- synthetic dataset dir: 40 users, 80 items; per user ≥3 train + 1 val + 1 test ---
N_USERS, N_ITEMS = 40, 80
rng = np.random.default_rng(38210573)

with open(os.path.join(tmp, 'user_ids.csv'), 'w') as f:
    f.write('user_id\n' + '\n'.join(str(u) for u in range(N_USERS)) + '\n')
with open(os.path.join(tmp, 'item_ids.csv'), 'w') as f:
    f.write('item_id\n' + '\n'.join(str(i) for i in range(N_ITEMS)) + '\n')

colors = ['blue', 'green', 'red', 'black']
shapes = ['square', 'circle', 'stripe']
with open(os.path.join(tmp, 'item_features.csv'), 'w') as f:
    f.write('item_id,color,shape\n')
    for i in range(N_ITEMS):
        f.write(f'{i},{colors[i % 4]},{shapes[i % 3]}\n')

train_rows, val_rows, test_rows = [], [], []
for u in range(N_USERS):
    items = rng.choice(N_ITEMS, size=6, replace=False)
    for it in items[:4]:
        train_rows.append((u, it))
    val_rows.append((u, items[4]))
    test_rows.append((u, items[5]))
for name, rows in [('train', train_rows), ('val', val_rows), ('test', test_rows)]:
    with open(os.path.join(tmp, f'listening_history_{name}.csv'), 'w') as f:
        f.write('user_id,item_id\n')
        for u, it in rows:
            f.write(f'{u},{it}\n')

EVAL_NEG = 50  # 1+50 candidates ≥ K=50; 80 - ≤6 consumed leaves ≥74 candidates — feasible


def run_conf(use_id):
    conf = argparse.Namespace(
        data_path=tmp, seed=38210573, device='cpu',
        n_epochs=10, neg_train=4, train_neg_strategy='uniform', eval_neg_strategy='uniform',
        loss_func_name='bce', loss_func_aggr='mean', batch_size=32, val_batch_size=64,
        rec_sys_param={'use_bias': 0},
        optim_param={'optim': 'adam', 'lr': 1e-2, 'wd': 1e-4},
        ft_ext_param={
            'ft_type': 'lightfm', 'embedding_dim': 16,
            'user_ft_ext_param': {'ft_type': 'embedding'},
            'item_ft_ext_param': {'ft_type': 'lightfm', 'use_id_feature': use_id,
                                  'feature_fields': ['color', 'shape']},
        })
    ckpt_dir = tempfile.mkdtemp(prefix=f'ckpt_{use_id}_', dir=tmp)

    train_loader = get_protorecdataset_dataloader(
        data_path=tmp, split_set='train', n_neg=conf.neg_train,
        neg_strategy='uniform', batch_size=conf.batch_size, shuffle=True, num_workers=0)
    val_loader = get_protorecdataset_dataloader(
        data_path=tmp, split_set='val', n_neg=EVAL_NEG,
        neg_strategy='uniform', batch_size=conf.val_batch_size, num_workers=0)

    reproducible(conf.seed)
    trainer = Trainer(train_loader, val_loader, conf, use_ray=False)
    trainer.run(checkpoint_dir=ckpt_dir)
    ckpt = os.path.join(ckpt_dir, 'best_model.pth')

    test_loader = get_protorecdataset_dataloader(
        data_path=tmp, split_set='test', n_neg=EVAL_NEG,
        neg_strategy='uniform', batch_size=conf.val_batch_size, num_workers=0)
    reproducible(conf.seed)
    tester = Tester(test_loader, conf, ckpt)
    metrics = tester.test()
    return ckpt, metrics


for use_id, tag in [(False, 'tags'), (True, 'tags+ids')]:
    ckpt, metrics = run_conf(use_id)
    check(f"[{tag}] best checkpoint saved", os.path.exists(ckpt))
    check(f"[{tag}] test metrics finite",
          all(np.isfinite(v) for v in metrics.values()),
          f"hr@10={metrics.get('hit_ratio@10'):.4f}")
    check(f"[{tag}] metric set complete",
          all(f'hit_ratio@{k}' in metrics for k in (1, 3, 5, 10, 50)))

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
