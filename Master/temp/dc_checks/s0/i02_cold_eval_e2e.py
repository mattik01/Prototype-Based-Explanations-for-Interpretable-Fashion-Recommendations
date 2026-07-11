# i02 — cold-eval end-to-end on a toy variant: train a real mf model on the variant with the
# repo Trainer (exercises training-negative exclusion + relaxed val invariant + nnz divisor in
# the real loop), then run the FULL cold_eval runner on the checkpoint and verify the report:
# both rankings, decile slices, bootstrap CIs, attr-kNN patch (with an identity check),
# tie-block diagnostic, popularity reference, and the use_bias refusal guard.
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from _harness import make_check, build_toy_canonical, REPO_ROOT

sys.path.insert(0, os.path.join(REPO_ROOT, 'data', 'hm'))
from make_cold_variant import make_cold_variant

from rec_sys.protomf_dataset import get_protorecdataset_dataloader
from rec_sys.trainer import Trainer
from utilities.cold_eval import (attr_knn_patch, build_model, load_checkpoint, load_config,
                                 run_cold_eval, _find_item_embedding_weight)
from utilities.utils import reproducible

check, fails = make_check()

tmp = tempfile.mkdtemp(prefix='s0_i02_')
canon = os.path.join(tmp, 'toy')
variant = os.path.join(tmp, 'toy_cold')
# bigger toy than t04/t05: the cold-vs-cold pool must hold >= 50 non-consumed candidates per
# user (K=50 metric floor -> n_neg=50)
build_toy_canonical(canon, n_users=200, n_items=800)
make_cold_variant(canon, variant)

EVAL_NEG = 50

conf_dict = {
    'data_path': variant, 'seed': 38210573, 'device': 'cpu',
    'n_epochs': 8, 'neg_train': 4, 'train_neg_strategy': 'uniform', 'eval_neg_strategy': 'uniform',
    'loss_func_name': 'bce', 'loss_func_aggr': 'mean', 'batch_size': 64, 'val_batch_size': 64,
    'rec_sys_param': {'use_bias': 0},
    'optim_param': {'optim': 'adam', 'lr': 1e-2, 'wd': 1e-4},
    'ft_ext_param': {'ft_type': 'detached', 'embedding_dim': 12,
                     'user_ft_ext_param': {'ft_type': 'embedding'},
                     'item_ft_ext_param': {'ft_type': 'embedding'}},
}

# --- train mf on the variant with the real Trainer (use_ray=False) ---
results_dir = os.path.join(tmp, 'mf_toy_cold_s38210573')
os.makedirs(results_dir, exist_ok=True)
train_loader = get_protorecdataset_dataloader(
    data_path=variant, split_set='train', n_neg=4, neg_strategy='uniform',
    batch_size=64, shuffle=True, num_workers=0)
val_loader = get_protorecdataset_dataloader(
    data_path=variant, split_set='val', n_neg=EVAL_NEG, neg_strategy='uniform',
    batch_size=64, num_workers=0)
check("variant val has fewer rows than users (relaxed invariant path)",
      val_loader.dataset.coo_matrix.nnz < val_loader.dataset.n_users,
      f"{val_loader.dataset.coo_matrix.nnz} < {val_loader.dataset.n_users}")
check("train loader excludes cold items from negatives",
      train_loader.dataset.neg_exclude_items is not None)

reproducible(38210573)
trainer = Trainer(train_loader, val_loader, argparse.Namespace(**conf_dict), use_ray=False)
trainer.run(checkpoint_dir=results_dir)
check("variant training ran and saved a checkpoint (nnz divisor path green)",
      os.path.exists(os.path.join(results_dir, 'best_model.pth')))

with open(os.path.join(results_dir, 'config.json'), 'w') as f:
    json.dump(conf_dict, f, indent=2, default=str)

# --- the full runner ---
report = run_cold_eval(results_dir, variant, canonical_path=canon, device='cpu',
                       batch_size=64, n_neg=EVAL_NEG, seed=38210573)

check("report blocks present",
      all(k in report for k in ('warm_test', 'cold_vs_cold', 'cold_vs_all',
                                'attr_knn', 'tie_block_diagnostic', 'popularity_reference')))
cvc = report['cold_vs_cold']
check("cold-vs-cold rows == cold_test rows", cvc['n_rows'] == report['cold_vs_all']['n_rows'])
check("metrics finite and in [0,1]",
      all(0. <= cvc[m] <= 1. for m in ('hit_ratio@10', 'ndcg@10')),
      f"HR@10={cvc['hit_ratio@10']:.3f}")
check("chance level declared for cold-vs-cold", abs(cvc['chance_hit_ratio@10'] - 10 / 51) < 1e-9)
ci = cvc['hit_ratio@10_ci95']
check("bootstrap CI brackets the point estimate", ci[0] <= cvc['hit_ratio@10'] <= ci[1],
      f"[{ci[0]:.3f}, {ci[1]:.3f}] vs {cvc['hit_ratio@10']:.3f}")
check("decile slices cover the rows",
      sum(e['n_rows'] for e in cvc['by_pop_decile']) == cvc['n_rows'])
check("attr-kNN re-eval present for a CF model",
      'cold_vs_cold' in report['attr_knn'], str(report['attr_knn'])[:60])
td = report['tie_block_diagnostic']
check("tie diagnostic sane", 1 <= td['mean_distinct_signatures_in_topk'] <= td['k'])
check("report files written",
      all(os.path.exists(os.path.join(results_dir, 'cold_eval', f))
          for f in ('report.json', 'report.md')))

# --- attr-kNN patch identity check (direct) ---
conf = load_config(results_dir)
model, _, _ = build_model(conf, variant, 'cpu')
load_checkpoint(model, results_dir, 'cpu')
w = _find_item_embedding_weight(model.item_feature_extractor)
before = w.data.clone()
info = attr_knn_patch(model, variant, n_neighbors=5)
import pandas as pd
cold_ids = np.sort(pd.read_csv(os.path.join(variant, 'cold_items.csv'))['item_id'].to_numpy())
warm_mask = np.ones(before.shape[0], dtype=bool)
warm_mask[cold_ids] = False
check("kNN patch: warm rows untouched", torch.equal(w.data[warm_mask], before[warm_mask]))
check("kNN patch: cold rows changed", not torch.equal(w.data[cold_ids], before[cold_ids]))
check("kNN patch: patched rows within trained-row convex hull (mean of 5 neighbors)",
      bool((w.data[cold_ids].abs().max() <= before[warm_mask].abs().max() + 1e-6)))
check("kNN patch info", info['n_cold_patched'] == len(cold_ids) and info['n_neighbors'] == 5)

# --- use_bias refusal ---
biased_dir = os.path.join(tmp, 'biased_run')
os.makedirs(biased_dir, exist_ok=True)
bad_conf = dict(conf_dict)
bad_conf['rec_sys_param'] = {'use_bias': 1}
with open(os.path.join(biased_dir, 'config.json'), 'w') as f:
    json.dump(bad_conf, f)
try:
    load_config(biased_dir)
    check("use_bias != 0 REFUSED without cold-bias policy", False)
except RuntimeError as e:
    check("use_bias != 0 REFUSED without cold-bias policy", True, str(e)[:60])

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
