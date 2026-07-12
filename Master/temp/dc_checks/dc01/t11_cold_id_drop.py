# t11 — F-S0-07(b): the cold-inference ID-column drop for dc01's NESTED item branch
# (feature_item_proto; design doc §3.4 cold notes / §3.7 deviation 6).
# Part A/C re-pointed at the SC.3 re-run (2026-07-12) to the fI host: the item branch is now
# PrototypeEmbedding(embedding_ext=FeatureEmbedding) directly (no Concatenate, no projection
# half); the UI-shaped variant lives in git history.
#
# Verifies, on a toy canonical -> committed cold-variant pipeline:
#   A. drop_cold_id_rows reaches the FeatureEmbedding nested inside the item branch
#      (PrototypeEmbedding.embedding_ext): cold representations become the pure feature
#      composition q_cold = Σ_f e_f (the activation t* changes for cold items); warm outputs
#      stay bit-identical; the _noid branch is a clean no-op.
#   B. the load_config refusal is LIFTED (dc01+ids accepted, incl. absent-key=default-True);
#      use_bias refusal intact; config_expects_id_drop truth table.
#   C. FULL runner end-to-end on a REAL trained fI toy checkpoint: id_column_drop in the
#      report, attr-kNN skips cleanly (CF-only seam raises ValueError on the feature branch),
#      finite metrics, report files written.
#   D. restore + re-apply: after a checkpoint reload the ID rows are live again; re-applying
#      the drop restores the zeroed state (the runner's defensive re-apply after kNN restore).
import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile

from _harness import make_check, feature_item_proto_param, build_recsys  # dc01 harness

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'data', 'hm'))

# the s0 harness is ALSO named _harness — load it under a distinct module name
_spec = importlib.util.spec_from_file_location(
    's0_harness', os.path.join(os.path.dirname(__file__), '..', 's0', '_harness.py'))
_s0h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s0h)
build_toy_canonical = _s0h.build_toy_canonical

from make_cold_variant import make_cold_variant  # noqa: E402
from feature_extraction.feature_ids import build_feature_ids  # noqa: E402
from rec_sys.protomf_dataset import get_protorecdataset_dataloader  # noqa: E402
from rec_sys.trainer import Trainer  # noqa: E402
from utilities.cold_eval import (CANONICAL_FIELDS, attr_knn_patch, config_expects_id_drop,  # noqa: E402
                                 drop_cold_id_rows, load_config, run_cold_eval,
                                 _find_feature_embedding)
from utilities.utils import reproducible  # noqa: E402

check, fails = make_check()
tmp = tempfile.mkdtemp(prefix='t11_')
canonical = os.path.join(tmp, 'toy')
variant = os.path.join(tmp, 'toy_cold')
# same sizing rationale as s0/i02: cold-vs-cold pool must hold >= 50 candidates (K=50 floor)
n_users, n_items = build_toy_canonical(canonical, n_users=200, n_items=800)
make_cold_variant(canonical, variant, seed=38210573, fraction=0.2)

cold_ids = np.sort(pd.read_csv(os.path.join(variant, 'cold_items.csv'))['item_id'].to_numpy())
cold_t = torch.as_tensor(cold_ids)
warm_ids = np.setdiff1d(np.arange(n_items), cold_ids)[:60]
warm_t = torch.as_tensor(warm_ids)

# ---------------- A. nested drop semantics (structural, no training) ----------------
feat_ids, n_feat = build_feature_ids(variant, CANONICAL_FIELDS)
d, Kt = 12, 7
reproducible(38210573)
model = build_recsys(feature_item_proto_param(d, Kt, feat_ids, n_feat), n_users, n_items)
item_fe = model.item_feature_extractor          # PrototypeEmbedding (fI host — the single branch)
shared = item_fe.embedding_ext                  # FeatureEmbedding

check('A: _find_feature_embedding locates the nested instance (PrototypeEmbedding.embedding_ext)',
      _find_feature_embedding(item_fe) is shared)

state_before = copy.deepcopy(model.state_dict())
warm_before = item_fe(warm_t).detach().clone()      # (60, Kt) — the activation t*
cold_before = item_fe(cold_t).detach().clone()

info = drop_cold_id_rows(model, variant)
check('A: drop applied and reports the full cold set',
      info is not None and info['n_cold_id_rows_zeroed'] == len(cold_ids), str(info))

# cold composed embedding == PURE feature composition (metadata rows only)
pure = shared.embedding_layer(shared.feature_ids[cold_t]).sum(dim=-2)
check('A: cold composed embedding == pure feature composition after drop',
      bool(torch.allclose(shared(cold_t), pure, atol=1e-7)))

# the activation surface (the WHOLE fI score path) changed for cold items …
cold_after = item_fe(cold_t).detach()
check('A: prototype activation t* (the single score surface) changed for cold items',
      not bool(torch.allclose(cold_after, cold_before)))

# … while warm outputs are bit-identical
check('A: warm item-branch outputs bit-identical after drop',
      bool(torch.equal(item_fe(warm_t).detach(), warm_before)))

# manual-zeroing equivalence: drop == zeroing rows n_features + cold_ids, nothing else
model2 = build_recsys(feature_item_proto_param(d, Kt, feat_ids, n_feat), n_users, n_items)
model2.load_state_dict(state_before)
fe2 = model2.item_feature_extractor.embedding_ext
fe2.embedding_layer.weight.data[fe2.n_features + cold_t] = 0.
check('A: drop == manual zeroing of exactly the cold ID rows (state_dict equality)',
      all(torch.equal(a, b) for (ka, a), (kb, b)
          in zip(model.state_dict().items(), model2.state_dict().items())))

# _noid branch: clean no-op
model_noid = build_recsys(
    feature_item_proto_param(d, Kt, feat_ids, n_feat, use_id_feature=False),
    n_users, n_items)
check('A: _noid branch -> drop returns None', drop_cold_id_rows(model_noid, variant) is None)

# ---------------- B. guard lift + config_expects_id_drop ----------------
run_dir = os.path.join(tmp, 'fake_run')
os.makedirs(run_dir, exist_ok=True)


def write_conf(ft_ext_param, use_bias=0):
    conf = {'rec_sys_param': {'use_bias': use_bias}, 'ft_ext_param': ft_ext_param,
            'loss_func_name': 'bce'}
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(conf, f)
    return conf


def accepted(conf_dir=run_dir):
    try:
        load_config(conf_dir)
        return True
    except RuntimeError:
        return False


c_ids = write_conf({'ft_type': 'feature_item_proto',
                    'item_ft_ext_param': {'use_id_feature': True}})
check('B: dc01+use_id_feature now ACCEPTED by load_config (guard lifted)', accepted())
check('B: config_expects_id_drop == True for dc01+ids', config_expects_id_drop(c_ids))

c_abs = write_conf({'ft_type': 'feature_item_proto', 'item_ft_ext_param': {}})
check('B: absent use_id_feature key ACCEPTED (drop still expected downstream)', accepted())
check('B: config_expects_id_drop == True for absent key (factory default True)',
      config_expects_id_drop(c_abs))

c_noid = write_conf({'ft_type': 'feature_item_proto',
                     'item_ft_ext_param': {'use_id_feature': False}})
check('B: _noid accepted, no drop expected',
      accepted() and not config_expects_id_drop(c_noid))

c_lf = write_conf({'ft_type': 'lightfm', 'item_ft_ext_param': {'use_id_feature': True}})
check('B: lightfm_tags_ids expects the drop', config_expects_id_drop(c_lf))
c_mf = write_conf({'ft_type': 'detached', 'item_ft_ext_param': {}})
check('B: CF config expects no drop', not config_expects_id_drop(c_mf))

write_conf({'ft_type': 'feature_item_proto',
            'item_ft_ext_param': {'use_id_feature': True}}, use_bias=1)
check('B: use_bias != 0 still REFUSED', not accepted())

# ---------------- C. full runner e2e on a trained dc01 toy checkpoint ----------------
EVAL_NEG = 50
conf_dict = {
    'data_path': variant, 'seed': 38210573, 'device': 'cpu',
    'n_epochs': 6, 'neg_train': 4, 'train_neg_strategy': 'uniform',
    'eval_neg_strategy': 'uniform', 'loss_func_name': 'bce', 'loss_func_aggr': 'mean',
    'batch_size': 64, 'val_batch_size': 64,
    'rec_sys_param': {'use_bias': 0},
    'optim_param': {'optim': 'adam', 'lr': 1e-2, 'wd': 1e-4},
    'ft_ext_param': {
        'ft_type': 'feature_item_proto', 'embedding_dim': 12,
        'user_ft_ext_param': {'ft_type': 'embedding'},
        'item_ft_ext_param': {'ft_type': 'feature_item_proto', 'sim_proto_weight': 1.,
                              'sim_batch_weight': 1., 'use_weight_matrix': False,
                              'n_prototypes': 6, 'cosine_type': 'shifted',
                              'reg_proto_type': 'max', 'reg_batch_type': 'max',
                              'use_id_feature': True,
                              'feature_fields': list(CANONICAL_FIELDS)},
    },
}
results_dir = os.path.join(tmp, 'dc01_toy_cold_run')
os.makedirs(results_dir, exist_ok=True)
train_loader = get_protorecdataset_dataloader(
    data_path=variant, split_set='train', n_neg=4, neg_strategy='uniform',
    batch_size=64, shuffle=True, num_workers=0)
val_loader = get_protorecdataset_dataloader(
    data_path=variant, split_set='val', n_neg=EVAL_NEG, neg_strategy='uniform',
    batch_size=64, num_workers=0)
reproducible(38210573)
trainer = Trainer(train_loader, val_loader, argparse.Namespace(**conf_dict), use_ray=False)
trainer.run(checkpoint_dir=results_dir)
check('C: dc01 toy training saved a checkpoint',
      os.path.exists(os.path.join(results_dir, 'best_model.pth')))
with open(os.path.join(results_dir, 'config.json'), 'w') as f:
    json.dump(conf_dict, f, indent=2, default=str)

report = run_cold_eval(results_dir, variant, canonical_path=canonical, device='cpu',
                       batch_size=64, n_neg=EVAL_NEG, seed=38210573)
check('C: runner ran the _ids config and recorded the drop',
      report.get('id_column_drop', {}).get('n_cold_id_rows_zeroed') == len(cold_ids),
      str(report.get('id_column_drop')))
check('C: attr-kNN skipped cleanly on the feature branch (CF-only seam)',
      'skipped' in report.get('attr_knn', {}), str(report.get('attr_knn'))[:70])
for key in ('warm_test', 'cold_vs_cold', 'cold_vs_all'):
    b = report[key]
    check(f'C: {key} metrics finite and in [0,1]',
          all(0. <= b[m] <= 1. for m in ('hit_ratio@10', 'ndcg@10')),
          f"HR@10={b['hit_ratio@10']:.3f}")
check('C: report files written',
      all(os.path.exists(os.path.join(results_dir, 'cold_eval', f))
          for f in ('report.json', 'report.md')))

# direct seam pin: attr_knn_patch raises ValueError on the dc01 branch (no per-item ID table
# in the Embedding sense — the fallback is defined for CF rows only)
try:
    attr_knn_patch(model, variant)
    check('C: attr_knn_patch raises ValueError on the dc01 item branch', False)
except ValueError:
    check('C: attr_knn_patch raises ValueError on the dc01 item branch', True)

# ---------------- D. restore + re-apply (the runner's defensive path) ----------------
zeroed_rows = shared.embedding_layer.weight.data[shared.n_features + cold_t].clone()
check('D: post-drop cold ID rows are exactly zero',
      bool(torch.equal(zeroed_rows, torch.zeros_like(zeroed_rows))))
model.load_state_dict(state_before)  # simulate the kNN-restore checkpoint reload
restored = shared.embedding_layer.weight.data[shared.n_features + cold_t]
check('D: checkpoint reload brings the cold ID rows back (drop is eval-time-only)',
      not bool(torch.equal(restored, torch.zeros_like(restored))))
drop_cold_id_rows(model, variant)
re_zeroed = shared.embedding_layer.weight.data[shared.n_features + cold_t]
check('D: re-applying the drop restores the zeroed state',
      bool(torch.equal(re_zeroed, torch.zeros_like(re_zeroed))))

shutil.rmtree(tmp, ignore_errors=True)
print()
print('ALL PASS' if not fails else f'FAILURES: {fails}')
sys.exit(1 if fails else 0)
