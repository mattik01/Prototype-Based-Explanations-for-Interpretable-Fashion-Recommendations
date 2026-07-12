# t12 — dc01 SC.5.4: the shared user-perspective breakdown renderer + prototype cards +
# dual naming route + explainer registration.
#
# Verifies:
#   A. fI slot exactness (untrained model, pure math): baseline Σu_k + Σ_k u_k(t*_k−1)
#      == forward score; zoom rows sum exactly to the top prototype's bar; ID row present
#      iff use_id_feature; _noid has no ID line; _f0 zoom is the ID row alone
#      (features: 0%); the self-check assert actually raises on a mismatch.
#   B. host slot (item_proto): identical contribution math; post-hoc exemplar zoom
#      declared as post-hoc.
#   C. lightfm slot: Σ_r u·e_r == forward score, ID line for the _ids arm; mf slot:
#      opaque statement + verified total; popularity: dataset-derived statement.
#   D. artifacts: figure + markdown written; md carries the arithmetic-honesty markers
#      (rank-inert baseline, ID row, exact-sum line, self-check line) and NO
#      effect-epistemics (no determinacy/gauge annotations — the SC.5 gate re-scope).
#   E. FULL pipeline e2e on a REAL trained fI toy checkpoint: intrinsic pass (cosine/)
#      + dual post-hoc naming pass (lift/) both written (F-DC01-13), breakdown +
#      prototype cards artifacts present, top1_item_for_user matches brute force.
#   F. explainer registration/gating: breakdown + proto_cards registered; supports()
#      sets match the SC.5 scope (fI + item_proto for breakdown; no attr_item_proto in
#      proto_cards at dc01's SC.5; user_proto/user_item_proto breakdown slot raises the
#      deferral error).
import argparse
import importlib.util
import json
import os
import sys
import tempfile

from _harness import make_check, feature_item_proto_param, item_proto_param, mf_param, build_recsys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO_ROOT)

_spec = importlib.util.spec_from_file_location(
    's0_harness', os.path.join(os.path.dirname(__file__), '..', 's0', '_harness.py'))
_s0h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s0h)
build_toy_canonical = _s0h.build_toy_canonical

from feature_extraction.feature_ids import build_feature_ids  # noqa: E402
from rec_sys.protomf_dataset import get_protorecdataset_dataloader  # noqa: E402
from rec_sys.trainer import Trainer  # noqa: E402
from utilities.cold_eval import CANONICAL_FIELDS  # noqa: E402
from utilities.explanations.breakdown import (  # noqa: E402
    _assert_close, compute_breakdown_feature_item_proto, compute_breakdown_item_proto,
    compute_breakdown_lightfm, compute_breakdown_mf, compute_breakdown_popularity,
    render_for_results_dir, top1_item_for_user, write_breakdown)
from utilities.utils import reproducible  # noqa: E402

check, fails = make_check()
tmp = tempfile.mkdtemp(prefix='t12_')
toy = os.path.join(tmp, 'toy')
n_users, n_items = build_toy_canonical(toy, n_users=120, n_items=300)
items_info = pd.read_csv(os.path.join(toy, 'item_features.csv'))
feat_ids, n_feat = build_feature_ids(toy, CANONICAL_FIELDS)
FIELDS = list(CANONICAL_FIELDS)
d, Kt = 12, 7
UID, IID = 3, 42

def fi_param(use_id=True, fields=True):
    p = feature_item_proto_param(d, Kt, feat_ids if fields else feat_ids[:, :0],
                                 n_feat if fields else 0, use_id_feature=use_id)
    p['item_ft_ext_param']['feature_fields'] = FIELDS if fields else []
    return p

# ---------------- A. fI slot exactness (untrained, pure math) ----------------
reproducible(38210573)
model_fi = build_recsys(fi_param(), n_users, n_items)
bd = compute_breakdown_feature_item_proto(model_fi, UID, IID, items_info, FIELDS)

S = float(model_fi(torch.tensor([UID]), torch.tensor([[IID]])).squeeze())
parts = bd.baseline + sum(l.value for l in bd.lines)
check('A: baseline + Σ prototype bars == forward score',
      abs(parts - S) < 5e-4 * max(1, abs(S)), f'{parts:.6f} vs {S:.6f}')
zoom_sum = sum(l.value for l in bd.zoom_lines)
check('A: zoom rows sum exactly to the top prototype bar',
      abs(zoom_sum - bd.zoom_parent_value) < 5e-4, f'{zoom_sum:.6f} vs {bd.zoom_parent_value:.6f}')
check('A: ID row present and flagged (ids arm)',
      any(l.is_id for l in bd.zoom_lines))
check('A: zoom has F metadata rows + 1 ID row', len(bd.zoom_lines) == len(FIELDS) + 1)
check('A: feature-explained fraction line present', bd.feature_explained is not None,
      str(bd.feature_explained))

model_noid = build_recsys(fi_param(use_id=False), n_users, n_items)
bd_noid = compute_breakdown_feature_item_proto(model_noid, UID, IID, items_info, FIELDS)
check('A: _noid arm has no ID line', not any(l.is_id for l in bd_noid.zoom_lines)
      and len(bd_noid.zoom_lines) == len(FIELDS))

model_f0 = build_recsys(fi_param(fields=False), n_users, n_items)
bd_f0 = compute_breakdown_feature_item_proto(model_f0, UID, IID, items_info, [])
check('A: _f0 zoom is the ID row alone', len(bd_f0.zoom_lines) == 1
      and bd_f0.zoom_lines[0].is_id)
check('A: _f0 feature-explained is 0%', 'features: 0.0%' in (bd_f0.feature_explained or ''),
      str(bd_f0.feature_explained))

try:
    _assert_close(1.0, 2.0, 'tamper probe')
    check('A: self-check assert raises on mismatch', False)
except AssertionError:
    check('A: self-check assert raises on mismatch', True)

# ---------------- B. host slot (item_proto) ----------------
reproducible(38210573)
model_ip = build_recsys(item_proto_param(d, Kt), n_users, n_items)
bd_ip = compute_breakdown_item_proto(model_ip, UID, IID, items_info)
S_ip = float(model_ip(torch.tensor([UID]), torch.tensor([[IID]])).squeeze())
parts_ip = bd_ip.baseline + sum(l.value for l in bd_ip.lines)
check('B: host baseline + Σ bars == forward score',
      abs(parts_ip - S_ip) < 5e-4 * max(1, abs(S_ip)), f'{parts_ip:.6f} vs {S_ip:.6f}')
check('B: host zoom is exemplars and declared post-hoc',
      bd_ip.zoom_kind == 'exemplars' and 'post-hoc' in bd_ip.zoom_title.lower()
      and any('POST-HOC' in n for n in bd_ip.notes))
check('B: host zoom lists items', len(bd_ip.zoom_lines) == 8)

# ---------------- C. lightfm / mf / popularity slots ----------------
def lightfm_param(use_id=True):
    return {'ft_type': 'lightfm', 'embedding_dim': d,
            'user_ft_ext_param': {'ft_type': 'embedding'},
            'item_ft_ext_param': {'ft_type': 'lightfm', 'use_id_feature': use_id,
                                  'feature_fields': FIELDS,
                                  'feature_ids': feat_ids, 'n_features': n_feat}}

reproducible(38210573)
model_lf = build_recsys(lightfm_param(), n_users, n_items)
bd_lf = compute_breakdown_lightfm(model_lf, UID, IID, items_info, FIELDS,
                                  model_label='LightFM-style CBF (lightfm_tags_ids)')
S_lf = float(model_lf(torch.tensor([UID]), torch.tensor([[IID]])).squeeze())
check('C: lightfm Σ per-feature bars == forward score',
      abs(sum(l.value for l in bd_lf.lines) - S_lf) < 5e-4 * max(1, abs(S_lf)))
check('C: lightfm _ids arm carries the ID line', any(l.is_id for l in bd_lf.lines))
check('C: lightfm declares no-prototype-layer', 'No prototype layer' in bd_lf.statement)

reproducible(38210573)
model_mf = build_recsys(mf_param(d), n_users, n_items)
bd_mf = compute_breakdown_mf(model_mf, UID, IID, items_info)
S_mf = float(model_mf(torch.tensor([UID]), torch.tensor([[IID]])).squeeze())
check('C: mf slot is opaque with verified total',
      bd_mf.mechanism == 'opaque' and abs(bd_mf.total_score - S_mf) < 1e-6
      and 'no decomposable explanation surface' in bd_mf.statement)

bd_pop = compute_breakdown_popularity(toy, UID)
check('C: popularity statement carries rank + non-personalized declaration',
      'rank' in bd_pop.statement and 'Non-personalized' in bd_pop.statement
      and bd_pop.total_score > 0, bd_pop.statement[:60])

# ---------------- D. artifacts + honesty markers ----------------
out = os.path.join(tmp, 'art')
stem = write_breakdown(bd, out)
check('D: figure + markdown written',
      os.path.exists(stem + '.png') and os.path.exists(stem + '.md'))
md = open(stem + '.md').read()
for marker in ('rank-inert baseline', 'ID row', 'Self-check passed', 'anti-affinities'):
    check(f'D: md carries "{marker}"', marker in md)
for absent in ('determinacy', 'gauge', 'canonical representative'):
    check(f'D: md carries NO effect-epistemics ("{absent}")', absent not in md.lower())
write_breakdown(bd_ip, out)   # host figure renders too
write_breakdown(bd_lf, out)
write_breakdown(bd_mf, out)
write_breakdown(bd_pop, out)
check('D: all five slots render without error', True)

# ---------------- E. pipeline e2e on a trained fI toy checkpoint ----------------
conf_dict = {
    'data_path': toy, 'seed': 38210573, 'device': 'cpu',
    'n_epochs': 4, 'neg_train': 4, 'train_neg_strategy': 'uniform',
    'eval_neg_strategy': 'uniform', 'loss_func_name': 'bce', 'loss_func_aggr': 'mean',
    'batch_size': 64, 'val_batch_size': 64,
    'rec_sys_param': {'use_bias': 0},
    'optim_param': {'optim': 'adam', 'lr': 1e-2, 'wd': 1e-4},
    'ft_ext_param': {
        'ft_type': 'feature_item_proto', 'embedding_dim': d,
        'user_ft_ext_param': {'ft_type': 'embedding'},
        'item_ft_ext_param': {'ft_type': 'feature_item_proto', 'sim_proto_weight': 1.,
                              'sim_batch_weight': 1., 'use_weight_matrix': False,
                              'n_prototypes': Kt, 'cosine_type': 'shifted',
                              'reg_proto_type': 'max', 'reg_batch_type': 'max',
                              'use_id_feature': True, 'feature_fields': FIELDS},
    },
}
results_dir = os.path.join(tmp, 'fi_toy_run')
os.makedirs(results_dir, exist_ok=True)
train_loader = get_protorecdataset_dataloader(
    data_path=toy, split_set='train', n_neg=4, neg_strategy='uniform',
    batch_size=64, shuffle=True, num_workers=0)
val_loader = get_protorecdataset_dataloader(
    data_path=toy, split_set='val', n_neg=50, neg_strategy='uniform',
    batch_size=64, num_workers=0)  # >= K=50 so the metric set is computable (as t11)
reproducible(38210573)
trainer = Trainer(train_loader, val_loader, argparse.Namespace(**conf_dict), use_ray=False)
trainer.run(checkpoint_dir=results_dir)
with open(os.path.join(results_dir, 'config.json'), 'w') as f:
    json.dump(conf_dict, f, indent=2, default=str)
with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
    json.dump({'model': 'feature_item_proto', 'dataset': 'toy'}, f)

# point the pipeline's DATA_PATH-bound loaders at the tmp dir (dataset name = 'toy')
import utilities.explanations.loader as _loader_mod
import utilities.explanations.items_info as _items_mod
_loader_mod.DATA_PATH = tmp
_items_mod.DATA_PATH = tmp

from utilities.explanations.pipeline import run_explanations_pipeline  # noqa: E402
exp_dir = run_explanations_pipeline(results_dir, tsne_sample_size=150)
check('E: pipeline ran the intrinsic pass (cosine/)',
      exp_dir is not None and exp_dir.endswith('cosine')
      and os.path.exists(os.path.join(exp_dir, 'prototype_names_item.csv')))
lift_dir = os.path.join(results_dir, 'explanations', 'lift')
check('E: dual naming route wrote the post-hoc pass (lift/) — F-DC01-13',
      os.path.exists(os.path.join(lift_dir, 'prototype_names_item.csv'))
      and os.path.exists(os.path.join(lift_dir, 'prototype_naming_stats_item.csv')))
check('E: breakdown artifact rendered on the trained checkpoint',
      any(f.startswith('breakdown_user') and f.endswith('.png')
          for f in os.listdir(exp_dir)))
check('E: prototype cards rendered',
      os.path.exists(os.path.join(exp_dir, 'prototype_cards_item.png')))

# standalone results-dir entry point + brute-force top-1 pin
model_tr, _, _ = _loader_mod.load_recsys_from_results_dir(results_dir, data_dir=toy)
with torch.no_grad():
    all_scores = model_tr(torch.tensor([UID]),
                          torch.arange(n_items).unsqueeze(0)).squeeze(0)
check('E: top1_item_for_user matches brute force',
      top1_item_for_user(model_tr, UID, n_items) == int(torch.argmax(all_scores)))
stem2 = render_for_results_dir(results_dir, UID, data_dir=toy)
check('E: render_for_results_dir writes into explanations/breakdown/',
      os.path.exists(stem2 + '.png') and 'explanations' in stem2)

# ---------------- F. registration + gating scope ----------------
from utilities.explanations.explainers import REGISTERED_EXPLAINERS  # noqa: E402
names = {e.name for e in REGISTERED_EXPLAINERS}
check('F: breakdown + proto_cards registered',
      {'breakdown', 'proto_cards'} <= names, str(sorted(names)))
bde = next(e for e in REGISTERED_EXPLAINERS if e.name == 'breakdown')
check('F: breakdown supports exactly the dc01 stage pair',
      bde.supports('feature_item_proto') and bde.supports('item_proto')
      and not bde.supports('user_item_proto') and not bde.supports('attr_item_proto'))
pce = next(e for e in REGISTERED_EXPLAINERS if e.name == 'proto_cards')
check('F: proto_cards excludes attr_item_proto at dc01 SC.5',
      pce.supports('feature_item_proto') and pce.supports('item_proto')
      and not pce.supports('attr_item_proto'))
try:
    render_for_results_dir.__wrapped__  # noqa: B018 — no wrapper expected
except AttributeError:
    pass
# deferral error for user-side hosts
with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
    json.dump({'model': 'user_item_proto', 'dataset': 'toy'}, f)
try:
    render_for_results_dir(results_dir, UID, data_dir=toy)
    check('F: user_item_proto slot raises the stage-deferral error', False)
except NotImplementedError as e:
    check('F: user_item_proto slot raises the stage-deferral error',
          'deferred' in str(e))
except Exception as e:  # noqa: BLE001 — config mismatch would land here
    check('F: user_item_proto slot raises the stage-deferral error', False, repr(e))
finally:
    with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
        json.dump({'model': 'feature_item_proto', 'dataset': 'toy'}, f)

print(f"\n{'ALL PASS' if not fails else f'FAILURES: {fails}'} "
      f"({len(fails)} failed)")
sys.exit(1 if fails else 0)
