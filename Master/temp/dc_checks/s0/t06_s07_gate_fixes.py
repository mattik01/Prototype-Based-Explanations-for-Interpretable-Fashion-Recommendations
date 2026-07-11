# t06 — S0.7 gate fixes (F-S0-07 guard, F-S0-08 popularity scorer, F-S0-09 eval-negative
# exclusion, F-S0-13 ID-column drop). Self-contained: toy canonical -> committed generator ->
# toy cold variant in a temp dir.
import json
import os
import shutil
import sys
import tempfile

from _harness import make_check, build_toy_canonical, REPO_ROOT  # noqa: E402 (sys.path setup)

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, REPO_ROOT)
from data.hm.make_cold_variant import make_cold_variant  # noqa: E402
from rec_sys.protomf_dataset import ProtoRecDataset  # noqa: E402
from utilities.cold_eval import (PopularityScorer, drop_cold_id_rows,  # noqa: E402
                                 load_config)
from utilities.eval import Hit_Ratio_at_k_batch  # noqa: E402

check, fails = make_check()
tmp = tempfile.mkdtemp(prefix='t06_')
canonical = os.path.join(tmp, 'toy')
variant = os.path.join(tmp, 'toy_cold')
n_users, n_items = build_toy_canonical(canonical, n_users=80, n_items=300)
make_cold_variant(canonical, variant, seed=38210573, fraction=0.2)

# ---- 1. F-S0-07: guard refuses feature_item_proto with use_id_feature enabled ----
run_dir = os.path.join(tmp, 'fake_dc01_run')
os.makedirs(run_dir)


def write_conf(ft_ext_param):
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'rec_sys_param': {'use_bias': 0}, 'ft_ext_param': ft_ext_param,
                   'loss_func_name': 'bce'}, f)


write_conf({'ft_type': 'feature_item_proto',
            'item_ft_ext_param': {'ft_type': 'feature_item_proto', 'use_id_feature': True}})
try:
    load_config(run_dir)
    check('F-S0-07: dc01+use_id_feature REFUSED', False)
except RuntimeError as e:
    check('F-S0-07: dc01+use_id_feature REFUSED', 'F-S0-07' in str(e), str(e)[:60])

write_conf({'ft_type': 'feature_item_proto',
            'item_ft_ext_param': {'ft_type': 'feature_item_proto'}})  # key absent -> default True
try:
    load_config(run_dir)
    check('F-S0-07: absent key treated as enabled (factory default)', False)
except RuntimeError:
    check('F-S0-07: absent key treated as enabled (factory default)', True)

write_conf({'ft_type': 'feature_item_proto',
            'item_ft_ext_param': {'ft_type': 'feature_item_proto', 'use_id_feature': False}})
try:
    conf = load_config(run_dir)
    check('F-S0-07: _noid config passes the guard', True)
except RuntimeError as e:
    check('F-S0-07: _noid config passes the guard', False, str(e)[:60])

write_conf({'ft_type': 'lightfm',
            'item_ft_ext_param': {'ft_type': 'lightfm', 'use_id_feature': True}})
try:
    load_config(run_dir)
    check('F-S0-07: lightfm_tags_ids NOT refused (drop implemented for it)', True)
except RuntimeError as e:
    check('F-S0-07: lightfm_tags_ids NOT refused (drop implemented for it)', False, str(e)[:60])

# ---- 2. F-S0-08: PopularityScorer rank semantics ----
scorer = PopularityScorer(variant, n_items, seed=38210573)
scores = scorer.pop.numpy()
check('F-S0-08: scores in [0,1) — no sigmoid saturation', 0. <= scores.min() and scores.max() < 1.,
      f'range [{scores.min():.4f}, {scores.max():.4f}]')

train = pd.read_csv(os.path.join(variant, 'listening_history_train.csv'))
pop = np.zeros(n_items, dtype=np.int64)
c = train.groupby('item_id').size()
pop[c.index.to_numpy()] = c.to_numpy()
hi, lo = int(np.argmax(pop)), int(np.argmin(pop))
order_ok = True
# distinct popularity levels must never be reordered by the tie-break noise
for a in range(0, n_items, 7):
    for b in range(0, n_items, 13):
        if pop[a] > pop[b] and scores[a] <= scores[b]:
            order_ok = False
check('F-S0-08: noise never reorders distinct popularity levels', order_ok)

# all-tie chance behavior: rows of 100 candidates drawn from the cold pool (all pop 0)
cold_ids = pd.read_csv(os.path.join(variant, 'cold_items.csv'))['item_id'].to_numpy()
zero_pool = cold_ids[pop[cold_ids] == 0]  # cold items have 0 variant-train pop by construction
rng = np.random.default_rng(7)
n_rows, hits = 4000, []
for _ in range(n_rows):
    cand = rng.choice(zero_pool, size=min(100, len(zero_pool)), replace=False)
    k = min(10, len(cand) - 1)
    logits = scores[cand][None, :]
    hits.append(int(Hit_Ratio_at_k_batch(logits, k=k, sum=True)))
hr = np.mean(hits)
expected = k / min(100, len(zero_pool))
check('F-S0-08: all-tie rows read ~chance (not 0.00)', abs(hr - expected) < 0.04,
      f'HR@{k} = {hr:.3f} vs chance {expected:.3f} over {n_rows} rows, pool {len(zero_pool)}')

check('F-S0-08: scorer is seed-deterministic',
      bool(torch.equal(scorer.pop, PopularityScorer(variant, n_items, seed=38210573).pop)))

# ---- 3. F-S0-09: variant val/test negatives exclude the user's removed cold purchases ----
ct = pd.read_csv(os.path.join(variant, 'cold_test.csv'))
removed = {}
for u, i in zip(ct['user_id'], ct['item_id']):
    removed.setdefault(int(u), set()).add(int(i))

violations = 0
for split in ('val', 'test'):
    ds = ProtoRecDataset(variant, split, n_neg=30, neg_strategy='uniform')
    np.random.seed(4242)
    for idx in range(len(ds)):
        u, item_idxs, _ = ds[idx]
        negs = set(int(x) for x in item_idxs[1:])
        violations += len(negs & removed.get(int(u), set()))
check('F-S0-09: no removed cold purchase drawn as variant val/test negative', violations == 0,
      f'{violations} violations over full val+test sweep')

# canonical (unmarked) datasets are untouched by the new path
ds_canon = ProtoRecDataset(canonical, 'val', n_neg=10, neg_strategy='uniform')
check('F-S0-09: canonical path unaffected (no marker, no fold-in)',
      ds_canon.csr_matrix.nnz == len(pd.read_csv(os.path.join(canonical, 'listening_history_val.csv')))
      + len(pd.read_csv(os.path.join(canonical, 'listening_history_train.csv'))))

# ---- 4. F-S0-13: ID-column drop on a lightfm-style FeatureEmbedding branch ----
sys.path.insert(0, REPO_ROOT)
from feature_extraction.feature_extractors import Embedding, FeatureEmbedding  # noqa: E402
from feature_extraction.feature_ids import build_feature_ids  # noqa: E402
from rec_sys.rec_sys import RecSys  # noqa: E402

FIELDS = ['department_name', 'product_type_name', 'section_name',
          'colour_group_name', 'graphical_appearance_name']
feat_ids, n_feat = build_feature_ids(variant, FIELDS)
d = 16
user_fe = Embedding(n_users, d)
item_fe = FeatureEmbedding(n_items, feat_ids, n_feat, d, use_id_feature=True)
model = RecSys(n_users, n_items, {'use_bias': 0}, user_fe, item_fe, 'bce')
model.init_parameters()

cold_t = torch.as_tensor(np.sort(cold_ids))
before = item_fe(cold_t).clone()
info = drop_cold_id_rows(model, variant)
check('F-S0-13: drop reports the full cold set', info is not None
      and info['n_cold_id_rows_zeroed'] == len(cold_ids), str(info))

# after the drop, cold representations equal the pure feature composition (sum of field rows)
after = item_fe(cold_t)
tags_only = item_fe.embedding_layer(item_fe.feature_ids[cold_t]).sum(dim=-2)
check('F-S0-13: cold repr == pure feature composition after drop',
      bool(torch.allclose(after, tags_only, atol=1e-6)))
check('F-S0-13: drop actually changed cold reprs', not bool(torch.allclose(before, after)))

warm_ids = torch.as_tensor([i for i in range(n_items) if i not in set(cold_ids.tolist())][:50])
warm_before = item_fe(warm_ids).clone()
id_rows = item_fe.embedding_layer.weight.data[item_fe.n_features + warm_ids]
check('F-S0-13: warm ID rows untouched', not bool(torch.allclose(id_rows, torch.zeros_like(id_rows))))

# no-ID branch: drop is a no-op returning None
item_fe_noid = FeatureEmbedding(n_items, feat_ids, n_feat, d, use_id_feature=False)
model_noid = RecSys(n_users, n_items, {'use_bias': 0}, user_fe, item_fe_noid, 'bce')
check('F-S0-13: tags-only branch -> drop returns None', drop_cold_id_rows(model_noid, variant) is None)

# plain CF branch: no FeatureEmbedding -> None
model_cf = RecSys(n_users, n_items, {'use_bias': 0}, Embedding(n_users, d), Embedding(n_items, d), 'bce')
check('F-S0-13: CF branch -> drop returns None', drop_cold_id_rows(model_cf, variant) is None)

shutil.rmtree(tmp)
print()
print('ALL PASS' if not fails else f'FAILURES: {fails}')
sys.exit(1 if fails else 0)
