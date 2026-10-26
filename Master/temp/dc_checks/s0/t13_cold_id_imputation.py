# S0 pin — the cold ID-row convention (S0.3 §5 as amended 2026-10-26, F-DC09-08).
# For every feature-composed family with ID rows (dc01 fI-ids, S0.4 lightfm_tags_ids, dc05
# fU-ids user side), on a fixed one-word-per-field toy layout, float64:
#   1. imputation sets cold rows to the mean of the WARM rows and touches nothing else;
#   2. the translation gauge — every word of field φ += v_φ, every ID row −= Σ_φ v_φ — leaves
#      every WARM score unchanged (the symmetry is real);
#   3. under imputation the COLD scores are unchanged by the gauge (the convention is
#      invariant); under the old drop they move (the convention was gauge-dependent);
#   4. user side: an empty-basket cold user keeps q_u = 0 under imputation, so it stays
#      invariant too (imputing the mean there would break it — SC.2 toy D7c).
# Precision: item compositions are unweighted sums, so the gauge is exact (1e-12 here). The user
# composition weighs each basket word by n_{u,f}/|H_u|, stored as float32 by
# HistoryFeatureEmbedding, so a field's weights sum to 1 only to float32 rounding (3 × 1/3): the
# gauge — and with it the invariance — holds to ~1e-7 on that side, in real models too.
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import copy  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory  # noqa: E402
from rec_sys.rec_sys import RecSys  # noqa: E402
from utilities.cold_eval import (_find_feature_embedding, _find_history_feature_embedding,  # noqa: E402
                                 apply_cold_id_convention)
from utilities.utils import reproducible  # noqa: E402

fails = []


def check(name, cond, evidence=''):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f' | {evidence}' if evidence else ''))
    if not cond:
        fails.append(name)


VOCAB, M, N, D, K = (3, 4, 2), 14, 9, 6, 4
OFFS = [sum(VOCAB[:k]) for k in range(len(VOCAB))]
V = sum(VOCAB)
g = torch.Generator().manual_seed(0)
feature_ids = torch.stack([torch.randint(0, v, (M,), generator=g) + o for v, o in zip(VOCAB, OFFS)], 1)
baskets = [sorted(torch.randperm(M, generator=g)[:int(torch.randint(2, 5, (1,), generator=g))].tolist())
           for _ in range(N)]
baskets[7] = []  # an empty-basket (cold) user
hist_ids = torch.zeros((N, V), dtype=torch.long)
hist_w = torch.zeros((N, V))
for u, b in enumerate(baskets):
    c = torch.zeros(V)
    for i in b:
        c.index_add_(0, feature_ids[i], torch.ones(len(VOCAB)))
    nz = c.nonzero().flatten()
    hist_ids[u, :len(nz)] = nz
    hist_w[u, :len(nz)] = c[nz] / max(len(b), 1)
COLD_I, COLD_U = [3, 10, 12], [2, 7]
tmp = tempfile.mkdtemp()
pd.DataFrame({'item_id': COLD_I}).to_csv(os.path.join(tmp, 'cold_items.csv'), index=False)
pd.DataFrame({'user_id': COLD_U}).to_csv(os.path.join(tmp, 'cold_users.csv'), index=False)
proto = dict(sim_proto_weight=1., sim_batch_weight=1., use_weight_matrix=False, n_prototypes=K,
             cosine_type='shifted', reg_proto_type='max', reg_batch_type='max')
PARAMS = {
    'fI_ids': {'ft_type': 'feature_item_proto', 'embedding_dim': D, 'user_ft_ext_param': {'ft_type': 'embedding'},
               'item_ft_ext_param': {'ft_type': 'feature_item_proto', **proto, 'use_id_feature': True,
                                     'feature_ids': feature_ids, 'n_features': V}},
    'lightfm_tags_ids': {'ft_type': 'lightfm', 'embedding_dim': D, 'user_ft_ext_param': {'ft_type': 'embedding'},
                         'item_ft_ext_param': {'ft_type': 'lightfm', 'use_id_feature': True,
                                               'feature_ids': feature_ids, 'n_features': V}},
    'fU_ids': {'ft_type': 'feature_user_proto', 'embedding_dim': D, 'item_ft_ext_param': {'ft_type': 'embedding'},
               'user_ft_ext_param': {'ft_type': 'feature_user_proto', **proto, 'use_id_feature': True,
                                     'hist_value_ids': hist_ids, 'hist_weights': hist_w, 'n_features': V}},
}


def build(name):
    reproducible(9)
    u_fe, i_fe = FeatureExtractorFactory.create_models(copy.copy(PARAMS[name]), N, M)
    rec = RecSys(N, M, {'use_bias': 0}, u_fe, i_fe, 'sampled_softmax')
    rec.init_parameters()
    return rec.double().eval()


def gauge_shift(rec, side, seed=4):
    """words of field φ += v_φ; every ID row on that side −= Σ_φ v_φ (score-preserving)."""
    fe = (_find_feature_embedding(rec.item_feature_extractor) if side == 'item'
          else _find_history_feature_embedding(rec.user_feature_extractor))
    W = fe.embedding_layer.weight.data
    gv = torch.Generator().manual_seed(seed)
    v = [torch.randn(D, generator=gv, dtype=torch.float64) * 3 for _ in VOCAB]
    for o, n_v, vv in zip(OFFS, VOCAB, v):
        W[o:o + n_v] += vv
    off = fe.n_features if side == 'item' else fe.id_row_offset
    W[off:off + fe.n_objects] -= sum(v)


def scores(rec, users, items):
    with torch.no_grad():
        return rec(torch.as_tensor(users), torch.as_tensor(items)).detach()


all_users = np.arange(N)
all_items = np.tile(np.arange(M), (N, 1))
warm_users = [u for u in range(N) if u not in COLD_U]
for name in PARAMS:
    side = 'user' if name.startswith('fU') else 'item'
    a = build(name)
    b = build(name)
    gauge_shift(b, side)
    warm_u = warm_users if side == 'user' else list(all_users)
    warm_items = [i for i in range(M) if i not in COLD_I] if side == 'item' else list(range(M))
    sa = scores(a, warm_u, np.tile(warm_items, (len(warm_u), 1)))
    sb = scores(b, warm_u, np.tile(warm_items, (len(warm_u), 1)))
    tol = 1e-12 if side == 'item' else 1e-6  # see the precision note in the header
    err = ((sa - sb).abs().max() / sa.abs().max()).item()
    check(f'{name}: the gauge is real — warm scores unchanged', err < tol, f'max rel {err:.1e}')

    # 1. imputation values
    fe = (_find_feature_embedding(a.item_feature_extractor) if side == 'item'
          else _find_history_feature_embedding(a.user_feature_extractor))
    off = fe.n_features if side == 'item' else fe.id_row_offset
    cold = COLD_I if side == 'item' else COLD_U
    before = fe.embedding_layer.weight.detach().clone()
    info_i, info_u = apply_cold_id_convention(a, tmp, 'impute_mean')
    W = fe.embedding_layer.weight
    warm_rows = [r for r in range(fe.n_objects) if r not in cold]
    mean = before[torch.as_tensor(warm_rows) + off].mean(0)
    basket_cold = [c for c in cold if side == 'item' or baskets[c]]
    empty_cold = [c for c in cold if side == 'user' and not baskets[c]]
    check(f'{name}: cold rows = warm mean (empty-basket users = 0)',
          all(torch.equal(W[off + c], mean) for c in basket_cold)
          and all(bool((W[off + c] == 0).all()) for c in empty_cold), str(info_i or info_u))
    changed = (W != before).any(1).nonzero().flatten().tolist()
    check(f'{name}: nothing else changes', changed == sorted(off + c for c in cold), f'{changed}')

    # 3./4. invariance of the convention under the gauge
    for conv in ('impute_mean', 'drop'):
        x, y = build(name), build(name)
        gauge_shift(y, side)
        apply_cold_id_convention(x, tmp, conv)
        apply_cold_id_convention(y, tmp, conv)
        if side == 'item':
            cx, cy = scores(x, all_users, np.tile(COLD_I, (N, 1))), scores(y, all_users, np.tile(COLD_I, (N, 1)))
        else:
            cx, cy = scores(x, COLD_U, np.tile(np.arange(M), (len(COLD_U), 1))), \
                     scores(y, COLD_U, np.tile(np.arange(M), (len(COLD_U), 1)))
        rel = ((cx - cy).abs().max() / cx.abs().max()).item()
        if conv == 'impute_mean':
            check(f'{name}: imputation is gauge-invariant (cold scores unchanged)', rel < tol, f'max rel {rel:.1e}')
        else:
            check(f'{name}: drop is gauge-dependent (cold scores move)', rel > 1e-3, f'max rel {rel:.1e}')

print()
print('ALL PASS' if not fails else f'FAILURES: {fails}')
sys.exit(1 if fails else 0)
