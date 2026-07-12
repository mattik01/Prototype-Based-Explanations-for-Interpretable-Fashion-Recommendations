# t08 — fU builder multi-value support (S0-build extension component b):
# build_user_history_weights(layout='bags').
# Keystone: on one-value-per-field (H&M-shaped) inputs, layout='bags' is BIT-IDENTICAL to the
# original layout='fixed' path (toy + real V1). Plus: hand-computed multi-value aggregation
# (n_{u,f}/|H_u| over variable-size bags, short-bag items included), per-user vote mass =
# mean tokens per purchase (raw bags, S0.5-addendum decision A), layout guard.
import os
import shutil
import sys
import tempfile

import torch

from _harness import make_check, REPO_ROOT

from feature_extraction.feature_ids import build_user_history_weights
from feature_extraction.feature_extractors import HistoryFeatureEmbedding

check, fails = make_check()

tmp = tempfile.mkdtemp(prefix='s0_t08_')


def write_dir(feature_rows, feature_header, train_pairs, n_users):
    d = tempfile.mkdtemp(prefix='case_', dir=tmp)
    with open(os.path.join(d, 'item_features.csv'), 'w') as f:
        f.write(feature_header + '\n')
        for r in feature_rows:
            f.write(','.join(str(x) for x in r) + '\n')
    with open(os.path.join(d, 'user_ids.csv'), 'w') as f:
        f.write('user_id,user\n' + '\n'.join(f'{u},u{u}' for u in range(n_users)) + '\n')
    with open(os.path.join(d, 'listening_history_train.csv'), 'w') as f:
        f.write('user_id,item_id\n')
        for u, i in train_pairs:
            f.write(f'{u},{i}\n')
    return d


# ---------------------------------------------------------------------------
# K1 — fixed-width toy: layout='bags' ≡ layout='fixed', bit-identical
# ---------------------------------------------------------------------------
fixed_dir = write_dir(
    [(0, 'blue', 'square'), (1, 'green', 'square'), (2, 'red', 'circle'), (3, 'blue', 'circle')],
    'item_id,color,shape',
    [(0, 0), (0, 1), (1, 2), (2, 0), (2, 2), (2, 3), (2, 1)], n_users=3)
ids_f, w_f, nf_f = build_user_history_weights(fixed_dir, ['color', 'shape'], layout='fixed')
ids_b, w_b, nf_b = build_user_history_weights(fixed_dir, ['color', 'shape'], layout='bags')
check("K1: fixed toy — hist_value_ids bit-identical", torch.equal(ids_f, ids_b),
      f"{ids_f.tolist()} vs {ids_b.tolist()}")
check("K1: fixed toy — hist_weights bit-identical", torch.equal(w_f, w_b), "")
check("K1: fixed toy — n_features equal", nf_f == nf_b, f"{nf_f} vs {nf_b}")

# ---------------------------------------------------------------------------
# K2 — real V1 (hm_1_month): bags path reproduces the shipped fixed path bit-identically
# ---------------------------------------------------------------------------
V1_FIELDS = ['department_name', 'product_type_name', 'section_name',
             'colour_group_name', 'graphical_appearance_name']
v1 = os.path.join(REPO_ROOT, 'data', 'hm_1_month')
if os.path.exists(os.path.join(v1, 'item_features.csv')):
    v_ids_f, v_w_f, v_nf_f = build_user_history_weights(v1, V1_FIELDS, layout='fixed')
    v_ids_b, v_w_b, v_nf_b = build_user_history_weights(v1, V1_FIELDS, layout='bags')
    check("K2: V1 — hist_value_ids bit-identical", torch.equal(v_ids_f, v_ids_b),
          f"shape {tuple(v_ids_f.shape)}")
    check("K2: V1 — hist_weights bit-identical", torch.equal(v_w_f, v_w_b),
          f"maxΔ={float((v_w_f - v_w_b).abs().max()):.3e}")
    check("K2: V1 — n_features == 426 in both", v_nf_f == v_nf_b == 426, f"{v_nf_f}/{v_nf_b}")
else:
    print("[SKIP] hm_1_month not present")

# ---------------------------------------------------------------------------
# Multi-value aggregation, hand-computed (ml-1m-shaped)
# genres vocab: Action=0, Comedy=1, Drama=2 ; tags (+3): dark=3, funny=4, long=5
# item bags: i0={0,1,4}, i1={2,3,5}, i2={1} (short bag), i3={0,2,3,4,5}
# ---------------------------------------------------------------------------
ml_dir = write_dir(
    [(0, 'Action|Comedy', 'funny'), (1, 'Drama', 'dark|long'),
     (2, 'Comedy', ''), (3, 'Action|Drama', 'dark|funny|long')],
    'item_id,genres,tags',
    [(0, 0), (0, 1),            # u0: |H|=2, all six tokens once each -> w = 1/2
     (1, 2),                    # u1: |H|=1, token 1 -> w = 1
     (2, 0), (2, 2), (2, 3)],   # u2: |H|=3, n: {0:2, 1:2, 2:1, 3:1, 4:2, 5:1}
    n_users=3)
m_ids, m_w, m_nf = build_user_history_weights(ml_dir, ['genres', 'tags'], layout='bags')
check("bags: n_features = 6", m_nf == 6, f"{m_nf}")
check("bags: D_max = 6 (u0/u2 carry six distinct tokens)", m_ids.shape == (3, 6),
      f"{tuple(m_ids.shape)}")


def row_as_dict(u):
    return {int(t): float(w) for t, w in zip(m_ids[u], m_w[u]) if float(w) > 0}


check("bags: u0 weights = 1/2 on all six tokens",
      row_as_dict(0) == {t: 0.5 for t in range(6)}, f"{row_as_dict(0)}")
check("bags: u1 weight = 1.0 on token 1 only (short-bag item)",
      row_as_dict(1) == {1: 1.0}, f"{row_as_dict(1)}")
u2_expect = {0: 2 / 3, 1: 2 / 3, 2: 1 / 3, 3: 1 / 3, 4: 2 / 3, 5: 1 / 3}
u2_got = row_as_dict(2)
check("bags: u2 weights = n_{u,f}/|H_u| over variable bags",
      set(u2_got) == set(u2_expect) and all(abs(u2_got[t] - u2_expect[t]) < 1e-6 for t in u2_expect),
      f"{u2_got}")
# raw-bag vote mass: Σ_f w̄_{u,f} == mean tokens per purchase (u2: (3+1+5)/3 = 3)
check("bags: per-user vote mass == mean tokens/purchase (raw bags, decision A)",
      abs(float(m_w[2].sum()) - 3.0) < 1e-6 and abs(float(m_w[0].sum()) - 3.0) < 1e-6
      and abs(float(m_w[1].sum()) - 1.0) < 1e-6,
      f"masses {[round(float(m_w[u].sum()), 4) for u in range(3)]}")

# HistoryFeatureEmbedding consumes the bag-derived tensors unchanged (layout-agnostic)
hfe = HistoryFeatureEmbedding(3, m_ids, m_w, m_nf, embedding_dim=4, use_id_feature=True)
hfe.init_parameters()
out = hfe(torch.arange(3))
check("HistoryFeatureEmbedding consumes bag-derived tensors (finite forward)",
      out.shape == (3, 4) and bool(torch.isfinite(out).all()), "")

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
try:
    build_user_history_weights(ml_dir, ['genres'], layout='nosuch')
    check("guard: unknown layout raises", False, "no raise")
except ValueError as e:
    check("guard: unknown layout raises", 'layout' in str(e), str(e)[:60])

# degenerate F=0 path unchanged in bags mode
z_ids, z_w, z_nf = build_user_history_weights(ml_dir, [], layout='bags')
check("bags: F=0 degenerate path — all-zero (n_users, 1) tensors, n_features=0",
      z_ids.shape == (3, 1) and bool((z_w == 0).all()) and z_nf == 0,
      f"{tuple(z_ids.shape)}, nf={z_nf}")

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
