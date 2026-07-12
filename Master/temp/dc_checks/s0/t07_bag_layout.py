# t07 — bag layout (S0-build extension component a, charter C5 ml-1m amendment):
# build_feature_bags + FeatureEmbedding weighted-bag mode.
# Keystones:
#   K1  bag-vs-padded BIT-IDENTITY on fixed-width (H&M-shaped) inputs — builder tensors equal,
#       FeatureEmbedding outputs bitwise equal in both id modes and both index shapes.
#   K2  F=0 reduction in bag mode: use_id_feature=True + zero fields ≡ plain Embedding, bitwise.
# Plus: multi-value semantics (vocab/offsets/weights), padding zero-contribution, the
# missingness policy ('' = short bag), and the F-S0-06-family guards.
import os
import shutil
import sys
import tempfile

import torch

from _harness import make_check, REPO_ROOT

from feature_extraction.feature_ids import build_feature_ids, build_feature_bags
from feature_extraction.feature_extractors import Embedding, FeatureEmbedding

check, fails = make_check()

tmp = tempfile.mkdtemp(prefix='s0_t07_')
torch.manual_seed(38210573)


def write_csv(rows, header):
    d = tempfile.mkdtemp(prefix='case_', dir=tmp)
    with open(os.path.join(d, 'item_features.csv'), 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    return d


def clone_feature_embedding(src: FeatureEmbedding, dst: FeatureEmbedding):
    dst.load_state_dict(src.state_dict())


# ---------------------------------------------------------------------------
# K1a — builder equivalence on fixed-width data (one value per field)
# ---------------------------------------------------------------------------
fixed = write_csv([(0, 'blue', 'square'), (1, 'green', 'square'), (2, 'red', 'circle'),
                   (3, 'blue', 'circle')], 'item_id,color,shape')
ids_f, nf_f = build_feature_ids(fixed, ['color', 'shape'])
ids_b, w_b, nf_b = build_feature_bags(fixed, ['color', 'shape'])
check("K1a: fixed-width bag ids == build_feature_ids ids", torch.equal(ids_b, ids_f),
      f"{ids_b.tolist()} vs {ids_f.tolist()}")
check("K1a: fixed-width n_features equal", nf_b == nf_f, f"{nf_b} vs {nf_f}")
check("K1a: fixed-width weights all exactly 1.0",
      w_b.shape == ids_f.shape and bool((w_b == 1.0).all()), f"{w_b.tolist()}")

# ---------------------------------------------------------------------------
# K1b — FeatureEmbedding bit-identity: fixed path vs bag path (all-ones weights)
# ---------------------------------------------------------------------------
for use_id in (True, False):
    m_fixed = FeatureEmbedding(4, ids_f, nf_f, 7, use_id_feature=use_id)
    m_bag = FeatureEmbedding(4, ids_b, nf_b, 7, use_id_feature=use_id, feature_weights=w_b)
    m_fixed.init_parameters()
    clone_feature_embedding(m_fixed, m_bag)
    idx1 = torch.arange(4)                      # 1D shape
    idx2 = torch.tensor([[0, 2, 3], [1, 1, 0]])  # 2D (batch, 1+n_neg) shape
    for label, idx in (('1D', idx1), ('2D', idx2)):
        out_fixed = m_fixed(idx)
        out_bag = m_bag(idx)
        check(f"K1b: bit-identical outputs (use_id={use_id}, {label} idxs)",
              torch.equal(out_fixed, out_bag),
              f"max|Δ|={float((out_fixed - out_bag).abs().max()):.3e}")

# ---------------------------------------------------------------------------
# K1c — real V1 (hm_1_month, canonical 5): bag builder reproduces the fixed builder exactly
# ---------------------------------------------------------------------------
V1_FIELDS = ['department_name', 'product_type_name', 'section_name',
             'colour_group_name', 'graphical_appearance_name']
v1 = os.path.join(REPO_ROOT, 'data', 'hm_1_month')
if os.path.exists(os.path.join(v1, 'item_features.csv')):
    v_ids_f, v_nf_f = build_feature_ids(v1, V1_FIELDS)
    v_ids_b, v_w_b, v_nf_b = build_feature_bags(v1, V1_FIELDS)
    check("K1c: V1 bag ids == fixed ids (13651x5)",
          torch.equal(v_ids_b, v_ids_f), f"shapes {tuple(v_ids_b.shape)}/{tuple(v_ids_f.shape)}")
    check("K1c: V1 n_features == 426 in both", v_nf_b == v_nf_f == 426, f"{v_nf_b}/{v_nf_f}")
    check("K1c: V1 weights all 1.0, no padding (D_max == F == 5)",
          v_w_b.shape == (13651, 5) and bool((v_w_b == 1.0).all()), f"{tuple(v_w_b.shape)}")
else:
    print("[SKIP] hm_1_month item_features.csv not present")

# ---------------------------------------------------------------------------
# Multi-value semantics (ml-1m-shaped): bags, offsets, short-bag missingness
# ---------------------------------------------------------------------------
# genres vocab (sorted): Action=0, Comedy=1, Drama=2 ; tags vocab (+3): dark=3, funny=4, long=5
ml = write_csv([
    (0, 'Action|Comedy', 'funny'),
    (1, 'Drama', 'dark|long'),
    (2, 'Comedy', ''),            # '' = short bag (genres-only tail — legal, disclosed)
    (3, 'Action|Drama', 'dark|funny|long'),
], 'item_id,genres,tags')
b_ids, b_w, b_nf = build_feature_bags(ml, ['genres', 'tags'])
check("bags: n_features = 3 genres + 3 tags = 6", b_nf == 6, f"{b_nf}")
check("bags: D_max = 5 (item 3: 2 genres + 3 tags)", b_ids.shape == (4, 5), f"{tuple(b_ids.shape)}")
expect_ids = torch.tensor([
    [0, 1, 4, 0, 0],   # Action, Comedy, funny + padding
    [2, 3, 5, 0, 0],   # Drama, dark, long + padding
    [1, 0, 0, 0, 0],   # Comedy + padding (short bag)
    [0, 2, 3, 4, 5],   # Action, Drama, dark, funny, long
])
expect_w = torch.tensor([
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 0., 0.],
    [1., 0., 0., 0., 0.],
    [1., 1., 1., 1., 1.],
])
check("bags: token ids as expected (field-offset encoding, field order)",
      torch.equal(b_ids, expect_ids), f"{b_ids.tolist()}")
check("bags: weights 1.0 real / 0.0 padding", torch.equal(b_w, expect_w), f"{b_w.tolist()}")

# padding zero-contribution: perturbing embedding row 0 (the padding id, also a REAL vocab row)
# must not change items whose real tokens exclude id 0.
m = FeatureEmbedding(4, b_ids, b_nf, 5, use_id_feature=True, feature_weights=b_w)
m.init_parameters()
before = m(torch.arange(4))
with torch.no_grad():
    m.embedding_layer.weight[0] += 100.0
after = m(torch.arange(4))
uses_tok0 = torch.tensor([True, False, False, True])  # items whose REAL bag contains token id 0
check("padding contributes exactly nothing (row-0 perturbation isolated to real users of token 0)",
      torch.equal(before[~uses_tok0], after[~uses_tok0]) and not torch.equal(before[uses_tok0], after[uses_tok0]),
      "")

# composition correctness: q_i == Σ_{real tokens} e_tok (+ e_ID) — manual per-item sum
m2 = FeatureEmbedding(4, b_ids, b_nf, 5, use_id_feature=True, feature_weights=b_w)
m2.init_parameters()
W = m2.embedding_layer.weight.detach()
out = m2(torch.arange(4)).detach()
manual = torch.stack([
    W[0] + W[1] + W[4] + W[6 + 0],
    W[2] + W[3] + W[5] + W[6 + 1],
    W[1] + W[6 + 2],
    W[0] + W[2] + W[3] + W[4] + W[5] + W[6 + 3],
])
check("bag composition == manual token sum (+ ID row)", torch.allclose(out, manual, atol=1e-6),
      f"max|Δ|={float((out - manual).abs().max()):.3e}")

# ---------------------------------------------------------------------------
# K2 — F=0 reduction in bag mode: fields=[] + use_id_feature=True ≡ plain Embedding, bitwise
# ---------------------------------------------------------------------------
z_ids, z_w, z_nf = build_feature_bags(ml, [])
check("K2: F=0 returns (n_items, 0) tensors, n_features=0",
      z_ids.shape == (4, 0) and z_w.shape == (4, 0) and z_nf == 0,
      f"{tuple(z_ids.shape)}, nf={z_nf}")
m_f0 = FeatureEmbedding(4, z_ids, z_nf, 6, use_id_feature=True, feature_weights=z_w)
m_ref = Embedding(4, 6)
m_ref.init_parameters()
with torch.no_grad():
    m_f0.embedding_layer.weight.copy_(m_ref.embedding_layer.weight)
idx = torch.tensor([[0, 3], [2, 1]])
check("K2: F=0 bag mode ≡ plain Embedding (bitwise)",
      torch.equal(m_f0(idx), m_ref(idx)), "")

# ---------------------------------------------------------------------------
# Guards (F-S0-06 family)
# ---------------------------------------------------------------------------
def expect_raise(name, rows, fields, header='item_id,genres,tags', substr=''):
    d = write_csv(rows, header)
    try:
        build_feature_bags(d, fields)
        check(name, False, "no raise")
    except ValueError as e:
        check(name, substr in str(e), str(e)[:80])


expect_raise("guard: literal 'nan' token raises (F-S0-06 phantom value)",
             [(0, 'Action|nan', 'x'), (1, 'Drama', 'y')], ['genres', 'tags'], substr='nan')
expect_raise("guard: duplicate token within a cell raises",
             [(0, 'Action|Action', 'x'), (1, 'Drama', 'y')], ['genres', 'tags'], substr='duplicate')
expect_raise("guard: stray separator (empty token) raises",
             [(0, 'Action|', 'x'), (1, 'Drama', 'y')], ['genres', 'tags'], substr='empty token')
expect_raise("guard: all-fields-empty item raises (degenerate composition)",
             [(0, 'Action', 'x'), (1, '', '')], ['genres', 'tags'], substr='ZERO tokens')
expect_raise("guard: missing field raises",
             [(0, 'Action', 'x')], ['genres', 'nosuch'], substr='not in item_features.csv')
expect_raise("guard: item_id coverage (gap) raises",
             [(0, 'Action', 'x'), (2, 'Drama', 'y')], ['genres', 'tags'], substr='item_id')

# shuffled/reversed rows on disk align to numeric item_id order (dtype=str sort hazard pinned:
# lexicographic '10' < '2' must NOT reorder rows)
big_rows = [(i, f'g{i % 3}', f't{i % 4}') for i in range(12)]
shuf = write_csv(list(reversed(big_rows)), 'item_id,genres,tags')
plain = write_csv(big_rows, 'item_id,genres,tags')
s_ids, s_w, s_nf = build_feature_bags(shuf, ['genres', 'tags'])
p_ids, p_w, p_nf = build_feature_bags(plain, ['genres', 'tags'])
check("rows align to NUMERIC item_id order regardless of on-disk order (n_items=12 > 10)",
      torch.equal(s_ids, p_ids) and torch.equal(s_w, p_w) and s_nf == p_nf, "")

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
