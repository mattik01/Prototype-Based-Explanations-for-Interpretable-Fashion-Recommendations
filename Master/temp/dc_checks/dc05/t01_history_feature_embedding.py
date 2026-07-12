# dc05 t01 — HistoryFeatureEmbedding unit test (design doc §3.2).
# Pins: weighted gather-sum vs hand-computed; padding contributes exactly nothing; the ID row
# offset (n_features + u); (B,) and (B, n) index shapes; max_norm live on the word table;
# non-persistent buffers excluded from state_dict.
# Run: python Master/temp/dc_checks/dc05/t01_history_feature_embedding.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check

from feature_extraction.feature_extractors import HistoryFeatureEmbedding

check, FAILS = make_check()
torch.manual_seed(0)

N, V, DMAX, D = 5, 7, 3, 6
ids = torch.tensor([[1, 4, 0],
                    [2, 0, 0],
                    [0, 0, 0],    # empty history (all padding)
                    [6, 3, 2],
                    [5, 5, 0]])   # duplicate id (allowed: weights just add)
w = torch.tensor([[0.5, 0.5, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.4, 0.4, 0.2],
                  [0.3, 0.7, 0.0]])

fe = HistoryFeatureEmbedding(N, ids, w, n_features=V, embedding_dim=D, use_id_feature=True)
fe.init_parameters()
E = fe.embedding_layer.weight.detach()

# 1. hand-computed composition, user by user
with torch.no_grad():
    out = fe(torch.arange(N))
expected = torch.stack([
    0.5 * E[1] + 0.5 * E[4] + E[V + 0],
    1.0 * E[2] + E[V + 1],
    E[V + 2],                                     # empty basket -> ID row alone
    0.4 * E[6] + 0.4 * E[3] + 0.2 * E[2] + E[V + 3],
    0.3 * E[5] + 0.7 * E[5] + E[V + 4],
])
check("t01.weighted_sum_hand_computed", torch.allclose(out, expected, atol=1e-6),
      f"max abs diff {(out - expected).abs().max():.3e}")

# 2. padding contributes exactly nothing: perturb word row 0, padded users unaffected
with torch.no_grad():
    before = fe(torch.tensor([0, 1, 2]))
    fe.embedding_layer.weight[0] += 100.0
    after = fe(torch.tensor([0, 1, 2]))
    fe.embedding_layer.weight[0] -= 100.0
check("t01.padding_exact_noop", torch.equal(before, after),
      f"max abs diff {(before - after).abs().max():.3e}")

# 3. noid variant: table has V rows only; empty user composes to exactly zero
fe_noid = HistoryFeatureEmbedding(N, ids, w, n_features=V, embedding_dim=D, use_id_feature=False)
fe_noid.init_parameters()
check("t01.noid_table_rows", fe_noid.embedding_layer.weight.shape[0] == V)
with torch.no_grad():
    q2 = fe_noid(torch.tensor([2]))
check("t01.noid_empty_user_zero_vector", torch.equal(q2, torch.zeros_like(q2)),
      f"norm {q2.norm():.3e}")

# 4. (B, n) index shape matches flattened (B,) calls
idx2d = torch.tensor([[0, 1, 2], [3, 4, 0]])
with torch.no_grad():
    out2d = fe(idx2d)
    outflat = fe(idx2d.reshape(-1)).reshape(2, 3, D)
check("t01.2d_index_shape", tuple(out2d.shape) == (2, 3, D), f"{tuple(out2d.shape)}")
check("t01.2d_matches_flat", torch.equal(out2d, outflat))

# 5. max_norm live: oversized row gets renormed on lookup
fe_mn = HistoryFeatureEmbedding(N, ids, w, n_features=V, embedding_dim=D,
                                use_id_feature=True, max_norm=1.0)
with torch.no_grad():
    fe_mn.embedding_layer.weight[2] = torch.full((D,), 10.0)
    _ = fe_mn(torch.tensor([1]))  # user 1 uses word 2 -> triggers renorm
    row_norm = fe_mn.embedding_layer.weight[2].norm().item()
check("t01.max_norm_applied", abs(row_norm - 1.0) < 1e-5, f"row norm {row_norm:.4f}")

# 6. buffers non-persistent; only the embedding table round-trips
sd = fe.state_dict()
check("t01.buffers_not_in_state_dict",
      not any(k.startswith('hist_') for k in sd), f"keys: {sorted(sd)}")
check("t01.weight_in_state_dict", 'embedding_layer.weight' in sd)

# 7. shape guards fire
try:
    HistoryFeatureEmbedding(N, ids[:3], w, n_features=V, embedding_dim=D)
    guard1 = False
except AssertionError:
    guard1 = True
try:
    HistoryFeatureEmbedding(N, ids, w[:, :2], n_features=V, embedding_dim=D)
    guard2 = False
except AssertionError:
    guard2 = True
check("t01.shape_guards_fire", guard1 and guard2)

print("\nt01: ALL PASS" if not FAILS else f"\nt01: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
