# t01 — AttributeLookup: parameter-free multi-hot base (dc02 P1).
import sys

import torch

from _harness import make_check, toy_attr_data

from feature_extraction.feature_extractors import AttributeLookup

check, fails = make_check()

M, V_fs = 12, (3, 4, 5)
X, field_offsets = toy_attr_data(M=M, V_fs=V_fs, seed=0)
V = X.shape[1]

m = AttributeLookup(X)

# --- shapes & values ---
idx1 = torch.tensor([0, 3, 7])
out1 = m(idx1)
check("1-dim idxs -> (B, V)", out1.shape == (3, V), f"{tuple(out1.shape)}")
check("1-dim rows equal X rows", torch.equal(out1, X[idx1]))

idx2 = torch.tensor([[0, 1, 2], [5, 6, 11]])
out2 = m(idx2)
check("2-dim idxs -> (B, n, V)", out2.shape == (2, 3, V), f"{tuple(out2.shape)}")
check("2-dim rows equal X rows", torch.equal(out2, X[idx2]))

# --- dtype: non-float input is stored as float32 ---
m_int = AttributeLookup(X.long())
check("buffer is float32 even from int input", m_int.attr_multi_hot.dtype == torch.float32)
# forward of the INT-built lookup must still emit float32 (this is the non-vacuous check —
# using m here instead of m_int would pass trivially since m's buffer is already float).
check("forward output float32 (from int-built lookup)", m_int(idx1).dtype == torch.float32)

# --- parameter-free / loss-free ---
check("no parameters", list(m.parameters()) == [])
check("get_and_reset_loss == 0.0", m.get_and_reset_loss() == 0.0)
check("buffer requires_grad False", m.attr_multi_hot.requires_grad is False)

# --- non-persistent buffer: excluded from state_dict ---
check("attr_multi_hot NOT in state_dict", 'attr_multi_hot' not in m.state_dict())
check("state_dict is empty", len(m.state_dict()) == 0, f"keys={list(m.state_dict().keys())}")

# --- exposed metadata ---
check("n_objects == M", m.n_objects == M)
check("n_attr_values == V", m.n_attr_values == V)

# --- rank contract: 3-dim idxs rejected (host convention) ---
try:
    m(torch.zeros(2, 2, 2, dtype=torch.long))
    check("rank-3 idxs raise", False)
except AssertionError:
    check("rank-3 idxs raise", True)

# --- rank-1 vs rank-2 same values for same indices ---
check("rank-1/rank-2 consistent",
      torch.equal(m(torch.tensor([4, 9])), m(torch.tensor([[4, 9]]))[0]))

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
