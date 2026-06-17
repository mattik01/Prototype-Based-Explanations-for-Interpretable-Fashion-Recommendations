# dc01 P8 unit test — intrinsic per-feature read-out (exact decomposition + global profile).
# Run: python Master/temp/dc_checks/dc01/t08_feature_readout.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

torch.set_default_dtype(torch.float64)
TOL = 1e-10

from utilities.explanations.feature_readout import (
    per_feature_shares, activation_from_shares, global_prototype_profile, item_feature_rows)
from feature_extraction.feature_extractors import FeatureEmbedding

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


def shifted_cos(x, P):
    return 1.0 + (P @ x) / (x.norm() * P.norm(dim=1))


# --- (1) C1 exactness: shares sum to t*_k - 1 (NOT t*_k) ---
max_err = 0.0
max_recon = 0.0
for seed in range(10):
    g = torch.Generator().manual_seed(seed)
    d, Kt, V, M = 8, 4, 20, 6
    E = torch.randn(V + M, d, generator=g) * (10.0 ** (seed % 3 - 1))
    Pt = torch.randn(Kt, d, generator=g)
    for F in [0, 1, 3, 7]:
        rows = torch.randperm(V, generator=g)[:F].tolist() + [V + (seed % M)]
        E_rows = E[rows]
        q = E_rows.sum(0)
        tstar = shifted_cos(q, Pt)
        shares = per_feature_shares(E_rows, Pt)
        err = (tstar - 1.0 - shares.sum(dim=0)).abs().max().item()
        recon = (activation_from_shares(shares) - tstar).abs().max().item()
        max_err = max(max_err, err)
        max_recon = max(max_recon, recon)
check("t08.shares_sum_to_tstar_minus_1", max_err < TOL, f"max abs err {max_err:.3e} (10 seeds x F in 0,1,3,7)")
check("t08.activation_from_shares_reconstructs_tstar", max_recon < TOL, f"max abs err {max_recon:.3e}")

# --- (2) global_prototype_profile shape + range ---
g = torch.Generator().manual_seed(123)
V, d, Kt = 20, 8, 4
E = torch.randn(V, d, generator=g)
Pt = torch.randn(Kt, d, generator=g)
prof = global_prototype_profile(E, Pt)
check("t08.profile_shape", tuple(prof.shape) == (V, Kt), f"{tuple(prof.shape)}")
check("t08.profile_in_range", bool((prof >= -1 - 1e-9).all() and (prof <= 1 + 1e-9).all()),
      f"min {prof.min().item():.4f}, max {prof.max().item():.4f}")
# zero row -> cosine 0 (no NaN)
E2 = E.clone(); E2[0] = 0.0
prof2 = global_prototype_profile(E2, Pt)
check("t08.profile_zero_row_is_zero_not_nan",
      bool(torch.isfinite(prof2).all()) and prof2[0].abs().max().item() < 1e-9,
      f"row0 max |cos| {prof2[0].abs().max().item():.2e}")

# --- (3) item_feature_rows ties to the actual FeatureEmbedding forward ---
torch.manual_seed(0)
n_objects, Ffields, n_features = 6, 3, 12
feature_ids = torch.randint(0, n_features, (n_objects, Ffields))
fe = FeatureEmbedding(n_objects, feature_ids, n_features, d, use_id_feature=True)
fe.init_parameters()
item = 2
with torch.no_grad():
    q_model = fe(torch.tensor([item]))[0]      # (1,) idx -> (1,d) -> (d,)
    rows = item_feature_rows(fe, item)         # (F+1, d)
    q_rows = rows.sum(0)
    rows_err = (q_model - q_rows).abs().max().item()
check("t08.item_feature_rows_sum_eq_forward", rows_err < 1e-6, f"max abs err {rows_err:.2e}")
check("t08.item_feature_rows_count", rows.shape[0] == Ffields + 1, f"{rows.shape[0]} (expect F+1={Ffields + 1})")

# read-out wired end-to-end: shares from the model's own rows reconstruct its shifted-cos activation
Pt_d = torch.randn(Kt, d)
with torch.no_grad():
    shares = per_feature_shares(rows, Pt_d)
    tstar = shifted_cos(q_model, Pt_d)
    e2e_err = (activation_from_shares(shares) - tstar).abs().max().item()
check("t08.readout_end_to_end", e2e_err < 1e-8, f"max abs err {e2e_err:.2e}")

print("\nt08: ALL PASS" if not FAILS else f"\nt08: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
