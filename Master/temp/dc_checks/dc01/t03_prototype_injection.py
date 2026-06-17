# dc01 P3 unit test — PrototypeEmbedding.embedding_ext injection (REGRESSION-CRITICAL).
# Requires t03_golden.pt captured on the UNMODIFIED code by t03_capture_golden.py.
# Run: python Master/temp/dc_checks/dc01/t03_prototype_injection.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from utilities.utils import reproducible
from feature_extraction.feature_extractors import PrototypeEmbedding, FeatureEmbedding

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


GOLDEN = os.path.join(os.path.dirname(__file__), 't03_golden.pt')
if not os.path.exists(GOLDEN):
    print("FAIL: t03_golden.pt missing — run t03_capture_golden.py on the UNMODIFIED code first.")
    sys.exit(1)

golden = torch.load(GOLDEN)
SEED = golden['meta']['seed']
N, D, K = golden['meta']['n'], golden['meta']['d'], golden['meta']['k']
INPUT = golden['meta']['input']

# --- embedding_ext=None path must reproduce the golden bit-identically ---
reproducible(SEED)
m = PrototypeEmbedding(N, D, K, use_weight_matrix=False, cosine_type='shifted',
                       reg_proto_type='max', reg_batch_type='max')
m.init_parameters()
with torch.no_grad():
    out = m(INPUT)

# state_dict bit-identity
g_sd, m_sd = golden['state_dict'], m.state_dict()
keys_match = set(g_sd.keys()) == set(m_sd.keys())
check("t03.state_dict_keys_match", keys_match, f"golden={sorted(g_sd.keys())} vs new={sorted(m_sd.keys())}")
max_w = 0.0
if keys_match:
    for k in g_sd:
        max_w = max(max_w, (g_sd[k] - m_sd[k]).abs().max().item())
check("t03.state_dict_bit_identical", keys_match and max_w == 0.0, f"max abs diff {max_w:.2e}")

# output match. The state_dict above is the regression teeth — it must be EXACTLY identical
# (same params, same RNG draw order as the pre-edit code). The forward OUTPUT is a derived
# quantity: identical to 0.0 on the capture machine, but it drifts at float32 epsilon (~1e-7) when
# the golden was captured under a different BLAS/torch build (e.g. local torch 1.9.1 golden vs the
# LEO5 torch 2.5.1 runtime). So require allclose at a tight tolerance, not bitwise equality.
out_diff = (out - golden['output']).abs().max().item()
check("t03.output_matches", out_diff < 1e-5, f"max abs diff {out_diff:.2e} (exact on capture env; ≤1e-5 cross-BLAS)")

# --- injected path uses the passed object ---
feature_ids = torch.randint(0, 7, (N, 2))
inj = FeatureEmbedding(N, feature_ids, 7, D, use_id_feature=True)
m2 = PrototypeEmbedding(N, D, K, use_weight_matrix=False, cosine_type='shifted',
                        reg_proto_type='max', reg_batch_type='max', embedding_ext=inj)
check("t03.injected_is_passed_object", m2.embedding_ext is inj, "m2.embedding_ext is inj")

# the injected model still forwards to the right shape
m2_inj_owner_init = inj.init_parameters() or True  # single-owner init done by caller
with torch.no_grad():
    out2 = m2(INPUT)
check("t03.injected_forward_shape", tuple(out2.shape) == (INPUT.shape[0], INPUT.shape[1], K), f"{tuple(out2.shape)}")

print("\nt03: ALL PASS" if not FAILS else f"\nt03: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
