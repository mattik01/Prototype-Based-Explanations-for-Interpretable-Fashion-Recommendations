# dc01 P2 unit test — FeatureEmbeddingW (projection over a SHARED FeatureEmbedding).
# Run: python Master/temp/dc_checks/dc01/t02_feature_embedding_w.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from feature_extraction.feature_extractors import FeatureEmbedding, FeatureEmbeddingW

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


torch.manual_seed(1)
n_objects, F, n_features, d, out_dim = 6, 3, 12, 8, 5
feature_ids = torch.randint(0, n_features, (n_objects, F))

shared = FeatureEmbedding(n_objects, feature_ids, n_features, d, use_id_feature=True)
proj = FeatureEmbeddingW(shared=shared, out_dimension=out_dim)
shared.init_parameters()
proj.init_parameters()

bn = torch.tensor([[0, 2, 4], [1, 3, 5]])
out = proj(bn)
check("t02.output_shape", tuple(out.shape) == (2, 3, out_dim), f"{tuple(out.shape)}")

# embedding identity is shared (is)
check("t02.shared_instance_identity", proj.shared is shared, "proj.shared is shared")

# forward == linear(shared(idxs))
with torch.no_grad():
    manual = proj.linear_layer(shared(bn))
    err = (out - manual).abs().max().item()
check("t02.forward_eq_linear_of_shared", err < 1e-6, f"max abs err {err:.2e}")

# calling init_parameters does NOT change the shared embedding's weights
w_before = shared.embedding_layer.weight.detach().clone()
proj.init_parameters()
w_after = shared.embedding_layer.weight.detach().clone()
delta = (w_before - w_after).abs().max().item()
check("t02.init_does_not_touch_shared", delta == 0.0, f"max abs delta {delta:.2e}")

print("\nt02: ALL PASS" if not FAILS else f"\nt02: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
