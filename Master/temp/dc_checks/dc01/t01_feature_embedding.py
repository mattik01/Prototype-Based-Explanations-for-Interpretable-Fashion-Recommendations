# dc01 P1 unit test — FeatureEmbedding (sum-of-feature-embeddings item encoder).
# Run: python Master/temp/dc_checks/dc01/t01_feature_embedding.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from feature_extraction.feature_extractors import FeatureEmbedding

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


torch.manual_seed(0)
n_objects, F, n_features, d = 6, 3, 12, 8
feature_ids = torch.randint(0, n_features, (n_objects, F))

fe = FeatureEmbedding(n_objects, feature_ids, n_features, d, use_id_feature=True)
fe.init_parameters()

# --- shapes ---
b = torch.tensor([0, 1, 2, 3])
bn = torch.tensor([[0, 2, 4], [1, 3, 5]])
out_b = fe(b)
out_bn = fe(bn)
check("t01.shape_(B,)", tuple(out_b.shape) == (4, d), f"{tuple(out_b.shape)}")
check("t01.shape_(B,1+n_neg)", tuple(out_bn.shape) == (2, 3, d), f"{tuple(out_bn.shape)}")


# --- output equals manual emb(feat).sum(-2) ---
def manual(idxs):
    feat = feature_ids[idxs]
    id_col = (idxs + n_features).unsqueeze(-1)
    feat = torch.cat([feat, id_col], dim=-1)
    return fe.embedding_layer(feat).sum(-2)


with torch.no_grad():
    err_b = (out_b - manual(b)).abs().max().item()
    err_bn = (out_bn - manual(bn)).abs().max().item()
check("t01.manual_sum_(B,)", err_b < 1e-6, f"max abs err {err_b:.2e}")
check("t01.manual_sum_(B,1+n_neg)", err_bn < 1e-6, f"max abs err {err_bn:.2e}")

# --- ID-on vs ID-off differ by EXACTLY the ID row ---
fe_off = FeatureEmbedding(n_objects, feature_ids, n_features, d, use_id_feature=False)
with torch.no_grad():
    fe_off.embedding_layer.weight.data[:n_features] = fe.embedding_layer.weight.data[:n_features]
    out_on = fe(b)
    out_off = fe_off(b)
    id_rows = fe.embedding_layer.weight.data[n_features + b]  # (4, d)
    diff_err = ((out_on - out_off) - id_rows).abs().max().item()
check("t01.id_on_minus_off_eq_id_row", diff_err < 1e-6, f"max abs err {diff_err:.2e}")

# --- F=0 + ID returns EXACTLY the ID-row vector ---
fe0 = FeatureEmbedding(n_objects, torch.zeros((n_objects, 0), dtype=torch.long), 0, d, use_id_feature=True)
fe0.init_parameters()
with torch.no_grad():
    out0 = fe0(b)
    expected0 = fe0.embedding_layer.weight.data[b]  # n_features=0 -> ID rows at idx 0..n_objects-1
    err0 = (out0 - expected0).abs().max().item()
check("t01.F0_plus_id_eq_id_row", err0 < 1e-6, f"max abs err {err0:.2e}; out0 shape {tuple(out0.shape)}")

# --- loss == 0 (no regularization) ---
loss = fe.get_and_reset_loss()
check("t01.loss_is_zero", float(loss) == 0.0, f"loss={loss}")

# --- feature_ids NOT in state_dict (persistent=False); embedding_layer.weight IS ---
sd_keys = list(fe.state_dict().keys())
check("t01.feature_ids_not_in_state_dict", not any('feature_ids' in k for k in sd_keys), f"keys={sd_keys}")
check("t01.embedding_layer_weight_in_state_dict", any('embedding_layer.weight' in k for k in sd_keys), f"keys={sd_keys}")

print("\nt01: ALL PASS" if not FAILS else f"\nt01: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
