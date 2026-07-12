# dc01 I3 — gradient reaches a feature row from BOTH UI-score halves (u*·t̂ and û·t*).
# Confirms the shared-instance tie routes gradient through both the projection (t̂ = W^t q_i)
# and the cosine (t* = sim(q_i, P^t)). Integration analogue of check_claims C3.
# Run: python Master/temp/dc_checks/dc01/i03_gradient_both_halves.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check, feature_item_proto_param, build_recsys, toy_batch

check, FAILS = make_check()

torch.manual_seed(0)
N_USERS, N_ITEMS, D, KU, KT, F, NF = 5, 6, 8, 3, 4, 2, 7
feature_ids = torch.randint(0, NF, (N_ITEMS, F))

rec = build_recsys(feature_item_proto_param(D, KU, KT, feature_ids, NF, use_id_feature=True),
                   N_USERS, N_ITEMS)
item_feat_embed = rec.item_feature_extractor.model_1.embedding_ext

u, i, _ = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2, seed=5)
# a metadata feature row used by the first positive item
used_row = int(feature_ids[int(i[0, 0]), 0])


def half_grad_on_row(which):
    rec.zero_grad(set_to_none=True)
    u_out = rec.user_feature_extractor(u)
    i_out = rec.item_feature_extractor(i)
    if which == 1:  # u*·t̂  (first KU dims)
        half = (u_out[:, :KU].unsqueeze(1) * i_out[..., :KU]).sum()
    else:           # û·t*  (last KT dims)
        half = (u_out[:, KU:].unsqueeze(1) * i_out[..., KU:]).sum()
    half.backward()
    g = item_feat_embed.embedding_layer.weight.grad
    return g[used_row].norm().item()


n1 = half_grad_on_row(1)
n2 = half_grad_on_row(2)
check("i03.grad_from_ustar_dot_that", n1 > 1e-8, f"||∂(u*·t̂)/∂e_row|| = {n1:.3e}")
check("i03.grad_from_uhat_dot_tstar", n2 > 1e-8, f"||∂(û·t*)/∂e_row|| = {n2:.3e}")

print("\ni03: ALL PASS" if not FAILS else f"\ni03: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
