# dc01 I2 — KEYSTONE: feature_item_proto at F=0 (ID-only) == user_item_proto, by weight-copy.
# RNG draw order differs (the shared design allocates no pre-tie item table), so equivalence is
# proven by COPYING the shared params from user_item_proto into feature_item_proto and asserting
# forward scores AND per-parameter gradients match to 1e-6. Ties check_claims C5.
# Run: python Master/temp/dc_checks/dc01/i02_keystone_equivalence.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import (make_check, feature_item_proto_param, user_item_proto_param, build_recsys, toy_batch)

check, FAILS = make_check()

torch.manual_seed(0)
N_USERS, N_ITEMS, D, KU, KT = 5, 6, 8, 3, 4
feature_ids_F0 = torch.zeros((N_ITEMS, 0), dtype=torch.long)  # F=0 metadata

# A = feature_item_proto (F=0, use_id_feature=True);  B = user_item_proto. Both use_bias=0.
A = build_recsys(feature_item_proto_param(D, KU, KT, feature_ids_F0, 0, use_id_feature=True),
                 N_USERS, N_ITEMS, use_bias=0)
B = build_recsys(user_item_proto_param(D, KU, KT), N_USERS, N_ITEMS, use_bias=0)

# Handles
A_up, A_ue = A.user_feature_extractor.model_1, A.user_feature_extractor.model_2
A_ip, A_iproj = A.item_feature_extractor.model_1, A.item_feature_extractor.model_2
B_up, B_ue = B.user_feature_extractor.model_1, B.user_feature_extractor.model_2
B_ip, B_iembed = B.item_feature_extractor.model_1, B.item_feature_extractor.model_2

# F=0 + use_id_feature -> the FeatureEmbedding table has exactly n_items ID rows == B's item table.
check("i02.item_table_shapes_match",
      tuple(A_ip.embedding_ext.embedding_layer.weight.shape) == tuple(B_ip.embedding_ext.embedding_layer.weight.shape)
      == (N_ITEMS, D),
      f"A {tuple(A_ip.embedding_ext.embedding_layer.weight.shape)} vs B {tuple(B_ip.embedding_ext.embedding_layer.weight.shape)}")

# Copy ALL shared params B -> A (so A becomes a numerical clone of B at F=0).
with torch.no_grad():
    A_up.prototypes.copy_(B_up.prototypes)                                              # P^u
    A_up.embedding_ext.embedding_layer.weight.copy_(B_up.embedding_ext.embedding_layer.weight)  # user table (tied)
    A_ue.linear_layer.weight.copy_(B_ue.linear_layer.weight)                            # W^u (user proj)
    A_ip.prototypes.copy_(B_ip.prototypes)                                              # P^t
    A_ip.embedding_ext.embedding_layer.weight.copy_(B_ip.embedding_ext.embedding_layer.weight)  # item table (shared/tied)
    A_iproj.linear_layer.weight.copy_(B_iembed.linear_layer.weight)                     # W^t (item proj)

u, i, labels = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2, seed=3)

# Forward equivalence
logits_A = A(u, i)
logits_B = B(u, i)
fwd_err = (logits_A - logits_B).abs().max().item()
check("i02.forward_scores_allclose", fwd_err < 1e-6, f"max abs score diff {fwd_err:.3e}")

# Loss equivalence (incl. the in-batch reg losses, which depend only on identical sim matrices)
loss_A = A.loss_func(logits_A, labels)
loss_B = B.loss_func(logits_B, labels)
loss_err = abs(loss_A.item() - loss_B.item())
check("i02.loss_allclose", loss_err < 1e-6, f"|loss_A - loss_B| {loss_err:.3e}")

# Gradient equivalence, per-parameter pair
loss_A.backward()
loss_B.backward()
pairs = [
    ("P^u", A_up.prototypes, B_up.prototypes),
    ("user_table", A_up.embedding_ext.embedding_layer.weight, B_up.embedding_ext.embedding_layer.weight),
    ("W^u_userproj", A_ue.linear_layer.weight, B_ue.linear_layer.weight),
    ("P^t", A_ip.prototypes, B_ip.prototypes),
    ("item_table", A_ip.embedding_ext.embedding_layer.weight, B_ip.embedding_ext.embedding_layer.weight),
    ("W^t_itemproj", A_iproj.linear_layer.weight, B_iembed.linear_layer.weight),
]
max_grad_err = 0.0
for name, pa, pb in pairs:
    ga, gb = pa.grad, pb.grad
    ok = ga is not None and gb is not None
    err = (ga - gb).abs().max().item() if ok else float('inf')
    max_grad_err = max(max_grad_err, err)
    check(f"i02.grad[{name}]_allclose", ok and err < 1e-6, f"max abs grad diff {err:.3e}")

check("i02.KEYSTONE_overall", fwd_err < 1e-6 and loss_err < 1e-6 and max_grad_err < 1e-6,
      f"fwd {fwd_err:.2e}, loss {loss_err:.2e}, grad {max_grad_err:.2e}")

print("\ni02: ALL PASS (KEYSTONE)" if not FAILS else f"\ni02: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
