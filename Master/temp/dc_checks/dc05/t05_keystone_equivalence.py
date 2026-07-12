# dc05 t05 — KEYSTONE: feature_user_proto at V=0 (ID-only) == user_proto (U-ProtoMF), by
# weight-copy (4b C5′; dc01 i02′ pattern; brief M3 amplified floor). RNG draw order differs, so
# equivalence is proven by COPYING the three parameter tensors from user_proto into
# feature_user_proto and asserting BIT-IDENTICAL forward scores plus per-parameter gradient
# equality. This is the function part of A1 (vault 2026-07-12_1739); the objective terms are t06.
# Run: python Master/temp/dc_checks/dc05/t05_keystone_equivalence.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import (make_check, feature_user_proto_param, user_proto_param, build_recsys,
                      toy_batch, zero_word_payload)

check, FAILS = make_check()
torch.manual_seed(0)
N_USERS, N_ITEMS, D, KU = 6, 8, 7, 4

ids0, w0, nf0 = zero_word_payload(N_USERS)

# A = feature_user_proto (V=0, use_id_feature=True); B = user_proto. Both use_bias=0.
A = build_recsys(feature_user_proto_param(D, KU, ids0, w0, nf0, use_id_feature=True,
                                          feature_fields=[]),
                 N_USERS, N_ITEMS, use_bias=0)
B = build_recsys(user_proto_param(D, KU), N_USERS, N_ITEMS, use_bias=0)

A_up, B_up = A.user_feature_extractor, B.user_feature_extractor
A_i, B_i = A.item_feature_extractor, B.item_feature_extractor

# V=0 + use_id_feature -> the HistoryFeatureEmbedding table has exactly n_users ID rows == B's
# free user table.
check("t05.user_table_shapes_match",
      tuple(A_up.embedding_ext.embedding_layer.weight.shape)
      == tuple(B_up.embedding_ext.embedding_layer.weight.shape) == (N_USERS, D))
check("t05.item_table_shapes_match",
      tuple(A_i.embedding_layer.weight.shape) == tuple(B_i.embedding_layer.weight.shape)
      == (N_ITEMS, KU))

# Copy ALL params B -> A (so A becomes a numerical clone of B at V=0).
with torch.no_grad():
    A_up.embedding_ext.embedding_layer.weight.copy_(B_up.embedding_ext.embedding_layer.weight)
    A_up.prototypes.copy_(B_up.prototypes)
    A_i.embedding_layer.weight.copy_(B_i.embedding_layer.weight)

u, i, labels = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2, seed=3)

logits_A = A(u, i)
logits_B = B(u, i)
check("t05.forward_scores_bit_identical", torch.equal(logits_A, logits_B),
      f"max abs score diff {(logits_A - logits_B).abs().max().item():.3e}")

loss_A = A.loss_func(logits_A, labels)
loss_B = B.loss_func(logits_B, labels)
check("t05.loss_bit_identical", loss_A.item() == loss_B.item(),
      f"|Δ| {abs(loss_A.item() - loss_B.item()):.3e}")

reg_A = A_up.get_and_reset_loss()
reg_B = B_up.get_and_reset_loss()
check("t05.inclusion_regularizers_bit_identical", float(reg_A) == float(reg_B),
      f"|Δ| {abs(float(reg_A) - float(reg_B)):.3e}")

(loss_A + reg_A).backward() if torch.is_tensor(reg_A) else loss_A.backward()
(loss_B + reg_B).backward() if torch.is_tensor(reg_B) else loss_B.backward()
pairs = [
    ("user_table(ID block)", A_up.embedding_ext.embedding_layer.weight,
     B_up.embedding_ext.embedding_layer.weight),
    ("P^u", A_up.prototypes, B_up.prototypes),
    ("item_table", A_i.embedding_layer.weight, B_i.embedding_layer.weight),
]
all_equal = True
for name, pa, pb in pairs:
    ga, gb = pa.grad, pb.grad
    ok = ga is not None and gb is not None and torch.equal(ga, gb)
    all_equal = all_equal and ok
    err = (ga - gb).abs().max().item() if (ga is not None and gb is not None) else float('inf')
    check(f"t05.grad[{name}]_bit_identical", ok, f"max abs grad diff {err:.3e}")

check("t05.KEYSTONE_overall", torch.equal(logits_A, logits_B) and all_equal,
      "fU(V=0, +ID) ≡ user_proto, bit-identical fwd + grads")

print("\nt05: ALL PASS (KEYSTONE)" if not FAILS else f"\nt05: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
