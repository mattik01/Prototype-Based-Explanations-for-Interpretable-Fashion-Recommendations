# dc01 I2′ — KEYSTONE: feature_item_proto at F=0 (ID-only) == item_proto (I-ProtoMF), by
# weight-copy. Re-pointed at the SC.3 re-run (2026-07-12): the host is I-ProtoMF (was
# user_item_proto under the UI build — that keystone lives in git history). RNG draw order
# differs, so equivalence is proven by COPYING the three parameter tensors from item_proto into
# feature_item_proto and asserting BIT-IDENTICAL forward scores (the fI re-check C5′ found exact
# bit-identity, not tolerance-level) plus per-parameter gradient equality.
# Run: python Master/temp/dc_checks/dc01/i02_keystone_equivalence.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import (make_check, feature_item_proto_param, item_proto_param, build_recsys, toy_batch)

check, FAILS = make_check()

torch.manual_seed(0)
N_USERS, N_ITEMS, D, KT = 5, 6, 8, 4
feature_ids_F0 = torch.zeros((N_ITEMS, 0), dtype=torch.long)  # F=0 metadata

# A = feature_item_proto (F=0, use_id_feature=True);  B = item_proto. Both use_bias=0.
A = build_recsys(feature_item_proto_param(D, KT, feature_ids_F0, 0, use_id_feature=True),
                 N_USERS, N_ITEMS, use_bias=0)
B = build_recsys(item_proto_param(D, KT), N_USERS, N_ITEMS, use_bias=0)

# Handles: A item = PrototypeEmbedding(ext=FeatureEmbedding); B item = PrototypeEmbedding(ext=Embedding)
A_ip, B_ip = A.item_feature_extractor, B.item_feature_extractor
A_u, B_u = A.user_feature_extractor, B.user_feature_extractor

# F=0 + use_id_feature -> the FeatureEmbedding table has exactly n_items ID rows == B's item table.
check("i02.item_table_shapes_match",
      tuple(A_ip.embedding_ext.embedding_layer.weight.shape) == tuple(B_ip.embedding_ext.embedding_layer.weight.shape)
      == (N_ITEMS, D),
      f"A {tuple(A_ip.embedding_ext.embedding_layer.weight.shape)} vs B {tuple(B_ip.embedding_ext.embedding_layer.weight.shape)}")
check("i02.user_table_shapes_match",
      tuple(A_u.embedding_layer.weight.shape) == tuple(B_u.embedding_layer.weight.shape) == (N_USERS, KT),
      f"A {tuple(A_u.embedding_layer.weight.shape)} vs B {tuple(B_u.embedding_layer.weight.shape)}")

# Copy ALL params B -> A (so A becomes a numerical clone of B at F=0).
with torch.no_grad():
    A_u.embedding_layer.weight.copy_(B_u.embedding_layer.weight)                                # user table (Kt)
    A_ip.prototypes.copy_(B_ip.prototypes)                                                      # P^t
    A_ip.embedding_ext.embedding_layer.weight.copy_(B_ip.embedding_ext.embedding_layer.weight)  # item table

u, i, labels = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2, seed=3)

# Forward equivalence — BIT-IDENTICAL (same modules, same op sequence; the F=0 composition is a
# sum over a single ID row, which is an exact identity)
logits_A = A(u, i)
logits_B = B(u, i)
check("i02.forward_scores_bit_identical", torch.equal(logits_A, logits_B),
      f"max abs score diff {(logits_A - logits_B).abs().max().item():.3e}")

# Loss equivalence (incl. the in-batch reg losses, which depend only on identical sim matrices)
loss_A = A.loss_func(logits_A, labels)
loss_B = B.loss_func(logits_B, labels)
check("i02.loss_bit_identical", loss_A.item() == loss_B.item(),
      f"|loss_A - loss_B| {abs(loss_A.item() - loss_B.item()):.3e}")

# Gradient equivalence, per-parameter pair
loss_A.backward()
loss_B.backward()
pairs = [
    ("user_table", A_u.embedding_layer.weight, B_u.embedding_layer.weight),
    ("P^t", A_ip.prototypes, B_ip.prototypes),
    ("item_table", A_ip.embedding_ext.embedding_layer.weight, B_ip.embedding_ext.embedding_layer.weight),
]
all_equal = True
for name, pa, pb in pairs:
    ga, gb = pa.grad, pb.grad
    ok = ga is not None and gb is not None and torch.equal(ga, gb)
    all_equal = all_equal and ok
    err = (ga - gb).abs().max().item() if (ga is not None and gb is not None) else float('inf')
    check(f"i02.grad[{name}]_bit_identical", ok, f"max abs grad diff {err:.3e}")

check("i02.KEYSTONE_overall", torch.equal(logits_A, logits_B) and all_equal,
      "fI(F=0, +ID) ≡ item_proto, bit-identical fwd + grads")

print("\ni02: ALL PASS (KEYSTONE)" if not FAILS else f"\ni02: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
