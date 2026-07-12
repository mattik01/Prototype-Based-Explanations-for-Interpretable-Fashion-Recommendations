# dc01 I1 — feature_item_proto (fI host): build -> forward -> backward sanity (toy synthetic).
# Re-pointed at the SC.3 re-run (2026-07-12): I-ProtoMF shape (user = plain Embedding in R^{Kt},
# item = PrototypeEmbedding over the composed FeatureEmbedding).
# Run: python Master/temp/dc_checks/dc01/i01_build_forward_backward.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check, feature_item_proto_param, build_recsys, toy_batch

check, FAILS = make_check()

torch.manual_seed(0)
N_USERS, N_ITEMS, D, KT, F, NF = 5, 6, 8, 4, 2, 7
feature_ids = torch.randint(0, NF, (N_ITEMS, F))

rec = build_recsys(feature_item_proto_param(D, KT, feature_ids, NF, use_id_feature=True),
                   N_USERS, N_ITEMS)

u, i, labels = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2)

# output last-dims equal (user vector · item activation, both Kt)
with torch.no_grad():
    u_out = rec.user_feature_extractor(u)
    i_out = rec.item_feature_extractor(i)
check("i01.user_item_lastdims_equal", u_out.shape[-1] == i_out.shape[-1] == KT,
      f"user {tuple(u_out.shape)}, item {tuple(i_out.shape)}")

logits = rec(u, i)
check("i01.logits_shape", tuple(logits.shape) == (4, 3), f"{tuple(logits.shape)}")
check("i01.logits_finite", bool(torch.isfinite(logits).all()), "")

loss = rec.loss_func(logits, labels)
check("i01.loss_finite_scalar", loss.dim() == 0 and bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")
loss.backward()

# grads exist & are nonzero on the key params
item_proto = rec.item_feature_extractor
item_feat_embed = item_proto.embedding_ext
fe_grad = item_feat_embed.embedding_layer.weight.grad
check("i01.grad_on_item_feat_embed", fe_grad is not None and fe_grad.abs().sum().item() > 0,
      f"||grad||={None if fe_grad is None else fe_grad.norm().item():.3e}")
check("i01.grad_on_item_prototypes",
      item_proto.prototypes.grad is not None and item_proto.prototypes.grad.abs().sum().item() > 0, "")
user_grad = rec.user_feature_extractor.embedding_layer.weight.grad
check("i01.grad_on_user_table",
      user_grad is not None and user_grad.abs().sum().item() > 0, "")

print("\ni01: ALL PASS" if not FAILS else f"\ni01: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
