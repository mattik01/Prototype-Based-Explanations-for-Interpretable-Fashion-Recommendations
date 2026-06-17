# dc01 I6a — LOCAL regression build-check: existing models still build/forward/backward unchanged
# after the P3 PrototypeEmbedding edit (embedding_ext=None default path). t03 already proved unit
# byte-identity; this checks the assembled RecSys for user_item_proto and mf on a toy batch.
# Run: python Master/temp/dc_checks/dc01/i06a_regression_buildcheck.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check, user_item_proto_param, mf_param, build_recsys, toy_batch

check, FAILS = make_check()

torch.manual_seed(0)
N_USERS, N_ITEMS, D, KU, KT = 5, 6, 8, 3, 4
u, i, labels = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2, seed=11)

cases = [
    ('user_item_proto', user_item_proto_param(D, KU, KT)),
    ('mf', mf_param(D)),
]
for name, param in cases:
    rec = build_recsys(param, N_USERS, N_ITEMS)
    logits = rec(u, i)
    check(f"i06a.{name}.logits_shape", tuple(logits.shape) == (4, 3), f"{tuple(logits.shape)}")
    check(f"i06a.{name}.logits_finite", bool(torch.isfinite(logits).all()), "")
    loss = rec.loss_func(logits, labels)
    loss.backward()
    grads = [p.grad for p in rec.parameters() if p.requires_grad]
    nonzero = any(g is not None and g.abs().sum().item() > 0 for g in grads)
    check(f"i06a.{name}.backward_grads_nonzero", bool(torch.isfinite(loss)) and nonzero,
          f"loss={loss.item():.4f}")

print("\ni06a: ALL PASS" if not FAILS else f"\ni06a: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
