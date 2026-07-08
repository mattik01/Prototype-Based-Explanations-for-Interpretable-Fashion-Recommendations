# i01 — factory -> RecSys -> init -> forward -> loss -> backward (dc02 P5).
# Also claim 5: gradient reaches all learned params, X receives none.
import sys

import torch

from _harness import make_check, toy_attr_data, attr_item_proto_param, build_recsys, toy_batch

torch.manual_seed(5)
check, fails = make_check()

N_u, M, d, Lu, K = 9, 13, 8, 4, 5
X, offs = toy_attr_data(M=M, V_fs=(3, 4, 5), seed=0)

rec = build_recsys(attr_item_proto_param(d, Lu, K, X, offs), N_u, M, use_bias=0, loss='bce')

# --- no bias modules under use_bias=0 (dc02 §3.5(e)) ---
check("use_bias False", rec.use_bias is False)
check("no user/item bias modules", not hasattr(rec, 'user_bias') and not hasattr(rec, 'item_bias'))

u, i, labels = toy_batch(N_u, M, B=4, n_neg=2, seed=0)
out = rec(u, i)
check("logits shape (B, 1+n_neg)", out.shape == (4, 3), f"{tuple(out.shape)}")
check("logits finite", bool(torch.isfinite(out).all()))

loss = rec.loss_func(out, labels)
check("loss finite", bool(torch.isfinite(loss).all()), f"{float(loss):.4f}")
loss.backward()

user_fe, item_fe = rec.user_feature_extractor, rec.item_feature_extractor
named = {
    'user table': user_fe.model_1.embedding_ext.embedding_layer.weight,
    'user prototypes': user_fe.model_1.prototypes,
    'user projection W_u': user_fe.model_2.linear_layer.weight,
    'item prototypes A': item_fe.model_1.prototypes,
    'item projection W_t': item_fe.model_2.linear_layer.weight,
}
for name, p in named.items():
    check(f"grad reaches {name}", p.grad is not None and float(p.grad.norm()) > 0,
          f"|grad|={float(p.grad.norm()):.4f}" if p.grad is not None else "grad None")

X_buf = item_fe.model_1.embedding_ext.attr_multi_hot
check("claim 5: X requires_grad False", X_buf.requires_grad is False)
check("claim 5: X.grad is None", X_buf.grad is None)

# exactly 5 learned parameter tensors in the whole model (no biases, no per-item table);
# the user table is SHARED between the two user branches (weight tie) -> counted once.
n_params = len({id(p) for p in rec.parameters()})
check("exactly 5 distinct parameter tensors", n_params == 5, f"{n_params}")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
