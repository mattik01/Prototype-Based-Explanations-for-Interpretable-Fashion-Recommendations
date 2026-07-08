# i03 — the two UI-score halves route gradients to their own item branch only,
# and the full score reaches both (proves the concat wiring; both branches consume the same X).
import sys

import torch

from _harness import make_check, toy_attr_data, attr_item_proto_param, build_recsys, toy_batch

torch.manual_seed(13)
check, fails = make_check()

N_u, M, d, Lu, K = 9, 13, 8, 4, 5
X, offs = toy_attr_data(M=M, V_fs=(3, 4, 5), seed=0)

rec = build_recsys(attr_item_proto_param(d, Lu, K, X, offs), N_u, M, use_bias=0)
user_fe, item_fe = rec.user_feature_extractor, rec.item_feature_extractor
A = item_fe.model_1.prototypes
W_t = item_fe.model_2.linear_layer.weight

u, i, _ = toy_batch(N_u, M, B=4, n_neg=2, seed=1)


def grad_norm(p):
    return 0.0 if p.grad is None else float(p.grad.norm())


def half_loss(which):
    u_embed = user_fe(u)            # (B, Lu+K) = [u* ; û]
    i_embed = item_fe(i)            # (B, n, Lu+K) = [t̂ ; t*]  (invert=True)
    if which == 'proj':             # u*ᵀ t̂ — the projection half
        dots = (u_embed[:, :Lu].unsqueeze(1) * i_embed[..., :Lu]).sum(-1)
    else:                           # ûᵀ t* — the prototype half
        dots = (u_embed[:, Lu:].unsqueeze(1) * i_embed[..., Lu:]).sum(-1)
    user_fe.get_and_reset_loss()
    item_fe.get_and_reset_loss()
    return (dots ** 2).sum()


# --- projection half only: W_t gets gradient, A does not ---
rec.zero_grad()
half_loss('proj').backward()
check("proj half: W_t grad nonzero", grad_norm(W_t) > 0, f"{grad_norm(W_t):.4f}")
check("proj half: A grad zero", grad_norm(A) == 0.0, f"{grad_norm(A):.4f}")

# --- prototype half only: A gets gradient, W_t does not ---
rec.zero_grad()
half_loss('proto').backward()
check("proto half: A grad nonzero", grad_norm(A) > 0, f"{grad_norm(A):.4f}")
check("proto half: W_t grad zero", grad_norm(W_t) == 0.0, f"{grad_norm(W_t):.4f}")

# --- full score: both halves receive gradient ---
rec.zero_grad()
out = rec(u, i)
(out ** 2).sum().backward()
user_fe.get_and_reset_loss()
item_fe.get_and_reset_loss()
check("full score: A grad nonzero", grad_norm(A) > 0)
check("full score: W_t grad nonzero", grad_norm(W_t) > 0)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
