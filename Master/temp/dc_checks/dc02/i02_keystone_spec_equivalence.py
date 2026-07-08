# i02 — KEYSTONE: the factory-built repo model is numerically equivalent to the independent
# toy mechanism that the 11 claims were verified on (check_claims.py, frozen).
# The spec functions below are copied VERBATIM from check_claims.py lines 42-81 (sc / forward /
# regs), with only the closure constants (X, K) turned into arguments — do not let them drift.
# Weight mapping repo -> spec (nn.Linear stores (out, in), the spec stores (in, out)):
#   U   := user table (tied)             P   := user prototypes
#   W_u := user EmbeddingW weight.T      A   := item prototypes
#   W_t := AttributeProjection weight.T
import sys

import torch

from _harness import make_check, toy_attr_data, attr_item_proto_param, build_recsys, toy_batch

torch.manual_seed(23)
check, fails = make_check()

N_u, M, d, Lu, K = 9, 13, 8, 4, 5
X, offs = toy_attr_data(M=M, V_fs=(3, 4, 5), seed=0)


# ---- spec mechanism (verbatim from check_claims.py, closure vars -> args) ----
def sc(a, b):
    """shifted cosine: a (..., D), b (R, D) -> (..., R)"""
    num = a @ b.t()
    den = a.norm(dim=-1, keepdim=True) * b.norm(dim=-1)
    return 1.0 + num / den


def spec_forward(params, u_idxs, i_idxs, Xm):
    u = params["U"][u_idxs]                     # (B, d)
    u_star = sc(u, params["P"])                 # (B, L_u)
    u_hat = u @ params["W_u"]                   # (B, K)
    x = Xm[i_idxs]                              # (B, n, V)
    t_star = sc(x, params["A"])                 # (B, n, K)
    t_hat = x @ params["W_t"]                   # (B, n, L_u)
    score = (u_star.unsqueeze(1) * t_hat).sum(-1) + \
            (u_hat.unsqueeze(1) * t_star).sum(-1)   # (B, n)
    return score, t_star


def spec_regs(t_star, K_):
    S = t_star.reshape(-1, K_)                  # (B*n, K)
    R_batch = -S.max(dim=1).values.mean()
    R_proto = -S.max(dim=0).values.mean()
    return R_batch, R_proto


# ---- repo model ----
rec = build_recsys(attr_item_proto_param(d, Lu, K, X, offs), N_u, M, use_bias=0)
user_fe, item_fe = rec.user_feature_extractor, rec.item_feature_extractor

# ---- weight-copy repo -> spec ----
params = {
    "U": user_fe.model_1.embedding_ext.embedding_layer.weight.detach().clone().requires_grad_(True),
    "P": user_fe.model_1.prototypes.detach().clone().requires_grad_(True),
    "W_u": user_fe.model_2.linear_layer.weight.detach().clone().T.contiguous().requires_grad_(True),
    "A": item_fe.model_1.prototypes.detach().clone().requires_grad_(True),
    "W_t": item_fe.model_2.linear_layer.weight.detach().clone().T.contiguous().requires_grad_(True),
}

u, i, _ = toy_batch(N_u, M, B=6, n_neg=3, seed=2)

# ---- scores match ----
score_spec, t_star_spec = spec_forward(params, u, i, X)
score_repo = rec(u, i)
err = (score_spec - score_repo).abs().max()
check("scores match the spec mechanism", float(err) < 1e-5, f"max err {float(err):.2e}")

# ---- regularizer values match (isolate R_proto / R_batch via the weight knobs) ----
rb_spec, rp_spec = spec_regs(t_star_spec, K)
item_fe.get_and_reset_loss()   # drain the forward above
user_fe.get_and_reset_loss()

state = rec.state_dict()
rec_p = build_recsys(attr_item_proto_param(d, Lu, K, X, offs,
                                           sim_proto_weight=1.0, sim_batch_weight=0.0), N_u, M)
rec_p.load_state_dict(state)
rec_p(u, i)
rp_repo = rec_p.item_feature_extractor.model_1.get_and_reset_loss()
rec_b = build_recsys(attr_item_proto_param(d, Lu, K, X, offs,
                                           sim_proto_weight=0.0, sim_batch_weight=1.0), N_u, M)
rec_b.load_state_dict(state)
rec_b(u, i)
rb_repo = rec_b.item_feature_extractor.model_1.get_and_reset_loss()
check("R_proto matches spec", abs(float(rp_repo) - float(rp_spec)) < 1e-6,
      f"{float(rp_repo):.6f} vs {float(rp_spec):.6f}")
check("R_batch matches spec", abs(float(rb_repo) - float(rb_spec)) < 1e-6,
      f"{float(rb_repo):.6f} vs {float(rb_spec):.6f}")

# ---- per-parameter gradients match under the mapping ----
loss_spec = (score_spec ** 2).sum()
loss_spec.backward()
rec.zero_grad()
out = rec(u, i)
(out ** 2).sum().backward()
item_fe.get_and_reset_loss()
user_fe.get_and_reset_loss()

repo_grads = {
    "U": user_fe.model_1.embedding_ext.embedding_layer.weight.grad,
    "P": user_fe.model_1.prototypes.grad,
    "W_u": user_fe.model_2.linear_layer.weight.grad.T,
    "A": item_fe.model_1.prototypes.grad,
    "W_t": item_fe.model_2.linear_layer.weight.grad.T,
}
for name in params:
    g_err = (params[name].grad - repo_grads[name]).abs().max()
    check(f"grad({name}) matches spec", float(g_err) < 1e-4,
          f"max err {float(g_err):.2e}, |g|={float(repo_grads[name].norm()):.3f}")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
