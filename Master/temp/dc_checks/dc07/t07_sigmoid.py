# dc07 t07 — sigmoid membership: m_l = σ(⟨q,p_l⟩/τ + b_l); b is a registered nn.Parameter of
# shape (K,) initialised to 0 (and left alone by init_parameters); memberships are independent
# per prototype (no competition: changing p_k leaves m_l untouched); the b→+∞ limit gives
# m→1 for everyone = score → 1ᵀt, the per-item scalar (documented degenerate, guarded by the
# entropy read-out); construction order keeps the bias LAST in the parameter list.
# Run: python Master/temp/dc_checks/dc07/t07_sigmoid.py
from _common import *  # noqa: F401,F403

check, FAILS = make_check('t07')
torch.manual_seed(7)
N, D, K, TAU = 40, 12, 7, 0.5
m = proto(N, D, K, 'sigmoid', TAU)
names = [n for n, _ in m.named_parameters()]
check("bias_registered", 'membership_bias' in names and tuple(m.membership_bias.shape) == (K,), f"{names}")
check("bias_init_zero", bool((m.membership_bias == 0).all()))
check("bias_after_prototypes_in_param_order", names.index('membership_bias') > names.index('prototypes'), f"{names}")
with torch.no_grad():
    idx = torch.arange(N); out = m(idx); q = m.embedding_ext(idx)
    ref = torch.sigmoid(q @ m.prototypes.T / TAU + m.membership_bias)
check("closed_form", torch.allclose(out, ref, atol=1e-12))
check("in_0_1", bool((out > 0).all() and (out < 1).all()))
check("rows_do_not_sum_to_1", not torch.allclose(out.sum(1), torch.ones(N), atol=1e-3), "no normalisation across prototypes")
# independence: perturb prototype k → only column k moves
with torch.no_grad():
    m.prototypes.data[3] += 5.0
    out2 = m(idx)
    m.prototypes.data[3] -= 5.0
mask = torch.ones(K, dtype=torch.bool); mask[3] = False
check("independent_per_prototype", torch.allclose(out2[:, mask], out[:, mask], atol=1e-12) and not torch.allclose(out2[:, 3], out[:, 3]))
# b → +inf limit
T = torch.randn(15, K)
with torch.no_grad():
    m.membership_bias.data.fill_(60.0)
    mem = m(idx)
    m.membership_bias.data.zero_()
check("bias_to_inf_all_one", torch.allclose(mem, torch.ones(N, K), atol=1e-12))
check("bias_to_inf_score_is_1T_t", torch.allclose(mem @ T.T, T.sum(1).expand(N, -1), atol=1e-9),
      "score → Σ_l t_{i,l}: per-item scalar (popularity channel rebuilt) — guard via entropy read-out")
# bias receives gradient, init_parameters leaves it alone
m.membership_bias.data.fill_(0.3); m.init_parameters()
check("init_parameters_leaves_bias", bool((m.membership_bias == 0.3).all()))
m.membership_bias.data.zero_()
loss = (m(idx) * torch.randn(N, K)).sum() + m.get_and_reset_loss()
loss.backward()
check("bias_has_grad", m.membership_bias.grad is not None and bool(m.membership_bias.grad.abs().sum() > 0))
m.get_and_reset_loss()
finish('t07', FAILS)
