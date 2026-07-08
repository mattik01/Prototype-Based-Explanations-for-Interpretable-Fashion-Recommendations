# t07 — the four dc02 §3.5 constraint knobs on the REPO class (claims 8–11 of the claims spec),
# plus per-step resampling, global-RNG isolation, and the knobs-off short-circuit.
import math
import sys

import torch

from _harness import make_check, toy_attr_data

from feature_extraction.feature_extractors import AttributeLookup, AttributePrototypeEmbedding

torch.manual_seed(17)
check, fails = make_check()

M, V_fs, K = 12, (3, 4, 5), 5
X, offs = toy_attr_data(M=M, V_fs=V_fs, seed=0)
V = X.shape[1]
F = len(V_fs)


def build(**knobs):
    return AttributePrototypeEmbedding(M, AttributeLookup(X), K, offs,
                                       reg_proto_type='max', reg_batch_type='max',
                                       cosine_type='shifted', **knobs)


# ============ K1 — nonnegativity reparameterization (claim 8) ============
m1 = build(nonneg_prototypes=True)
m1.prototypes.data.normal_(0, 1)                     # generic Ã with negative entries
check("K1: stored parameter has negative entries (it is Ã)",
      bool((m1.prototypes.detach() < 0).any()))
A_eff = m1.effective_prototypes()
check("K1: effective prototypes strictly positive", bool((A_eff > 0).all()))
check("K1: softplus(stored) == effective",
      torch.equal(torch.nn.functional.softplus(m1.state_dict()['prototypes']), A_eff.detach()))
check("K1: state_dict key stays 'prototypes'", list(m1.state_dict().keys()) == ['prototypes'])

t_star = m1(torch.arange(M))
check("K1: t* within [1, 2] (repulsion half unreachable)",
      bool((t_star >= 1 - 1e-6).all()) and bool((t_star <= 2 + 1e-6).all()),
      f"range [{float(t_star.min()):.4f}, {float(t_star.max()):.4f}]")
m1.zero_grad()
(t_star ** 2).sum().backward()
check("K1: grad reaches raw parameter through softplus",
      m1.prototypes.grad is not None and float(m1.prototypes.grad.norm()) > 0)
m1.get_and_reset_loss()

# ============ K2 — field crispness (claim 9) ============
m2 = build(crisp_weight=1.0)
# one-hot-like: one dominating coordinate per field block -> entropy ~ 0
A_crisp = torch.zeros(K, V)
for f in range(F):
    A_crisp[:, offs[f]] = 50.0
m2.prototypes.data.copy_(A_crisp)
l_crisp = m2.get_and_reset_loss()                     # no forward -> pure knob term
check("K2: one-hot-like blocks -> ~0", float(l_crisp) < 1e-6, f"{float(l_crisp):.2e}")
# uniform blocks -> mean ln(V_f), the entropy maximum
m2.prototypes.data.zero_()
l_uniform = m2.get_and_reset_loss()
expected = sum(math.log(v) for v in V_fs) / F
check("K2: uniform blocks == mean ln(V_f)",
      abs(float(l_uniform) - expected) < 1e-5, f"{float(l_uniform):.5f} vs {expected:.5f}")
# random < uniform
m2.prototypes.data.normal_(0, 1)
l_rand = m2.get_and_reset_loss()
check("K2: random < uniform maximum", float(l_rand) < float(l_uniform),
      f"{float(l_rand):.4f} < {float(l_uniform):.4f}")
# gradient reaches A via the knob loss alone
m2.zero_grad()
m2.get_and_reset_loss().backward()
check("K2: grad reaches prototypes via loss", float(m2.prototypes.grad.norm()) > 0)
# linear scaling in crisp_weight
m2b = build(crisp_weight=2.0)
m2b.prototypes.data.copy_(m2.prototypes.data)
check("K2: term scales linearly with crisp_weight",
      abs(float(m2b.get_and_reset_loss()) - 2 * float(m2.get_and_reset_loss())) < 1e-6)

# ============ K3 — separation hinge (claim 10) ============
m3 = build(sep_weight=1.0, sep_margin=0.3)
# orthogonal prototypes: disjoint one-hot support -> cos == 0 <= m -> L_sep == 0 exactly
A_orth = torch.zeros(K, V)
for k in range(K):
    A_orth[k, k] = 1.0
m3.prototypes.data.copy_(A_orth)
check("K3: orthogonal prototypes -> exactly 0", float(m3.get_and_reset_loss()) == 0.0)
# near-duplicate pair -> positive
A_dup = A_orth.clone()
A_dup[1] = A_dup[0] + 0.01 * torch.randn(V)
m3.prototypes.data.copy_(A_dup)
check("K3: near-duplicate pair -> positive", float(m3.get_and_reset_loss()) > 0)
# anti-parallel pair: PLAIN cosine == -1 <= m -> contributes 0 (proves unshifted cosine)
A_anti = A_orth.clone()
A_anti[1] = -A_anti[0]
m3.prototypes.data.copy_(A_anti)
check("K3: anti-parallel pair -> 0 (plain, unshifted cosine)",
      float(m3.get_and_reset_loss()) == 0.0)
# gradient
m3.prototypes.data.copy_(A_dup)
m3.zero_grad()
m3.get_and_reset_loss().backward()
check("K3: grad reaches prototypes", float(m3.prototypes.grad.norm()) > 0)

# ============ K4 — data pull (claim 11) ============
# full-catalog deterministic path (push_n_items >= M)
m4 = build(push_weight=1.0, push_n_items=M)
A_pull = torch.stack([2.0 * X[k] for k in range(K)])          # positive multiples of real rows
m4.prototypes.data.copy_(A_pull)
l0 = m4.get_and_reset_loss()
check("K4: positive multiples of catalog rows -> ~0", float(l0) < 1e-6, f"{float(l0):.2e}")
A_pert = A_pull.clone()
A_pert[2] = A_pull[2] + torch.ones(V)                          # off-support perturbation
m4.prototypes.data.copy_(A_pert)
check("K4: perturbed prototype -> positive", float(m4.get_and_reset_loss()) > 0)
A_neg = A_pull.clone()
A_neg[0] = -A_pull[0]                                          # negated multiple: cos = -1
m4.prototypes.data.copy_(A_neg)
check("K4: negated multiple -> positive", float(m4.get_and_reset_loss()) > 0)
m4.zero_grad()
m4.get_and_reset_loss().backward()
check("K4: grad reaches prototypes", float(m4.prototypes.grad.norm()) > 0)

# per-step resampling with push_n_items < M: consecutive subsets generically differ
m4s = build(push_weight=1.0, push_n_items=3, push_seed=42)
m4s.prototypes.data.normal_(0, 1)
vals = [float(m4s.get_and_reset_loss()) for _ in range(4)]
check("K4: per-step resampling -> consecutive values differ", len(set(vals)) > 1, f"{vals}")

# RNG isolation: global torch RNG state bitwise unchanged by a forward+loss with push on
state_before = torch.random.get_rng_state().clone()
m4s(torch.arange(M))
m4s.get_and_reset_loss()
state_after = torch.random.get_rng_state()
check("K4: global RNG state bitwise unchanged", torch.equal(state_before, state_after))

# same push_seed -> same subset sequence (determinism of the dedicated generator)
m4a = build(push_weight=1.0, push_n_items=3, push_seed=7)
m4b = build(push_weight=1.0, push_n_items=3, push_seed=7)
m4b.prototypes.data.copy_(m4a.prototypes.data)
seq_a = [float(m4a.get_and_reset_loss()) for _ in range(3)]
seq_b = [float(m4b.get_and_reset_loss()) for _ in range(3)]
check("K4: same push_seed -> same value sequence", seq_a == seq_b)

# ============ knobs-off short-circuit ============
m_off = build()
m_base = build()
m_base.prototypes.data.copy_(m_off.prototypes.data)
idxs = torch.tensor([[0, 3, 7], [2, 5, 11]])
out_off, out_base = m_off(idxs), m_base(idxs)
check("knobs-off: forward identical to base build", torch.equal(out_off, out_base))
check("knobs-off: loss has no phantom knob terms",
      torch.equal(m_off.get_and_reset_loss(), m_base.get_and_reset_loss()))

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
