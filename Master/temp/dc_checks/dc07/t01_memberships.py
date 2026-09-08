# dc07 t01 — memberships live on the simplex (softmax) / in (0,1) (sigmoid), are permutation-
# equivariant in the prototypes, keep the [..., K] output contract for 1-D and 2-D indices, and
# match the closed form softmax(⟨q,P⟩/τ) computed by hand.
# Run: python Master/temp/dc_checks/dc07/t01_memberships.py
from _common import *  # noqa: F401,F403

check, FAILS = make_check('t01')
torch.manual_seed(1)
N, D, K = 40, 12, 7
for tau in (0.1, 1.0, 3.0):
    m = proto(N, D, K, 'softmax', tau)
    idx1 = torch.randint(0, N, (9,))
    idx2 = torch.randint(0, N, (5, 4))
    with torch.no_grad():
        o1, o2 = m(idx1), m(idx2)
    check(f"shape_1d_tau{tau}", tuple(o1.shape) == (9, K), f"{tuple(o1.shape)}")
    check(f"shape_2d_tau{tau}", tuple(o2.shape) == (5, 4, K), f"{tuple(o2.shape)}")
    check(f"rows_sum_to_1_tau{tau}", torch.allclose(o2.sum(-1), torch.ones(5, 4), atol=1e-12),
          f"max |Σm−1| = {(o2.sum(-1) - 1).abs().max():.2e}")
    # closed interval: at small τ a float64 softmax underflows to exactly 0 (legitimately)
    check(f"in_0_1_tau{tau}", bool((o2 >= 0).all() and (o2 <= 1).all()) and (tau < 1 or bool((o2 > 0).all())))
    # closed form
    q = m.embedding_ext(idx1)
    ref = torch.softmax(q @ m.prototypes.T / tau, dim=-1)
    check(f"closed_form_tau{tau}", torch.allclose(o1, ref, atol=1e-12), f"{(o1 - ref).abs().max():.2e}")
    m.get_and_reset_loss()

# permutation equivariance in the prototypes
m = proto(N, D, K, 'softmax', 0.7)
perm = torch.randperm(K)
with torch.no_grad():
    o = m(torch.arange(N))
    m.prototypes.data = m.prototypes.data[perm]
    o_p = m(torch.arange(N))
check("prototype_permutation_equivariant", torch.allclose(o_p, o[:, perm], atol=1e-12),
      f"{(o_p - o[:, perm]).abs().max():.2e}")

# scale sensitivity (P4): scaling q sharpens the softmax — unlike cosine, which is invariant
m = proto(N, D, K, 'softmax', 1.0)
with torch.no_grad():
    q = m.embedding_ext(torch.arange(N))
    m1 = m.memberships(q); m3 = m.memberships(3.0 * q)
    H = lambda p: -(p * p.log()).sum(-1).mean()
check("norm_sensitive_scaling_sharpens", H(m3) < H(m1), f"H(3q)={H(m3):.4f} < H(q)={H(m1):.4f}")
c = proto(N, D, K, 'shifted', 1.0)
with torch.no_grad():
    qc = c.embedding_ext(torch.arange(N))
    a1 = c.cosine_sim_func(qc.unsqueeze(-2), c.prototypes); a3 = c.cosine_sim_func((3 * qc).unsqueeze(-2), c.prototypes)
check("cosine_control_scale_invariant", torch.allclose(a1, a3, atol=1e-12))

# the forward and the regulariser see the same matrix (memberships, not logits)
m = proto(N, D, K, 'softmax', 0.5)
with torch.no_grad():
    o = m(torch.arange(N))
    loss = m.get_and_reset_loss()
    ref = -o.max(1).values.mean() - o.max(0).values.mean()
check("reg_consumes_memberships", torch.isclose(loss, ref, atol=1e-12), f"{float(loss):.6f} vs {float(ref):.6f}")

# bad temperature refuses
try:
    PrototypeEmbedding(N, D, K, cosine_type='softmax', temperature=0.0); bad = False
except ValueError:
    bad = True
check("temperature_must_be_positive", bad)
finish('t01', FAILS)
