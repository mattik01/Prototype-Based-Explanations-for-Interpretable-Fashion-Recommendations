# dc07 t04 — regulariser contract under the membership family (D1): 'max' = −mean row / column
# max of the MEMBERSHIP matrix (saturates at −1 per side; reads as "sharpen"); 'soft' / 'incl'
# consume the membership directly with a single sum-normalisation — no second softmax — and
# match hand formulas; the cosine family's 'soft'/'incl' still double-softmax exactly as before.
# Run: python Master/temp/dc_checks/dc07/t04_regulariser_contract.py
from _common import *  # noqa: F401,F403

check, FAILS = make_check('t04')
torch.manual_seed(4)
N, D, K = 50, 12, 8
idx = torch.arange(N)


def H(p, axis):
    return -(p * p.clamp_min(1e-300).log()).sum(axis).mean()


for ct in ('softmax', 'sigmoid'):
    # max / max
    m = proto(N, D, K, ct, 0.8, reg=('max', 'max'))
    with torch.no_grad():
        mem = m(idx); loss = m.get_and_reset_loss()
    ref = -mem.max(0).values.mean() - mem.max(1).values.mean()
    check(f"{ct}.max_max_on_memberships", torch.isclose(loss, ref, atol=1e-12), f"{float(loss):.6f}")
    check(f"{ct}.max_saturates_at_minus_1_per_side", bool(loss >= -2.0 - 1e-12), "each side ≥ −1")
    # soft / soft — single normalisation
    m = proto(N, D, K, ct, 0.8, reg=('soft', 'soft'))
    with torch.no_grad():
        mem = m(idx); loss = m.get_and_reset_loss()
    row_p = mem / mem.sum(1, keepdim=True)
    col_p = mem / mem.sum(0, keepdim=True)
    ref = H(col_p, 0) + H(row_p, 1)
    check(f"{ct}.soft_soft_single_normalisation", torch.isclose(loss, ref, atol=1e-10), f"{float(loss):.6f} vs {float(ref):.6f}")
    double = PrototypeEmbedding._entropy_reg_loss(mem, 0) + PrototypeEmbedding._entropy_reg_loss(mem, 1)
    check(f"{ct}.soft_is_not_the_double_softmax", not torch.isclose(loss, double, atol=1e-6),
          f"membership-native {float(loss):.4f} vs double-softmax {float(double):.4f}")
    if ct == 'softmax':
        check("softmax.soft_batch_term_is_plain_membership_entropy",
              torch.isclose(H(row_p, 1), H(mem, 1), atol=1e-12), "rows already sum to 1")
    # incl
    m = proto(N, D, K, ct, 0.8, reg=('incl', 'max'))
    with torch.no_grad():
        mem = m(idx); loss = m.get_and_reset_loss()
    q_k = mem.sum(0) / mem.sum()
    ref = (q_k * q_k.log()).sum() - mem.max(1).values.mean()
    check(f"{ct}.incl_on_membership_loads", torch.isclose(loss, ref, atol=1e-10), f"{float(loss):.6f} vs {float(ref):.6f}")

# cosine family unchanged: 'soft' still = entropy of Softmax(sim), 'incl' = column loads of Softmax(sim)
m = proto(N, D, K, 'shifted', reg=('incl', 'soft'))
with torch.no_grad():
    sim = m(idx); loss = m.get_and_reset_loss()
ref = PrototypeEmbedding._inclusiveness_constraint(sim) + PrototypeEmbedding._entropy_reg_loss(sim, 1)
check("cosine.soft_incl_untouched", torch.isclose(loss, ref, atol=1e-12))
# unknown types still refuse
try:
    PrototypeEmbedding(N, D, K, cosine_type='softmax', reg_batch_type='incl'); bad = False
except ValueError:
    bad = True
check("batch_incl_still_refused", bad)
finish('t04', FAILS)
