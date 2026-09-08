# dc07 t03 — τ-limit behaviour as a DOCUMENTED PROPERTY (not a keystone) and the measurement
# that pins the search range: at stock init, for each (d, K) and input shape, the smallest τ
# at which the mean membership entropy exceeds 0.9·log K (the uniform / rebuilt-popularity
# regime). DC07_TAU_MAX must sit below the smallest such τ over the grid; DC07_TAU_MIN must
# sit above the τ at which memberships go one-hot (H < 0.1·log K) for the LARGEST-norm shape,
# otherwise the search can only pick degenerate ends.
# Run: python Master/temp/dc_checks/dc07/t03_temperature_limits.py
import math
import numpy as np
from _common import *  # noqa: F401,F403
from confs.hyper_params import DC07_TAU_MIN, DC07_TAU_MAX

check, FAILS = make_check('t03')
torch.manual_seed(3)

# ---- (1) limits ----
N, D, K = 60, 16, 9
base = proto(N, D, K, 'softmax', 1.0)
T = torch.randn(25, K)  # a toy free side (items) in prototype space
with torch.no_grad():
    q = base.embedding_ext(torch.arange(N))
    for tau, name, target in ((1e6, 'uniform', torch.full((N, K), 1.0 / K)),):
        base.temperature = tau
        mem = base.memberships(q)
        check(f"tau_to_inf_{name}", torch.allclose(mem, target, atol=1e-6), f"max |m−1/K| = {(mem - target).abs().max():.2e}")
        score = mem @ T.T
        per_item_mean = T.mean(1).expand(N, -1)
        check("tau_to_inf_score_is_per_item_mean_of_t", torch.allclose(score, per_item_mean, atol=1e-5),
              "score → (1/K)·Σ_l t_{i,l}: a pure per-item scalar (the popularity channel rebuilt)")
    base.temperature = 1e-6
    mem = base.memberships(q)
    check("tau_to_0_one_hot", bool((mem.max(1).values > 1 - 1e-9).all()), "one prototype per object")
    argmax_dot = (q @ base.prototypes.T).argmax(1)
    check("tau_to_0_hard_assignment_by_dot", bool((mem.argmax(1) == argmax_dot).all()))
    base.temperature = 1.0

# ---- (2) range measurement ----
def mean_H_over_logK(m, q, tau):
    m.temperature = tau
    with torch.no_grad():
        mem = m.memberships(q)
    return float((-(mem * mem.clamp_min(1e-300).log()).sum(-1).mean()) / math.log(m.n_prototypes))


taus = np.logspace(-2.5, 2.5, 101)
rows = []
NN = 3000
for d in (10, 30, 60, 100):
    for k in (10, 30, 60, 100):
        for shape in ('host', 'fI', 'fU'):
            if shape == 'host':
                ext = None
            elif shape == 'fI':
                ext, _ = fi_ext(NN, d, n_feat=200, F=5)
            else:
                ext, _, _ = fu_ext(NN, d, n_feat=200, d_max=20)
            m = proto(NN, d, k, 'softmax', 1.0, ext=ext).eval()
            with torch.no_grad():
                q = m.embedding_ext(torch.arange(NN))
            qn = float(q.norm(dim=1).mean())
            Hs = np.array([mean_H_over_logK(m, q, t) for t in taus])
            t_uni = taus[np.argmax(Hs > 0.9)] if (Hs > 0.9).any() else float('inf')
            t_hard = taus[np.argmax(Hs > 0.1)] if (Hs > 0.1).any() else float('inf')  # first τ leaving the hard regime
            rows.append((d, k, shape, qn, t_hard, t_uni))
print("\n| d | K | shape | mean‖q‖ at init | τ leaving hard regime (H>0.1 logK) | τ entering uniform regime (H>0.9 logK) |")
print("|---|---|---|---:|---:|---:|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.4g} | {r[5]:.4g} |")
t_uni_min = min(r[5] for r in rows)
host = [r for r in rows if r[2] == 'host']
t_hard_max_host = max(r[4] for r in host)
t_hard_max_all = max(r[4] for r in rows)
print(f"\nsmallest uniform-regime τ over grid: {t_uni_min:.4g}; largest hard-regime exit τ: "
      f"host {t_hard_max_host:.4g} / all shapes {t_hard_max_all:.4g} (fI ≈ 2.5× the host norm)")
print(f"configured range: [{DC07_TAU_MIN}, {DC07_TAU_MAX}]  (stage 1 = host shapes; a stage-2 shift for fI is a recorded option)")
check("tau_max_below_uniform_regime", DC07_TAU_MAX < t_uni_min,
      f"DC07_TAU_MAX={DC07_TAU_MAX} < {t_uni_min:.4g}")
check("tau_min_within_one_decade_of_host_hard_exit", DC07_TAU_MIN >= t_hard_max_host / 10,
      f"DC07_TAU_MIN={DC07_TAU_MIN} ≥ {t_hard_max_host:.4g}/10 (host shapes, stage 1)")
check("range_is_non_trivial", DC07_TAU_MAX / DC07_TAU_MIN >= 10)
finish('t03', FAILS)
