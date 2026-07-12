"""
fI re-check (re-host amendment, 2026-07-12) — claims C3', C4'(a/b/c), C5' ONLY.

Mechanism (fI model, per the spec's "Mechanism delta"):
  - Item i composed embedding: q_i = sum_{r in rows(i)} E[r]  over F+1 rows
    (F feature-value rows + 1 ID row). E in R^{(V+M) x d}.
  - Item prototypes P_t in R^{K_t x d}; tstar_k = 1 + cos(q_i, P_t[k]).
  - User side: free vector u = U[u] in R^{K_t}. No P_u, W_u, W_t, ustar,
    uhat, that.
  - Score: S(u,i) = u . tstar.

Adversarial sweep: >=12 seeds, F in {0,1,2,3,5,11}, parameter scales
{1e-3, 1, 1e3} (uniform across all tensors for all seeds, plus the full
27 per-tensor scale combinations for 3 seeds), shapes (K_t,d) including
K_t=1 and d=1, float64 throughout. No training.

C4'(c) uses candidate sets of M=25 items (>=20) and checks the exact
argsort identity plus the pairwise-difference identity.
C5' requires bit-identity (max abs diff exactly 0.0) of tstar and S
under matched weights Q[i] = E[V+i] (V=0 when F=0).
"""

import itertools
import math

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

SEEDS = list(range(12))               # 12 seeds >= 10
F_VALUES = [0, 1, 2, 3, 5, 11]
SCALES = [1e-3, 1.0, 1e3]
SHAPES = [(4, 8), (1, 8), (3, 2), (7, 16), (1, 1)]  # (K_t, d), incl. K_t=1
N_USERS = 4
M_ITEMS = 25                          # candidate set size for C4'(c) (>=20)
V_PER_FEAT = 6                        # distinct values per categorical field

REL_TOL = 1e-9                        # float64 exactness threshold (relative)


def build(seed, F, K_t, d, sE, sU, sP):
    g = torch.Generator().manual_seed(seed)
    V = F * V_PER_FEAT
    E = torch.randn(V + M_ITEMS, d, generator=g) * sE
    U = torch.randn(N_USERS, K_t, generator=g) * sU
    P_t = torch.randn(K_t, d, generator=g) * sP
    # rows(i): one value per categorical field + the ID row V+i
    rows = []
    for i in range(M_ITEMS):
        rr = [f * V_PER_FEAT
              + int(torch.randint(V_PER_FEAT, (1,), generator=g))
              for f in range(F)]
        rr.append(V + i)
        rows.append(rr)
    return E, U, P_t, rows, V


def tstar_of(q, P_t):
    # q: (d,) ; P_t: (K_t, d)  ->  (K_t,)
    return 1.0 + (P_t @ q) / (q.norm() * P_t.norm(dim=1))


def relerr(a, b):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    denom = torch.maximum(torch.maximum(a.abs(), b.abs()),
                          torch.tensor(1e-300))
    return ((a - b).abs() / denom).max().item(), (a - b).abs().max().item()


# ------------------------------------------------------------------ sweep --
stats = {
    # C3' split by d==1 (mathematically degenerate: 1-D cosine depends only
    # on sign, so the true gradient is exactly zero there) vs d>=2
    "C3_trials_d2": 0, "C3_nonzero_d2": 0,
    "C3_min_gradnorm_rel_d2": math.inf, "C3_max_relerr_analytic_d2": 0.0,
    "C3_trials_d1": 0, "C3_zero_d1": 0, "C3_max_gradnorm_d1": 0.0,
    "C4a_max_rel": 0.0, "C4a_max_abs": 0.0, "C4a_trials": 0,
    # scale-aware error: |lhs-rhs| / (sum of magnitudes of all terms)
    "C4b_max_scaled": 0.0, "C4b_max_abs": 0.0, "C4b_trials": 0,
    "C4c_argsort_ok": 0, "C4c_argsort_trials": 0,
    "C4c_pair_max_scaled": 0.0, "C4c_pair_max_abs": 0.0,
    "C5_max_abs_tstar": 0.0, "C5_max_abs_S": 0.0, "C5_trials": 0,
    "configs": 0,
}
failures = []

scale_combos_all = list(itertools.product(SCALES, SCALES, SCALES))
configs = []
for seed, F, (K_t, d) in itertools.product(SEEDS, F_VALUES, SHAPES):
    for s in SCALES:                                  # uniform scales
        configs.append((seed, F, K_t, d, s, s, s))
    if seed < 3:                                      # full 27 combos
        for sE, sU, sP in scale_combos_all:
            if not (sE == sU == sP):
                configs.append((seed, F, K_t, d, sE, sU, sP))

for (seed, F, K_t, d, sE, sU, sP) in configs:
    stats["configs"] += 1
    E, U, P_t, rows, V = build(seed, F, K_t, d, sE, sU, sP)
    uvec = U[seed % N_USERS]
    tag = f"seed={seed} F={F} K_t={K_t} d={d} scales=({sE},{sU},{sP})"

    # ---------------- C3': gradient through the single path ---------------
    # (needs a feature row -> F >= 1)
    if F >= 1:
        i = seed % M_ITEMS
        Eg = E.clone().requires_grad_(True)
        q = Eg[rows[i]].sum(dim=0)
        S = uvec @ tstar_of(q, P_t)
        S.backward()
        # analytic dS/dq (identical for every composed row r of item i)
        with torch.no_grad():
            qn = q.detach()
            nq = qn.norm()
            npk = P_t.norm(dim=1)
            cos = (P_t @ qn) / (nq * npk)
            dSdq = (P_t / (nq * npk)[:, None]
                    - (cos / nq**2)[:, None] * qn[None, :])
            dSdq = (uvec[:, None] * dSdq).sum(dim=0)
        for r in rows[i][:-1]:                        # feature rows only
            gnorm = Eg.grad[r].norm().item()
            # scale-free non-zero test: compare to a natural magnitude
            ref = max((uvec.abs().sum() / max(nq.item(), 1e-300)).item(),
                      1e-300)
            if d == 1:
                # true gradient is exactly 0 (1-D cosine is sign-only);
                # confirm it is zero up to float noise
                stats["C3_trials_d1"] += 1
                stats["C3_max_gradnorm_d1"] = max(
                    stats["C3_max_gradnorm_d1"], gnorm / ref)
                if gnorm <= 1e-12 * ref:
                    stats["C3_zero_d1"] += 1
                else:
                    failures.append(("C3'-d1", tag,
                                     f"row {r} gradnorm/ref={gnorm/ref}"))
            else:
                stats["C3_trials_d2"] += 1
                if gnorm > 1e-12 * ref:
                    stats["C3_nonzero_d2"] += 1
                else:
                    failures.append(("C3'", tag,
                                     f"row {r} gradnorm={gnorm}"))
                stats["C3_min_gradnorm_rel_d2"] = min(
                    stats["C3_min_gradnorm_rel_d2"], gnorm / ref)
                re_, _ = relerr(Eg.grad[r], dSdq)
                stats["C3_max_relerr_analytic_d2"] = max(
                    stats["C3_max_relerr_analytic_d2"], re_)

    # ---------------- C4'(a): S = sum_k u_k tstar_k ------------------------
    for i in range(M_ITEMS):
        q = E[rows[i]].sum(dim=0)
        ts = tstar_of(q, P_t)
        S = (uvec @ ts).item()
        # each summand computed independently, then summed
        summands = [(uvec[k] * (1.0 + (q @ P_t[k])
                                / (q.norm() * P_t[k].norm()))).item()
                    for k in range(K_t)]
        re_, ae_ = relerr(S, math.fsum(summands))
        stats["C4a_trials"] += 1
        stats["C4a_max_rel"] = max(stats["C4a_max_rel"], re_)
        stats["C4a_max_abs"] = max(stats["C4a_max_abs"], ae_)
        if re_ > REL_TOL:
            failures.append(("C4'a", tag, f"item {i} relerr={re_}"))

        # ------------ C4'(b): S = sum u_k + sum_r sum_k u_k c_{r,k} --------
        c = (E[rows[i]] @ P_t.T) / (q.norm() * P_t.norm(dim=1))  # (F+1, K_t)
        terms = ([uvec[k].item() for k in range(K_t)]
                 + [(uvec[k] * c[j, k]).item()
                    for j in range(len(rows[i])) for k in range(K_t)])
        rhs = math.fsum(terms)
        ae_ = abs(S - rhs)
        # condition-aware denominator: total magnitude of the terms summed
        denom = max(math.fsum(abs(t) for t in terms), abs(S), 1e-300)
        se_ = ae_ / denom
        stats["C4b_trials"] += 1
        stats["C4b_max_scaled"] = max(stats["C4b_max_scaled"], se_)
        stats["C4b_max_abs"] = max(stats["C4b_max_abs"], ae_)
        if se_ > REL_TOL:
            failures.append(("C4'b", tag, f"item {i} scaled_err={se_}"))

    # ---------------- C4'(c): rank-inert baseline --------------------------
    Svec = torch.stack([uvec @ tstar_of(E[rows[i]].sum(dim=0), P_t)
                        for i in range(M_ITEMS)])
    base = uvec.sum()
    feat = Svec - base                       # feature part as spec defines it
    stats["C4c_argsort_trials"] += 1
    # stable argsort handles exact ties (which occur at d=1, cos = +/-1)
    ok = np.array_equal(np.argsort(Svec.numpy(), kind="stable"),
                        np.argsort(feat.numpy(), kind="stable"))
    if ok:
        stats["C4c_argsort_ok"] += 1
    else:
        failures.append(("C4'c-argsort", tag, "argsort mismatch"))
    D_S = Svec[:, None] - Svec[None, :]
    D_f = feat[:, None] - feat[None, :]
    ae_ = (D_S - D_f).abs().max().item()
    # scale-aware: compare against the magnitude of the scores involved
    denom = max(Svec.abs().max().item(), feat.abs().max().item(),
                base.abs().item(), 1e-300)
    se_ = ae_ / denom
    stats["C4c_pair_max_scaled"] = max(stats["C4c_pair_max_scaled"], se_)
    stats["C4c_pair_max_abs"] = max(stats["C4c_pair_max_abs"], ae_)
    if se_ > REL_TOL:
        failures.append(("C4'c-pairs", tag, f"pairwise scaled_err={se_}"))

    # ---------------- C5': host reduction (F = 0 only) ---------------------
    if F == 0:
        assert V == 0
        Q = E.clone()                        # matched weights Q[i] = E[V+i]
        for i in range(M_ITEMS):
            q_fi = E[rows[i]].sum(dim=0)     # composition over the 1 ID row
            ts_fi = tstar_of(q_fi, P_t)
            ts_id = tstar_of(Q[i], P_t)      # plain per-item embedding table
            S_fi = uvec @ ts_fi
            S_id = uvec @ ts_id
            stats["C5_trials"] += 1
            d_ts = (ts_fi - ts_id).abs().max().item()
            d_S = abs((S_fi - S_id).item())
            stats["C5_max_abs_tstar"] = max(stats["C5_max_abs_tstar"], d_ts)
            stats["C5_max_abs_S"] = max(stats["C5_max_abs_S"], d_S)
            if d_ts != 0.0 or d_S != 0.0:
                failures.append(("C5'", tag, f"item {i} dts={d_ts} dS={d_S}"))

# also report the mathematically-equivalent alt feature part u.(tstar-1)
# for C4'(c), on a few configs, as a cross-check
alt_max_rel = 0.0
alt_argsort_ok = 0
alt_trials = 0
for seed in SEEDS[:5]:
    E, U, P_t, rows, V = build(seed, 3, 4, 8, 1.0, 1.0, 1.0)
    uvec = U[0]
    Svec = torch.stack([uvec @ tstar_of(E[rows[i]].sum(dim=0), P_t)
                        for i in range(M_ITEMS)])
    feat2 = torch.stack([uvec @ (tstar_of(E[rows[i]].sum(dim=0), P_t) - 1.0)
                         for i in range(M_ITEMS)])
    alt_trials += 1
    if np.array_equal(np.argsort(Svec.numpy(), kind="stable"),
                      np.argsort(feat2.numpy(), kind="stable")):
        alt_argsort_ok += 1
    re_, _ = relerr(Svec[:, None] - Svec[None, :],
                    feat2[:, None] - feat2[None, :])
    alt_max_rel = max(alt_max_rel, re_)

# -------------------------------------------------------------- report ----
print(f"configs run: {stats['configs']}")
print()
print("C3' (gradient through the single path):")
print(f"  d>=2: trials={stats['C3_trials_d2']}"
      f"  non-zero={stats['C3_nonzero_d2']}"
      f"  min gradnorm/ref={stats['C3_min_gradnorm_rel_d2']:.3e}"
      f"  max relerr autograd~analytic="
      f"{stats['C3_max_relerr_analytic_d2']:.3e}")
print(f"  d=1 (degenerate, true grad = 0): trials={stats['C3_trials_d1']}"
      f"  zero-as-expected={stats['C3_zero_d1']}"
      f"  max gradnorm/ref={stats['C3_max_gradnorm_d1']:.3e}")
print()
print("C4'(a) (S = sum_k u_k tstar_k, independent summands):")
print(f"  trials={stats['C4a_trials']}  max_rel={stats['C4a_max_rel']:.3e}"
      f"  max_abs={stats['C4a_max_abs']:.3e}")
print("C4'(b) (S = sum u_k + sum_r sum_k u_k c_rk):")
print(f"  trials={stats['C4b_trials']}"
      f"  max_scaled_err={stats['C4b_max_scaled']:.3e}"
      f"  max_abs={stats['C4b_max_abs']:.3e}")
print("C4'(c) (rank-inert baseline, candidate sets of "
      f"{M_ITEMS} items):")
print(f"  argsort identical : {stats['C4c_argsort_ok']}"
      f"/{stats['C4c_argsort_trials']}")
print(f"  pairwise-diff max_scaled_err={stats['C4c_pair_max_scaled']:.3e}"
      f"  max_abs={stats['C4c_pair_max_abs']:.3e}")
print(f"  alt feature part u.(tstar-1): argsort {alt_argsort_ok}/{alt_trials},"
      f" pairwise max_rel={alt_max_rel:.3e}")
print()
print("C5' (host reduction, F=0, bit-identity):")
print(f"  trials={stats['C5_trials']}"
      f"  max_abs_diff_tstar={stats['C5_max_abs_tstar']}"
      f"  max_abs_diff_S={stats['C5_max_abs_S']}")
print()
if failures:
    print(f"FAILURES ({len(failures)}):")
    for f in failures[:50]:
        print("  ", f)
else:
    print("NO FAILURES — all checks passed within tolerance / bit-identity.")
