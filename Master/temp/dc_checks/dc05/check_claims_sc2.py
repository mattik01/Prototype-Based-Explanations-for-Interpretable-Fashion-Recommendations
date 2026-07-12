"""
dc05 SC.2 black-box toy re-check (addendum round).

Re-implements the mechanism from `claims_spec_sc2.md` ALONE (no project code):

- F categorical fields, field g has V_g values, total V word rows.
- Frozen word table E in R^{V x d}, iid N(0,1), float64, no training.
- Each item carries exactly one value per field.
- User basket H_u -> weights w_{u,f} = n_{u,f} / |H_u|;
  composed metadata vector m_u = sum_f w_{u,f} E[f].
- Optional ID row e_ID(u) ~ N(0, I_d), weight 1: q_u = m_u + e_ID(u).
- Heterogeneous population: per field g, p_pop^g ~ Dirichlet(1,...,1)
  (once per seed); per user p_u^g ~ Dirichlet(alpha * V_g * p_pop^g);
  items iid with field-g value ~ p_u^g.
- User centroid c_u = sum_g sum_{f in g} p_{u,f} E[f].

Claims checked: C10' (a)-(d) [no ID row] and C11' (a)-(b) [ID row, alpha=2].

Toy dims: F=5, V_g in {15,19,22,26,30} (V=112), d=32, N=300 users,
5 seeds, float64, CPU, seconds of runtime.

Variance-reduction coupling (noted as an interpretation choice): for each
(seed, alpha, user) we draw 300 items once and use PREFIXES as the baskets
of size 3, 10, 30, 100, 300 — nested baskets, so the |H|-trends are measured
on coupled samples. User tastes p_u are fixed per (seed, alpha) across |H|.
"""

import numpy as np

# ----------------------------- configuration ------------------------------
F = 5
V_G = [15, 19, 22, 26, 30]          # V_g per field, within spec's 15-30
V = sum(V_G)                         # 112
D = 32
N_USERS = 300
H_SIZES = [3, 10, 30, 100, 300]
ALPHAS = [0.2, 2.0, 20.0]
SEEDS = [0, 1, 2, 3, 4]
H_MAX = max(H_SIZES)

FIELD_OFFSETS = np.cumsum([0] + V_G)  # word-row offsets per field


# ------------------------------- helpers ----------------------------------
def angles_deg(A, B):
    """Per-row angle (degrees) between rows of A and rows of B."""
    num = np.sum(A * B, axis=-1)
    den = np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1)
    cos = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def mean_pairwise_angle_deg(X):
    """Mean angle (degrees) over all distinct unordered pairs of rows of X."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = np.clip(Xn @ Xn.T, -1.0, 1.0)
    iu = np.triu_indices(X.shape[0], k=1)
    return float(np.degrees(np.arccos(G[iu])).mean())


def check_monotone(vals, direction):
    """Return list of (i, delta) adjacent violations of strict monotonicity."""
    out = []
    for i in range(len(vals) - 1):
        d = vals[i + 1] - vals[i]
        if direction == "dec" and d >= 0:
            out.append((i, d))
        if direction == "inc" and d <= 0:
            out.append((i, d))
    return out


# --------------------------- one seed simulation ---------------------------
def run_seed(seed):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((V, D))                     # frozen word table
    p_pop = [rng.dirichlet(np.ones(vg)) for vg in V_G]  # once per seed

    res = {}
    for alpha in ALPHAS:
        # per-user tastes, per field
        p_u = [rng.dirichlet(alpha * V_G[g] * p_pop[g], size=N_USERS)
               for g in range(F)]                       # each (N, V_g)

        # centroids c_u
        C = np.zeros((N_USERS, D))
        for g in range(F):
            Eg = E[FIELD_OFFSETS[g]:FIELD_OFFSETS[g + 1]]
            C += p_u[g] @ Eg

        # draw H_MAX items per user per field; prefix counts -> nested baskets
        # counts[g] has shape (N, len(H_SIZES), V_g): n_{u,f} for each |H|
        counts = []
        for g in range(F):
            vg = V_G[g]
            cum = np.cumsum(
                np.eye(vg)[
                    np.array([rng.choice(vg, size=H_MAX, p=p_u[g][u])
                              for u in range(N_USERS)])
                ], axis=1)                              # (N, H_MAX, V_g)
            counts.append(cum[:, np.array(H_SIZES) - 1, :])

        # composed vectors m_u for each basket size
        M = {}
        for hi, h in enumerate(H_SIZES):
            m = np.zeros((N_USERS, D))
            for g in range(F):
                Eg = E[FIELD_OFFSETS[g]:FIELD_OFFSETS[g + 1]]
                m += (counts[g][:, hi, :] / h) @ Eg
            M[h] = m

        res[alpha] = {
            "own_angle": {h: float(angles_deg(M[h], C).mean())
                          for h in H_SIZES},
            "pair_angle_m": {h: mean_pairwise_angle_deg(M[h])
                             for h in H_SIZES},
            "pair_angle_c": mean_pairwise_angle_deg(C),
            "mean_norm_m": {h: float(np.linalg.norm(M[h], axis=1).mean())
                            for h in H_SIZES},
            "mean_norm_c": float(np.linalg.norm(C, axis=1).mean()),
        }

        # C11' quantities (ID row), only needed at alpha = 2
        if alpha == 2.0:
            e_id = rng.standard_normal((N_USERS, D))
            n_id2 = np.sum(e_id ** 2, axis=1)
            id_share, ratio = {}, {}
            for h in H_SIZES:
                n_m2 = np.sum(M[h] ** 2, axis=1)
                id_share[h] = float((n_id2 / (n_m2 + n_id2)).mean())
                ratio[h] = float((np.sqrt(n_m2) / np.sqrt(n_id2)).mean())
            res[alpha]["id_share"] = id_share
            res[alpha]["ratio_m_id"] = ratio
    return res


# ------------------------------ aggregation --------------------------------
def main():
    per_seed = {s: run_seed(s) for s in SEEDS}

    def seed_vals(alpha, key, h=None):
        if h is None:
            return [per_seed[s][alpha][key] for s in SEEDS]
        return [per_seed[s][alpha][key][h] for s in SEEDS]

    def mean(alpha, key, h=None):
        return float(np.mean(seed_vals(alpha, key, h)))

    fmt = lambda x: f"{x:10.4f}"
    print(f"Config: F={F}, V_g={V_G} (V={V}), d={D}, N={N_USERS} users, "
          f"seeds={SEEDS}, |H| in {H_SIZES}, alpha in {ALPHAS}, float64\n")

    # ---------------- C10'(a): mean angle(m_u, c_u) vs |H| ----------------
    print("C10'(a)  mean angle(m_u, own centroid c_u) [deg]  (seed-mean)")
    print("  |H|   " + "".join(f"a={a:<9g}" for a in ALPHAS))
    for h in H_SIZES:
        print(f"  {h:<5d}" + "".join(fmt(mean(a, "own_angle", h))
                                     for a in ALPHAS))
    viol_a = []
    for a in ALPHAS:
        curve = [mean(a, "own_angle", h) for h in H_SIZES]
        for i, d in check_monotone(curve, "dec"):
            viol_a.append(("mean", a, H_SIZES[i], H_SIZES[i+1], d))
        for s in SEEDS:
            sc = [per_seed[s][a]["own_angle"][h] for h in H_SIZES]
            for i, d in check_monotone(sc, "dec"):
                viol_a.append((f"seed{s}", a, H_SIZES[i], H_SIZES[i+1], d))
    print(f"  monotone-decrease violations: {viol_a if viol_a else 'NONE'}\n")

    # ---------------- C10'(b): pairwise m-angle vs centroid angle ---------
    print("C10'(b)  mean pairwise angle between users [deg]  (seed-mean)")
    for a in ALPHAS:
        cang = mean(a, "pair_angle_c")
        print(f"  alpha={a:g}: centroid-centroid = {cang:.4f}")
        print("    |H|      m-m angle      gap to centroid curve")
        for h in H_SIZES:
            ma = mean(a, "pair_angle_m", h)
            print(f"    {h:<6d} {ma:12.4f}   {ma - cang:+12.4f}")
    print()

    # ---------------- C10'(c): centroid angle vs alpha --------------------
    print("C10'(c)  between-centroid mean pairwise angle [deg] per alpha")
    print("  alpha   seed-mean   per-seed")
    cvals = {}
    for a in ALPHAS:
        vs = seed_vals(a, "pair_angle_c")
        cvals[a] = float(np.mean(vs))
        print(f"  {a:<7g} {cvals[a]:9.4f}   " +
              " ".join(f"{v:8.4f}" for v in vs))
    viol_c = check_monotone([cvals[a] for a in ALPHAS], "dec")
    for s in SEEDS:
        sc = [per_seed[s][a]["pair_angle_c"] for a in ALPHAS]
        v = check_monotone(sc, "dec")
        if v:
            viol_c.append((f"seed{s}", v))
    print(f"  decrease-with-alpha violations: {viol_c if viol_c else 'NONE'}\n")

    # ---------------- C10'(d): SNR table -----------------------------------
    print("C10'(d)  SNR = pairwise m-m angle / own-centroid angle  (seed-mean)")
    print("  |H|   " + "".join(f"a={a:<9g}" for a in ALPHAS))
    snr = {a: {h: mean(a, "pair_angle_m", h) / mean(a, "own_angle", h)
               for h in H_SIZES} for a in ALPHAS}
    for h in H_SIZES:
        print(f"  {h:<5d}" + "".join(fmt(snr[a][h]) for a in ALPHAS))
    viol_d = []
    for a in ALPHAS:
        for i, d in check_monotone([snr[a][h] for h in H_SIZES], "inc"):
            viol_d.append(("mean", a, H_SIZES[i], H_SIZES[i+1], d))
        for s in SEEDS:
            sc = [per_seed[s][a]["pair_angle_m"][h] /
                  per_seed[s][a]["own_angle"][h] for h in H_SIZES]
            for i, d in check_monotone(sc, "inc"):
                viol_d.append((f"seed{s}", a, H_SIZES[i], H_SIZES[i+1], d))
    print(f"  monotone-increase violations: {viol_d if viol_d else 'NONE'}\n")

    # ---------------- C11'(a): norm handover, alpha = 2 --------------------
    a = 2.0
    print("C11'(a)  mean ||m_u|| vs |H|  (alpha=2, seed-mean)")
    cn = mean(a, "mean_norm_c")
    print("  |H|      mean ||m_u||")
    for h in H_SIZES:
        print(f"  {h:<6d} {mean(a, 'mean_norm_m', h):12.4f}")
    print(f"  mean ||c_u|| (asymptote) = {cn:.4f}")
    viol_11a = check_monotone([mean(a, "mean_norm_m", h) for h in H_SIZES],
                              "dec")
    for s in SEEDS:
        sc = [per_seed[s][a]["mean_norm_m"][h] for h in H_SIZES]
        v = check_monotone(sc, "dec")
        if v:
            viol_11a.append((f"seed{s}", v))
    print(f"  monotone-decrease violations: "
          f"{viol_11a if viol_11a else 'NONE'}\n")

    # ---------------- C11'(b): ID share -------------------------------------
    print("C11'(b)  ID-row share  (alpha=2, seed-mean)")
    print("  |H|    mean ||e_ID||^2/(||m||^2+||e_ID||^2)   mean ||m||/||e_ID||")
    for h in H_SIZES:
        print(f"  {h:<6d} {mean(a, 'id_share', h):24.4f}"
              f" {mean(a, 'ratio_m_id', h):22.4f}")
    viol_share = check_monotone([mean(a, "id_share", h) for h in H_SIZES],
                                "inc")
    viol_ratio = check_monotone([mean(a, "ratio_m_id", h) for h in H_SIZES],
                                "dec")
    for s in SEEDS:
        v1 = check_monotone([per_seed[s][a]["id_share"][h]
                             for h in H_SIZES], "inc")
        v2 = check_monotone([per_seed[s][a]["ratio_m_id"][h]
                             for h in H_SIZES], "dec")
        if v1:
            viol_share.append((f"seed{s}", v1))
        if v2:
            viol_ratio.append((f"seed{s}", v2))
    print(f"  share monotone-increase violations: "
          f"{viol_share if viol_share else 'NONE'}")
    print(f"  ratio monotone-decrease violations: "
          f"{viol_ratio if viol_ratio else 'NONE'}")


if __name__ == "__main__":
    main()
