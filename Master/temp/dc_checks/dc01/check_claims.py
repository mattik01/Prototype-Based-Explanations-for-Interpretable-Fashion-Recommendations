# dc01 claims check — toy-scale verification of C1..C6 from claims_spec.md
# Pure torch (float64), no training, seconds of runtime.
import torch

torch.set_default_dtype(torch.float64)
TOL = 1e-10  # float64 exactness tolerance


def shifted_cos(x, P):
    # x: (d,), P: (K,d) -> (K,) of 1 + cos(x, P[k])
    return 1.0 + (P @ x) / (x.norm() * P.norm(dim=1))


def make_params(g, d, Ku, Kt, N, V, M, scale=1.0):
    E = torch.randn(V + M, d, generator=g) * scale
    U = torch.randn(N, d, generator=g)
    Pu = torch.randn(Ku, d, generator=g)
    Pt = torch.randn(Kt, d, generator=g)
    Wu = torch.randn(Kt, d, generator=g)
    Wt = torch.randn(Ku, d, generator=g)
    return E, U, Pu, Pt, Wu, Wt


def forward(u_vec, q, Pu, Pt, Wu, Wt):
    ustar = shifted_cos(u_vec, Pu)            # (Ku,)
    tstar = shifted_cos(q, Pt)                # (Kt,)
    uhat = Wu @ u_vec                         # (Kt,)
    that = Wt @ q                             # (Ku,)
    S = ustar @ that + uhat @ tstar
    return ustar, tstar, uhat, that, S


def report(name, ok, evidence):
    print(f"{name}: {'CONFIRMED' if ok else 'FALSIFIED'} | {evidence}")
    return ok


results = {}

# ---------------- C1: exact per-feature decomposition of tstar ----------------
# tstar_k - 1 = sum_r c_{r,k},  c_{r,k} = (E[r]·Pt[k]) / (||q||·||Pt[k]||)
max_err_c1 = 0.0
for seed in range(10):
    g = torch.Generator().manual_seed(seed)
    for F in [0, 1, 2, 3, 5, 7, 11]:  # odd & even feature-set sizes, incl. F=0
        d, Ku, Kt, V, M, N = 8, 3, 4, 20, 6, 5
        E, U, Pu, Pt, Wu, Wt = make_params(g, d, Ku, Kt, N, V, M,
                                           scale=10.0 ** (seed % 3 - 1))
        rows = torch.randperm(V, generator=g)[:F].tolist() + [V + 0]  # F features + ID
        q = E[rows].sum(dim=0)
        tstar = shifted_cos(q, Pt)
        c = (E[rows] @ Pt.T) / (q.norm() * Pt.norm(dim=1))  # (F+1, Kt)
        err = (tstar - 1.0 - c.sum(dim=0)).abs().max().item()
        max_err_c1 = max(max_err_c1, err)
results["C1"] = report("C1", max_err_c1 < TOL, f"max abs error {max_err_c1:.3e} over 10 seeds x F in {{0,1,2,3,5,7,11}}, param scales 0.1/1/10")

# ---------------- C2: cold-item non-degeneracy ----------------
c2a_min_diff, c2b_max_diff, c2c_min_spread = float("inf"), 0.0, float("inf")
for seed in range(10):
    g = torch.Generator().manual_seed(100 + seed)
    d, Ku, Kt, V, M, N, F = 8, 3, 4, 20, 6, 5, 3
    E, U, Pu, Pt, Wu, Wt = make_params(g, d, Ku, Kt, N, V, M)
    # cold IDs: zero rows
    E[V + 0] = 0.0
    E[V + 1] = 0.0
    # (a) different feature sets, zero ID rows
    feats_a = [0, 1, 2]
    feats_b = [3, 4, 5]
    q_a = E[feats_a + [V + 0]].sum(0)
    q_b = E[feats_b + [V + 1]].sum(0)
    diff_ab = (shifted_cos(q_a, Pt) - shifted_cos(q_b, Pt)).abs().max().item()
    c2a_min_diff = min(c2a_min_diff, diff_ab)
    # (b) identical feature sets, zero ID rows
    q_c = E[feats_a + [V + 1]].sum(0)
    diff_ac = (shifted_cos(q_a, Pt) - shifted_cos(q_c, Pt)).abs().max().item()
    c2b_max_diff = max(c2b_max_diff, diff_ac)
    # also (b) with tiny random ID rows -> near-identical but not exactly
    # (c) tstar non-constant across k
    spread = (shifted_cos(q_a, Pt).max() - shifted_cos(q_a, Pt).min()).item()
    c2c_min_spread = min(c2c_min_spread, spread)
ok_c2 = (c2a_min_diff > 1e-3) and (c2b_max_diff < TOL) and (c2c_min_spread > 1e-3)
results["C2"] = report("C2", ok_c2,
    f"(a) min max-abs tstar diff (different feats) {c2a_min_diff:.3e}; "
    f"(b) max diff (identical feats, zero IDs) {c2b_max_diff:.3e}; "
    f"(c) min activation spread across k {c2c_min_spread:.3e}; 10 seeds")

# ---------------- C3: gradient flows through tie from BOTH score halves ----------------
min_g1, min_g2 = float("inf"), float("inf")
for seed in range(10):
    g = torch.Generator().manual_seed(200 + seed)
    d, Ku, Kt, V, M, N, F = 8, 3, 4, 20, 6, 5, 3
    E, U, Pu, Pt, Wu, Wt = make_params(g, d, Ku, Kt, N, V, M)
    E.requires_grad_(True)
    rows = [0, 1, 2, V + 0]
    u_vec = U[1]

    def halves():
        q = E[rows].sum(0)
        ustar = shifted_cos(u_vec, Pu)
        tstar = shifted_cos(q, Pt)
        uhat = Wu @ u_vec
        that = Wt @ q
        return ustar @ that, uhat @ tstar

    t1, _ = halves()
    g1 = torch.autograd.grad(t1, E)[0]      # grad of ustar·that wrt E
    _, t2 = halves()
    g2 = torch.autograd.grad(t2, E)[0]      # grad of uhat·tstar wrt E
    # feature row r = 0 (a feature row of the scored item)
    n1 = g1[0].norm().item()
    n2 = g2[0].norm().item()
    min_g1, min_g2 = min(min_g1, n1), min(min_g2, n2)
ok_c3 = min_g1 > 1e-6 and min_g2 > 1e-6
results["C3"] = report("C3", ok_c3,
    f"min ||dS_half/dE[feature row]|| over 10 seeds: ustar·that term {min_g1:.3e}, uhat·tstar term {min_g2:.3e} (both nonzero)")

# ---------------- C4: per-prototype additive decomposition of S ----------------
max_err_c4 = 0.0
for seed in range(10):
    g = torch.Generator().manual_seed(300 + seed)
    for (Ku, Kt) in [(3, 4), (1, 1), (7, 2)]:
        d, V, M, N, F = 8, 20, 6, 5, 3
        E, U, Pu, Pt, Wu, Wt = make_params(g, d, Ku, Kt, N, V, M)
        rows = [0, 1, 2, V + 0]
        q = E[rows].sum(0)
        ustar, tstar, uhat, that, S = forward(U[2], q, Pu, Pt, Wu, Wt)
        contribs = torch.cat([ustar * that, uhat * tstar])  # Ku + Kt scalars
        err = abs(S.item() - contribs.sum().item())
        max_err_c4 = max(max_err_c4, err)
results["C4"] = report("C4", max_err_c4 < TOL,
    f"max |S - sum of Ku+Kt per-prototype contributions| = {max_err_c4:.3e} over 10 seeds x (Ku,Kt) in {{(3,4),(1,1),(7,2)}}")

# ---------------- C5: F=0 reduces to ID-only (plain item embedding) model ----------------
max_err_c5 = 0.0
for seed in range(10):
    g = torch.Generator().manual_seed(400 + seed)
    d, Ku, Kt, V, M, N = 8, 3, 4, 20, 6, 5
    E, U, Pu, Pt, Wu, Wt = make_params(g, d, Ku, Kt, N, V, M)
    Q = E[V:].clone()  # matched plain per-item embedding table
    for i in range(M):
        q_comp = E[[V + i]].sum(0)          # composed, F=0: only ID row
        q_plain = Q[i]                       # plain table lookup
        out_a = forward(U[i % N], q_comp, Pu, Pt, Wu, Wt)
        out_b = forward(U[i % N], q_plain, Pu, Pt, Wu, Wt)
        for a, b in zip(out_a, out_b):
            max_err_c5 = max(max_err_c5, (a - b).abs().max().item())
results["C5"] = report("C5", max_err_c5 < TOL,
    f"max abs diff across all outputs (ustar,tstar,uhat,that,S) composed-F=0 vs plain table = {max_err_c5:.3e}, 10 seeds x 6 items")

# ---------------- C6: norm coupling — scaling one row preserves other rows' ratios ----------------
max_ratio_err, max_common_factor_err = 0.0, 0.0
for seed in range(10):
    g = torch.Generator().manual_seed(500 + seed)
    for alpha in [10.0, 0.1, -3.0]:
        d, Ku, Kt, V, M, N, F = 8, 3, 4, 20, 6, 5, 5
        E, U, Pu, Pt, Wu, Wt = make_params(g, d, Ku, Kt, N, V, M)
        rows = [0, 1, 2, 3, 4, V + 0]
        r0_idx = 2  # scale rows[2]

        def contributions(Em):
            q = Em[rows].sum(0)
            return (Em[rows] @ Pt.T) / (q.norm() * Pt.norm(dim=1))  # (F+1, Kt)

        c_before = contributions(E)
        E2 = E.clone()
        E2[rows[r0_idx]] *= alpha
        c_after = contributions(E2)
        others = [j for j in range(len(rows)) if j != r0_idx]
        cb, ca = c_before[others].flatten(), c_after[others].flatten()
        # ratios between other rows' contributions unchanged
        ratio_b = cb[:, None] / cb[None, :]
        ratio_a = ca[:, None] / ca[None, :]
        max_ratio_err = max(max_ratio_err, (ratio_a - ratio_b).abs().max().item())
        # absolute values change by one common factor = ||q_old|| / ||q_new||
        q_old = E[rows].sum(0).norm()
        q_new = E2[rows].sum(0).norm()
        factor = (q_old / q_new).item()
        max_common_factor_err = max(max_common_factor_err,
                                    (ca - factor * cb).abs().max().item())
ok_c6 = max_ratio_err < 1e-8 and max_common_factor_err < TOL
results["C6"] = report("C6", ok_c6,
    f"max ratio change among non-scaled rows {max_ratio_err:.3e}; max |c_after - (||q_old||/||q_new||)·c_before| = {max_common_factor_err:.3e}; 10 seeds x alpha in {{10, 0.1, -3}}")

print("\nALL CONFIRMED" if all(results.values()) else "\nSOME FALSIFIED")
