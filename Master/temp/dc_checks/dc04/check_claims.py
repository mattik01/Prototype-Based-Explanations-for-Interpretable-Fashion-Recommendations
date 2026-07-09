"""
Black-box toy verification of dc04 claims CL1-CL12.

Only input: claims_spec.md in this directory. Implements the mechanism from
scratch in torch (float64, CPU, toy dims), no repo code, no datasets.

Run: conda activate protomf; python check_claims.py
"""
import math
import torch

torch.set_default_dtype(torch.float64)

RESULTS = []


def report(claim, verdict, detail):
    RESULTS.append((claim, verdict, detail))
    print(f"[{claim}] {verdict}\n{detail}\n")


# ----------------------------------------------------------------------------
# Mechanism implementation
# ----------------------------------------------------------------------------

def cos_mat(x, P):
    """cosine similarity between x (..., d) and each row of P (n, d) -> (..., n)."""
    num = x @ P.t()
    den = x.norm(dim=-1, keepdim=True) * P.norm(dim=-1)
    return num / den


def make_feature_ids(M, F, V, gen):
    """(M, F) int matrix; field f draws values from its own block of V//F ids."""
    assert V % F == 0
    per = V // F
    cols = []
    for f in range(F):
        cols.append(torch.randint(0, per, (M,), generator=gen) + f * per)
    return torch.stack(cols, dim=1)


def refresh_anchors(T, feature_ids, warm_mask):
    """Vectorized refresh: c_v = mean of T rows over warm items containing v."""
    V_ = int(feature_ids.max().item()) + 1
    d = T.shape[1]
    C = torch.zeros(V_, d, dtype=T.dtype)
    counts = torch.zeros(V_, dtype=T.dtype)
    warm_ids = feature_ids[warm_mask]                     # (Mw, F)
    Tw = T[warm_mask]                                     # (Mw, d)
    F = warm_ids.shape[1]
    flat = warm_ids.reshape(-1)                           # (Mw*F,)
    C.index_add_(0, flat, Tw.repeat_interleave(F, dim=0))
    counts.index_add_(0, flat, torch.ones(flat.shape[0], dtype=T.dtype))
    C = C / counts.clamp(min=1.0).unsqueeze(1)
    return C, counts


def refresh_anchors_naive(T, feature_ids, warm_mask):
    V_ = int(feature_ids.max().item()) + 1
    d = T.shape[1]
    C = torch.zeros(V_, d, dtype=T.dtype)
    counts = torch.zeros(V_, dtype=T.dtype)
    M = T.shape[0]
    for v in range(V_):
        acc = torch.zeros(d, dtype=T.dtype)
        n = 0
        for i in range(M):
            if warm_mask[i] and (feature_ids[i] == v).any():
                acc = acc + T[i]
                n += 1
        counts[v] = n
        if n > 0:
            C[v] = acc / n
    return C, counts


class ToyModel:
    def __init__(self, N_u, M, d, L_u, K, V, F, gen, requires_grad=False):
        self.N_u, self.M, self.d = N_u, M, d
        self.L_u, self.K, self.V, self.F = L_u, K, V, F
        rg = requires_grad
        self.U = torch.randn(N_u, d, generator=gen).requires_grad_(rg)
        self.P_u = torch.randn(L_u, d, generator=gen).requires_grad_(rg)
        self.W_u = torch.randn(d, K, generator=gen).requires_grad_(rg)
        self.T = torch.randn(M, d, generator=gen).requires_grad_(rg)
        self.W_t = torch.randn(d, L_u, generator=gen).requires_grad_(rg)
        self.Atil = torch.randn(K, V, generator=gen).requires_grad_(rg)
        self.feature_ids = make_feature_ids(M, F, V, gen)
        self.warm_mask = torch.ones(M, dtype=torch.bool)
        self.refresh()

    def refresh(self):
        with torch.no_grad():
            C, counts = refresh_anchors(self.T, self.feature_ids, self.warm_mask)
        self.C = C            # buffer: requires_grad is False by construction
        self.counts = counts

    def item_repr(self, i_idxs, cold_mask=None, C_override=None):
        C = self.C if C_override is None else C_override
        t = self.T[i_idxs]                                  # (B, N, d)
        if cold_mask is not None:
            t_cold_all = C[self.feature_ids].mean(dim=1)    # (M, d), (1/F) sum
            cm = cold_mask[i_idxs]                          # (B, N)
            t = torch.where(cm.unsqueeze(-1), t_cold_all[i_idxs], t)
        return t

    def forward(self, u_idxs, i_idxs, tau=None, cold_mask=None, C_override=None,
                center=False):
        C = self.C if C_override is None else C_override
        if center:
            cbar = self.T[self.warm_mask].mean(dim=0)
            C = C - cbar
        A = torch.softmax(self.Atil if tau is None else self.Atil / tau, dim=-1)
        u = self.U[u_idxs]                                  # (B, d)
        u_star = 1 + cos_mat(u, self.P_u)                   # (B, L_u)
        u_hat = u @ self.W_u                                # (B, K)
        P_t = A @ C                                         # (K, d)
        t = self.item_repr(i_idxs, cold_mask, C)            # (B, N, d)
        t_star = 1 + cos_mat(t, P_t)                        # (B, N, K)
        t_hat = t @ self.W_t                                # (B, N, L_u)
        score = (u_star.unsqueeze(1) * t_hat).sum(-1) \
              + (u_hat.unsqueeze(1) * t_star).sum(-1)       # (B, N)
        return dict(u_star=u_star, u_hat=u_hat, t_star=t_star, t_hat=t_hat,
                    score=score, A=A, P_t=P_t, t=t)


def toy_dims():
    return dict(B=8, N=5, d=16, L_u=4, K=6, V=15, F=3, M=40, N_u=30)


def make_model_and_batch(seed, requires_grad=False):
    D = toy_dims()
    gen = torch.Generator().manual_seed(seed)
    m = ToyModel(D['N_u'], D['M'], D['d'], D['L_u'], D['K'], D['V'], D['F'],
                 gen, requires_grad=requires_grad)
    u_idxs = torch.randint(0, D['N_u'], (D['B'],), generator=gen)
    i_idxs = torch.randint(0, D['M'], (D['B'], D['N']), generator=gen)
    return m, u_idxs, i_idxs, D, gen


# ----------------------------------------------------------------------------
# CL1: shapes
# ----------------------------------------------------------------------------

def cl1():
    m, u_idxs, i_idxs, D, _ = make_model_and_batch(0)
    out = m.forward(u_idxs, i_idxs)
    want = dict(u_star=(8, 4), u_hat=(8, 6), t_star=(8, 5, 6),
                t_hat=(8, 5, 4), score=(8, 5))
    got = {k: tuple(out[k].shape) for k in want}
    ok = got == want
    report('CL1', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  expected {want}\n  got      {got}")


# ----------------------------------------------------------------------------
# CL2: exact per-anchor decomposition of t_star - 1
# ----------------------------------------------------------------------------

def cl2():
    worst = 0.0
    for seed in range(5):
        m, u_idxs, i_idxs, D, _ = make_model_and_batch(seed)
        out = m.forward(u_idxs, i_idxs)
        t, A, P_t = out['t'], out['A'], out['P_t']          # (B,N,d),(K,V),(K,d)
        C = m.C
        # RHS: sum_v A[k,v] * (t . C[v]) / (|t| |P_t[k]|)
        tC = t @ C.t()                                      # (B,N,V)
        rhs = tC @ A.t()                                    # (B,N,K) = sum_v A[k,v] t.C[v]
        rhs = rhs / (t.norm(dim=-1, keepdim=True) * P_t.norm(dim=-1))
        lhs = out['t_star'] - 1.0
        worst = max(worst, (lhs - rhs).abs().max().item())
    ok = worst < 1e-12
    report('CL2', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  max |lhs - rhs| over 5 seeds = {worst:.3e} (tol 1e-12)")


# ----------------------------------------------------------------------------
# CL3: exact score decomposition into L_u + K scalar addends
# ----------------------------------------------------------------------------

def cl3():
    worst = 0.0
    for seed in range(5):
        m, u_idxs, i_idxs, D, _ = make_model_and_batch(seed)
        out = m.forward(u_idxs, i_idxs)
        us, uh, ts, th, sc = (out[k] for k in
                              ('u_star', 'u_hat', 't_star', 't_hat', 'score'))
        B, N = sc.shape
        for b in range(B):
            for n in range(N):
                addends = ([us[b, l].item() * th[b, n, l].item()
                            for l in range(D['L_u'])] +
                           [uh[b, k].item() * ts[b, n, k].item()
                            for k in range(D['K'])])
                assert len(addends) == D['L_u'] + D['K']
                worst = max(worst, abs(sum(addends) - sc[b, n].item()))
    ok = worst < 1e-12
    report('CL3', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  {toy_dims()['L_u'] + toy_dims()['K']} addends per (u,i); "
           f"max residual over 5 seeds = {worst:.3e} (tol 1e-12)")


# ----------------------------------------------------------------------------
# CL4: gradient routing
# ----------------------------------------------------------------------------

def cl4():
    m, u_idxs, i_idxs, D, _ = make_model_and_batch(3, requires_grad=True)
    m.refresh()  # C built under no_grad from current T
    assert m.C.requires_grad is False
    out = m.forward(u_idxs, i_idxs)
    loss = (out['score'] ** 2).sum()
    loss.backward()
    gnorms = {name: getattr(m, name).grad.norm().item()
              for name in ('Atil', 'T', 'U', 'P_u', 'W_u', 'W_t')}
    all_nonzero = all(v > 0 for v in gnorms.values())
    c_grad_none = m.C.grad is None and not m.C.requires_grad

    # Counterfactual: C as a leaf with requires_grad=True (still detached values)
    C_leaf = m.C.detach().clone().requires_grad_(True)
    out2 = m.forward(u_idxs, i_idxs, C_override=C_leaf)
    (out2['score'] ** 2).sum().backward()
    c_math_grad = C_leaf.grad.norm().item()

    ok = all_nonzero and c_grad_none and c_math_grad > 0
    report('CL4', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  grad norms: {({k: f'{v:.3e}' for k, v in gnorms.items()})}\n"
           f"  buffer C: requires_grad={m.C.requires_grad}, C.grad is "
           f"{'None' if m.C.grad is None else 'set'}\n"
           f"  counterfactual leaf-C grad norm (mathematical dScore/dC) = "
           f"{c_math_grad:.3e}")


# ----------------------------------------------------------------------------
# CL5: convex hull property & temperature limit
# ----------------------------------------------------------------------------

def cl5():
    hull_ok, hull_detail = True, []
    for seed in range(3):
        m, u_idxs, i_idxs, D, _ = make_model_and_batch(seed + 10)
        A = torch.softmax(m.Atil, dim=-1)
        P_t = A @ m.C
        nonneg = (A >= 0).all().item()
        rowsum_err = (A.sum(dim=-1) - 1).abs().max().item()
        recon_err = (A @ m.C - P_t).abs().max().item()
        hull_ok &= nonneg and rowsum_err < 1e-12 and recon_err == 0.0
        hull_detail.append((nonneg, rowsum_err, recon_err))
    # temperature limit
    temp_ok, temp_detail = True, []
    for seed in range(3):
        m, u_idxs, i_idxs, D, _ = make_model_and_batch(seed + 20)
        A = torch.softmax(m.Atil / 1e-6, dim=-1)
        P_t = A @ m.C
        arg = m.Atil.argmax(dim=-1)
        target = m.C[arg]
        err = (P_t - target).abs().max().item()
        exact = torch.equal(P_t, target)
        temp_ok &= err == 0.0
        temp_detail.append((exact, err))
    ok = hull_ok and temp_ok
    report('CL5', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  hull (3 seeds): A>=0 / max|rowsum-1| / |A@C - P_t|: {hull_detail}\n"
           f"  tau=1e-6 (3 seeds): (bitwise equal to C[argmax], max err): "
           f"{temp_detail}")


# ----------------------------------------------------------------------------
# CL6: cold-item rule
# ----------------------------------------------------------------------------

def cl6():
    # (a) identical attribute rows -> bitwise identical reps and scores
    m, u_idxs, _, D, gen = make_model_and_batch(30)
    cold_mask = torch.zeros(D['M'], dtype=torch.bool)
    cold_mask[0] = cold_mask[1] = True
    m.warm_mask = ~cold_mask
    m.feature_ids[1] = m.feature_ids[0]          # identical attribute rows
    m.refresh()
    i_idxs = torch.tensor([[0, 1, 2, 3, 4]]).repeat(D['B'], 1)
    out = m.forward(u_idxs, i_idxs, cold_mask=cold_mask)
    rep_equal = torch.equal(out['t'][:, 0, :], out['t'][:, 1, :])
    score_equal = torch.equal(out['score'][:, 0], out['score'][:, 1])
    tstar_equal = torch.equal(out['t_star'][:, 0, :], out['t_star'][:, 1, :])

    # (b) finite & non-constant across k
    finite = torch.isfinite(out['t_star'][:, 0, :]).all().item()
    std_k = out['t_star'][0, 0, :].std().item()

    # (c) differing attribute rows -> different t_star; min over seeds of max diff
    maxdiffs = []
    for seed in range(12):
        m2, u2, _, D2, g2 = make_model_and_batch(100 + seed)
        cm = torch.zeros(D2['M'], dtype=torch.bool)
        cm[0] = cm[1] = True
        m2.warm_mask = ~cm
        m2.feature_ids[1] = m2.feature_ids[0].clone()
        # change one field's value
        old = m2.feature_ids[1, 0].item()
        per = D2['V'] // D2['F']
        m2.feature_ids[1, 0] = (old + 1) % per          # stay in field-0 block
        m2.refresh()
        ii = torch.tensor([[0, 1]]).repeat(D2['B'], 1)
        o2 = m2.forward(u2, ii, cold_mask=cm)
        maxdiffs.append((o2['t_star'][:, 0, :] - o2['t_star'][:, 1, :])
                        .abs().max().item())
    min_maxdiff = min(maxdiffs)
    ok = (rep_equal and score_equal and tstar_equal and finite
          and std_k > 1e-8 and min_maxdiff > 1e-8)
    report('CL6', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  (a) identical attrs: rep bitwise equal={rep_equal}, "
           f"t_star equal={tstar_equal}, scores equal={score_equal}\n"
           f"  (b) finite={finite}, std over k of one cold t_star={std_k:.4f}\n"
           f"  (c) min over 12 seeds of max |t_star diff| = {min_maxdiff:.4e}")


# ----------------------------------------------------------------------------
# CL7: spread vs table structure
# ----------------------------------------------------------------------------

def spread_tstar(T_items, P):
    ts = 1 + cos_mat(T_items, P)     # (M, K)
    return ts.std(dim=1).mean().item()


def build_tables(seed, M=200, d=16, F=3, V=15, mu=None):
    gen = torch.Generator().manual_seed(seed)
    fids = make_feature_ids(M, F, V, gen)
    T_iso = torch.randn(M, d, generator=gen)
    if mu is not None:
        T_iso = T_iso + mu
    centers = torch.randn(V, d, generator=gen)
    T_struct = centers[fids].mean(dim=1) + 0.1 * torch.randn(M, d, generator=gen)
    return fids, T_iso, T_struct, gen


def cl7():
    K, d, V = 6, 16, 15
    lines = []
    ratios_a = {1.0: [], 3.0: []}
    ratios_b = {1.0: [], 3.0: []}
    for seed in range(6):
        fids, T_iso, T_struct, gen = build_tables(seed)
        warm = torch.ones(T_iso.shape[0], dtype=torch.bool)
        C_iso, _ = refresh_anchors(T_iso, fids, warm)
        C_struct, _ = refresh_anchors(T_struct, fids, warm)
        P_free = torch.randn(K, d, generator=gen)
        for scale in (1.0, 3.0):
            Atil = scale * torch.randn(K, V, generator=gen)
            A = torch.softmax(Atil, dim=-1)
            s_anch_iso = spread_tstar(T_iso, A @ C_iso)
            s_free_iso = spread_tstar(T_iso, P_free)
            s_anch_st = spread_tstar(T_struct, A @ C_struct)
            s_free_st = spread_tstar(T_struct, P_free)
            ratios_a[scale].append(s_free_iso / s_anch_iso)
            ratios_b[scale].append(s_anch_st / s_free_st)
            lines.append(f"    seed={seed} scale={scale}: ISO anch={s_anch_iso:.4f} "
                         f"free={s_free_iso:.4f} (free/anch={s_free_iso/s_anch_iso:.1f}) | "
                         f"STRUCT anch={s_anch_st:.4f} free={s_free_st:.4f} "
                         f"(anch/free={s_anch_st/s_free_st:.2f})")
    # Supplementary scan: profile logit scale x table size, to probe the
    # ambiguous "random profiles A" reading (near-uniform vs peaked profiles).
    scan_lines = []
    for M in (200, 1000):
        for scale in (0.01, 0.1, 0.3, 1.0, 3.0, 10.0):
            ra, rb = [], []
            for seed in range(5):
                fids, T_iso, T_struct, gen = build_tables(1000 + seed, M=M)
                warm = torch.ones(M, dtype=torch.bool)
                C_iso, _ = refresh_anchors(T_iso, fids, warm)
                C_st, _ = refresh_anchors(T_struct, fids, warm)
                P_free = torch.randn(K, d, generator=gen)
                A = torch.softmax(scale * torch.randn(K, V, generator=gen), -1)
                ra.append(spread_tstar(T_iso, P_free)
                          / spread_tstar(T_iso, A @ C_iso))
                rb.append(spread_tstar(T_struct, A @ C_st)
                          / spread_tstar(T_struct, P_free))
            scan_lines.append(
                f"    M={M:4d} logit-scale={scale:5.2f}: "
                f"CL7a free/anch min={min(ra):7.1f} max={max(ra):7.1f} | "
                f"CL7b anch/free min={min(rb):.3f} max={max(rb):.3f}")
    ok_a = {sc: min(r) >= 5.0 for sc, r in ratios_a.items()}
    ok_b = {sc: min(r) >= 0.3 for sc, r in ratios_b.items()}
    verdict_a = all(ok_a.values())
    verdict_b = all(ok_b.values())
    det = "\n".join(lines)
    summary = (f"  CL7a min(free/anchored) under ISO: scale1={min(ratios_a[1.0]):.1f}, "
               f"scale3={min(ratios_a[3.0]):.1f} (claim >= ~5) -> "
               f"{'ok' if verdict_a else 'VIOLATED'}\n"
               f"  CL7b min(anchored/free) under STRUCT: scale1={min(ratios_b[1.0]):.2f}, "
               f"scale3={min(ratios_b[3.0]):.2f} (claim >= ~0.3) -> "
               f"{'ok' if verdict_b else 'VIOLATED'}\n" + det
               + "\n  scan over profile peakedness (5 seeds each):\n"
               + "\n".join(scan_lines))
    report('CL7a', 'CONFIRMED' if verdict_a else 'FALSIFIED', summary)
    report('CL7b', 'CONFIRMED' if verdict_b else 'FALSIFIED',
           '  (see CL7a detail above)')


# ----------------------------------------------------------------------------
# CL8: weight-decay uniformity drift
# ----------------------------------------------------------------------------

def entropy(p):
    return -(p * p.clamp_min(1e-300).log()).sum(dim=-1)


def cl8():
    gen = torch.Generator().manual_seed(7)
    min_inc = float('inf')
    n_rows = 0
    for sigma in (0.5, 2.0, 5.0):
        for gamma in (0.5, 0.9, 0.99):
            Z = sigma * torch.randn(500, 15, generator=gen)
            H0 = entropy(torch.softmax(Z, dim=-1))
            H1 = entropy(torch.softmax(gamma * Z, dim=-1))
            inc = (H1 - H0).min().item()
            min_inc = min(min_inc, inc)
            n_rows += 500
    ok = min_inc > 0
    report('CL8', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  {n_rows} random non-uniform rows, sigma in (0.5,2,5), "
           f"gamma=(1-eta*lambda) in (0.5,0.9,0.99):\n"
           f"  min entropy increase = {min_inc:.3e} (strictly > 0 required)")


# ----------------------------------------------------------------------------
# CL9: K-ENT mechanics
# ----------------------------------------------------------------------------

def L_ent(Atil, tau=None):
    A = torch.softmax(Atil if tau is None else Atil / tau, dim=-1)
    return entropy(A).mean()


def cl9():
    V, K = 15, 6
    # one-hot-like
    Atil_oh = torch.zeros(K, V)
    Atil_oh[torch.arange(K), torch.arange(K)] = 30.0
    l_oh = L_ent(Atil_oh).item()
    # uniform
    l_unif = L_ent(torch.zeros(K, V)).item()
    lnV = math.log(V)
    # gradient + GD step from mildly peaked
    gen = torch.Generator().manual_seed(11)
    Atil = (0.5 * torch.randn(K, V, generator=gen)).requires_grad_(True)
    with torch.no_grad():
        Atil[torch.arange(K), torch.randint(0, V, (K,), generator=gen)] += 1.5
    l0 = L_ent(Atil)
    l0.backward()
    gnorm = Atil.grad.norm().item()
    with torch.no_grad():
        Atil2 = Atil - 0.5 * Atil.grad
    l1 = L_ent(Atil2).item()
    ok = l_oh < 1e-10 and abs(l_unif - lnV) < 1e-12 and gnorm > 0 and l1 < l0.item()
    report('CL9', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  L_ent(one-hot-like, logit=30) = {l_oh:.3e} (~0)\n"
           f"  L_ent(uniform) = {l_unif:.12f} vs ln V = {lnV:.12f}\n"
           f"  grad norm on Atil = {gnorm:.3e}; GD step (lr=0.5): "
           f"L_ent {l0.item():.6f} -> {l1:.6f} (decrease={l0.item()-l1:.3e})")


# ----------------------------------------------------------------------------
# CL10: K-SEP mechanics
# ----------------------------------------------------------------------------

def L_sep_from_A(A):
    K = A.shape[0]
    tot, n = 0.0 * A.sum(), 0
    for k in range(K):
        for j in range(K):
            if k != j:
                tot = tot + torch.minimum(A[k], A[j]).sum()
                n += 1
    return tot / n


def cl10():
    V = 15
    # identical rows
    gen = torch.Generator().manual_seed(13)
    z = torch.randn(V, generator=gen)
    A_id = torch.softmax(torch.stack([z, z]), dim=-1)
    l_id = L_sep_from_A(A_id).item()
    # disjoint supports: near-one-hot on different values
    z1 = torch.full((V,), -20.0); z1[2] = 20.0
    z2 = torch.full((V,), -20.0); z2[9] = 20.0
    A_dis = torch.softmax(torch.stack([z1, z2]), dim=-1)
    l_dis = L_sep_from_A(A_dis).item()
    # near-duplicate pair: GD step decreases overlap
    zb = z + 0.05 * torch.randn(V, generator=gen)
    Atil = torch.stack([z, zb]).clone().requires_grad_(True)
    A0 = torch.softmax(Atil, dim=-1)
    cos_prof = torch.nn.functional.cosine_similarity(A0[0], A0[1], dim=0).item()
    l0 = L_sep_from_A(A0)
    l0.backward()
    with torch.no_grad():
        Atil1 = Atil - 1.0 * Atil.grad
    l1 = L_sep_from_A(torch.softmax(Atil1, dim=-1)).item()
    # exact duplicate: what does autograd do?
    Atil_d = torch.stack([z, z]).clone().requires_grad_(True)
    ld = L_sep_from_A(torch.softmax(Atil_d, dim=-1))
    ld.backward()
    g = Atil_d.grad
    grad_zero = g.abs().max().item() == 0.0
    rows_equal_grad = torch.equal(g[0], g[1])
    with torch.no_grad():
        Atil_d1 = Atil_d - 1.0 * g
    ld1 = L_sep_from_A(torch.softmax(Atil_d1, dim=-1)).item()
    ok = (abs(l_id - 1.0) < 1e-12 and l_dis < 1e-8
          and cos_prof > 0.99 and l1 < l0.item())
    report('CL10', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  identical rows: L_sep = {l_id:.15f} (claim 1)\n"
           f"  disjoint near-one-hot: L_sep = {l_dis:.3e} (claim ~0)\n"
           f"  near-duplicate (profile cos={cos_prof:.6f}): GD step lr=1 "
           f"L_sep {l0.item():.6f} -> {l1:.6f} (decrease={l0.item()-l1:.3e})\n"
           f"  EXACT duplicate: grad max|.|={g.abs().max().item():.3e} "
           f"(zero={grad_zero}), grad rows identical={rows_equal_grad}, "
           f"L_sep after step = {ld1:.6f} (stays 1 iff rows move together)")


# ----------------------------------------------------------------------------
# CL11: K-CENTER in the degenerate (common-mean) regime
# ----------------------------------------------------------------------------

def cl11():
    K, d, V = 6, 16, 15
    ratios = []
    for seed in range(6):
        gen0 = torch.Generator().manual_seed(500 + seed)
        mu_dir = torch.randn(d, generator=gen0)
        mu = math.sqrt(d) * mu_dir / mu_dir.norm()      # ||mu|| = sqrt(d) ~ row norm
        fids, T_iso, _, gen = build_tables(600 + seed, mu=mu)
        warm = torch.ones(T_iso.shape[0], dtype=torch.bool)
        C, _ = refresh_anchors(T_iso, fids, warm)
        cbar = T_iso[warm].mean(dim=0)
        Cc = C - cbar
        Atil = torch.randn(K, V, generator=gen)
        A = torch.softmax(Atil, dim=-1)
        s_unc = spread_tstar(T_iso, A @ C)
        s_cen = spread_tstar(T_iso, A @ Cc)
        ratios.append(s_cen / s_unc)
    ok = all(r > 1.0 for r in ratios)
    report('CL11', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  spread(centered)/spread(uncentered) over 6 seeds: "
           f"{[f'{r:.1f}' for r in ratios]} (all must be > 1)")


# ----------------------------------------------------------------------------
# CL12: refresh correctness
# ----------------------------------------------------------------------------

def cl12():
    gen = torch.Generator().manual_seed(77)
    M, d, F, V = 40, 16, 3, 15
    fids = make_feature_ids(M, F, V, gen)
    T = torch.randn(M, d, generator=gen)
    warm = torch.rand(M, generator=gen) < 0.7
    C_vec, cnt_vec = refresh_anchors(T, fids, warm)
    C_nai, cnt_nai = refresh_anchors_naive(T, fids, warm)
    bitwise = torch.equal(C_vec, C_nai)
    maxerr = (C_vec - C_nai).abs().max().item()
    cnt_ok = torch.equal(cnt_vec, cnt_nai)
    # manual count check
    manual_counts = torch.tensor(
        [float(sum(1 for i in range(M) if warm[i] and (fids[i] == v).any()))
         for v in range(V)])
    cnt_manual_ok = torch.equal(cnt_vec, manual_counts)
    # cold items do not influence C
    T2 = T.clone()
    T2[~warm] += 1000.0 * torch.randn((~warm).sum(), d, generator=gen)
    C2, cnt2 = refresh_anchors(T2, fids, warm)
    cold_indep = torch.equal(C_vec, C2) and torch.equal(cnt_vec, cnt2)
    ok = (bitwise or maxerr < 1e-12) and cnt_ok and cnt_manual_ok and cold_indep
    report('CL12', 'CONFIRMED' if ok else 'FALSIFIED',
           f"  vectorized vs naive: bitwise equal={bitwise}, max err={maxerr:.3e}\n"
           f"  counts equal (naive={cnt_ok}, manual={cnt_manual_ok}); "
           f"warm counts = {cnt_vec.tolist()}\n"
           f"  perturbing non-warm rows by ~1000x leaves C bitwise unchanged: "
           f"{cold_indep}")


if __name__ == '__main__':
    print(f"torch {torch.__version__}, default dtype {torch.get_default_dtype()}\n")
    cl1(); cl2(); cl3(); cl4(); cl5(); cl6(); cl7(); cl8(); cl9(); cl10()
    cl11(); cl12()
    print("=" * 60)
    for c, v, _ in RESULTS:
        print(f"{c}: {v}")
