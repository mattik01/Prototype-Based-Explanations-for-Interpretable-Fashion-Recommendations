"""dc05 black-box toy check (protocol Step 4b).

Implements the mechanism EXACTLY as described in claims_spec.md (the only
input) and checks claims C1-C9. float64 everywhere, CPU, toy dims,
multiple seeds, adversarial scales.

Mechanism:
  E in R^{(V+N)xd}   word rows 0..V-1, ID row V+u (optional)
  P in R^{Kxd}       user prototypes
  T in R^{MxK}       free item vectors t_i
  w_{u,f} = n_{u,f}/|H_u|  (mean normalization, default)
  q_u = sum_f w_{u,f} E[f] (+ E[V+u] with weight 1 if ID enabled)
  sim(a,b) = 1 + cos(a,b)   (torch nn.CosineSimilarity semantics)
  u*_l = sim(q_u, P[l]);  S(u,i) = sum_l t_{i,l} u*_l
"""

import math
import torch
import torch.nn.functional as Fn

torch.set_default_dtype(torch.float64)

SEEDS = [0, 1, 2, 3]
REPORT = []


def log(msg):
    print(msg)
    REPORT.append(msg)


# ----------------------------------------------------------------------
# mechanism
# ----------------------------------------------------------------------

def shifted_cos(q, P):
    """u* = 1 + cos(q, P[l]), torch eps-guarded semantics. q:(d,), P:(K,d)."""
    return 1.0 + Fn.cosine_similarity(q.unsqueeze(0), P, dim=1)


def make_basket(V_per_field, F, size):
    """Multiset of items; each item = tuple of F word ids (one per field,
    field j owns global ids [j*V_per_field, (j+1)*V_per_field))."""
    items = []
    for _ in range(size):
        item = tuple(int(torch.randint(0, V_per_field, (1,)).item()) + j * V_per_field
                     for j in range(F))
        items.append(item)
    return items


def counts_from_basket(basket, V):
    n = torch.zeros(V)
    for item in basket:
        for f in item:
            n[f] += 1.0
    return n


def compose_q(w, E_words, id_row=None):
    q = w @ E_words
    if id_row is not None:
        q = q + id_row
    return q


def c1_row_shares(w, E_words, id_row, P, q):
    """Per-row shares omega_r * E[r].P[l] / (|q| |P[l]|). Returns
    (word_shares dict f->(K,), id_share (K,) or None)."""
    qn = q.norm()
    Pn = P.norm(dim=1)
    shares = {}
    for f in torch.nonzero(w).flatten().tolist():
        shares[f] = w[f] * (P @ E_words[f]) / (qn * Pn)
    id_share = (P @ id_row) / (qn * Pn) if id_row is not None else None
    return shares, id_share


# ======================================================================
# C1 exact per-row decomposition
# ======================================================================
def exact_shifted_cos(q, P):
    """1 + q.P[l]/(|q||P[l]|) with true norms, no eps clamp."""
    return 1.0 + (P @ q) / (q.norm() * P.norm(dim=1))


def check_c1():
    worst_exact = 0.0          # identity vs exact-cosine u*
    worst_torch_safe = 0.0     # identity vs torch u*, |q||P| >= eps
    worst_torch_clamp = 0.0    # identity vs torch u*, |q||P| < eps (clamped)
    n_cases = n_clamped = 0
    eps = 1e-8
    for seed in SEEDS:
        torch.manual_seed(seed)
        for F in [1, 2, 5]:
            for K in [2, 7]:
                for scaleE in [1.0, 1e6, 1e-6]:
                    for scaleP in [1.0, 1e3, 1e-3]:
                        for use_id in [False, True]:
                            d, Vpf, N = 6, 8, 3
                            V = Vpf * F
                            E_words = torch.randn(V, d) * scaleE
                            E_id = torch.randn(N, d) * scaleE
                            P = torch.randn(K, d) * scaleP
                            basket = make_basket(Vpf, F, size=7)
                            n = counts_from_basket(basket, V)
                            w = n / len(basket)
                            id_row = E_id[0] if use_id else None
                            q = compose_q(w, E_words, id_row)
                            u_torch = shifted_cos(q, P)
                            u_exact = exact_shifted_cos(q, P)
                            shares, id_share = c1_row_shares(
                                w, E_words, id_row, P, q)
                            total = torch.zeros(K)
                            for s in shares.values():
                                total = total + s
                            if id_share is not None:
                                total = total + id_share
                            err_e = (u_exact - 1.0 - total).abs().max().item()
                            worst_exact = max(worst_exact, err_e)
                            err_t = (u_torch - 1.0 - total).abs()
                            clamped = (q.norm() * P.norm(dim=1)) < eps
                            n_cases += 1
                            if clamped.any():
                                n_clamped += 1
                                worst_torch_clamp = max(
                                    worst_torch_clamp,
                                    err_t[clamped].max().item())
                            if (~clamped).any():
                                worst_torch_safe = max(
                                    worst_torch_safe,
                                    err_t[~clamped].max().item())
    log(f"C1: cases={n_cases}; identity vs EXACT cosine: max abs err = "
        f"{worst_exact:.3e}")
    log(f"C1: identity vs TORCH CosineSimilarity where |q||P_l| >= eps=1e-8: "
        f"max abs err = {worst_torch_safe:.3e}")
    log(f"C1: cases with |q||P_l| < eps (torch clamps denominator): "
        f"{n_clamped}/{n_cases} cases -> max abs err = {worst_torch_clamp:.3e} "
        f"(clamp regime: scaleE=1e-6 x scaleP=1e-3)")
    return worst_exact < 1e-12 and worst_torch_safe < 1e-12


# ======================================================================
# C2 per-purchase regrouping
# ======================================================================
def check_c2():
    worst = 0.0
    for seed in SEEDS:
        torch.manual_seed(seed)
        for F in [1, 2, 5]:
            for use_id in [False, True]:
                d, Vpf, K = 6, 8, 7
                V = Vpf * F
                E_words = torch.randn(V, d)
                E_id_row = torch.randn(d) if use_id else None
                P = torch.randn(K, d)
                basket = make_basket(Vpf, F, size=9)   # multiset, repeats OK
                n = counts_from_basket(basket, V)
                w = n / len(basket)
                q = compose_q(w, E_words, E_id_row)
                qn, Pn = q.norm(), P.norm(dim=1)
                ustar = shifted_cos(q, P)
                # per-purchase shares
                per_item_total = torch.zeros(K)
                for item in basket:
                    c_i = torch.zeros(K)
                    for f in item:
                        c_i = c_i + (1.0 / len(basket)) * (P @ E_words[f]) / (qn * Pn)
                    per_item_total = per_item_total + c_i
                # word-grouped metadata part
                shares, id_share = c1_row_shares(w, E_words, E_id_row, P, q)
                word_total = torch.zeros(K)
                for s in shares.values():
                    word_total = word_total + s
                meta_target = ustar - 1.0 - (id_share if id_share is not None
                                             else torch.zeros(K))
                e1 = (per_item_total - word_total).abs().max().item()
                e2 = (per_item_total - meta_target).abs().max().item()
                worst = max(worst, e1, e2)
    log(f"C2: max|purchase-grouped - word-grouped| and vs (u*-1-ID): {worst:.3e}")
    return worst < 1e-12


# ======================================================================
# C3 degenerate inputs
# ======================================================================
def check_c3():
    ok = True
    # (a) empty basket, no ID
    torch.manual_seed(0)
    d, V, K, M = 6, 8, 5, 4
    E_words = torch.randn(V, d)
    P = torch.randn(K, d)
    T = torch.randn(M, K)
    w = torch.zeros(V)
    q = compose_q(w, E_words, None)
    ustar = shifted_cos(q, P)
    dev_ones = (ustar - 1.0).abs().max().item()
    S = T @ ustar
    dev_S = (S - T.sum(dim=1)).abs().max().item()
    log(f"C3a: q=0 -> max|u* - 1|={dev_ones:.3e}, "
        f"max|S - sum_l t_il|={dev_S:.3e} (torch eps-guard)")
    ok &= dev_ones == 0.0 and dev_S < 1e-14

    # (b) identical normalized weights -> bit-identical u*
    bit_identical = True
    for seed in SEEDS:
        torch.manual_seed(seed)
        F, Vpf = 2, 8
        V2 = Vpf * F
        E2 = torch.randn(V2, d)
        P2 = torch.randn(K, d)
        base = make_basket(Vpf, F, size=3)
        basketA = base                       # each item once
        basketB = base + base                # each item twice -> same w
        wA = counts_from_basket(basketA, V2) / len(basketA)
        wB = counts_from_basket(basketB, V2) / len(basketB)
        uA = shifted_cos(compose_q(wA, E2, None), P2)
        uB = shifted_cos(compose_q(wB, E2, None), P2)
        bit_identical &= torch.equal(uA, uB)
    log(f"C3b: identical normalized weights -> bit-identical u*: {bit_identical}")
    ok &= bit_identical

    # (c) different baskets -> different u*
    min_over_seeds = math.inf
    for seed in SEEDS:
        torch.manual_seed(seed + 100)
        F, Vpf = 2, 8
        V2 = Vpf * F
        E2 = torch.randn(V2, d)
        P2 = torch.randn(K, d)
        b1 = make_basket(Vpf, F, size=5)
        b2 = make_basket(Vpf, F, size=5)
        w1 = counts_from_basket(b1, V2) / len(b1)
        w2 = counts_from_basket(b2, V2) / len(b2)
        u1 = shifted_cos(compose_q(w1, E2, None), P2)
        u2 = shifted_cos(compose_q(w2, E2, None), P2)
        min_over_seeds = min(min_over_seeds, (u1 - u2).abs().max().item())
    log(f"C3c: min over seeds of max|Delta u*| (different baskets) = "
        f"{min_over_seeds:.3e}")
    ok &= min_over_seeds > 1e-6
    return ok


# ======================================================================
# C4 score decomposition + baseline non-inertness
# ======================================================================
def check_c4():
    ok = True
    worst = 0.0
    for seed in SEEDS:
        torch.manual_seed(seed)
        d, Vpf, F, K, M = 6, 8, 2, 7, 20
        V = Vpf * F
        E_words = torch.randn(V, d)
        P = torch.randn(K, d)
        T = torch.randn(M, K)
        basket = make_basket(Vpf, F, size=6)
        w = counts_from_basket(basket, V) / len(basket)
        q = compose_q(w, E_words, None)
        ustar = shifted_cos(q, P)
        S = T @ ustar
        B = T.sum(dim=1)
        pers = T @ (ustar - 1.0)
        worst = max(worst, (S - (B + pers)).abs().max().item())
    log(f"C4a: max|S - (B + personalized)| = {worst:.3e}")
    ok &= worst < 1e-12

    # (b) two items, equal personalized part, different B
    torch.manual_seed(7)
    d, K = 6, 7
    P = torch.randn(K, d)
    Vpf, F = 8, 2
    E_words = torch.randn(Vpf * F, d)
    basket = make_basket(Vpf, F, size=6)
    w = counts_from_basket(basket, Vpf * F) / len(basket)
    ustar = shifted_cos(compose_q(w, E_words, None), P)
    g = ustar - 1.0
    t1 = torch.randn(K)
    ones = torch.ones(K)
    v = ones - (ones @ g) / (g @ g) * g          # v.g = 0, sum(v) != 0
    t2 = t1 + v
    pers1, pers2 = (t1 @ g).item(), (t2 @ g).item()
    B1, B2 = t1.sum().item(), t2.sum().item()
    S1, S2 = (t1 @ ustar).item(), (t2 @ ustar).item()
    order_by_B = (S2 > S1) == (B2 > B1)
    log(f"C4b: personalized parts {pers1:.15f} vs {pers2:.15f} "
        f"(diff {abs(pers1-pers2):.3e}); B: {B1:.6f} vs {B2:.6f}; "
        f"scores {S1:.6f} vs {S2:.6f}; order decided by B: {order_by_B}")
    torch.manual_seed(11)
    Bvar = torch.randn(5000, K).sum(dim=1).var().item()
    log(f"C4b: empirical var of B(t) over 5000 random items (randn init, K={K}): "
        f"{Bvar:.4f} (nonzero)")
    ok &= abs(pers1 - pers2) < 1e-12 and order_by_B and Bvar > 0.1
    return ok


# ======================================================================
# C5 keystone reduction (V=0 words, ID enabled == host)
# ======================================================================
def check_c5():
    all_bit = True
    for seed in SEEDS:
        torch.manual_seed(seed)
        d, K, N, M = 6, 5, 4, 10
        P = torch.randn(K, d)
        T = torch.randn(M, K)
        U_host = torch.randn(N, d)
        E_id = U_host.clone()                    # copy host rows into ID rows
        V = 0
        E_words = torch.zeros(V, d)
        for u in range(N):
            w = torch.zeros(V)
            q_model = compose_q(w, E_words, E_id[u])   # 0-vector + ID row
            q_host = U_host[u]
            u_model = shifted_cos(q_model, P)
            u_host = shifted_cos(q_host, P)
            all_bit &= torch.equal(u_model, u_host)
            all_bit &= torch.equal(T @ u_model, T @ u_host)
    log(f"C5: V=0 + ID row copied from host -> u* and S bit-identical: {all_bit}")
    return all_bit


# ======================================================================
# C6 joint-normalization coupling
# ======================================================================
def check_c6():
    worst_common, worst_ratio = 0.0, 0.0
    for seed in SEEDS:
        torch.manual_seed(seed)
        d, Vpf, F, K = 6, 8, 2, 7
        V = Vpf * F
        E_words = torch.randn(V, d)
        P = torch.randn(K, d)
        basket = make_basket(Vpf, F, size=6)
        w = counts_from_basket(basket, V) / len(basket)
        present = torch.nonzero(w).flatten().tolist()
        f0 = present[0]
        for alpha in [10.0, 0.1, -3.0]:
            E_new = E_words.clone()
            E_new[f0] = alpha * E_words[f0]
            q_old = compose_q(w, E_words, None)
            q_new = compose_q(w, E_new, None)
            sh_old, _ = c1_row_shares(w, E_words, None, P, q_old)
            sh_new, _ = c1_row_shares(w, E_new, None, P, q_new)
            factor = (q_old.norm() / q_new.norm()).item()
            others = [f for f in present if f != f0]
            # absolute rescale by common factor |q_old|/|q_new|
            for f in others:
                dev = (sh_new[f] - factor * sh_old[f]).abs().max().item()
                worst_common = max(worst_common, dev)
            # pairwise ratios unchanged (guard tiny denominators)
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    a_old, b_old = sh_old[others[i]], sh_old[others[j]]
                    a_new, b_new = sh_new[others[i]], sh_new[others[j]]
                    mask = b_old.abs() > 1e-8
                    if mask.any():
                        r_old = (a_old / b_old)[mask]
                        r_new = (a_new / b_new)[mask]
                        worst_ratio = max(worst_ratio,
                                          (r_old - r_new).abs().max().item())
    log(f"C6: alpha in {{10, 0.1, -3}} on one word row: "
        f"max|share_new - (|q_old|/|q_new|)*share_old| (other rows) = "
        f"{worst_common:.3e}; max|pairwise ratio change| = {worst_ratio:.3e}")
    return worst_common < 1e-12 and worst_ratio < 1e-10


# ======================================================================
# C7 item-vector gauge  (needs K > d+1)
# ======================================================================
def _c7_setup(seed):
    torch.manual_seed(seed)
    d, K, N, M = 3, 6, 12, 8         # K > d+1
    P = torch.randn(K, d)
    Phat = P / P.norm(dim=1, keepdim=True)
    # w in null space of C = [1^T ; Phat^T]  ((d+1) x K)
    C = torch.cat([torch.ones(1, K), Phat.T], dim=0)
    _, s, Vt = torch.linalg.svd(C)
    null = Vt[C.shape[0]:]          # rows spanning null space (dim K-(d+1)=2)
    wvec = null[0]
    wvec = wvec / wvec.norm()
    # users: random baskets -> u* matrix
    Vpf, F = 8, 2
    E_words = torch.randn(Vpf * F, d)
    U = torch.zeros(N, K)
    for u in range(N):
        basket = make_basket(Vpf, F, size=5)
        wgt = counts_from_basket(basket, Vpf * F) / len(basket)
        U[u] = shifted_cos(compose_q(wgt, E_words, None), P)
    T = torch.randn(M, K)
    return d, K, N, M, P, Phat, wvec, U, T


def _softmax_loss(U, T, targets):
    scores = U @ T.T
    return Fn.cross_entropy(scores, targets)


def check_c7():
    ok_a = ok_b = ok_d = True
    worst_a = worst_b = 0.0
    bitwise_c_all = True
    worst_c_drift = 0.0
    worst_d = 0.0
    for seed in SEEDS:
        d, K, N, M, P, Phat, wvec, U, T = _c7_setup(seed)
        targets = torch.randint(0, M, (N,))
        # sanity: w satisfies constraints
        # (a) t_i + w changes no score
        i0 = 0
        T2 = T.clone()
        T2[i0] = T2[i0] + wvec
        dS = ((U @ T.T) - (U @ T2.T)).abs().max().item()
        worst_a = max(worst_a, dS)
        # (b) autograd grad of softmax loss has zero component along w
        Tg = T.clone().requires_grad_(True)
        loss = _softmax_loss(U, Tg, targets)
        grad, = torch.autograd.grad(loss, Tg)
        for i in range(M):
            gn = grad[i].norm().item()
            comp = abs((grad[i] @ wvec).item()) / max(gn, 1e-300)
            worst_b = max(worst_b, comp)
        # (c) plain SGD: component along w bitwise frozen?
        eta = 0.05
        Tc = T.clone()
        comp0 = ((Tc[i0] @ wvec) / (wvec @ wvec)).item()
        comps = [comp0]
        first_flip = None
        for step in range(20):
            Tg = Tc.clone().requires_grad_(True)
            loss = _softmax_loss(U, Tg, targets)
            g, = torch.autograd.grad(loss, Tg)
            Tc = Tc - eta * g
            c = ((Tc[i0] @ wvec) / (wvec @ wvec)).item()
            comps.append(c)
            if first_flip is None and c != comp0:
                first_flip = step + 1
        bitwise = all(c == comp0 for c in comps)
        bitwise_c_all &= bitwise
        if not bitwise:
            log(f"C7c seed {seed}: comp0={comp0!r}, first bit flip at step "
                f"{first_flip}, final={comps[-1]!r}")
        worst_c_drift = max(worst_c_drift,
                            max(abs(c - comp0) for c in comps))
        # (d) with L2: component decays by exactly (1-2*eta*lam) per step
        lam = 0.01
        Td = T.clone()
        prev = ((Td[i0] @ wvec) / (wvec @ wvec)).item()
        for _ in range(20):
            Tg = Td.clone().requires_grad_(True)
            loss = _softmax_loss(U, Tg, targets) + lam * (Tg ** 2).sum()
            g, = torch.autograd.grad(loss, Tg)
            Td = Td - eta * g
            cur = ((Td[i0] @ wvec) / (wvec @ wvec)).item()
            worst_d = max(worst_d, abs(cur / prev - (1 - 2 * eta * lam)))
            prev = cur
    log(f"C7a: max|Delta S| after t_i -> t_i + w : {worst_a:.3e}")
    log(f"C7b: max over items of |grad . w|/(|grad||w|) (softmax loss): "
        f"{worst_b:.3e}")
    log(f"C7c: SGD 20 steps, component of t_i along w — bitwise frozen: "
        f"{bitwise_c_all}; max abs drift observed: {worst_c_drift:.3e}")
    log(f"C7d: L2 lam=0.01 eta=0.05 — max|comp ratio - (1-2*eta*lam)| "
        f"per step: {worst_d:.3e}")
    ok_a = worst_a < 1e-12
    ok_b = worst_b < 1e-12
    ok_d = worst_d < 1e-10
    return ok_a, ok_b, bitwise_c_all, worst_c_drift, ok_d


# ======================================================================
# C8 normalization scale invariance / where it bites
# ======================================================================
def check_c8():
    worst_noid = 0.0
    for seed in SEEDS:
        torch.manual_seed(seed)
        d, Vpf, F, K = 6, 8, 2, 7
        V = Vpf * F
        E_words = torch.randn(V, d)
        P = torch.randn(K, d)
        basket = make_basket(Vpf, F, size=6)
        n = counts_from_basket(basket, V)
        w = n / len(basket)
        u_ref = shifted_cos(compose_q(w, E_words, None), P)
        for alpha in [2.0, 0.5, 3.7, 1e6, 1e-6, float(len(basket))]:
            u_s = shifted_cos(compose_q(alpha * w, E_words, None), P)
            worst_noid = max(worst_noid, (u_ref - u_s).abs().max().item())
        # explicit sum-vs-mean
        u_sum = shifted_cos(compose_q(n, E_words, None), P)
        worst_noid = max(worst_noid, (u_ref - u_sum).abs().max().item())
    log(f"C8a: no-ID, w -> alpha*w (alpha in {{2, .5, 3.7, 1e6, 1e-6, |H|}}) "
        f"and sum-vs-mean: max|Delta u*| = {worst_noid:.3e}")

    # (b) ID enabled: sum vs mean genuinely differ
    torch.manual_seed(0)
    d, Vpf, F, K = 6, 8, 2, 7
    V = Vpf * F
    E_words = torch.randn(V, d)
    E_id_row = torch.randn(d)
    P = torch.randn(K, d)
    basket = make_basket(Vpf, F, size=20)     # heavy basket
    n = counts_from_basket(basket, V)
    u_mean = shifted_cos(compose_q(n / len(basket), E_words, E_id_row), P)
    u_sum = shifted_cos(compose_q(n, E_words, E_id_row), P)
    diff = (u_mean - u_sum).abs().max().item()
    log(f"C8b: ID enabled, |H|=20: sum vs mean weights -> max|Delta u*| = "
        f"{diff:.4f} (u*_mean={u_mean.tolist()}, u*_sum={u_sum.tolist()})")
    return worst_noid < 1e-12, diff > 1e-3


# ======================================================================
# C9 mean-regression of heavy baskets (trend)
# ======================================================================
def check_c9():
    sizes = [3, 10, 30, 100, 300]
    all_mono = True
    for seed in SEEDS[:3]:
        torch.manual_seed(seed)
        V, d, n_users = 40, 8, 2000
        E_words = torch.randn(V, d)
        p = torch.rand(V)
        p = p / p.sum()
        mu = p @ E_words
        mu_n = mu / mu.norm()
        means = []
        for H in sizes:
            counts = torch.distributions.Multinomial(
                H, probs=p).sample((n_users,))          # (n_users, V)
            w = counts / H                                # mean normalization
            Q = w @ E_words                               # (n_users, d)
            cosang = (Q @ mu_n) / Q.norm(dim=1).clamp_min(1e-30)
            ang = torch.rad2deg(torch.acos(cosang.clamp(-1, 1)))
            means.append(ang.mean().item())
        mono = all(means[i] > means[i + 1] for i in range(len(means) - 1))
        all_mono &= mono
        log(f"C9 seed {seed}: mean angle(q_u, mu) deg for |H|={sizes}: "
            f"{[f'{m:.2f}' for m in means]}  monotone decreasing: {mono}")
    return all_mono


# ======================================================================
if __name__ == "__main__":
    log("=" * 70)
    r1 = check_c1()
    r2 = check_c2()
    r3 = check_c3()
    r4 = check_c4()
    r5 = check_c5()
    r6 = check_c6()
    r7a, r7b, r7c_bit, r7c_drift, r7d = check_c7()
    r8a, r8b = check_c8()
    r9 = check_c9()
    log("=" * 70)
    log(f"C1 exact per-row decomposition       : {'PASS' if r1 else 'FAIL'}")
    log(f"C2 per-purchase regrouping           : {'PASS' if r2 else 'FAIL'}")
    log(f"C3 degenerate inputs                 : {'PASS' if r3 else 'FAIL'}")
    log(f"C4 score decomposition / baseline    : {'PASS' if r4 else 'FAIL'}")
    log(f"C5 keystone reduction                : {'PASS' if r5 else 'FAIL'}")
    log(f"C6 joint-normalization coupling      : {'PASS' if r6 else 'FAIL'}")
    log(f"C7a gauge score invariance           : {'PASS' if r7a else 'FAIL'}")
    log(f"C7b gauge gradient orthogonality     : {'PASS' if r7b else 'FAIL'}")
    log(f"C7c SGD component bitwise frozen     : "
        f"{'PASS' if r7c_bit else f'FAIL (drift {r7c_drift:.3e})'}")
    log(f"C7d L2 exact (1-2*eta*lam) decay     : {'PASS' if r7d else 'FAIL'}")
    log(f"C8a no-ID scale invariance           : {'PASS' if r8a else 'FAIL'}")
    log(f"C8b ID sum-vs-mean differs           : {'PASS' if r8b else 'FAIL'}")
    log(f"C9 mean-regression trend             : {'PASS' if r9 else 'FAIL'}")
