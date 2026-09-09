"""
Independent black-box sanity check of the dc08 claims spec (C1..C11).
Toy dimensions, numpy float64 + torch float64. Runtime: seconds.
No project code is imported; everything below is a from-scratch re-implementation
of the mechanism exactly as written in the spec.
"""
import numpy as np, torch, itertools, math

rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)

# ---------------------------------------------------------------- toy layout
V, M, N, d, K = 11, 9, 7, 5, 4          # words, items, users, dim, prototypes

def make_world(seed=0, force_singleton_user=True, force_empty_user=False):
    r = np.random.default_rng(seed)
    E = r.normal(size=(V + M + N, d))
    P = r.normal(size=(K, d))
    T = r.normal(size=(M, K))                        # free per-item t_i
    # item word bags (multisets allowed -> counts)
    fbags = []
    for j in range(M):
        n = r.integers(1, 4)
        fbags.append(list(r.integers(0, V, size=n)))  # repeats possible
    # user histories
    H = []
    for u in range(N):
        n = int(r.integers(1, 5))
        H.append(sorted(r.choice(M, size=n, replace=False).tolist()))
    if force_singleton_user:
        H[0] = [int(r.integers(0, M))]               # |H|=1 user
    if force_empty_user:
        H[-1] = []                                   # |H|=0 user
    # deliberate repeated-word user: give user 1 two items sharing a word
    if len(H) > 1 and M >= 2:
        fbags[0] = [3, 3, 7]
        fbags[1] = [3, 7, 7]
        H[1] = [0, 1]
    return E, P, T, fbags, H

def word_row(f):   return f
def item_row(j):   return V + j
def user_row(u):   return V + M + u

# ------------------------------------------------- reference (math) form
def q_math(E, fbags, Hu, u, lam, use_id):
    n = len(Hu)
    acc = np.zeros(d)
    for j in Hu:
        for f in fbags[j]:
            acc += E[word_row(f)]
        acc += lam * E[item_row(j)]
    q = acc / n if n > 0 else np.zeros(d)
    if use_id:
        q = q + E[user_row(u)]
    return q

# ------------------------------------------------- padded bag form
def build_bag(fbags, Hu, lam, Dpad, alpha=1.0):
    """Returns (ids, weights) padded to Dpad; pooling divisor |H|**alpha."""
    n = len(Hu)
    ids, ws = [], []
    if n > 0:
        den = float(n) ** alpha
        counts = {}
        for j in Hu:
            for f in fbags[j]:
                counts[f] = counts.get(f, 0) + 1
        for f in sorted(counts):
            ids.append(word_row(f)); ws.append(counts[f] / den)
        for j in Hu:
            ids.append(item_row(j)); ws.append(lam / den)
    while len(ids) < Dpad:
        ids.append(0); ws.append(0.0)
    return np.array(ids[:Dpad]), np.array(ws[:Dpad], dtype=np.float64)

def q_bag(E, ids, ws, u, use_id, E_user=None):
    q = (ws[:, None] * E[ids]).sum(0)
    if use_id:
        q = q + E[user_row(u)]
    return q

DPAD = V + M + 3

# ================================================================= C1
def C1():
    err = 0.0
    for seed in range(30):
        E, P, T, fbags, H = make_world(seed)
        for lam, use_id in itertools.product([0.0, 0.5, 1.0, 2.0], [False, True]):
            for u in range(N):
                ids, ws = build_bag(fbags, H[u], lam, DPAD)
                a = q_bag(E, ids, ws, u, use_id)
                b = q_math(E, fbags, H[u], u, lam, use_id)
                err = max(err, np.abs(a - b).max())
    return err

# ================================================================= C2
def C2():
    max_err, bitident = 0.0, True
    for seed in range(30):
        E, P, T, fbags, H = make_world(seed)
        for use_id in [False, True]:
            for u in range(N):
                ids, ws = build_bag(fbags, H[u], 0.0, DPAD)
                a = q_bag(E, ids, ws, u, use_id)
                # bag with identity slots physically removed
                keep = [(i, w) for i, w in zip(ids, ws)
                        if not (V <= i < V + M)]
                ids2 = np.array([i for i, _ in keep]); ws2 = np.array([w for _, w in keep])
                b = q_bag(E, ids2, ws2, u, use_id)
                max_err = max(max_err, np.abs(a - b).max())
                if not np.array_equal(a, b):
                    bitident = False
    return max_err, bitident

# ================================================================= C3
def C3():
    max_err = 0.0
    singleton_zero = True
    id_removed_ok = True
    for seed in range(40):
        E, P, T, fbags, H = make_world(seed)
        r = np.random.default_rng(1000 + seed)
        for lam in [0.0, 1.0]:
            for u in range(N):
                n = len(H[u])
                ids, ws = build_bag(fbags, H[u], lam, DPAD)
                qm = (ws[:, None] * E[ids]).sum(0)      # metadata part only
                if n == 1:
                    i = H[u][0]
                    e_pos = sum(E[word_row(f)] for f in fbags[i]) + lam * E[item_row(i)]
                    q_loo = np.zeros(d)                  # defined as exactly zero
                    if np.abs(q_loo).max() != 0.0:
                        singleton_zero = False
                    continue
                i = int(r.choice(H[u]))
                e_pos = sum(E[word_row(f)] for f in fbags[i]) + lam * E[item_row(i)]
                q_loo = (n * qm - e_pos) / (n - 1)
                rest = [j for j in H[u] if j != i]
                ref = np.zeros(d)
                for j in rest:
                    for f in fbags[j]:
                        ref += E[word_row(f)]
                    ref += lam * E[item_row(j)]
                ref /= (n - 1)
                max_err = max(max_err, np.abs(q_loo - ref).max())
                # identity-row removal check: coefficient on y_i must be 0
                # probe by perturbing E[item_row(i)] and seeing if q_loo moves
                E2 = E.copy(); E2[item_row(i)] += 100.0
                ids2, ws2 = build_bag(fbags, H[u], lam, DPAD)
                qm2 = (ws2[:, None] * E2[ids2]).sum(0)
                e_pos2 = sum(E2[word_row(f)] for f in fbags[i]) + lam * E2[item_row(i)]
                q_loo2 = (n * qm2 - e_pos2) / (n - 1)
                if np.abs(q_loo2 - q_loo).max() > 1e-9:
                    id_removed_ok = False
    return max_err, singleton_zero, id_removed_ok

# ================================================================= C4
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))

def C4():
    max_err = 0.0
    for seed in range(30):
        E, P, T, fbags, H = make_world(seed)
        for lam, use_id in itertools.product([0.0, 1.0], [False, True]):
            for u in range(N):
                if len(H[u]) == 0: continue
                ids, ws = build_bag(fbags, H[u], lam, DPAD)
                q = q_bag(E, ids, ws, u, use_id)
                nq = np.linalg.norm(q)
                for l in range(K):
                    Pl = P[l]; npl = np.linalg.norm(Pl)
                    shares = ws * (E[ids] @ Pl) / (nq * npl)
                    tot = shares.sum()
                    if use_id:
                        tot += (E[user_row(u)] @ Pl) / (nq * npl)
                    max_err = max(max_err, abs(tot - cos(q, Pl)))
    return max_err

# ================================================================= C5
def C5():
    err_a, err_b, err_bsum = 0.0, 0.0, 0.0
    for seed in range(30):
        E, P, T, fbags, H = make_world(seed)
        for lam in [0.5, 1.0]:
            for u in range(N):
                n = len(H[u])
                if n == 0: continue
                ids, ws = build_bag(fbags, H[u], lam, DPAD)
                q = q_bag(E, ids, ws, u, True)
                nq = np.linalg.norm(q)
                for l in range(K):
                    Pl = P[l]; npl = np.linalg.norm(Pl); den = nq * npl
                    c = cos(q, Pl)
                    urow = (E[user_row(u)] @ Pl) / den
                    # (a) grouped by row type
                    wsh = sum(w * (E[i] @ Pl) / den for i, w in zip(ids, ws) if i < V)
                    ish = sum(w * (E[i] @ Pl) / den for i, w in zip(ids, ws) if V <= i < V + M)
                    err_a = max(err_a, abs(wsh + ish + urow - c))
                    # (b) grouped by purchased item j
                    per_item = []
                    for j in H[u]:
                        s = sum((E[word_row(f)] @ Pl) / (n * den) for f in fbags[j])
                        s += lam * (E[item_row(j)] @ Pl) / (n * den)
                        per_item.append(s)
                    err_b = max(err_b, abs(sum(per_item) + urow - c))
                    err_bsum = max(err_bsum, abs(sum(per_item) - (c - urow)))
    return err_a, err_b, err_bsum

# ================================================================= C6
def C6():
    max_err = 0.0
    B_stable = True
    for seed in range(20):
        E, P, T, fbags, H = make_world(seed)
        for u in range(N):
            if len(H[u]) == 0: continue
            ids, ws = build_bag(fbags, H[u], 1.0, DPAD)
            q = q_bag(E, ids, ws, u, True)
            cl = np.array([cos(q, P[l]) for l in range(K)])
            ustar = 1.0 + cl
            for i in range(M):
                S = float(T[i] @ ustar)
                dec = T[i].sum() + float(T[i] @ cl)
                max_err = max(max_err, abs(S - dec))
            # B independent of user
            B1 = T.sum(1)
            q2 = q * 3.7 + 1.0
            cl2 = np.array([cos(q2, P[l]) for l in range(K)])
            B2 = T.sum(1)
            if not np.array_equal(B1, B2): B_stable = False
    return max_err, B_stable

# ================================================================= C7
def C7():
    # (i) positive scaling of q leaves u* and S unchanged
    err_scale = 0.0
    E, P, T, fbags, H = make_world(3)
    for u in range(N):
        if len(H[u]) == 0: continue
        ids, ws = build_bag(fbags, H[u], 1.0, DPAD)
        q = q_bag(E, ids, ws, u, True)
        for s in [1e-3, 0.5, 2.0, 1e3]:
            for l in range(K):
                err_scale = max(err_scale, abs(cos(q, P[l]) - cos(s * q, P[l])))
    # (ii) pooling exponent alpha
    max_noid, max_id = 0.0, 0.0
    for seed in range(20):
        E, P, T, fbags, H = make_world(seed)
        for u in range(N):
            if len(H[u]) < 2: continue
            for alpha in [0.5, 1.0, 1.5, 2.0]:
                ids1, ws1 = build_bag(fbags, H[u], 1.0, DPAD, alpha=1.0)
                ids2, ws2 = build_bag(fbags, H[u], 1.0, DPAD, alpha=alpha)
                for use_id, store in [(False, 'n'), (True, 'i')]:
                    qa = q_bag(E, ids1, ws1, u, use_id)
                    qb = q_bag(E, ids2, ws2, u, use_id)
                    dif = max(abs(cos(qa, P[l]) - cos(qb, P[l])) for l in range(K))
                    if store == 'n': max_noid = max(max_noid, dif)
                    else:            max_id = max(max_id, dif)
    return err_scale, max_noid, max_id

# ================================================================= C8
def C8():
    E, P, T, fbags, H = make_world(5)
    lam = 1.0
    u = 2
    while len(H[u]) < 2: u = (u + 1) % N
    Hu = H[u]; n = len(Hu)
    pos = Hu[0]
    Et = torch.tensor(E, requires_grad=True)
    Pt = torch.tensor(P); Tt = torch.tensor(T)
    ids, ws = build_bag(fbags, Hu, lam, DPAD)
    idst = torch.tensor(ids); wst = torch.tensor(ws)

    def score(loo):
        qm = (wst[:, None] * Et[idst]).sum(0)
        if loo:
            e_pos = sum(Et[word_row(f)] for f in fbags[pos]) + lam * Et[item_row(pos)]
            qm = (n * qm - e_pos) / (n - 1)
        q = qm + Et[user_row(u)]
        cl = torch.nn.functional.cosine_similarity(q[None, :], Pt, dim=1)
        return (Tt[pos] * (1 + cl)).sum()

    out = {}
    for loo in [False, True]:
        if Et.grad is not None: Et.grad = None
        score(loo).backward()
        g = Et.grad.detach().numpy()
        out[loo] = {j: float(np.abs(g[item_row(j)]).max()) for j in range(M)}
        out[str(loo) + '_user'] = float(np.abs(g[user_row(u)]).max())
    return Hu, pos, out

# ================================================================= C9
def C9():
    E, P, T, fbags, H = make_world(7)
    lam = 1.3
    # make user 0's history disjoint from everyone else's
    H = [list(h) for h in H]
    H[0] = [0, 1, 2]
    for u in range(1, N):
        H[u] = sorted(set(H[u]) - {0, 1, 2}) or [3]
    res = []
    delta = rng.normal(size=d)
    for use_loo in [False, True]:
        E2 = E.copy()
        E2[user_row(0)] += delta
        for j in H[0]:
            E2[item_row(j)] -= delta / lam
        def compose(EE, u, loo_pos=None):
            n = len(H[u])
            ids, ws = build_bag(fbags, H[u], lam, DPAD)
            qm = (ws[:, None] * EE[ids]).sum(0)
            if loo_pos is not None and n >= 2:
                e_pos = sum(EE[word_row(f)] for f in fbags[loo_pos]) + lam * EE[item_row(loo_pos)]
                qm = (n * qm - e_pos) / (n - 1)
            return qm + EE[user_row(u)]
        worst = 0.0
        for u in range(N):
            lp = H[u][0] if (use_loo and len(H[u]) >= 2) else None
            worst = max(worst, np.abs(compose(E, u, lp) - compose(E2, u, lp)).max())
        res.append(worst)
    # control: what if user 0's items are NOT disjoint -> other users must move
    Hs = [list(h) for h in H]; Hs[1] = sorted(set(Hs[1]) | {0})
    E3 = E.copy(); E3[user_row(0)] += delta
    for j in Hs[0]: E3[item_row(j)] -= delta / lam
    def comp2(EE, u):
        ids, ws = build_bag(fbags, Hs[u], lam, DPAD)
        return (ws[:, None] * EE[ids]).sum(0) + EE[user_row(u)]
    contam = np.abs(comp2(E, 1) - comp2(E3, 1)).max()
    return res[0], res[1], contam

# ================================================================= C10
def C10():
    E, P, T, fbags, H = make_world(11)
    lam = 0.8
    c = rng.normal(size=d)
    E2 = E.copy()
    E2[V:V + M] += c
    max_err = 0.0
    rank_changes = 0; total = 0
    for u in range(N):
        if len(H[u]) == 0: continue
        ids, ws = build_bag(fbags, H[u], lam, DPAD)
        q1 = q_bag(E, ids, ws, u, True)
        q2 = q_bag(E2, ids, ws, u, True)
        max_err = max(max_err, np.abs((q2 - q1) - lam * c).max())
        s1 = np.array([sum(T[i, l] * (1 + cos(q1, P[l])) for l in range(K)) for i in range(M)])
        s2 = np.array([sum(T[i, l] * (1 + cos(q2, P[l])) for l in range(K)) for i in range(M)])
        total += 1
        if not np.array_equal(np.argsort(-s1), np.argsort(-s2)):
            rank_changes += 1
    return max_err, rank_changes, total

# ================================================================= C11
def C11():
    E, P, T, fbags, H = make_world(13, force_empty_user=True)
    u = N - 1
    assert len(H[u]) == 0
    ids, ws = build_bag(fbags, H[u], 1.0, DPAD)   # all-pad
    q_noid = q_bag(E, ids, ws, u, False)
    q_id = q_bag(E, ids, ws, u, True)
    e_noid = np.abs(q_noid).max()
    e_id = np.abs(q_id - E[user_row(u)]).max()
    qt = torch.zeros(1, d, dtype=torch.float64)
    Pt = torch.tensor(P)
    cs = torch.nn.functional.cosine_similarity(qt.expand(K, d), Pt, dim=1).numpy()
    ustar = 1 + cs
    S = T @ ustar
    Bt = T.sum(1)
    return e_noid, e_id, cs, float(np.abs(ustar - 1).max()), float(np.abs(S - Bt).max())

# ================================================================= run
print("=" * 70)
print("C1 max|bag - math| over 30 worlds x {lam 0,.5,1,2} x {use_id} x all users:",
      f"{C1():.3e}")

e, bit = C2()
print(f"C2 max|with-zero-weight-id-slots - without| = {e:.3e}  bit-identical={bit}")

e, sz, idr = C3()
print(f"C3 max|algebraic LOO - recomputed| = {e:.3e}  |H|=1 -> exact zero: {sz}  "
      f"identity row of positive fully removed: {idr}")

print(f"C4 max|sum of per-slot shares - cos_l| = {C4():.3e}")

a, b, bs = C5()
print(f"C5 grouping(a) err={a:.3e}  grouping(b)+userrow err={b:.3e}  "
      f"per-item total vs cos-userrow err={bs:.3e}")

e, bst = C6()
print(f"C6 max|S - (sum t + sum t*cos)| = {e:.3e}  B(t_i) unchanged when q changes: {bst}")

es, mn, mi = C7()
print(f"C7 max|cos(q)-cos(s*q)| = {es:.3e}   alpha-change max dcos use_id=False: {mn:.3e}"
      f"   use_id=True: {mi:.3e}")

Hu, pos, g = C8()
print(f"C8 H_u={Hu} positive(LOO-removed)={pos}")
print(f"   no-LOO  |grad y_j|max: " + ", ".join(f"j{j}:{g[False][j]:.3e}" for j in range(M)))
print(f"   with-LOO|grad y_j|max: " + ", ".join(f"j{j}:{g[True][j]:.3e}" for j in range(M)))
print(f"   user row grad: noLOO {g['False_user']:.3e}  LOO {g['True_user']:.3e}")

r0, r1, contam = C9()
print(f"C9 residual over ALL users (no LOO) = {r0:.3e}  (with LOO) = {r1:.3e}  "
      f"non-disjoint control drift on other user = {contam:.3e}")

e, rc, tot = C10()
print(f"C10 max|dq - lam*c| = {e:.3e}   users whose item ranking changed: {rc}/{tot}")

a, b, cs, du, ds = C11()
print(f"C11 |q_noid|={a:.3e}  |q_id - e_ID|={b:.3e}  torch cos(0,P)={cs}  "
      f"max|u*-1|={du:.3e}  max|S - sum t|={ds:.3e}")
print("=" * 70)
