#!/usr/bin/env python
"""check_claims.py -- independent black-box toy verification of claims C1-C14 in
claims_spec.md (dc09 "merged model"). The mechanism is implemented here from the
spec alone (no project code was read). CPU, float64 default, toy sizes, seconds.

Run (from this directory):
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate protomf && python check_claims.py

Writes report.md next to this file.
"""
import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
EPS = 1e-8
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4, 5]
TOL_MP = 1e-11          # "machine precision" (relative) threshold used for verdicts
TOL_FLOAT = 1e-9        # "float tolerance" for cosine invariances

CONFIGS = [
    # K_u > d+1 and K_t > d+1 ; fixed word-set size ; user 0 forced to |H|=1
    dict(name="A", V=6, M=8, N=6, d=3, K_u=8, K_t=6, F=3, variable=False, bs=(1, 5), force_single=True),
    # variable word-set sizes, bigger K
    dict(name="B", V=8, M=10, N=7, d=5, K_u=12, K_t=9, F=2, variable=True, bs=(1, 6), force_single=False),
    # K_u <= d+1 (4 <= 5) and K_t == d+1 (5) : no gauge null space
    dict(name="C", V=5, M=8, N=6, d=4, K_u=4, K_t=5, F=3, variable=True, bs=(2, 4), force_single=False),
    # many words per item, small d, large K_u
    dict(name="D", V=7, M=9, N=8, d=3, K_u=10, K_t=7, F=4, variable=False, bs=(1, 3), force_single=True),
]
# parameter scale sets (randn * scale) -- adversarial mixtures incl. 1e-3 and 1e3; randn gives negatives
SCALES = [
    dict(),
    dict(E_I=1e-3, E_U=1e3),
    dict(E_I=1e3, E_U=1e-3, P_u=1e-3, P_t=1e3),
    dict(W_u=1e3, W_t=1e-3),
    dict(E_I=1e-3, E_U=1e-3, P_u=1e-3, P_t=1e-3, W_u=1e-3, W_t=1e-3),
]

REPORT = []


def log(s=""):
    print(s)
    REPORT.append(s)


def fmt(x):
    return f"{float(x):.3e}"


def maxabs(a, b):
    return (a - b).abs().max().item() if a.numel() else 0.0


def maxrel(a, b):
    den = max(a.abs().max().item(), b.abs().max().item(), 1e-300)
    return maxabs(a, b) / den


# ----------------------------------------------------------------------------- data
class Data:
    """Items with word sets f_i (subsets of 0..V-1), users with baskets H_u (sets of items)."""

    def __init__(self, V, M, N, F=3, variable=False, bs=(1, 5), seed=0, force_single=False, **_):
        rng = np.random.default_rng(1000 + seed)
        self.V, self.M, self.N = V, M, N
        self.item_words = []
        for i in range(M):
            if V == 0:
                self.item_words.append([])
                continue
            k = F if not variable else int(rng.integers(1, F + 2))
            k = min(k, V)
            self.item_words.append(sorted(rng.choice(V, size=k, replace=False).tolist()))
        self.item_word_mat = torch.zeros(M, V)
        for i, ws in enumerate(self.item_words):
            if ws:
                self.item_word_mat[i, ws] = 1.0
        self.baskets = []
        for u in range(N):
            k = int(rng.integers(bs[0], bs[1] + 1))
            if force_single and u == 0:
                k = 1
            k = min(k, M)
            self.baskets.append(sorted(rng.choice(M, size=k, replace=False).tolist()))
        self.rebuild()

    def rebuild(self):
        self.basket_mat = torch.zeros(self.N, self.M)
        for u, b in enumerate(self.baskets):
            if b:
                self.basket_mat[u, b] = 1.0
        self.basket_sizes = self.basket_mat.sum(1)

    def user_word_weights(self):
        """(N,V): count_f / |H_u| (0 for an empty basket)."""
        counts = self.basket_mat @ self.item_word_mat
        return counts / self.basket_sizes.clamp_min(1.0)[:, None]

    def user_word_weights_loo(self, users, pos):
        """(B,V): leave-one-out weights for user users[b] with positive pos[b] removed."""
        counts = self.basket_mat[users] @ self.item_word_mat - self.item_word_mat[pos]
        n = self.basket_sizes[users] - 1.0
        w = counts / n.clamp_min(1.0)[:, None]
        return torch.where(n[:, None] > 0, w, torch.zeros_like(w))


def make_params(V_I, V_U, M, N, d, K_u, K_t, seed, scales=None):
    g = torch.Generator().manual_seed(seed)
    sc = dict(E_I=1.0, E_U=1.0, P_u=1.0, P_t=1.0, W_u=1.0, W_t=1.0)
    if scales:
        sc.update(scales)
    return {
        "E_I": torch.randn(V_I + M, d, generator=g) * sc["E_I"],
        "E_U": torch.randn(V_U + N, d, generator=g) * sc["E_U"],
        "P_u": torch.randn(K_u, d, generator=g) * sc["P_u"],
        "P_t": torch.randn(K_t, d, generator=g) * sc["P_t"],
        "W_u": torch.randn(K_t, d, generator=g) * sc["W_u"],
        "W_t": torch.randn(K_u, d, generator=g) * sc["W_t"],
    }


def params_from_cfg(cfg, seed, scales=None, V_I=None, V_U=None):
    V_I = cfg["V"] if V_I is None else V_I
    V_U = cfg["V"] if V_U is None else V_U
    return make_params(V_I, V_U, cfg["M"], cfg["N"], cfg["d"], cfg["K_u"], cfg["K_t"], seed, scales)


# ----------------------------------------------------------------------------- mechanism
def sim(q, P):
    """1 + guarded cosine; q (B,d), P (K,d) -> (B,K)."""
    return 1.0 + F.cosine_similarity(q.unsqueeze(1), P.unsqueeze(0), dim=-1, eps=EPS)


def compose(E, word_w, ids, V, use_id=True):
    """q = sum_f w_f E[f] (+ E[V+id]).  word_w (B,V) (counts or mean weights)."""
    q = word_w.to(E.dtype) @ E[:V]
    if use_id:
        q = q + E[V + ids]
    return q


def normalise(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)


def units(q_cos, q_lin, P, W):
    return sim(q_cos, P), q_lin @ W.T


def side_units(q, P, W, norm=None):
    if norm == "both":
        qn = normalise(q)
        return units(qn, qn, P, W)
    if norm == "linear":
        return units(q, normalise(q), P, W)
    return units(q, q, P, W)


def score(u_star, u_hat, t_star, t_hat):
    return u_star @ t_hat.T + u_hat @ t_star.T


def merged_forward(p, data, V_I, V_U, item_ids=True, user_ids=True, norm=None, user_w=None):
    M, N = data.M, data.N
    iw = data.item_word_mat if V_I > 0 else torch.zeros(M, 0)
    q_i = compose(p["E_I"], iw, torch.arange(M), V_I, item_ids)
    if user_w is None:
        user_w = data.user_word_weights() if V_U > 0 else torch.zeros(N, 0)
    q_u = compose(p["E_U"], user_w, torch.arange(N), V_U, user_ids)
    t_star, t_hat = side_units(q_i, p["P_t"], p["W_t"], norm)
    u_star, u_hat = side_units(q_u, p["P_u"], p["W_u"], norm)
    S = score(u_star, u_hat, t_star, t_hat)
    return dict(q_i=q_i, q_u=q_u, t_star=t_star, t_hat=t_hat, u_star=u_star, u_hat=u_hat, S=S, iw=iw, uw=user_w)


def ce_first(logits):
    return F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long))


def sample_batch(data, rng, n_neg=4):
    users = torch.arange(data.N)
    pos = torch.tensor([int(rng.choice(b)) for b in data.baskets])
    negs = torch.tensor(rng.integers(0, data.M, size=(data.N, n_neg)))
    return users, pos, negs


def merged_loss(p, data, V_I, V_U, users, pos, negs, item_ids=True, user_ids=True, loo=True, norm=None):
    """Sampled-softmax CE (1 positive + n_neg negatives per user). LOO: q_u from H_u \\ {pos}."""
    M = data.M
    iw = data.item_word_mat if V_I > 0 else torch.zeros(M, 0)
    q_i = compose(p["E_I"], iw, torch.arange(M), V_I, item_ids)
    t_star, t_hat = side_units(q_i, p["P_t"], p["W_t"], norm)
    if V_U > 0:
        uw = data.user_word_weights_loo(users, pos) if loo else data.user_word_weights()[users]
    else:
        uw = torch.zeros(len(users), 0)
    q_u = compose(p["E_U"], uw, users, V_U, user_ids)
    u_star, u_hat = side_units(q_u, p["P_u"], p["W_u"], norm)
    cand = torch.cat([pos[:, None], negs], 1)
    logits = (u_star[:, None, :] * t_hat[cand]).sum(-1) + (u_hat[:, None, :] * t_star[cand]).sum(-1)
    return ce_first(logits)


# host (UI-ProtoMF): free tables, base vector looked up TWICE per side
def host_forward(ph):
    T, U = ph["T"], ph["U"]
    ii = torch.arange(T.shape[0])
    uu = torch.arange(U.shape[0])
    t_star = sim(T[ii], ph["P_t"])
    t_hat = T[ii] @ ph["W_t"].T
    u_star = sim(U[uu], ph["P_u"])
    u_hat = U[uu] @ ph["W_u"].T
    return score(u_star, u_hat, t_star, t_hat)


def host_loss(ph, users, pos, negs):
    T, U = ph["T"], ph["U"]
    ii = torch.arange(T.shape[0])
    t_star = sim(T[ii], ph["P_t"])
    t_hat = T[ii] @ ph["W_t"].T
    u_star = sim(U[users], ph["P_u"])
    u_hat = U[users] @ ph["W_u"].T
    cand = torch.cat([pos[:, None], negs], 1)
    logits = (u_star[:, None, :] * t_hat[cand]).sum(-1) + (u_hat[:, None, :] * t_star[cand]).sum(-1)
    return ce_first(logits)


def A_of(P):
    """A = [1^T ; Phat^T] in R^{(d+1) x K}."""
    Phat = P / P.norm(dim=1, keepdim=True)
    return torch.cat([torch.ones(1, P.shape[0]), Phat.T], 0)


def null_basis(A, tol=1e-10):
    _, s, Vh = torch.linalg.svd(A, full_matrices=True)
    rank = int((s > tol * s.max()).sum())
    return Vh[rank:].T, Vh[:rank].T, rank  # null basis (K, K-rank), row-space basis (K, rank)


def instances():
    for cfg in CONFIGS:
        for seed in SEEDS:
            for sc in SCALES:
                yield cfg, seed, sc


# ============================================================================= claims
def c1():
    log("## C1 -- four-component decomposition; intercept kinds")
    worst = 0.0
    inert_mismatch = 0
    near_ties = 0
    n_inst = 0
    inst_with_flips = 0
    flip_users_min, flip_users_max, flip_pairs_tot = 10**9, 0, 0
    min_spread_1t = float("inf")
    for cfg, seed, sc in instances():
        for item_ids, user_ids in [(True, True), (False, True), (True, False)]:
            data = Data(seed=seed, **cfg)
            p = params_from_cfg(cfg, seed, sc)
            o = merged_forward(p, data, cfg["V"], cfg["V"], item_ids, user_ids)
            S, us, uh, ts, th = o["S"], o["u_star"], o["u_hat"], o["t_star"], o["t_hat"]
            one_th, one_uh = th.sum(1), uh.sum(1)
            dec = one_th[None, :] + (us - 1) @ th.T + one_uh[:, None] + uh @ (ts - 1).T
            worst = max(worst, maxrel(S, dec))
            # pairwise item orders; pairs tied to within 1e-9*max|S| (e.g. twin items with identical word sets in
            # the noid arm, whose scores differ only at the ULP level) are not counted as order changes
            tol = 1e-9 * S.abs().max().item()
            dS = S[:, :, None] - S[:, None, :]
            sgn = torch.sign(dS)
            S1 = S - one_uh[:, None]
            dS1 = S1[:, :, None] - S1[:, None, :]
            mm1 = (torch.sign(dS1) != sgn) & (dS.abs() > tol) & (dS1.abs() > tol)
            inert_mismatch += int(mm1.sum())
            near_ties += int(((dS.abs() <= tol) & ~torch.eye(S.shape[1], dtype=torch.bool)[None]).sum()) // 2
            S2 = S - one_th[None, :]
            dS2 = S2[:, :, None] - S2[:, None, :]
            mm = (torch.sign(dS2) != sgn) & (dS.abs() > tol) & (dS2.abs() > tol)
            fu = int(mm.any(-1).any(-1).sum())
            flip_users_min, flip_users_max = min(flip_users_min, fu), max(flip_users_max, fu)
            flip_pairs_tot += int(mm.sum()) // 2
            inst_with_flips += int(fu > 0)
            min_spread_1t = min(min_spread_1t, ((one_th.max() - one_th.min()) / one_th.abs().max()).item())
            n_inst += 1
    log(f"- instances: {n_inst} (4 configs x 6 seeds x 5 scale sets x 3 id-arms)")
    log(f"- decomposition S = 1^T t_hat + sum_l (u*_l-1) t_hat_l + 1^T u_hat + sum_k u_hat_k (t*_k-1): max rel err {fmt(worst)}")
    log(f"- 1^T u_hat rank-inert: pairwise item-order mismatches between S and S - 1^T u_hat = {inert_mismatch} (over all users/instances; "
        f"{near_ties} exactly/near-tied pairs -- twin items with identical word sets in the noid arm, |dS| <= 1e-9 max|S| -- excluded from all order comparisons)")
    log(f"- 1^T t_hat varies across items: min over instances of (max_i - min_i) 1^T t_hat_i / max_i |1^T t_hat_i| = {fmt(min_spread_1t)}")
    log(f"- removing 1^T t_hat changes item rankings: instances with >=1 user whose ranking changes = {inst_with_flips}/{n_inst}; "
        f"users with changed ranking per instance min/max = {flip_users_min}/{flip_users_max}; total flipped pairs = {flip_pairs_tot}")
    ok = worst < TOL_MP and inert_mismatch == 0 and inst_with_flips > 0
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c2():
    log("## C2 -- exact linear-half attribution, both sides")
    err_a1 = err_rem = err_b1 = err_shares = err_derived = 0.0
    full_dev_min, full_dev_max = float("inf"), 0.0
    cf_dev_min, cf_dev_max = float("inf"), 0.0
    n_rem = n_cf = 0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V, M, N = cfg["V"], cfg["M"], cfg["N"]
        EI, EU, Wt, Wu, Pt, Pu = p["E_I"], p["E_U"], p["W_t"], p["W_u"], p["P_t"], p["P_u"]
        o = merged_forward(p, data, V, V)
        iw, uw = o["iw"], o["uw"]
        # (a) t_hat identity
        th_dec = iw @ (EI[:V] @ Wt.T) + EI[V:] @ Wt.T
        err_a1 = max(err_a1, maxrel(o["t_hat"], th_dec))
        # (a) removal effect on the u*^T t_hat half (one word per item, first two items to keep runtime low)
        for i in range(M):
            for f in data.item_words[i][:2]:
                iw2 = iw.clone()
                iw2[i, f] = 0.0
                qi2 = compose(EI, iw2, torch.arange(M), V, True)[i]
                lin_before = o["u_star"] @ o["t_hat"][i]
                lin_after = o["u_star"] @ (Wt @ qi2)
                pred = -(o["u_star"] @ (Wt @ EI[f]))
                err_rem = max(err_rem, maxrel(lin_after - lin_before, pred))
                ts2, th2 = units(qi2[None], qi2[None], Pt, Wt)
                S_after = o["u_star"] @ th2[0] + o["u_hat"] @ ts2[0]
                dS = S_after - o["S"][:, i]
                dev = ((dS - pred).abs() / pred.abs().clamp_min(1e-300))
                full_dev_min = min(full_dev_min, dev.min().item())
                full_dev_max = max(full_dev_max, dev.max().item())
                n_rem += 1
        # (b) u_hat identity + shares
        uh_dec = uw @ (EU[:V] @ Wu.T) + EU[V:] @ Wu.T
        err_b1 = max(err_b1, maxrel(o["u_hat"], uh_dec))
        a = iw @ EU[:V] @ Wu.T  # (M,K_t): W_u sum_{f in f_j} E_U[f]
        word_part = uw @ EU[:V] @ Wu.T
        for u in range(N):
            H = data.baskets[u]
            n = len(H)
            s = a[H] / n
            err_shares = max(err_shares, maxrel(s.sum(0), word_part[u]))
            if n >= 2:
                abar = a[H].mean(0)
                for j in H:
                    H2 = [k for k in H if k != j]
                    uw2 = iw[H2].sum(0) / (n - 1)
                    q2 = uw2 @ EU[:V] + EU[V + u]
                    delta = Wu @ q2 - o["u_hat"][u]
                    sj = a[j] / n
                    dev = (delta + sj).norm().item() / max(sj.norm().item(), 1e-300)
                    cf_dev_min, cf_dev_max = min(cf_dev_min, dev), max(cf_dev_max, dev)
                    # normalised by the share scale (delta and the prediction are both exactly 0 when the basket
                    # holds only twins of j, so a plain relative error would compare rounding noise with itself)
                    err_derived = max(err_derived, maxabs(delta, (abar - a[j]) / (n - 1)) / a[H].abs().max().item())
                    n_cf += 1
    log(f"- (a) t_hat = sum_f W_t E_I[f] + W_t E_I[V+i]: max rel err {fmt(err_a1)}")
    log(f"- (a) removal of word f from q_i: change of the u*^T t_hat half vs -u*^T W_t E_I[f]: max rel err {fmt(err_rem)} over {n_rem} removals x all users")
    log(f"- (a) side note: change of the FULL score S vs -u*^T W_t E_I[f]: relative deviation min/max = {fmt(full_dev_min)}/{fmt(full_dev_max)} "
        f"(the cosine half also moves, so the per-word share is exact for the linear half only)")
    log(f"- (b) u_hat = (1/|H|) sum_j sum_f W_u E_U[f] + W_u E_U[V+u]: max rel err {fmt(err_b1)}")
    log(f"- (b) per-purchase shares s_j sum to the word part of u_hat: max rel err {fmt(err_shares)}")
    log(f"- (b) counterfactual removal of purchase j (re-mean over |H|-1) vs -s_j: ||delta + s_j||/||s_j|| min/max = {fmt(cf_dev_min)}/{fmt(cf_dev_max)} over {n_cf} (u,j) pairs -> NOT equal in general (confirmed)")
    log(f"- (b) derived closed form delta = (a_bar - a_j)/(|H|-1) with a_j = W_u sum_f E_U[f], a_bar = basket mean: max rel err {fmt(err_derived)}")
    # constructed equality-condition tests
    g = torch.Generator().manual_seed(123)
    d, Kt = 4, 6
    rows = []
    for trial in range(4):
        m = torch.randn(d, generator=g) * (10.0 ** (trial - 1))
        dl = torch.randn(d, generator=g)
        uid = torch.randn(d, generator=g)
        Wu = torch.randn(Kt, d, generator=g)
        # basket {A,B,C}, disjoint single words a,b,c; word sums: m-dl, m, m+dl  -> a_B equals the basket mean word sum
        EA, EB, EC = m - dl, m, m + dl
        uh_full = Wu @ ((EA + EB + EC) / 3 + uid)
        uh_noB = Wu @ ((EA + EC) / 2 + uid)
        delta_B = uh_noB - uh_full
        s_B = Wu @ EB / 3
        # alternative: the OTHER purchases cancel (EA = -EC): then delta_B = -s_B although a_B != a_bar
        EA2, EC2 = EA, -EA
        uh_full2 = Wu @ ((EA2 + EB + EC2) / 3 + uid)
        uh_noB2 = Wu @ ((EA2 + EC2) / 2 + uid)
        delta_B2 = uh_noB2 - uh_full2
        rows.append((delta_B.norm().item(), s_B.norm().item(), (delta_B + s_B).norm().item(),
                     (delta_B2 + s_B).norm().item(), ((EB - (EA2 + EB + EC2) / 3)).norm().item()))
    log("- (b) equality condition test, basket {A,B,C} with single disjoint words, word sums (m-dl, m, m+dl) so a_B == basket mean word sum:")
    for r in rows:
        log(f"    ||delta_B|| = {fmt(r[0])}   ||s_B|| = {fmt(r[1])}   ||delta_B + s_B|| = {fmt(r[2])}   "
            f"| alt. construction (others cancel, a_B != mean by {fmt(r[4])}): ||delta_B + s_B|| = {fmt(r[3])}")
    log("    -> when a_j equals the basket mean, removing j leaves the mean unchanged (delta = 0) and delta != -s_j; "
        "equality delta = -s_j holds instead iff W_u a_j = W_u * (basket TOTAL word sum), i.e. the other purchases cancel.")
    ok_ident = max(err_a1, err_rem, err_b1, err_shares, err_derived) < TOL_MP and cf_dev_min > 1e-6
    cond_falsified = all(r[2] > 1e-6 * max(r[1], 1e-300) for r in rows)
    v = "FALSIFIED (final clause of (b) only; all identities and the 'not in general equal' statement CONFIRMED)" if (ok_ident and cond_falsified) else ("CONFIRMED" if ok_ident else "FALSIFIED")
    log(f"- **Verdict: {v}**")
    return v


def c3():
    log("## C3 -- cosine-unit exact shares inside the merged score")
    err_i = err_u = err_regroup = 0.0
    max_count = 0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V = cfg["V"]
        EI, EU, Pt, Pu = p["E_I"], p["E_U"], p["P_t"], p["P_u"]
        o = merged_forward(p, data, V, V)
        iw, uw = o["iw"], o["uw"]
        qn, Pn = o["q_i"].norm(dim=1), Pt.norm(dim=1)
        shares_i = (iw @ (EI[:V] @ Pt.T) + EI[V:] @ Pt.T) / (qn[:, None] * Pn[None, :])
        err_i = max(err_i, maxabs(o["t_star"] - 1, shares_i))
        qun, Pun = o["q_u"].norm(dim=1), Pu.norm(dim=1)
        shares_u = (uw @ (EU[:V] @ Pu.T) + EU[V:] @ Pu.T) / (qun[:, None] * Pun[None, :])
        err_u = max(err_u, maxabs(o["u_star"] - 1, shares_u))
        # regrouping: per-word (count_f/|H|) vs per-purchase (1/|H|) sum_{f in f_j}
        per_word = (uw @ (EU[:V] @ Pu.T)) / (qun[:, None] * Pun[None, :])
        per_item = iw @ (EU[:V] @ Pu.T)
        per_purchase = (data.basket_mat @ per_item) / data.basket_sizes.clamp_min(1)[:, None] / (qun[:, None] * Pun[None, :])
        err_regroup = max(err_regroup, maxrel(per_word, per_purchase))
        max_count = max(max_count, int((data.basket_mat @ data.item_word_mat).max()))
    log(f"- t*_k - 1 = sum_r E_I[r].P_t[k]/(||q_i|| ||P_t[k]||) (rows = words + ID): max abs err {fmt(err_i)}")
    log(f"- u*_l - 1 = sum_r w_r E_U[r].P_u[l]/(||q_u|| ||P_u[l]||), w = count/|H| (words), 1 (ID): max abs err {fmt(err_u)} (max word count within a basket = {max_count})")
    log(f"- per-purchase regrouping == per-word grouping: max rel err {fmt(err_regroup)}")
    ok = max(err_i, err_u, err_regroup) < TOL_MP
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c4():
    log("## C4 -- keystone reduction to the host (V = 0, ID rows both sides)")
    res = {}
    grad_worst = {}
    loss_equal = True
    for dtype in [torch.float64, torch.float32]:
        all_equal, worst = True, 0.0
        n = 0
        for cfg in CONFIGS:
            for seed in SEEDS:
                for sc in SCALES:
                    data = Data(seed=seed, **{**cfg, "V": 0})
                    p = params_from_cfg(cfg, seed, sc, V_I=0, V_U=0)
                    p = {k: v.to(dtype) for k, v in p.items()}
                    ph = {"T": p["E_I"].clone(), "U": p["E_U"].clone(), "P_u": p["P_u"].clone(), "P_t": p["P_t"].clone(),
                          "W_u": p["W_u"].clone(), "W_t": p["W_t"].clone()}
                    S_m = merged_forward(p, data, 0, 0)["S"]
                    S_h = host_forward(ph)
                    eq = torch.equal(S_m, S_h)
                    all_equal &= eq
                    worst = max(worst, maxabs(S_m, S_h))
                    n += 1
                    if dtype == torch.float64:
                        rng = np.random.default_rng(seed)
                        users, pos, negs = sample_batch(data, rng)
                        pr = {k: v.clone().requires_grad_() for k, v in p.items()}
                        lm = merged_loss(pr, data, 0, 0, users, pos, negs)
                        gm = torch.autograd.grad(lm, list(pr.values()))
                        hr = {k: v.clone().requires_grad_() for k, v in ph.items()}
                        lh = host_loss(hr, users, pos, negs)
                        gh = torch.autograd.grad(lh, list(hr.values()))
                        loss_equal &= torch.equal(lm, lh)
                        names_m = list(pr.keys())
                        names_h = list(hr.keys())
                        for nm_m, nm_h in zip(names_m, names_h):
                            a, b = gm[names_m.index(nm_m)], gh[names_h.index(nm_h)]
                            key = f"{nm_m}~{nm_h}"
                            grad_worst[key] = max(grad_worst.get(key, 0.0), maxrel(a, b))
        res[dtype] = (all_equal, worst, n)
    for dtype, (eq, w, n) in res.items():
        log(f"- (a) scores bit-identical ({str(dtype).split('.')[-1]}): torch.equal on all {n} instances = {eq}; max abs diff = {fmt(w)}")
    log(f"- (b) sampled-softmax loss values torch.equal (float64): {loss_equal}")
    for k, v in grad_worst.items():
        log(f"- (b) grad max rel err {k}: {fmt(v)}")
    ok = all(eq for eq, _, _ in res.values()) and max(grad_worst.values()) < TOL_MP
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c5():
    log("## C5 -- single-side reductions")
    eq_item, eq_user, w_item, w_user, n = True, True, 0.0, 0.0, 0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        V, M, N = cfg["V"], cfg["M"], cfg["N"]
        # item side reduced: V_items = 0, item ID on ; user side composed from the items' word sets
        p = params_from_cfg(cfg, seed, sc, V_I=0, V_U=V)
        S_m = merged_forward(p, data, 0, V)["S"]
        T = p["E_I"]
        ii = torch.arange(M)
        t_star, t_hat = sim(T[ii], p["P_t"]), T[ii] @ p["W_t"].T
        q_u = compose(p["E_U"], data.user_word_weights(), torch.arange(N), V, True)
        u_star, u_hat = units(q_u, q_u, p["P_u"], p["W_u"])
        S_h = score(u_star, u_hat, t_star, t_hat)
        eq_item &= torch.equal(S_m, S_h)
        w_item = max(w_item, maxabs(S_m, S_h))
        # user side reduced: V_users = 0, user ID on ; item side composed
        p2 = params_from_cfg(cfg, seed, sc, V_I=V, V_U=0)
        S_m2 = merged_forward(p2, data, V, 0)["S"]
        U = p2["E_U"]
        uu = torch.arange(N)
        u_star2, u_hat2 = sim(U[uu], p2["P_u"]), U[uu] @ p2["W_u"].T
        q_i = compose(p2["E_I"], data.item_word_mat, torch.arange(M), V, True)
        t_star2, t_hat2 = units(q_i, q_i, p2["P_t"], p2["W_t"])
        S_h2 = score(u_star2, u_hat2, t_star2, t_hat2)
        eq_user &= torch.equal(S_m2, S_h2)
        w_user = max(w_user, maxabs(S_m2, S_h2))
        n += 1
    log(f"- item side reduced (V_items=0, ID on) vs free item table looked up twice + composed users: torch.equal on all {n} instances = {eq_item}; max abs diff {fmt(w_item)}")
    log(f"- user side reduced (V_users=0, ID on) vs free user table looked up twice + composed items: torch.equal on all {n} instances = {eq_user}; max abs diff {fmt(w_user)}")
    v = "CONFIRMED" if (eq_item and eq_user) else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c6():
    log("## C6 -- gauge freedom lives in W")
    # (a),(b),(e) for W_t with A_u ; mirror for W_u with A_t
    worst_a = worst_a_norm = worst_b = 0.0
    worst_a_m = worst_a_norm_m = worst_b_m = 0.0
    nullity_ok = True
    nullity_rows = set()
    n_gauge = 0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V, d, K_u, K_t = cfg["V"], cfg["d"], cfg["K_u"], cfg["K_t"]
        o = merged_forward(p, data, V, V)
        rng = np.random.default_rng(seed)
        users, pos, negs = sample_batch(data, rng)
        for side in ["t", "u"]:
            P = p["P_u"] if side == "t" else p["P_t"]
            K = K_u if side == "t" else K_t
            A = A_of(P)
            Nb, Rb, rank = null_basis(A)
            nullity = K - rank
            expected = max(K - d - 1, 0)
            nullity_ok &= (nullity == expected)
            nullity_rows.add((K, d, nullity, expected))
            if nullity == 0:
                continue
            g = torch.Generator().manual_seed(100 + seed)
            for dscale in [1e-3, 1.0, 1e3]:
                Delta = Nb @ (torch.randn(nullity, d, generator=g) * dscale)
                assert (A @ Delta).abs().max() < 1e-9 * Delta.abs().max()
                Wname = "W_t" if side == "t" else "W_u"
                p2 = dict(p)
                p2[Wname] = p[Wname] + Delta
                S2 = merged_forward(p2, data, V, V)["S"]
                q_other = o["q_i"] if side == "t" else o["q_u"]
                nat = Delta.norm().item() * q_other.norm(dim=1).max().item() * math.sqrt(K) * 2  # natural scale of a would-be change
                o2 = merged_forward(p2, data, V, V)
                # rounding floor of S: the magnitude of the summed terms (max over both models)
                floor = max((o["u_star"].abs() @ o["t_hat"].abs().T + o["u_hat"].abs() @ o["t_star"].abs().T).max().item(),
                            (o2["u_star"].abs() @ o2["t_hat"].abs().T + o2["u_hat"].abs() @ o2["t_star"].abs().T).max().item())
                e_abs = maxabs(o["S"], S2)
                if side == "t":
                    worst_a = max(worst_a, e_abs / floor)
                    worst_a_norm = max(worst_a_norm, e_abs / nat)
                else:
                    worst_a_m = max(worst_a_m, e_abs / floor)
                    worst_a_norm_m = max(worst_a_norm_m, e_abs / nat)
                # (b) gradient orthogonality
                pr = {k: v.clone().requires_grad_(k == Wname) for k, v in p.items()}
                loss = merged_loss(pr, data, V, V, users, pos, negs)
                (gW,) = torch.autograd.grad(loss, [pr[Wname]])
                ip = abs((gW * Delta).sum().item()) / (gW.norm().item() * Delta.norm().item())
                if side == "t":
                    worst_b = max(worst_b, ip)
                else:
                    worst_b_m = max(worst_b_m, ip)
                n_gauge += 1
    log(f"- (a) S invariant under W_t -> W_t + Delta (A_u Delta = 0), Delta scales 1e-3/1/1e3 (incl. ||Delta|| = 1e-6 ||W_t|| and 1e6 ||W_t||): "
        f"max |dS| / rounding floor of S (max_(u,i) sum_l |u*_l||t_hat_l| + sum_k |u_hat_k||t*_k|) = {fmt(worst_a)}; "
        f"max |dS| / (||Delta||_F max||q_i|| sqrt(K_u) 2) = {fmt(worst_a_norm)} (this second ratio is only informative when ||Delta|| >> ||W||; "
        f"for ||Delta|| << ||W|| the floor is S's own rounding)  [{n_gauge} gauge shifts incl. mirror]")
    log(f"- (b) <grad_W_t L, Delta> / (||grad|| ||Delta||): max {fmt(worst_b)}")
    log(f"- (a,b) mirror W_u with A_t: max |dS| / floor {fmt(worst_a_m)} (vs would-be change {fmt(worst_a_norm_m)}); |<grad_W_u, Delta>| normalised max {fmt(worst_b_m)}")
    log(f"- (e) nullity of A = [1^T; Phat^T] measured vs K-d-1 (0 if K<=d+1), all seeds/scales agree = {nullity_ok}; (K, d, measured, expected): {sorted(nullity_rows)}")
    # (c),(d) SGD
    drift_frozen, move_row, drift_joint, wd_step_err, wd_total_err = 0.0, float("inf"), 0.0, 0.0, 0.0
    T_steps, lr, wd = 40, 0.05, 0.1
    for cfg in [CONFIGS[0], CONFIGS[3]]:
        for seed in SEEDS[:4]:
            data = Data(seed=seed, **cfg)
            V = cfg["V"]
            p = params_from_cfg(cfg, seed, None)
            A = A_of(p["P_u"])
            Nb, Rb, _ = null_basis(A)

            def run(freeze, wd_):
                rng = np.random.default_rng(seed)
                params = {k: v.clone().requires_grad_(k not in freeze) for k, v in p.items()}
                names = [k for k in params if params[k].requires_grad]
                traj = [params["W_t"].detach().clone()]
                for _ in range(T_steps):
                    users, pos, negs = sample_batch(data, rng)
                    loss = merged_loss(params, data, V, V, users, pos, negs)
                    grads = torch.autograd.grad(loss, [params[k] for k in names])
                    with torch.no_grad():
                        for k, g_ in zip(names, grads):
                            if wd_:
                                g_ = g_ + wd_ * params[k]
                            params[k] -= lr * g_
                    traj.append(params["W_t"].detach().clone())
                return traj

            # (c) plain SGD, P_u frozen (A_u fixed), everything else trained
            traj = run(("P_u",), 0.0)
            nc0, ncT = Nb.T @ traj[0], Nb.T @ traj[-1]
            drift_frozen = max(drift_frozen, maxrel(nc0, ncT))
            rc0, rcT = Rb.T @ traj[0], Rb.T @ traj[-1]
            move_row = min(move_row, maxrel(rc0, rcT))
            # joint training (P_u also trained): the null space defined at step 0 rotates -> informative only
            trajJ = run((), 0.0)
            drift_joint = max(drift_joint, maxrel(Nb.T @ trajJ[0], Nb.T @ trajJ[-1]))
            # (d) weight decay, P_u frozen
            trajW = run(("P_u",), wd)
            for t in range(T_steps):
                a, b = Nb.T @ trajW[t + 1], (1 - lr * wd) * (Nb.T @ trajW[t])
                wd_step_err = max(wd_step_err, maxrel(a, b))
            wd_total_err = max(wd_total_err, maxrel(Nb.T @ trajW[-1], (1 - lr * wd) ** T_steps * (Nb.T @ trajW[0])))
    log(f"- (c) plain SGD, {T_steps} steps, lr={lr}, P_u frozen (A_u fixed), all other params trained, 2 configs x 4 seeds: "
        f"null-space coords of W_t max rel drift {fmt(drift_frozen)}; row-space coords min rel movement {fmt(move_row)}")
    log(f"- (c) informative: with P_u ALSO trained the step-0 null space rotates; drift of step-0 null coords = {fmt(drift_joint)} (claim presumes a fixed A_u)")
    log(f"- (d) weight decay wd={wd} (grad += wd*W): per-step null coords vs (1-lr*wd) x previous: max rel err {fmt(wd_step_err)}; "
        f"after {T_steps} steps vs (1-lr*wd)^T x initial: max rel err {fmt(wd_total_err)}")
    ok = (worst_a < TOL_MP and worst_b < TOL_MP and worst_a_m < TOL_MP and worst_b_m < TOL_MP and nullity_ok
          and drift_frozen < TOL_MP and move_row > 1e-6 and wd_step_err < TOL_MP and wd_total_err < 1e-9)
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}** (interpretation for (c)/(d): A_u fixed, i.e. P_u frozen while W_t and all other tables train)")
    return v


def c7():
    log("## C7 -- live intercept attributable and not gauge")
    err_dec = err_gauge = err_gauge_norm = 0.0
    n = 0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V, d, K_u = cfg["V"], cfg["d"], cfg["K_u"]
        o = merged_forward(p, data, V, V)
        one_th = o["t_hat"].sum(1)
        v = p["W_t"].T @ torch.ones(K_u)
        dec = o["iw"] @ (p["E_I"][:V] @ v) + p["E_I"][V:] @ v
        err_dec = max(err_dec, maxrel(one_th, dec))
        A = A_of(p["P_u"])
        Nb, _, rank = null_basis(A)
        if K_u - rank > 0:
            g = torch.Generator().manual_seed(200 + seed)
            for dscale in [1e-3, 1.0, 1e3]:
                Delta = Nb @ (torch.randn(K_u - rank, d, generator=g) * dscale)
                th2 = o["q_i"] @ (p["W_t"] + Delta).T
                one_th2 = th2.sum(1)
                floor = max(o["t_hat"].abs().sum(1).max().item(), th2.abs().sum(1).max().item())  # rounding floor of the sum
                err_gauge = max(err_gauge, maxabs(one_th, one_th2) / floor)
                nat = Delta.norm().item() * o["q_i"].norm(dim=1).max().item() * math.sqrt(K_u)
                err_gauge_norm = max(err_gauge_norm, maxabs(one_th, one_th2) / nat)
                n += 1
    log(f"- 1^T t_hat = sum_f (W_t^T 1).E_I[f] + (W_t^T 1).E_I[V+i]: max rel err {fmt(err_dec)}")
    log(f"- 1^T t_hat invariant under gauge shifts Delta (A_u Delta=0), {n} shifts (Delta scales 1e-3/1/1e3): max |change| / rounding floor (max_i sum_l |t_hat_l|) = {fmt(err_gauge)}; "
        f"max |change| / (||Delta|| max||q_i|| sqrt(K_u)) = {fmt(err_gauge_norm)}")
    ok = err_dec < TOL_MP and err_gauge < TOL_MP
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c8():
    log("## C8 -- leave-one-out reaches both roles")
    err_ident = 0.0
    mins, maxs = float("inf"), 0.0
    n = 0
    n_eq_mean, eq_mean_max = 0, 0.0
    single_min = float("inf")
    ustar_diff_min = float("inf")
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V, N = cfg["V"], cfg["N"]
        EU, Wu, Pu = p["E_U"], p["W_u"], p["P_u"]
        word_sums = data.item_word_mat @ EU[:V]  # (M,d)
        for u in range(N):
            H = data.baskets[u]
            mean_ws = word_sums[H].mean(0)
            for i in data.baskets[u]:
                eq_mean = (word_sums[i] - mean_ws).norm().item() <= 1e-9 * word_sums[H].norm(dim=1).max().item()
                users = torch.tensor([u])
                pos = torch.tensor([i])
                w_full = data.user_word_weights()[users]
                w_loo = data.user_word_weights_loo(users, pos)
                q_full = compose(EU, w_full, users, V, True)
                q_loo = compose(EU, w_loo, users, V, True)
                u_hat = q_loo @ Wu.T                 # correct: both roles from q_loo
                u_hat_wrong = q_full @ Wu.T          # LOO applied to the cosine unit only
                diff = u_hat_wrong - u_hat
                err_ident = max(err_ident, maxabs(diff, (q_full - q_loo) @ Wu.T) / max(u_hat.abs().max().item(), u_hat_wrong.abs().max().item()))
                nrm = diff.norm().item() / max(u_hat.norm().item(), 1e-300)
                if len(data.baskets[u]) == 1:
                    single_min = min(single_min, nrm)
                elif eq_mean:
                    n_eq_mean += 1
                    eq_mean_max = max(eq_mean_max, nrm)
                else:
                    mins, maxs = min(mins, nrm), max(maxs, nrm)
                ustar_diff_min = min(ustar_diff_min, maxabs(sim(q_loo, Pu), sim(q_full, Pu)))
                n += 1
    log(f"- u_hat' - u_hat = W_u (q_full - q_loo): max rel err {fmt(err_ident)} over {n} (u, positive) pairs")
    log(f"- ||u_hat' - u_hat|| / ||u_hat||, |H|>=2 and positive's word sum != basket mean: min {fmt(mins)}, max {fmt(maxs)} (generically nonzero)")
    log(f"- pairs where the positive's word sum == basket mean (baskets of twin items): {n_eq_mean}, max ||u_hat' - u_hat||/||u_hat|| there = {fmt(eq_mean_max)} (zero as predicted)")
    log(f"- same for |H|=1 (LOO word part = 0): min {fmt(single_min)}")
    log(f"- control: u* also differs between q_loo and q_full (min over pairs of max abs diff {fmt(ustar_diff_min)}) -- both roles see the same q_loo in the merged model")
    # constructed zero case: positive's word sum equals the basket mean
    g = torch.Generator().manual_seed(321)
    d, Kt = 4, 6
    m, dl, uid = torch.randn(d, generator=g), torch.randn(d, generator=g), torch.randn(d, generator=g)
    Wu = torch.randn(Kt, d, generator=g)
    EA, EB, EC = m - dl, m, m + dl
    q_full = (EA + EB + EC) / 3 + uid
    q_loo_B = (EA + EC) / 2 + uid
    q_loo_A = (EB + EC) / 2 + uid
    zero_case = (Wu @ (q_full - q_loo_B)).norm().item()
    nonzero_case = (Wu @ (q_full - q_loo_A)).norm().item()
    log(f"- constructed: positive B with word sum == basket mean -> ||u_hat' - u_hat|| = {fmt(zero_case)}; positive A (!= mean) -> {fmt(nonzero_case)}")
    ok = err_ident < TOL_MP and mins > 1e-6 and single_min > 1e-6 and zero_case < 1e-12
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c9():
    log("## C9 -- normalisation controls")
    # (a)
    worst_star, min_change_hat, min_change_S = 0.0, float("inf"), float("inf")
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V = cfg["V"]
        o = merged_forward(p, data, V, V)
        on = merged_forward(p, data, V, V, norm="both")
        worst_star = max(worst_star, maxabs(o["u_star"], on["u_star"]), maxabs(o["t_star"], on["t_star"]))
        min_change_hat = min(min_change_hat, maxrel(o["u_hat"], on["u_hat"]), maxrel(o["t_hat"], on["t_hat"]))
        min_change_S = min(min_change_S, maxrel(o["S"], on["S"]))
    log(f"- (a) normalised model: u*, t* max abs change {fmt(worst_star)}; u_hat/t_hat min rel change {fmt(min_change_hat)}; S min rel change {fmt(min_change_S)}")
    # (b) models without linear units
    worst_S, worst_g = 0.0, 0.0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V, M, N, K_u, K_t = cfg["V"], cfg["M"], cfg["N"], cfg["K_u"], cfg["K_t"]
        g = torch.Generator().manual_seed(400 + seed)
        rng = np.random.default_rng(seed)
        users, pos, negs = sample_batch(data, rng)
        cand = torch.cat([pos[:, None], negs], 1)
        # user-cosine model: S = u*^T t, free t (M x K_u)
        Tfree = torch.randn(M, K_u, generator=g) * sc.get("W_t", 1.0)
        for normed in [False, True]:
            EU = p["E_U"].clone().requires_grad_()
            Pu = p["P_u"].clone().requires_grad_()
            Tf = Tfree.clone().requires_grad_()
            uw = data.user_word_weights_loo(users, pos)
            q_u = compose(EU, uw, users, V, True)
            if normed:
                q_u = normalise(q_u)
            u_star = sim(q_u, Pu)
            logits = (u_star[:, None, :] * Tf[cand]).sum(-1)
            loss = ce_first(logits)
            grads = torch.autograd.grad(loss, [EU, Pu, Tf])
            if not normed:
                ref = (logits.detach(), [x.detach() for x in grads])
            else:
                worst_S = max(worst_S, maxrel(ref[0], logits.detach()))
                for a, b in zip(ref[1], grads):
                    worst_g = max(worst_g, maxrel(a, b))
        # item-cosine model: S = u^T t*, free u (N x K_t)
        Ufree = torch.randn(N, K_t, generator=g) * sc.get("W_u", 1.0)
        for normed in [False, True]:
            EI = p["E_I"].clone().requires_grad_()
            Pt = p["P_t"].clone().requires_grad_()
            Uf = Ufree.clone().requires_grad_()
            q_i = compose(EI, data.item_word_mat, torch.arange(M), V, True)
            if normed:
                q_i = normalise(q_i)
            t_star = sim(q_i, Pt)
            logits = (Uf[users][:, None, :] * t_star[cand]).sum(-1)
            loss = ce_first(logits)
            grads = torch.autograd.grad(loss, [EI, Pt, Uf])
            if not normed:
                ref = (logits.detach(), [x.detach() for x in grads])
            else:
                worst_S = max(worst_S, maxrel(ref[0], logits.detach()))
                for a, b in zip(ref[1], grads):
                    worst_g = max(worst_g, maxrel(a, b))
    log(f"- (b) no-linear-unit models (u*^T t free t; u^T t* free u), q normalised before the cosine unit: scores max rel err {fmt(worst_S)}; grads (E tables, P, free table) max rel err {fmt(worst_g)}")
    # (c) one-sided block normalisation
    worst_star_c, worst_scale_free = 0.0, 0.0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V = cfg["V"]
        o = merged_forward(p, data, V, V)
        oc = merged_forward(p, data, V, V, norm="linear")
        worst_star_c = max(worst_star_c, maxabs(o["u_star"], oc["u_star"]), maxabs(o["t_star"], oc["t_star"]))
        for c in [1e-3, 0.37, 1.0, 2.0, 1e3]:
            _, uh_c = side_units(c * o["q_u"], p["P_u"], p["W_u"], norm="linear")
            _, th_c = side_units(c * o["q_i"], p["P_t"], p["W_t"], norm="linear")
            worst_scale_free = max(worst_scale_free, maxrel(uh_c, oc["u_hat"]), maxrel(th_c, oc["t_hat"]))
    log(f"- (c) one-sided block normalisation: u*, t* max abs change {fmt(worst_star_c)} (bit-identical path); u_hat(c q) vs u_hat(q), c in {{1e-3,0.37,1,2,1e3}}: max rel err {fmt(worst_scale_free)}")
    ok = worst_star < TOL_FLOAT and min_change_hat > 1e-6 and min_change_S > 1e-6 and worst_S < TOL_MP and worst_g < TOL_MP and worst_star_c == 0.0 and worst_scale_free < TOL_MP
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c10():
    log("## C10 -- twin separation: cosine squashed, linear not")
    cs = np.geomspace(1e-3, 1e3, 13)
    log(f"- sweep of word-mass scale c (m = c*m0, ||m|| = c||m0||): {[f'{c:.0e}' for c in cs]}")
    all_mono, all_tend, lin_worst_abs, lin_worst_rel = True, True, 0.0, 0.0
    tail_mono, n_nonmono = True, 0
    for seed in range(8):
        g = torch.Generator().manual_seed(500 + seed)
        d, V, K_t, K_u = 4, 5, 7, 6
        E0 = torch.randn(V, d, generator=g)
        e_i, e_j = torch.randn(d, generator=g), torch.randn(d, generator=g)
        Pt = torch.randn(K_t, d, generator=g)
        Wt = torch.randn(K_u, d, generator=g)
        words = [0, 1, 2]
        ref_lin = Wt @ (e_i - e_j)
        diffs = []
        for c in cs:
            m = (c * E0)[words].sum(0)
            qi, qj = m + e_i, m + e_j
            diffs.append((sim(qi[None], Pt)[0] - sim(qj[None], Pt)[0]).norm().item())
            lin = Wt @ qi - Wt @ qj
            lin_worst_abs = max(lin_worst_abs, maxabs(lin, ref_lin))
            lin_worst_rel = max(lin_worst_rel, maxabs(lin, ref_lin) / ref_lin.norm().item())
        viol = sum(1 for k in range(len(cs) - 1) if diffs[k + 1] > diffs[k] * (1 + 1e-12))
        viol_tail = sum(1 for k in range(len(cs) - 1) if cs[k] >= 1.0 and diffs[k + 1] > diffs[k] * (1 + 1e-12))
        slope = math.log(diffs[-1] / diffs[-5]) / math.log(cs[-1] / cs[-5])
        tend = diffs[-1] < 1e-2 * diffs[0]
        all_mono &= viol == 0
        tail_mono &= viol_tail == 0
        all_tend &= tend
        n_nonmono += int(viol > 0)
        ratio_e = (e_i.norm() / (E0[words].sum(0)).norm()).item()
        log(f"  seed {seed}: ||t*(i)-t*(j)|| = {' '.join(f'{x:.2e}' for x in diffs)}  | increases: whole sweep {viol}, c>=1 only {viol_tail} | "
            f"last/first = {diffs[-1]/diffs[0]:.2e} | log-log slope over c in [10,1000] = {slope:.3f} | ||e_i||/||m0|| = {ratio_e:.2f}")
    log(f"- cosine-unit difference tends to 0 in all 8 seeds: {all_tend} (tail slope ~ -1, i.e. O(1/||m||))")
    log(f"- strictly decreasing over the WHOLE sweep in all seeds: {all_mono} (non-monotone in {n_nonmono}/8 seeds while ||m|| < ||e||: the difference first grows slightly before the word mass dominates); "
        f"decreasing for all c >= 1 (||m|| >= ||m0||, i.e. word mass at or above the ID-row scale) in all seeds: {tail_mono}")
    log(f"- linear-unit difference t_hat_i - t_hat_j vs W_t(e_i - e_j) across the sweep: max abs err {fmt(lin_worst_abs)}, max rel err {fmt(lin_worst_rel)} (float64 rounding at c=1e3)")
    ok = all_tend and tail_mono and lin_worst_rel < 1e-10
    if ok and not all_mono:
        v = ("CONFIRMED (tends to 0; linear-unit difference exactly independent of m); the parenthetical 'decreasing in ||m|| over a sweep' "
             f"FALSIFIED as a global statement -- non-monotone at small ||m|| in {n_nonmono}/8 seeds, monotone once ||m|| >= ID-row scale")
    else:
        v = "CONFIRMED" if (ok and all_mono) else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c11():
    log("## C11 -- degenerate paths")
    q0_ok = us_ok = uh_ok = S_eq = S_elem_eq = True
    S_maxabs = S_maxrel = 0.0
    min_q, min_spread, min_th = float("inf"), float("inf"), float("inf")
    twins_eq = True
    twins_maxabs = 0.0
    n = 0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        data.baskets[0] = []
        data.rebuild()
        p = params_from_cfg(cfg, seed, sc)
        V = cfg["V"]
        # (a) cold user 0: empty basket, no user-ID row
        o = merged_forward(p, data, V, V, user_ids=False)
        q0_ok &= torch.equal(o["q_u"][0], torch.zeros(cfg["d"]))
        us_ok &= torch.equal(o["u_star"][0], torch.ones(cfg["K_u"]))
        uh_ok &= torch.equal(o["u_hat"][0], torch.zeros(cfg["K_t"]))
        one_th = o["t_hat"].sum(1)
        S_eq &= torch.equal(o["S"][0], one_th)
        S_maxabs = max(S_maxabs, maxabs(o["S"][0], one_th))
        S_maxrel = max(S_maxrel, maxabs(o["S"][0], one_th) / o["t_hat"].abs().sum(1).max().item())
        # element-wise product then sum (same reduction as 1^T t_hat) is bit-identical since u* == 1.0 exactly
        S_elem_eq &= torch.equal((o["u_star"][0][None, :] * o["t_hat"]).sum(1) + (o["u_hat"][0][None, :] * o["t_star"]).sum(1), one_th)
        # (b) cold items: ID row dropped, words present
        ob = merged_forward(p, data, V, V, item_ids=False)
        min_q = min(min_q, ob["q_i"].norm(dim=1).min().item() / p["E_I"][:V].norm(dim=1).mean().item())
        min_spread = min(min_spread, (ob["t_star"].max(1).values - ob["t_star"].min(1).values).min().item())
        min_th = min(min_th, (ob["t_hat"].norm(dim=1) / (p["W_t"].norm() * ob["q_i"].norm(dim=1))).min().item())
        d2 = Data(seed=seed, **cfg)
        d2.item_word_mat[1] = d2.item_word_mat[0]
        d2.item_words[1] = list(d2.item_words[0])
        ob2 = merged_forward(p, d2, V, V, item_ids=False)
        twins_eq &= torch.equal(ob2["t_star"][0], ob2["t_star"][1]) and torch.equal(ob2["t_hat"][0], ob2["t_hat"][1])
        twins_maxabs = max(twins_maxabs, maxabs(ob2["t_star"][0], ob2["t_star"][1]), maxabs(ob2["t_hat"][0], ob2["t_hat"][1]))
        n += 1
    log(f"- (a) cold user (empty basket, no ID row), {n} instances: q_u == 0 exactly: {q0_ok}; u* == 1 exactly (torch.equal): {us_ok}; u_hat == 0 exactly: {uh_ok}")
    log(f"- (a) S(u,i) == 1^T t_hat_i: via the matmul score path torch.equal = {S_eq}, max abs diff {fmt(S_maxabs)}, max diff / sum_l |t_hat_l| = {fmt(S_maxrel)} "
        f"(machine precision; bitwise identity fails only because the matmul reduces in a different order than .sum); "
        f"via element-wise product + .sum torch.equal = {S_elem_eq}")
    log(f"- (b) cold items (no ID row, words present): min ||q_i||/mean||E_I word row|| = {fmt(min_q)}; min over items of (max_k - min_k) t*_k = {fmt(min_spread)}; "
        f"min ||t_hat||/(||W_t|| ||q_i||) = {fmt(min_th)}")
    log(f"- (b) twin cold items with identical word sets: identical t* and t_hat (torch.equal) = {twins_eq}; max abs diff {fmt(twins_maxabs)}")
    ok = q0_ok and us_ok and uh_ok and S_maxrel < TOL_MP and S_elem_eq and min_q > 1e-3 and min_spread > 1e-6 and min_th > 1e-6 and twins_eq
    v = ("CONFIRMED" + ("" if S_eq else " (S = 1^T t_hat to machine precision, 1e-16 relative; not bitwise through the matmul path)")) if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c12():
    log("## C12 -- bag-size channel (statistical trend)")
    rng = np.random.default_rng(777)
    n_max, trials, M, Fw = 10, 20000, 12, 3
    all_mono = True
    # paired design: per trial one random unit-row table E_U, M items with F distinct words each, one random item
    # order; the basket of size n is the first n items of that order (nested), so norms across n share the randomness
    for V, d in [(8, 3), (50, 5)]:
        E = rng.standard_normal((trials, V, d))
        E /= np.linalg.norm(E, axis=-1, keepdims=True)                        # equal-norm (unit) rows, random directions
        words = np.argsort(rng.random((trials, M, V)), axis=-1)[:, :, :Fw]    # F distinct words per item
        tri = np.arange(trials)[:, None, None]
        wsum = E[tri, words].sum(axis=2)                                        # (trials, M, d) word sums per item
        order = np.argsort(rng.random((trials, M)), axis=-1)                    # random item order per trial
        wsum = np.take_along_axis(wsum, order[:, :, None], axis=1)
        csum = np.cumsum(wsum, axis=1)[:, :n_max]                               # (trials, n_max, d) sum over first n items
        means_vec = csum / np.arange(1, n_max + 1)[None, :, None]
        norms = np.linalg.norm(means_vec, axis=-1)                              # (trials, n_max)
        means = norms.mean(0)
        sems = norms.std(0) / math.sqrt(trials)
        mono = all(means[k + 1] < means[k] for k in range(n_max - 1))
        all_mono &= mono
        log(f"  V={V}, d={d}, F={Fw}, {trials} paired baskets: E||basket mean|| for n=1..{n_max} = {' '.join(f'{x:.3f}' for x in means)} "
            f"(s.e. <= {sems.max():.4f}) | strictly decreasing: {mono}")
    # u* invariance under positive rescaling; ||u_hat|| linear in ||q_u||
    worst_star, worst_lin = 0.0, 0.0
    for cfg, seed, sc in instances():
        data = Data(seed=seed, **cfg)
        p = params_from_cfg(cfg, seed, sc)
        V = cfg["V"]
        o = merged_forward(p, data, V, V)
        for c in [1e-3, 0.5, 3.0, 1e3]:
            us_c, uh_c = units(c * o["q_u"], c * o["q_u"], p["P_u"], p["W_u"])
            worst_star = max(worst_star, maxabs(us_c, o["u_star"]))
            worst_lin = max(worst_lin, maxrel(uh_c.norm(dim=1), c * o["u_hat"].norm(dim=1)))
    log(f"- u*(c q_u) vs u*(q_u), c in {{1e-3,0.5,3,1e3}}: max abs diff {fmt(worst_star)}; ||u_hat(c q)|| vs c ||u_hat(q)||: max rel err {fmt(worst_lin)}")
    ok = all_mono and worst_star < TOL_FLOAT and worst_lin < TOL_MP
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def c13():
    log("## C13 -- collapsed function class")
    worst = 0.0
    n = 0
    for cfg, seed, sc in instances():
        for item_ids, user_ids in [(True, True), (False, False)]:
            data = Data(seed=seed, **cfg)
            p = params_from_cfg(cfg, seed, sc)
            V, K_u, K_t = cfg["V"], cfg["K_u"], cfg["K_t"]
            o = merged_forward(p, data, V, V, item_ids, user_ids)
            Wt, Wu, Pu, Pt = p["W_t"], p["W_u"], p["P_u"], p["P_t"]
            Pu_hat, Pt_hat = Pu / Pu.norm(dim=1, keepdim=True), Pt / Pt.norm(dim=1, keepdim=True)
            qu_hat, qi_hat = normalise(o["q_u"]), normalise(o["q_i"])
            t1 = o["q_i"] @ (Wt.T @ torch.ones(K_u))                     # (M,)
            t2 = qu_hat @ (Pu_hat.T @ Wt) @ o["q_i"].T                    # (N,M)
            t3 = o["q_u"] @ (Wu.T @ torch.ones(K_t))                     # (N,)
            t4 = (qi_hat @ (Pt_hat.T @ Wu) @ o["q_u"].T).T                # (N,M)
            S13 = t1[None, :] + t2 + t3[:, None] + t4
            worst = max(worst, maxrel(o["S"], S13))
            n += 1
    log(f"- S = (W_t^T 1).q_i + qhat_u^T (Phat_u^T W_t) q_i + (W_u^T 1).q_u + qhat_i^T (Phat_t^T W_u) q_u: max rel err {fmt(worst)} over {n} instances (ids and noid arms)")
    ok = worst < TOL_MP
    v = "CONFIRMED" if ok else "FALSIFIED"
    log(f"- **Verdict: {v}**")
    return v


def procrustes_R(EU, EI):
    """R orthogonal (d x d) minimising sum_f ||EU[f] - R EI[f]||^2  (rows: EU ~ EI R^T)."""
    Uu, _, Vh = torch.linalg.svd(EI.T @ EU)
    return (Uu @ Vh).T


def L_align(EU, EI, R):
    return ((EU - EI @ R.T) ** 2).sum()


def c14():
    log("## C14 -- K10 Procrustes alignment penalty (interpretation: R in O(d) acts on the embedding dimension, row-wise)")
    zero_max, rand_min, grad_min_U, grad_min_I, grad_norm_err = 0.0, float("inf"), float("inf"), float("inf"), 0.0
    for seed in SEEDS:
        for V, d in [(6, 3), (8, 5), (5, 4)]:
            for scale in [1e-3, 1.0, 1e3]:
                g = torch.Generator().manual_seed(600 + seed)
                EI = torch.randn(V, d, generator=g) * scale
                Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
                for det_sign in [1.0, -1.0]:
                    R0 = Q.clone()
                    if torch.det(R0).sign().item() != det_sign:
                        R0[:, 0] = -R0[:, 0]
                    EU = EI @ R0.T
                    R = procrustes_R(EU, EI)
                    zero_max = max(zero_max, (L_align(EU, EI, R) / (EI ** 2).sum()).item())
                # random (non-orthogonally-related) tables
                EU_r = torch.randn(V, d, generator=g) * scale
                R = procrustes_R(EU_r, EI)
                L = L_align(EU_r, EI, R)
                rand_min = min(rand_min, (L / (EI ** 2).sum()).item())
                # (b) gradients with R constant
                EUg, EIg = EU_r.clone().requires_grad_(), EI.clone().requires_grad_()
                Lg = L_align(EUg, EIg, R.detach())
                gU, gI = torch.autograd.grad(Lg, [EUg, EIg])
                grad_min_U = min(grad_min_U, (gU.norm() / (2 * Lg.sqrt())).item())
                grad_min_I = min(grad_min_I, (gI.norm() / (2 * Lg.sqrt())).item())
                grad_norm_err = max(grad_norm_err, abs((gU.norm() / (2 * Lg.sqrt())).item() - 1), abs((gI.norm() / (2 * Lg.sqrt())).item() - 1))
    log(f"- (a) E_U = R0 E_I (R0 random orthogonal, det +1 and -1), 6 seeds x 3 shapes x 3 scales: max L_align/||E_I||^2 = {fmt(zero_max)} (zero to machine precision)")
    log(f"- (a) random unrelated tables: min L_align/||E_I||^2 = {fmt(rand_min)} (> 0); the converse (L=0 => E_U = R E_I) is definitional with R the Procrustes minimiser")
    log(f"- (b) R held constant: min ||grad_E_U||/(2 sqrt L) = {fmt(grad_min_U)}, min ||grad_E_I||/(2 sqrt L) = {fmt(grad_min_I)}; both equal 2 sqrt(L) to rel err {fmt(grad_norm_err)} -> nonzero whenever L > 0")
    # (c) scaling
    rows = []
    for seed in SEEDS[:3]:
        g = torch.Generator().manual_seed(700 + seed)
        V, d = 6, 3 + seed
        EI = torch.randn(V, d, generator=g)
        for c in [1e-3, 0.5, 0.999, 1.001, 2.0, 1e3, -0.5, -1.0, -2.0]:
            EU = c * EI
            R = procrustes_R(EU, EI)
            L = L_align(EU, EI, R).item()
            rows.append((seed, d, c, L / (EI ** 2).sum().item(), (abs(c) - 1) ** 2))
    log("- (c) E_U = c * E_I (otherwise identical): L_align/||E_I||^2 per (seed, d, c), predicted (|c|-1)^2:")
    for r in rows:
        log(f"    seed {r[0]} d={r[1]} c={r[2]:+.4g}: L/||E_I||^2 = {r[3]:.6e}  predicted (|c|-1)^2 = {r[4]:.6e}")
    pos_ok = all(r[3] > 1e-8 for r in rows if r[2] > 0 and r[2] != 1.0)
    neg_one_zero = all(r[3] < 1e-12 for r in rows if r[2] == -1.0)
    log(f"- (c) L > 0 for every positive c != 1 tested: {pos_ok}; c = -1 gives L = 0 (R = -I is orthogonal): {neg_one_zero} -> the literal 'c != 1' wording fails at c = -1; not scale-invariant for c > 0 confirmed")
    ok = zero_max < 1e-20 and rand_min > 1e-6 and grad_min_U > 0.5 and grad_min_I > 0.5 and grad_norm_err < TOL_MP and pos_ok
    v = ("CONFIRMED for c > 0 (a, b, c); literal (c) wording FALSIFIED at c = -1 (sign flip is orthogonal)" if (ok and neg_one_zero) else ("CONFIRMED" if ok else "FALSIFIED"))
    log(f"- **Verdict: {v}**")
    return v


INTERPRETATIONS = """## Interpretations made (spec ambiguities resolved by the verifier)

1. Sampled-softmax training step: one positive per user (drawn from H_u), n_neg = 4 uniform negatives, the LOO user
   vector q_u^{LOO}(positive) scores the positive AND its negatives; loss = mean cross-entropy, target index 0.
2. C2(a) "changes the score by exactly -u*^T W_t E_I[f]" is read, per its own dash clause, as the change of the
   u*^T t_hat half (q_u fixed, u* fixed). The change of the full S is also reported (it differs, since t* moves).
3. C5 "V_items = 0": the item table has no word rows (E_I in R^{M x d}); the users still compose from the items'
   word sets through E_U's V word rows (and symmetrically for V_users = 0). The comparison model looks the free table
   up twice (host style) and composes the other side once.
4. C6 A_u = [1^T ; P_hat_u] in R^{(d+1) x K_u} is read as [1^T ; P_hat_u^T] (P_hat_u is K_u x d). For (c)/(d) the
   null space is the one of A_u at step 0, so P_u is frozen while W_t, W_u, P_t, E_I, E_U train (if P_u also trains
   the null space rotates; that drift is reported separately as informative).
5. C9(b) "free t" / "free u": a free per-item table in R^{M x K_u} / per-user table in R^{N x K_t}; both the
   user-cosine and the item-cosine variants are tested, with the sampled-softmax loss and LOO.
6. C10: twin items share word set {0,1,2} (F = 3) with m = c * m0 swept over c in [1e-3, 1e3]; e_i, e_j fixed.
7. C11(a): the cold user is user 0 with an emptied basket in the noid-user arm (q_u = 0 exactly by construction).
8. C14: R in O(d) (reflections allowed, as the SVD Procrustes minimiser U V^T yields) acting on the embedding
   dimension row-wise: L = sum_f ||E_U[f] - R E_I[f]||^2 = ||E_U - E_I R^T||_F^2.
9. "Machine precision" verdict threshold: max relative error < 1e-11 (float64); "float tolerance" for cosine
   invariances: < 1e-9 abs; bit identity: torch.equal. Scale sets mix 1e-3 and 1e3 across tables, randn gives negatives.
10. Invariance claims (C6a, C7) are measured against the rounding floor of the invariant quantity (the magnitude of the
    terms being summed, e.g. sum_k |u_hat_k||t*_k|), because with ||Delta|| = 1e-6 ||W|| the only observable change is
    S's own float64 rounding; the deviation relative to the would-be change ||Delta|| ||q|| is reported alongside.
11. Ranking comparisons (C1) ignore pairs tied to within 1e-9 max|S| (twin items with identical word sets in the noid
    arm have scores that differ only at the ULP level and can tie after a constant shift).
12. C12 uses a paired design (nested baskets of one random item order per trial, 20000 trials) so that the trend is
    not masked by Monte-Carlo noise; the claim is statistical, so sample means and their standard errors are reported.
"""


def main():
    t0 = time.time()
    verdicts = {}
    for name, fn in [("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5), ("C6", c6), ("C7", c7),
                     ("C8", c8), ("C9", c9), ("C10", c10), ("C11", c11), ("C12", c12), ("C13", c13), ("C14", c14)]:
        verdicts[name] = fn()
        log()
    runtime = time.time() - t0
    header = [
        "# dc09 claims check -- independent black-box toy verification (C1-C14)",
        "",
        f"- Script: `{os.path.join(HERE, 'check_claims.py')}` (mechanism implemented from `claims_spec.md` alone; no other repository file read)",
        "- Run command (from this directory): `source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate protomf && python check_claims.py`",
        f"- Environment: torch {torch.__version__}, numpy {np.__version__}, CPU, float64 default (float32 where C4 names it)",
        f"- Toy sizes: 4 shape configs (V 5-8, M 8-10, N 6-8, d 3-5, K_u 4-12, K_t 5-9; K_u<=d+1 and K_t=d+1 included), 6 seeds, 5 parameter scale sets (1e-3 ... 1e3, mixed), basket sizes 1-6 with forced |H|=1 users, fixed and variable word-set sizes",
        f"- Runtime: {runtime:.1f} s",
        "",
        "## Verdict summary",
        "",
        "| Claim | Verdict |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in verdicts.items()] + ["", INTERPRETATIONS, "## Per-claim measurements", ""]
    with open(os.path.join(HERE, "report.md"), "w") as fh:
        fh.write("\n".join(header + REPORT) + "\n")
    print(f"\nRuntime {runtime:.1f} s; report written to {os.path.join(HERE, 'report.md')}")


if __name__ == "__main__":
    main()
