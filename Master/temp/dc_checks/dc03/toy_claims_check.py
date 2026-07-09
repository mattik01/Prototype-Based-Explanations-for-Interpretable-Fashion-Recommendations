"""
dc03 black-box toy check of claims CL1-CL10 from claims_spec.md.

Only input: claims_spec.md. Toy dims, fixed seeds, deterministic, CPU, seconds.
Run: conda run -n protomf python toy_claims_check.py
"""

import numpy as np
import torch
import torch.nn.functional as Fn

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float64)  # tighter float tolerances for exactness checks

# ---------------- dimensions (per spec) ----------------
M, F, d, K, L_u, N_users, B, n = 40, 12, 8, 5, 4, 20, 6, 3
EPS = 1e-8

results = {}  # claim id -> (verdict, evidence string)


def cos(a, b):
    """eps-guarded cosine along last dim, broadcasting."""
    return Fn.cosine_similarity(a, b, dim=-1, eps=EPS)


def sim(a, b):
    """shifted cosine 1 + cos."""
    return 1.0 + cos(a, b)


def make_params(seed=0):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(M, F, generator=g)
    V = V / V.norm(dim=1, keepdim=True)  # frozen, rows L2-normalized
    E = torch.randn(F, d, generator=g) * 0.5
    P = torch.randn(K, d, generator=g)
    W_t = torch.randn(d, L_u, generator=g) * 0.5
    U = torch.randn(N_users, d, generator=g)
    P_u = torch.randn(L_u, d, generator=g)
    W_u = torch.randn(d, K, generator=g) * 0.5
    return V, E, P, W_t, U, P_u, W_u


def make_batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    u_idxs = torch.randint(0, N_users, (B,), generator=g)
    i_idxs = torch.randint(0, M, (B, n), generator=g)
    return u_idxs, i_idxs


def forward(V, E, P, W_t, U, P_u, W_u, u_idxs, i_idxs):
    Q = V @ E                                      # (M, d)
    q = Q[i_idxs]                                  # (B, n, d)
    t_star = sim(q.unsqueeze(2), P.view(1, 1, K, d))       # (B, n, K)
    t_hat = q @ W_t                                # (B, n, L_u)
    u = U[u_idxs]                                  # (B, d)
    u_star = sim(u.unsqueeze(1), P_u.view(1, L_u, d))      # (B, L_u)
    u_hat = u @ W_u                                # (B, K)
    term1 = (u_star.unsqueeze(1) * t_hat).sum(-1)  # (B, n)
    term2 = (u_hat.unsqueeze(1) * t_star).sum(-1)  # (B, n)
    return Q, t_star, t_hat, u_star, u_hat, term1 + term2, term1, term2


def push(Q, P):
    """no-grad push: P[k] <- Q[argmax_i cos(Q_i, P_k)]. Returns new P, indices."""
    with torch.no_grad():
        c = cos(Q.unsqueeze(1), P.unsqueeze(0))    # (M, K)
        i_star = c.argmax(dim=0)                   # (K,)
        return Q[i_star].clone(), i_star


V, E, P, W_t, U, P_u, W_u = make_params(seed=0)
u_idxs, i_idxs = make_batch(seed=1)

# ---------------- CL1: decomposition ----------------
Q, t_star, t_hat, u_star, u_hat, score, term1, term2 = forward(
    V, E, P, W_t, U, P_u, W_u, u_idxs, i_idxs)
c_contrib = u_hat.unsqueeze(1) * t_star            # (B, n, K)
max_abs_diff = (c_contrib.sum(-1) - term2).abs().max().item()
ok = max_abs_diff < 1e-12
results["CL1"] = ("confirmed" if ok else "falsified",
                  f"max |sum_k c[b,j,k] - term2[b,j]| = {max_abs_diff:.2e} over all (b,j)")

# ---------------- CL2: tie gradient through both paths ----------------
E2 = E.clone().requires_grad_(True)
q2 = (V @ E2)[i_idxs]
loss_star = sim(q2.unsqueeze(2), P.view(1, 1, K, d)).sum()
g_star = torch.autograd.grad(loss_star, E2)[0].norm().item()
E3 = E.clone().requires_grad_(True)
q3 = (V @ E3)[i_idxs]
loss_hat = (q3 @ W_t).sum()
g_hat = torch.autograd.grad(loss_hat, E3)[0].norm().item()
ok = g_star > 1e-8 and g_hat > 1e-8
results["CL2"] = ("confirmed" if ok else "falsified",
                  f"||grad_E from t_star-only|| = {g_star:.4f}, ||grad_E from t_hat-only|| = {g_hat:.4f} (both nonzero)")

# ---------------- CL3: cold item ----------------
variances = []
for s in range(20):
    g = torch.Generator().manual_seed(100 + s)
    v_new = torch.randn(F, generator=g)
    v_new = v_new / v_new.norm()
    Es = torch.randn(F, d, generator=g)
    Ps = torch.randn(K, d, generator=g)
    act = sim(v_new @ Es, Ps)                      # (K,)
    variances.append(act.var(unbiased=False).item())
min_var, mean_var = min(variances), float(np.mean(variances))
ok = min_var > 1e-3
results["CL3"] = ("confirmed" if ok else "falsified",
                  f"activation variance across K over 20 random (v_new,E,P): min = {min_var:.4f}, mean = {mean_var:.4f} (> 1e-3)")

# ---------------- CL4: zero-image neutrality ----------------
v_zero = torch.zeros(F)
act0 = sim(v_zero @ E, P)                          # (K,)
exact_one = bool((act0 == 1.0).all())
results["CL4"] = ("confirmed" if exact_one else "falsified",
                  f"activations for v=0: {act0.tolist()} (all exactly 1.0: {exact_one})")

# ---------------- CL5a: push idempotence ----------------
P1, idx1 = push(Q, P)
P2, idx2 = push(Q, P1)
diff = (P2 - P1).abs().max().item()
ok = diff == 0.0 and bool((idx1 == idx2).all())
results["CL5a"] = ("confirmed" if ok else "falsified",
                   f"max |P_after_2nd_push - P_after_1st_push| = {diff:.2e}, argmax indices identical: {bool((idx1 == idx2).all())}")

# ---------------- CL5b: push perturbation monotone in pre-push proximity ----------------
ratios = []
for s in range(5):
    g = torch.Generator().manual_seed(200 + s)
    chosen = torch.randperm(M, generator=g)[:K]
    P_near = Q[chosen] + 0.01 * torch.randn(K, d, generator=g)
    P_rand = torch.randn(K, d, generator=g)
    d_scores = []
    for P0 in (P_near, P_rand):
        s0 = forward(V, E, P0, W_t, U, P_u, W_u, u_idxs, i_idxs)[5]
        P_pushed, _ = push(Q, P0)
        s1 = forward(V, E, P_pushed, W_t, U, P_u, W_u, u_idxs, i_idxs)[5]
        d_scores.append((s1 - s0).abs().mean().item())
    ratios.append(d_scores[1] / max(d_scores[0], 1e-300))
min_ratio = min(ratios)
ok = min_ratio >= 10.0
results["CL5b"] = ("confirmed" if ok else "falsified",
                   f"mean|dScore| ratio random-P / near-P over 5 seeds: min = {min_ratio:.1f}x, all = {[f'{r:.0f}' for r in ratios]} (claim: >= 10x)")

# ---------------- CL5c: exact grounding ----------------
P_pushed, idxp = push(Q, torch.randn(K, d, generator=torch.Generator().manual_seed(7)))
all_match = all(any((P_pushed[k] == Q[i]).all() for i in range(M)) for k in range(K))
results["CL5c"] = ("confirmed" if all_match else "falsified",
                   f"every pushed prototype row bitwise-equals some row of Q: {all_match} (exemplar idx: {idxp.tolist()})")

# ---------------- CL6: R_proto pulls prototypes to batch items ----------------
Pg = torch.randn(K, d, generator=torch.Generator().manual_seed(11)).requires_grad_(True)
q_flat = Q[i_idxs].reshape(-1, d)                   # (B*n, d) batch fixed
S = sim(q_flat.unsqueeze(1), Pg.unsqueeze(0))       # (B*n, K)
R_proto = -S.max(dim=0).values.mean()
grad_P = torch.autograd.grad(R_proto, Pg)[0]
lr = 0.1
P_new = (Pg - lr * grad_P).detach()
mm_before = cos(q_flat.unsqueeze(1), Pg.detach().unsqueeze(0)).max(dim=0).values.mean().item()
mm_after = cos(q_flat.unsqueeze(1), P_new.unsqueeze(0)).max(dim=0).values.mean().item()
ok = mm_after > mm_before
results["CL6"] = ("confirmed" if ok else "falsified",
                  f"mean_k max_j cos(q_j, P_k): {mm_before:.6f} -> {mm_after:.6f} after one GD step (lr=0.1), strictly increased: {ok}")


# ---------------- CL7: L_div behavior ----------------
def L_div(Pm, tau):
    Cm = cos(Pm.unsqueeze(1), Pm.unsqueeze(0))      # (K, K)
    iu, ju = torch.triu_indices(Pm.shape[0], Pm.shape[0], offset=1)
    pair = Cm[iu, ju]
    Kp = Pm.shape[0]
    return (2.0 / (Kp * (Kp - 1))) * torch.clamp(pair - tau, min=0).sum(), pair


tau = 0.5
# (a) zero when all pairwise cosines <= tau: orthogonal prototypes (cos = 0)
P_orth = torch.zeros(K, d)
for k in range(K):
    P_orth[k, k] = 1.0
val_zero, _ = L_div(P_orth, tau)
a_ok = val_zero.item() == 0.0
# (c) gradient exactly 0 when all pairwise cosines <= tau (strictly below)
P_orth_g = P_orth.clone().requires_grad_(True)
val, _ = L_div(P_orth_g, tau)
grad_orth = torch.autograd.grad(val, P_orth_g)[0]
c_ok = grad_orth.abs().max().item() == 0.0
# (b) two identical prototypes -> one gradient step strictly decreases their cosine
P_id = torch.randn(K, d, generator=torch.Generator().manual_seed(13))
P_id[1] = P_id[0].clone()                            # exactly identical pair
P_id_g = P_id.clone().requires_grad_(True)
val, pair = L_div(P_id_g, tau)
cos01_before = cos(P_id[0], P_id[1]).item()
grad_id = torch.autograd.grad(val, P_id_g)[0]
grad_pair_norm = grad_id[:2].norm().item()
P_id_new = P_id - 0.5 * grad_id
cos01_after = cos(P_id_new[0], P_id_new[1]).item()
b_ok = cos01_after < cos01_before
# nearby interpretation: nearly identical pair (cos < 1)
P_near = P_id.clone()
P_near[1] = P_id[0] + 0.05 * torch.randn(d, generator=torch.Generator().manual_seed(14))
P_near_g = P_near.clone().requires_grad_(True)
val_n, _ = L_div(P_near_g, tau)
cosn_before = cos(P_near[0], P_near[1]).item()
grad_n = torch.autograd.grad(val_n, P_near_g)[0]
P_near_new = P_near - 0.5 * grad_n
cosn_after = cos(P_near_new[0], P_near_new[1]).item()
near_ok = cosn_after < cosn_before
verdict7 = "confirmed" if (a_ok and b_ok and c_ok) else "falsified"
results["CL7"] = (verdict7,
                  f"L_div(orth)={val_zero.item():.1e} (==0: {a_ok}); grad(all-below-tau) max|.|={grad_orth.abs().max().item():.1e} (==0: {c_ok}); "
                  f"identical pair: cos {cos01_before:.10f} -> {cos01_after:.10f} (strict decrease: {b_ok}, ||grad on pair||={grad_pair_norm:.2e}); "
                  f"NEARBY (cos={cosn_before:.4f} pair): -> {cosn_after:.4f} (decrease: {near_ok})")

# ---------------- CL8: L_orth counteracts collapse ----------------
def run_orth_steps(E0, steps=50, lr=0.005):
    Ee = E0.clone().requires_grad_(True)
    for _ in range(steps):
        G = Ee.t() @ Ee - torch.eye(d)
        loss = (G ** 2).sum()
        g_ = torch.autograd.grad(loss, Ee)[0]
        Ee = (Ee - lr * g_).detach().requires_grad_(True)
    return Ee.detach()


def rank_proxies(Emat):
    sv = np.linalg.svd(Emat.numpy(), compute_uv=False)
    p = sv / sv.sum()
    p = p[p > 0]
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    return float(sv[-1]), eff_rank


col = torch.randn(F, 1, generator=torch.Generator().manual_seed(21))
col = col / col.norm()                               # unit-norm column, keeps L_orth GD stable
E_collapsed = col.repeat(1, d)                       # literal: all columns equal
smin0, er0 = rank_proxies(E_collapsed)
E_after = run_orth_steps(E_collapsed)
smin1, er1 = rank_proxies(E_after)
# meaningful (above machine-noise) increase: sigma_min relative to s1 must exceed
# float64 epsilon scale, and effective rank must move by more than round-off.
s1_after = float(np.linalg.svd(E_after.numpy(), compute_uv=False)[0])
literal_ok = (smin1 / s1_after > 1e-10) and (er1 - er0 > 1e-6)
# nearby interpretation: near-collapse (tiny noise)
E_nearc = E_collapsed + 1e-3 * torch.randn(F, d, generator=torch.Generator().manual_seed(22))
smin0n, er0n = rank_proxies(E_nearc)
E_after_n = run_orth_steps(E_nearc)
smin1n, er1n = rank_proxies(E_after_n)
near_ok8 = (smin1n > smin0n) and (er1n > er0n)
results["CL8"] = ("confirmed" if literal_ok else "falsified",
                  f"exact collapse (all cols equal): sigma_min {smin0:.2e} -> {smin1:.2e} (round-off only, sigma_min/s1 = {smin1/s1_after:.1e}), "
                  f"eff-rank {er0:.10f} -> {er1:.10f} (meaningful increase: {literal_ok}); "
                  f"NEARBY near-collapse (+1e-3 noise): sigma_min {smin0n:.2e} -> {smin1n:.2e}, eff-rank {er0n:.2f} -> {er1n:.2f} (increase: {near_ok8})")

# ---------------- CL9: cosine-argmax vs L2-argmin ----------------
# (a) exhibit a divergent configuration with unnormalized q
q_ex = torch.zeros(2, d)
q_ex[0, 0] = 10.0                                    # same direction as p, far in L2
q_ex[1, 0], q_ex[1, 1] = 1.0, 0.5                    # closer in L2, lower cosine
p_ex = torch.zeros(d)
p_ex[0] = 1.0
cos_ex = cos(q_ex, p_ex.unsqueeze(0).expand(2, d))
l2_ex = (q_ex - p_ex).norm(dim=1)
diverge = int(cos_ex.argmax()) != int(l2_ex.argmin())
# (b) equal-norm data: selections coincide for all prototypes
g = torch.Generator().manual_seed(31)
Q_eq = torch.randn(M, d, generator=g)
Q_eq = 3.7 * Q_eq / Q_eq.norm(dim=1, keepdim=True)   # all rows same norm 3.7
P_test = torch.randn(50, d, generator=g)
cos_mat = cos(Q_eq.unsqueeze(1), P_test.unsqueeze(0))            # (M, 50)
l2_mat = (Q_eq.unsqueeze(1) - P_test.unsqueeze(0)).norm(dim=-1)  # (M, 50)
coincide = bool((cos_mat.argmax(dim=0) == l2_mat.argmin(dim=0)).all())
ok = diverge and coincide
results["CL9"] = ("confirmed" if ok else "falsified",
                  f"exhibit: cos-argmax=item{int(cos_ex.argmax())} (cos {cos_ex[0]:.3f} vs {cos_ex[1]:.3f}), L2-argmin=item{int(l2_ex.argmin())} "
                  f"(dist {l2_ex[0]:.2f} vs {l2_ex[1]:.2f}) -> differ: {diverge}; equal-norm: argmaxes coincide for all 50 prototypes: {coincide}")


# ---------------- CL10: greedy unique push ----------------
def greedy_unique_push(Qm, Pm):
    with torch.no_grad():
        c = cos(Qm.unsqueeze(1), Pm.unsqueeze(0))    # (M, K)
        Kp = Pm.shape[0]
        order = torch.argsort(c.max(dim=0).values, descending=True)  # prototypes by best cosine
        taken, assign = set(), {}
        for k in order.tolist():
            ranked = torch.argsort(c[:, k], descending=True).tolist()
            for i in ranked:
                if i not in taken:
                    taken.add(i)
                    assign[k] = i
                    break
        return [assign[k] for k in range(Kp)]


all_unique = True
worst_case_note = ""
for s in range(10):
    g = torch.Generator().manual_seed(300 + s)
    Q_r = torch.randn(M, d, generator=g) @ torch.randn(d, d, generator=g)
    P_r = torch.randn(K, d, generator=g)
    if s >= 5:
        P_r = P_r[0:1].repeat(K, 1)                  # stress: all prototypes identical
    idxs = greedy_unique_push(Q_r, P_r)
    if len(set(idxs)) != K:
        all_unique = False
        worst_case_note = f" (failed at seed {300+s}: {idxs})"
# extra stress: M == K
Q_min = torch.randn(K, d, generator=torch.Generator().manual_seed(400))
idxs_min = greedy_unique_push(Q_min, torch.randn(K, d, generator=torch.Generator().manual_seed(401)))
mk_ok = len(set(idxs_min)) == K
ok = all_unique and mk_ok
results["CL10"] = ("confirmed" if ok else "falsified",
                   f"10 trials (incl. 5 with all-identical prototypes): all yielded K={K} distinct indices: {all_unique}{worst_case_note}; "
                   f"M==K edge case distinct: {mk_ok} ({sorted(idxs_min)})")

# ---------------- report ----------------
print("=" * 100)
print(f"{'claim':6s} {'verdict':12s} evidence")
print("-" * 100)
for cid in ["CL1", "CL2", "CL3", "CL4", "CL5a", "CL5b", "CL5c", "CL6", "CL7", "CL8", "CL9", "CL10"]:
    v, e = results[cid]
    print(f"{cid:6s} {v:12s} {e}")
print("=" * 100)
