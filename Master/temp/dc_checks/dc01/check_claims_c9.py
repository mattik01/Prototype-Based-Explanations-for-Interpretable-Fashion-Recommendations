"""
C9 — user-side gauge checks for the fI model (SC.1b-delta re-check, 2026-07-12).

Implements ONLY the fI mechanism from the spec's "fI re-check / Mechanism delta":
  - item i's composed embedding q_i = sum of its F+1 rows of E  (F feature rows + 1 ID row)
  - item prototypes P_t in R^{K_t x d}; tstar_k = 1 + cos(q_i, P_t[k])
  - user side: free vector u = U[u] in R^{K_t} (no P_u, no W_u/W_t)
  - score S(u,i) = u . tstar

Gauge space: G(Phat) = { w in R^{K_t} : Phat^T w = 0 and 1^T w = 0 },
Phat = row-normalized P_t. dim G >= K_t - d - 1, so toys use K_t > d+1.

Claims:
  C9(a)  exact score invariance S(u+w,i) = S(u,i) for all w in G(Phat)
  C9(b1) gauge component of u frozen under SGD training (P_t, E frozen)
  C9(b2) everything trainable: per-step loss gradient wrt each user vector
         is orthogonal to the CURRENT gauge space
  C9(c)  with SGD weight_decay=wd, gauge component decays exactly by
         (1 - lr*wd)^T after T steps

float64 throughout. Shapes (K_t=8, d=3) and (K_t=12, d=5).
Seeds: 5 for (a), 3 for (b1)/(b2)/(c).
"""

import math
import torch

torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------------
# fI model utilities
# ----------------------------------------------------------------------------

def make_params(seed, N, M, V, F, d, K_t, scale=1.0):
    """Random fI parameters + random item->rows assignment."""
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(V + M, d, generator=g) * scale        # feature/ID table
    U = torch.randn(N, K_t, generator=g) * scale          # free user vectors
    P_t = torch.randn(K_t, d, generator=g) * scale        # item prototypes
    # item i: F random feature-value rows (ids in [0, V)) + its ID row (V + i)
    feat = torch.randint(0, V, (M, F), generator=g) if F > 0 \
        else torch.zeros(M, 0, dtype=torch.long)
    ids = torch.arange(M).unsqueeze(1) + V
    item_rows = torch.cat([feat, ids], dim=1)             # M x (F+1)
    return E, U, P_t, item_rows


def tstar_of(E, P_t, item_rows, items):
    """tstar for a batch of item indices. items: LongTensor (B,) -> (B, K_t)."""
    q = E[item_rows[items]].sum(dim=1)                    # B x d
    qn = q.norm(dim=1, keepdim=True)                      # B x 1
    pn = P_t.norm(dim=1)                                  # K_t
    cos = (q @ P_t.t()) / (qn * pn.unsqueeze(0))
    return 1.0 + cos


def score_of(E, U, P_t, item_rows, users, items):
    return (U[users] * tstar_of(E, P_t, item_rows, items)).sum(dim=1)


def gauge_basis(P_t):
    """Orthonormal basis (columns) of G(Phat) via SVD null space of the
    (d+1) x K_t matrix stacking Phat^T and 1^T."""
    K_t, d = P_t.shape
    Phat = P_t / P_t.norm(dim=1, keepdim=True)
    A = torch.cat([Phat.t(), torch.ones(1, K_t)], dim=0)  # (d+1) x K_t
    try:
        _, S, Vh = torch.linalg.svd(A, full_matrices=True)
        V = Vh.t()
    except AttributeError:  # older torch
        _, S, V = torch.svd(A, some=False)
    tol = S.max() * max(A.shape) * torch.finfo(torch.float64).eps
    rank = int((S > tol).sum())
    W = V[:, rank:]                                       # K_t x m, orthonormal
    return W


def make_dataset(seed, N, M, n_pairs):
    """Fixed (user, item) pairs with fixed binary labels; every user covered."""
    g = torch.Generator().manual_seed(seed + 10_000)
    users = torch.cat([torch.arange(N),
                       torch.randint(0, N, (n_pairs - N,), generator=g)])
    items = torch.randint(0, M, (n_pairs,), generator=g)
    labels = torch.randint(0, 2, (n_pairs,), generator=g).to(torch.float64)
    return users, items, labels


def bce_loss(E, U, P_t, item_rows, users, items, labels):
    s = score_of(E, U, P_t, item_rows, users, items)
    return torch.nn.functional.binary_cross_entropy_with_logits(s, labels)


SHAPES = [  # (K_t, d, F, V, N, M)  — K_t > d+1 in both
    (8, 3, 3, 9, 6, 10),
    (12, 5, 4, 12, 7, 12),
]

# ----------------------------------------------------------------------------
# C9(a) — exact score invariance under gauge shifts
# ----------------------------------------------------------------------------

def check_c9a():
    print("=" * 72)
    print("C9(a): S(u+w, i) = S(u, i) exactly for w in G(Phat)")
    worst = 0.0
    worst_cfg = None
    n_checks = 0
    for (K_t, d, F, V, N, M) in SHAPES:
        for seed in range(5):
            for scale in (0.1, 1.0, 10.0):
                E, U, P_t, item_rows = make_params(seed, N, M, V, F, d, K_t,
                                                   scale=scale)
                W = gauge_basis(P_t)
                m = W.shape[1]
                assert m >= K_t - d - 1, "gauge space unexpectedly small"
                g = torch.Generator().manual_seed(seed + 777)
                users = torch.arange(N).repeat(3)
                items = torch.randint(0, M, (users.numel(),), generator=g)
                s0 = score_of(E, U, P_t, item_rows, users, items)
                for wmag in (0.5, 5.0, 100.0):
                    coeff = torch.randn(m, generator=g)
                    w = (W @ coeff)
                    w = w / w.norm() * wmag
                    U2 = U.clone()
                    U2[users] = U2[users] + w.unsqueeze(0)
                    s1 = score_of(E, U2, P_t, item_rows, users, items)
                    # tolerance scaled by the size of the cancelled term w.tstar
                    tst = tstar_of(E, P_t, item_rows, items)
                    denom = (wmag * tst.norm(dim=1)).clamp(min=1.0)
                    rel = ((s1 - s0).abs() / denom).max().item()
                    n_checks += s0.numel()
                    if rel > worst:
                        worst = rel
                        worst_cfg = (K_t, d, seed, scale, wmag)
    print(f"  checks: {n_checks} scored pairs "
          f"(2 shapes x 5 seeds x 3 param scales x 3 gauge magnitudes)")
    print(f"  max relative error |S(u+w,i)-S(u,i)| / (|w||tstar|): "
          f"{worst:.3e}  at (K_t,d,seed,scale,|w|)={worst_cfg}")
    ok = worst < 1e-12
    print(f"  VERDICT: {'CONFIRMED' if ok else 'FALSIFIED'}")
    return ok, worst


# ----------------------------------------------------------------------------
# C9(b1) — gauge component frozen when P_t (and E) are frozen
# ----------------------------------------------------------------------------

def check_c9b1(train_E=False):
    tag = "U+E trainable" if train_E else "U only trainable"
    print("=" * 72)
    print(f"C9(b1): gauge component of u frozen under SGD (P_t frozen; {tag})")
    worst_dev = 0.0
    min_inspan_move = math.inf
    worst_cfg = None
    for (K_t, d, F, V, N, M) in SHAPES:
        for seed in range(3):
            E, U, P_t, item_rows = make_params(seed, N, M, V, F, d, K_t)
            E = E.clone().requires_grad_(train_E)
            U = U.clone().requires_grad_(True)
            P_t = P_t.clone()  # frozen
            users, items, labels = make_dataset(seed, N, M, n_pairs=40)
            W = gauge_basis(P_t)                     # fixed: P_t frozen
            gauge0 = U.detach() @ W                  # N x m initial coeffs
            inspan0 = U.detach() - (U.detach() @ W) @ W.t()
            params = [U, E] if train_E else [U]
            opt = torch.optim.SGD(params, lr=0.05)
            T = 300
            for t in range(T):
                opt.zero_grad()
                loss = bce_loss(E, U, P_t, item_rows, users, items, labels)
                loss.backward()
                opt.step()
                dev = ((U.detach() @ W) - gauge0).abs().max().item()
                if dev > worst_dev:
                    worst_dev = dev
                    worst_cfg = (K_t, d, seed, t)
            inspanT = U.detach() - (U.detach() @ W) @ W.t()
            move = (inspanT - inspan0).norm(dim=1).max().item()
            min_inspan_move = min(min_inspan_move, move)
    print(f"  runs: 2 shapes x 3 seeds, T=300 SGD steps, lr=0.05, BCE loss")
    print(f"  max |gauge coeff drift| over ALL steps/users: {worst_dev:.3e} "
          f"at (K_t,d,seed,step)={worst_cfg}")
    print(f"  min (over runs) max-user in-span movement ||Δu_inspan||: "
          f"{min_inspan_move:.3e}")
    ok = worst_dev < 1e-11 and min_inspan_move > 1e-3
    print(f"  VERDICT: {'CONFIRMED' if ok else 'FALSIFIED'}")
    return ok, worst_dev, min_inspan_move


# ----------------------------------------------------------------------------
# C9(b2) — everything trainable: per-step grad wrt U orthogonal to CURRENT G
# ----------------------------------------------------------------------------

def check_c9b2():
    print("=" * 72)
    print("C9(b2): everything trainable — grad_U(loss) ⊥ G(Phat_current) "
          "at every step")
    worst = 0.0
    worst_cfg = None
    for (K_t, d, F, V, N, M) in SHAPES:
        for seed in range(3):
            E, U, P_t, item_rows = make_params(seed, N, M, V, F, d, K_t)
            E = E.clone().requires_grad_(True)
            U = U.clone().requires_grad_(True)
            P_t = P_t.clone().requires_grad_(True)
            users, items, labels = make_dataset(seed, N, M, n_pairs=40)
            opt = torch.optim.SGD([E, U, P_t], lr=0.05)
            T = 300
            for t in range(T):
                opt.zero_grad()
                loss = bce_loss(E, U, P_t, item_rows, users, items, labels)
                loss.backward()
                W = gauge_basis(P_t.detach())        # CURRENT gauge space
                gnorm = U.grad.norm().clamp(min=1.0)
                proj = ((U.grad @ W).abs().max() / gnorm).item()
                if proj > worst:
                    worst = proj
                    worst_cfg = (K_t, d, seed, t)
                opt.step()
    print(f"  runs: 2 shapes x 3 seeds, T=300 steps; gauge basis recomputed "
          f"per step from current P_t")
    print(f"  max_t max_{{u,w}} |g^T w| / ||g||: {worst:.3e} "
          f"at (K_t,d,seed,step)={worst_cfg}")
    ok = worst < 1e-12
    print(f"  VERDICT: {'CONFIRMED' if ok else 'FALSIFIED'}")
    return ok, worst


# ----------------------------------------------------------------------------
# C9(c) — L2 canonicalization: gauge component decays as (1 - lr*wd)^T
# ----------------------------------------------------------------------------

def check_c9c():
    print("=" * 72)
    print("C9(c): with weight_decay=wd, gauge coeffs of u decay exactly by "
          "(1 - lr*wd)^T")
    lr, wd = 0.05, 0.01
    worst_rel = 0.0
    worst_cfg = None
    final_ratio_report = None
    for (K_t, d, F, V, N, M) in SHAPES:
        for seed in range(3):
            E, U, P_t, item_rows = make_params(seed, N, M, V, F, d, K_t)
            U = U.clone().requires_grad_(True)
            users, items, labels = make_dataset(seed, N, M, n_pairs=40)
            W = gauge_basis(P_t)
            gauge0 = U.detach() @ W
            opt = torch.optim.SGD([U], lr=lr, weight_decay=wd)
            T = 300
            for t in range(1, T + 1):
                opt.zero_grad()
                loss = bce_loss(E, U, P_t, item_rows, users, items, labels)
                loss.backward()
                opt.step()
                predicted = gauge0 * (1.0 - lr * wd) ** t
                dev = (U.detach() @ W - predicted).abs().max().item()
                rel = dev / gauge0.abs().max().item()
                if rel > worst_rel:
                    worst_rel = rel
                    worst_cfg = (K_t, d, seed, t)
            # decay-ratio evidence at final step
            gT = U.detach() @ W
            mask = gauge0.abs() > 1e-8
            emp = (gT[mask] / gauge0[mask])
            theo = (1.0 - lr * wd) ** T
            final_ratio_report = (emp.min().item(), emp.max().item(), theo)
    emp_min, emp_max, theo = final_ratio_report
    print(f"  runs: 2 shapes x 3 seeds, T=300 steps, lr={lr}, wd={wd} "
          f"(theoretical factor (1-lr*wd)^300 = {theo:.12f})")
    print(f"  max relative deviation |gauge_t - gauge_0*(1-lr*wd)^t| / "
          f"max|gauge_0| over ALL steps: {worst_rel:.3e} "
          f"at (K_t,d,seed,step)={worst_cfg}")
    print(f"  empirical per-coeff decay ratio at T (last run): "
          f"[{emp_min:.12f}, {emp_max:.12f}] vs theoretical {theo:.12f}")
    ok = worst_rel < 1e-11
    print(f"  VERDICT: {'CONFIRMED' if ok else 'FALSIFIED'}")
    return ok, worst_rel


if __name__ == "__main__":
    torch.manual_seed(0)
    r_a = check_c9a()
    r_b1 = check_c9b1(train_E=False)
    r_b1e = check_c9b1(train_E=True)
    r_b2 = check_c9b2()
    r_c = check_c9c()
    print("=" * 72)
    print("SUMMARY")
    print(f"  C9(a) : {'CONFIRMED' if r_a[0] else 'FALSIFIED'}")
    print(f"  C9(b1): {'CONFIRMED' if (r_b1[0] and r_b1e[0]) else 'FALSIFIED'}"
          f"  (U-only and U+E variants)")
    print(f"  C9(b2): {'CONFIRMED' if r_b2[0] else 'FALSIFIED'}")
    print(f"  C9(c) : {'CONFIRMED' if r_c[0] else 'FALSIFIED'}")
