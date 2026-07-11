"""SC.2 amendment re-check for dc01 claims spec: C7a, C7b, C8.

Implements ONLY the mechanism defined in claims_spec.md (this directory).
All algebraic checks in float64. Runtime: seconds.

Run:
    conda activate protomf
    python check_claims_sc2.py
"""

import math

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

TOL = 1e-12


# ---------------------------------------------------------------------------
# Mechanism (exactly as in the spec)
# ---------------------------------------------------------------------------

def shifted_cos(x, P):
    """x: (d,), P: (K,d) -> (K,) of 1 + cos(x, P[k])."""
    return 1.0 + (P @ x) / (x.norm() * P.norm(dim=1))


def forward_pair(u_vec, q, P_u, P_t, W_u, W_t):
    ustar = shifted_cos(u_vec, P_u)          # (K_u,)
    tstar = shifted_cos(q, P_t)              # (K_t,)
    uhat = W_u @ u_vec                       # (K_t,)
    that = W_t @ q                           # (K_u,)
    S = ustar @ that + uhat @ tstar
    return ustar, tstar, uhat, that, S


# ---------------------------------------------------------------------------
# C7a / C7b — algebraic checks, adversarial sweep
# ---------------------------------------------------------------------------

def check_c7():
    max_err_c7a = 0.0
    max_relerr_c7a = 0.0
    max_err_c7b_lin = 0.0
    max_relerr_c7b_lin = 0.0
    # cosine half: verify c'_{r,k} = c_{r,k} * (||q||/||q'||) exactly, and that
    # the common factor is generically != 1 (shares DO change on removal)
    max_err_c7b_cos_ratio = 0.0
    min_factor_dev = math.inf   # min over trials of |(||q||/||q'||) - 1|
    n_trials = 0

    seeds = range(10)
    F_sizes = [1, 2, 3, 5, 20]
    scales = [1e-3, 1.0, 1e3]
    dims = [(4, 1, 1), (8, 3, 4), (16, 5, 7)]   # (d, K_u, K_t)

    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        for F in F_sizes:
            for scale in scales:
                for (d, K_u, K_t) in dims:
                    n_trials += 1
                    R = F + 1  # F feature rows + 1 ID row
                    E_rows = torch.randn(R, d, generator=g) * scale
                    u_vec = torch.randn(d, generator=g) * scale
                    P_u = torch.randn(K_u, d, generator=g) * scale
                    P_t = torch.randn(K_t, d, generator=g) * scale
                    W_u = torch.randn(K_t, d, generator=g)
                    W_t = torch.randn(K_u, d, generator=g)

                    q = E_rows.sum(dim=0)
                    ustar, tstar, uhat, that, S = forward_pair(
                        u_vec, q, P_u, P_t, W_u, W_t)

                    # ---- C7a: ustar·(W_t q) == sum_r ustar·(W_t E[r]) ----
                    lhs = ustar @ that
                    per_row = torch.stack(
                        [ustar @ (W_t @ E_rows[r]) for r in range(R)])
                    rhs = per_row.sum()
                    err = (lhs - rhs).abs().item()
                    max_err_c7a = max(max_err_c7a, err)
                    denom = max(abs(lhs.item()), abs(rhs.item()), 1e-30)
                    max_relerr_c7a = max(max_relerr_c7a, err / denom)

                    # ---- C7b linear half: removal-exact ----
                    r0 = seed % R  # vary which row is removed
                    q_prime = q - E_rows[r0]
                    lhs_new = ustar @ (W_t @ q_prime)
                    predicted = lhs - per_row[r0]
                    err = (lhs_new - predicted).abs().item()
                    max_err_c7b_lin = max(max_err_c7b_lin, err)
                    denom = max(abs(lhs_new.item()), abs(predicted.item()),
                                1e-30)
                    max_relerr_c7b_lin = max(max_relerr_c7b_lin, err / denom)

                    # ---- C7b cosine half: remaining shares change through
                    #      ||q|| only, by common factor ||q||/||q'|| != 1 ----
                    # c_{r,k} before and after removal
                    c_before = (E_rows @ P_t.T) / (q.norm() * P_t.norm(dim=1))
                    c_after = (E_rows @ P_t.T) / (q_prime.norm()
                                                  * P_t.norm(dim=1))
                    keep = [r for r in range(R) if r != r0]
                    factor = (q.norm() / q_prime.norm()).item()
                    pred_after = c_before[keep] * factor
                    err = (c_after[keep] - pred_after).abs().max().item()
                    max_err_c7b_cos_ratio = max(max_err_c7b_cos_ratio, err)
                    min_factor_dev = min(min_factor_dev, abs(factor - 1.0))

    print("== C7a ==")
    print(f"  trials: {n_trials} (10 seeds x F in {F_sizes} x scales "
          f"{scales} x dims {dims})")
    print(f"  max abs error : {max_err_c7a:.3e}")
    print(f"  max rel error : {max_relerr_c7a:.3e}")
    print("== C7b ==")
    print(f"  linear half removal-exactness: max abs err "
          f"{max_err_c7b_lin:.3e}, max rel err {max_relerr_c7b_lin:.3e}")
    print(f"  cosine half: remaining shares rescale by common factor "
          f"||q||/||q'|| (max abs err vs exact rescale "
          f"{max_err_c7b_cos_ratio:.3e})")
    print(f"  min over trials of |factor - 1| : {min_factor_dev:.3e} "
          f"(shares of remaining rows DO change in every trial)")


# ---------------------------------------------------------------------------
# C8 — lockstep gauge freedom under training
# ---------------------------------------------------------------------------

# Fixed toy dataset (independent of model init seed)
N_USERS = 10
M_ITEMS = 30
V_VALS = 12          # categorical feature values 0..11; a=0, b=1 co-occur
F_FEAT = 3           # features per item (+1 ID row)
D = 8
K_U = 3
K_T = 4
A_ID, B_ID = 0, 1


def build_dataset(structured_labels=False):
    rng = np.random.default_rng(12345)
    rows = np.zeros((M_ITEMS, F_FEAT + 1), dtype=np.int64)
    carries_ab = np.zeros(M_ITEMS, dtype=bool)
    for i in range(M_ITEMS):
        if i % 2 == 0:  # half the items carry the co-occurring pair {a, b}
            other = rng.choice(np.arange(2, V_VALS), size=F_FEAT - 2,
                               replace=False)
            feat = np.concatenate(([A_ID, B_ID], other))
            carries_ab[i] = True
        else:           # the rest never carry a or b
            feat = rng.choice(np.arange(2, V_VALS), size=F_FEAT,
                              replace=False)
        rows[i, :F_FEAT] = feat
        rows[i, F_FEAT] = V_VALS + i  # ID row in shared table
    if structured_labels:
        # learnable signal: first half of users like {a,b}-items, second half
        # the others -> the pair's combined embedding is a trained quantity
        labels = np.zeros((N_USERS, M_ITEMS))
        labels[: N_USERS // 2, :] = carries_ab[None, :]
        labels[N_USERS // 2:, :] = ~carries_ab[None, :]
    else:
        labels = rng.integers(0, 2, size=(N_USERS, M_ITEMS)).astype(
            np.float64)
    return torch.from_numpy(rows), torch.from_numpy(labels), carries_ab


def init_params(seed):
    g = torch.Generator().manual_seed(seed)
    p = {
        "E": torch.randn(V_VALS + M_ITEMS, D, generator=g) * 0.5,
        "U": torch.randn(N_USERS, D, generator=g) * 0.5,
        "P_u": torch.randn(K_U, D, generator=g) * 0.5,
        "P_t": torch.randn(K_T, D, generator=g) * 0.5,
        "W_u": torch.randn(K_T, D, generator=g) * 0.5,
        "W_t": torch.randn(K_U, D, generator=g) * 0.5,
    }
    for t in p.values():
        t.requires_grad_(True)
    return p


def full_scores(p, item_rows):
    U, E = p["U"], p["E"]
    q = E[item_rows].sum(dim=1)                                    # (M, d)
    ustar = 1.0 + (U @ p["P_u"].T) / (U.norm(dim=1, keepdim=True)
                                      * p["P_u"].norm(dim=1))      # (N, K_u)
    tstar = 1.0 + (q @ p["P_t"].T) / (q.norm(dim=1, keepdim=True)
                                      * p["P_t"].norm(dim=1))      # (M, K_t)
    uhat = U @ p["W_u"].T                                          # (N, K_t)
    that = q @ p["W_t"].T                                          # (M, K_u)
    return ustar @ that.T + uhat @ tstar.T                         # (N, M)


def train(seed, steps=300, lr=0.05, weight_decay=0.0,
          structured_labels=False):
    item_rows, labels, _ = build_dataset(structured_labels)
    p = init_params(seed)
    params = list(p.values())
    opt = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss()

    diff0 = (p["E"][A_ID] - p["E"][B_ID]).detach().clone()
    sum0 = (p["E"][A_ID] + p["E"][B_ID]).detach().clone()

    max_grad_gap = 0.0      # max_t || grad E[a] - grad E[b] ||_inf
    max_diff_drift = 0.0    # max_t || (E[a]-E[b]) - diff0 ||_inf
    diff_norms = []
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(full_scores(p, item_rows), labels)
        loss.backward()
        gap = (p["E"].grad[A_ID] - p["E"].grad[B_ID]).abs().max().item()
        max_grad_gap = max(max_grad_gap, gap)
        opt.step()
        d = p["E"].detach()[A_ID] - p["E"].detach()[B_ID]
        max_diff_drift = max(max_diff_drift,
                             (d - diff0).abs().max().item())
        diff_norms.append(d.norm().item())

    sum_move = ((p["E"].detach()[A_ID] + p["E"].detach()[B_ID])
                - sum0).norm().item()
    return p, item_rows, {
        "max_grad_gap": max_grad_gap,
        "max_diff_drift": max_diff_drift,
        "diff_norm_start": diff0.norm().item(),
        "diff_norm_end": diff_norms[-1],
        "sum_move_norm": sum_move,
        "final_loss": loss.item(),
    }


def cosine_shares(p, item_rows, item_idx):
    """c_{r,k} per C1 for rows a and b of the given item. Returns (K_t,) x2."""
    with torch.no_grad():
        E, P_t = p["E"], p["P_t"]
        q = E[item_rows[item_idx]].sum(dim=0)
        denom = q.norm() * P_t.norm(dim=1)
        c_a = (P_t @ E[A_ID]) / denom
        c_b = (P_t @ E[B_ID]) / denom
    return c_a, c_b


def check_c8():
    steps, lr, wd = 300, 0.05, 0.1

    # ---- (a) plain SGD, no weight decay ----
    _, _, st = train(seed=0, steps=steps, lr=lr, weight_decay=0.0)
    print("== C8(a) == (seed 0, plain SGD, 300 steps, no weight decay)")
    print(f"  max_t |grad E[a] - grad E[b]|_inf : {st['max_grad_gap']:.3e}")
    print(f"  max_t drift of (E[a]-E[b]) from init (inf-norm): "
          f"{st['max_diff_drift']:.3e}")
    print(f"  ||E[a]+E[b]|| moved by {st['sum_move_norm']:.4f} "
          f"(final loss {st['final_loss']:.4f})")
    # extra seeds for (a)
    a_ok = True
    for s in range(1, 5):
        _, _, st_s = train(seed=s, steps=steps, lr=lr, weight_decay=0.0)
        print(f"  seed {s}: grad gap {st_s['max_grad_gap']:.3e}, "
              f"diff drift {st_s['max_diff_drift']:.3e}, "
              f"sum moved {st_s['sum_move_norm']:.4f}")
        a_ok &= st_s['max_grad_gap'] < TOL and st_s['max_diff_drift'] < TOL

    # ---- (b) with L2 weight decay ----
    _, _, stw = train(seed=0, steps=steps, lr=lr, weight_decay=wd)
    predicted_ratio = (1.0 - lr * wd) ** steps
    actual_ratio = stw["diff_norm_end"] / stw["diff_norm_start"]
    print(f"== C8(b) == (seed 0, SGD weight_decay={wd})")
    print(f"  ||E[a]-E[b]||: start {stw['diff_norm_start']:.6f} -> "
          f"end {stw['diff_norm_end']:.6f}")
    print(f"  decay ratio actual {actual_ratio:.6e} vs "
          f"(1-lr*wd)^T = {predicted_ratio:.6e} "
          f"(rel dev {abs(actual_ratio - predicted_ratio) / predicted_ratio:.2e})")

    # ---- (c) split vs sum dispersion across seeds ----
    for structured, c_steps, c_lr, tag, item in [
            (False, 300, 0.05, "literal: random labels, 300 steps", None),
            (True, 2000, 0.1, "learnable labels, 2000 steps", None),
            (True, 2000, 0.1, "learnable labels, 2000 steps, 2nd item", 4)]:
        c8c_run(structured, c_steps, c_lr, tag, eval_item=item)
    return a_ok


def c8c_run(structured, steps, lr, tag, eval_item=None):
    seeds = list(range(8))
    _, _, carries_ab = build_dataset()
    if eval_item is None:
        eval_item = int(np.flatnonzero(carries_ab)[0])
    CA, CB = [], []
    for s in seeds:
        p, item_rows, st = train(seed=s, steps=steps, lr=lr,
                                 weight_decay=0.0,
                                 structured_labels=structured)
        c_a, c_b = cosine_shares(p, item_rows, eval_item)
        CA.append(c_a.numpy())
        CB.append(c_b.numpy())
    CA = np.stack(CA)   # (seeds, K_t)
    CB = np.stack(CB)

    def rel_disp(x):    # std/|mean| across seeds, x: (seeds,)
        return np.std(x, ddof=1) / max(abs(np.mean(x)), 1e-30)

    # k-aggregated shares (prototype-permutation-invariant across seeds)
    Ca_tot, Cb_tot = CA.sum(axis=1), CB.sum(axis=1)
    tot = Ca_tot + Cb_tot
    print(f"== C8(c) == [{tag}] ({len(seeds)} seeds, eval item "
          f"{eval_item}, final loss seed7 {st['final_loss']:.4f})")
    print(f"  k-summed shares per seed:")
    print(f"    c_a     : {np.array2string(Ca_tot, precision=4)}")
    print(f"    c_b     : {np.array2string(Cb_tot, precision=4)}")
    print(f"    c_a+c_b : {np.array2string(tot, precision=4)}")
    print(f"  abs std across seeds:  c_a {np.std(Ca_tot, ddof=1):.4f}   "
          f"c_b {np.std(Cb_tot, ddof=1):.4f}   "
          f"split (c_a-c_b) {np.std(Ca_tot - Cb_tot, ddof=1):.4f}   "
          f"sum {np.std(tot, ddof=1):.4f}")
    print(f"  rel dispersion (std/|mean|):  c_a {rel_disp(Ca_tot):.3f}   "
          f"c_b {rel_disp(Cb_tot):.3f}   "
          f"split {rel_disp(Ca_tot - Cb_tot):.3f}   "
          f"sum {rel_disp(tot):.3f}")
    frac = Ca_tot / tot
    print(f"  a-fraction c_a/(c_a+c_b) per seed: "
          f"{np.array2string(frac, precision=3)}")
    # literal per-k reading (note: prototype identity k is not aligned
    # across seeds, which inflates all per-k dispersions)
    perk_a = np.array([np.std(CA[:, k], ddof=1) for k in range(K_T)])
    perk_b = np.array([np.std(CB[:, k], ddof=1) for k in range(K_T)])
    perk_s = np.array([np.std(CA[:, k] + CB[:, k], ddof=1)
                       for k in range(K_T)])
    print(f"  per-k abs std (literal reading; k unaligned across seeds):")
    print(f"    c_a,k     : {np.array2string(perk_a, precision=3)}")
    print(f"    c_b,k     : {np.array2string(perk_b, precision=3)}")
    print(f"    c_a+c_b,k : {np.array2string(perk_s, precision=3)}")


if __name__ == "__main__":
    check_c7()
    print()
    check_c8()
