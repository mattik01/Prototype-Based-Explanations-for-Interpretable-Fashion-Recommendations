"""
Black-box toy sanity check of mechanism M as defined in claims_spec.md.
Implements ONLY what the spec defines; verifies claims 1-11 and prints
per-claim evidence. Runs in seconds on CPU (torch 1.9.1 compatible).

Re-run:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate protomf
  python Master/temp/dc_checks/dc02/check_claims.py
"""
import math
import torch

torch.manual_seed(0)

# ---------------- toy dimensions (per spec suggestion) ----------------
N_u, M, F = 30, 40, 3
V_fs = [4, 5, 6]
V = sum(V_fs)          # 15
d, L_u, K = 8, 4, 5
B, n_neg = 8, 3
n = 1 + n_neg          # 4

FIELD_OFFSETS = [0]
for vf in V_fs[:-1]:
    FIELD_OFFSETS.append(FIELD_OFFSETS[-1] + vf)

# ---------------- fixed data X: one value per field, multi-hot --------
def make_X():
    X = torch.zeros(M, V)
    for i in range(M):
        for f, vf in enumerate(V_fs):
            v = torch.randint(0, vf, (1,)).item()
            X[i, FIELD_OFFSETS[f] + v] = 1.0
    # force item M-1 (cold twin) to have X row identical to item 0 (warm)
    X[M - 1] = X[0]
    return X

X = make_X()                     # fixed constant, NOT a parameter
assert not X.requires_grad

# ---------------- mechanism ----------------
def sc(a, b):
    """shifted cosine: a (..., D), b (R, D) -> (..., R)"""
    num = a @ b.t()
    den = a.norm(dim=-1, keepdim=True) * b.norm(dim=-1)
    return 1.0 + num / den

def make_params(seed=1):
    g = torch.Generator().manual_seed(seed)
    def rnd(*shape):
        t = torch.randn(*shape, generator=g)
        t.requires_grad_(True)
        return t
    return {
        "U": rnd(N_u, d),
        "P": rnd(L_u, d),
        "W_u": rnd(d, K),
        "A": rnd(K, V),
        "W_t": rnd(V, L_u),
    }

def forward(params, u_idxs, i_idxs, Xmat=None, return_intermediates=False):
    Xm = X if Xmat is None else Xmat
    u = params["U"][u_idxs]                     # (B, d)
    u_star = sc(u, params["P"])                 # (B, L_u)
    u_hat = u @ params["W_u"]                   # (B, K)
    x = Xm[i_idxs]                              # (B, n, V)
    t_star = sc(x, params["A"])                 # (B, n, K)
    t_hat = x @ params["W_t"]                   # (B, n, L_u)
    score = (u_star.unsqueeze(1) * t_hat).sum(-1) + \
            (u_hat.unsqueeze(1) * t_star).sum(-1)   # (B, n)
    if return_intermediates:
        return score, dict(u=u, u_star=u_star, u_hat=u_hat, x=x,
                           t_star=t_star, t_hat=t_hat)
    return score

def regs(t_star):
    S = t_star.reshape(-1, K)                   # (B*n, K)
    R_batch = -S.max(dim=1).values.mean()
    R_proto = -S.max(dim=0).values.mean()
    return R_batch, R_proto, S

results = {}

def report(claim, verdict, evidence):
    results[claim] = (verdict, evidence)

# =====================================================================
# Claim 1: shapes
# =====================================================================
params = make_params()
u_idxs = torch.randint(0, N_u, (B,))
i_idxs = torch.randint(0, M, (B, n))
score, inter = forward(params, u_idxs, i_idxs, return_intermediates=True)
shapes_ok = (
    inter["u"].shape == (B, d) and inter["u_star"].shape == (B, L_u)
    and inter["u_hat"].shape == (B, K) and inter["x"].shape == (B, n, V)
    and inter["t_star"].shape == (B, n, K) and inter["t_hat"].shape == (B, n, L_u)
    and score.shape == (B, n)
)
report(1, shapes_ok,
       f"score {tuple(score.shape)}=(B,n); u* {tuple(inter['u_star'].shape)}, "
       f"u_hat {tuple(inter['u_hat'].shape)}, t* {tuple(inter['t_star'].shape)}, "
       f"t_hat {tuple(inter['t_hat'].shape)} — all as annotated: {shapes_ok}")

# =====================================================================
# Claim 2: cold-item non-degeneracy (short toy optimization)
# =====================================================================
COLD = M - 1   # excluded from training; X[COLD] == X[0] (warm twin)
tr_params = make_params(seed=2)
opt = torch.optim.Adam(tr_params.values(), lr=0.05)
losses = []
gtr = torch.Generator().manual_seed(3)
for step in range(200):
    ub = torch.randint(0, N_u, (B,), generator=gtr)
    ib = torch.randint(0, M - 1, (B, n), generator=gtr)   # never samples COLD
    assert (ib != COLD).all()
    sc_out = forward(tr_params, ub, ib)
    labels = torch.zeros(B, n)
    labels[:, 0] = 1.0
    rb, rp, _ = regs(sc(X[ib], tr_params["A"]))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(sc_out, labels) \
           + 0.1 * (rb + rp)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
loss_decreased = sum(losses[-20:]) / 20 < sum(losses[:20]) / 20
with torch.no_grad():
    t_star_all = sc(X, tr_params["A"])          # (M, K)
cold_t = t_star_all[COLD]
warm_t = t_star_all[0]
finite = torch.isfinite(cold_t).all().item()
nonconst = cold_t.std().item() > 1e-6
exact_tie = torch.equal(cold_t, warm_t)
report(2, loss_decreased and finite and nonconst and exact_tie,
       f"loss {sum(losses[:20])/20:.4f}->{sum(losses[-20:])/20:.4f}; cold t* finite={finite}, "
       f"std across k={cold_t.std().item():.4f} (>0), bitwise equal to warm twin={exact_tie}")

# =====================================================================
# Claim 3: exact per-attribute decomposition
# =====================================================================
with torch.no_grad():
    A = params["A"]
    t_star_full = sc(X, A)                              # (M, K)
    lhs = t_star_full - 1.0
    rhs = (X @ A.t()) / (math.sqrt(F) * A.norm(dim=-1))  # (M, K)
    err3 = (lhs - rhs).abs().max().item()
report(3, err3 < 1e-5, f"max |lhs-rhs| over all (item,proto) = {err3:.2e} (tol 1e-5)")

# =====================================================================
# Claim 4: exact score decomposition
# =====================================================================
with torch.no_grad():
    addends_u = inter["u_star"].unsqueeze(1) * inter["t_hat"]   # (B,n,L_u)
    addends_i = inter["u_hat"].unsqueeze(1) * inter["t_star"]   # (B,n,K)
    recon = addends_u.sum(-1) + addends_i.sum(-1)
    err4 = (recon - score).abs().max().item()
report(4, err4 < 1e-5,
       f"|sum of {L_u}+{K} per-prototype addends - score| max = {err4:.2e} (tol 1e-5)")

# =====================================================================
# Claim 5: gradient flow to all params, none to X
# =====================================================================
p5 = make_params(seed=5)
s5 = forward(p5, u_idxs, i_idxs)
(s5 ** 2).sum().backward()
gnorms = {k: v.grad.norm().item() for k, v in p5.items()}
all_nonzero = all(g > 0 for g in gnorms.values())
x_no_grad = (X.grad is None) and (not X.requires_grad)
report(5, all_nonzero and x_no_grad,
       "grad norms " + ", ".join(f"{k}={g:.3f}" for k, g in gnorms.items())
       + f"; X.requires_grad={X.requires_grad}, X.grad={X.grad}")

# =====================================================================
# Claim 6: same-signature ties; per-item bias breaks tie
# =====================================================================
with torch.no_grad():
    all_u = torch.arange(N_u)
    pair = torch.tensor([[0, COLD]]).expand(N_u, 2)      # identical X rows
    sc_pair = forward(params, all_u, pair)               # (N_u, 2)
    tie = torch.equal(sc_pair[:, 0], sc_pair[:, 1])
    b = torch.randn(M)                                    # per-item bias
    sc_bias = sc_pair + b[pair]
    tie_broken = not torch.allclose(sc_bias[:, 0], sc_bias[:, 1])
    max_gap = (sc_pair[:, 0] - sc_pair[:, 1]).abs().max().item()
report(6, tie and tie_broken,
       f"exact tie across all {N_u} users (max gap {max_gap:.1e}); "
       f"with bias b_i: max gap {(sc_bias[:,0]-sc_bias[:,1]).abs().max().item():.3f} -> tie destroyed")

# =====================================================================
# Claim 7: regularizer direction
# =====================================================================
p7 = make_params(seed=7)
with torch.no_grad():
    x7 = X[i_idxs].reshape(-1, V)          # (B*n, V) batch items
def R_of(Amat):
    S = sc(x7, Amat)
    return -S.max(dim=1).values.mean(), -S.max(dim=0).values.mean(), S
with torch.no_grad():
    Rb0, Rp0, S0 = R_of(p7["A"])
    # (a) align item r's best-matching prototype k* more with x_r
    r = 0
    k_star = S0[r].argmax().item()
    alpha = 0.05
    A_a = p7["A"].clone(); A_a[k_star] += alpha * x7[r]
    Rb_a, _, S_a = R_of(A_a)
    align_up = S_a[r, k_star].item() > S0[r, k_star].item()
    dec_a = Rb_a.item() < Rb0.item()
    # (b) rotate worst-claimed prototype toward the x of its best-claiming item
    col_max = S0.max(dim=0).values
    j_star = col_max.argmin().item()
    r2 = S0[:, j_star].argmax().item()
    A_b = p7["A"].clone(); A_b[j_star] += alpha * x7[r2]
    _, Rp_b, _ = R_of(A_b)
    dec_b = Rp_b.item() < Rp0.item()
# autograd sign checks at alpha=0
al = torch.zeros(1, requires_grad=True)
Aa = p7["A"].detach().clone(); mask = torch.zeros(K, V); mask[k_star] = x7[r]
Rb_al, _, _ = R_of(Aa + al * mask)
gRb = torch.autograd.grad(Rb_al, al)[0].item()
al2 = torch.zeros(1, requires_grad=True)
mask2 = torch.zeros(K, V); mask2[j_star] = x7[r2]
_, Rp_al, _ = R_of(Aa + al2 * mask2)
gRp = torch.autograd.grad(Rp_al, al2)[0].item()
report(7, dec_a and dec_b and gRb < 0 and gRp < 0 and align_up,
       f"(a) R_batch {Rb0.item():.6f}->{Rb_a.item():.6f} (dR/da={gRb:.4f}<0); "
       f"(b) R_proto {Rp0.item():.6f}->{Rp_b.item():.6f} (dR/da={gRp:.4f}<0)")

# =====================================================================
# Claim 8: C1 mechanics (softplus reparam)
# =====================================================================
A_tilde = torch.randn(K, V, requires_grad=True)
A_pos = torch.nn.functional.softplus(A_tilde)
all_pos = (A_pos > 0).all().item()
t_star8 = sc(X, A_pos)
t_star8.sum().backward()
grad_reaches = A_tilde.grad is not None and A_tilde.grad.norm().item() > 0
in_range = t_star8.min().item() >= 1.0 - 1e-6 and t_star8.max().item() <= 2.0 + 1e-6
report(8, all_pos and grad_reaches and in_range,
       f"A>0 all: {all_pos}; grad(A_tilde) norm={A_tilde.grad.norm().item():.3f}; "
       f"t* range [{t_star8.min().item():.4f}, {t_star8.max().item():.4f}] within [1,2]")

# =====================================================================
# Claim 9: C2 mechanics (field crispness)
# =====================================================================
def L_crisp(Amat):
    tot = 0.0
    for f, vf in enumerate(V_fs):
        blk = Amat[:, FIELD_OFFSETS[f]:FIELD_OFFSETS[f] + vf]     # (K, V_f)
        p = torch.softmax(blk, dim=-1)
        H = -(p * torch.log(p + 1e-12)).sum(-1)                    # (K,)
        tot = tot + H.sum()
    return tot / (K * F)
A_onehot = torch.zeros(K, V)
for f, vf in enumerate(V_fs):
    A_onehot[:, FIELD_OFFSETS[f]] = 50.0    # one dominant coord per block
A_unif = torch.ones(K, V)
Lc_onehot = L_crisp(A_onehot).item()
Lc_unif = L_crisp(A_unif).item()
Lc_max_theory = sum(math.log(vf) for vf in V_fs) / F
A9 = torch.randn(K, V, requires_grad=True)
Lc_rand = L_crisp(A9)
Lc_rand.backward()
g9 = A9.grad.norm().item()
report(9, Lc_onehot < 1e-6 and abs(Lc_unif - Lc_max_theory) < 1e-5
          and Lc_rand.item() < Lc_unif and g9 > 0,
       f"one-hot-like: L={Lc_onehot:.2e}~0; uniform: L={Lc_unif:.5f}=mean ln(V_f)"
       f"={Lc_max_theory:.5f} (max); random {Lc_rand.item():.4f}<uniform; grad(A) norm={g9:.3f}")

# =====================================================================
# Claim 10: C3 mechanics (separation)
# =====================================================================
m10 = 0.3
def L_sep(Amat, m):
    An = Amat / Amat.norm(dim=-1, keepdim=True)
    C = An @ An.t()
    iu = torch.triu_indices(K, K, offset=1)
    return torch.relu(C[iu[0], iu[1]] - m).mean()
A_orth = torch.zeros(K, V)
for k in range(K):
    A_orth[k, k] = 1.0                       # pairwise cos = 0 <= m
L_orth = L_sep(A_orth, m10).item()
A_dup = torch.randn(K, V); A_dup[1] = A_dup[0] + 1e-4 * torch.randn(V)
L_dup = L_sep(A_dup, m10).item()
A10 = torch.randn(K, V, requires_grad=True)
Ls = L_sep(A10, m10)
Ls.backward()
g10 = A10.grad.norm().item() if A10.grad is not None else 0.0
report(10, L_orth == 0.0 and L_dup > 0 and g10 > 0,
       f"orthogonal protos (cos=0<=m={m10}): L_sep={L_orth}; near-duplicate pair: "
       f"L_sep={L_dup:.4f}>0; grad(A) norm={g10:.3f}")

# =====================================================================
# Claim 11: C4 mechanics (data pull)
# =====================================================================
S_items = torch.arange(0, 10)                 # fixed item subset
Xs = X[S_items]
def L_push(Amat):
    An = Amat / Amat.norm(dim=-1, keepdim=True)
    Xn = Xs / Xs.norm(dim=-1, keepdim=True)
    cosm = An @ Xn.t()                        # (K, |S|)
    return (1.0 - cosm.max(dim=1).values).mean()
# every prototype = positive multiple of some x_i in S_items -> 0
A_pull = torch.stack([ (0.5 + k) * Xs[k % len(S_items)] for k in range(K) ])
L_zero = L_push(A_pull).item()
# perturb one prototype off the data manifold -> positive
A_off = A_pull.clone(); A_off[0] = A_off[0] + torch.randn(V) * 0.5
L_off = L_push(A_off).item()
# negative multiple -> positive
A_neg = A_pull.clone(); A_neg[0] = -A_neg[0]
L_neg = L_push(A_neg).item()
A11 = torch.randn(K, V, requires_grad=True)
Lp = L_push(A11)
Lp.backward()
g11 = A11.grad.norm().item()
report(11, L_zero < 1e-6 and L_off > 1e-4 and L_neg > 1e-4 and g11 > 0,
       f"pos. multiples of x_i: L_push={L_zero:.2e}~0; perturbed proto: {L_off:.4f}>0; "
       f"negated proto: {L_neg:.4f}>0; grad(A) norm={g11:.3f}")

# =====================================================================
print("=" * 72)
for c in range(1, 12):
    v, e = results[c]
    print(f"Claim {c:2d}: {'CONFIRMED' if v else 'FALSIFIED'} — {e}")
