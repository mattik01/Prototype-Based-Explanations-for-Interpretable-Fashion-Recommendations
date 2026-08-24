"""dc06 black-box toy claims check (C1-C10).

Self-contained: implements the mechanism exactly as in claims_spec.md and
adversarially tests each claim. torch float64 throughout.

Toy scale: d=8, F=3 fields, vocab sizes [4,3,5], n=10 items, K=4 prototypes,
U=5 users. Tolerance for exact float64 identities: TOL = 1e-9 (float64 dot
products over ~10 terms carry ~1e-15 relative error; 1e-9 is generous but
strict enough to catch any structural residual). "Nonzero" threshold: 1e-6.

Run: conda run -n protomf python t01_claims_check.py
"""
import torch

torch.manual_seed(0)
DT = torch.float64
TOL = 1e-9
NZ = 1e-6  # threshold above which a quantity counts as "nonzero"

d, F, K, n, U = 8, 3, 4, 10, 5
vocab_sizes = [4, 3, 5]

# ---- catalog: item i carries value x[i, f] in field f (fixed layout) ----
# Deliberately leave value 3 of field 0 and value 4 of field 2 UNUSED
# (zero catalog frequency) to probe C6/C7 edge conditions.
x = torch.stack([
    torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),          # field 0: value 3 unused
    torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 1]),          # field 1: all used
    torch.tensor([0, 1, 2, 0, 0, 1, 2, 3, 0, 1]),          # field 2: value 4 unused; items 0,3 twin
], dim=1)  # (n, F)

def make_params(seed=0):
    g = torch.Generator().manual_seed(seed)
    E = [torch.randn(V, d, dtype=DT, generator=g, requires_grad=True) for V in vocab_sizes]
    EID = torch.randn(n, d, dtype=DT, generator=g, requires_grad=True)
    P = torch.randn(K, d, dtype=DT, generator=g, requires_grad=True)
    Uaff = torch.randn(U, K, dtype=DT, generator=g, requires_grad=True)   # cosine-host users
    Q = torch.randn(U, d, dtype=DT, generator=g, requires_grad=True)      # dot-host users
    return E, EID, P, Uaff, Q

def freqs():
    return [torch.bincount(x[:, f], minlength=vocab_sizes[f]).to(DT) / n for f in range(F)]

PF = freqs()

def mu(E, flavour, detach=False):
    out = []
    for f in range(F):
        Ef = E[f].detach() if detach else E[f]
        if flavour == "weighted":
            out.append(PF[f] @ Ef)
        else:
            out.append(Ef.mean(dim=0))
    return out  # list of (d,)

def t_raw(E, EID):
    return sum(E[f][x[:, f]] for f in range(F)) + EID  # (n, d)

def t_centred(E, EID, flavour, detach=False):
    m = mu(E, flavour, detach)
    return sum(E[f][x[:, f]] - m[f] for f in range(F)) + EID

def cos_act(T, P):  # t*_k(i) = 1 + cos(t_i, p_k) -> (n, K)
    return 1 + (T @ P.T) / (T.norm(dim=1, keepdim=True) * P.norm(dim=1))

def ranking(scores):  # (U, n) -> per-user ordering of items
    return torch.argsort(scores, dim=1, descending=True)

E, EID, P, Uaff, Q = make_params()
verdicts = {}

def report(c, verdict, msg):
    verdicts[c] = verdict
    print(f"\n[{c}] {verdict}\n{msg}")

print("=" * 72)
print("dc06 toy claims check  (d=8, F=3, vocabs=[4,3,5], n=10, K=4, U=5)")
print(f"exact-identity tolerance TOL={TOL}, nonzero threshold NZ={NZ}")
print("=" * 72)

# -------------------- C1: shared vector, both flavours --------------------
with torch.no_grad():
    msgs = []
    ok = True
    for fl in ("weighted", "unweighted"):
        C = sum(mu(E, fl))
        resid = (t_centred(E, EID, fl) - (t_raw(E, EID) - C)).abs().max().item()
        spread = 0.0  # verify C is item-independent by direct per-item recompute
        per_item_C = t_raw(E, EID) - t_centred(E, EID, fl)  # (n, d)
        spread = (per_item_C - per_item_C[0]).abs().max().item()
        msgs.append(f"  {fl}: max|t_i - (t_raw - C)| = {resid:.3e}; "
                    f"max spread of per-item subtracted vector = {spread:.3e}")
        ok &= resid < TOL and spread < TOL
report("C1", "CONFIRMED" if ok else "FALSIFIED", "\n".join(msgs))

# -------------------- C2: gauge invariance --------------------
msgs, ok = [], True
c_shift = torch.full((d,), 3.7, dtype=DT)
shift_field = 1
for fl in ("weighted", "unweighted"):
    # loss on centred t: use a fixed nonlinear loss so gradient comparison is meaningful
    def loss_of(Elist, EIDp):
        T = t_centred(Elist, EIDp, fl)
        return (T @ P.detach().T).tanh().sum() + T.pow(2).sum() * 0.1

    E1, EID1, *_ = make_params()
    L1 = loss_of(E1, EID1)
    g1 = torch.autograd.grad(L1, [E1[0], E1[2], EID1])

    E2, EID2, *_ = make_params()
    E2sh = list(E2)
    E2sh[shift_field] = E2[shift_field] + c_shift  # add c to EVERY row of field 1
    with torch.no_grad():
        dT = (t_centred([e.detach() for e in E2sh], EID2.detach(), fl)
              - t_centred([e.detach() for e in E1], EID1.detach(), fl)).abs().max().item()
        dRaw = (t_raw([e.detach() for e in E2sh], EID2.detach())
                - t_raw([e.detach() for e in E1], EID1.detach())).abs().max().item()
    L2 = loss_of(E2sh, EID2)
    g2 = torch.autograd.grad(L2, [E2[0], E2[2], EID2])
    dG = max((a - b).abs().max().item() for a, b in zip(g1, g2))
    msgs.append(f"  {fl}: shift c=3.7*1 on all rows of field {shift_field}: "
                f"max|dt_centred|={dT:.3e}, max|dgrad(other params)|={dG:.3e}, "
                f"max|dt_raw|={dRaw:.3e} (raw must change)")
    ok &= dT < TOL and dG < TOL and dRaw > NZ
report("C2", "CONFIRMED" if ok else "FALSIFIED", "\n".join(msgs))

# -------------------- C3: exact share decomposition --------------------
with torch.no_grad():
    fl = "weighted"
    m = mu(E, fl)
    T = t_centred(E, EID, fl)
    total = T @ P.T  # (n, K)
    shares = sum((E[f][x[:, f]] - m[f]) @ P.T for f in range(F)) + EID @ P.T
    resid = (total - shares).abs().max().item()
report("C3", "CONFIRMED" if resid < TOL else "FALSIFIED",
       f"  max|t_i^T p_k - (sum of field shares + ID share)| = {resid:.3e} "
       f"(< TOL={TOL}; exact up to float64 rounding)")

# -------------------- C4: cold path --------------------
with torch.no_grad():
    fl = "weighted"
    m = mu(E, fl)
    t_cold = lambda vals: sum(E[f][vals[f]] - m[f] for f in range(F))
    t_cold_raw = lambda vals: sum(E[f][vals[f]] for f in range(F))
    a, b = x[0], x[1]           # different field values
    same1, same2 = x[0], x[3]   # items 0 and 3 share ALL field values (see catalog)
    assert torch.equal(same1, same2)
    n_a = t_cold(a).norm().item()
    diff_ab = (t_cold(a) - t_cold(b)).norm().item()
    eq_c = (t_cold(same1) - t_cold(same2)).abs().max().item()
    eq_r = (t_cold_raw(same1) - t_cold_raw(same2)).abs().max().item()
    ok = n_a > NZ and diff_ab > NZ and eq_c < TOL and eq_r < TOL
report("C4", "CONFIRMED" if ok else "FALSIFIED",
       f"  ||t_cold(item0)||={n_a:.4f} (nonzero); ||t_cold(0)-t_cold(1)||={diff_ab:.4f} "
       f"(differ); items 0,3 share all values: max diff centred={eq_c:.3e}, raw={eq_r:.3e}")

# -------------------- C5: rank inertness split --------------------
with torch.no_grad():
    fl = "weighted"
    C = sum(mu(E, fl))
    Traw, Tc = t_raw(E, EID), t_centred(E, EID, fl)
    # dot host
    s_raw, s_c = Q @ Traw.T, Q @ Tc.T
    rank_same = torch.equal(ranking(s_raw), ranking(s_c))
    max_pair_flip = 0.0  # also check pairwise score differences are preserved exactly
    pd = ((s_raw[:, :, None] - s_raw[:, None, :]) - (s_c[:, :, None] - s_c[:, None, :])).abs().max().item()
    # cosine host: search users for a ranking change
    a_raw, a_c = cos_act(Traw, P), cos_act(Tc, P)
    s2_raw, s2_c = Uaff @ a_raw.T, Uaff @ a_c.T
    r2_raw, r2_c = ranking(s2_raw), ranking(s2_c)
    changed_users = [u for u in range(U) if not torch.equal(r2_raw[u], r2_c[u])]
    cex = ""
    if changed_users:
        u = changed_users[0]
        cex = (f"    counterexample: user {u} ranking with t_raw = {r2_raw[u].tolist()}\n"
               f"                    user {u} ranking with t_raw - C = {r2_c[u].tolist()}")
    ok = rank_same and pd < TOL and bool(changed_users)
report("C5", "CONFIRMED" if ok else "FALSIFIED",
       f"  dot host: rankings identical = {rank_same}; max|pairwise score-gap change| = {pd:.3e}\n"
       f"  cosine host: ranking changed for {len(changed_users)}/{U} users\n{cex}")

# -------------------- C6: live vs detached mean gradients --------------------
msgs, ok, caveat = [], True, False
item = 0
for fl in ("weighted", "unweighted"):
    for detach in (False, True):
        E1, EID1, *_ = make_params()
        T = t_centred(E1, EID1, fl, detach=detach)
        L = T[item].pow(2).sum()  # loss depends only on item 0's centred t
        gE = torch.autograd.grad(L, E1, allow_unused=False)
        # rows of same-field values item 0 does NOT carry:
        rows = {}
        for f in range(F):
            for v in range(vocab_sizes[f]):
                if v != x[item, f].item():
                    rows[(f, v)] = gE[f][v].abs().max().item()
        mx = max(rows.values())
        mode = "detached" if detach else "live"
        if detach:
            msgs.append(f"  {fl}/{mode}: max grad on non-carried same-field rows = {mx:.3e}")
            ok &= mx < TOL
        else:
            # claim: NONZERO on those rows. Check the zero-frequency rows separately.
            zero_freq = {(0, 3), (2, 4)}
            g_used = max(v for k, v in rows.items() if k not in zero_freq)
            g_unused = max(v for k, v in rows.items() if k in zero_freq)
            msgs.append(f"  {fl}/{mode}: max grad on non-carried rows with catalog freq>0 = "
                        f"{g_used:.3e}; on ZERO-frequency vocab rows (f0,v3),(f2,v4) = {g_unused:.3e}")
            ok &= g_used > NZ
            if fl == "weighted" and g_unused < TOL:
                caveat = True
verdict = "CONFIRMED" if ok else "FALSIFIED"
cav = ("\n  CAVEAT (falsifies the universal 'nonzero' reading in the weighted flavour): a\n"
       "  vocabulary value with zero catalog frequency gets p_f(v)=0 and thus EXACTLY zero\n"
       "  live-mean gradient — nonzero holds only for rows with positive frequency\n"
       "  (unweighted flavour reaches all rows)." if caveat else "")
report("C6", verdict + (" (with stated condition)" if caveat else ""), "\n".join(msgs) + cav)

# -------------------- C7: gauge pin --------------------
E1, EID1, *_ = make_params()
mW = mu(E1, "weighted")
Lpin = sum(m.pow(2).sum() for m in mW)
gE = torch.autograd.grad(Lpin, E1)
row_g = {(f, v): gE[f][v].abs().max().item() for f in range(F) for v in range(vocab_sizes[f])}
zero_freq = {(0, 3), (2, 4)}
g_pos = min(v for k, v in row_g.items() if k not in zero_freq)
g_zero = max(v for k, v in row_g.items() if k in zero_freq)
with torch.no_grad():
    nonneg = Lpin.item() >= 0
    # construct embeddings with exactly-zero weighted means; check L_pin=0 and raw==centred
    E0 = [e.detach().clone() for e in E1]
    for f in range(F):
        # shifting ALL rows of field f by -mcur shifts the weighted mean by exactly
        # -mcur (weights sum to 1) => new weighted mean is zero
        mcur = PF[f] @ E0[f]
        E0[f] = E0[f] - mcur
    m0 = [PF[f] @ E0[f] for f in range(F)]
    L0 = sum(mm.pow(2).sum() for mm in m0).item()
    Tc0 = sum(E0[f][x[:, f]] - m0[f] for f in range(F)) + EID1.detach()
    Tr0 = sum(E0[f][x[:, f]] for f in range(F)) + EID1.detach()
    coincide = (Tc0 - Tr0).abs().max().item()
ok = nonneg and Lpin.item() > NZ and g_pos > NZ and L0 < TOL and coincide < TOL
cav = g_zero < TOL
report("C7", ("CONFIRMED (with stated condition)" if cav else "CONFIRMED") if ok else "FALSIFIED",
       f"  L_pin = {Lpin.item():.4f} >= 0; generic embeddings: min |grad| over positive-frequency\n"
       f"  rows = {g_pos:.3e} (all reached); after zeroing every weighted mean: L_pin = {L0:.3e}\n"
       f"  and max|t_centred - t_raw| = {coincide:.3e} (coincide).\n"
       f"  CAVEAT: grad on zero-frequency vocab rows = {g_zero:.3e} — 'reaches every metadata\n"
       f"  row' holds only for rows with positive catalog frequency (grad_v = 2 p_f(v) mu_f).")

# -------------------- C8: ID-mean pin --------------------
E1, EID1, *_ = make_params()
B = [0, 2, 4, 7]  # batch
m_id = EID1[B].mean(dim=0)
Lid = m_id.pow(2).sum()
gID, = torch.autograd.grad(Lid, [EID1])
rows = gID[B]
same = (rows - rows[0]).abs().max().item()
off_batch = gID[[i for i in range(n) if i not in B]].abs().max().item()
with torch.no_grad():
    # zero-mean batch => L=0
    EIDz = EID1.detach().clone()
    EIDz[B] -= EIDz[B].mean(dim=0)
    Lz = EIDz[B].mean(dim=0).pow(2).sum().item()
    # nonzero mean => L>0
ok = Lid.item() > NZ and Lz < TOL and same < TOL and off_batch < TOL
report("C8", "CONFIRMED" if ok else "FALSIFIED",
       f"  generic batch: L_idpin = {Lid.item():.4f} > 0; after removing batch mean L = {Lz:.3e};\n"
       f"  max row-to-row gradient difference within batch = {same:.3e} (identical rows =>\n"
       f"  pairwise differences e_ID(i)-e_ID(j) get exactly zero net gradient); grad off-batch = {off_batch:.3e}")

# -------------------- C9: bag layout is weaker --------------------
with torch.no_grad():
    fl = "weighted"
    m = mu(E, fl)
    # counterexample: item A has 1 token (field 0, value 0, w=1);
    #                 item B has 3 tokens (field 0 values 0,1,2, each w=1)
    massA = 1.0 * m[0]
    # B carries tokens (f0,v0),(f0,v1),(f0,v2) all weight 1 -> mass = 3*mu_0
    massB = 3.0 * m[0]
    dep = (massA - massB).norm().item()
    # weighted-count variant: same tokens, weights 0.5 each -> mass = 1.5*mu_0
    massC = 0.5 * 3 * m[0]
    dep2 = (massA - massC).norm().item()
    # reduction: every item exactly one token per field, weight 1 -> mass = sum_f mu_f = C
    C = sum(m)
    mass_fixed = sum(1.0 * m[f] for f in range(F))
    red = (mass_fixed - C).norm().item()
ok = dep > NZ and dep2 > NZ and red < TOL
report("C9", "CONFIRMED" if ok else "FALSIFIED",
       f"  counterexample: item A = 1 token in field 0 (mass = mu_0), item B = 3 field-0 tokens\n"
       f"  w=1 (mass = 3*mu_0): ||massA - massB|| = {dep:.4f} != 0 -> subtracted mass is\n"
       f"  item-dependent, no shared C. Reweighted variant differs too ({dep2:.4f}).\n"
       f"  Reduction check (one token per field, w=1): ||mass - C|| = {red:.3e}")

# -------------------- C10: shared C not activation-inert on cosine host --------------------
with torch.no_grad():
    fl = "weighted"
    Tc = t_centred(E, EID, fl)
    Cvec = sum(mu(E, fl))
    a0, a1 = cos_act(Tc, P), cos_act(Tc + Cvec, P)
    delta = a1 - a0  # (n, K)
    # item-dependence: find i, j, k with (a(i,k)-a(j,k)) changed
    gap0 = a0[:, None, :] - a0[None, :, :]
    gap1 = a1[:, None, :] - a1[None, :, :]
    dgap = (gap1 - gap0).abs()
    idx = torch.argmax(dgap).item()
    i = torch.tensor(idx // (n * K))
    j = torch.tensor((idx // K) % n)
    k = torch.tensor(idx % K)
    ok = dgap.max().item() > NZ
report("C10", "CONFIRMED" if ok else "FALSIFIED",
       f"  adding shared C to all t_i: max|delta activation| = {delta.abs().max().item():.4f};\n"
       f"  item-dependence witness: items i={i.item()}, j={j.item()}, prototype k={k.item()}:\n"
       f"  gap t*_k(i)-t*_k(j) moves {gap0[i, j, k].item():+.4f} -> {gap1[i, j, k].item():+.4f} "
       f"(change {dgap[i, j, k].item():.4f} > NZ)")

print("\n" + "=" * 72)
print("SUMMARY")
for c in sorted(verdicts, key=lambda s: int(s[1:])):
    print(f"  {c}: {verdicts[c]}")
