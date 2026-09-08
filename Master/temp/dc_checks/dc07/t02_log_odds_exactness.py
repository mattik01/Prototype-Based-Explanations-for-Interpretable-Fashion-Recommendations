# dc07 t02 — the lineage's additive attribution contract under softmax membership lives on the
# LOG-ODDS: log m_l − log m_k = Σ_f c_f ⟨e_f, p_l − p_k⟩ / τ for composed inputs q = Σ_f c_f e_f.
# Verified against brute force for the fI shape (FeatureEmbedding: sum of F word rows + ID row)
# and the fU shape (HistoryFeatureEmbedding: weighted word rows + ID row), ids and noid, in float64.
# Also: the decomposition does NOT hold on m itself (documented loss of the affine share zoom).
# Run: python Master/temp/dc_checks/dc07/t02_log_odds_exactness.py
from _common import *  # noqa: F401,F403

check, FAILS = make_check('t02')
torch.manual_seed(2)
N, D, K, NF, TAU = 30, 10, 6, 11, 0.6


def brute_rows(kind, ext, ids, w, i):
    E = ext.embedding_layer.weight
    rows = []
    if kind == 'fI':
        rows += [E[t] for t in ids[i]]
    else:
        # the module stores hist_weights as a float32 buffer (rounded once); brute force must use
        # the SAME rounded weights, otherwise the comparison measures float32 rounding, not exactness
        wb = ext.hist_weights
        rows += [wb[i, j] * E[ids[i, j]] for j in range(ids.shape[1]) if wb[i, j] > 0]
    if ext.use_id_feature:
        rows.append(E[NF + i])
    return rows


for kind in ('fI', 'fU'):
    for use_id in (True, False):
        if kind == 'fI':
            ext, ids = fi_ext(N, D, NF, use_id=use_id); w = None
        else:
            ext, ids, w = fu_ext(N, D, NF, use_id=use_id)
        m = proto(N, D, K, 'softmax', TAU, ext=ext)
        m.eval()
        tag = f"{kind}.{'ids' if use_id else 'noid'}"
        with torch.no_grad():
            idx = torch.arange(N)
            q = m.embedding_ext(idx)
            mem = m(idx)
            # (a) the rows sum to q (the composition contract this rides on)
            maxdiff = max(((torch.stack(brute_rows(kind, ext, ids, w, i)).sum(0) - q[i]).abs().max()
                           for i in range(N) if brute_rows(kind, ext, ids, w, i)))
            check(f"{tag}.rows_sum_to_q", maxdiff < 1e-12, f"{maxdiff:.2e}")
            # (b) log-odds exactness for every pair (l, k) and every object
            P = m.prototypes
            worst = 0.0
            for i in range(N):
                rows = brute_rows(kind, ext, ids, w, i)
                if not rows:
                    continue
                for l in range(K):
                    for k in range(K):
                        lhs = mem[i, l].log() - mem[i, k].log()
                        rhs = sum((r @ (P[l] - P[k])) / TAU for r in rows)
                        worst = max(worst, float((lhs - rhs).abs()))
            check(f"{tag}.log_odds_exact_over_rows", worst < 1e-10, f"worst |lhs−rhs| = {worst:.2e}")
            # (c) NOT additive on m itself: m(q) ≠ Σ_f m(c_f e_f) − (F−1)·(anything constant)
            i = next(i for i in range(N) if len(brute_rows(kind, ext, ids, w, i)) >= 2)
            rows = brute_rows(kind, ext, ids, w, i)
            per_row_m = torch.stack([m.memberships(r) for r in rows]).sum(0)
            check(f"{tag}.not_additive_on_m", not torch.allclose(per_row_m, mem[i], atol=1e-6),
                  "membership of the sum ≠ sum of memberships (share zoom on m is lost, log-odds replaces it)")
        m.get_and_reset_loss()

# sigmoid: the LOGIT is exactly linear in the rows (t07 pins the rest)
ext, ids = fi_ext(N, D, NF, use_id=True)
TAU_SG = 4.0  # keeps σ away from saturation so the inverse (log m − log(1−m)) is numerically exact
m = proto(N, D, K, 'sigmoid', TAU_SG, ext=ext).eval()
with torch.no_grad():
    idx = torch.arange(N); mem = m(idx)
    logit = torch.log(mem) - torch.log(1 - mem)
    worst = 0.0
    for i in range(N):
        rows = brute_rows('fI', ext, ids, None, i)
        rhs = sum(r @ m.prototypes.T for r in rows) / TAU_SG + m.membership_bias
        worst = max(worst, float(((logit[i] - rhs).abs() / (1 + rhs.abs())).max()))
check("sigmoid.logit_linear_in_rows", worst < 1e-9, f"worst relative |lhs−rhs| = {worst:.2e}")
finish('t02', FAILS)
