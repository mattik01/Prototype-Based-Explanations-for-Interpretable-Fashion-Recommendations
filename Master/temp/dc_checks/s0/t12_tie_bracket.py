# t12 — F-S0-14 tie-bracket instrument selftest (utilities/cold_eval.py).
# Verifies: (1) tie counting on crafted logits, (2) no-tie degeneracy (bracket collapses to the
# standard HR/NDCG point), (3) hand-computed straddle and all-tied cases, (4) the closed-form
# expected values against an INDEPENDENT Monte-Carlo of uniform-random tie-breaking, (5) the
# argpartition point metric always falls inside the bracket on tied data.
import numpy as np

from _harness import make_check

from utilities.cold_eval import _tie_counts, tie_bracket
from utilities.eval import Hit_Ratio_at_k_batch, NDCG_at_k_batch

check, fails = make_check()
rng = np.random.default_rng(38210573)
K = 10


# ---- (1) tie counting on a crafted row -------------------------------------------------------
logits = np.full((1, 100), 0.5, dtype=np.float32)
logits[0, 0] = 1.0          # target
logits[0, 1:4] = 2.0        # 3 strictly above
logits[0, 4:9] = 1.0        # 5 exact ties
a, t = _tie_counts(logits)
check('crafted counts: 3 above, 5 tied', a[0] == 3 and t[0] == 5, f'a={a[0]} t={t[0]}')

# ---- (2) no ties -> bracket degenerates to the standard point metrics ------------------------
uniq = rng.permutation(10000).astype(np.float64)[:100 * 50].reshape(50, 100)  # all values distinct
a, t = _tie_counts(uniq)
check('no-tie rows: n_tied all zero', (t == 0).all())
tb = tie_bracket(a, t, k=K)
hr_ref = Hit_Ratio_at_k_batch(uniq, k=K) / 50
ndcg_ref = NDCG_at_k_batch(uniq, k=K) / 50
for side in ('worst', 'expected', 'best'):
    check(f'no ties: HR {side} == standard HR', np.isclose(tb['hit_ratio'][side], hr_ref),
          f"{tb['hit_ratio'][side]:.6f} vs {hr_ref:.6f}")
    check(f'no ties: NDCG {side} == standard NDCG', np.isclose(tb['ndcg'][side], ndcg_ref),
          f"{tb['ndcg'][side]:.6f} vs {ndcg_ref:.6f}")
check('no ties: zero straddle/any-tie, block size 1',
      tb['share_rows_block_straddles_cut'] == 0 and tb['share_rows_any_tie'] == 0
      and tb['mean_tie_block_size'] == 1 and tb['max_tie_block_size'] == 1)

# ---- (3a) all 100 tied (the popularity-floor case): expected == chance floor -----------------
a = np.zeros(1, dtype=np.int64)
t = np.full(1, 99, dtype=np.int64)
tb = tie_bracket(a, t, k=K)
exp_ndcg = sum(1. / np.log2(p + 1) for p in range(1, K + 1)) / 100
check('all-tied: HR worst/expected/best = 0 / 0.10 / 1',
      tb['hit_ratio']['worst'] == 0 and np.isclose(tb['hit_ratio']['expected'], K / 100)
      and tb['hit_ratio']['best'] == 1)
check('all-tied: NDCG worst/expected/best = 0 / chance / 1',
      tb['ndcg']['worst'] == 0 and np.isclose(tb['ndcg']['expected'], exp_ndcg)
      and tb['ndcg']['best'] == 1, f"exp {tb['ndcg']['expected']:.6f} vs {exp_ndcg:.6f}")

# ---- (3b) hand-computed straddle: 8 above, 3 tied (block at ranks 9-12, cut at 10) -----------
tb = tie_bracket(np.array([8]), np.array([3]), k=K)
hand_ndcg = (1. / np.log2(10) + 1. / np.log2(11)) / 4    # target at rank 9 or 10, each prob 1/4
check('straddle: HR = 0 / 0.5 / 1',
      tb['hit_ratio']['worst'] == 0 and np.isclose(tb['hit_ratio']['expected'], 0.5)
      and tb['hit_ratio']['best'] == 1)
check('straddle: NDCG expected matches hand computation',
      np.isclose(tb['ndcg']['expected'], hand_ndcg),
      f"{tb['ndcg']['expected']:.6f} vs {hand_ndcg:.6f}")
check('straddle: flagged as straddling', tb['share_rows_block_straddles_cut'] == 1)

# ---- (4) Monte-Carlo cross-check on random tied data -----------------------------------------
# Rows with heavy ties from a tiny value alphabet; MC breaks ties with fresh uniform noise and
# reads the target's rank directly from the jittered scores — no shared code with the formulas.
n_rows, n_mc = 40, 20000
tied = rng.integers(0, 12, size=(n_rows, 100)).astype(np.float64)
a, t = _tie_counts(tied)
check('MC data actually tied', (t > 0).any(), f'mean block {(t + 1).mean():.1f}')
tb = tie_bracket(a, t, k=K)
hr_mc = np.zeros(n_rows)
ndcg_mc = np.zeros(n_rows)
for _ in range(n_mc // 100):
    jit = tied[None] + rng.uniform(0, 1e-6, size=(100, n_rows, 100))
    ranks = 1 + (jit > jit[:, :, :1]).sum(axis=2)        # target rank per (draw, row)
    hr_mc += (ranks <= K).mean(axis=0)
    ndcg_mc += np.where(ranks <= K, 1. / np.log2(ranks + 1), 0.).mean(axis=0)
hr_mc = hr_mc.mean() / (n_mc // 100)
ndcg_mc = ndcg_mc.mean() / (n_mc // 100)
check('MC: expected HR matches closed form (tol 3e-3)',
      abs(tb['hit_ratio']['expected'] - hr_mc) < 3e-3,
      f"closed {tb['hit_ratio']['expected']:.5f} vs MC {hr_mc:.5f}")
check('MC: expected NDCG matches closed form (tol 3e-3)',
      abs(tb['ndcg']['expected'] - ndcg_mc) < 3e-3,
      f"closed {tb['ndcg']['expected']:.5f} vs MC {ndcg_mc:.5f}")

# ---- (5) argpartition point falls inside the bracket on tied data ----------------------------
point_hr = Hit_Ratio_at_k_batch(tied, k=K) / n_rows
point_ndcg = NDCG_at_k_batch(tied, k=K) / n_rows
tb = tie_bracket(a, t, k=K, point_hr=point_hr, point_ndcg=point_ndcg)
check('point within bracket on tied data', tb['point_within_bracket'],
      f"HR point {point_hr:.4f} in [{tb['hit_ratio']['worst']:.4f}, {tb['hit_ratio']['best']:.4f}]")
check('bracket ordering: worst <= expected <= best',
      tb['hit_ratio']['worst'] <= tb['hit_ratio']['expected'] <= tb['hit_ratio']['best']
      and tb['ndcg']['worst'] <= tb['ndcg']['expected'] <= tb['ndcg']['best'])

print(f"\n{'ALL PASS' if not fails else f'FAILURES: {fails}'}")
raise SystemExit(1 if fails else 0)
