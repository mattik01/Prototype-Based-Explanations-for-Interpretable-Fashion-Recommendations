# t01 — Evaluator divisor semantics (F-S0-03): metrics divide by the COUNTED number of
# evaluated rows; the count is asserted against the declared expectation (n_users);
# results are bit-identical to the historical divide-by-n_users while rows == n_users.
import sys

import numpy as np

from _harness import make_check

from utilities.consts import K_VALUES
from utilities.eval import Evaluator, Hit_Ratio_at_k_batch, NDCG_at_k_batch

check, fails = make_check()

rng = np.random.default_rng(38210573)


def make_logits(n_rows, n_cols=100, hit_rows=()):
    """Random logits; rows in hit_rows get the positive (col 0) as the argmax → HR@1 hit."""
    logits = rng.standard_normal((n_rows, n_cols)).astype(np.float64)
    logits[:, 0] = logits.min(axis=1) - 1.0          # positive ranked LAST by default
    for r in hit_rows:
        logits[r, 0] = logits[r].max() + 1.0         # positive ranked FIRST
    return logits


# --- 1. counted-rows divisor == exact hand computation (single batch) ---
logits = make_logits(8, hit_rows=(0, 3))
ev = Evaluator(8)
ev.eval_batch(logits)
res = ev.get_results()
check("hit_ratio@1 == 2/8", np.isclose(res['hit_ratio@1'], 2 / 8), f"{res['hit_ratio@1']}")
check("ndcg@1 == 2/8", np.isclose(res['ndcg@1'], 2 / 8), f"{res['ndcg@1']}")
check("all K present", all(f'hit_ratio@{k}' in res and f'ndcg@{k}' in res for k in K_VALUES))

# --- 2. multi-batch accumulation: rows counted across batches ---
b1, b2 = make_logits(5, hit_rows=(1,)), make_logits(3, hit_rows=(0, 2))
ev = Evaluator(8)
ev.eval_batch(b1)
ev.eval_batch(b2)
res = ev.get_results()
check("multi-batch hit_ratio@1 == 3/8", np.isclose(res['hit_ratio@1'], 3 / 8), f"{res['hit_ratio@1']}")

# --- 3. bit-identical to historical divide-by-n_users while rows == n_users ---
b1, b2 = make_logits(6, hit_rows=(0,)), make_logits(4, hit_rows=(1, 3))
manual = {}
for k in K_VALUES:
    for name, metric in [(f'ndcg@{k}', NDCG_at_k_batch), (f'hit_ratio@{k}', Hit_Ratio_at_k_batch)]:
        manual[name] = (metric(b1, k) + metric(b2, k)) / 10  # the historical formula, n_users=10
ev = Evaluator(10)
ev.eval_batch(b1)
ev.eval_batch(b2)
res = ev.get_results()
check("bit-identical to historical divisor (rows == n_users)",
      all(res[m] == manual[m] for m in manual),
      f"max |Δ| = {max(abs(res[m] - manual[m]) for m in manual)}")

# --- 4. assert fires when counted != expected ---
ev = Evaluator(9)                    # expects 9, will see 8
ev.eval_batch(make_logits(8))
try:
    ev.get_results()
    check("row-count mismatch raises", False)
except AssertionError as e:
    check("row-count mismatch raises", True, str(e)[:60])

# --- 5. expectation None → divides by counted rows, no assert ---
ev = Evaluator(None)
ev.eval_batch(make_logits(4, hit_rows=(0, 1)))
res = ev.get_results()
check("None expectation: divides by counted rows", np.isclose(res['hit_ratio@1'], 2 / 4),
      f"{res['hit_ratio@1']}")

# --- 6. aggregated=False path: per-row lists, count still checked ---
ev = Evaluator(4)
ev.eval_batch(make_logits(4, hit_rows=(2,)), sum=False)
res = ev.get_results(aggregated=False)
check("non-aggregated: per-row list of len 4", len(res['hit_ratio@1']) == 4, f"{len(res['hit_ratio@1'])}")
check("non-aggregated: values are per-row 0/1", list(res['hit_ratio@1']) == [0, 0, 1, 0])

ev = Evaluator(5)                    # expects 5, will see 4 → assert also on this path
ev.eval_batch(make_logits(4), sum=False)
try:
    ev.get_results(aggregated=False)
    check("non-aggregated row-count mismatch raises", False)
except AssertionError:
    check("non-aggregated row-count mismatch raises", True)

# --- 7. state reset after get_results (fresh accumulation) ---
ev = Evaluator(3)
ev.eval_batch(make_logits(3, hit_rows=(0,)))
_ = ev.get_results()
ev.eval_batch(make_logits(3, hit_rows=(0, 1, 2)))
res = ev.get_results()
check("state resets between get_results calls", np.isclose(res['hit_ratio@1'], 1.0), f"{res['hit_ratio@1']}")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
