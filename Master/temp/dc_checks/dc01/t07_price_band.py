# dc01 P7 unit test — price-band decile (synthetic, in-memory). Full H&M regeneration is GATED.
# Run: python Master/temp/dc_checks/dc01/t07_price_band.py
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO)

import pandas as pd

from data.hm.price_band import compute_price_band

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


N_ITEMS = 20
# item_id 0..19 -> article 'A0'..'A19'
item_ids = pd.DataFrame({'item_id': list(range(N_ITEMS)),
                         'item': [f'A{k}' for k in range(N_ITEMS)]})
# transactions: 2 rows per article, mean price increasing with k (so deciles are clean)
tx_rows = []
for k in range(N_ITEMS):
    tx_rows.append({'article_id': f'A{k}', 'price': (k + 1) * 0.01})
    tx_rows.append({'article_id': f'A{k}', 'price': (k + 1) * 0.01 + 0.002})
transactions = pd.DataFrame(tx_rows)

bands = compute_price_band(transactions, item_ids, n_bins=10)

vals = bands['price_band'].tolist()
check("t07.length_eq_n_items", len(bands) == N_ITEMS, f"{len(bands)}")
check("t07.deciles_0_to_9", set(int(v) for v in vals) == set(range(10)), f"{sorted(set(int(v) for v in vals))}")
check("t07.aligned_by_item_id",
      bands['item_id'].tolist() == list(range(N_ITEMS))
      and bands['item'].tolist() == [f'A{k}' for k in range(N_ITEMS)],
      "item_id 0..N-1 ascending, item matches")
# monotone: higher mean price -> higher (or equal) band
mono = all(int(vals[i]) <= int(vals[i + 1]) for i in range(N_ITEMS - 1))
check("t07.monotone_in_price", mono, f"bands={[int(v) for v in vals]}")

# determinism: rebuild (and with shuffled inputs) -> identical
bands2 = compute_price_band(transactions.sample(frac=1.0, random_state=0),
                            item_ids.sample(frac=1.0, random_state=1), n_bins=10)
check("t07.deterministic_under_shuffle",
      bands2['price_band'].tolist() == vals and bands2['item_id'].tolist() == list(range(N_ITEMS)),
      "shuffled tx + item_ids give identical aligned bands")

print("\nt07: ALL PASS" if not FAILS else f"\nt07: {len(FAILS)} FAILED -> {FAILS}")
print("NOTE: full H&M price_band regeneration is GATED (heavy raw join); price_band is OPTIONAL for the base model.")
sys.exit(1 if FAILS else 0)
