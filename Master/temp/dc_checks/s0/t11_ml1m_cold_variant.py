# t11 — generalized cold generator + ml-1m_cold artifact (S0-build extension component e).
# Pins: the generated data/ml-1m_cold against the S0.3-addendum A1/A2 spec (draw magnitudes,
# byte-identical ID universe, no cold positive in variant train/val/test, determinism of the
# seeded draw, A2.3 pool sufficiency incl. the report's new min-pool fields), the loud
# item_features precondition, ColdTestDataset semantics on the real artifact, and registration.
# Requires data/ml-1m_cold (gitignored; regenerate:
#   python data/hm/make_cold_variant.py --canonical data/ml-1m --out data/ml-1m_cold).
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from _harness import make_check, REPO_ROOT

sys.path.insert(0, os.path.join(REPO_ROOT, 'data', 'hm'))
from make_cold_variant import sample_cold_items

from rec_sys.protomf_dataset import ColdTestDataset

check, fails = make_check()

ML = os.path.join(REPO_ROOT, 'data', 'ml-1m')
MLC = os.path.join(REPO_ROOT, 'data', 'ml-1m_cold')
if not os.path.exists(os.path.join(MLC, 'cold_variant_report.json')):
    print("[SKIP] data/ml-1m_cold not present — generate it first")
    sys.exit(0)

report = json.load(open(os.path.join(MLC, 'cold_variant_report.json')))

# --- A1 draw magnitudes (the committed generator's numbers are canonical from here on) ---
check("report: C = 625 of 3,125 eligible (F-S0-04: 0 train-unseen on ml-1m)",
      report['n_cold_items'] == 625 and report['n_items_eligible'] == 3125,
      f"C={report['n_cold_items']}, eligible={report['n_items_eligible']}")
check("report: removed train ~20% (B5 fraction)",
      19.0 <= report['removed_train_pct'] <= 21.0, f"{report['removed_train_pct']}%")
check("report: zero zero-train collateral (heavy histories)",
      report['dropped_zero_train_users'] == 0 and report['dropped_zero_train_rows'] == 0, "")

# --- A2.3 pool sufficiency: report fields + independent recompute ---
check("report: min cold pool >= 100, no user below (A2.3)",
      report['min_cold_pool_over_users'] >= 100 and report['cold_users_with_pool_below_100'] == 0,
      f"min={report['min_cold_pool_over_users']}")
cold_items = pd.read_csv(os.path.join(MLC, 'cold_items.csv'))
cold_test = pd.read_csv(os.path.join(MLC, 'cold_test.csv'))
C = set(cold_items['item_id'])
consumption = pd.concat([pd.read_csv(os.path.join(ML, f'listening_history_{s}.csv'),
                                     usecols=['user_id', 'item_id']) for s in ('train', 'val', 'test')])
in_c = consumption[consumption['item_id'].isin(C)].groupby('user_id')['item_id'].nunique()
pools = len(C) - in_c.reindex(cold_test['user_id'].unique(), fill_value=0)
check("recompute: min pool over cold users matches report (341, ~the addendum's 342 arithmetic)",
      int(pools.min()) == report['min_cold_pool_over_users'] == 341, f"{int(pools.min())}")

# --- ID universe byte-identical (S0.3 §2.4) ---
def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()

for f in ('user_ids.csv', 'item_ids.csv', 'item_features.csv'):
    check(f"byte-identical copy: {f}", md5(os.path.join(ML, f)) == md5(os.path.join(MLC, f)), "")

# --- no cold positive anywhere in the variant splits; cold_test complete ---
ok = True
for s in ('train', 'val', 'test'):
    df = pd.read_csv(os.path.join(MLC, f'listening_history_{s}.csv'), usecols=['item_id'])
    ok = ok and not df['item_id'].isin(C).any()
check("variant train/val/test contain NO cold item", ok, "")
check("cold_test rows == report (112,979)", len(cold_test) == report['cold_test_rows'] == 112979,
      f"{len(cold_test)}")
check("every cold_test item is in C", cold_test['item_id'].isin(C).all(), "")

# --- determinism: seeded re-draw reproduces cold_items.csv exactly ---
train = pd.read_csv(os.path.join(ML, 'listening_history_train.csv'))
redraw, n_eligible = sample_cold_items(train, seed=38210573, fraction=0.2)
check("determinism: re-draw == cold_items.csv (item set + deciles)",
      n_eligible == 3125 and redraw['item_id'].tolist() == cold_items['item_id'].tolist()
      and redraw['pop_decile'].tolist() == cold_items['pop_decile'].tolist(), "")

# --- ColdTestDataset on the real artifact (cold-vs-cold primary) ---
ds = ColdTestDataset(MLC, ranking='cold', n_neg=99)
check("ColdTestDataset: rows == 112,979; pool == C", len(ds) == 112979 and len(ds.pool) == 625, "")
rng = np.random.default_rng(0)
ok = True
for idx in rng.choice(len(ds), 25, replace=False):
    u, items, labels = ds[int(idx)]
    csr = ds.consumption_csr
    consumed = set(csr.indices[csr.indptr[u]:csr.indptr[u + 1]].tolist())
    negs = set(items[1:].tolist())
    ok = ok and items[0] == ds.items[idx] and labels[0] == 1.0 and len(negs) == 99 \
        and negs.issubset(C) and not (negs & consumed)
check("ColdTestDataset: 25 sampled rows — negatives all cold, none canonically consumed",
      ok, "")

# --- loud precondition: missing item_features.csv refuses with the named ordering ---
tmp = tempfile.mkdtemp(prefix='s0_t11_')
for f in ('listening_history_train.csv', 'listening_history_val.csv',
          'listening_history_test.csv', 'user_ids.csv', 'item_ids.csv'):
    shutil.copy2(os.path.join(ML, f), os.path.join(tmp, f))
from make_cold_variant import make_cold_variant
try:
    make_cold_variant(tmp, os.path.join(tmp, 'out'))
    check("precondition: missing item_features.csv fails loud", False, "no raise")
except FileNotFoundError as e:
    check("precondition: missing item_features.csv fails loud (names the ordering)",
          'MODEL-GRADE' in str(e), str(e)[:80])
shutil.rmtree(tmp, ignore_errors=True)

# --- registration ---
start_src = open(os.path.join(REPO_ROOT, 'start.py')).read()
combo_src = open(os.path.join(REPO_ROOT, 'Master', 'scripts', 'run_combo.py')).read()
check("registration: 'ml-1m_cold' in start.py choices", "'ml-1m_cold'" in start_src, "")
check("registration: 'ml-1m_cold' in run_combo VALID_DATASETS", "'ml-1m_cold'" in combo_src, "")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
