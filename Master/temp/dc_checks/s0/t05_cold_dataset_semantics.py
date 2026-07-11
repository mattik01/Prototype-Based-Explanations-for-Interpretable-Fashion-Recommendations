# t05 — dataset-side cold-variant semantics (S0.3 §3/§4): training-negative exclusion of C
# (leakage item 6), eval negatives keep the full catalog, the F-S0-03 one-row-per-user
# invariant (fatal on canonical, relaxed on the marked variant), and ColdTestDataset
# (pool membership, canonical-consumption exclusion, determinism, labels).
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from _harness import make_check, build_toy_canonical, REPO_ROOT

sys.path.insert(0, os.path.join(REPO_ROOT, 'data', 'hm'))
from make_cold_variant import make_cold_variant

from rec_sys.protomf_dataset import ColdTestDataset, ProtoRecDataset

check, fails = make_check()

tmp = tempfile.mkdtemp(prefix='s0_t05_')
canon = os.path.join(tmp, 'toy')
variant = os.path.join(tmp, 'toy_cold')
build_toy_canonical(canon)
make_cold_variant(canon, variant)

C = set(pd.read_csv(os.path.join(variant, 'cold_items.csv'))['item_id'])

# --- 1. training negatives on the variant NEVER draw a cold item ---
np.random.seed(0)
train_ds = ProtoRecDataset(variant, 'train', n_neg=20, neg_strategy='uniform')
check("variant train dataset detects cold_items.csv",
      train_ds.neg_exclude_items is not None and len(train_ds.neg_exclude_items) == len(C))
drawn = set()
for idx in range(0, len(train_ds), max(1, len(train_ds) // 60)):
    _, item_idxs, _ = train_ds[idx]
    drawn.update(item_idxs[1:].tolist())
check("uniform training negatives avoid C", not (drawn & C), f"{len(drawn)} distinct negatives drawn")

np.random.seed(0)
train_pop = ProtoRecDataset(variant, 'train', n_neg=20, neg_strategy='popular')
drawn_p = set()
for idx in range(0, len(train_pop), max(1, len(train_pop) // 60)):
    _, item_idxs, _ = train_pop[idx]
    drawn_p.update(item_idxs[1:].tolist())
check("popular training negatives avoid C", not (drawn_p & C))

# --- 2. eval splits keep the FULL catalog (no exclusion mask) ---
val_ds = ProtoRecDataset(variant, 'val', n_neg=20, neg_strategy='uniform')
check("variant val dataset has NO exclusion mask", val_ds.neg_exclude_items is None)
np.random.seed(1)
drawn_v = set()
for idx in range(len(val_ds)):
    _, item_idxs, _ = val_ds[idx]
    drawn_v.update(item_idxs[1:].tolist())
check("val negatives CAN include cold items (full catalog)", bool(drawn_v & C),
      f"{len(drawn_v & C)} cold items appeared")

# --- 3. canonical dataset with NO exclusion (unchanged behavior) ---
canon_tr = ProtoRecDataset(canon, 'train', n_neg=5)
check("canonical train has no exclusion mask", canon_tr.neg_exclude_items is None)

# --- 4. F-S0-03 invariant: fatal on a broken canonical split, relaxed on the variant ---
broken = os.path.join(tmp, 'broken')
shutil.copytree(canon, broken)
va = pd.read_csv(os.path.join(broken, 'listening_history_val.csv'))
va.iloc[:-3].to_csv(os.path.join(broken, 'listening_history_val.csv'), index=False)
try:
    ProtoRecDataset(broken, 'val', n_neg=5)
    check("broken canonical val (rows != n_users) raises", False)
except AssertionError as e:
    check("broken canonical val (rows != n_users) raises", True, str(e)[:60])
check("variant val (rows != n_users, marker present) constructs fine",
      val_ds.coo_matrix.nnz < val_ds.n_users)

# --- 5. ColdTestDataset semantics ---
consumption = {}
for s in ('train', 'val', 'test'):
    df = pd.read_csv(os.path.join(canon, f'listening_history_{s}.csv'))
    for u, i in zip(df['user_id'], df['item_id']):
        consumption.setdefault(u, set()).add(i)

cold_test = pd.read_csv(os.path.join(variant, 'cold_test.csv'))
ds_cold = ColdTestDataset(variant, ranking='cold', n_neg=20, seed=123)
check("cold rows count matches cold_test.csv", len(ds_cold) == len(cold_test))

ok_pos, ok_pool, ok_cons, ok_labels = True, True, True, True
for idx in range(len(ds_cold)):
    u, item_idxs, labels = ds_cold[idx]
    pos, negs = int(item_idxs[0]), item_idxs[1:].tolist()
    ok_pos &= pos in C and pos == cold_test['item_id'].iloc[idx]
    ok_pool &= all(n in C for n in negs)
    ok_cons &= not (set(negs) & consumption[int(u)])
    ok_labels &= labels[0] == 1. and labels[1:].sum() == 0.
check("positives are the cold_test rows (in order)", ok_pos)
check("cold-vs-cold negatives all from C", ok_pool)
check("no canonical-consumed item as negative (leakage item 5)", ok_cons)
check("labels: [1, 0...0]", ok_labels)

ds_all = ColdTestDataset(variant, ranking='all', n_neg=20, seed=123)
outside = False
for idx in range(len(ds_all)):
    _, item_idxs, _ = ds_all[idx]
    outside |= any(n not in C for n in item_idxs[1:].tolist())
check("cold-vs-all negatives include warm items (full-catalog pool)", outside)

# determinism: fresh instance, same seed -> identical draws
d1 = ColdTestDataset(variant, ranking='cold', n_neg=20, seed=123)
d2 = ColdTestDataset(variant, ranking='cold', n_neg=20, seed=123)
check("seeded determinism across fresh instances (sequential iteration)",
      all(np.array_equal(d1[i][1], d2[i][1]) for i in range(len(d1))))

# canonical-path inference guard
try:
    ColdTestDataset(os.path.join(tmp, 'not_suffixed'), ranking='cold')
    check("non-'_cold' dir without explicit canonical_path raises", False)
except AssertionError:
    check("non-'_cold' dir without explicit canonical_path raises", True)

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
