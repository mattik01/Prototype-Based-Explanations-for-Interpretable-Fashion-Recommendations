# dc05 t02 — build_user_history_weights (design doc §3.2 data plumbing).
# Pins: (a) w̄ vs hand-computed baskets on a tiny hand-written split; vocabulary/offsets
# identical to build_feature_ids; dedup on (user_id, item_id) keep-first matches the binary
# training matrix; per-field masses sum to 1 per user with history; users absent from train
# get all-zero rows. (b) guards fire on out-of-range ids. (c) VARIANT-TRAIN REBUILD (M6.1a
# leakage pin, review build obligation): on a variant dir the weights come from the VARIANT
# train file — a purchase removed in the variant provably contributes nothing.
# Run: python Master/temp/dc_checks/dc05/t02_history_weights_builder.py
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check

from feature_extraction.feature_ids import build_feature_ids, build_user_history_weights

check, FAILS = make_check()
tmp = tempfile.mkdtemp(prefix='t02_dc05_')
canonical = os.path.join(tmp, 'toy')
os.makedirs(canonical)

# --- tiny hand-written split: 3 users, 4 items, 2 fields ---
with open(os.path.join(canonical, 'user_ids.csv'), 'w') as f:
    f.write('user_id,user\n0,u0\n1,u1\n2,u2\n3,u3\n')      # user 3 has NO train rows
with open(os.path.join(canonical, 'item_ids.csv'), 'w') as f:
    f.write('item_id,item\n0,a0\n1,a1\n2,a2\n3,a3\n')
with open(os.path.join(canonical, 'item_features.csv'), 'w') as f:
    f.write('item_id,item,dep,col\n'
            '0,a0,Blouse,Black\n'
            '1,a1,Trouser,Black\n'
            '2,a2,Blouse,White\n'
            '3,a3,Trouser,Red\n')
# user 0: items 0,1  | user 1: items 0,2,3 (+ a DUPLICATE row for item 0) | user 2: item 3
with open(os.path.join(canonical, 'listening_history_train.csv'), 'w') as f:
    f.write('t_dat,customer_id,article_id,user_id,user,item_id,item\n')
    for u, i in [(0, 0), (0, 1), (1, 0), (1, 0), (1, 2), (1, 3), (2, 3)]:
        f.write(f'2020-09-01,u{u},a{i},{u},u{u},{i},a{i}\n')

FIELDS = ['dep', 'col']
feature_ids, n_features = build_feature_ids(canonical, FIELDS)
ids, w, nf = build_user_history_weights(canonical, FIELDS)

# vocab: dep {Blouse:0, Trouser:1}, col (offset 2) {Black:2, Red:3, White:4}
check("t02.n_features_matches_build_feature_ids", nf == n_features == 5, f"nf={nf}")
check("t02.shapes", ids.shape == w.shape and ids.shape[0] == 4, f"{tuple(ids.shape)}")

def basket(u):
    return {int(ids[u, j]): float(w[u, j]) for j in range(ids.shape[1]) if w[u, j] > 0}

# user 0: H={0,1} -> dep: Blouse 1/2, Trouser 1/2; col: Black 2/2
b0 = basket(0)
check("t02.user0_hand_computed",
      abs(b0.get(0, 0) - 0.5) < 1e-6 and abs(b0.get(1, 0) - 0.5) < 1e-6
      and abs(b0.get(2, 0) - 1.0) < 1e-6 and len(b0) == 3, f"{b0}")
# user 1: H={0,2,3} (duplicate row for item 0 MUST NOT count twice) ->
#   dep: Blouse 2/3, Trouser 1/3; col: Black 1/3, White 1/3, Red 1/3
b1 = basket(1)
check("t02.user1_dedup_keep_first",
      abs(b1.get(0, 0) - 2 / 3) < 1e-6 and abs(b1.get(1, 0) - 1 / 3) < 1e-6
      and abs(b1.get(2, 0) - 1 / 3) < 1e-6 and abs(b1.get(4, 0) - 1 / 3) < 1e-6
      and abs(b1.get(3, 0) - 1 / 3) < 1e-6, f"{b1}")
# user 2: H={3} -> Trouser 1, Red 1
b2 = basket(2)
check("t02.user2_single_item", abs(b2.get(1, 0) - 1.0) < 1e-6 and abs(b2.get(3, 0) - 1.0) < 1e-6
      and len(b2) == 2, f"{b2}")
# user 3: absent from train -> all-zero row
check("t02.absent_user_zero_row", float(w[3].abs().sum()) == 0.0)

# per-field mass sums to F for every user with history (mean normalization invariant)
rowsums = w.sum(dim=1)
check("t02.total_mass_F_per_active_user",
      all(abs(float(rowsums[u]) - 2.0) < 1e-5 for u in (0, 1, 2)),
      f"{[round(float(x), 4) for x in rowsums]}")

# guard: train file referencing an unknown user
bad = os.path.join(tmp, 'bad')
shutil.copytree(canonical, bad)
with open(os.path.join(bad, 'listening_history_train.csv'), 'a') as f:
    f.write('2020-09-01,u9,a0,9,u9,0,a0\n')
try:
    build_user_history_weights(bad, FIELDS)
    guard = False
except ValueError:
    guard = True
check("t02.guard_unknown_user_fires", guard)

# --- (c) variant-train rebuild / leakage pin ---
variant = os.path.join(tmp, 'toy_variant')
shutil.copytree(canonical, variant)
# variant removes user 1's purchase of item 3 (the only 'Red' in their basket)
with open(os.path.join(variant, 'listening_history_train.csv'), 'w') as f:
    f.write('t_dat,customer_id,article_id,user_id,user,item_id,item\n')
    for u, i in [(0, 0), (0, 1), (1, 0), (1, 2), (2, 3)]:
        f.write(f'2020-09-01,u{u},a{i},{u},u{u},{i},a{i}\n')
ids_v, w_v, nf_v = build_user_history_weights(variant, FIELDS)

def basket_v(u):
    return {int(ids_v[u, j]): float(w_v[u, j]) for j in range(ids_v.shape[1]) if w_v[u, j] > 0}

bv1 = basket_v(1)
# variant user 1: H={0,2} -> Blouse 1, Black 1/2, White 1/2 — and NO Red (code 3) mass
check("t02c.variant_weights_from_variant_train",
      abs(bv1.get(0, 0) - 1.0) < 1e-6 and abs(bv1.get(2, 0) - 0.5) < 1e-6
      and abs(bv1.get(4, 0) - 0.5) < 1e-6, f"{bv1}")
check("t02c.removed_row_cannot_leak", 3 not in bv1,
      "code 3 ('col=Red') carries zero weight after variant removal")
check("t02c.untouched_user_unchanged", basket_v(0) == basket(0))

shutil.rmtree(tmp)
print("\nt02: ALL PASS" if not FAILS else f"\nt02: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
