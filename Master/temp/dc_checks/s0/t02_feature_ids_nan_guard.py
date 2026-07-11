# t02 — build_feature_ids NaN guard (F-S0-06): the dc01 builder must raise on NaN cells,
# symmetric with build_attr_multi_hot, instead of silently encoding a phantom 'nan' vocab value.
# Also pins: no behavior change on clean data (identical tensor pre/post guard semantics),
# and the two builders now agree on rejecting the same NaN input.
import os
import shutil
import sys
import tempfile

import torch

from _harness import make_check, REPO_ROOT

from feature_extraction.feature_ids import build_feature_ids, build_attr_multi_hot

check, fails = make_check()

tmp = tempfile.mkdtemp(prefix='s0_t02_')


def write_csv(rows, header='item_id,color,shape'):
    d = tempfile.mkdtemp(prefix='case_', dir=tmp)
    with open(os.path.join(d, 'item_features.csv'), 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    return d


# --- clean data: unchanged behavior ---
clean = write_csv([(0, 'blue', 'square'), (1, 'green', 'square'), (2, 'red', 'circle')])
ids, n_feat = build_feature_ids(clean, ['color', 'shape'])
check("clean data builds", ids.shape == (3, 2) and n_feat == 5, f"shape={tuple(ids.shape)}, n={n_feat}")
# lexicographic vocab: blue=0, green=1, red=2; circle=3, square=4
check("clean data encoding unchanged",
      torch.equal(ids, torch.tensor([[0, 4], [1, 4], [2, 3]])), f"{ids.tolist()}")

# --- NaN cell (empty CSV cell) raises ---
nan_dir = write_csv([(0, 'blue', 'square'), (1, '', 'circle'), (2, 'red', 'circle')])
try:
    build_feature_ids(nan_dir, ['color', 'shape'])
    check("NaN cell raises ValueError", False)
except ValueError as e:
    check("NaN cell raises ValueError", True, str(e)[:70])

# --- literal string 'nan' in the CSV: pd.read_csv parses it as NaN (default na_values),
# so it is indistinguishable from a missing cell and MUST also raise — in BOTH builders
# (this is exactly the phantom-'nan'-vocab-value scenario the guard exists to catch) ---
lit = write_csv([(0, 'nan', 'square'), (1, 'blue', 'circle')])
lit_raises = 0
for builder in (build_feature_ids, build_attr_multi_hot):
    try:
        builder(lit, ['color', 'shape'])
    except ValueError:
        lit_raises += 1
check("literal 'nan' cell raises in both builders (pandas parses it as NaN)",
      lit_raises == 2, f"{lit_raises}/2 raised")

# --- symmetry: both builders reject the same NaN input ---
sym = 0
for builder, name in [(build_feature_ids, 'build_feature_ids'), (build_attr_multi_hot, 'build_attr_multi_hot')]:
    try:
        builder(nan_dir, ['color', 'shape'])
    except ValueError:
        sym += 1
check("builders symmetric on NaN input (both raise)", sym == 2, f"{sym}/2 raised")

# --- real data: V1 canonical fields still clean (0 NaN) ---
V1_FIELDS = ['department_name', 'product_type_name', 'section_name',
             'colour_group_name', 'graphical_appearance_name']
v1 = os.path.join(REPO_ROOT, 'data', 'hm_1_month')
if os.path.exists(os.path.join(v1, 'item_features.csv')):
    ids_v1, n_feat_v1 = build_feature_ids(v1, V1_FIELDS)
    check("hm_1_month: builds under guard (0 NaN)", ids_v1.shape == (13651, 5), f"{tuple(ids_v1.shape)}")
    check("hm_1_month: canonical 5-field vocab V == 426", n_feat_v1 == 426, f"{n_feat_v1}")
else:
    print("[SKIP] hm_1_month item_features.csv not present")

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
