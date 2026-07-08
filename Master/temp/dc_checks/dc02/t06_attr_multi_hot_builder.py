# t06 — build_attr_multi_hot: CSV -> multi-hot X + field_offsets + column names (dc02 P2),
# plus the inject_feature_ids attr branch (non-mutating copy discipline) and real-data sanity.
import os
import shutil
import sys
import tempfile

import torch

from _harness import make_check, REPO_ROOT

from feature_extraction.feature_ids import build_attr_multi_hot, inject_feature_ids

check, fails = make_check()

tmp = tempfile.mkdtemp(prefix='dc02_t06_')


def write_csv(rows, header='item_id,item,color,shape', name='item_features.csv', dirpath=None):
    d = dirpath or tempfile.mkdtemp(prefix='dc02_t06_case_', dir=tmp)
    with open(os.path.join(d, name), 'w') as f:
        f.write(header + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    return d


# --- synthetic: 4 items, 2 fields; rows deliberately SHUFFLED on disk ---
rows = [
    (2, 'a2', 'red', 'circle'),
    (0, 'a0', 'blue', 'square'),
    (3, 'a3', 'blue', 'circle'),
    (1, 'a1', 'green', 'square'),
]
d = write_csv(rows)
X, offs, cols = build_attr_multi_hot(d, ['color', 'shape'])

check("shape (M, V)", X.shape == (4, 5), f"{tuple(X.shape)}")   # colors {blue,green,red}=3 + shapes {circle,square}=2
check("dtype float32", X.dtype == torch.float32)
check("field_offsets", offs == [0, 3, 5], f"{offs}")
check("row sums == F", bool((X.sum(dim=1) == 2.0).all()))
# lexicographic vocab: colors -> blue=0, green=1, red=2; shapes -> circle=3, square=4
check("column meta lexicographic + field-tagged",
      cols == [('color', 'blue'), ('color', 'green'), ('color', 'red'),
               ('shape', 'circle'), ('shape', 'square')], f"{cols}")
# shuffled rows realign by item_id: item 0 is blue/square -> columns 0 and 4
expected = torch.tensor([
    [1., 0., 0., 0., 1.],   # 0: blue square
    [0., 1., 0., 0., 1.],   # 1: green square
    [0., 0., 1., 1., 0.],   # 2: red circle
    [1., 0., 0., 1., 0.],   # 3: blue circle
])
check("shuffled rows realigned by item_id", torch.equal(X, expected))
# exactly one 1 inside each field block
check("one-hot per field block",
      bool((X[:, 0:3].sum(dim=1) == 1.0).all()) and bool((X[:, 3:5].sum(dim=1) == 1.0).all()))

# --- determinism ---
X2, offs2, cols2 = build_attr_multi_hot(d, ['color', 'shape'])
check("deterministic rebuild", torch.equal(X, X2) and offs == offs2 and cols == cols2)

# --- field order defines block order ---
Xr, offsr, colsr = build_attr_multi_hot(d, ['shape', 'color'])
check("field order respected", offsr == [0, 2, 5] and colsr[0][0] == 'shape')

# --- guards ---
d_missing = write_csv(rows[:3])                                   # item 1 missing
try:
    build_attr_multi_hot(d_missing, ['color', 'shape'])
    check("missing item_id raises", False)
except ValueError:
    check("missing item_id raises", True)

d_dup = write_csv(rows + [(2, 'a2b', 'red', 'circle')])           # duplicate item 2
try:
    build_attr_multi_hot(d_dup, ['color', 'shape'])
    check("duplicate item_id raises", False)
except ValueError:
    check("duplicate item_id raises", True)

d_nan = write_csv([(0, 'a0', 'blue', 'square'), (1, 'a1', '', 'circle')])   # empty cell -> NaN
try:
    build_attr_multi_hot(d_nan, ['color', 'shape'])
    check("NaN cell raises", False)
except ValueError:
    check("NaN cell raises", True)

try:
    build_attr_multi_hot(d, ['color', 'no_such_field'])
    check("missing field raises", False)
except ValueError:
    check("missing field raises", True)

try:
    build_attr_multi_hot(os.path.join(tmp, 'nowhere'), ['color'])
    check("missing csv raises FileNotFoundError", False)
except FileNotFoundError:
    check("missing csv raises FileNotFoundError", True)

# --- inject_feature_ids: attr branch, non-mutating copy discipline ---
spec = {
    'ft_type': 'attr_item_proto', 'embedding_dim': 8,
    'user_ft_ext_param': {'ft_type': 'attr_item_proto', 'n_prototypes': 4},
    'item_ft_ext_param': {'ft_type': 'attr_item_proto', 'n_prototypes': 5,
                          'attr_fields': ['color', 'shape']},
}
out = inject_feature_ids(spec, d)
check("inject returns a new object", out is not spec)
check("inject adds attr_multi_hot to the copy", torch.equal(out['item_ft_ext_param']['attr_multi_hot'], X))
check("inject adds n_attr_values + field_offsets",
      out['item_ft_ext_param']['n_attr_values'] == 5 and out['item_ft_ext_param']['field_offsets'] == [0, 3, 5])
check("caller item dict NOT mutated (no tensor leak into config)",
      'attr_multi_hot' not in spec['item_ft_ext_param'] and 'field_offsets' not in spec['item_ft_ext_param'])
check("user sub-dict copied (factory mutates it in place)",
      out['user_ft_ext_param'] is not spec['user_ft_ext_param'])

foreign = {'ft_type': 'prototypes_double_tie', 'item_ft_ext_param': {}, 'user_ft_ext_param': {}}
check("foreign ft_type returned unchanged (same object)", inject_feature_ids(foreign, d) is foreign)

# --- real data: hm_3_month ---
HM_FIELDS = ['product_group_name', 'product_type_name', 'graphical_appearance_name',
             'colour_group_name', 'perceived_colour_master_name', 'index_group_name',
             'garment_group_name', 'section_name', 'department_name']
hm3 = os.path.join(REPO_ROOT, 'data', 'hm_3_month')
if os.path.exists(os.path.join(hm3, 'item_features.csv')):
    Xh, offsh, colsh = build_attr_multi_hot(hm3, HM_FIELDS)
    card = [offsh[i + 1] - offsh[i] for i in range(len(HM_FIELDS))]
    check("hm_3_month: M == 26963", Xh.shape[0] == 26963, f"{Xh.shape[0]}")
    check("hm_3_month: V == 534", Xh.shape[1] == 534, f"{Xh.shape[1]}")
    check("hm_3_month: per-field cardinalities",
          card == [17, 118, 29, 49, 20, 5, 21, 54, 221], f"{card}")
    check("hm_3_month: all row sums == 9", bool((Xh.sum(dim=1) == 9.0).all()))
    check("hm_3_month: column meta length == V", len(colsh) == 534)
else:
    print("[SKIP] hm_3_month item_features.csv not present")

hmf = os.path.join(REPO_ROOT, 'data', 'hm_full')
if os.path.exists(os.path.join(hmf, 'item_features.csv')):
    Xf, offsf, _ = build_attr_multi_hot(hmf, HM_FIELDS)
    check("hm_full: M == 90690", Xf.shape[0] == 90690, f"{Xf.shape[0]}")
    check("hm_full: all row sums == 9", bool((Xf.sum(dim=1) == 9.0).all()))
    check("hm_full: V within expected range", 534 <= Xf.shape[1] <= 600, f"V={Xf.shape[1]}")
else:
    print("[SKIP] hm_full item_features.csv not present")

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
