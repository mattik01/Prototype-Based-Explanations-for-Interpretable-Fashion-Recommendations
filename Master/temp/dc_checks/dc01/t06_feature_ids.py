# dc01 P6 unit test — build_feature_ids (field-offset encoding + alignment guard + determinism).
# Run: python Master/temp/dc_checks/dc01/t06_feature_ids.py
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import pandas as pd

from feature_extraction.feature_ids import build_feature_ids, inject_feature_ids

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


def write_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


FIELDS = ['fieldA', 'fieldB']
BASE_ROWS = [
    {'item_id': 0, 'item': 'a0', 'fieldA': 'x', 'fieldB': 'p'},
    {'item_id': 1, 'item': 'a1', 'fieldA': 'y', 'fieldB': 'p'},
    {'item_id': 2, 'item': 'a2', 'fieldA': 'x', 'fieldB': 'q'},
]
# fieldA sorted vocab ['x','y'] -> x=0,y=1 (offset 0); fieldB sorted ['p','q'] -> p=2,q=3 (offset 2)
EXPECTED = torch.tensor([[0, 2], [1, 2], [0, 3]], dtype=torch.long)

with tempfile.TemporaryDirectory() as d:
    csv = os.path.join(d, 'item_features.csv')
    write_csv(BASE_ROWS, csv)
    feature_ids, n_features = build_feature_ids(d, FIELDS)

    check("t06.shape", tuple(feature_ids.shape) == (3, 2), f"{tuple(feature_ids.shape)}")
    check("t06.dtype_long", feature_ids.dtype == torch.long, f"{feature_ids.dtype}")
    check("t06.field_offset_correct", torch.equal(feature_ids, EXPECTED), f"{feature_ids.tolist()}")
    check("t06.n_features_sum_of_vocabs", n_features == 4, f"n_features={n_features} (expected 2+2=4)")

    # determinism: build twice -> bit-identical
    f2, n2 = build_feature_ids(d, FIELDS)
    check("t06.determinism", torch.equal(feature_ids, f2) and n_features == n2, "")

# alignment: shuffled rows still align by item_id
with tempfile.TemporaryDirectory() as d:
    csv = os.path.join(d, 'item_features.csv')
    write_csv([BASE_ROWS[2], BASE_ROWS[0], BASE_ROWS[1]], csv)  # shuffled
    f_shuf, _ = build_feature_ids(d, FIELDS)
    check("t06.alignment_by_item_id", torch.equal(f_shuf, EXPECTED), f"{f_shuf.tolist()}")

# missing item_id raises
with tempfile.TemporaryDirectory() as d:
    csv = os.path.join(d, 'item_features.csv')
    write_csv([BASE_ROWS[0], BASE_ROWS[2]], csv)  # item_ids {0,2}, n_rows=2 -> not 0..1
    try:
        build_feature_ids(d, FIELDS)
        check("t06.missing_item_id_raises", False, "no exception raised")
    except ValueError as e:
        check("t06.missing_item_id_raises", True, f"raised ValueError: {str(e)[:60]}")

# duplicate item_id raises
with tempfile.TemporaryDirectory() as d:
    csv = os.path.join(d, 'item_features.csv')
    dup = dict(BASE_ROWS[1]); dup['item_id'] = 0
    write_csv([BASE_ROWS[0], dup, BASE_ROWS[2]], csv)  # item_ids {0,0,2}
    try:
        build_feature_ids(d, FIELDS)
        check("t06.duplicate_item_id_raises", False, "no exception raised")
    except ValueError as e:
        check("t06.duplicate_item_id_raises", True, f"raised ValueError: {str(e)[:60]}")

# unknown field raises
with tempfile.TemporaryDirectory() as d:
    csv = os.path.join(d, 'item_features.csv')
    write_csv(BASE_ROWS, csv)
    try:
        build_feature_ids(d, ['fieldA', 'nope'])
        check("t06.unknown_field_raises", False, "no exception raised")
    except ValueError as e:
        check("t06.unknown_field_raises", True, f"raised ValueError: {str(e)[:60]}")

# --- inject_feature_ids non-mutation contract (tensor must NOT persist in caller's config) ---
with tempfile.TemporaryDirectory() as d:
    write_csv(BASE_ROWS, os.path.join(d, 'item_features.csv'))
    cfg = {'ft_type': 'feature_item_proto', 'embedding_dim': 8,
           'user_ft_ext_param': {'ft_type': 'feature_item_proto', 'n_prototypes': 3},
           'item_ft_ext_param': {'ft_type': 'feature_item_proto', 'n_prototypes': 4,
                                 'use_id_feature': True, 'feature_fields': FIELDS}}
    out = inject_feature_ids(cfg, d)
    check("t06.inject_returns_tensor",
          'feature_ids' in out['item_ft_ext_param'] and 'n_features' in out['item_ft_ext_param'],
          "returned param carries feature_ids/n_features for the factory")
    check("t06.inject_does_not_mutate_caller",
          'feature_ids' not in cfg['item_ft_ext_param'] and 'n_features' not in cfg['item_ft_ext_param'],
          "caller's config keeps only the spec (tensor would otherwise leak into config.json)")
    check("t06.inject_returns_new_object", out is not cfg and out['item_ft_ext_param'] is not cfg['item_ft_ext_param'], "")

# non-feature ft_type returns the SAME object unchanged (existing models: factory mutates in place)
other = {'ft_type': 'prototypes_double_tie', 'item_ft_ext_param': {}, 'user_ft_ext_param': {}}
check("t06.inject_noop_for_other_fttype", inject_feature_ids(other, '/nonexistent') is other,
      "returns same object for non-feature_item_proto")

# --- REAL H&M data (hm_3_month locally; hm_1_month after P0 on the cluster) ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
real_dir = os.path.join(repo_root, 'data', 'hm_3_month')
real_fields = ['department_name', 'product_type_name', 'section_name',
               'colour_group_name', 'graphical_appearance_name']
if os.path.exists(os.path.join(real_dir, 'item_features.csv')):
    f_real, n_real = build_feature_ids(real_dir, real_fields)
    n_items = f_real.shape[0]
    check("t06.real_rows", n_items > 10000, f"n_items={n_items} (hm_3_month ~26963)")
    check("t06.real_shape", f_real.shape[1] == len(real_fields), f"F={f_real.shape[1]}")
    check("t06.real_no_nan_codes", bool((f_real >= 0).all()) and bool((f_real < n_real).all()),
          f"min={int(f_real.min())}, max={int(f_real.max())}, n_features={n_real}")
else:
    print("[SKIP] t06.real_* — data/hm_3_month/item_features.csv not present")

print("\nt06: ALL PASS" if not FAILS else f"\nt06: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
