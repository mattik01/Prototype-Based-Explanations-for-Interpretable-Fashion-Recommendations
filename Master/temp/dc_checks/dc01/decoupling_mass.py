# decoupling_mass.py — F-DC01-06's committed counting script (dc01 design doc §3.4
# "Share identifiability across correlated fields"; plain-language source: vault
# 2026-07-11_2310).
#
# For a pair of feature values a (field X), b (field Y != X) that co-occur on items, the
# rendered split of their combined share is determined only by the loss curvature along the
# redistribution direction e_a -> e_b (sum fixed). That curvature is generated ONLY by items
# where the pair decouples (carry exactly one of the two values) — so the pair's DECOUPLING
# MASS in the catalog is the first-order determinacy proxy:
#
#   decoupling(a,b)      = |items with a but not b| + |items with b but not a|
#   norm_decoupling(a,b) = decoupling / |items with a or b|          in [0,1]
#     0   -> perfect lockstep: the split is pure gauge (frozen initialization noise)
#     ->1 -> the pair is almost never composed together; the split is well determined
#
# Reported at two levels (threshold-free — the continuum is the result, per the
# "production decides; research characterizes" principle, learnings ledger):
#   VALUE level — every item-observed cross-field value pair (where the mechanism lives;
#     two fields can be weakly associated overall while one value pair is in perfect lockstep).
#   FIELD level — presentation summary: item-weighted mean normalized decoupling of each
#     item's (X,Y) value pair ("for a random catalog item, how determined is its X-vs-Y
#     split"), plus lockstep counts (counts, not cutoffs).
#
# Usage:
#   python decoupling_mass.py [--data-dir ../../../../data/hm_1_month] [--out-dir decoupling_out]
#   python decoupling_mass.py --selftest
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

CANONICAL_FIELDS = ['department_name', 'product_type_name', 'section_name',
                    'colour_group_name', 'graphical_appearance_name']


def value_level(df: pd.DataFrame, fields) -> pd.DataFrame:
    """One row per item-observed cross-field value pair, with decoupling masses."""
    rows = []
    for fx, fy in itertools.combinations(fields, 2):
        ct = pd.crosstab(df[fx], df[fy])                      # co-occurrence counts
        n_x = df[fx].value_counts()
        n_y = df[fy].value_counts()
        both = ct.stack()
        both = both[both > 0]                                 # observed pairs only
        for (a, b), n_both in both.items():
            na, nb = int(n_x[a]), int(n_y[b])
            dec = (na - int(n_both)) + (nb - int(n_both))
            union = na + nb - int(n_both)
            rows.append({'field_x': fx, 'value_x': a, 'field_y': fy, 'value_y': b,
                         'n_x': na, 'n_y': nb, 'n_both': int(n_both),
                         'decoupling_mass': dec,
                         'norm_decoupling': dec / union})
    out = pd.DataFrame(rows).sort_values(
        ['norm_decoupling', 'n_both'], ascending=[True, False]).reset_index(drop=True)
    return out


def field_level(df: pd.DataFrame, values: pd.DataFrame, fields) -> pd.DataFrame:
    """Presentation summary per field pair: item-weighted mean normalized decoupling of each
    item's own (X,Y) value pair + lockstep counts."""
    key = values.set_index(['field_x', 'value_x', 'field_y', 'value_y'])['norm_decoupling']
    rows = []
    for fx, fy in itertools.combinations(fields, 2):
        nd = key.loc[fx, :, fy, :]                            # observed pairs of this field pair
        item_nd = df.apply(lambda r: key.loc[(fx, r[fx], fy, r[fy])], axis=1).to_numpy()
        pair_slice = values[(values.field_x == fx) & (values.field_y == fy)]
        lock = pair_slice[pair_slice.decoupling_mass == 0]
        rows.append({'field_x': fx, 'field_y': fy,
                     'n_observed_value_pairs': len(pair_slice),
                     'item_weighted_mean_norm_decoupling': float(item_nd.mean()),
                     'item_median_norm_decoupling': float(np.median(item_nd)),
                     'n_lockstep_value_pairs': len(lock),
                     'n_items_on_lockstep_pairs': int(lock['n_both'].sum()),
                     'share_items_on_lockstep_pairs': float(lock['n_both'].sum() / len(df))})
    return pd.DataFrame(rows).sort_values(
        'item_weighted_mean_norm_decoupling').reset_index(drop=True)


def run(data_dir: str, out_dir: str, fields=None):
    fields = fields or CANONICAL_FIELDS
    df = pd.read_csv(os.path.join(data_dir, 'item_features.csv'))
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(f'fields missing from item_features.csv: {missing}')
    vals = value_level(df, fields)
    flds = field_level(df, vals, fields)

    os.makedirs(out_dir, exist_ok=True)
    vals.to_csv(os.path.join(out_dir, 'decoupling_value_level.csv'), index=False)
    flds.to_csv(os.path.join(out_dir, 'decoupling_field_level.csv'), index=False)

    n_lock = int((vals.decoupling_mass == 0).sum())
    print(f'\n=== decoupling mass — {os.path.basename(os.path.normpath(data_dir))} '
          f'({len(df)} items, {len(fields)} fields) ===')
    print(f'observed cross-field value pairs: {len(vals)}; '
          f'perfect-lockstep pairs (mass 0): {n_lock}')
    print('\nFIELD level (item-weighted; low = splits weakly determined):')
    print(flds.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nVALUE level — 15 least-determined pairs with the largest shared mass:')
    head = vals[vals.n_both >= vals.n_both.quantile(0.5)].head(15)
    print(head.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'\nCSVs written to {out_dir}')
    return vals, flds


def selftest():
    """Synthetic fixture: a strictly lockstep pair must read mass 0; an independent pair
    must read high normalized decoupling."""
    n = 400
    rng = np.random.default_rng(0)
    dep = np.where(np.arange(n) % 2 == 0, 'Denim', 'Knit')     # lockstep with ptype
    ptype = np.where(dep == 'Denim', 'Jeans', 'Sweater')       # deterministic partner
    colour = rng.choice(['Black', 'Blue'], n)                  # independent of dep
    df = pd.DataFrame({'item_id': range(n), 'department_name': dep,
                       'product_type_name': ptype, 'colour_group_name': colour})
    vals = value_level(df, ['department_name', 'product_type_name', 'colour_group_name'])
    lock = vals[(vals.value_x == 'Denim') & (vals.value_y == 'Jeans')]
    assert len(lock) == 1 and lock.iloc[0].decoupling_mass == 0, 'lockstep pair must read 0'
    ind = vals[(vals.value_x == 'Denim') & (vals.value_y == 'Black')]
    assert len(ind) == 1 and ind.iloc[0].norm_decoupling > 0.5, \
        'independent pair must read high decoupling'
    flds = field_level(df, vals, ['department_name', 'product_type_name', 'colour_group_name'])
    dp = flds[(flds.field_x == 'department_name') & (flds.field_y == 'product_type_name')]
    assert float(dp.iloc[0].item_weighted_mean_norm_decoupling) == 0., \
        'fully-lockstep field pair must read 0 at field level'
    print('SELFTEST PASS (lockstep -> 0; independent -> high; field roll-up consistent)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='F-DC01-06 decoupling-mass counter')
    ap.add_argument('--data-dir', default=os.path.join(REPO_ROOT, 'data', 'hm_1_month'))
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__) or '.',
                                                      'decoupling_out'))
    ap.add_argument('--fields', nargs='*', default=None)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        sys.exit(0)
    run(args.data_dir, args.out_dir, args.fields)
