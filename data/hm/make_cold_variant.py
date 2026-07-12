"""Cold-variant generator (Scrutiny S0.3 §2) — derive `<dataset>_cold` from a frozen canonical split.

Derivation is deterministic and NEVER re-splits from raw or re-k-cores (the frozen artifact is the
single source; re-coring would silently reshape the warm regime — S0.3 leakage item 3):

1. Item train-popularity from the canonical train split; eligible = items with >= 1 train
   interaction. Eligible items are ordered by (popularity, item_id) ascending and cut into 10
   equal-frequency deciles (``np.array_split``); ``fraction`` (default 20%, B5/LightFM §5) of each
   decile is drawn without replacement with ``numpy.random.default_rng(seed)`` (default
   38210573), deciles processed in ascending-popularity order -> cold set C. (Stratification is a
   variance-reduction refinement of B5's uniform draw — identical marginal inclusion
   probability.)
2. Variant train/val := canonical minus rows with item in C; removed train/val rows plus the
   canonical test rows with item in C become ``cold_test.csv`` (tagged with their original
   split); variant test := canonical test minus cold rows (the warm test).
3. Cold rows of users left with ZERO variant-train interactions are dropped (an item-side eval
   cannot score a user with no warm history — S0.2); users falling below 3 train rows are
   counted and reported, not dropped (uniform across all models).
4. ID universe unchanged: ``user_ids.csv`` / ``item_ids.csv`` / ``item_features.csv`` are copied
   byte-identically — same n_users/n_items, same integer ids, same feature tensors.

Outputs in --out: the 3 reduced/warm listening_history files, the 3 byte-identical ID/feature
copies, ``cold_items.csv`` (item_id, item, train_pop, pop_decile), ``cold_test.csv``
(user_id, item_id, orig_split), and ``cold_variant_report.json`` (the collateral report, also
printed). Training-time negative-sampling exclusion of C is handled downstream by
``ProtoRecDataset`` detecting ``cold_items.csv`` in the data dir (S0.3 leakage item 6).

DATASET-GENERIC (S0.3 addendum A2, 2026-07-12): the machinery reads only the generic split
files (``listening_history_*`` with ``user_id``/``item_id`` columns + the 3 ID/feature files),
so any canonical split dir works — only the DEFAULTS are H&M. The script stays in ``data/hm/``
(provenance; moving it would churn history for zero behavior gain). Precondition (ordering
inside the S0-build extension): the MODEL-GRADE ``item_features.csv`` must exist in the
canonical dir BEFORE generation — it is copied byte-identically into the variant and feature
models build their vocab from the variant's own copy. The report includes
``min_cold_pool_over_users`` (A2.3): the worst-case per-user cold-vs-cold negative pool
|C \\ consumed_u| — must be >= NEG_VAL+1 = 100 for the primary ranking (heavy-history hazard on
ml-1m: |C| ~ 625 vs |H_u| up to 1,415); ``ColdTestDataset`` additionally asserts it per row at
eval time.

Usage:
    python data/hm/make_cold_variant.py                       # hm_1_month -> hm_1_month_cold
    python data/hm/make_cold_variant.py --canonical data/hm_1_month --out data/hm_1_month_cold
    python data/hm/make_cold_variant.py --canonical data/ml-1m --out data/ml-1m_cold
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

DEFAULT_SEED = 38210573
DEFAULT_FRACTION = 0.2


def sample_cold_items(train_lhs: pd.DataFrame, seed: int = DEFAULT_SEED,
                      fraction: float = DEFAULT_FRACTION):
    """Popularity-stratified cold-item draw (S0.3 §2.1). Returns (cold_items DataFrame with
    item_id/train_pop/pop_decile, eligible_count)."""
    pop = train_lhs.groupby('item_id').size()
    # deterministic order: ascending popularity, item_id as tiebreak
    eligible = pop.reset_index(name='train_pop').sort_values(
        ['train_pop', 'item_id']).reset_index(drop=True)
    deciles = np.array_split(np.arange(len(eligible)), 10)

    rng = np.random.default_rng(seed)
    rows = []
    for d, idx in enumerate(deciles):
        n_take = int(round(fraction * len(idx)))
        take = rng.choice(idx, size=n_take, replace=False)
        block = eligible.iloc[np.sort(take)].copy()
        block['pop_decile'] = d
        rows.append(block)
    cold = pd.concat(rows).sort_values('item_id').reset_index(drop=True)
    return cold, len(eligible)


def make_cold_variant(canonical: str, out: str, seed: int = DEFAULT_SEED,
                      fraction: float = DEFAULT_FRACTION) -> dict:
    # Precondition (loud, named — S0.3 addendum A2.2): the ID/feature files are copied
    # byte-identically into the variant; item_features.csv must be the MODEL-GRADE build
    # (for ml-1m: genres + tags@0.8 via data/ml-1m/build_item_features.py), generated BEFORE
    # this script runs — a variant generated from a missing/display-grade file would freeze
    # the wrong feature universe into the cold artifact.
    for f in ('user_ids.csv', 'item_ids.csv', 'item_features.csv'):
        p = os.path.join(canonical, f)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{p} not found — the cold variant copies it byte-identically; for '
                f'item_features.csv build the MODEL-GRADE file first (S0-build extension '
                f'ordering: features builder (d) before cold generator (e))')

    train = pd.read_csv(os.path.join(canonical, 'listening_history_train.csv'))
    val = pd.read_csv(os.path.join(canonical, 'listening_history_val.csv'))
    test = pd.read_csv(os.path.join(canonical, 'listening_history_test.csv'))
    item_ids = pd.read_csv(os.path.join(canonical, 'item_ids.csv'))

    cold, n_eligible = sample_cold_items(train, seed, fraction)
    C = set(cold['item_id'].tolist())

    # 2. split reduction; removed rows -> cold_test (tagged)
    parts = []
    reduced = {}
    for name, df in [('train', train), ('val', val), ('test', test)]:
        mask = df['item_id'].isin(C)
        part = df.loc[mask, ['user_id', 'item_id']].copy()
        part['orig_split'] = name
        parts.append(part)
        reduced[name] = df.loc[~mask].reset_index(drop=True)
    cold_test = pd.concat(parts, ignore_index=True)

    # 3. drop cold rows of users with ZERO variant-train interactions; count <3-row users
    users_with_train = set(reduced['train']['user_id'].unique())
    zero_mask = ~cold_test['user_id'].isin(users_with_train)
    n_zero_users = cold_test.loc[zero_mask, 'user_id'].nunique()
    n_zero_rows = int(zero_mask.sum())
    cold_test = cold_test.loc[~zero_mask].reset_index(drop=True)

    train_counts = reduced['train'].groupby('user_id').size()
    n_users = len(pd.read_csv(os.path.join(canonical, 'user_ids.csv')))
    n_below3 = int((train_counts < 3).sum())

    # Cold-pool sufficiency (S0.3 addendum A2.3): the cold-vs-cold negative pool for user u is
    # |C \ consumed_u| (ColdTestDataset excludes the user's CANONICAL train+val+test consumption
    # from the pool). Report the minimum over cold-test users — must be >= NEG_VAL+1 = 100 for
    # the primary ranking; the dataset asserts it per row at eval time (fails loud).
    consumption = pd.concat([train[['user_id', 'item_id']], val[['user_id', 'item_id']],
                             test[['user_id', 'item_id']]], ignore_index=True)
    consumed_in_C = consumption[consumption['item_id'].isin(C)].groupby('user_id')['item_id'].nunique()
    cold_users = cold_test['user_id'].unique()
    per_user_pool = len(C) - consumed_in_C.reindex(cold_users, fill_value=0)
    min_cold_pool = int(per_user_pool.min()) if len(cold_users) else len(C)
    n_pool_below_100 = int((per_user_pool < 100).sum())

    # 4. write the variant
    os.makedirs(out, exist_ok=True)
    for name in ('train', 'val', 'test'):
        reduced[name].to_csv(os.path.join(out, f'listening_history_{name}.csv'), index=False)
    cold_test.to_csv(os.path.join(out, 'cold_test.csv'), index=False)
    cold_out = cold.merge(item_ids, on='item_id')[['item_id', 'item', 'train_pop', 'pop_decile']]
    cold_out.to_csv(os.path.join(out, 'cold_items.csv'), index=False)
    for f in ('user_ids.csv', 'item_ids.csv', 'item_features.csv'):
        shutil.copy2(os.path.join(canonical, f), os.path.join(out, f))

    report = {
        'canonical': os.path.abspath(canonical), 'out': os.path.abspath(out),
        'seed': seed, 'fraction': fraction,
        'n_items_total': len(item_ids), 'n_items_eligible': n_eligible,
        'n_cold_items': len(cold),
        'canonical_rows': {'train': len(train), 'val': len(val), 'test': len(test)},
        'variant_rows': {k: len(v) for k, v in reduced.items()},
        'removed_rows': {'train': len(train) - len(reduced['train']),
                         'val': len(val) - len(reduced['val']),
                         'test': len(test) - len(reduced['test'])},
        'removed_train_pct': round(100 * (len(train) - len(reduced['train'])) / len(train), 2),
        'cold_test_rows': len(cold_test),
        'dropped_zero_train_users': int(n_zero_users),
        'dropped_zero_train_rows': n_zero_rows,
        'n_users': n_users,
        'users_below_3_train_rows': n_below3,
        'users_with_no_train_rows_at_all': int(n_users - len(train_counts)),
        'min_cold_pool_over_users': min_cold_pool,
        'cold_users_with_pool_below_100': n_pool_below_100,
    }
    with open(os.path.join(out, 'cold_variant_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print('=== cold-variant collateral report (S0.3 §2) ===')
    for k, v in report.items():
        print(f'  {k}: {v}')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    here = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(here, '..'))
    ap.add_argument('--canonical', default=os.path.join(data_root, 'hm_1_month'))
    ap.add_argument('--out', default=os.path.join(data_root, 'hm_1_month_cold'))
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--fraction', type=float, default=DEFAULT_FRACTION)
    args = ap.parse_args()
    make_cold_variant(args.canonical, args.out, args.seed, args.fraction)


if __name__ == '__main__':
    main()
