"""
Read-only sizing probe for a potential hm_1_month variant.
Does NOT create any dataset / does NOT run the splitter. Only prints stats.

Mirrors hm_splitter.py logic: last-N-months window -> dedup (keep first)
-> iterative k-core (item pass, then user pass, until convergence).
Tests k in {3, 5, 10} to see what keeps the dataset 'functional' and
in the ml-1m-ish target zone.
"""
import pandas as pd

RAW = './data/hm/raw/transactions_train.csv'
MAX_DATE = pd.Timestamp('2020-09-22')          # documented end of data
CUTOFF_1MO = (MAX_DATE - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
print(f'1-month window cutoff: t_dat >= {CUTOFF_1MO}')

# Chunked read, keep only the last-month rows + only needed cols.
cols = ['t_dat', 'customer_id', 'article_id']
parts = []
for chunk in pd.read_csv(RAW, usecols=cols, dtype=str, chunksize=2_000_000):
    parts.append(chunk[chunk.t_dat >= CUTOFF_1MO])
lhs = pd.concat(parts, ignore_index=True)
print(f'\nRaw rows in 1-month window: {len(lhs):,}')
print(f'  users: {lhs.customer_id.nunique():,}  items: {lhs.article_id.nunique():,}')

# Dedup (keep first occurrence of customer-article pair)
lhs = lhs.drop_duplicates(subset=['customer_id', 'article_id'], keep='first')
print(f'After dedup: {len(lhs):,}  users: {lhs.customer_id.nunique():,}  items: {lhs.article_id.nunique():,}')


def kcore(df, core):
    df = df.copy()
    it = 0
    while True:
        n0 = len(df)
        ic = df.article_id.value_counts()
        df = df[df.article_id.isin(set(ic[ic >= core].index))]
        uc = df.customer_id.value_counts()
        df = df[df.customer_id.isin(set(uc[uc >= core].index))]
        it += 1
        if len(df) == n0:
            break
    return df, it


print('\n=== iterative k-core sweep (1-month window) ===')
print(f"{'k':>3} {'interactions':>14} {'users':>10} {'items':>9} {'dens%':>7} {'i/u':>6} {'i/it':>7} {'iters':>5}")
for k in (3, 5, 10):
    d, iters = kcore(lhs, k)
    u, i, n = d.customer_id.nunique(), d.article_id.nunique(), len(d)
    dens = 100 * n / (u * i) if u and i else 0
    print(f'{k:>3} {n:>14,} {u:>10,} {i:>9,} {dens:>7.3f} {n/u:>6.1f} {n/i:>7.1f} {iters:>5}')

print('\nReference: ml-1m = 574,376 int / 6,034 u / 3,125 it (5-core)')
print('Reference: hm_3_month = 2,898,804 int / 256,701 u / 26,963 it (5-core)')
