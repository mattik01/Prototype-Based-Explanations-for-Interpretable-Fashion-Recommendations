import argparse
import os

import pandas as pd
from tqdm import tqdm

INF_STR = "{:10d} entries {:7d} users {:7d} items for {}"

parser = argparse.ArgumentParser(description='H&M dataset splitter for ProtoMF')

parser.add_argument('--listening_history_path', '-lh', type=str,
                    help="Path to directory containing transactions_train.csv and articles.csv",
                    default='./raw/')
parser.add_argument('--saving_path', '-s', type=str,
                    help="Path where to save the split data. Default to './'",
                    default='./')
parser.add_argument('--months', '-mo', type=int, default=6,
                    help="Temporal window: keep last N months of data (default: 6)")
parser.add_argument('--core', '-c', type=int, default=10,
                    help="K-core filtering threshold (default: 10)")

args = parser.parse_args()

listening_history_path = args.listening_history_path
saving_path = args.saving_path
k = 1  # leave-k-out

# ── Step 1: Load transactions ────────────────────────────────────────────────

print('Loading transactions...')
txn_path = os.path.join(listening_history_path, 'transactions_train.csv')
lhs = pd.read_csv(txn_path,
                   usecols=['t_dat', 'customer_id', 'article_id'],
                   dtype={'customer_id': str, 'article_id': str},
                   parse_dates=['t_dat'])

print(INF_STR.format(len(lhs), lhs.customer_id.nunique(), lhs.article_id.nunique(), 'Original Data'))

# ── Step 2: Temporal filter ──────────────────────────────────────────────────

max_date = lhs['t_dat'].max()
cutoff = max_date - pd.DateOffset(months=args.months)
lhs = lhs[lhs['t_dat'] >= cutoff]

print(INF_STR.format(len(lhs), lhs.customer_id.nunique(), lhs.article_id.nunique(),
                     f'Last {args.months} months (>= {cutoff.date()})'))

# ── Step 3: Deduplicate ─────────────────────────────────────────────────────

n_before = len(lhs)
lhs = lhs.sort_values('t_dat')
lhs = lhs.drop_duplicates(subset=['customer_id', 'article_id'], keep='first')
n_removed = n_before - len(lhs)

print(f'Removed {n_removed:,} duplicate (customer, article) pairs')
print(INF_STR.format(len(lhs), lhs.customer_id.nunique(), lhs.article_id.nunique(), 'After dedup'))

# ── Step 4: K-core filtering ────────────────────────────────────────────────

core = args.core
print(f'\n{core}-core filtering...')
iteration = 0
while True:
    start_number = len(lhs)

    # Item pass
    item_counts = lhs.article_id.value_counts()
    item_above = set(item_counts[item_counts >= core].index)
    lhs = lhs[lhs.article_id.isin(item_above)]
    print(f'  Pass {iteration + 1} - after items: {len(lhs):,}')

    # User pass
    user_counts = lhs.customer_id.value_counts()
    user_above = set(user_counts[user_counts >= core].index)
    lhs = lhs[lhs.customer_id.isin(user_above)]
    print(f'  Pass {iteration + 1} - after users: {len(lhs):,}')

    iteration += 1
    if len(lhs) == start_number:
        print(f'  Converged after {iteration} iterations')
        break

print(INF_STR.format(len(lhs), lhs.customer_id.nunique(), lhs.article_id.nunique(),
                     f'{core}-core filtering'))

# ── Step 5: ID remap ────────────────────────────────────────────────────────

print('\nRemapping IDs to contiguous 0-based integers...')

user_ids = lhs.customer_id.drop_duplicates().reset_index(drop=True)
item_ids = lhs.article_id.drop_duplicates().reset_index(drop=True)

user_ids.index.name = 'user_id'
item_ids.index.name = 'item_id'

user_ids.name = 'user'
item_ids.name = 'item'

user_ids = user_ids.reset_index()
item_ids = item_ids.reset_index()

# Merge to add user_id and item_id columns
lhs = lhs.merge(user_ids, left_on='customer_id', right_on='user')
lhs = lhs.merge(item_ids, left_on='article_id', right_on='item')

print(f'  Users: {len(user_ids):,}  Items: {len(item_ids):,}')

# ── Step 6: Leave-1-out split ───────────────────────────────────────────────

print('\nSplitting the data, temporal ordered - leave-k-out.')

lhs = lhs.sort_values('t_dat')
train_idxs = []
val_idxs = []
test_idxs = []

for user, user_group in tqdm(lhs.groupby('customer_id'), desc='Splitting users', unit='user'):
    # Data is already sorted by timestamp
    if len(user_group) <= k * 2:
        # Not enough data for val/test. Place entirely in train.
        train_idxs += list(user_group.index)
    else:
        train_idxs += list(user_group.index[:-2 * k])
        val_idxs += list(user_group.index[-2 * k:-k])
        test_idxs += list(user_group.index[-k:])

train_data = lhs.loc[train_idxs]
val_data = lhs.loc[val_idxs]
test_data = lhs.loc[test_idxs]

print(INF_STR.format(len(train_data), train_data.user_id.nunique(), train_data.item_id.nunique(), 'Train Data'))
print(INF_STR.format(len(val_data), val_data.user_id.nunique(), val_data.item_id.nunique(), 'Val Data'))
print(INF_STR.format(len(test_data), test_data.user_id.nunique(), test_data.item_id.nunique(), 'Test Data'))

# ── Step 7: Save 5 standard CSVs ────────────────────────────────────────────

print(f'\nSaving data to {saving_path}')
os.makedirs(saving_path, exist_ok=True)

train_data.to_csv(os.path.join(saving_path, 'listening_history_train.csv'), index=False)
val_data.to_csv(os.path.join(saving_path, 'listening_history_val.csv'), index=False)
test_data.to_csv(os.path.join(saving_path, 'listening_history_test.csv'), index=False)

user_ids.to_csv(os.path.join(saving_path, 'user_ids.csv'), index=False)
item_ids.to_csv(os.path.join(saving_path, 'item_ids.csv'), index=False)

# ── Step 8: Save item_features.csv ──────────────────────────────────────────

print('Building item_features.csv...')
articles_path = os.path.join(listening_history_path, 'articles.csv')
articles = pd.read_csv(articles_path, dtype={'article_id': str})

feature_cols = [
    'product_group_name',
    'product_type_name',
    'graphical_appearance_name',
    'colour_group_name',
    'perceived_colour_master_name',
    'index_group_name',
    'garment_group_name',
    'section_name',
    'department_name',
]

# Join with item_ids to keep only items in our filtered dataset and add item_id
item_features = item_ids.merge(articles[['article_id'] + feature_cols],
                               left_on='item', right_on='article_id')
item_features = item_features.drop(columns=['article_id'])
item_features = item_features.sort_values('item_id')

item_features.to_csv(os.path.join(saving_path, 'item_features.csv'), index=False)
print(f'  Saved item_features.csv with {len(item_features)} items, {len(feature_cols)} feature columns')

# ── Summary ──────────────────────────────────────────────────────────────────

n_users = len(user_ids)
n_items = len(item_ids)
n_total = len(train_data) + len(val_data) + len(test_data)
density = n_total / (n_users * n_items) * 100

print(f'\n{"=" * 50}')
print(f'SUMMARY')
print(f'{"=" * 50}')
print(f'  Users:              {n_users:>10,}')
print(f'  Items:              {n_items:>10,}')
print(f'  Train interactions: {len(train_data):>10,}')
print(f'  Val interactions:   {len(val_data):>10,}')
print(f'  Test interactions:  {len(test_data):>10,}')
print(f'  Total interactions: {n_total:>10,}')
print(f'  Density:            {density:>10.4f}%')
print(f'  Avg per user:       {n_total / n_users:>10.1f}')
print(f'  Avg per item:       {n_total / n_items:>10.1f}')
print(f'{"=" * 50}')
