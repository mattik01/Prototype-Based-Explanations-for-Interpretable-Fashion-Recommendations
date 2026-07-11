# S0-build test harness — shared check() helper + repo-root path setup.
# Imported by the t0x/i0x checks in this directory. Not a test itself.
# (Mirrors dc01/dc02 _harness.py conventions.)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def make_check():
    fails = []

    def check(name, cond, evidence=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
        if not cond:
            fails.append(name)

    return check, fails


def build_toy_canonical(path, n_users=60, n_items=300, seed=38210573):
    """Synthetic canonical split dir in the hm layout (user_ids/item_ids/item_features +
    listening_history_{train,val,test}, one val+test row per user, >=3 train rows) with the
    S0.5 CANONICAL FIELD NAMES so cold_eval's kNN / tie-diagnostic paths run unmodified.
    Returns (n_users, n_items)."""
    import numpy as np
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'user_ids.csv'), 'w') as f:
        f.write('user_id,user\n' + '\n'.join(f'{u},u{u:04d}' for u in range(n_users)) + '\n')
    with open(os.path.join(path, 'item_ids.csv'), 'w') as f:
        f.write('item_id,item\n' + '\n'.join(f'{i},a{i:06d}' for i in range(n_items)) + '\n')

    deps = ['Blouse', 'Trouser', 'Knitwear', 'Shoe', 'Jersey']
    types = ['Top', 'Bottom', 'Dress', 'Sneaker']
    secs = ['Womens', 'Mens', 'Divided']
    cols = ['Black', 'White', 'Red', 'Blue', 'Green', 'Beige']
    gaps = ['Solid', 'Stripe', 'All over pattern']
    with open(os.path.join(path, 'item_features.csv'), 'w') as f:
        f.write('item_id,item,department_name,product_type_name,section_name,'
                'colour_group_name,graphical_appearance_name\n')
        for i in range(n_items):
            f.write(f'{i},a{i:06d},{deps[i % 5]},{types[i % 4]},{secs[i % 3]},'
                    f'{cols[i % 6]},{gaps[i % 3]}\n')

    rng = np.random.default_rng(seed)
    # popularity skew so popularity deciles are non-degenerate
    weights = 1. / (np.arange(n_items) + 1.)
    weights = weights / weights.sum()
    rows = {'train': [], 'val': [], 'test': []}
    for u in range(n_users):
        k = int(rng.integers(6, 10))
        items = rng.choice(n_items, size=k, replace=False, p=weights)
        for it in items[:k - 2]:
            rows['train'].append((u, it))
        rows['val'].append((u, items[k - 2]))
        rows['test'].append((u, items[k - 1]))
    for name in ('train', 'val', 'test'):
        with open(os.path.join(path, f'listening_history_{name}.csv'), 'w') as f:
            f.write('t_dat,customer_id,article_id,user_id,user,item_id,item\n')
            for u, it in rows[name]:
                f.write(f'2020-09-01,u{u:04d},a{it:06d},{u},u{u:04d},{it},a{it:06d}\n')
    return n_users, n_items
