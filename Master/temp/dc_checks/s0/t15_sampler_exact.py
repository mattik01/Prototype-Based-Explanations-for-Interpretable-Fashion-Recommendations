# S0 pin — the fast negative sampler draws EXACTLY the negatives of the inherited one (2026-10-27).
# `ProtoRecDataset` now hoists pop^0.75 out of the per-sample work and runs numpy's own
# choice(replace=False, p) loop without its validation passes. Claim: every negative of every
# split is bit-identical, so all landed rows stay reproducible and every run trained after the
# change is trained exactly as before. Reference = the original sampler code, verbatim, in a
# subclass. Checked, per (dataset, split, strategy), over a FULL pass:
#   per-sample — every index in order under one seed, sha256 over all negatives;
#   loader     — the training path itself: DataLoader(num_workers=1, shuffle, the trainer's
#                prefetch), same torch + numpy seeds, sha256 over every batch.
# Also: the construction-time self-check passes, and leaves the global RNG state untouched.
# Run locally on a toy split (default), or on LEO5 with `--real` (hm_1_month, ml-1m, hm_1_month_cold).
import hashlib
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils import data  # noqa: E402

from rec_sys.protomf_dataset import ProtoRecDataset  # noqa: E402

fails = []


def check(name, cond, evidence=''):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f' | {evidence}' if evidence else ''), flush=True)
    if not cond:
        fails.append(name)


class OriginalSampler(ProtoRecDataset):
    """The samplers exactly as inherited (pre-2026-10-27), for reference."""

    def _neg_sample_uniform(self, row_idx):
        consumed_items = self.csr_matrix.indices[self.csr_matrix.indptr[row_idx]:self.csr_matrix.indptr[row_idx + 1]]
        p = np.ones(self.n_items)
        p[consumed_items] = 0.
        if self.neg_exclude_items is not None:
            p[self.neg_exclude_items] = 0.
        p = p / p.sum()
        return np.random.choice(np.arange(self.n_items), self.n_neg, replace=False, p=p)

    def _neg_sample_popular(self, row_idx):
        consumed_items = self.csr_matrix.indices[self.csr_matrix.indptr[row_idx]:self.csr_matrix.indptr[row_idx + 1]]
        p = self.pop_distribution.copy()
        p[consumed_items] = 0.
        if self.neg_exclude_items is not None:
            p[self.neg_exclude_items] = 0.
        p = np.power(p, .75)
        p = p / p.sum()
        return np.random.choice(np.arange(self.n_items), self.n_neg, replace=False, p=p)


def per_sample_digest(ds, seed=11):
    np.random.seed(seed)
    h = hashlib.sha256()
    for idx in range(len(ds)):
        h.update(ds[idx][1].tobytes())
    return h.hexdigest()


def loader_digest(ds, batch_size, seed=11):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # fork, as on LEO5 (Linux default): the worker inherits the parent's numpy RNG state — the
    # property the fleet's single-worker stream rests on (macOS would spawn a fresh process)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=ds.split_set == 'train', num_workers=1,
                         prefetch_factor=5, multiprocessing_context='fork')
    h = hashlib.sha256()
    for u, i, lab in dl:
        h.update(u.numpy().tobytes())
        h.update(i.numpy().tobytes())
        h.update(lab.numpy().tobytes())
    return h.hexdigest()


def toy_split():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    n_u, n_i = 60, 140
    pd.DataFrame({'user_id': range(n_u)}).to_csv(os.path.join(d, 'user_ids.csv'), index=False)
    pd.DataFrame({'item_id': range(n_i)}).to_csv(os.path.join(d, 'item_ids.csv'), index=False)
    pop = rng.zipf(1.6, n_i).astype(float)
    rows = []
    for u in range(n_u):
        items = rng.choice(n_i, size=int(rng.integers(8, 30)), replace=False, p=pop / pop.sum())
        rows += [(u, int(i)) for i in items]
    df = pd.DataFrame(rows, columns=['user_id', 'item_id'])
    last = df.groupby('user_id').tail(2)
    val = last.groupby('user_id').head(1)
    test = last.groupby('user_id').tail(1)
    train = df.drop(last.index)
    for name, part in (('train', train), ('val', val), ('test', test)):
        part.to_csv(os.path.join(d, f'listening_history_{name}.csv'), index=False)
    return d


if __name__ == '__main__':
    real = '--real' in sys.argv
    if real:
        root = os.environ.get('PROTOMF_DATA_PATH', '/scratch/c7031336/protomf_data')
        cases = [  # (dataset, split, strategy, n_neg, mode)
            ('hm_1_month', 'train', 'popular', 50, 'loader'),
            ('hm_1_month', 'train', 'uniform', 50, 'per-sample'),
            ('hm_1_month', 'val', 'uniform', 99, 'loader'),
            ('hm_1_month', 'test', 'uniform', 99, 'per-sample'),
            ('ml-1m', 'train', 'popular', 50, 'loader'),
            ('ml-1m', 'val', 'uniform', 99, 'loader'),
            ('hm_1_month_cold', 'train', 'popular', 50, 'loader'),
            ('hm_1_month_cold', 'val', 'uniform', 99, 'loader'),
        ]
        paths = {c[0]: os.path.join(root, c[0]) for c in cases}
    else:
        toy = toy_split()
        cases = [('toy', s, st, 10, m) for s, st in (('train', 'popular'), ('train', 'uniform'), ('val', 'uniform'),
                                                    ('test', 'uniform')) for m in ('per-sample', 'loader')]
        paths = {'toy': toy}

    for dataset, split, strategy, n_neg, mode in cases:
        np.random.seed(5)
        before = np.random.get_state()[1].copy()
        new = ProtoRecDataset(paths[dataset], split, n_neg, strategy)
        after = np.random.get_state()[1]
        check(f'{dataset}/{split}: self-check passes and leaves the RNG untouched',
              new._exact_sampler and np.array_equal(before, after))
        ref = OriginalSampler(paths[dataset], split, n_neg, strategy)
        t = time.perf_counter()
        if mode == 'loader':
            a, b = loader_digest(ref, 256), loader_digest(new, 256)
        else:
            a, b = per_sample_digest(ref), per_sample_digest(new)
        check(f'{dataset}/{split}/{strategy}/{mode}: identical over the full pass', a == b,
              f'{len(new)} rows, {time.perf_counter() - t:.0f}s, sha256 {b[:12]}')

    print('\nALL PASS' if not fails else f'\nFAILURES: {fails}')
    sys.exit(1 if fails else 0)
