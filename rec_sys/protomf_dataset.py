import os

import numpy as np
import pandas as pd
from scipy import sparse as sp
from torch.utils import data
try:
    from torch.utils.data.dataset import T_co
except ImportError:
    from typing import Any as T_co


class ProtoRecDataset(data.Dataset):
    """
    Dataset class to be used in ProtoRec. To use this class for any dataset, please refer to the splitter functions
    (e.g. movielens_splitter.py)

    This class implements some basic functionalities about negative sampling. The negative sampling for a specific user
    is influenced by the split_set:
        - split_set = train: The other training items are excluded from the sampling.
        - split_set = val: The other validation items and training items are excluded from the sampling.
        - split_set = test: The other test items and training items are excluded from the sampling.

    About the data management and access:
    To perform a fast iteration and sampling over the dataset, we use two sparse matrices (COO and CSR). The COO
    is used for iteration over the training data while the CSR for fast negative sampling. We always load the train
    CSR since it is used to exclude the training data from the negative sampling also for Validation and Testing.
    NB. Depending on the split_set, the matrices may have different data. Train COO and Train CSR have always the
    same data. However, Val CSR has Val + Train data (same applies for test). This is due to the negative sampling
    in the csr matrix, for which we also exclude items from training (see below).
    """

    def __init__(self, data_path: str, split_set: str, n_neg: int, neg_strategy: str = 'uniform'):
        """
        :param data_path: path to the directory with the listening_history_*, item_ids, and user_ids files.
        :param split_set: Value in [train, val, test].
        :param n_neg: Number of negative samples.
        :param neg_strategy: Strategy to select the negative samples.
        """
        assert split_set in ['train', 'val', 'test'], f'<{split_set}> is not a valid value for split set!'

        self.data_path = data_path
        self.split_set = split_set
        self.n_neg = n_neg
        self.neg_strategy = neg_strategy

        self.n_users = None
        self.n_items = None

        self.item_ids = None

        self.coo_matrix = None
        self.csr_matrix = None

        self.pop_distribution = None

        # S0.3 cold-variant support (leakage item 6): on a derived cold dataset (marked by
        # cold_items.csv in the data dir) the cold items must not be drawn as TRAINING negatives —
        # the model would learn "cold items are disliked" (mimics "not yet in catalog during the
        # training period"). Eval negatives keep the full catalog as a pool (a launched item
        # competes with everything), so this global mask applies to the train split only; eval
        # splits instead exclude the user's own removed cold purchases per-user (F-S0-09, below).
        self.neg_exclude_items = None
        cold_items_path = os.path.join(data_path, 'cold_items.csv')
        if split_set == 'train' and os.path.exists(cold_items_path):
            self.neg_exclude_items = pd.read_csv(cold_items_path)['item_id'].to_numpy()
            print(f'Cold variant detected ({cold_items_path}): '
                  f'{len(self.neg_exclude_items)} cold items EXCLUDED from training negatives')

        self.load_data()

        self._exact_sampler = self._check_exact_sampler()
        if not self._exact_sampler:
            print(f'WARNING: the fast exact negative sampler does not reproduce np.random.choice '
                  f'under numpy {np.__version__} — falling back to np.random.choice (same negatives, '
                  f'original speed)')

        print(f'Built ProtoRecDataset module \n'
              f'- data_path: {self.data_path} \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- n_interactions: {self.coo_matrix.nnz} \n'
              f'- split_set: {self.split_set} \n'
              f'- n_neg: {self.n_neg} \n'
              f'- neg_strategy: {self.neg_strategy} \n')

    def load_data(self):
        print('Loading data')

        user_ids = pd.read_csv(os.path.join(self.data_path, 'user_ids.csv'))
        item_ids = pd.read_csv(os.path.join(self.data_path, 'item_ids.csv'))

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)

        train_lhs = pd.read_csv(os.path.join(self.data_path, 'listening_history_train.csv'))

        train_csr = sp.csr_matrix(
            (np.ones(len(train_lhs), dtype=np.int16), (train_lhs.user_id, train_lhs.item_id)),
            shape=(self.n_users, self.n_items))

        # Computing the popularity distribution (see _neg_sample_popular)
        item_popularity = np.array(train_csr.sum(axis=0)).flatten()
        self.pop_distribution = item_popularity / item_popularity.sum()
        # pop^0.75 once, instead of once per sample: zeroing excluded items AFTER the power gives the
        # same values (0^0.75 = 0), so every sampling distribution is bit-identical to the original.
        self.pop_distribution_075 = np.power(self.pop_distribution, .75)  # Squashing factor alpha = .75
        self._n_zero_pop = int(np.count_nonzero(self.pop_distribution_075 == 0.))

        if self.split_set == 'val':
            val_lhs = pd.read_csv(os.path.join(self.data_path, 'listening_history_val.csv'))

            val_csr = sp.csr_matrix(
                (np.ones(len(val_lhs), dtype=np.int16), (val_lhs.user_id, val_lhs.item_id)),
                shape=(self.n_users, self.n_items))

            val_coo = sp.coo_matrix(val_csr)

            self.coo_matrix = val_coo
            self.csr_matrix = val_csr + train_csr

        elif self.split_set == 'test':
            test_lhs = pd.read_csv(os.path.join(self.data_path, 'listening_history_test.csv'))

            test_csr = sp.csr_matrix(
                (np.ones(len(test_lhs), dtype=np.int16), (test_lhs.user_id, test_lhs.item_id)),
                shape=(self.n_users, self.n_items))

            test_coo = sp.coo_matrix(test_csr)

            self.coo_matrix = test_coo
            self.csr_matrix = test_csr + train_csr

        elif self.split_set == 'train':
            train_coo = sp.coo_matrix(train_csr)

            self.coo_matrix = train_coo
            self.csr_matrix = train_csr

        # F-S0-09: on a marked cold variant, a user's REMOVED cold purchases (cold_test.csv)
        # are canonically consumed items and must not be drawn as warm eval negatives — that
        # would put the user's actual purchases in the negative slots and directionally
        # penalize cold-generalizing models (incl. in early stopping). Folded into the
        # exclusion CSR only (never into the positives); canonical datasets are unaffected.
        if self.split_set in ('val', 'test'):
            cold_test_path = os.path.join(self.data_path, 'cold_test.csv')
            if os.path.exists(os.path.join(self.data_path, 'cold_items.csv')) \
                    and os.path.exists(cold_test_path):
                ct = pd.read_csv(cold_test_path, usecols=['user_id', 'item_id'])
                ct_csr = sp.csr_matrix(
                    (np.ones(len(ct), dtype=np.int16), (ct.user_id, ct.item_id)),
                    shape=(self.n_users, self.n_items))
                self.csr_matrix = self.csr_matrix + ct_csr
                print(f'Cold variant: {len(ct)} removed cold purchases folded into the '
                      f'{self.split_set} negative-exclusion set (F-S0-09)')

        # F-S0-03 invariant: canonical eval splits carry EXACTLY one row per user (leave-one-out).
        # A derived cold variant (marked by cold_items.csv) legitimately has fewer (cold positives
        # were removed); anything else with a mismatch is a broken split and must fail loudly.
        if self.split_set in ('val', 'test') and self.coo_matrix.nnz != self.n_users:
            if os.path.exists(os.path.join(self.data_path, 'cold_items.csv')):
                print(f'Cold variant: {self.split_set} has {self.coo_matrix.nnz} rows for '
                      f'{self.n_users} users (one-row-per-user invariant relaxed by design)')
            else:
                raise AssertionError(
                    f'{self.split_set} split has {self.coo_matrix.nnz} rows but {self.n_users} users — '
                    f'the one-eval-row-per-user invariant is broken (F-S0-03)')

    def _neg_sample_uniform(self, row_idx: int) -> np.array:
        """
        For a specific user, it samples n_neg items u.a.r.
        :param row_idx: user id (or row in the matrix)
        :return: npy array containing the negatively sampled items.
        """

        consumed_items = self.csr_matrix.indices[self.csr_matrix.indptr[row_idx]:self.csr_matrix.indptr[row_idx + 1]]

        # Uniform distribution without items consumed by the user
        p = np.ones(self.n_items)
        p[consumed_items] = 0.  # Excluding consumed items
        if self.neg_exclude_items is not None:
            p[self.neg_exclude_items] = 0.  # Excluding cold-variant items (train split only)
        p = p / p.sum()

        return self._choice(p, len(consumed_items), n_zero=0)

    def _neg_sample_popular(self, row_idx: int) -> np.array:
        """
        For a specific user, it samples n_neg items considering the frequency of appearance of items in the dataset, i.e.
        p(i being neg) ∝ (pop_i)^0.75.
        :param row_idx: user id (or row in the matrix)
        :return: npy array containing the negatively sampled items.
        """
        consumed_items = self.csr_matrix.indices[self.csr_matrix.indptr[row_idx]:self.csr_matrix.indptr[row_idx + 1]]

        p = self.pop_distribution_075.copy()  # (pop_i)^0.75, precomputed in load_data
        p[consumed_items] = 0.  # Excluding consumed items
        if self.neg_exclude_items is not None:
            p[self.neg_exclude_items] = 0.  # Excluding cold-variant items (train split only;
            # already popularity-0 on the variant — the explicit mask makes the exclusion structural)
        p = p / p.sum()

        return self._choice(p, len(consumed_items), n_zero=self._n_zero_pop)

    def _choice(self, p: np.ndarray, n_consumed: int, n_zero: int) -> np.ndarray:
        """``np.random.choice(n_items, n_neg, replace=False, p=p)`` — the same draws, faster.

        The per-sample cost of the inherited samplers sits in numpy's input validation (several
        full passes over the catalog per call). ``_sample_exact`` is numpy's own algorithm for
        this call without those passes: the same ``random_sample`` draws from the global RNG and
        the same cumsum / searchsorted arithmetic, hence bit-identical negatives (verified at
        construction by ``_check_exact_sampler``, pinned over full epochs by dc_checks/s0 t15;
        2.1–2.5× faster per sample on hm_1_month / ml-1m, dc09 SC.4 audit 2026-10-27). numpy's
        "fewer non-zero entries in p than size" error is kept: checked exactly whenever the cheap
        bound cannot rule it out. ``p`` is a fresh array per call and is modified in place.
        """
        if not self._exact_sampler:
            return np.random.choice(np.arange(self.n_items), self.n_neg, replace=False, p=p)
        n_excluded = 0 if self.neg_exclude_items is None else len(self.neg_exclude_items)
        if self.n_items - n_zero - n_consumed - n_excluded < self.n_neg \
                and np.count_nonzero(p > 0) < self.n_neg:
            raise ValueError('Fewer non-zero entries in p than size')
        return self._sample_exact(p)

    def _sample_exact(self, p: np.ndarray) -> np.ndarray:
        """numpy's legacy ``RandomState.choice`` loop for ``replace=False`` with ``p`` (mtrand)."""
        found = np.zeros(self.n_neg, dtype=np.int64)
        n_uniq = 0
        while n_uniq < self.n_neg:
            x = np.random.random_sample((self.n_neg - n_uniq,))
            if n_uniq > 0:
                p[found[0:n_uniq]] = 0
            cdf = np.cumsum(p)
            cdf /= cdf[-1]
            new = cdf.searchsorted(x, side='right')
            _, unique_indices = np.unique(new, return_index=True)
            unique_indices.sort()
            new = new.take(unique_indices)
            found[n_uniq:n_uniq + new.size] = new
            n_uniq += new.size
        return found

    def _check_exact_sampler(self, n_rows: int = 64) -> bool:
        """Self-check at construction: ``_sample_exact`` must reproduce ``np.random.choice`` on
        this dataset's own distributions (both strategies, the first ``n_rows`` users with
        history). The global RNG state is saved and restored, so the check consumes nothing from
        the stream training uses. On any mismatch (e.g. a numpy release that changed the
        algorithm) the dataset falls back to ``np.random.choice`` and says so."""
        state = np.random.get_state()
        try:
            users = np.unique(self.coo_matrix.row)[:n_rows]
            for u in users:
                consumed = self.csr_matrix.indices[self.csr_matrix.indptr[u]:self.csr_matrix.indptr[u + 1]]
                for base in (np.ones(self.n_items), self.pop_distribution_075):
                    p = base.copy()
                    p[consumed] = 0.
                    if self.neg_exclude_items is not None:
                        p[self.neg_exclude_items] = 0.
                    if np.count_nonzero(p > 0) < self.n_neg:
                        continue
                    p = p / p.sum()
                    np.random.seed(int(u))
                    ref = np.random.choice(np.arange(self.n_items), self.n_neg, replace=False, p=p.copy())
                    np.random.seed(int(u))
                    if not np.array_equal(ref, self._sample_exact(p.copy())):
                        return False
            return True
        finally:
            np.random.set_state(state)

    def __len__(self) -> int:
        return self.coo_matrix.nnz

    def __getitem__(self, index) -> T_co:
        """
        Loads the (user,item) pair associated to the index and performs the negative sampling.
        :param index: (user,item) index pair (as defined by the COO.data vector)
        :return: (user_idx,item_idxs,labels) where
            user_idx: is the index of the user
            item_idxs: is a npy array containing the items indexes. The positive item is in the 1st position followed
                        by the negative items indexes. Shape is (1 + n_neg,)
            labels: npy array containing the labels. First position is 1, the others are 0. Shape is (1 + n_neg,).

        """

        user_idx = self.coo_matrix.row[index].astype('int64')
        item_idx_pos = self.coo_matrix.col[index]

        # Select the correct negative sampling strategy
        if self.neg_strategy == 'uniform':
            neg_samples = self._neg_sample_uniform(user_idx)
        elif self.neg_strategy == 'popular':
            neg_samples = self._neg_sample_popular(user_idx)
        else:
            raise ValueError(f'Negative Sampling Strategy <{self.neg_strategy}> not implemented ... Yet')

        item_idxs = np.concatenate(([item_idx_pos], neg_samples)).astype('int64')

        labels = np.zeros(1 + self.n_neg, dtype='float32')
        labels[0] = 1.

        return user_idx, item_idxs, labels


class ColdTestDataset(data.Dataset):
    """
    S0.3 cold-test rows over a derived cold variant (see data/hm/make_cold_variant.py).

    Each row is 1 cold positive (from ``cold_test.csv``) + ``n_neg`` negatives drawn from the
    ranking pool:
        - ranking='cold' (PRIMARY, LightFM/B5-faithful): negatives sampled from the cold set C —
          every candidate slot is feature-only, so cold-warm score calibration cannot masquerade
          as cold skill (the "new-arrivals" scenario).
        - ranking='all' (secondary diagnostic): negatives from the FULL catalog (warm + cold) —
          deployment-shaped, deliberately calibration-confounded.

    Negative exclusion (leakage item 5): the user's consumption from the CANONICAL histories
    (train+val+test), so no user-consumed item can ever appear as a negative. Negatives are drawn
    from a dedicated, seeded ``numpy.random.default_rng`` — deterministic with num_workers=0
    (the cold-eval runner's setting), independent of the global RNG stream.

    Returns the standard ``(user_idx, item_idxs, labels)`` triple, so the usual model forward and
    Evaluator machinery apply unchanged. The divisor expectation is ``len(self)`` — NEVER n_users
    (F-S0-03: multiple cold rows per user are the norm here).
    """

    def __init__(self, variant_path: str, ranking: str = 'cold', n_neg: int = 99,
                 canonical_path: str = None, seed: int = 38210573):
        assert ranking in ('cold', 'all'), f'<{ranking}> is not a valid ranking pool!'
        self.variant_path = variant_path
        self.ranking = ranking
        self.n_neg = n_neg
        self.seed = seed

        if canonical_path is None:
            base = variant_path.rstrip('/')
            assert base.endswith('_cold'), \
                'canonical_path not given and variant dir does not end in "_cold" — pass it explicitly'
            canonical_path = base[:-len('_cold')]
        self.canonical_path = canonical_path

        self.n_users = len(pd.read_csv(os.path.join(variant_path, 'user_ids.csv')))
        self.n_items = len(pd.read_csv(os.path.join(variant_path, 'item_ids.csv')))

        cold_test = pd.read_csv(os.path.join(variant_path, 'cold_test.csv'))
        self.users = cold_test['user_id'].to_numpy()
        self.items = cold_test['item_id'].to_numpy()
        self.orig_split = cold_test['orig_split'].to_numpy()

        self.cold_items = np.sort(pd.read_csv(
            os.path.join(variant_path, 'cold_items.csv'))['item_id'].to_numpy())
        self.pool = self.cold_items if ranking == 'cold' else np.arange(self.n_items)

        # canonical consumption CSR (train+val+test) — the negative-exclusion source
        frames = [pd.read_csv(os.path.join(canonical_path, f'listening_history_{s}.csv'),
                              usecols=['user_id', 'item_id']) for s in ('train', 'val', 'test')]
        allc = pd.concat(frames, ignore_index=True)
        self.consumption_csr = sp.csr_matrix(
            (np.ones(len(allc), dtype=np.int16), (allc.user_id, allc.item_id)),
            shape=(self.n_users, self.n_items))

        self.rng = np.random.default_rng(seed)

        print(f'Built ColdTestDataset module \n'
              f'- variant_path: {self.variant_path} \n'
              f'- canonical_path: {self.canonical_path} \n'
              f'- ranking: {self.ranking} (pool size {len(self.pool)}) \n'
              f'- n_cold_rows: {len(self.users)} \n'
              f'- n_neg: {self.n_neg} \n')

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index):
        user_idx = int(self.users[index])
        item_idx_pos = self.items[index]

        csr = self.consumption_csr
        consumed = csr.indices[csr.indptr[user_idx]:csr.indptr[user_idx + 1]]
        allowed = self.pool[~np.isin(self.pool, consumed)]
        assert len(allowed) >= self.n_neg, \
            f'user {user_idx}: only {len(allowed)} candidates in the {self.ranking} pool for {self.n_neg} negatives'
        neg_samples = self.rng.choice(allowed, self.n_neg, replace=False)

        item_idxs = np.concatenate(([item_idx_pos], neg_samples)).astype('int64')
        labels = np.zeros(1 + self.n_neg, dtype='float32')
        labels[0] = 1.

        return np.int64(user_idx), item_idxs, labels


def get_protorecdataset_dataloader(data_path: str, split_set: str, n_neg: int, neg_strategy='uniform',
                                   **loader_params) -> data.DataLoader:
    """
    Returns the dataloader for a ProtoRecDataset
    :param data_path, ... ,neg_strategy: check ProtoRecDataset class for info about these parameters
    :param loader_params: parameters for the Dataloader
    :return:
    """
    protorec_dataset = ProtoRecDataset(data_path, split_set, n_neg, neg_strategy)

    return data.DataLoader(protorec_dataset, **loader_params)
