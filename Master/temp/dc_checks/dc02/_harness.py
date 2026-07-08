# dc02 test harness — shared toy-data + param builders + a check() helper.
# Imported by t01..t10 / i01..i06. Not a test itself. (Mirrors dc01's _harness.py conventions.)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from rec_sys.rec_sys import RecSys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def make_check():
    fails = []

    def check(name, cond, evidence=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
        if not cond:
            fails.append(name)

    return check, fails


def toy_attr_data(M=12, V_fs=(3, 4, 5), seed=0):
    """Toy multi-hot X (M, V) with exactly one value per field block.

    Item M-1 is forced to DUPLICATE item 0's row — the cold/tie twin used by the
    cold-item (claim 2) and exact-tie (claim 6) tests.

    :return: (X FloatTensor (M, V), field_offsets list[int] of length F+1)
    """
    g = torch.Generator().manual_seed(seed)
    field_offsets = [0]
    for v in V_fs:
        field_offsets.append(field_offsets[-1] + v)
    X = torch.zeros(M, field_offsets[-1])
    for f, v in enumerate(V_fs):
        choices = torch.randint(0, v, (M,), generator=g)
        X[torch.arange(M), field_offsets[f] + choices] = 1.0
    X[M - 1] = X[0]  # cold/tie twin
    return X, field_offsets


def _proto_block(k):
    return dict(sim_proto_weight=1.0, sim_batch_weight=1.0, use_weight_matrix=False,
                n_prototypes=k, cosine_type='shifted', reg_proto_type='max', reg_batch_type='max')


def attr_item_proto_param(d, Lu, K, X, field_offsets, **item_overrides):
    """Build a fresh attr_item_proto ft_ext_param in INJECTED form (tensors already present,
    as `inject_feature_ids` would produce). `item_overrides` sets/overrides item-side keys
    (e.g. the constraint knobs)."""
    item = {'ft_type': 'attr_item_proto', **_proto_block(K),
            'attr_fields': [f'f{i}' for i in range(len(field_offsets) - 1)],
            'attr_multi_hot': X, 'n_attr_values': int(X.shape[1]),
            'field_offsets': list(field_offsets)}
    item.update(item_overrides)
    return {
        'ft_type': 'attr_item_proto', 'embedding_dim': d,
        'user_ft_ext_param': {'ft_type': 'attr_item_proto', **_proto_block(Lu)},
        'item_ft_ext_param': item,
    }


def user_item_proto_param(d, Ku, Kt):
    """Build a fresh prototypes_double_tie (== user_item_proto) ft_ext_param."""
    return {
        'ft_type': 'prototypes_double_tie', 'embedding_dim': d,
        'user_ft_ext_param': {'ft_type': 'prototypes_double_tie', **_proto_block(Ku)},
        'item_ft_ext_param': {'ft_type': 'prototypes_double_tie', **_proto_block(Kt)},
    }


def mf_param(d):
    return {'ft_type': 'detached', 'embedding_dim': d,
            'user_ft_ext_param': {'ft_type': 'embedding'},
            'item_ft_ext_param': {'ft_type': 'embedding'}}


def build_recsys(param, n_users, n_items, use_bias=0, loss='bce'):
    u_fe, i_fe = FeatureExtractorFactory.create_models(param, n_users, n_items)
    rec = RecSys(n_users, n_items, {'use_bias': use_bias}, u_fe, i_fe, loss)
    rec.init_parameters()
    return rec


def toy_batch(n_users, n_items, B=4, n_neg=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randint(0, n_users, (B,), generator=g)
    pos = torch.randint(0, n_items, (B, 1), generator=g)
    neg = torch.randint(0, n_items, (B, n_neg), generator=g)
    i = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(B, 1 + n_neg)
    labels[:, 0] = 1.0
    return u, i, labels
