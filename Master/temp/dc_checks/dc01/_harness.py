# dc01 integration-test harness — shared param builders + a check() helper.
# Imported by i01..i06a. Not a test itself.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from rec_sys.rec_sys import RecSys


def make_check():
    fails = []

    def check(name, cond, evidence=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
        if not cond:
            fails.append(name)

    return check, fails


def _proto_block(k):
    return dict(sim_proto_weight=1.0, sim_batch_weight=1.0, use_weight_matrix=False,
                n_prototypes=k, cosine_type='shifted', reg_proto_type='max', reg_batch_type='max')


def feature_item_proto_param(d, Kt, feature_ids, n_features, use_id_feature=True):
    """Build a fresh feature_item_proto (fI host) ft_ext_param with feature_ids already injected.
    User side = plain embedding (I-ProtoMF free user vector in R^{Kt}, sized by the factory)."""
    F = feature_ids.shape[1]
    return {
        'ft_type': 'feature_item_proto', 'embedding_dim': d,
        'user_ft_ext_param': {'ft_type': 'embedding'},
        'item_ft_ext_param': {'ft_type': 'feature_item_proto', **_proto_block(Kt),
                              'use_id_feature': use_id_feature,
                              'feature_fields': [f'f{i}' for i in range(F)],
                              'feature_ids': feature_ids, 'n_features': n_features},
    }


def item_proto_param(d, Kt):
    """Build a fresh 'prototypes' Item-Proto (== item_proto) ft_ext_param — the fI host."""
    return {
        'ft_type': 'prototypes', 'embedding_dim': d,
        'user_ft_ext_param': {'ft_type': 'embedding'},
        'item_ft_ext_param': {'ft_type': 'prototypes', **_proto_block(Kt)},
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
