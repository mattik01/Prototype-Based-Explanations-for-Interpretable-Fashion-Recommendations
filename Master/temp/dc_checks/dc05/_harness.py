# dc05 test harness — shared param builders + a check() helper (dc01/_harness pattern).
# Imported by t01..t10 / i06. Not a test itself.
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


def feature_user_proto_param(d, Ku, hist_value_ids, hist_weights, n_features,
                             use_id_feature=True, max_norm=None, feature_fields=None):
    """Build a fresh feature_user_proto (fU host) ft_ext_param with the history payload already
    injected. Item side = plain embedding (U-ProtoMF free item vector in R^{Ku}, sized by the
    factory)."""
    user = {'ft_type': 'feature_user_proto', **_proto_block(Ku),
            'use_id_feature': use_id_feature,
            'feature_fields': feature_fields if feature_fields is not None
            else [f'f{i}' for i in range(5)],
            'hist_value_ids': hist_value_ids, 'hist_weights': hist_weights,
            'n_features': n_features}
    if max_norm is not None:
        user['max_norm'] = max_norm
    return {
        'ft_type': 'feature_user_proto', 'embedding_dim': d,
        'user_ft_ext_param': user,
        'item_ft_ext_param': {'ft_type': 'embedding'},
    }


def user_proto_param(d, Ku, max_norm=None):
    """Build a fresh 'prototypes' User-Proto (== user_proto) ft_ext_param — the fU host."""
    user = {'ft_type': 'prototypes', **_proto_block(Ku)}
    if max_norm is not None:
        user['max_norm'] = max_norm
    return {
        'ft_type': 'prototypes', 'embedding_dim': d,
        'user_ft_ext_param': user,
        'item_ft_ext_param': {'ft_type': 'embedding'},
    }


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


def make_toy_history(n_users, V, d_max=4, seed=0, empty_users=()):
    """Random valid padded history tensors: per user, a few distinct word ids with positive
    mean-normalized-style weights (padding: id 0 / weight 0.0). ``empty_users`` get all-zero
    rows (the 4b C3 degenerate path)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros((n_users, d_max), dtype=torch.long)
    w = torch.zeros((n_users, d_max), dtype=torch.float32)
    for u in range(n_users):
        if u in empty_users:
            continue
        r = int(torch.randint(1, d_max + 1, (1,), generator=g))
        perm = torch.randperm(V, generator=g)[:r]
        raw = torch.rand(r, generator=g) + 0.1
        ids[u, :r] = perm
        w[u, :r] = raw / raw.sum()
    return ids, w


def zero_word_payload(n_users):
    """The keystone / A1 payload shape: zero metadata vocabulary (n_features=0), one all-padding
    slot per user — q_u = e_ID(u) exactly."""
    return (torch.zeros((n_users, 1), dtype=torch.long),
            torch.zeros((n_users, 1), dtype=torch.float32), 0)
