# dc09 t08 — one vocabulary index on both sides, on the REAL splits (plan P2 / stage D).
# Needs data: PROTOMF_DATA_PATH with hm_1_month, ml-1m and their _cold variants (run on LEO5).
# Per dataset, through the production seam (canonical resolution -> inject_feature_ids):
#   A. vocabulary: both sides resolve the same fields/layout; item n_features == user
#      n_features == len(code->(field, value) labels); the user builder's per-item token table
#      (the leave-one-out payload) IS the item composition's code table (ids + weights); every
#      code is a word (< V), never an ID or dc08 row;
#   B. history: an independent float64 rebuild of w = n_{u,f}/|H_u| from the item codes and the
#      variant's own train pairs equals the injected history (every (user, word) cell); |H_u| matches;
#      on _cold variants no train pair touches a cold item (leakage rule);
#   C. the four arms built by the factory on the real payloads: table layout and row counts,
#      the borrowed table, and a manual float64 recomposition of q_i / q_u from the table rows
#      at the arm's offsets (sampled objects) equal the bases' own outputs; scores finite.
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from confs.hyper_params import (feature_user_item_proto_hyper_params,  # noqa: E402
                                feature_user_item_proto_noid_hyper_params,
                                feature_user_item_proto_noid_sep_hyper_params,
                                feature_user_item_proto_sep_hyper_params)
from feature_extraction.feature_extractor_factories import FeatureExtractorFactory  # noqa: E402
from feature_extraction.feature_ids import (build_code_to_field_value, inject_feature_ids,  # noqa: E402
                                            resolve_feature_fields_in_conf)
from rec_sys.rec_sys import RecSys  # noqa: E402
from utilities.consts import DATA_PATH  # noqa: E402
from utilities.utils import reproducible  # noqa: E402

FAILS = []


def check(name, cond, evidence=''):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f' | {evidence}' if evidence else ''))
    if not cond:
        FAILS.append(name)


ARMS = {
    'ids_shared': feature_user_item_proto_hyper_params,
    'noid_shared': feature_user_item_proto_noid_hyper_params,
    'ids_sep': feature_user_item_proto_sep_hyper_params,
    'noid_sep': feature_user_item_proto_noid_sep_hyper_params,
}
DATASETS = sys.argv[1:] or ['hm_1_month', 'ml-1m', 'hm_1_month_cold', 'ml-1m_cold']
D_EMB, K_U, K_T, N_SAMPLE = 16, 7, 9, 512


def concrete(space, dataset):
    """The arm's ft_ext_param with the tune samplers replaced by fixed values, canonical resolved."""
    conf = {'ft_ext_param': copy.deepcopy(space['ft_ext_param'])}
    resolve_feature_fields_in_conf(conf, dataset)
    ft = conf['ft_ext_param']
    ft['embedding_dim'] = D_EMB
    for side, k in (('user_ft_ext_param', K_U), ('item_ft_ext_param', K_T)):
        ft[side].update(n_prototypes=k, sim_proto_weight=0.1, sim_batch_weight=0.1)
    return ft


def sparse_history(rows, cols, vals, n_users, V):
    """(user, word) -> weight, duplicates summed (coalesced COO, float64)."""
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals.double(), (n_users, V)).coalesce()


for dataset in DATASETS:
    data_path = os.path.join(DATA_PATH, dataset)
    if not os.path.isdir(data_path):
        check(f'{dataset}: data present', False, f'{data_path} missing')
        continue
    print(f'\n===== {dataset} =====')
    n_users = len(pd.read_csv(os.path.join(data_path, 'user_ids.csv')))
    n_items = len(pd.read_csv(os.path.join(data_path, 'item_ids.csv')))

    # ---- A. vocabulary, through the production seam (default arm: ids, shared) ----
    ft = inject_feature_ids(concrete(feature_user_item_proto_hyper_params, dataset), data_path)
    ip, up = ft['item_ft_ext_param'], ft['user_ft_ext_param']
    check(f'{dataset}.A.same_fields_layout',
          ip['feature_fields'] == up['feature_fields'] and ip['feature_layout'] == up['feature_layout'],
          f"{ip['feature_fields']} / {ip['feature_layout']}")
    V = ip['n_features']
    labels = build_code_to_field_value(data_path, ip['feature_fields'], ip['feature_layout'])
    check(f'{dataset}.A.one_index', V == up['n_features'] == len(labels),
          f"item {V}, user {up['n_features']}, labels {len(labels)}")
    item_codes = ip['feature_ids']
    item_w = ip.get('feature_weights', torch.ones(item_codes.shape, dtype=torch.float32))
    check(f'{dataset}.A.loo_token_table_is_item_code_table',
          torch.equal(item_codes, up['hist_item_token_ids'])
          and torch.equal((item_w > 0).float(), up['hist_item_token_w']),
          f'item codes {tuple(item_codes.shape)}, layout {ip["feature_layout"]}')
    check(f'{dataset}.A.codes_are_words',
          int(item_codes[item_w > 0].max()) < V and int(up['hist_value_ids'][up['hist_weights'] > 0].max()) < V
          and int(up.get('n_history_rows', 0)) == 0)
    check(f'{dataset}.A.object_counts',
          item_codes.shape[0] == n_items and up['hist_value_ids'].shape[0] == n_users,
          f'M={n_items}, N={n_users}')

    # ---- B. independent rebuild of the user history ----
    pairs = pd.read_csv(os.path.join(data_path, 'listening_history_train.csv'), usecols=['user_id', 'item_id'])
    pairs = pairs.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    u = torch.as_tensor(pairs['user_id'].to_numpy(), dtype=torch.long)
    i = torch.as_tensor(pairs['item_id'].to_numpy(), dtype=torch.long)
    sizes = torch.bincount(u, minlength=n_users).double()
    check(f'{dataset}.B.basket_sizes', torch.equal(sizes.float(), up['hist_sizes']),
          f'{int((sizes > 0).sum())} users with a basket, {int((sizes == 0).sum())} empty')
    toks, real = item_codes[i], item_w[i] > 0  # (P, D_item): one unit per real token of a purchase
    u_rep = u.unsqueeze(1).expand_as(toks)[real]
    expected = sparse_history(u_rep, toks[real], 1.0 / sizes[u_rep], n_users, V)
    hv, hw = up['hist_value_ids'], up['hist_weights']
    got = sparse_history(torch.arange(n_users).unsqueeze(1).expand_as(hv)[hw > 0], hv[hw > 0], hw[hw > 0],
                         n_users, V)
    same_support = torch.equal(expected.indices(), got.indices())
    err = (got.values() - expected.values()).abs().max().item() if same_support else float('inf')
    check(f'{dataset}.B.history_rebuild', same_support and err < 1e-6,
          f'{expected.indices().shape[1]} (user, word) cells, max abs {err:.1e} (float32 storage)')
    del expected, got
    if dataset.endswith('_cold'):
        cold = set(pd.read_csv(os.path.join(data_path, 'cold_items.csv'))['item_id'].tolist())
        touched = len(cold & set(pairs['item_id'].unique().tolist()))
        check(f'{dataset}.B.no_cold_item_in_train', touched == 0, f'{len(cold)} cold items, {touched} in train')

    # ---- C. the four arms on the real payloads ----
    g = torch.Generator().manual_seed(0)
    s_items = torch.randperm(n_items, generator=g)[:N_SAMPLE]
    s_users = torch.randperm(n_users, generator=g)[:N_SAMPLE]
    for arm, space in ARMS.items():
        ftp = inject_feature_ids(concrete(space, dataset), data_path)
        reproducible(0)
        ufe, ife = FeatureExtractorFactory.create_models(ftp, n_users, n_items)
        rec = RecSys(n_users, n_items, {'use_bias': 0}, ufe, ife, 'sampled_softmax')
        rec.init_parameters()
        rec.eval()
        ib, ub = ife.base, ufe.base
        ids = 'noid' not in arm
        shared = 'shared' in arm
        Ti, Tu = ib.embedding_layer.weight, ub.embedding_layer.weight
        if shared:
            layout_ok = (Ti is Tu) and Ti.shape[0] == V + (n_items + n_users if ids else 0)
        else:
            layout_ok = (Ti is not Tu) and Ti.shape[0] == V + (n_items if ids else 0) \
                and Tu.shape[0] == V + (n_users if ids else 0)
        check(f'{dataset}.C.{arm}.table_layout', layout_ok,
              f'rows item {Ti.shape[0]}, user {Tu.shape[0]} (V={V}, M={n_items}, N={n_users})')
        exp_uoff = V + (n_items if shared and ids else 0)
        check(f'{dataset}.C.{arm}.user_id_offset', (not ids) or ub.id_row_offset == exp_uoff,
              f'{getattr(ub, "id_row_offset", None)} vs {exp_uoff}')

        Wi, Wu = Ti.detach().double(), Tu.detach().double()
        c, w = item_codes[s_items], item_w[s_items].double()
        q_i = (Wi[c] * w.unsqueeze(-1)).sum(1) + (Wi[V + s_items] if ids else 0)
        hv, hw = ftp['user_ft_ext_param']['hist_value_ids'][s_users], \
            ftp['user_ft_ext_param']['hist_weights'][s_users].double()
        q_u = (Wu[hv] * hw.unsqueeze(-1)).sum(1) + (Wu[exp_uoff + s_users] if ids else 0)
        with torch.no_grad():
            bi, bu = ib(s_items).double(), ub(s_users).double()
            sc = rec(s_users[:64], s_items[:200].repeat(64, 1))
        ei = ((bi - q_i).abs().max() / q_i.abs().max()).item()
        eu = ((bu - q_u).abs().max() / q_u.abs().max()).item()
        check(f'{dataset}.C.{arm}.recomposition', ei < 1e-5 and eu < 1e-5, f'item {ei:.1e}, user {eu:.1e}')
        check(f'{dataset}.C.{arm}.scores_finite', bool(torch.isfinite(sc).all()), f'shape {tuple(sc.shape)}')
        ufe.get_and_reset_loss()
        ife.get_and_reset_loss()
        del rec, ufe, ife, ftp

print('\nt08: ALL PASS' if not FAILS else f'\nt08: {len(FAILS)} FAILED -> {FAILS}')
sys.exit(1 if FAILS else 0)
