# dc05 t13 — the lightfm_hist decoupled control (F-DC05-21, dc05 SC.8 gate 2026-08-20).
# The fU user representation WITHOUT the prototype layer: user = bare HistoryFeatureEmbedding,
# item = plain CF Embedding in R^d, plain dot score (user-side mirror of the S0.4 lightfm rows).
# Pins: branch shapes; the two guards; forward == manual q_u·t_i (exact); ids/noid differ only
# in the ID term; gradient reach (word rows, ID rows, item table; none on buffers); the
# F=0+ID keystone (user branch == per-user embedding lookup, bit-identical under weight copy);
# inject_feature_ids routes the USER-side payload for ft_type 'lightfm_hist'; config
# registration (explicit use_id_feature, canonical sentinel, start.py choices).
# Run: python Master/temp/dc_checks/dc05/t13_lightfm_hist.py
import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check, toy_batch, make_toy_history, zero_word_payload

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_extractors import Embedding, HistoryFeatureEmbedding
from feature_extraction.feature_ids import inject_feature_ids
from rec_sys.rec_sys import RecSys

check, FAILS = make_check()
torch.manual_seed(0)

N_USERS, N_ITEMS, V, D = 9, 12, 7, 6
ids, w = make_toy_history(N_USERS, V, d_max=3, seed=1)


def lightfm_hist_param(d, hist_value_ids, hist_weights, n_features, use_id_feature=True):
    return {
        'ft_type': 'lightfm_hist', 'embedding_dim': d,
        'user_ft_ext_param': {'ft_type': 'lightfm_hist', 'use_id_feature': use_id_feature,
                              'feature_fields': [f'f{i}' for i in range(5)],
                              'hist_value_ids': hist_value_ids, 'hist_weights': hist_weights,
                              'n_features': n_features},
        'item_ft_ext_param': {'ft_type': 'embedding'},
    }


# --- 1. branch shapes -------------------------------------------------------------------
param = lightfm_hist_param(D, ids, w, V)
u_fe, i_fe = FeatureExtractorFactory.create_models(copy.deepcopy(param), N_USERS, N_ITEMS)
check("t13.user_branch_type", isinstance(u_fe, HistoryFeatureEmbedding),
      type(u_fe).__name__)
check("t13.item_branch_type_and_dim",
      isinstance(i_fe, Embedding) and tuple(i_fe.embedding_layer.weight.shape) == (N_ITEMS, D),
      f"item table {tuple(i_fe.embedding_layer.weight.shape)} (plain CF in R^d)")
check("t13.word_table_rows_ids_arm",
      u_fe.embedding_layer.weight.shape[0] == V + N_USERS)

# --- 2. guards --------------------------------------------------------------------------
bad = copy.deepcopy(param); bad['item_ft_ext_param']['ft_type'] = 'prototypes'
try:
    FeatureExtractorFactory.create_models(bad, N_USERS, N_ITEMS); g1 = False
except AssertionError:
    g1 = True
check("t13.guard_item_branch", g1)
bad = copy.deepcopy(param); del bad['user_ft_ext_param']['hist_value_ids']
try:
    FeatureExtractorFactory.create_models(bad, N_USERS, N_ITEMS); g2 = False
except AssertionError:
    g2 = True
check("t13.guard_missing_injection", g2)

# --- 3. forward identity: RecSys score == manual q_u · t_i ------------------------------
rec = RecSys(N_USERS, N_ITEMS, {'use_bias': 0}, u_fe, i_fe, 'bce')
rec.init_parameters()
u_idx, i_idx, labels = toy_batch(N_USERS, N_ITEMS, B=5, n_neg=3, seed=2)
with torch.no_grad():
    out = rec(u_idx, i_idx)
    E = u_fe.embedding_layer.weight
    q = (E[ids[u_idx]] * w[u_idx].unsqueeze(-1)).sum(dim=-2) + E[V + u_idx]
    t = i_fe.embedding_layer.weight[i_idx]
    manual = (q.unsqueeze(1) * t).sum(-1)
check("t13.forward_equals_manual_dot",
      torch.allclose(out, manual, atol=1e-6), f"maxΔ {(out - manual).abs().max():.2e}")

# --- 4. noid arm: table rows and forward without ID term --------------------------------
param_noid = lightfm_hist_param(D, ids, w, V, use_id_feature=False)
u_fe_n, i_fe_n = FeatureExtractorFactory.create_models(copy.deepcopy(param_noid), N_USERS, N_ITEMS)
check("t13.word_table_rows_noid_arm", u_fe_n.embedding_layer.weight.shape[0] == V)
with torch.no_grad():
    En = u_fe_n.embedding_layer.weight
    qn = (En[ids[u_idx]] * w[u_idx].unsqueeze(-1)).sum(dim=-2)
check("t13.noid_forward_no_id_term",
      torch.allclose(u_fe_n(u_idx), qn, atol=1e-6))

# --- 5. gradient reach ------------------------------------------------------------------
rec.zero_grad()
loss = rec.loss_func(rec(u_idx, i_idx), labels) + u_fe.get_and_reset_loss()
loss.backward()
gw = u_fe.embedding_layer.weight.grad
check("t13.grads_reach_word_rows", gw is not None and gw[:V].abs().sum() > 0)
check("t13.grads_reach_id_rows", gw[V:].abs().sum() > 0)
check("t13.grads_reach_item_table",
      i_fe.embedding_layer.weight.grad is not None
      and i_fe.embedding_layer.weight.grad.abs().sum() > 0)
check("t13.buffers_have_no_grad",
      u_fe.hist_value_ids.grad is None and u_fe.hist_weights.grad is None)

# --- 6. F=0+ID keystone: user branch == per-user embedding lookup (weight copy) ---------
z_ids, z_w, z_nf = zero_word_payload(N_USERS)
param_f0 = lightfm_hist_param(D, z_ids, z_w, z_nf, use_id_feature=True)
u_f0, _ = FeatureExtractorFactory.create_models(copy.deepcopy(param_f0), N_USERS, N_ITEMS)
plain = Embedding(N_USERS, D)
with torch.no_grad():
    plain.embedding_layer.weight.copy_(u_f0.embedding_layer.weight)  # rows n_features..: the ID table
    all_u = torch.arange(N_USERS)
    same = torch.equal(u_f0(all_u), plain(all_u))
check("t13.f0_id_keystone_bit_identical", same)

# --- 7. inject_feature_ids routes the USER side for lightfm_hist ------------------------
with tempfile.TemporaryDirectory() as td:
    with open(os.path.join(td, 'item_features.csv'), 'w') as f:
        f.write("item_id,fa,fb\n0,x,u\n1,y,u\n2,x,v\n")
    with open(os.path.join(td, 'user_ids.csv'), 'w') as f:
        f.write("user_id,user\n0,u0\n1,u1\n")
    with open(os.path.join(td, 'listening_history_train.csv'), 'w') as f:
        f.write("user_id,item_id\n0,0\n0,1\n1,2\n")
    spec = {'ft_type': 'lightfm_hist',
            'user_ft_ext_param': {'ft_type': 'lightfm_hist', 'use_id_feature': True,
                                  'feature_fields': ['fa', 'fb'], 'feature_layout': 'fixed'},
            'item_ft_ext_param': {'ft_type': 'embedding'}}
    out_spec = inject_feature_ids(spec, td)
    u_spec = out_spec['user_ft_ext_param']
    check("t13.inject_routes_user_payload",
          'hist_value_ids' in u_spec and 'hist_weights' in u_spec and u_spec['n_features'] == 4,
          f"n_features={u_spec.get('n_features')}")
    check("t13.inject_caller_untouched",
          'hist_value_ids' not in spec['user_ft_ext_param'])

# --- 8. config registration -------------------------------------------------------------
from confs.hyper_params import lightfm_hist_hyper_params, lightfm_hist_ids_hyper_params
for name, cfg, want_id in (('lightfm_hist', lightfm_hist_hyper_params, False),
                           ('lightfm_hist_ids', lightfm_hist_ids_hyper_params, True)):
    u_cfg = cfg['ft_ext_param']['user_ft_ext_param']
    check(f"t13.config_{name}_explicit_flags",
          cfg['ft_ext_param']['ft_type'] == 'lightfm_hist'
          and u_cfg['use_id_feature'] is want_id
          and u_cfg['feature_fields'] == 'canonical'
          and cfg['ft_ext_param']['item_ft_ext_param']['ft_type'] == 'embedding')
with open(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'start.py')) as f:
    start_src = f.read()
check("t13.start_py_choices", "'lightfm_hist'" in start_src and "'lightfm_hist_ids'" in start_src)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} "
      f"({'t13' if not FAILS else len(FAILS)})")
sys.exit(1 if FAILS else 0)
