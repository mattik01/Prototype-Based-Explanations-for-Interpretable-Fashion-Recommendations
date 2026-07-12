# dc05 t03 — the feature_user_proto factory branch (design doc §3.2).
# Pins: branch shapes (user = PrototypeEmbedding(ext=HistoryFeatureEmbedding), item table
# M×K_u); the three guards fire; the factory single-owner-inits the history word table;
# forward/backward runs through RecSys.
# Run: python Master/temp/dc_checks/dc05/t03_factory_branch.py
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import (make_check, feature_user_proto_param, build_recsys, toy_batch,
                      make_toy_history)

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_extractors import HistoryFeatureEmbedding, PrototypeEmbedding, Embedding

check, FAILS = make_check()
torch.manual_seed(0)

N_USERS, N_ITEMS, V, D, KU = 9, 12, 7, 6, 4
ids, w = make_toy_history(N_USERS, V, d_max=3, seed=1)
param = feature_user_proto_param(D, KU, ids, w, V)

u_fe, i_fe = FeatureExtractorFactory.create_models(copy.deepcopy(param), N_USERS, N_ITEMS)
check("t03.user_branch_types",
      isinstance(u_fe, PrototypeEmbedding) and isinstance(u_fe.embedding_ext, HistoryFeatureEmbedding),
      f"{type(u_fe).__name__}(ext={type(u_fe.embedding_ext).__name__})")
check("t03.item_branch_type_and_dim",
      isinstance(i_fe, Embedding) and tuple(i_fe.embedding_layer.weight.shape) == (N_ITEMS, KU),
      f"item table {tuple(i_fe.embedding_layer.weight.shape)} (host property: M×K_u)")
check("t03.word_table_rows",
      u_fe.embedding_ext.embedding_layer.weight.shape[0] == V + N_USERS)

# factory single-owner init reached: general_weight_init draws N(0,1); nn.Embedding's default
# is ALSO N(0,1), so equality of distributions can't be checked — instead verify the factory
# CALL happens by monkeypatching init_parameters.
called = {}
orig = HistoryFeatureEmbedding.init_parameters
HistoryFeatureEmbedding.init_parameters = lambda self: called.setdefault('init', True) or orig(self)
FeatureExtractorFactory.create_models(copy.deepcopy(param), N_USERS, N_ITEMS)
HistoryFeatureEmbedding.init_parameters = orig
check("t03.factory_single_owner_init_called", called.get('init', False))

# guards
bad = copy.deepcopy(param); bad['item_ft_ext_param']['ft_type'] = 'prototypes'
try:
    FeatureExtractorFactory.create_models(bad, N_USERS, N_ITEMS); g1 = False
except AssertionError:
    g1 = True
bad = copy.deepcopy(param); bad['user_ft_ext_param']['use_weight_matrix'] = True
try:
    FeatureExtractorFactory.create_models(bad, N_USERS, N_ITEMS); g2 = False
except AssertionError:
    g2 = True
bad = copy.deepcopy(param); del bad['user_ft_ext_param']['hist_value_ids']
try:
    FeatureExtractorFactory.create_models(bad, N_USERS, N_ITEMS); g3 = False
except AssertionError:
    g3 = True
check("t03.guards_fire", g1 and g2 and g3, f"item_ft={g1}, weight_matrix={g2}, payload={g3}")

# end-to-end forward/backward through RecSys
rec = build_recsys(copy.deepcopy(param), N_USERS, N_ITEMS)
u, i, labels = toy_batch(N_USERS, N_ITEMS, seed=2)
logits = rec(u, i)
loss = rec.loss_func(logits, labels) + rec.user_feature_extractor.get_and_reset_loss()
loss.backward()
check("t03.forward_shape", tuple(logits.shape) == (4, 3), f"{tuple(logits.shape)}")
check("t03.backward_runs", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")

print("\nt03: ALL PASS" if not FAILS else f"\nt03: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
