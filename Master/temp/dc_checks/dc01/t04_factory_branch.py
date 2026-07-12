# dc01 P4 unit test — factory 'feature_item_proto' branch (fI host: I-ProtoMF shape,
# feature-composed item embedding, factory single-owner init).
# Re-pointed at the SC.3 re-run (2026-07-12): UI double-tie → I host. The UI-shaped test lives
# in git history with the UI factory branch.
# Run: python Master/temp/dc_checks/dc01/t04_factory_branch.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_extractors import Embedding, FeatureEmbedding, PrototypeEmbedding
from rec_sys.rec_sys import RecSys

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


torch.manual_seed(7)
N_USERS, N_ITEMS, D = 5, 6, 8
KT = 4  # item_n_prototypes (== user vector dimension, host property)
N_FEATURES = 7
feature_ids = torch.tensor([[0, 3], [1, 4], [2, 5], [0, 6], [1, 3], [2, 4]], dtype=torch.long)


def make_param():
    proto_block = dict(sim_proto_weight=1.0, sim_batch_weight=1.0, use_weight_matrix=False,
                       n_prototypes=KT, cosine_type='shifted', reg_proto_type='max', reg_batch_type='max')
    return {
        'ft_type': 'feature_item_proto',
        'embedding_dim': D,
        'user_ft_ext_param': {'ft_type': 'embedding'},
        'item_ft_ext_param': {'ft_type': 'feature_item_proto', **proto_block,
                              'use_id_feature': True, 'feature_fields': ['fieldA', 'fieldB'],
                              'feature_ids': feature_ids, 'n_features': N_FEATURES},
    }


user_fe, item_fe = FeatureExtractorFactory.create_models(make_param(), N_USERS, N_ITEMS)

# fI host shape: user = plain Embedding in R^{Kt}; item = PrototypeEmbedding over FeatureEmbedding
check("t04.user_is_plain_Embedding", isinstance(user_fe, Embedding), f"{type(user_fe).__name__}")
check("t04.user_dim_is_Kt", tuple(user_fe.embedding_layer.weight.shape) == (N_USERS, KT),
      f"{tuple(user_fe.embedding_layer.weight.shape)} (expect ({N_USERS}, {KT}))")
check("t04.item_is_PrototypeEmbedding", isinstance(item_fe, PrototypeEmbedding),
      f"{type(item_fe).__name__}")
check("t04.item_ext_is_FeatureEmbedding", isinstance(item_fe.embedding_ext, FeatureEmbedding),
      f"{type(item_fe.embedding_ext).__name__}")
check("t04.item_no_weight_matrix", item_fe.use_weight_matrix is False, "")
check("t04.feature_table_rows", tuple(item_fe.embedding_ext.embedding_layer.weight.shape)
      == (N_FEATURES + N_ITEMS, D), f"{tuple(item_fe.embedding_ext.embedding_layer.weight.shape)}")

# non-'embedding' user branch must be refused (fI host contract)
try:
    bad = make_param()
    bad['user_ft_ext_param'] = {'ft_type': 'prototypes_double_tie'}
    FeatureExtractorFactory.create_models(bad, N_USERS, N_ITEMS)
    check("t04.non_embedding_user_branch_refused", False, "no assert raised")
except AssertionError:
    check("t04.non_embedding_user_branch_refused", True, "AssertionError raised")

# Build RecSys; the feature table must already be finite (factory single-owner init) and
# must NOT be re-initialized by RecSys.init_parameters (PrototypeEmbedding never inits its ext).
rec = RecSys(N_USERS, N_ITEMS, {'use_bias': 0}, user_fe, item_fe, 'bce')
shared_w_before = item_fe.embedding_ext.embedding_layer.weight.detach().clone()
user_w_before = user_fe.embedding_layer.weight.detach().clone()
rec.init_parameters()
shared_w_after = item_fe.embedding_ext.embedding_layer.weight.detach().clone()
check("t04.feat_table_finite_after_init", bool(torch.isfinite(shared_w_after).all()), "")
check("t04.recsys_init_does_not_redraw_feat_table",
      torch.equal(shared_w_before, shared_w_after),
      f"max delta {(shared_w_before - shared_w_after).abs().max().item():.2e}")
check("t04.recsys_init_redraws_user_table",
      not torch.equal(user_w_before, user_fe.embedding_layer.weight.detach()),
      "user Embedding.init_parameters ran (host behavior)")

# output last-dims: both Kt (user vector · item activation)
u_idxs = torch.tensor([0, 1, 2, 3])           # (B,)
i_idxs = torch.tensor([[0, 1, 2], [3, 4, 5],  # (B, 1+n_neg)
                       [1, 2, 3], [4, 5, 0]])
with torch.no_grad():
    u_out = user_fe(u_idxs)
    i_out = item_fe(i_idxs)
check("t04.user_out_lastdim", u_out.shape[-1] == KT, f"{tuple(u_out.shape)} (expect last {KT})")
check("t04.item_out_lastdim", i_out.shape[-1] == KT, f"{tuple(i_out.shape)} (expect last {KT})")

# item activation is the shifted cosine: t* in [0, 2]
check("t04.item_activation_range", bool((i_out >= 0).all() and (i_out <= 2).all()),
      f"min {i_out.min().item():.3f}, max {i_out.max().item():.3f}")

# full forward through RecSys
with torch.no_grad():
    logits = rec(u_idxs, i_idxs)
check("t04.recsys_forward_shape", tuple(logits.shape) == (4, 3), f"{tuple(logits.shape)}")
check("t04.recsys_forward_finite", bool(torch.isfinite(logits).all()), "")

print("\nt04: ALL PASS" if not FAILS else f"\nt04: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
