# dc01 P4 unit test — factory 'feature_item_proto' branch (shared-instance tie, init ownership).
# Run: python Master/temp/dc_checks/dc01/t04_factory_branch.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_extractors import (
    ConcatenateFeatureExtractors, FeatureEmbedding, PrototypeEmbedding, EmbeddingW)
from rec_sys.rec_sys import RecSys

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


torch.manual_seed(7)
N_USERS, N_ITEMS, D = 5, 6, 8
KU, KT = 3, 4  # user_n_prototypes, item_n_prototypes
N_FEATURES = 7
feature_ids = torch.tensor([[0, 3], [1, 4], [2, 5], [0, 6], [1, 3], [2, 4]], dtype=torch.long)


def make_param():
    proto_block = lambda k: dict(sim_proto_weight=1.0, sim_batch_weight=1.0, use_weight_matrix=False,
                                 n_prototypes=k, cosine_type='shifted', reg_proto_type='max', reg_batch_type='max')
    return {
        'ft_type': 'feature_item_proto',
        'embedding_dim': D,
        'user_ft_ext_param': {'ft_type': 'prototypes_double_tie', **proto_block(KU)},
        'item_ft_ext_param': {'ft_type': 'feature_item_proto', **proto_block(KT),
                              'use_id_feature': True, 'feature_fields': ['fieldA', 'fieldB'],
                              'feature_ids': feature_ids, 'n_features': N_FEATURES},
    }


user_fe, item_fe = FeatureExtractorFactory.create_models(make_param(), N_USERS, N_ITEMS)

check("t04.two_concatenate_extractors",
      isinstance(user_fe, ConcatenateFeatureExtractors) and isinstance(item_fe, ConcatenateFeatureExtractors),
      f"{type(user_fe).__name__}, {type(item_fe).__name__}")

# item-side tie = shared instance identity
item_proto, item_proj = item_fe.model_1, item_fe.model_2
check("t04.item_shared_instance_tie", item_proto.embedding_ext is item_proj.shared,
      "item_proto.embedding_ext is item_proj.shared")
check("t04.item_ext_is_FeatureEmbedding", isinstance(item_proto.embedding_ext, FeatureEmbedding),
      f"{type(item_proto.embedding_ext).__name__}")

# user side unchanged: weight-tied plain Embedding (EmbeddingW + PrototypeEmbedding)
user_proto, user_embed = user_fe.model_1, user_fe.model_2
check("t04.user_proto_is_PrototypeEmbedding", isinstance(user_proto, PrototypeEmbedding), "")
check("t04.user_embed_is_EmbeddingW", isinstance(user_embed, EmbeddingW), "")
check("t04.user_weight_tie",
      user_embed.embedding_layer.weight is user_proto.embedding_ext.embedding_layer.weight,
      "user_embed.weight is user_proto.embedding_ext.weight")

# Build RecSys; the shared table must already be finite (factory single-owner init) and
# must NOT be re-initialized by RecSys.init_parameters (single-owner check).
rec = RecSys(N_USERS, N_ITEMS, {'use_bias': 0}, user_fe, item_fe, 'bce')
shared_w_before = item_proto.embedding_ext.embedding_layer.weight.detach().clone()
rec.init_parameters()
shared_w_after = item_proto.embedding_ext.embedding_layer.weight.detach().clone()
check("t04.shared_finite_after_init", bool(torch.isfinite(shared_w_after).all()), "")
check("t04.recsys_init_does_not_redraw_shared",
      torch.equal(shared_w_before, shared_w_after),
      f"max delta {(shared_w_before - shared_w_after).abs().max().item():.2e}")

# output last-dims: both Ku+Kt
u_idxs = torch.tensor([0, 1, 2, 3])           # (B,)
i_idxs = torch.tensor([[0, 1, 2], [3, 4, 5],  # (B, 1+n_neg)
                       [1, 2, 3], [4, 5, 0]])
with torch.no_grad():
    u_out = user_fe(u_idxs)
    i_out = item_fe(i_idxs)
check("t04.user_out_lastdim", u_out.shape[-1] == KU + KT, f"{tuple(u_out.shape)} (expect last {KU + KT})")
check("t04.item_out_lastdim", i_out.shape[-1] == KU + KT, f"{tuple(i_out.shape)} (expect last {KU + KT})")
check("t04.user_item_lastdims_match", u_out.shape[-1] == i_out.shape[-1], "")

# full forward through RecSys
with torch.no_grad():
    logits = rec(u_idxs, i_idxs)
check("t04.recsys_forward_shape", tuple(logits.shape) == (4, 3), f"{tuple(logits.shape)}")
check("t04.recsys_forward_finite", bool(torch.isfinite(logits).all()), "")

print("\nt04: ALL PASS" if not FAILS else f"\nt04: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
