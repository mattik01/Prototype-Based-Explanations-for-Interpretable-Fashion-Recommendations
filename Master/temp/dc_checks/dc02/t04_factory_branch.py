# t04 — factory branch 'attr_item_proto' (dc02 P5): structure, tie, concat order, no per-item params.
import sys

import torch

from _harness import make_check, toy_attr_data, attr_item_proto_param

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_extractors import (AttributeLookup, AttributePrototypeEmbedding,
                                                   AttributeProjection, ConcatenateFeatureExtractors,
                                                   EmbeddingW, PrototypeEmbedding)

torch.manual_seed(3)
check, fails = make_check()

# Dims chosen pairwise-distinct so "no parameter dim == n_items" is meaningful:
N_u, M, d, Lu, K = 9, 13, 8, 4, 5
X, offs = toy_attr_data(M=M, V_fs=(3, 4, 5), seed=0)          # V = 12
V = X.shape[1]
assert M not in (N_u, d, Lu, K, V)

param = attr_item_proto_param(d, Lu, K, X, offs)
user_fe, item_fe = FeatureExtractorFactory.create_models(param, N_u, M)

# --- structure ---
check("user side is ConcatenateFeatureExtractors", isinstance(user_fe, ConcatenateFeatureExtractors))
check("item side is ConcatenateFeatureExtractors", isinstance(item_fe, ConcatenateFeatureExtractors))
check("user model_1 is host PrototypeEmbedding (not attribute subclass)",
      type(user_fe.model_1) is PrototypeEmbedding)
check("user model_2 is EmbeddingW", type(user_fe.model_2) is EmbeddingW)
check("user invert False", user_fe.invert is False)
check("item model_1 is AttributePrototypeEmbedding", type(item_fe.model_1) is AttributePrototypeEmbedding)
check("item model_2 is AttributeProjection", type(item_fe.model_2) is AttributeProjection)
check("item invert True", item_fe.invert is True)

# --- ties ---
check("user weight tie (same tensor object)",
      user_fe.model_2.embedding_layer.weight is user_fe.model_1.embedding_ext.embedding_layer.weight)
check("item lookup instance shared between branches",
      item_fe.model_1.embedding_ext is item_fe.model_2.lookup)
check("shared lookup is an AttributeLookup", isinstance(item_fe.model_1.embedding_ext, AttributeLookup))

# --- concat order: item output = [t̂ (Lu) ; t* (K)] (invert=True) ---
idxs = torch.tensor([[0, 3], [7, 12]])
i_out = item_fe(idxs)
check("item output last dim == Lu+K", i_out.shape == (2, 2, Lu + K), f"{tuple(i_out.shape)}")
t_hat = item_fe.model_2(idxs)
t_star = item_fe.model_1(idxs)
check("concat order [t_hat; t_star]",
      torch.equal(i_out[..., :Lu], t_hat) and torch.equal(i_out[..., Lu:], t_star))
u_out = user_fe(torch.tensor([0, 5]))
check("user output last dim == Lu+K", u_out.shape == (2, Lu + K), f"{tuple(u_out.shape)}")
item_fe.get_and_reset_loss()
user_fe.get_and_reset_loss()

# --- parameter census: NO per-item parameters anywhere ---
item_shapes = sorted(tuple(p.shape) for p in item_fe.parameters())
check("item branch params exactly {(K,V), (Lu,V)}",
      item_shapes == sorted([(K, V), (Lu, V)]), f"{item_shapes}")
all_shapes = [tuple(p.shape) for p in list(user_fe.parameters()) + list(item_fe.parameters())]
check("no parameter has any dim == n_items", all(M not in s for s in all_shapes), f"{all_shapes}")

# --- injected-keys guard ---
bad = attr_item_proto_param(d, Lu, K, X, offs)
del bad['item_ft_ext_param']['attr_multi_hot']
try:
    FeatureExtractorFactory.create_models(bad, N_u, M)
    check("missing injected keys raise", False)
except AssertionError:
    check("missing injected keys raise", True)

# --- knob passthrough from config to module ---
param_k = attr_item_proto_param(d, Lu, K, X, offs,
                                nonneg_prototypes=True, crisp_weight=0.01,
                                sep_weight=0.02, sep_margin=0.3,
                                push_weight=0.03, push_n_items=7, push_seed=123)
_, item_fe_k = FeatureExtractorFactory.create_models(param_k, N_u, M)
pk = item_fe_k.model_1
check("knobs reach the module",
      pk.nonneg_prototypes is True and pk.crisp_weight == 0.01 and pk.sep_weight == 0.02
      and pk.sep_margin == 0.3 and pk.push_weight == 0.03 and pk.push_n_items == 7
      and pk.push_seed == 123)
check("knob defaults off when absent",
      item_fe.model_1.nonneg_prototypes is False and item_fe.model_1.crisp_weight == 0.
      and item_fe.model_1.sep_weight == 0. and item_fe.model_1.push_weight == 0.)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
