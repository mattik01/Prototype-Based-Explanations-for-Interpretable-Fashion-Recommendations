# t10 — explanations integration (dc02 P9): accessor registry + surface, pipeline wiring
# (attr route reads `attr_fields`), score-matrix naming, and the dc01 intrinsic-naming
# REGRESSION after the namer refactor.
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

from _harness import make_check, toy_attr_data, attr_item_proto_param, build_recsys, REPO_ROOT

from utilities.explanations.accessor import get_accessor, _ACCESSORS
from utilities.explanations.accessor.attr_item_proto import AttrItemProtoAccessor
from utilities.explanations.naming import (NamingConfig, FeatureSpec, attr_naming_config,
                                           intrinsic_naming_config,
                                           name_prototypes_from_score_matrix,
                                           name_prototypes_intrinsic)

torch.manual_seed(43)
check, fails = make_check()

N_u, M, d, Lu, K = 9, 13, 8, 4, 5
X, offs = toy_attr_data(M=M, V_fs=(3, 4, 5), seed=0)
V = X.shape[1]
F = len(offs) - 1
FIELDS = [f'f{f}' for f in range(F)]

# --- registry + flags ---
check("accessor registered", _ACCESSORS.get('attr_item_proto') is AttrItemProtoAccessor)
rec = build_recsys(attr_item_proto_param(d, Lu, K, X, offs), N_u, M, use_bias=0)
acc = get_accessor('attr_item_proto', rec)
check("capability flags",
      acc.has_attr_space_prototypes and acc.has_intrinsic_item_grounding
      and acc.has_item_prototypes and acc.has_user_prototypes and acc.has_projections)

# --- accessor surface ---
A_eff = rec.item_feature_extractor.model_1.effective_prototypes().detach().numpy()
check("item_prototypes() == effective A", np.array_equal(acc.item_prototypes(), A_eff))
check("item_embeddings() == X", np.array_equal(acc.item_embeddings(), X.numpy()))
check("user_prototype_attr_profiles == W_t (Lu, V)",
      np.array_equal(acc.user_prototype_attr_profiles(),
                     rec.item_feature_extractor.model_2.linear_layer.weight.detach().numpy())
      and acc.user_prototype_attr_profiles().shape == (Lu, V))
check("field_offsets passthrough", acc.field_offsets == list(offs))

# inherited sim == the model's own t* for all items
with torch.no_grad():
    t_star_all = rec.item_feature_extractor.model_1(torch.arange(M)).numpy()
rec.item_feature_extractor.get_and_reset_loss()
sim = acc.item_to_item_proto_sim()
check("item_to_item_proto_sim == forward t* (all items)",
      float(np.abs(sim - t_star_all).max()) < 1e-5,
      f"max err {float(np.abs(sim - t_star_all).max()):.2e}")

# attr_share_matrix: (V, K); rows gathered at an item's codes reproduce t*-1
S = acc.attr_share_matrix()
check("attr_share_matrix shape (V, K)", S.shape == (V, K))
i = 4
v_idx = (X[i] > 0).nonzero(as_tuple=False).squeeze(1).numpy()
gathered = S[v_idx].sum(axis=0)
check("share matrix gathered at item codes == t*-1",
      float(np.abs(gathered - (t_star_all[i] - 1)).max()) < 1e-6)

# nonneg build: item_prototypes() is the EFFECTIVE (positive) matrix, raw differs
rec_n = build_recsys(attr_item_proto_param(d, Lu, K, X, offs, nonneg_prototypes=True), N_u, M)
acc_n = get_accessor('attr_item_proto', rec_n)
check("nonneg: item_prototypes() strictly positive", bool((acc_n.item_prototypes() > 0).all()))
check("nonneg: raw_item_prototypes has negatives", bool((acc_n.raw_item_prototypes() < 0).any()))

# --- pipeline wiring (source-level) ---
from utilities.explanations.pipeline import EXPLAINABLE_MODELS as PIPE_EXPLAINABLE
check("pipeline EXPLAINABLE_MODELS contains attr_item_proto", 'attr_item_proto' in PIPE_EXPLAINABLE)
pipe_src = open(os.path.join(REPO_ROOT, 'utilities', 'explanations', 'pipeline.py')).read()
check("pipeline attr route reads attr_fields (not feature_fields)",
      re.search(r"attr_item.*\n.*attr_fields", pipe_src) is not None
      and "name_prototypes_from_score_matrix(" in pipe_src)
check("pipeline checks attr route BEFORE dc01 intrinsic route",
      pipe_src.find('has_attr_space_prototypes') < pipe_src.find('has_intrinsic_item_grounding'))

# explainers: the four grounding-compatible ones support attr_item_proto; weight_viz abstains.
from utilities.explanations.explainers import REGISTERED_EXPLAINERS
supported = {e.name for e in REGISTERED_EXPLAINERS if e.supports('attr_item_proto')}
check("tsne/top_k_items/naming/feature_small_multiples support attr_item_proto",
      {'tsne', 'top_k_items', 'naming', 'feature_small_multiples'} <= supported, f"{supported}")
# weight_viz is intentionally EXCLUDED: its shared weight_visualization util renders per-item
# bars û_k·t*_k, folding the ranking-irrelevant user-constant Σû forbidden by the §3.4 amendment.
check("weight_viz abstains from attr_item_proto (§3.4 amendment)", 'weight_viz' not in supported)

# --- naming: score-matrix route on a constructed share matrix ---
# toy items_info whose per-field sorted-unique vocab reproduces the toy column order
rows = []
for it in range(M):
    row = {'item_id': it}
    for f in range(F):
        j = int((X[it, offs[f]:offs[f + 1]] > 0).nonzero(as_tuple=False)[0])
        row[f'f{f}'] = f'v{j}'
    rows.append(row)
items_info = pd.DataFrame(rows)

scores = np.full((V, K), 0.001)
scores[offs[0] + 1, 0] = 0.30      # prototype 0: f0=v1 dominates
scores[offs[1] + 2, 0] = 0.20      # then f1=v2
cfg = attr_naming_config(NamingConfig(features=[FeatureSpec(f) for f in FIELDS]),
                         FIELDS, min_score=0.02)
check("attr_naming_config: scoring label 'attr', share-unit floor",
      cfg.scoring == 'attr' and cfg.min_score == 0.02 and cfg.min_count == 0)
result = name_prototypes_from_score_matrix(scores, FIELDS, items_info, cfg, side='item')
check("score-matrix naming: prototype 0 named by its top shares",
      result.names[0].name == 'v1, v2', f"'{result.names[0].name}'")
check("score-matrix naming: below-floor prototype falls back",
      result.names[1].name == 'proto 1', f"'{result.names[1].name}'")
check("profiles carry V stats per prototype", len(result.profiles[0].stats) == V)

# vocab mismatch raises
try:
    name_prototypes_from_score_matrix(scores[:-1], FIELDS, items_info, cfg, side='item')
    check("vocab-size mismatch raises", False)
except ValueError:
    check("vocab-size mismatch raises", True)

# --- dc01 REGRESSION: name_prototypes_intrinsic behaves identically after the refactor ---
e = np.array([[1.0, 0.0], [0.0, 1.0]])          # e_a, e_b for field 'g' values 'a','b'
p = np.array([[2.0, 0.0], [0.0, 3.0]])          # p0 ~ e_a, p1 ~ e_b
info_g = pd.DataFrame({'item_id': [0, 1], 'g': ['a', 'b']})
icfg = intrinsic_naming_config(NamingConfig(features=[FeatureSpec('g')]), ['g'], min_score=0.30)
r = name_prototypes_intrinsic(e, p, ['g'], info_g, icfg, side='item')
check("dc01 regression: intrinsic naming maps cos to correct names",
      r.names[0].name == 'a' and r.names[1].name == 'b',
      f"'{r.names[0].name}', '{r.names[1].name}'")
try:
    name_prototypes_intrinsic(e[:1], p, ['g'], info_g, icfg)
    check("dc01 regression: intrinsic vocab mismatch still raises", False)
except ValueError:
    check("dc01 regression: intrinsic vocab mismatch still raises", True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
