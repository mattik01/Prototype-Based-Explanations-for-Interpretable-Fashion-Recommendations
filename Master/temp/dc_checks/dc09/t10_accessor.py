# dc09 t10 — the explanation accessor sees the fUfI surfaces (plan G5, SC.3 half).
#   - every arm name resolves to the fUfI accessor (suffix normalisation incl. _sep/_noid_sep);
#   - both prototype sets, the composed factors (= the compositions' own outputs), both
#     projections (= the linear roles) have the right shapes and values;
#   - the word view is per side: identical rows under the shared table, the user's own rows
#     under `_sep` (what the pipeline's user-side intrinsic naming now reads);
#   - registration: pipeline EXPLAINABLE_MODELS and the generic explainers' supports();
#     weight_viz stays excluded until SC.5.
import sys

import numpy as np
import torch

from _fixture import D, K_T, K_U, N_ITEMS, N_USERS, build, make_check, param_fUfI, toy_world
from utilities.explanations.accessor import base_model_type, get_accessor
from utilities.explanations.explainers.naming import NamingExplainer
from utilities.explanations.explainers.proto_cards import ProtoCardsExplainer
from utilities.explanations.explainers.top_k_items import TopKItemsExplainer
from utilities.explanations.explainers.tsne import TSNEExplainer
from utilities.explanations.explainers.weight_viz import WeightVizExplainer
from utilities.explanations.pipeline import EXPLAINABLE_MODELS

check, FAILS = make_check()
w = toy_world()
V = w['V']

for name in ('feature_user_item_proto', 'feature_user_item_proto_noid', 'feature_user_item_proto_sep',
             'feature_user_item_proto_noid_sep'):
    check(f't10.base_model_type[{name}]', base_model_type(name) == 'feature_user_item_proto')

for share in (True, False):
    tag = 'shared' if share else 'sep'
    rec = build(param_fUfI(w, share=share)).eval()
    acc = get_accessor('feature_user_item_proto' + ('' if share else '_sep'), rec)
    ufe, ife = rec.user_feature_extractor, rec.item_feature_extractor
    check(f't10.{tag}.flags', acc.has_user_prototypes and acc.has_item_prototypes and acc.has_projections
          and acc.has_intrinsic_item_grounding and acc.has_intrinsic_user_grounding)
    check(f't10.{tag}.prototype_shapes', acc.user_prototypes().shape == (K_U, D)
          and acc.item_prototypes().shape == (K_T, D))
    with torch.no_grad():
        qu, qi = ufe.base(torch.arange(N_USERS)).numpy(), ife.base(torch.arange(N_ITEMS)).numpy()
        uh = ufe.linear_layer(ufe.base(torch.arange(N_USERS))).numpy()
        th = ife.linear_layer(ife.base(torch.arange(N_ITEMS))).numpy()
    check(f't10.{tag}.composed_factors', np.array_equal(acc.user_embeddings(), qu)
          and np.array_equal(acc.item_embeddings(), qi))
    check(f't10.{tag}.projections', np.array_equal(acc.user_proj_to_item_proto_space(torch.arange(N_USERS)), uh)
          and np.array_equal(acc.items_in_user_proto_space(), th) and th.shape == (N_ITEMS, K_U))
    fi, fu = acc.feature_value_embeddings('item'), acc.user_feature_value_embeddings()
    check(f't10.{tag}.word_view_per_side', fi.shape == fu.shape == (V, D)
          and (np.array_equal(fi, fu) if share else not np.array_equal(fi, fu))
          and acc.shares_word_table == share)
    check(f't10.{tag}.word_rows_are_the_table',
          np.array_equal(fu, ufe.base.embedding_layer.weight[:V].detach().numpy()))
    sim_u = acc.user_to_user_proto_sim()
    check(f't10.{tag}.user_proto_sim_shape', sim_u.shape == (N_USERS, K_U) and np.all(sim_u >= -1e-6))
    ufe.get_and_reset_loss(); ife.get_and_reset_loss()

check('t10.pipeline_explainable', 'feature_user_item_proto' in EXPLAINABLE_MODELS)
for ex in (NamingExplainer, TSNEExplainer, TopKItemsExplainer, ProtoCardsExplainer):
    check(f't10.supports[{ex.__name__}]', ex().supports('feature_user_item_proto'))
check('t10.weight_viz_still_excluded', not WeightVizExplainer().supports('feature_user_item_proto'))

print("\nt10: ALL PASS" if not FAILS else f"\nt10: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
