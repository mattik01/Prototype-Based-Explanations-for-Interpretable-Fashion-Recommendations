# dc09 t05 — the cold step reaches fUfI and covers both roles (F-DC09-01 code half, F-DC09-08).
#   - both finders reach the composition through TwoRoleComposedExtractor.base (the F-S0-07
#     silent-skip hazard), and config_expects_*_id_drop know the new ft_type (ids arms only);
#   - mean-ID imputation (the default convention) on the item side sets every cold item's ID row
#     to the mean of the WARM item ID rows; on the shared table it touches neither the words nor
#     the user-ID block; both roles of a cold item then read the imputed composition;
#   - the 'drop' convention zeroes them instead; noid arms are a no-op (None);
#   - the user side mirrors it at id_row_offset (after the item block on the shared table):
#     cold users with a basket get the warm-user mean, empty-basket cold users get 0 (q_u = 0);
#   - attr-kNN's target finder declines the branch (natively cold-capable, no free ID table).
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from _fixture import N_ITEMS, N_USERS, build, make_check, param_fUfI, toy_world
from utilities.cold_eval import (_find_feature_embedding, _find_history_feature_embedding,
                                 _find_item_embedding_weight, apply_cold_id_convention,
                                 config_expects_id_drop, config_expects_user_id_drop)

check, FAILS = make_check()
w = toy_world(empty_users=(6,))
V = w['V']
COLD_ITEMS, COLD_USERS = [2, 9], [1, 6]  # user 6 has an empty basket

tmp = tempfile.mkdtemp()
pd.DataFrame({'item_id': COLD_ITEMS}).to_csv(os.path.join(tmp, 'cold_items.csv'), index=False)
pd.DataFrame({'user_id': COLD_USERS}).to_csv(os.path.join(tmp, 'cold_users.csv'), index=False)

for share in (True, False):
    for ids in (True, False):
        tag = f'{"shared" if share else "sep"}.{"ids" if ids else "noid"}'
        conf = {'ft_ext_param': {'ft_type': 'feature_user_item_proto',
                                 'item_ft_ext_param': {'use_id_feature': ids},
                                 'user_ft_ext_param': {'use_id_feature': ids}}}
        check(f't05.{tag}.config_expects_item_step', config_expects_id_drop(conf) is ids)
        check(f't05.{tag}.config_expects_user_step', config_expects_user_id_drop(conf) is ids)

        rec = build(param_fUfI(w, share=share, item_id=ids, user_id=ids))
        ife, ufe = rec.item_feature_extractor, rec.user_feature_extractor
        fe, he = _find_feature_embedding(ife), _find_history_feature_embedding(ufe)
        check(f't05.{tag}.finders_reach_composition', fe is ife.base and he is ufe.base)
        try:  # attr_knn_patch's target finder: must decline (the runner then records 'skipped')
            _find_item_embedding_weight(ife)
            check(f't05.{tag}.attr_knn_declines', False, 'found a free ID table on a composed branch')
        except ValueError:
            check(f't05.{tag}.attr_knn_declines', True)

        if not ids:
            check(f't05.{tag}.noid_noop', apply_cold_id_convention(rec, tmp, 'impute_mean') == (None, None))
            continue

        W, UW = fe.embedding_layer.weight, he.embedding_layer.weight
        before, ubefore = W.detach().clone(), UW.detach().clone()
        uoff = he.id_row_offset
        check(f't05.{tag}.user_id_offset', uoff == V + (N_ITEMS if share else 0), f'{uoff}')
        item_info, user_info = apply_cold_id_convention(rec, tmp, 'impute_mean')
        warm_items = np.setdiff1d(np.arange(N_ITEMS), COLD_ITEMS)
        mean_i = before[torch.as_tensor(warm_items) + V].mean(0)
        check(f't05.{tag}.item_rows_are_warm_mean',
              all(torch.equal(W[V + c], mean_i) for c in COLD_ITEMS), str(item_info))
        # the user rows: cold user 1 (has a basket) -> warm-user mean; cold user 6 (empty) -> 0
        warm_users = np.setdiff1d(np.arange(N_USERS), COLD_USERS)
        mean_u = ubefore[torch.as_tensor(warm_users) + uoff].mean(0)
        check(f't05.{tag}.user_rows_mirror',
              torch.equal(UW[uoff + 1], mean_u) and bool((UW[uoff + 6] == 0).all())
              and user_info is not None and user_info.get('n_empty_basket_zeroed') == 1, str(user_info))
        item_rows = [V + c for c in COLD_ITEMS]
        user_rows = [uoff + c for c in COLD_USERS]
        changed_i = (W != before).any(dim=1).nonzero().flatten().tolist()
        changed_u = (UW != ubefore).any(dim=1).nonzero().flatten().tolist()
        want_i = sorted(item_rows + (user_rows if share else []))
        check(f't05.{tag}.only_cold_id_rows_change',
              changed_i == want_i and changed_u == (want_i if share else sorted(user_rows)),
              f'item table {changed_i}, user table {changed_u}')
        # both roles of a cold item read the imputed composition
        rec.eval()
        with torch.no_grad():
            q = ife.base(torch.tensor([COLD_ITEMS[0]]))
            want_q = W[w['feature_ids'][COLD_ITEMS[0]]].sum(0) + mean_i
            out = ife(torch.tensor([COLD_ITEMS[0]]))
            want = torch.cat([ife.linear_layer(q), ife.proto.sim_from_embed(q)], dim=-1)
        check(f't05.{tag}.cold_item_composition_imputed', torch.allclose(q[0], want_q, atol=1e-6))
        check(f't05.{tag}.both_roles_read_it', torch.equal(out, want))
        ife.get_and_reset_loss(); ufe.get_and_reset_loss()

        # 'drop' zeroes instead
        rec2 = build(param_fUfI(w, share=share))
        apply_cold_id_convention(rec2, tmp, 'drop')
        W2 = _find_feature_embedding(rec2.item_feature_extractor).embedding_layer.weight
        check(f't05.{tag}.drop_zeroes_item_rows', all(bool((W2[V + c] == 0).all()) for c in COLD_ITEMS))

print("\nt05: ALL PASS" if not FAILS else f"\nt05: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
