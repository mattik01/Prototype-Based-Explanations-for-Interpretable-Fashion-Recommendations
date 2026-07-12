# dc05 i06 — regression buildcheck (dc01 i06a pattern): after the dc05 additions, EVERY
# registered model family still assembles, runs forward/backward and takes one optimizer step
# on toy tensors. Guards the shared surfaces the build touched (factory dispatch, inject seam,
# extractors module).
# Run: python Master/temp/dc_checks/dc05/i06_regression_buildcheck.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import (make_check, _proto_block, feature_user_proto_param, user_proto_param,
                      build_recsys, toy_batch, make_toy_history)

check, FAILS = make_check()
torch.manual_seed(0)
N_USERS, N_ITEMS, V, D, K = 7, 9, 6, 5, 3

feature_ids = torch.randint(0, V, (N_ITEMS, 2))
attr_multi_hot = torch.zeros(N_ITEMS, V)
attr_multi_hot[torch.arange(N_ITEMS), feature_ids[:, 0]] = 1.0
attr_multi_hot[torch.arange(N_ITEMS), (feature_ids[:, 1] % 3) + 3] = 1.0
hist_ids, hist_w = make_toy_history(N_USERS, V, d_max=3, seed=6)

PARAMS = {
    'mf': {'ft_type': 'detached', 'embedding_dim': D,
           'user_ft_ext_param': {'ft_type': 'embedding'},
           'item_ft_ext_param': {'ft_type': 'embedding'}},
    'user_proto': user_proto_param(D, K),
    'item_proto': {'ft_type': 'prototypes', 'embedding_dim': D,
                   'user_ft_ext_param': {'ft_type': 'embedding'},
                   'item_ft_ext_param': {'ft_type': 'prototypes', **_proto_block(K)}},
    'user_item_proto': {'ft_type': 'prototypes_double_tie', 'embedding_dim': D,
                        'user_ft_ext_param': {'ft_type': 'prototypes_double_tie', **_proto_block(K)},
                        'item_ft_ext_param': {'ft_type': 'prototypes_double_tie', **_proto_block(K)}},
    'feature_item_proto': {'ft_type': 'feature_item_proto', 'embedding_dim': D,
                           'user_ft_ext_param': {'ft_type': 'embedding'},
                           'item_ft_ext_param': {'ft_type': 'feature_item_proto', **_proto_block(K),
                                                 'use_id_feature': True,
                                                 'feature_fields': ['f0', 'f1'],
                                                 'feature_ids': feature_ids, 'n_features': V}},
    'lightfm': {'ft_type': 'lightfm', 'embedding_dim': D,
                'user_ft_ext_param': {'ft_type': 'embedding'},
                'item_ft_ext_param': {'ft_type': 'lightfm', 'use_id_feature': True,
                                      'feature_fields': ['f0', 'f1'],
                                      'feature_ids': feature_ids, 'n_features': V}},
    'attr_item_proto': {'ft_type': 'attr_item_proto', 'embedding_dim': D,
                        'user_ft_ext_param': {'ft_type': 'attr_item_proto', **_proto_block(K)},
                        'item_ft_ext_param': {'ft_type': 'attr_item_proto', **_proto_block(K),
                                              'attr_fields': ['f0', 'f1'],
                                              'attr_multi_hot': attr_multi_hot,
                                              'n_attr_values': V,
                                              'field_offsets': [0, 3, V]}},
    'feature_user_proto': feature_user_proto_param(D, K, hist_ids, hist_w, V),
    'feature_user_proto_noid': feature_user_proto_param(D, K, hist_ids, hist_w, V,
                                                        use_id_feature=False),
}

for name, param in PARAMS.items():
    try:
        model = build_recsys(param, N_USERS, N_ITEMS)
        u, i, labels = toy_batch(N_USERS, N_ITEMS, seed=11)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = model.loss_func(model(u, i), labels) \
            + model.user_feature_extractor.get_and_reset_loss() \
            + model.item_feature_extractor.get_and_reset_loss()
        loss.backward()
        opt.step()
        check(f"i06.{name}", torch.isfinite(loss).item(), f"loss={float(loss):.4f}")
    except Exception as e:
        check(f"i06.{name}", False, f"{type(e).__name__}: {e}")

print("\ni06: ALL PASS" if not FAILS else f"\ni06: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
