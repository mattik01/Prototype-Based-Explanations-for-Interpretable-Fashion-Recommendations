# dc05 t07 — the cold-user ID-drop mirror (design doc §3.4/§3.7-6; F-S0-07/F-S0-13 mirrored to
# users; review build obligation). No cold-USER variant generator exists this phase (M6.2:
# scrutiny-stage work) — this test pins the SEAM with a hand-made cold_users.csv so the future
# testbed plugs in.
# Pins: drop zeroes exactly the cold users' ID rows (their composition becomes metadata-only);
# warm users bit-identical; noid arm clean no-op; missing cold_users.csv -> no-op; the
# config_expects_* truth tables (incl. the item-side drop being a clean no-op for fU configs).
# Run: python Master/temp/dc_checks/dc05/t07_cold_user_id_drop.py
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import (make_check, feature_user_proto_param, build_recsys, make_toy_history)

from utilities.cold_eval import (config_expects_id_drop, config_expects_user_id_drop,
                                 drop_cold_user_id_rows, drop_cold_id_rows)

check, FAILS = make_check()
torch.manual_seed(0)
N_USERS, N_ITEMS, V, D, KU = 8, 10, 6, 5, 3
COLD = [1, 3]

tmp = tempfile.mkdtemp(prefix='t07_dc05_')
variant = os.path.join(tmp, 'variant')
os.makedirs(variant)
with open(os.path.join(variant, 'cold_users.csv'), 'w') as f:
    f.write('user_id\n' + '\n'.join(str(u) for u in COLD) + '\n')

ids, w = make_toy_history(N_USERS, V, d_max=3, seed=2)
model = build_recsys(feature_user_proto_param(D, KU, ids, w, V, use_id_feature=True),
                     N_USERS, N_ITEMS)
hist = model.user_feature_extractor.embedding_ext

with torch.no_grad():
    q_before = hist(torch.arange(N_USERS)).clone()
    id_rows_before = hist.embedding_layer.weight[V:].clone()

info = drop_cold_user_id_rows(model, variant)
check("t07.drop_applied",
      info is not None and info['n_cold_user_id_rows_zeroed'] == len(COLD), f"{info}")

with torch.no_grad():
    q_after = hist(torch.arange(N_USERS))
warm = [u for u in range(N_USERS) if u not in COLD]
check("t07.warm_users_bit_identical", torch.equal(q_after[warm], q_before[warm]))
# cold users: composition is now metadata-only == q_before − e_ID(u)
expected_cold = q_before[COLD] - id_rows_before[COLD]
check("t07.cold_users_metadata_only",
      torch.allclose(q_after[COLD], expected_cold, atol=1e-7),
      f"max |Δ| {(q_after[COLD] - expected_cold).abs().max():.3e}")
check("t07.cold_id_rows_zeroed",
      torch.equal(hist.embedding_layer.weight[V + torch.tensor(COLD)],
                  torch.zeros(len(COLD), D)))

# noid arm: clean no-op
model_noid = build_recsys(feature_user_proto_param(D, KU, ids, w, V, use_id_feature=False),
                          N_USERS, N_ITEMS)
check("t07.noid_arm_noop", drop_cold_user_id_rows(model_noid, variant) is None)

# missing cold_users.csv: no-op (the current phase's standard situation)
empty = os.path.join(tmp, 'empty')
os.makedirs(empty)
model2 = build_recsys(feature_user_proto_param(D, KU, ids, w, V, use_id_feature=True),
                      N_USERS, N_ITEMS)
check("t07.no_cold_users_csv_noop", drop_cold_user_id_rows(model2, empty) is None)

# config truth tables
fu_conf = {'ft_ext_param': {'ft_type': 'feature_user_proto',
                            'user_ft_ext_param': {'use_id_feature': True},
                            'item_ft_ext_param': {'ft_type': 'embedding'}}}
fu_conf_absent = {'ft_ext_param': {'ft_type': 'feature_user_proto',
                                   'user_ft_ext_param': {},
                                   'item_ft_ext_param': {'ft_type': 'embedding'}}}
fu_conf_noid = {'ft_ext_param': {'ft_type': 'feature_user_proto',
                                 'user_ft_ext_param': {'use_id_feature': False},
                                 'item_ft_ext_param': {'ft_type': 'embedding'}}}
fi_conf = {'ft_ext_param': {'ft_type': 'feature_item_proto',
                            'item_ft_ext_param': {'use_id_feature': True}}}
check("t07.expects_user_drop[fU+ids]", config_expects_user_id_drop(fu_conf) is True)
check("t07.expects_user_drop[fU absent key -> default True]",
      config_expects_user_id_drop(fu_conf_absent) is True)
check("t07.expects_user_drop[fU noid]", config_expects_user_id_drop(fu_conf_noid) is False)
check("t07.expects_user_drop[fI]", config_expects_user_id_drop(fi_conf) is False)
# fU must NOT trigger the ITEM-side drop expectation (its item branch is a free Embedding)
check("t07.item_drop_not_expected_for_fU", config_expects_id_drop(fu_conf) is False)
# and the item-side drop function itself is a clean no-op on the fU model
check("t07.item_drop_noop_on_fU_model", drop_cold_id_rows(model2, variant) is None)

shutil.rmtree(tmp)
print("\nt07: ALL PASS" if not FAILS else f"\nt07: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
