# dc01 P5 unit test — config search space + registration in start.py and run_combo.py.
# Run: python Master/temp/dc_checks/dc01/t05_config_registration.py
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO)

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


# --- (1) hyper_params config ---
from confs.hyper_params import feature_item_proto_hyper_params as cfg, proto_double_tie_chose_original_hyper_params as base

ft = cfg['ft_ext_param']
item_ft = ft['item_ft_ext_param']
user_ft = ft['user_ft_ext_param']

check("t05.top_ft_type", ft['ft_type'] == 'feature_item_proto', f"{ft['ft_type']}")
check("t05.use_id_feature_present", item_ft.get('use_id_feature') is True, f"{item_ft.get('use_id_feature')}")
check("t05.feature_fields_present", isinstance(item_ft.get('feature_fields'), list) and len(item_ft['feature_fields']) >= 1,
      f"{item_ft.get('feature_fields')}")

# user side mirrors proto_double_tie (same keys/shape)
base_user = base['ft_ext_param']['user_ft_ext_param']
mirror_keys = ['sim_proto_weight', 'sim_batch_weight', 'use_weight_matrix', 'n_prototypes',
               'cosine_type', 'reg_proto_type', 'reg_batch_type']
check("t05.user_side_mirrors_double_tie",
      all(k in user_ft for k in mirror_keys) and user_ft['use_weight_matrix'] is False,
      f"missing: {[k for k in mirror_keys if k not in user_ft]}")
# item side also carries the prototype hyperparams (so the factory can read them directly)
check("t05.item_side_has_proto_hparams",
      all(k in item_ft for k in mirror_keys) and item_ft['use_weight_matrix'] is False, "")
# the tensor must NOT be in the static config (built in _build_model)
check("t05.no_tensor_in_config",
      'feature_ids' not in item_ft and 'n_features' not in item_ft,
      "feature_ids/n_features absent from static config")

# --- (2) start.py registration (source check; importing start.py would run argparse) ---
with open(os.path.join(REPO, 'start.py')) as f:
    start_src = f.read()
check("t05.start_cli_choice", "'feature_item_proto'" in start_src, "in argparse choices")
check("t05.start_elif_mapping", "feature_item_proto_hyper_params" in start_src, "config mapped in elif chain")

# --- (3) run_combo.py registration (source check; importing it pulls in Ray, which on the
#         laptop's Ray 1.6.0 lacks ray.tune.search — an env limitation, not a code issue) ---
import re

with open(os.path.join(REPO, 'Master', 'scripts', 'run_combo.py')) as f:
    rc_src = f.read()

mc = re.search(r"MODEL_CONFIGS\s*=\s*\{(.*?)\}", rc_src, re.DOTALL)
em = re.search(r"EXPLAINABLE_MODELS\s*=\s*\{([^}]*)\}", rc_src)
check("t05.run_combo_model_configs",
      mc is not None and "'feature_item_proto': feature_item_proto_hyper_params" in mc.group(1),
      "registered in MODEL_CONFIGS")
check("t05.run_combo_not_explainable_yet",
      em is not None and 'feature_item_proto' not in em.group(1),
      f"EXPLAINABLE_MODELS = {{{em.group(1).strip() if em else '?'}}}")

print("\nt05: ALL PASS" if not FAILS else f"\nt05: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
