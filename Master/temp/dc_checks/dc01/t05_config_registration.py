# dc01 P5 unit test — config search space + registration in start.py and run_combo.py.
# Re-pointed at the SC.3 re-run (2026-07-12): the config now mirrors item_proto_chose_original
# (fI host), not proto_double_tie.
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
from confs.hyper_params import feature_item_proto_hyper_params as cfg, item_proto_chose_original_hyper_params as base

ft = cfg['ft_ext_param']
item_ft = ft['item_ft_ext_param']
user_ft = ft['user_ft_ext_param']

check("t05.top_ft_type", ft['ft_type'] == 'feature_item_proto', f"{ft['ft_type']}")
check("t05.use_id_feature_present", item_ft.get('use_id_feature') is True, f"{item_ft.get('use_id_feature')}")
# (edited 2026-07-12, S0-build extension component c: configs now carry the 'canonical'
# sentinel, resolved per dataset in start_hyper — pin the sentinel AND its H&M resolution)
check("t05.feature_fields_present", item_ft.get('feature_fields') == 'canonical',
      f"{item_ft.get('feature_fields')}")
from feature_extraction.feature_ids import resolve_canonical_fields
_hm_fields, _hm_layout = resolve_canonical_fields('hm_1_month')
check("t05.canonical_resolves_to_s05_five_on_hm",
      len(_hm_fields) == 5 and _hm_layout == 'fixed', f"{_hm_fields} ({_hm_layout})")

# user side mirrors item_proto_chose_original: plain embedding, nothing else
base_user = base['ft_ext_param']['user_ft_ext_param']
check("t05.user_side_mirrors_item_proto",
      user_ft == base_user == {'ft_type': 'embedding'},
      f"user_ft = {user_ft}")
# item side carries exactly the host's prototype hyperparams (so the factory can read them
# directly) — same keys as item_proto's item side, plus ONLY the two dc01 additions
proto_keys = ['sim_proto_weight', 'sim_batch_weight', 'use_weight_matrix', 'n_prototypes',
              'cosine_type', 'reg_proto_type', 'reg_batch_type']
check("t05.item_side_has_proto_hparams",
      all(k in item_ft for k in proto_keys) and item_ft['use_weight_matrix'] is False, "")
base_item_keys = set(base['ft_ext_param']['item_ft_ext_param'].keys())
extra = set(item_ft.keys()) - base_item_keys
check("t05.item_side_additions_are_exactly_dc01s",
      extra == {'use_id_feature', 'feature_fields'}, f"extra keys vs item_proto: {sorted(extra)}")
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
# Updated at SC.5 (F-DC01-11, 2026-07-12): the headline model IS auto-explainable now;
# the _noid/_f0 ablation arms stay out (separate model keys, host convention).
check("t05.run_combo_explainable_headline_only",
      em is not None and 'feature_item_proto' in em.group(1)
      and 'feature_item_proto_noid' not in em.group(1)
      and 'feature_item_proto_f0' not in em.group(1),
      f"EXPLAINABLE_MODELS = {{{em.group(1).strip() if em else '?'}}} (F-DC01-11)")

print("\nt05: ALL PASS" if not FAILS else f"\nt05: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
