# dc01 ablation configs — feature_item_proto_noid (use_id_feature=False) and _f0 (feature_fields=[]).
# Asserts each changes exactly ONE item-side knob vs the base, leaves the base unmutated, and is
# registered. Importing the configs also exercises copy.deepcopy of the tune-search objects.
# Run: python Master/temp/dc_checks/dc01/t09_ablation_configs.py
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO)

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


from confs.hyper_params import (
    feature_item_proto_hyper_params as base,
    feature_item_proto_noid_hyper_params as noid,
    feature_item_proto_f0_hyper_params as f0,
)

base_item = base['ft_ext_param']['item_ft_ext_param']
noid_item = noid['ft_ext_param']['item_ft_ext_param']
f0_item = f0['ft_ext_param']['item_ft_ext_param']

# both remain feature_item_proto (same factory branch)
check("t09.noid_ft_type", noid['ft_ext_param']['ft_type'] == 'feature_item_proto', "")
check("t09.f0_ft_type", f0['ft_ext_param']['ft_type'] == 'feature_item_proto', "")

# noid: ONLY use_id_feature changed
check("t09.noid_use_id_false", noid_item['use_id_feature'] is False, f"{noid_item['use_id_feature']}")
check("t09.noid_fields_unchanged", noid_item['feature_fields'] == base_item['feature_fields'],
      "feature_fields same as base")

# f0: ONLY feature_fields emptied (ID feature stays on -> reduces to user_item_proto)
check("t09.f0_fields_empty", f0_item['feature_fields'] == [], f"{f0_item['feature_fields']}")
check("t09.f0_use_id_true", f0_item['use_id_feature'] is True, f"{f0_item['use_id_feature']}")

# base must be UNMUTATED by the deepcopies
check("t09.base_use_id_true", base_item['use_id_feature'] is True, "")
# (edited 2026-07-12, S0-build extension component c: base now carries the 'canonical'
# sentinel — resolved per dataset in start_hyper; f0's literal [] bypasses resolution)
check("t09.base_fields_intact", base_item['feature_fields'] == 'canonical',
      f"{base_item['feature_fields']}")

# registration: start.py
with open(os.path.join(REPO, 'start.py')) as f:
    s = f.read()
check("t09.start_choices",
      "'feature_item_proto_noid'" in s and "'feature_item_proto_f0'" in s, "both in argparse choices")
check("t09.start_elif",
      "feature_item_proto_noid_hyper_params" in s and "feature_item_proto_f0_hyper_params" in s,
      "both mapped in elif chain")

# registration: run_combo MODEL_CONFIGS
with open(os.path.join(REPO, 'Master', 'scripts', 'run_combo.py')) as f:
    rc = f.read()
mc = re.search(r"MODEL_CONFIGS\s*=\s*\{(.*?)\}", rc, re.DOTALL)
check("t09.run_combo_noid",
      mc is not None and "'feature_item_proto_noid': feature_item_proto_noid_hyper_params" in mc.group(1), "")
check("t09.run_combo_f0",
      mc is not None and "'feature_item_proto_f0': feature_item_proto_f0_hyper_params" in mc.group(1), "")

print("\nt09: ALL PASS" if not FAILS else f"\nt09: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
