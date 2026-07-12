# dc05 t04 — config registration (design doc §3.2 config; charter C7 / F-DC01-01).
# Pins: the fU search space mirrors user_proto_chose_original_hyper_params EXACTLY (M1 stage
# bar); the noid arm differs ONLY in use_id_feature (fixed flag, never searched); all three
# configs registered in start.py choices and run_combo MODEL_CONFIGS; no tensor payload in any
# serialized config (the hist tensors are injected at build time only).
# Run: python Master/temp/dc_checks/dc05/t04_config_registration.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _harness  # noqa: F401  (repo root on sys.path)
from _harness import make_check

from confs.hyper_params import (feature_user_proto_hyper_params,
                                feature_user_proto_noid_hyper_params,
                                feature_user_proto_debug_hyper_params,
                                user_proto_chose_original_hyper_params)

check, FAILS = make_check()

fu = feature_user_proto_hyper_params
noid = feature_user_proto_noid_hyper_params
host = user_proto_chose_original_hyper_params


def sig(o):
    """Canonical signature: tune Domain objects compare by (type, plain-value attrs), not by
    object identity (deepcopy/independent construction must count as equal search spaces)."""
    mod = getattr(type(o), '__module__', '') or ''
    if mod.startswith('ray.tune'):
        attrs = {k: v for k, v in vars(o).items()
                 if isinstance(v, (int, float, str, bool, list, tuple))}
        return f"{type(o).__name__}({sorted(attrs.items())!r})"
    if isinstance(o, dict):
        return {k: sig(v) for k, v in o.items()}
    return repr(o)

# 1. search space mirrors the host EXACTLY outside ft_ext_param
top_keys = set(fu.keys())
check("t04.same_top_level_keys", top_keys == set(host.keys()), f"delta={top_keys ^ set(host.keys())}")
mismatch = [k for k in top_keys - {'ft_ext_param'} if sig(fu[k]) != sig(host[k])]
check("t04.top_level_mirrors_host", not mismatch, f"mismatch={mismatch}")

# 2. user-side proto block mirrors the host's (same searched dims), + exactly the two additions
fu_user = dict(fu['ft_ext_param']['user_ft_ext_param'])
host_user = dict(host['ft_ext_param']['user_ft_ext_param'])
additions = {'use_id_feature', 'feature_fields'}
check("t04.user_side_additions_exact",
      set(fu_user.keys()) - set(host_user.keys()) == additions,
      f"{set(fu_user.keys()) - set(host_user.keys())}")
mirrors = all(sig(fu_user[k]) == sig(host_user[k])
              for k in set(host_user.keys()) - {'ft_type'})
check("t04.user_proto_block_mirrors_host", mirrors)
check("t04.item_side_plain_embedding",
      fu['ft_ext_param']['item_ft_ext_param'] == {'ft_type': 'embedding'})
check("t04.canonical_5_fields",
      fu_user['feature_fields'] == ['department_name', 'product_type_name', 'section_name',
                                    'colour_group_name', 'graphical_appearance_name'])

# 3. noid differs ONLY in use_id_feature (fixed flag — F-DC01-01: never a searched flag)
diffs = []
def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            walk(a.get(k), b.get(k), f"{path}.{k}")
    elif sig(a) != sig(b):
        diffs.append(path)
walk(fu, noid)
check("t04.noid_differs_only_in_use_id_feature",
      diffs == ['.ft_ext_param.user_ft_ext_param.use_id_feature'], f"{diffs}")
check("t04.noid_flag_is_fixed_bool",
      noid['ft_ext_param']['user_ft_ext_param']['use_id_feature'] is False)

# 4. debug config: fixed values only (no tune objects anywhere)
def has_tune(o):
    if isinstance(o, dict):
        return any(has_tune(v) for v in o.values())
    return 'ray.tune' in type(o).__module__ if hasattr(type(o), '__module__') and type(o).__module__ else False
check("t04.debug_config_fixed_values_only", not has_tune(feature_user_proto_debug_hyper_params))

# 5. no tensor payload in ANY of the three configs (lightweight spec only)
def has_payload(o):
    if isinstance(o, dict):
        return any(k in ('hist_value_ids', 'hist_weights') for k in o) or \
               any(has_payload(v) for v in o.values())
    return False
check("t04.no_tensor_payload_in_configs",
      not any(has_payload(c) for c in (fu, noid, feature_user_proto_debug_hyper_params)))

# 6. registration: start.py choices + run_combo MODEL_CONFIGS
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
start_src = open(os.path.join(repo, 'start.py')).read()
check("t04.start_py_choices",
      all(f"'{m}'" in start_src for m in
          ('feature_user_proto', 'feature_user_proto_noid', 'feature_user_proto_debug')))
# run_combo.py cannot be IMPORTED on the laptop (it pulls experiment_helper -> Ray 2.x API;
# local env has Ray 1.6 — the known pre-existing env mismatch, cf. dc02 smoke_train docstring),
# so the registration is pinned at source level: each model key must appear as a MODEL_CONFIGS
# entry ('<name>': <name>_hyper_params).
rc_src = open(os.path.join(repo, 'Master', 'scripts', 'run_combo.py')).read()
check("t04.run_combo_registered",
      all(f"'{m}': {m}_hyper_params" in rc_src for m in
          ('feature_user_proto', 'feature_user_proto_noid', 'feature_user_proto_debug')))

print("\nt04: ALL PASS" if not FAILS else f"\nt04: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
