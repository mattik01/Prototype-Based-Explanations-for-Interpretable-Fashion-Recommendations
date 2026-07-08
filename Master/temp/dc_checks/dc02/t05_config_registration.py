# t05 — dc02 configs + CLI registration (P7):
#  - attr_item_proto_hyper_params structure (search space, attr_fields, knob defaults OFF, use_bias=0)
#  - debug + debug_knobs confs (fixed values; knobs variant flips exactly the intended keys)
#  - start.py + run_combo.py registration (source scan); run_combo EXPLAINABLE_MODELS excludes dc02.
# (Membership in the explanations pipeline's EXPLAINABLE_MODELS is asserted in t10 — it lands with P9.)
import copy
import os
import re
import sys

from _harness import make_check, REPO_ROOT

from confs.hyper_params import (attr_item_proto_hyper_params, attr_item_proto_debug_hyper_params,
                                attr_item_proto_debug_knobs_hyper_params,
                                proto_double_tie_chose_original_hyper_params)

check, fails = make_check()

HM_FIELDS = ['product_group_name', 'product_type_name', 'graphical_appearance_name',
             'colour_group_name', 'perceived_colour_master_name', 'index_group_name',
             'garment_group_name', 'section_name', 'department_name']
KNOB_DEFAULTS = {'nonneg_prototypes': False, 'crisp_weight': 0.0, 'sep_weight': 0.0,
                 'sep_margin': 0.5, 'push_weight': 0.0, 'push_n_items': 256, 'push_seed': 98765}

cfg = attr_item_proto_hyper_params
ft = cfg['ft_ext_param']
item = ft['item_ft_ext_param']
user = ft['user_ft_ext_param']

# --- top level ---
check("ft_type == attr_item_proto (all three levels)",
      ft['ft_type'] == 'attr_item_proto' and item['ft_type'] == 'attr_item_proto'
      and user['ft_type'] == 'attr_item_proto')
check("use_bias == 0 (dc02 §3.5(e), via base_param)", cfg['rec_sys_param']['use_bias'] == 0)

# --- item side ---
check("attr_fields == the 9 CSV fields in order", item['attr_fields'] == HM_FIELDS)
check("all 7 knob keys present with OFF defaults",
      all(item.get(k) == v for k, v in KNOB_DEFAULTS.items()),
      f"{ {k: item.get(k) for k in KNOB_DEFAULTS} }")
check("no item-side embedding_dim (dimension is V, data-determined)", 'embedding_dim' not in item)
check("no item-side max_norm (inert — no item embedding)", 'max_norm' not in item)
check("item side keeps host proto knobs",
      item['use_weight_matrix'] is False and item['cosine_type'] == 'shifted'
      and item['reg_proto_type'] == 'max' and item['reg_batch_type'] == 'max')

# --- user side mirrors the double-tie user space (same keys, same fixed values) ---
host_user = proto_double_tie_chose_original_hyper_params['ft_ext_param']['user_ft_ext_param']
check("user sub-dict keys mirror double_tie's", set(user.keys()) == set(host_user.keys()))
check("user fixed values mirror double_tie's",
      all(user[k] == host_user[k] for k in ('use_weight_matrix', 'cosine_type',
                                            'reg_proto_type', 'reg_batch_type')))

# --- search-space keys are tune objects (not accidentally fixed) ---
def is_tune(o):
    return type(o).__module__.startswith('ray.tune')

check("embedding_dim / n_prototypes / sim weights are tune search objects",
      is_tune(ft['embedding_dim']) and is_tune(item['n_prototypes']) and is_tune(user['n_prototypes'])
      and is_tune(item['sim_proto_weight']) and is_tune(user['sim_batch_weight']))

# --- debug conf: single fixed trial ---
dbg = attr_item_proto_debug_hyper_params
dft = dbg['ft_ext_param']
check("debug: num_samples == 1, tiny fixed dims",
      dbg['num_samples'] == 1 and dft['embedding_dim'] == 8
      and dft['item_ft_ext_param']['n_prototypes'] == 4
      and dft['user_ft_ext_param']['n_prototypes'] == 4)
check("debug: no tune objects anywhere",
      not any(is_tune(v) for v in list(dbg.values()) + list(dft.values())
              + list(dft['item_ft_ext_param'].values()) + list(dft['user_ft_ext_param'].values())
              + list(dbg['optim_param'].values())))
check("debug: knobs OFF", all(dft['item_ft_ext_param'][k] == v for k, v in KNOB_DEFAULTS.items()))

# --- knobs conf: differs from debug by exactly the intended item-side keys; base unmutated ---
kn = attr_item_proto_debug_knobs_hyper_params
kn_item = kn['ft_ext_param']['item_ft_ext_param']
expected_flips = {'nonneg_prototypes': True, 'crisp_weight': 1e-2, 'sep_weight': 1e-2,
                  'push_weight': 1e-2, 'push_n_items': 64}
diff = {k for k in kn_item if kn_item[k] != dft['item_ft_ext_param'].get(k)}
check("knobs conf flips exactly the intended keys", diff == set(expected_flips),
      f"diff={diff}")
check("knobs conf values", all(kn_item[k] == v for k, v in expected_flips.items()))
check("base debug conf unmutated by the deepcopy-derived variant",
      dft['item_ft_ext_param']['nonneg_prototypes'] is False
      and dft['item_ft_ext_param']['crisp_weight'] == 0.0)
check("knobs conf shares no mutable state with debug conf",
      kn['ft_ext_param'] is not dbg['ft_ext_param'] and kn_item is not dft['item_ft_ext_param'])

# --- registration source scans ---
start_src = open(os.path.join(REPO_ROOT, 'start.py')).read()
for name in ('attr_item_proto', 'attr_item_proto_debug', 'attr_item_proto_debug_knobs'):
    check(f"start.py: '{name}' in -m choices", f"'{name}'" in start_src)
    check(f"start.py: elif branch for '{name}'",
          re.search(rf"elif model == '{name}':", start_src) is not None)

rc_src = open(os.path.join(REPO_ROOT, 'Master', 'scripts', 'run_combo.py')).read()
mc = re.search(r'MODEL_CONFIGS = \{(.*?)\}', rc_src, re.S).group(1)
for name in ('attr_item_proto', 'attr_item_proto_debug', 'attr_item_proto_debug_knobs'):
    check(f"run_combo MODEL_CONFIGS: '{name}'", f"'{name}':" in mc)
em = re.search(r'EXPLAINABLE_MODELS = \{(.*?)\}', rc_src, re.S).group(1)
check("run_combo EXPLAINABLE_MODELS excludes attr_item_proto (auto-run stays opt-in)",
      'attr_item_proto' not in em)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
