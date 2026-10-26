# dc09 t04 — the four 2×2 configs are registered and like-for-like with the UI host.
#   - both registries (start.py dispatch, run_combo.MODEL_CONFIGS) carry all four names;
#   - each config equals proto_double_tie_chose_original_hyper_params EXCEPT the declared
#     additions (ft_type names, share_word_table, per-side use_id_feature / feature_fields,
#     user-side loo_pooling) — same search space, same base hyperparameters;
#   - the arm flags are the 2×2 (ids/noid × shared/sep);
#   - 'canonical' resolves on BOTH sides to one field list + layout;
#   - the factory does not mutate the config it is handed (design doc §6.3).
import copy
import os
import sys

import torch

from _fixture import N_ITEMS, N_USERS, make_check, param_fUfI, toy_world
from confs import hyper_params as hp
from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_ids import resolve_feature_fields_in_conf

check, FAILS = make_check()
ARMS = {'feature_user_item_proto': (True, True), 'feature_user_item_proto_noid': (False, True),
        'feature_user_item_proto_sep': (True, False), 'feature_user_item_proto_noid_sep': (False, False)}

# --- registries ---
sys.argv = ['run_combo']
import Master.scripts.run_combo as rc  # noqa: E402

start_src = open(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'start.py')).read()
for name in ARMS:
    check(f't04.run_combo_has.{name}', name in rc.MODEL_CONFIGS)
    check(f't04.start_py_has.{name}', f"'{name}'" in start_src and f"model == '{name}'" in start_src)


def canon(x):
    """Comparable form of a config leaf (Ray Tune domains have no value equality)."""
    if hasattr(x, 'sampler') or hasattr(x, 'categories'):
        s = getattr(x, 'sampler', None)
        return (type(x).__name__, getattr(x, 'lower', None), getattr(x, 'upper', None),
                tuple(getattr(x, 'categories', ()) or ()), type(s).__name__, getattr(s, 'base', None))
    return x


def flat(d, p=''):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f'{p}{k}.') if isinstance(v, dict) else {f'{p}{k}': canon(v)})
    return out


host = flat(hp.proto_double_tie_chose_original_hyper_params)
ALLOWED_NEW = {'ft_ext_param.share_word_table', 'ft_ext_param.item_ft_ext_param.use_id_feature',
               'ft_ext_param.item_ft_ext_param.feature_fields', 'ft_ext_param.user_ft_ext_param.use_id_feature',
               'ft_ext_param.user_ft_ext_param.feature_fields', 'ft_ext_param.user_ft_ext_param.loo_pooling'}
FT_TYPES = {'ft_ext_param.ft_type', 'ft_ext_param.item_ft_ext_param.ft_type',
            'ft_ext_param.user_ft_ext_param.ft_type'}
for name, (ids, share) in ARMS.items():
    conf = rc.MODEL_CONFIGS[name]
    f = flat(conf)
    check(f't04.{name}.only_declared_additions', set(f) - set(host) == ALLOWED_NEW and not set(host) - set(f),
          f'extra {sorted(set(f) - set(host) - ALLOWED_NEW)}, missing {sorted(set(host) - set(f))}')
    same = [k for k in host if k not in FT_TYPES and f.get(k) != host[k]]
    check(f't04.{name}.host_search_space_verbatim', not same, f'differs: {same}')
    check(f't04.{name}.ft_types', all(f[k] == 'feature_user_item_proto' for k in FT_TYPES))
    check(f't04.{name}.arm_flags',
          f['ft_ext_param.share_word_table'] is share
          and f['ft_ext_param.item_ft_ext_param.use_id_feature'] is ids
          and f['ft_ext_param.user_ft_ext_param.use_id_feature'] is ids
          and f['ft_ext_param.user_ft_ext_param.loo_pooling'] is True,
          f'share={share} ids={ids}')
    for ds in ('hm_1_month', 'ml-1m'):
        c = copy.deepcopy(conf)
        resolve_feature_fields_in_conf(c, ds)
        it, us = c['ft_ext_param']['item_ft_ext_param'], c['ft_ext_param']['user_ft_ext_param']
        check(f't04.{name}.{ds}.canonical_resolves_identically_both_sides',
              it['feature_fields'] != 'canonical' and it['feature_fields'] == us['feature_fields']
              and it['feature_layout'] == us['feature_layout'], f"{it['feature_fields']} / {it['feature_layout']}")

# --- the factory does not mutate its input ---
w = toy_world()


def same_tree(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    return a == b


for share in (True, False):
    p = param_fUfI(w, share=share)
    before = copy.deepcopy(p)
    FeatureExtractorFactory.create_models(p, N_USERS, N_ITEMS)
    check(f't04.factory_does_not_mutate.{"shared" if share else "sep"}', same_tree(p, before))

print("\nt04: ALL PASS" if not FAILS else f"\nt04: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
