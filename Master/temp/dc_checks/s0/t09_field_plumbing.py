# t09 — per-dataset feature_fields plumbing (S0-build extension component c):
# CANONICAL_FEATURE_FIELDS registry + resolve_canonical_fields + resolve_feature_fields_in_conf
# + inject_feature_ids layout dispatch + factory feature_weights pass-through.
# Pins: sentinel resolution per dataset family (C8 manifests get literals), literal lists pass
# through untouched (F=0 / --retrain-config), unresolved-sentinel loud failure, bags payload
# reaching FeatureEmbedding through the factory bit-identically.
import copy
import os
import shutil
import sys
import tempfile

import torch

from _harness import make_check, REPO_ROOT

from feature_extraction.feature_ids import (resolve_canonical_fields,
                                            resolve_feature_fields_in_conf, inject_feature_ids)
from feature_extraction.feature_extractor_factories import FeatureExtractorFactory

check, fails = make_check()

S05_FIVE = ['department_name', 'product_type_name', 'section_name',
            'colour_group_name', 'graphical_appearance_name']

# ---------------------------------------------------------------------------
# 1. Registry resolution per dataset family
# ---------------------------------------------------------------------------
for ds in ('hm_1_month', 'hm_3_month', 'hm_full', 'hm_1_month_cold'):
    f, l = resolve_canonical_fields(ds)
    check(f"resolve: {ds} -> S0.5 five, fixed", f == S05_FIVE and l == 'fixed', f"{f} ({l})")
for ds in ('ml-1m', 'ml-1m_cold'):
    f, l = resolve_canonical_fields(ds)
    check(f"resolve: {ds} -> genres+tags, bags", f == ['genres', 'tags'] and l == 'bags',
          f"{f} ({l})")
try:
    resolve_canonical_fields('amazon2014')
    check("resolve: amazon2014 raises (no features — C2 amendment)", False, "no raise")
except ValueError as e:
    check("resolve: amazon2014 raises (no features — C2 amendment)", 'amazon2014' in str(e),
          str(e)[:70])

# ---------------------------------------------------------------------------
# 2. Conf resolution — sentinel replaced in place, literals untouched
# ---------------------------------------------------------------------------
from confs.hyper_params import (feature_item_proto_hyper_params, feature_user_proto_hyper_params,
                                lightfm_tags_ids_hyper_params, feature_item_proto_f0_hyper_params,
                                mf_hyper_params)

for name, cfg, side in (('fI', feature_item_proto_hyper_params, 'item_ft_ext_param'),
                        ('fU', feature_user_proto_hyper_params, 'user_ft_ext_param'),
                        ('lightfm_tags_ids', lightfm_tags_ids_hyper_params, 'item_ft_ext_param')):
    c = copy.deepcopy(cfg)
    resolve_feature_fields_in_conf(c, 'hm_1_month')
    spec = c['ft_ext_param'][side]
    check(f"conf({name}, hm_1_month): sentinel -> S0.5 five + layout fixed",
          spec['feature_fields'] == S05_FIVE and spec['feature_layout'] == 'fixed',
          f"{spec['feature_fields']} ({spec.get('feature_layout')})")
    c2 = copy.deepcopy(cfg)
    resolve_feature_fields_in_conf(c2, 'ml-1m')
    spec2 = c2['ft_ext_param'][side]
    check(f"conf({name}, ml-1m): sentinel -> genres+tags + layout bags",
          spec2['feature_fields'] == ['genres', 'tags'] and spec2['feature_layout'] == 'bags',
          f"{spec2['feature_fields']} ({spec2.get('feature_layout')})")

# literal [] (F=0 ablation) passes through untouched — no layout key added
c_f0 = copy.deepcopy(feature_item_proto_f0_hyper_params)
resolve_feature_fields_in_conf(c_f0, 'ml-1m')
f0_spec = c_f0['ft_ext_param']['item_ft_ext_param']
check("conf(f0): literal [] untouched by resolution (no feature_layout injected)",
      f0_spec['feature_fields'] == [] and 'feature_layout' not in f0_spec,
      f"{f0_spec['feature_fields']}")

# non-feature config untouched
c_mf = copy.deepcopy(mf_hyper_params)
before = repr(c_mf)
resolve_feature_fields_in_conf(c_mf, 'hm_1_month')
check("conf(mf): non-feature config untouched", repr(c_mf) == before, "")

# ---------------------------------------------------------------------------
# 3. inject_feature_ids — sentinel guard + layout dispatch
# ---------------------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix='s0_t09_')


def write_dir(feature_rows, feature_header, train_pairs=None, n_users=3):
    d = tempfile.mkdtemp(prefix='case_', dir=tmp)
    with open(os.path.join(d, 'item_features.csv'), 'w') as f:
        f.write(feature_header + '\n')
        for r in feature_rows:
            f.write(','.join(str(x) for x in r) + '\n')
    if train_pairs is not None:
        with open(os.path.join(d, 'user_ids.csv'), 'w') as f:
            f.write('user_id,user\n' + '\n'.join(f'{u},u{u}' for u in range(n_users)) + '\n')
        with open(os.path.join(d, 'listening_history_train.csv'), 'w') as f:
            f.write('user_id,item_id\n')
            for u, i in train_pairs:
                f.write(f'{u},{i}\n')
    return d


ml_dir = write_dir(
    [(0, 'Action|Comedy', 'funny'), (1, 'Drama', 'dark|long'),
     (2, 'Comedy', ''), (3, 'Action|Drama', 'dark|funny|long')],
    'item_id,genres,tags',
    train_pairs=[(0, 0), (0, 1), (1, 2), (2, 0), (2, 2), (2, 3)])

# unresolved sentinel fails loud at injection
try:
    inject_feature_ids({'ft_type': 'lightfm',
                        'item_ft_ext_param': {'ft_type': 'lightfm', 'feature_fields': 'canonical'},
                        'user_ft_ext_param': {'ft_type': 'embedding'}}, ml_dir)
    check("inject: unresolved sentinel raises", False, "no raise")
except ValueError as e:
    check("inject: unresolved sentinel raises", 'canonical' in str(e), str(e)[:70])

# bags layout: item payload gains feature_weights
lf_bags = inject_feature_ids({'ft_type': 'lightfm',
                              'item_ft_ext_param': {'ft_type': 'lightfm',
                                                    'feature_fields': ['genres', 'tags'],
                                                    'feature_layout': 'bags'},
                              'user_ft_ext_param': {'ft_type': 'embedding'}}, ml_dir)
lf_item = lf_bags['item_ft_ext_param']
check("inject(lightfm, bags): feature_ids + feature_weights + n_features injected",
      lf_item['feature_ids'].shape == (4, 5) and lf_item['feature_weights'].shape == (4, 5)
      and lf_item['n_features'] == 6, f"{tuple(lf_item['feature_ids'].shape)}, nf={lf_item['n_features']}")

# fixed layout / absent key (legacy saved configs): NO feature_weights key
lf_fixed = inject_feature_ids({'ft_type': 'lightfm',
                               'item_ft_ext_param': {'ft_type': 'lightfm',
                                                     'feature_fields': ['genres']},
                               'user_ft_ext_param': {'ft_type': 'embedding'}}, ml_dir)
check("inject(lightfm, absent layout = fixed): no feature_weights key (legacy-config compat)",
      'feature_weights' not in lf_fixed['item_ft_ext_param'], "")

# invalid layout raises
try:
    inject_feature_ids({'ft_type': 'lightfm',
                        'item_ft_ext_param': {'ft_type': 'lightfm', 'feature_fields': ['genres'],
                                              'feature_layout': 'nosuch'},
                        'user_ft_ext_param': {'ft_type': 'embedding'}}, ml_dir)
    check("inject: invalid feature_layout raises", False, "no raise")
except ValueError as e:
    check("inject: invalid feature_layout raises", 'feature_layout' in str(e), str(e)[:60])

# fU bags: hist tensors built through the layout
fu_bags = inject_feature_ids({'ft_type': 'feature_user_proto',
                              'user_ft_ext_param': {'ft_type': 'feature_user_proto',
                                                    'feature_fields': ['genres', 'tags'],
                                                    'feature_layout': 'bags',
                                                    'n_prototypes': 3, 'use_weight_matrix': False,
                                                    'use_id_feature': True},
                              'item_ft_ext_param': {'ft_type': 'embedding'}}, ml_dir)
fu_user = fu_bags['user_ft_ext_param']
check("inject(fU, bags): hist_value_ids/hist_weights/n_features injected",
      fu_user['hist_value_ids'].shape == fu_user['hist_weights'].shape
      and fu_user['n_features'] == 6, f"{tuple(fu_user['hist_value_ids'].shape)}")

# ---------------------------------------------------------------------------
# 4. Factory pass-through: bags payload reaches FeatureEmbedding bit-identically
# ---------------------------------------------------------------------------
torch.manual_seed(38210573)
# fixed-width toy so bags(all-ones) vs fixed must be bitwise equal through the factory
fx_dir = write_dir([(0, 'blue', 'square'), (1, 'green', 'square'), (2, 'red', 'circle'),
                    (3, 'blue', 'circle')], 'item_id,color,shape')
spec_fixed = inject_feature_ids({'ft_type': 'lightfm',
                                 'item_ft_ext_param': {'ft_type': 'lightfm',
                                                       'feature_fields': ['color', 'shape'],
                                                       'use_id_feature': True},
                                 'user_ft_ext_param': {'ft_type': 'embedding'}}, fx_dir)
spec_bags = inject_feature_ids({'ft_type': 'lightfm',
                                'item_ft_ext_param': {'ft_type': 'lightfm',
                                                      'feature_fields': ['color', 'shape'],
                                                      'feature_layout': 'bags',
                                                      'use_id_feature': True},
                                'user_ft_ext_param': {'ft_type': 'embedding'}}, fx_dir)
for s in (spec_fixed, spec_bags):
    s['embedding_dim'] = 6
u1, it1 = FeatureExtractorFactory.create_models(spec_fixed, n_users=3, n_items=4)
u2, it2 = FeatureExtractorFactory.create_models(spec_bags, n_users=3, n_items=4)
it1.init_parameters()
it2.load_state_dict(it1.state_dict())
idx = torch.tensor([[0, 2], [3, 1]])
check("factory(lightfm): bags payload consumed, fixed-vs-bags bitwise equal",
      torch.equal(it1(idx), it2(idx)), "")
check("factory(lightfm): bag model exposes feature_weights buffer",
      it2.feature_weights is not None and it1.feature_weights is None, "")

# fI branch: bags payload builds and forwards finitely on the multi-value toy
fi_spec = inject_feature_ids({'ft_type': 'feature_item_proto',
                              'item_ft_ext_param': {'ft_type': 'feature_item_proto',
                                                    'feature_fields': ['genres', 'tags'],
                                                    'feature_layout': 'bags',
                                                    'n_prototypes': 3, 'use_weight_matrix': False,
                                                    'use_id_feature': True},
                              'user_ft_ext_param': {'ft_type': 'embedding'}}, ml_dir)
fi_spec['embedding_dim'] = 6
fu_ext, fi_ext = FeatureExtractorFactory.create_models(fi_spec, n_users=3, n_items=4)
out = fi_ext(torch.arange(4))
check("factory(fI): bags payload builds; prototype forward finite, shape (4, K=3)",
      out.shape == (4, 3) and bool(torch.isfinite(out).all()), f"{tuple(out.shape)}")

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
