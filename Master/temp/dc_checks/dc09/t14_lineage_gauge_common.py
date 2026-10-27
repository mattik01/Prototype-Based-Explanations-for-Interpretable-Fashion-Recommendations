# dc09 t14 — F-DC09-19 (SC.5 gate, user "yes to all three"): the two honesty devices built at dc09's
# SC.5 now run in every earlier slot of the lineage — fI and its I host, fU and its U host, and the
# CBF rows (lightfm_tags(_ids), and the two-sided control lightfm_tags_hist_ids). On toy splits
# through the real injection seam, both layouts, briefly trained models:
#   1. every slot passes its own self-checks, and its "common to all users" value equals the
#      brute-force average over every user of the bars' total for that item;
#   2. canonical gauge: in every arm with an ID row a random translation gauge keeps the scores and
#      every rendered number, while the raw per-row numbers move; noid arms read raw;
#   3. intrinsic naming of fI-ids / fU-ids reads the canonical rows (names invariant under the gauge);
#   4. render_for_results_dir writes the artifacts with the new lines.
# Run: python Master/temp/dc_checks/dc09/t14_lineage_gauge_common.py
import copy
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

tmp = tempfile.mkdtemp(prefix='t14_dc09_')
os.environ['PROTOMF_DATA_PATH'] = tmp
os.environ.setdefault('WANDB_MODE', 'disabled')

import numpy as np
import pandas as pd
import torch

from _fixture import build, make_check, proto_block

from feature_extraction.feature_ids import build_field_index, inject_feature_ids  # noqa: E402
from utilities.explanations.accessor import get_accessor  # noqa: E402
from utilities.explanations.breakdown import (  # noqa: E402
    compute_breakdown_feature_item_proto, compute_breakdown_feature_user_proto,
    compute_breakdown_item_proto, compute_breakdown_lightfm, compute_breakdown_user_proto,
    mean_user_output, render_for_results_dir)
from utilities.explanations.feature_readout import item_feature_rows  # noqa: E402
from utilities.explanations.gauge import field_of_codes  # noqa: E402
from utilities.explanations.items_info import load_items_info  # noqa: E402
from utilities.explanations.naming import (get_naming_config, intrinsic_naming_config,  # noqa: E402
                                           name_prototypes_intrinsic)

check, FAILS = make_check()
torch.manual_seed(0)
D = 8

# the two toy splits of t13 (built here the same way)
_s0 = importlib.util.spec_from_file_location('s0h', os.path.join(os.path.dirname(__file__), '..', 's0', '_harness.py'))
s0h = importlib.util.module_from_spec(_s0); _s0.loader.exec_module(s0h)
FIXED = ['department_name', 'product_type_name', 'section_name', 'colour_group_name',
         'graphical_appearance_name']


def build_toy_bags(path, n_users=40, n_items=90, seed=5):
    rng = np.random.default_rng(seed)
    os.makedirs(path, exist_ok=True)
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']
    tags = ['dark', 'funny', 'long', 'quirky', 'slow', 'violent', 'witty']
    with open(os.path.join(path, 'user_ids.csv'), 'w') as f:
        f.write('user_id,user\n' + '\n'.join(f'{u},u{u}' for u in range(n_users)) + '\n')
    with open(os.path.join(path, 'item_ids.csv'), 'w') as f:
        f.write('item_id,item\n' + '\n'.join(f'{i},m{i}' for i in range(n_items)) + '\n')
    with open(os.path.join(path, 'item_features.csv'), 'w') as f:
        f.write('item_id,item,genres,tags\n')
        for i in range(n_items):
            g = [genres[i]] if i < len(genres) else sorted(rng.choice(genres, int(rng.integers(1, 4)), replace=False))
            t = [tags[i]] if i < len(tags) else sorted(rng.choice(tags, int(rng.integers(0, 5)), replace=False))
            f.write(f'{i},m{i},{"|".join(g)},{"|".join(t)}\n')
    rows = {'train': [], 'val': [], 'test': []}
    for u in range(n_users):
        items = rng.choice(n_items, size=int(rng.integers(6, 12)), replace=False)
        rows['train'] += [(u, it) for it in items[:-2]]
        rows['val'].append((u, items[-2])); rows['test'].append((u, items[-1]))
    for name, r in rows.items():
        with open(os.path.join(path, f'listening_history_{name}.csv'), 'w') as f:
            f.write('user_id,item_id\n' + '\n'.join(f'{u},{i}' for u, i in r) + '\n')
    return n_users, n_items


SPLITS = {}
p = os.path.join(tmp, 'toy_fixed'); SPLITS['toy_fixed'] = (p, *s0h.build_toy_canonical(p, n_users=50, n_items=120), FIXED, 'fixed')
p = os.path.join(tmp, 'toy_bags'); SPLITS['toy_bags'] = (p, *build_toy_bags(p), ['genres', 'tags'], 'bags')


def comp(ft, ids, fields, layout, extra=None):
    return {'ft_type': ft, 'use_id_feature': ids, 'feature_fields': fields, 'feature_layout': layout,
            **(extra or {})}


def specs(fields, layout):
    pb = lambda k: proto_block(k)
    return {
        'feature_item_proto': lambda ids: {'ft_type': 'feature_item_proto', 'embedding_dim': D,
            'item_ft_ext_param': comp('feature_item_proto', ids, fields, layout, pb(6)),
            'user_ft_ext_param': {'ft_type': 'embedding'}},
        'feature_user_proto': lambda ids: {'ft_type': 'feature_user_proto', 'embedding_dim': D,
            'user_ft_ext_param': comp('feature_user_proto', ids, fields, layout, {**pb(5), 'loo_pooling': True}),
            'item_ft_ext_param': {'ft_type': 'embedding'}},
        'lightfm_tags': lambda ids: {'ft_type': 'lightfm', 'embedding_dim': D,
            'item_ft_ext_param': comp('lightfm', ids, fields, layout),
            'user_ft_ext_param': {'ft_type': 'embedding'}},
        'lightfm_tags_hist': lambda ids: {'ft_type': 'lightfm_tags_hist', 'embedding_dim': D,
            'share_word_table': True, 'item_ft_ext_param': comp('lightfm_tags_hist', ids, fields, layout),
            'user_ft_ext_param': comp('lightfm_tags_hist', ids, fields, layout, {'loo_pooling': True})},
        'item_proto': lambda ids: {'ft_type': 'prototypes', 'embedding_dim': D,
            'item_ft_ext_param': {'ft_type': 'prototypes', **pb(6)}, 'user_ft_ext_param': {'ft_type': 'embedding'}},
        'user_proto': lambda ids: {'ft_type': 'prototypes', 'embedding_dim': D,
            'user_ft_ext_param': {'ft_type': 'prototypes', **pb(5)}, 'item_ft_ext_param': {'ft_type': 'embedding'}},
    }


def train_briefly(model, split, n_items, seed=1, steps=60):
    g = torch.Generator().manual_seed(seed)
    pairs = pd.read_csv(os.path.join(split, 'listening_history_train.csv'), usecols=['user_id', 'item_id'])
    u_all = torch.as_tensor(pairs['user_id'].to_numpy()); i_all = torch.as_tensor(pairs['item_id'].to_numpy())
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        sel = torch.randint(0, len(u_all), (32,), generator=g)
        i = torch.cat([i_all[sel].unsqueeze(1), torch.randint(0, n_items, (32, 4), generator=g)], 1)
        labels = torch.zeros(32, 5); labels[:, 0] = 1.0
        opt.zero_grad()
        loss = model.loss_func(model(u_all[sel], i), labels)
        for fe in (model.user_feature_extractor, model.item_feature_extractor):
            if hasattr(fe, 'get_and_reset_loss'):
                loss = loss + fe.get_and_reset_loss()
        loss.backward(); opt.step()
    return model.eval()


def reset(model):
    for fe in (model.user_feature_extractor, model.item_feature_extractor):
        if hasattr(fe, 'get_and_reset_loss'):
            fe.get_and_reset_loss()


def slot(name, model, u, i, items_info, fields, split, layout, common=None):
    kw = dict(common_mean=common)
    if name == 'feature_item_proto':
        return compute_breakdown_feature_item_proto(model, u, i, items_info, fields, dataset_dir=split,
                                                    feature_layout=layout, **kw)
    if name == 'feature_user_proto':
        return compute_breakdown_feature_user_proto(model, u, i, items_info, fields, split,
                                                    feature_layout=layout, **kw)
    if name == 'item_proto':
        return compute_breakdown_item_proto(model, u, i, items_info, **kw)
    if name == 'user_proto':
        return compute_breakdown_user_proto(model, u, i, items_info, split, **kw)
    return compute_breakdown_lightfm(model, u, i, items_info, fields, dataset_dir=split,
                                     feature_layout=layout, **kw)


def brute_bars_avg(name, model, item):
    """mean over every user of S − (baseline lines) for one item, from the forward alone."""
    n = model.n_users
    with torch.no_grad():
        S = model(torch.arange(n), torch.full((n, 1), item)).squeeze(1)
        uo = model.user_feature_extractor(torch.arange(n))
        io = model.item_feature_extractor(torch.tensor([item])).squeeze(0)
    reset(model)
    if name in ('feature_item_proto', 'item_proto'):
        S = S - uo.sum(dim=1)                 # rank-inert baseline Σ_k u_k (shifted cosine)
    elif name in ('feature_user_proto', 'user_proto'):
        S = S - io.sum()                      # item baseline B(t) = 1ᵀt
    return float(S.mean())


def tables(model):
    """(word table owner, [(ID-row offset, n objects, per-object field mass fn)]) for the gauge."""
    ufe, ife = model.user_feature_extractor, model.item_feature_extractor
    item_fe = getattr(ife, 'embedding_ext', ife)
    user_fe = getattr(ufe, 'embedding_ext', ufe)
    return item_fe, user_fe


def apply_gauge(model, name, split, fields, layout, seed=3):
    m = copy.deepcopy(model)
    foc = build_field_index(split, fields, layout)
    v = torch.randn(len(fields), D, generator=torch.Generator().manual_seed(seed)) * 0.5
    item_fe, user_fe = tables(m)
    comps = []
    if hasattr(item_fe, 'feature_ids'):
        comps.append(('item', item_fe))
    if hasattr(user_fe, 'user_slots'):
        comps.append(('user', user_fe))
    with torch.no_grad():
        shifted = set()
        for side, fe in comps:
            W = fe.embedding_layer.weight
            V = fe.n_features
            if id(W) not in shifted:
                W[:V] += v[foc]; shifted.add(id(W))
            if not fe.use_id_feature:
                continue
            n = m.n_items if side == 'item' else m.n_users
            off = V if side == 'item' else fe.id_row_offset
            for o in range(n):
                if side == 'item':
                    codes = fe.feature_ids[o]
                    w = fe.feature_weights[o] if getattr(fe, 'feature_weights', None) is not None else torch.ones(len(codes))
                else:
                    codes, w = fe.user_slots(o)
                mass = torch.zeros(len(fields)).index_add_(0, foc[codes], w.float())
                W[off + o] -= mass @ v
    return m


def numbers(bd):
    vals = [bd.total_score, bd.baseline or 0.0, bd.common_value or 0.0]
    vals += [l.value for l in bd.lines + bd.zoom_lines + bd.alt_zoom_lines]
    return np.array(vals, dtype=np.float64)


IDS_ARMS = {'feature_item_proto', 'feature_user_proto', 'lightfm_tags', 'lightfm_tags_hist'}
for ds, (split, n_users, n_items, fields, layout) in SPLITS.items():
    items_info = load_items_info(ds, dataset_dir=split)
    base_cfg = get_naming_config(ds, items_info)
    S = specs(fields, layout)
    for name in S:
        for ids in ((True, False) if name in IDS_ARMS else (True,)):
            tag = f'{ds}.{name}{"" if ids else "_noid"}'
            spec = S[name](ids)
            model = train_briefly(build(inject_feature_ids(spec, split), n_users=n_users, n_items=n_items),
                                  split, n_items, seed=len(tag))
            common = mean_user_output(model)
            # 1. self-checks + common line vs brute force
            ok = True
            for u, i in ((3, 7), (11, 40), (n_users - 1, n_items - 2)):
                bd = slot(name, model, u, i, items_info, fields, split, layout, common)
                brute = brute_bars_avg(name, model, i)
                ok &= bd.common_value is not None and abs(bd.common_value - brute) < 1e-4 * max(1, abs(brute))
            check(f't14.{tag}.common_is_user_average', ok)
            # 2. canonical gauge on the composed arms with an ID row
            if name in IDS_ARMS and ids:
                g = apply_gauge(model, name, split, fields, layout)
                with torch.no_grad():
                    uu = torch.arange(n_users).repeat_interleave(3)
                    ii = torch.randint(0, n_items, (n_users * 3, 1), generator=torch.Generator().manual_seed(4))
                    d = float((model(uu, ii) - g(uu, ii)).abs().max())
                reset(model); reset(g)
                b0 = slot(name, model, 3, 7, items_info, fields, split, layout)
                b1 = slot(name, g, 3, 7, items_info, fields, split, layout)
                n0, n1 = numbers(b0), numbers(b1)
                item_fe = tables(model)[0]
                if name == 'feature_user_proto':      # raw word rows of the user composition
                    from utilities.explanations.history_readout import user_history_rows
                    r0 = user_history_rows(tables(model)[1], 3)[0]; r1 = user_history_rows(tables(g)[1], 3)[0]
                else:
                    r0 = item_feature_rows(item_fe, 7); r1 = item_feature_rows(tables(g)[0], 7)
                check(f't14.{tag}.gauge_invariant_render', d < 1e-4 and n0.shape == n1.shape
                      and np.allclose(n0, n1, atol=2e-4) and float((r0 - r1).abs().max()) > 1e-3,
                      f'score Δ {d:.1e}, rendered Δ {np.abs(n0 - n1).max():.1e}, raw rows Δ {float((r0 - r1).abs().max()):.2f}')
                check(f't14.{tag}.canonical_note', 'relative to their field' in ' '.join(b0.notes)
                      and 'gauge' not in (' '.join(b0.notes) + b0.naming_route).lower())
                if name in ('feature_item_proto', 'feature_user_proto'):
                    side = 'item' if name == 'feature_item_proto' else 'user'
                    a0, a1 = get_accessor(name, model), get_accessor(name, g)
                    from feature_extraction.feature_ids import build_code_to_field_value
                    c2fv = build_code_to_field_value(split, fields, layout)
                    foc = field_of_codes(fields, c2fv, items_info)
                    cfg = intrinsic_naming_config(base_cfg, fields)
                    protos0 = a0.item_prototypes() if side == 'item' else a0.user_prototypes()
                    protos1 = a1.item_prototypes() if side == 'item' else a1.user_prototypes()
                    nm0 = name_prototypes_intrinsic(a0.naming_word_rows(side, foc, len(fields)), protos0,
                                                    fields, items_info, cfg, side=side, code_to_field_value=c2fv)
                    nm1 = name_prototypes_intrinsic(a1.naming_word_rows(side, foc, len(fields)), protos1,
                                                    fields, items_info, cfg, side=side, code_to_field_value=c2fv)
                    prof = lambda r: np.array([[s.score for s in p.stats] for p in r.profiles])
                    check(f't14.{tag}.canonical_naming_invariant', np.allclose(prof(nm0), prof(nm1), atol=1e-4))
            elif name in IDS_ARMS:
                bd = slot(name, model, 3, 7, items_info, fields, split, layout)
                check(f't14.{tag}.noid_reads_raw', 'relative to their field' not in bd.naming_route
                      and not any('relative to their field' in n for n in bd.notes))
            # 4. render end to end (the model name keeps the arm, as in the fleet)
            mname = {'lightfm_tags': 'lightfm_tags_ids' if ids else 'lightfm_tags',
                     'lightfm_tags_hist': 'lightfm_tags_hist_ids' if ids else 'lightfm_tags_hist'}.get(
                name, name if ids else f'{name}_noid')
            rd = os.path.join(tmp, f'rd_{tag}')
            os.makedirs(rd, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(rd, 'best_model.pth'))
            json.dump({'rec_sys_param': {'use_bias': 0}, 'loss_func_name': 'sampled_softmax',
                       'loss_func_aggr': 'mean', 'ft_ext_param': spec}, open(os.path.join(rd, 'config.json'), 'w'), default=str)
            json.dump({'model': mname, 'dataset': ds, 'seed': 0}, open(os.path.join(rd, 'metadata.json'), 'w'))
            out = render_for_results_dir(rd, user_id=3, data_dir=split)
            md = open(out + '.md').read()
            check(f't14.{tag}.render', 'common to all users' in md and 'Self-check passed' in md)

print()
print('t14: ALL PASS' if not FAILS else f't14: {len(FAILS)} FAIL — {FAILS}')
shutil.rmtree(tmp)
sys.exit(1 if FAILS else 0)
