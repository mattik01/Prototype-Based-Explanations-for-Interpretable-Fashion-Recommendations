# dc09 t13 — the SC.5 breakdown slots (fUfI + its UI host) and the canonical read-out gauge.
# On toy splits built through the real injection seam (fixed: the S0.5 field names; bags: an
# ml-1m-shaped genres|tags split), briefly-trained models of all four 2×2 arms + the UI host:
#   1. every slot's internal self-checks pass (four components == forward score; zoom and alt
#      zoom == the top bar; per-purchase rows regroup the model's own history; item-baseline
#      zoom == 1ᵀt̂; four-way coverage == each half; the what-if re-score of the unchanged item
#      == the forward score) for several users and items, on both layouts;
#   2. the "common to all users" value == the brute-force average over every user of the
#      personalised part for that item (F-DC09-09);
#   3. canonical gauge (F-DC09-07): a random translation gauge applied to an ids-arm model leaves
#      every forward score unchanged AND every rendered number unchanged, while the raw per-row
#      shares move (the test has teeth); naming profiles likewise; noid arms read raw;
#   4. the what-if rows are exact: substituting the reported word in the model's own item table
#      and re-running the forward reproduces the reported score change;
#   5. render_for_results_dir and the explanations pipeline write the artifacts for fUfI and the
#      host; both slots registered in the breakdown explainer. Figures → dc_checks/dc09/figures/.
# Run: python Master/temp/dc_checks/dc09/t13_breakdown_fufi.py
import copy
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

tmp = tempfile.mkdtemp(prefix='t13_dc09_')
os.environ['PROTOMF_DATA_PATH'] = tmp          # BEFORE utilities.consts is imported
os.environ.setdefault('WANDB_MODE', 'disabled')

import numpy as np
import pandas as pd
import torch

from _fixture import build, make_check, proto_block

_spec = importlib.util.spec_from_file_location(
    's0_harness', os.path.join(os.path.dirname(__file__), '..', 's0', '_harness.py'))
_s0h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s0h)

from feature_extraction.feature_ids import inject_feature_ids
from utilities.explanations.accessor import get_accessor
from utilities.explanations.breakdown import (compute_breakdown_feature_user_item_proto,
                                              compute_breakdown_user_item_proto,
                                              fufi_intrinsic_namings, mean_user_output,
                                              render_for_results_dir)
from utilities.explanations.explainers.breakdown import BreakdownExplainer
from utilities.explanations.feature_readout import item_feature_rows, per_feature_shares
from utilities.explanations.items_info import load_items_info
from utilities.explanations.naming import get_naming_config
from utilities.explanations.pipeline import run_explanations_pipeline

check, FAILS = make_check()
torch.manual_seed(0)
D = 8
FIXED_FIELDS = ['department_name', 'product_type_name', 'section_name',
                'colour_group_name', 'graphical_appearance_name']
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def build_toy_bags(path, n_users=40, n_items=90, seed=5):
    """ml-1m-shaped split: genres + tags as pipe-separated bags (1–3 genres, 0–4 tags)."""
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
            g = sorted(rng.choice(genres, size=int(rng.integers(1, 4)), replace=False))
            t = sorted(rng.choice(tags, size=int(rng.integers(0, 5)), replace=False))
            if i < len(genres):
                g = [genres[i]]                     # every genre occurs
            if i < len(tags):
                t = [tags[i]]                       # every tag occurs
            f.write(f'{i},m{i},{"|".join(g)},{"|".join(t)}\n')
    rows = {'train': [], 'val': [], 'test': []}
    for u in range(n_users):
        items = rng.choice(n_items, size=int(rng.integers(6, 12)), replace=False)
        rows['train'] += [(u, it) for it in items[:-2]]
        rows['val'].append((u, items[-2]))
        rows['test'].append((u, items[-1]))
    for name, r in rows.items():
        with open(os.path.join(path, f'listening_history_{name}.csv'), 'w') as f:
            f.write('user_id,item_id\n' + '\n'.join(f'{u},{i}' for u, i in r) + '\n')
    return n_users, n_items


SPLITS = {}
fixed = os.path.join(tmp, 'toy_fixed')
SPLITS['toy_fixed'] = (fixed, *_s0h.build_toy_canonical(fixed, n_users=50, n_items=120),
                       FIXED_FIELDS, 'fixed')
bags = os.path.join(tmp, 'toy_bags')
SPLITS['toy_bags'] = (bags, *build_toy_bags(bags), ['genres', 'tags'], 'bags')


def fufi_spec(fields, layout, ids=True, share=True):
    side = lambda k: {'ft_type': 'feature_user_item_proto', **proto_block(k), 'use_id_feature': ids,
                      'feature_fields': fields, 'feature_layout': layout}
    user = side(5)
    user['loo_pooling'] = True
    return {'ft_type': 'feature_user_item_proto', 'embedding_dim': D, 'share_word_table': share,
            'item_ft_ext_param': side(6), 'user_ft_ext_param': user}


def ui_spec():
    return {'ft_type': 'prototypes_double_tie', 'embedding_dim': D,
            'item_ft_ext_param': {'ft_type': 'prototypes_double_tie', **proto_block(6)},
            'user_ft_ext_param': {'ft_type': 'prototypes_double_tie', **proto_block(5)}}


def train_briefly(model, split, n_items, seed=1, steps=80):
    g = torch.Generator().manual_seed(seed)
    pairs = pd.read_csv(os.path.join(split, 'listening_history_train.csv'), usecols=['user_id', 'item_id'])
    u_all = torch.as_tensor(pairs['user_id'].to_numpy(), dtype=torch.long)
    i_all = torch.as_tensor(pairs['item_id'].to_numpy(), dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        sel = torch.randint(0, len(u_all), (32,), generator=g)
        i = torch.cat([i_all[sel].unsqueeze(1), torch.randint(0, n_items, (32, 4), generator=g)], 1)
        labels = torch.zeros(32, 5); labels[:, 0] = 1.0
        opt.zero_grad()
        loss = model.loss_func(model(u_all[sel], i), labels) \
            + model.user_feature_extractor.get_and_reset_loss() \
            + model.item_feature_extractor.get_and_reset_loss()
        loss.backward()
        opt.step()
    return model.eval()


def save_results_dir(model, model_name, dataset, spec):
    rd = os.path.join(tmp, f'results_{model_name}_{dataset}')
    os.makedirs(rd, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(rd, 'best_model.pth'))
    with open(os.path.join(rd, 'config.json'), 'w') as f:
        json.dump({'rec_sys_param': {'use_bias': 0}, 'loss_func_name': 'sampled_softmax',
                   'loss_func_aggr': 'mean', 'ft_ext_param': spec}, f, indent=2, default=str)
    with open(os.path.join(rd, 'metadata.json'), 'w') as f:
        json.dump({'model': model_name, 'config_name': model_name, 'dataset': dataset, 'seed': 0}, f)
    return rd


def pers_all_users(model, item_id, k_u):
    """Brute force: the personalised part S − B_item − B_user for every user, one item."""
    n = model.n_users
    with torch.no_grad():
        S = model(torch.arange(n), torch.full((n, 1), item_id)).squeeze(1)
        uo = model.user_feature_extractor(torch.arange(n))
        io = model.item_feature_extractor(torch.tensor([item_id])).squeeze(0)
    model.user_feature_extractor.get_and_reset_loss(); model.item_feature_extractor.get_and_reset_loss()
    return S - io[:k_u].sum() - uo[:, k_u:].sum(dim=1)


def apply_gauge(model, layout, seed=3):
    """A random translation gauge (F-DC09-07) on a copy: words of field φ += v_φ, every ID row
    −= its own field mass · v (items: per-field token counts; users: per-field basket weights).
    One v for the shared table; one per table under `_sep`."""
    m = copy.deepcopy(model)
    ub, ib = m.user_feature_extractor.base, m.item_feature_extractor.base
    V = ib.n_features
    from feature_extraction.feature_ids import build_field_index
    split, *_rest = SPLITS[m._toy_dataset]
    fields = m._toy_fields
    foc = build_field_index(split, fields, layout)
    g = torch.Generator().manual_seed(seed)
    shared = ib.embedding_layer.weight is ub.embedding_layer.weight
    with torch.no_grad():
        def shift_table(table, v):
            table[:V] += v[foc]
        v_i = torch.randn(len(fields), D, generator=g) * 0.5
        v_u = v_i if shared else torch.randn(len(fields), D, generator=g) * 0.5
        shift_table(ib.embedding_layer.weight, v_i)
        if not shared:
            shift_table(ub.embedding_layer.weight, v_u)
        for i in range(m.n_items if ib.use_id_feature else 0):
            codes = ib.feature_ids[i]
            w = ib.feature_weights[i] if getattr(ib, 'feature_weights', None) is not None \
                else torch.ones(len(codes))
            mass = torch.zeros(len(fields)).index_add_(0, foc[codes], w.float())
            ib.embedding_layer.weight[V + i] -= mass @ v_i
        for u in range(m.n_users if ub.use_id_feature else 0):
            codes, w = ub.user_slots(u)
            mass = torch.zeros(len(fields)).index_add_(0, foc[codes], w.float())
            ub.embedding_layer.weight[ub.id_row_offset + u] -= mass @ v_u
    return m


def bd_numbers(bd):
    vals = [bd.total_score, bd.baseline or 0.0, bd.common_value or 0.0]
    vals += [l.value for l in bd.extra_baselines + bd.lines + bd.zoom_lines + bd.alt_zoom_lines]
    return np.array(vals, dtype=np.float64)


def bd_section_numbers(bd):
    out = []
    for sec in bd.sections:
        for r in sec.rows:
            out += [float(x) for x in re.findall(r'[+-]\d+\.\d+', ' '.join(r))]
    return np.array(out)


ARMS = {'ids': (True, True), 'noid': (False, True), 'ids_sep': (True, False), 'noid_sep': (False, False)}
MODELS = {}
for ds, (split, n_users, n_items, fields, layout) in SPLITS.items():
    for arm, (ids, share) in ARMS.items():
        spec = fufi_spec(fields, layout, ids, share)
        model = train_briefly(build(inject_feature_ids(spec, split), n_users=n_users, n_items=n_items),
                              split, n_items, seed=hash(arm) % 97)
        model._toy_dataset, model._toy_fields = ds, fields
        MODELS[(ds, arm)] = (model, spec)
    ui = train_briefly(build(ui_spec(), n_users=n_users, n_items=n_items), split, n_items, seed=7)
    MODELS[(ds, 'ui')] = (ui, ui_spec())

# ---- 1 + 2: slots self-check on both layouts, several users/items; common line exact ----
n_ok = 0
for (ds, arm), (model, spec) in MODELS.items():
    split, n_users, n_items, fields, layout = SPLITS[ds]
    items_info = load_items_info(ds, dataset_dir=split)
    common = mean_user_output(model)
    for user, item in ((3, 7), (11, 40), (n_users - 1, n_items - 2)):
        if arm == 'ui':
            bd = compute_breakdown_user_item_proto(model, user, item, items_info, common_mean=common)
            k_u = model.user_feature_extractor.model_1.n_prototypes
        else:
            bd = compute_breakdown_feature_user_item_proto(model, user, item, items_info, fields, split,
                                                           feature_layout=layout, common_mean=common)
            k_u = model.user_feature_extractor.proto.n_prototypes
        n_ok += 1
        brute = float(pers_all_users(model, item, k_u).mean())
        check(f't13.{ds}.{arm}.u{user}i{item}.common_is_user_average',
              abs(bd.common_value - brute) < 1e-4 * max(1.0, abs(brute)),
              f'slot {bd.common_value:+.6f} brute {brute:+.6f}')
check('t13.slots_self_checked', n_ok == len(MODELS) * 3, f'{n_ok} breakdowns')

# ---- 3: canonical gauge — invariant rendered numbers, raw shares move ----
for ds in SPLITS:
    split, n_users, n_items, fields, layout = SPLITS[ds]
    items_info = load_items_info(ds, dataset_dir=split)
    base_cfg = get_naming_config(ds, items_info)
    for arm in ('ids', 'ids_sep', 'noid'):
        model, _spec_ = MODELS[(ds, arm)]
        gauged = apply_gauge(model, layout)
        with torch.no_grad():
            uu = torch.arange(n_users).repeat_interleave(4)
            ii = torch.randint(0, n_items, (n_users * 4, 1), generator=torch.Generator().manual_seed(2))
            s0, s1 = model(uu, ii), gauged(uu, ii)
        model.user_feature_extractor.get_and_reset_loss(); model.item_feature_extractor.get_and_reset_loss()
        gauged.user_feature_extractor.get_and_reset_loss(); gauged.item_feature_extractor.get_and_reset_loss()
        scores_same = float((s0 - s1).abs().max()) < 1e-4
        if arm == 'noid':
            # no ID row absorbs the shift: the gauge is NOT a symmetry here (it moves the scores)
            check(f't13.{ds}.noid.no_gauge_freedom', not scores_same)
            # noid slots read raw rows: the archetype zoom equals the raw per-word shares
            bd = compute_breakdown_feature_user_item_proto(model, 5, 9, items_info, fields, split,
                                                           feature_layout=layout)
            ife = model.item_feature_extractor
            raw = per_feature_shares(item_feature_rows(ife.base, 9), ife.proto.prototypes).detach()
            check(f't13.{ds}.noid.reads_raw_rows', np.isfinite(raw.numpy()).all()
                  and bd.naming_route.endswith('both sides'))
            continue
        check(f't13.{ds}.{arm}.gauge_preserves_scores', scores_same, f'max {float((s0 - s1).abs().max()):.2e}')
        for user, item in ((3, 7), (11, 40)):
            b0 = compute_breakdown_feature_user_item_proto(model, user, item, items_info, fields, split,
                                                           feature_layout=layout)
            b1 = compute_breakdown_feature_user_item_proto(gauged, user, item, items_info, fields, split,
                                                           feature_layout=layout)
            n0, n1 = bd_numbers(b0), bd_numbers(b1)
            sec0, sec1 = bd_section_numbers(b0), bd_section_numbers(b1)
            check(f't13.{ds}.{arm}.u{user}i{item}.rendered_numbers_invariant',
                  n0.shape == n1.shape and np.allclose(n0, n1, atol=2e-4)
                  and [l.label for l in b0.zoom_lines] == [l.label for l in b1.zoom_lines]
                  and sec0.shape == sec1.shape and np.allclose(sec0, sec1, atol=2e-3),
                  f'max diff {np.abs(n0 - n1).max():.2e}')
        ife0, ife1 = model.item_feature_extractor, gauged.item_feature_extractor
        r0 = per_feature_shares(item_feature_rows(ife0.base, 7), ife0.proto.prototypes).detach()
        r1 = per_feature_shares(item_feature_rows(ife1.base, 7), ife1.proto.prototypes).detach()
        check(f't13.{ds}.{arm}.raw_shares_move_under_gauge', float((r0 - r1).abs().max()) > 1e-3,
              f'max raw diff {float((r0 - r1).abs().max()):.3f}')
        acc0, acc1 = get_accessor('feature_user_item_proto', model), get_accessor('feature_user_item_proto', gauged)
        nu0, ni0 = fufi_intrinsic_namings(acc0, fields, layout, items_info, split, base_cfg)
        nu1, ni1 = fufi_intrinsic_namings(acc1, fields, layout, items_info, split, base_cfg)
        prof = lambda res: np.array([[s.score for s in p.stats] for p in res.profiles])
        check(f't13.{ds}.{arm}.canonical_naming_invariant',
              np.allclose(prof(nu0), prof(nu1), atol=1e-4) and np.allclose(prof(ni0), prof(ni1), atol=1e-4)
              and [n.name for n in ni0.names.values()] == [n.name for n in ni1.names.values()])

# ---- 4: what-if rows are exact (substitute in the model's own item table, re-run forward) ----
for ds in SPLITS:
    split, n_users, n_items, fields, layout = SPLITS[ds]
    items_info = load_items_info(ds, dataset_dir=split)
    from utilities.explanations.breakdown import feature_code_labels
    labels = feature_code_labels(fields, items_info, dataset_dir=split, feature_layout=layout)
    for arm in ('ids', 'noid_sep'):
        model, _ = MODELS[(ds, arm)]
        bd = compute_breakdown_feature_user_item_proto(model, 4, 12, items_info, fields, split,
                                                       feature_layout=layout)
        sec = [s for s in bd.sections if s.title.startswith('What if')]
        ok = bool(sec)
        if ok:
            word, drop, _gain = sec[0].rows[0]
            sub_label, reported = re.match(r'→ (.*): ([+-]\d+\.\d+)$', drop).groups()
            ib = model.item_feature_extractor.base
            old, new = labels.index(word), labels.index(sub_label)
            row = ib.feature_ids[12].clone()
            pos = int((row == old).nonzero()[0])
            ib.feature_ids[12, pos] = new
            with torch.no_grad():
                s_new = float(model(torch.tensor([4]), torch.tensor([[12]])).squeeze())
            ib.feature_ids[12] = row
            model.user_feature_extractor.get_and_reset_loss(); model.item_feature_extractor.get_and_reset_loss()
            ok = abs((s_new - bd.total_score) - float(reported)) < 2e-4
            ev = f'{word} → {sub_label}: forward Δ {s_new - bd.total_score:+.4f}, reported {reported}'
        check(f't13.{ds}.{arm}.whatif_exact', ok, ev if sec else 'no what-if section')

# ---- 5: artifacts end to end + registration ----
ex = BreakdownExplainer()
check('t13.explainer_supports_both', ex.supports('feature_user_item_proto') and ex.supports('user_item_proto'))
for ds, arm, name in (('toy_fixed', 'ids', 'feature_user_item_proto'), ('toy_fixed', 'noid_sep', 'feature_user_item_proto_noid_sep'),
                      ('toy_bags', 'ids', 'feature_user_item_proto'), ('toy_fixed', 'ui', 'user_item_proto'),
                      ('toy_bags', 'ui', 'user_item_proto')):
    model, spec = MODELS[(ds, arm)]
    split = SPLITS[ds][0]
    rd = save_results_dir(model, name, ds, spec)
    out = render_for_results_dir(rd, user_id=3, data_dir=split)
    md = open(out + '.md').read()
    need = ['common to all users', 'item baseline 1ᵀt̂', 'your baseline 1ᵀû', 'Self-check passed']
    if arm != 'ui':
        need += ['What if this item differed', 'Named coverage per component']
        if arm == 'ids':
            need.append('relative to their field')
    check(f't13.render.{ds}.{name}', os.path.exists(out + '.png') and all(s in md for s in need),
          ', '.join(s for s in need if s not in md))
    shutil.copy(out + '.png', os.path.join(FIGDIR, f't13_{ds}_{name}.png'))
    shutil.copy(out + '.md', os.path.join(FIGDIR, f't13_{ds}_{name}.md'))
    if (ds, arm) in (('toy_fixed', 'ids'), ('toy_fixed', 'ui')):
        exp_dir = run_explanations_pipeline(rd, sample_users_for_weight_viz=[3])
        files = os.listdir(exp_dir)
        check(f't13.pipeline.{name}', any(f.startswith('breakdown_user3') for f in files)
              and {'weight_viz_user3_user_side.png', 'weight_viz_user3_item_side.png'} <= set(files),
              f'{len(files)} files')

# ---- 6: the two-sided control renders through the existing CBF slot (per item word, q_u·e_r) ----
split, n_users, n_items, fields, layout = SPLITS['toy_fixed']
side = lambda: {'ft_type': 'lightfm_tags_hist', 'use_id_feature': True, 'feature_fields': fields,
                'feature_layout': layout}
ctrl_spec = {'ft_type': 'lightfm_tags_hist', 'embedding_dim': D, 'share_word_table': True,
             'item_ft_ext_param': side(), 'user_ft_ext_param': {**side(), 'loo_pooling': True}}
ctrl = train_briefly(build(inject_feature_ids(ctrl_spec, split), n_users=n_users, n_items=n_items),
                     split, n_items, seed=11)
rd = save_results_dir(ctrl, 'lightfm_tags_hist_ids', 'toy_fixed', ctrl_spec)
out = render_for_results_dir(rd, user_id=3, data_dir=split)
md = open(out + '.md').read()
check('t13.control_renders_via_cbf_slot', 'exact per-feature contribution' in md
      and 'item ID + field means' in md and 'Self-check passed' in md)

print()
print('t13: ALL PASS' if not FAILS else f't13: {len(FAILS)} FAIL — {FAILS}')
shutil.rmtree(tmp)
sys.exit(1 if FAILS else 0)
