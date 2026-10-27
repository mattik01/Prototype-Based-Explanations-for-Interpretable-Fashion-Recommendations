# dc09 t15 — the SC.8 one-off measurement script (Master/scripts/sc8_fufi_readout.py).
# On toy splits built through the real injection seam (fixed: the S0.5 field names; bags: an
# ml-1m-shaped genres|tags split), briefly-trained models of all four 2×2 arms + the UI host:
#   1. A: every two-way decomposition (U-half, I-half, personalised, score) equals the brute-force
#      decomposition of the explicit user×item grid; the components sum to the forward score; the
#      common part equals the brute-force user average;
#   2. B: per pair, the same-word / different-word / ID-involving split equals an explicit double
#      sum over the user's and the item's rows (field-mean read-out in ids arms) and sums to the
#      forward personalised score;
#   3. reparametrisation stress (F-DC09-07): a random translation gauge on ids arms (shared and
#      `_sep`, both layouts) leaves every reported number of A, B, D and E unchanged, while the
#      RAW same-word split moves and, on bags, the raw variance split moves (the test has teeth);
#   4. C: the basket-twin fractions equal a brute-force count; D: row maxima equal brute force;
#   5. the CLI writes readout.json + readout.md for a saved toy results dir.
# Run: python Master/temp/dc_checks/dc09/t15_sc8_readout.py
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

tmp = tempfile.mkdtemp(prefix='t15_dc09_')
os.environ['PROTOMF_DATA_PATH'] = tmp          # BEFORE utilities.consts is imported
os.environ.setdefault('WANDB_MODE', 'disabled')

import copy

import numpy as np
import pandas as pd
import torch

from _fixture import build, make_check, proto_block

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
_spec = importlib.util.spec_from_file_location(
    's0_harness', os.path.join(os.path.dirname(__file__), '..', 's0', '_harness.py'))
_s0h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s0h)
_rspec = importlib.util.spec_from_file_location(
    'sc8_fufi_readout', os.path.join(REPO, 'Master', 'scripts', 'sc8_fufi_readout.py'))
R = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(R)

from feature_extraction.feature_ids import build_field_index, inject_feature_ids
from utilities.explanations.gauge import field_means

check, FAILS = make_check()
torch.manual_seed(0)
D = 8
FIXED_FIELDS = ['department_name', 'product_type_name', 'section_name',
                'colour_group_name', 'graphical_appearance_name']


def build_toy_bags(path, n_users=40, n_items=90, seed=5):
    """ml-1m-shaped split: genres + tags as pipe-separated bags (1–3 genres, 0–4 tags) — t13's."""
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
                g = [genres[i]]
            if i < len(tags):
                t = [tags[i]]
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


def apply_gauge(model, split, fields, layout, seed=3):
    """t13's random translation gauge (F-DC09-07) on a copy: words of field φ += v_φ, every ID row
    −= its own field mass · v. One v for the shared table; one per table under `_sep`."""
    m = copy.deepcopy(model)
    ub, ib = m.user_feature_extractor.base, m.item_feature_extractor.base
    V = ib.n_features
    foc = build_field_index(split, fields, layout)
    g = torch.Generator().manual_seed(seed)
    shared = ib.embedding_layer.weight is ub.embedding_layer.weight
    with torch.no_grad():
        v_i = torch.randn(len(fields), D, generator=g) * 0.5
        v_u = v_i if shared else torch.randn(len(fields), D, generator=g) * 0.5
        ib.embedding_layer.weight[:V] += v_i[foc]
        if not shared:
            ub.embedding_layer.weight[:V] += v_u[foc]
        for i in range(m.n_items):
            codes = ib.feature_ids[i]
            w = ib.feature_weights[i] if getattr(ib, 'feature_weights', None) is not None \
                else torch.ones(len(codes))
            mass = torch.zeros(len(fields)).index_add_(0, foc[codes], w.float())
            ib.embedding_layer.weight[V + i] -= mass @ v_i
        for u in range(m.n_users):
            codes, w = ub.user_slots(u)
            mass = torch.zeros(len(fields)).index_add_(0, foc[codes], w.float())
            ub.embedding_layer.weight[ub.id_row_offset + u] -= mass @ v_u
    return m


def numbers(d, prefix=''):
    """Every float in a readout dict, keyed by path (raw-split keys flagged for exclusion)."""
    out = {}
    for k, v in d.items():
        p = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(numbers(v, p))
        elif isinstance(v, float) and np.isfinite(v):
            out[p] = v
    return out


ARMS = {'ids': (True, True), 'noid': (False, True), 'ids_sep': (True, False), 'noid_sep': (False, False)}
MODELS = {}
for ds, (split, n_users, n_items, fields, layout) in SPLITS.items():
    for arm, (ids, share) in ARMS.items():
        spec = fufi_spec(fields, layout, ids, share)
        MODELS[(ds, arm)] = (train_briefly(build(inject_feature_ids(spec, split), n_users=n_users,
                                                 n_items=n_items), split, n_items,
                                           seed=sum(map(ord, arm)) % 97), spec)
    MODELS[(ds, 'ui')] = (train_briefly(build(ui_spec(), n_users=n_users, n_items=n_items), split,
                                        n_items, seed=7), ui_spec())

READ = {}
for (ds, arm), (model, _s) in MODELS.items():
    split, n_users, n_items, fields, layout = SPLITS[ds]
    READ[(ds, arm)] = R.run_on_model(model, split, None if arm == 'ui' else fields, layout,
                                     sample_users=12, seed=1)


def grid_parts(model, k_u, aff):
    with torch.no_grad():
        Uo = R._batched(model.user_feature_extractor, model.n_users)
        Io = R._batched(model.item_feature_extractor, model.n_items)
        S = torch.stack([model(torch.full((model.n_items,), u), torch.arange(model.n_items)[:, None]).squeeze(1)
                         for u in range(model.n_users)]).double()
    R._reset(model.user_feature_extractor, model.item_feature_extractor)
    a_u, c_u, a_t, c_t = aff
    U = (Uo[:, :k_u] - c_u) @ Io[:, :k_u].T
    I = Uo[:, k_u:] @ (Io[:, k_u:] - c_t).T
    B = c_u * Io[:, :k_u].sum(1)[None, :] + c_t * Uo[:, k_u:].sum(1)[:, None]
    return {'U_half': U, 'I_half': I, 'personalised': U + I, 'score': U + I + B}, S


def brute_two_way(H):
    H = H.numpy()
    m = H.mean(); a = H.mean(0) - m; b = H.mean(1) - m
    r = H - m - a[None, :] - b[:, None]
    nu, ni = H.shape
    return dict(grand_mean=m, ss_item=nu * (a ** 2).sum(), ss_user=ni * (b ** 2).sum(),
                ss_interaction=(r ** 2).sum(), common=m + a)


# ---- 1: A equals the brute-force decomposition; components sum to the forward score ----
for (ds, arm), (model, _s) in MODELS.items():
    up, ip, kind = R.proto_layers(model)
    aff = (*R.cosine_affine(up)[:2], *R.cosine_affine(ip)[:2])
    parts, S = grid_parts(model, up.n_prototypes, aff)
    check(f't15.{ds}.{arm}.A.components_sum_to_forward', float((parts['score'] - S).abs().max()) < 1e-4,
          f'max err {float((parts["score"] - S).abs().max()):.1e}')
    A = READ[(ds, arm)]['A_common_mode']
    worst = 0.0
    for c in ('U_half', 'I_half', 'personalised', 'score'):
        bf = brute_two_way(parts[c])
        tot = bf['ss_item'] + bf['ss_user'] + bf['ss_interaction']
        for k in ('ss_item', 'ss_user', 'ss_interaction'):
            worst = max(worst, abs(A[c][k] - bf[k]) / max(tot, 1e-12))
        worst = max(worst, abs(A[c]['grand_mean'] - bf['grand_mean']))
    check(f't15.{ds}.{arm}.A.two_way_equals_brute_force', worst < 1e-9, f'worst rel {worst:.1e}')

# ---- 2: B per pair equals an explicit row-by-row double sum ----
for (ds, arm), (model, _s) in MODELS.items():
    if arm == 'ui':
        continue
    split, n_users, n_items, fields, layout = SPLITS[ds]
    foc = build_field_index(split, fields, layout)
    nf = len(fields)
    ufe, ife = model.user_feature_extractor, model.item_feature_extractor
    ub, ib = ufe.base, ife.base
    up, ip, _ = R.proto_layers(model)
    a_u, c_u, _ = R.cosine_affine(up); a_t, c_t, _ = R.cosine_affine(ip)
    aff = (a_u, c_u, a_t, c_t)
    with torch.no_grad():
        Uo, Io = R._batched(ufe, n_users), R._batched(ife, n_items)
        Q_u, Q_i = R._batched(ub, n_users), R._batched(ib, n_items)
    R._reset(ufe, ife)
    TU, TI = ub.embedding_layer.weight.detach().double(), ib.embedding_layer.weight.detach().double()
    V = ib.n_features
    mU = field_means(TU[:V], foc, nf) if ub.use_id_feature else torch.zeros(nf, D, dtype=torch.float64)
    mI = field_means(TI[:V], foc, nf) if ib.use_id_feature else torch.zeros(nf, D, dtype=torch.float64)
    Pu, Pt = R._unit(up.prototypes.detach().double()), R._unit(ip.prototypes.detach().double())
    A_ = a_u * Pu.T @ ife.linear_layer.weight.detach().double()
    B_ = a_t * ufe.linear_layer.weight.detach().double().T @ Pt
    # same/different-word parts: both sides read the same float64 rows → tight; the total (and
    # ID-involving = total − words) use q from the float32 forward, as the forward does → float32
    worst, worst_q, worst_sum = 0.0, 0.0, 0.0
    for u, i in ((3, 7), (11, 40), (n_users - 1, n_items - 2)):
        codes_u, w_u = ub.user_slots(u)
        rows_u = [(int(c), float(w), TU[int(c)] - mU[foc[int(c)]]) for c, w in zip(codes_u, w_u)]
        mass_u = torch.zeros(nf, dtype=torch.float64).index_add_(0, foc[codes_u], w_u.double())
        id_u = (TU[ub.id_row_offset + u] + mass_u @ mU) if ub.use_id_feature else None
        codes_i = ib.feature_ids[i]
        w_i = ib.feature_weights[i] if getattr(ib, 'feature_weights', None) is not None else torch.ones(len(codes_i))
        rows_i = [(int(c), float(w), TI[int(c)] - mI[foc[int(c)]]) for c, w in zip(codes_i, w_i) if float(w) != 0]
        mass_i = torch.zeros(nf, dtype=torch.float64).index_add_(0, foc[codes_i], w_i.double())
        id_i = (TI[V + i] + mass_i @ mI) if ib.use_id_feature else None
        M = A_ / Q_u[u].norm() + B_ / Q_i[i].norm()
        same = sum(wu * wi * float(xu @ M @ xi) for cu, wu, xu in rows_u for ci, wi, xi in rows_i if cu == ci)
        diff = sum(wu * wi * float(xu @ M @ xi) for cu, wu, xu in rows_u for ci, wi, xi in rows_i if cu != ci)
        xu_all = [w * x for _c, w, x in rows_u] + ([id_u] if id_u is not None else [])
        xi_all = [w * x for _c, w, x in rows_i] + ([id_i] if id_i is not None else [])
        total = sum(float(a @ M @ b) for a in xu_all for b in xi_all)
        idinv = total - same - diff
        got = R.instrument_pairs(model, Uo, Io, Q_u, Q_i, up.n_prototypes, aff, foc, nf,
                                 {'one': {u: [i]}})['one']
        for k, v in (('mean_same_word', same), ('mean_different_word', diff)):
            worst = max(worst, abs(got[k] - v) / (1.0 + abs(v)))
        for k, v in (('mean_id_involving', idinv), ('mean_personalised', total)):
            worst_q = max(worst_q, abs(got[k] - v) / (1.0 + abs(v)))
        fwd = float(Uo[u] @ Io[i] - c_u * Io[i, :up.n_prototypes].sum() - c_t * Uo[u, up.n_prototypes:].sum())
        worst_sum = max(worst_sum, abs(total - fwd) / (1.0 + abs(fwd)))
    check(f't15.{ds}.{arm}.B.word_pairs_equal_double_sum', worst < 1e-12, f'worst rel {worst:.1e}')
    check(f't15.{ds}.{arm}.B.total_and_id_equal_double_sum', worst_q < 1e-5, f'worst rel {worst_q:.1e} (float32 q)')
    check(f't15.{ds}.{arm}.B.rows_reproduce_forward', worst_sum < 1e-4, f'worst rel {worst_sum:.1e}')

# ---- 3: reparametrisation stress — reported numbers invariant, raw splits move ----
for ds in SPLITS:
    split, n_users, n_items, fields, layout = SPLITS[ds]
    for arm in ('ids', 'ids_sep'):
        model, _s = MODELS[(ds, arm)]
        gauged = apply_gauge(model, split, fields, layout)
        with torch.no_grad():
            uu = torch.arange(n_users).repeat_interleave(3)
            ii = torch.randint(0, n_items, (n_users * 3, 1), generator=torch.Generator().manual_seed(2))
            ds_ = float((model(uu, ii) - gauged(uu, ii)).abs().max())
        R._reset(model.user_feature_extractor, model.item_feature_extractor,
                 gauged.user_feature_extractor, gauged.item_feature_extractor)
        check(f't15.{ds}.{arm}.gauge_keeps_scores', ds_ < 1e-4, f'max Δscore {ds_:.1e}')
        r0, r1 = READ[(ds, arm)], R.run_on_model(gauged, split, fields, layout, sample_users=12, seed=1)
        n0, n1 = numbers(r0), numbers(r1)
        keys = [k for k in n0 if '_raw' not in k and 'max_rel_err' not in k]
        delta = max(abs(n0[k] - n1[k]) / (1.0 + abs(n0[k])) for k in keys)
        check(f't15.{ds}.{arm}.reported_numbers_invariant', delta < 2e-4,
              f'{len(keys)} numbers, worst rel Δ {delta:.1e}')
        # teeth 1: the RAW same-word split of one pair moves under the gauge
        def raw_same(m, u=3, i=7):
            ub, ib = m.user_feature_extractor.base, m.item_feature_extractor.base
            TU, TI = ub.embedding_layer.weight.detach().double(), ib.embedding_layer.weight.detach().double()
            up, ip, _ = R.proto_layers(m)
            a_u = R.cosine_affine(up)[0]; a_t = R.cosine_affine(ip)[0]
            Pu, Pt = R._unit(up.prototypes.detach().double()), R._unit(ip.prototypes.detach().double())
            with torch.no_grad():
                qu, qi = ub(torch.tensor([u])).double()[0], ib(torch.tensor([i])).double()[0]
            M = a_u * Pu.T @ m.item_feature_extractor.linear_layer.weight.detach().double() / qu.norm() \
                + a_t * m.user_feature_extractor.linear_layer.weight.detach().double().T @ Pt / qi.norm()
            cu, wu = ub.user_slots(u)
            ci = ib.feature_ids[i]
            wi = ib.feature_weights[i] if getattr(ib, 'feature_weights', None) is not None else torch.ones(len(ci))
            return sum(float(a) * float(b) * float(TU[int(x)] @ M @ TI[int(y)])
                       for x, a in zip(cu, wu) for y, b in zip(ci, wi) if int(x) == int(y) and float(b) != 0)
        mv = abs(raw_same(model) - raw_same(gauged))
        check(f't15.{ds}.{arm}.teeth.raw_same_word_moves', mv > 1e-3, f'raw same-word Δ {mv:.3f}')
        if layout == 'bags':
            E0, E1 = r0['E_item_words_vs_id']['q_raw'], r1['E_item_words_vs_id']['q_raw']
            mvE = abs(E0['words'] - E1['words'])
            check(f't15.{ds}.{arm}.teeth.raw_variance_split_moves_on_bags', mvE > 1e-3,
                  f'raw words share {E0["words"]:.3f} → {E1["words"]:.3f}')

# ---- 4: C and D against brute force ----
for (ds, arm), r in READ.items():
    if arm == 'ui':
        check(f't15.{ds}.ui.host_runs_A_only', set(r) >= {'A_common_mode'} and 'B_word_pairs' not in r)
        continue
    split, n_users, n_items, fields, layout = SPLITS[ds]
    model, _s = MODELS[(ds, arm)]
    ib = model.item_feature_extractor.base
    words = []
    for i in range(n_items):
        c = ib.feature_ids[i]
        w = ib.feature_weights[i] if getattr(ib, 'feature_weights', None) is not None else torch.ones(len(c))
        words.append(frozenset(int(x) for x, y in zip(c, w) if float(y) != 0))
    train = pd.read_csv(os.path.join(split, 'listening_history_train.csv')).drop_duplicates()
    baskets = train.groupby('user_id')['item_id'].apply(list).to_dict()
    hits = [any(j != i and words[j] == words[i] for j in baskets[u]) for u, i in zip(train.user_id, train.item_id)]
    cat = np.mean([sum(words[j] == words[i] for j in range(n_items)) >= 2 for i in range(n_items)])
    C = r['C_basket_twins']
    check(f't15.{ds}.{arm}.C.twins_equal_brute_force',
          abs(C['frac_train_positives_with_basket_twin'] - np.mean(hits)) < 1e-12
          and abs(C['frac_catalogue_items_with_twin'] - cat) < 1e-12,
          f"train {C['frac_train_positives_with_basket_twin']:.3f} catalogue {C['frac_catalogue_items_with_twin']:.3f}")
    up, ip, _ = R.proto_layers(model)
    Pu, Pt = up.prototypes.detach().double(), ip.prototypes.detach().double()
    cos = torch.nn.functional.cosine_similarity(Pu[:, None, :], Pt[None, :, :], dim=-1)
    Dd = r['D_cross_prototype_cosine']
    check(f't15.{ds}.{arm}.D.rowmax_equals_brute_force',
          abs(Dd['user_proto_rowmax_vs_item_protos']['p50'] - float(np.percentile(cos.max(1).values.numpy(), 50))) < 1e-9
          and abs(Dd['item_proto_rowmax_vs_user_protos']['p50'] - float(np.percentile(cos.max(0).values.numpy(), 50))) < 1e-9)
    E = r['E_item_words_vs_id']
    if ARMS[arm][0]:
        ok = all(abs(sum(E[k].values()) - 1.0) < 1e-9 for k in ('q', 'q_raw', 't_hat', 'B_item'))
        check(f't15.{ds}.{arm}.E.splits_sum_to_one', ok)
        if layout == 'fixed':
            check(f't15.{ds}.{arm}.E.fixed_layout_raw_equals_convention_free',
                  abs(E['q']['words'] - E['q_raw']['words']) < 1e-9, f"{E['q']['words']:.4f}")
    else:
        check(f't15.{ds}.{arm}.E.noid_does_not_apply', E['applies'] is False)

# ---- 5: CLI end to end on a saved toy results dir ----
split, n_users, n_items, fields, layout = SPLITS['toy_bags']
model, spec = MODELS[('toy_bags', 'ids')]
rd = os.path.join(tmp, 'results_fufi_toy_bags')
os.makedirs(rd, exist_ok=True)
torch.save(model.state_dict(), os.path.join(rd, 'best_model.pth'))
with open(os.path.join(rd, 'config.json'), 'w') as f:
    json.dump({'rec_sys_param': {'use_bias': 0}, 'loss_func_name': 'sampled_softmax',
               'loss_func_aggr': 'mean', 'ft_ext_param': spec}, f, indent=2, default=str)
proc = subprocess.run([sys.executable, os.path.join(REPO, 'Master', 'scripts', 'sc8_fufi_readout.py'),
                       '--results-dir', rd, '--dataset', split, '--sample-users', '8'],
                      capture_output=True, text=True, env={**os.environ, 'PROTOMF_DATA_PATH': tmp})
out = os.path.join(rd, 'sc8_fufi_readout')
ok = proc.returncode == 0 and os.path.exists(os.path.join(out, 'readout.md'))
if ok:
    md = open(os.path.join(out, 'readout.md')).read()
    ok = all(s in md for s in ('## A.', '## B.', '## C.', '## D.', '## E.', 'NOT fixed by the scores'))
check('t15.cli_writes_json_and_md', ok, proc.stderr[-400:] if not ok else out)

print('\nt15', 'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL: {FAILS}')
sys.exit(1 if FAILS else 0)
