# cosbias2x2 t01 — cosine-offset × bias generalization of the breakdown renderer
# (2026-08-17 special experiments; utilities/explanations/breakdown.py).
#
# The trap this suite exists for: the arithmetic self-check is c-INVARIANT —
# t·(u*−c) + c·1ᵀt ≡ t·u* for ANY offset c — so "parts sum to the forward score" can NEVER
# validate the split. Every assertion below therefore pins the SPLIT SEMANTICS against
# constants defined HERE (independent of the code under test):
#   act = a·cos + c with shifted (1,1) | standard (1,0) | shifted_and_div (0.5,0.5).
#
# Coverage, per {shifted, standard, shifted_and_div} × {bias off, on}:
#   A. item_proto host   — (a) parts sum to forward score, (b) baseline == c·Σ_k u_k exactly
#      (c-table defined here), (c) baseline line ABSENT (None + no formula in md) when c=0,
#      (d) bias lines present exactly iff use_bias, values == model's own b_u/b_i/b_g,
#      (e) bars == u_k·(t*_k − c) with t* from the model's own extractor,
#      (f) md wording: coef '0.5·' under shifted_and_div; rank-relevance bias labels.
#   B. user_proto host   — same battery with baseline == c·1ᵀt (item_intercept side).
#   C. feature_item_proto (dc01) — offset semantics + zoom a-scale: zoom rows sum to the top
#      bar (catches the missing a under shifted_and_div), baseline value, c=0 absence, bias.
#   D. feature_user_proto (dc05) — same, both zoom regroupings self-check internally
#      (compute raises on any exactness failure), baseline == c·1ᵀt, c=0 absence, bias.
#   E. cosine_affine guardrail — unknown cosine_type refuses loudly.
#   F. standalone entry (render_for_results_dir) routes a SPECIAL-EXPERIMENT ARM KEY
#      ('user_proto_cosstd_bias') to its host slot via base_model_type normalization,
#      keeps the full arm name in the artifact, and renders the c=0 + bias semantics.
#
# Run: python Master/temp/dc_checks/cosbias2x2/t01_breakdown_cosine_offset.py
import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', '..', '..')))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), *rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s0h = _load('s0_harness', ['..', 's0', '_harness.py'])
_dc01h = _load('dc01_harness', ['..', 'dc01', '_harness.py'])
_dc05h = _load('dc05_harness', ['..', 'dc05', '_harness.py'])

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from feature_extraction.feature_ids import build_feature_ids, inject_feature_ids  # noqa: E402
from utilities.explanations.breakdown import (  # noqa: E402
    compute_breakdown_feature_item_proto, compute_breakdown_feature_user_proto,
    compute_breakdown_item_proto, compute_breakdown_user_proto, cosine_affine,
    render_breakdown_text, write_breakdown)

check, FAILS = _s0h.make_check()

# affine table defined INDEPENDENTLY of the code under test (the whole point)
AFFINE = {'shifted': (1.0, 1.0), 'standard': (1.0, 0.0), 'shifted_and_div': (0.5, 0.5)}
COSINE_TYPES = list(AFFINE.keys())

tmp = tempfile.mkdtemp(prefix='t01_cosbias2x2_')
split = os.path.join(tmp, 'toy')
N_USERS, N_ITEMS = _s0h.build_toy_canonical(split, n_users=60, n_items=120)
items_info = pd.read_csv(os.path.join(split, 'item_features.csv'))
FIELDS = ['department_name', 'product_type_name', 'section_name',
          'colour_group_name', 'graphical_appearance_name']
D, K = 12, 6
UID, IID = 7, 42
TOL = 5e-4


def randomize_biases(model, seed):
    """use_bias models: init_parameters zeroes the biases — give them non-trivial values so
    the sum/label checks are not vacuous."""
    if not model.use_bias:
        return
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.user_bias.weight.normal_(generator=g)
        model.item_bias.weight.normal_(generator=g)
        model.global_bias.fill_(0.37)


def forward_score(model):
    with torch.no_grad():
        return float(model(torch.tensor([UID]), torch.tensor([[IID]])).squeeze())


def model_bias_terms(model):
    with torch.no_grad():
        return (float(model.user_bias(torch.tensor([UID])).squeeze()),
                float(model.item_bias(torch.tensor([IID])).squeeze()),
                float(model.global_bias.squeeze()))


def check_common(tag, bd, model, ct, use_bias, expected_baseline, n_protos, side_var):
    """The shared battery: sum-to-score, baseline VALUE, c=0 absence, bias lines/wording."""
    a, c = AFFINE[ct]
    S = forward_score(model)
    parts = (bd.baseline or 0.0) + sum(l.value for l in bd.lines)
    check(f'{tag}: parts sum to forward score',
          abs(parts - S) < TOL * max(1, abs(S)), f'{parts:.6f} vs {S:.6f}')
    if c == 0.0:
        check(f'{tag}: baseline ABSENT when c=0', bd.baseline is None)
    else:
        check(f'{tag}: baseline == c·1ᵀ{side_var} exactly',
              bd.baseline is not None
              and abs(bd.baseline - expected_baseline) < TOL * max(1, abs(expected_baseline)),
              f'{bd.baseline} vs expected {expected_baseline:.6f}')
    md = render_breakdown_text(bd)
    if c == 0.0:
        check(f'{tag}: md carries NO baseline formula',
              '1ᵀt = ' not in md and 'Σ_k u_k = ' not in md and 'baseline' not in
              md.split('## Method notes')[0].replace('no baseline', ''))
    elif c == 0.5:
        check(f'{tag}: md baseline formula carries the 0.5 coefficient',
              ('0.5·1ᵀt' in md) or ('0.5·Σ_k u_k' in md))
    else:
        check(f'{tag}: md baseline formula is the historical c=1 wording',
              ('B(t) = 1ᵀt' in md) or ('Σ_k u_k' in md))
    # bias lines present exactly when use_bias, with the rank-relevance split
    bias_lines = [l for l in bd.lines if 'bias' in l.label]
    if use_bias:
        u_b, i_b, g_b = model_bias_terms(model)
        check(f'{tag}: three bias lines, values == model biases',
              len(bias_lines) == 3
              and abs(bias_lines[0].value - u_b) < 1e-6
              and abs(bias_lines[1].value - i_b) < 1e-6
              and abs(bias_lines[2].value - g_b) < 1e-6,
              f'{[(l.label, round(l.value, 4)) for l in bias_lines]}')
        check(f'{tag}: bias wording splits rank-relevance',
              'item bias b_i (popularity dial — rank-relevant)' in md
              and 'user bias b_u (rank-inert)' in md
              and 'bias terms' in md
              and 'rank-inert, identical for every item' in md)
        check(f'{tag}: bias_total == b_u + b_i + b_g',
              bd.bias_total is not None
              and abs(bd.bias_total - (u_b + i_b + g_b)) < 1e-6)
    else:
        check(f'{tag}: no bias lines without use_bias',
              len(bias_lines) == 0 and bd.bias_total is None and 'b_i' not in md)
    # prototype bars only (bias excluded) for the caller's semantics check
    return [l for l in bd.lines if 'bias' not in l.label][:n_protos]


# ---------------- A. item_proto host ----------------
for ct in COSINE_TYPES:
    for use_bias in (0, 1):
        tag = f'A item_proto/{ct}/bias={use_bias}'
        torch.manual_seed(11)
        param = _dc01h.item_proto_param(D, K)
        param['item_ft_ext_param']['cosine_type'] = ct
        model = _dc01h.build_recsys(param, N_USERS, N_ITEMS, use_bias=use_bias)
        randomize_biases(model, 100 + use_bias)
        model.eval()
        a, c = AFFINE[ct]
        with torch.no_grad():
            u = model.user_feature_extractor(torch.tensor([UID])).squeeze(0)
            tstar = model.item_feature_extractor(torch.tensor([IID])).squeeze(0)
        model.item_feature_extractor.get_and_reset_loss()
        bd = compute_breakdown_item_proto(model, UID, IID, items_info)
        bars = check_common(tag, bd, model, ct, use_bias,
                            expected_baseline=c * float(u.sum()), n_protos=K, side_var='u')
        expected = (u * (tstar - c)).numpy()
        check(f'{tag}: bars == u_k·(t*_k − c) (split semantics, c from the test table)',
              max(abs(bars[k].value - float(expected[k])) for k in range(K)) < TOL)
        check(f'{tag}: exemplar zoom unit label matches the sim function',
              bd.zoom_value_label == {'shifted': '1+cos', 'standard': 'cos',
                                      'shifted_and_div': '(1+cos)/2'}[ct])

# ---------------- B. user_proto host ----------------
for ct in COSINE_TYPES:
    for use_bias in (0, 1):
        tag = f'B user_proto/{ct}/bias={use_bias}'
        torch.manual_seed(22)
        param = _dc05h.user_proto_param(D, K)
        param['user_ft_ext_param']['cosine_type'] = ct
        model = _dc05h.build_recsys(param, N_USERS, N_ITEMS, use_bias=use_bias)
        randomize_biases(model, 200 + use_bias)
        model.eval()
        a, c = AFFINE[ct]
        with torch.no_grad():
            ustar = model.user_feature_extractor(torch.tensor([UID])).squeeze(0)
            t = model.item_feature_extractor(torch.tensor([IID])).squeeze(0)
        model.user_feature_extractor.get_and_reset_loss()
        bd = compute_breakdown_user_proto(model, UID, IID, items_info, split)
        bars = check_common(tag, bd, model, ct, use_bias,
                            expected_baseline=c * float(t.sum()), n_protos=K, side_var='t')
        expected = (t * (ustar - c)).numpy()
        check(f'{tag}: bars == t_l·(u*_l − c) (split semantics, c from the test table)',
              max(abs(bars[l].value - float(expected[l])) for l in range(K)) < TOL)
        check(f'{tag}: baseline_kind stays item_intercept (dc05 §3.4 wording)',
              bd.baseline_kind == 'item_intercept')

# figure + md artifacts render for the extreme combos (no baseline / half baseline + bias)
torch.manual_seed(33)
for ct in ('standard', 'shifted_and_div'):
    param = _dc05h.user_proto_param(D, K)
    param['user_ft_ext_param']['cosine_type'] = ct
    model = _dc05h.build_recsys(param, N_USERS, N_ITEMS, use_bias=1)
    randomize_biases(model, 300)
    model.eval()
    bd = compute_breakdown_user_proto(model, UID, IID, items_info, split)
    stem = write_breakdown(bd, os.path.join(tmp, 'art'), stem=f'up_{ct}_bias')
    check(f'B: figure+md written ({ct}, bias)',
          os.path.exists(stem + '.png') and os.path.exists(stem + '.md'))

# ---------------- C. feature_item_proto (dc01 candidate) ----------------
feat_ids, n_feat = build_feature_ids(split, FIELDS)
for ct in COSINE_TYPES:
    for use_bias in (0, 1):
        tag = f'C fI/{ct}/bias={use_bias}'
        torch.manual_seed(44)
        param = _dc01h.feature_item_proto_param(D, K, feat_ids, n_feat)
        param['item_ft_ext_param']['feature_fields'] = FIELDS
        param['item_ft_ext_param']['cosine_type'] = ct
        model = _dc01h.build_recsys(param, N_USERS, N_ITEMS, use_bias=use_bias)
        randomize_biases(model, 400 + use_bias)
        model.eval()
        a, c = AFFINE[ct]
        with torch.no_grad():
            u = model.user_feature_extractor(torch.tensor([UID])).squeeze(0)
        # compute self-checks internally (raise on failure), incl. the zoom a-scale
        bd = compute_breakdown_feature_item_proto(
            model, UID, IID, items_info, FIELDS,
            dataset_dir=split, feature_layout='fixed')
        check_common(tag, bd, model, ct, use_bias,
                     expected_baseline=c * float(u.sum()), n_protos=K, side_var='u')
        zoom_sum = sum(l.value for l in bd.zoom_lines)
        check(f'{tag}: zoom rows sum to the top bar (a-scale correct)',
              abs(zoom_sum - bd.zoom_parent_value) < TOL,
              f'{zoom_sum:.6f} vs {bd.zoom_parent_value:.6f}')

# ---------------- D. feature_user_proto (dc05 candidate) ----------------
for ct in COSINE_TYPES:
    for use_bias in (0, 1):
        tag = f'D fU/{ct}/bias={use_bias}'
        torch.manual_seed(55)
        fu_spec = {
            'ft_type': 'feature_user_proto', 'embedding_dim': D,
            'user_ft_ext_param': {'ft_type': 'feature_user_proto', 'sim_proto_weight': 1.0,
                                  'sim_batch_weight': 1.0, 'use_weight_matrix': False,
                                  'n_prototypes': K, 'cosine_type': ct,
                                  'reg_proto_type': 'max', 'reg_batch_type': 'max',
                                  'use_id_feature': True, 'feature_fields': FIELDS},
            'item_ft_ext_param': {'ft_type': 'embedding'},
        }
        fu_injected = inject_feature_ids(fu_spec, split)
        model = _dc05h.build_recsys(fu_injected, N_USERS, N_ITEMS, use_bias=use_bias)
        randomize_biases(model, 500 + use_bias)
        model.eval()
        a, c = AFFINE[ct]
        with torch.no_grad():
            t = model.item_feature_extractor(torch.tensor([IID])).squeeze(0)
        # both zoom regroupings self-check internally with the a-scale (raise on failure)
        bd = compute_breakdown_feature_user_proto(
            model, UID, IID, items_info, FIELDS, split, feature_layout='fixed')
        check_common(tag, bd, model, ct, use_bias,
                     expected_baseline=c * float(t.sum()), n_protos=K, side_var='t')
        zoom_sum = sum(l.value for l in bd.zoom_lines)
        alt_sum = sum(l.value for l in bd.alt_zoom_lines)
        check(f'{tag}: per-purchase AND per-word zooms both sum to the top bar',
              abs(zoom_sum - bd.zoom_parent_value) < TOL
              and abs(alt_sum - bd.zoom_parent_value) < TOL,
              f'{zoom_sum:.6f} / {alt_sum:.6f} vs {bd.zoom_parent_value:.6f}')

# ---------------- E. cosine_affine guardrail ----------------
class _FakeFE:
    cosine_type = 'not_a_cosine'


try:
    cosine_affine(_FakeFE())
    check('E: unknown cosine_type refuses loudly', False)
except ValueError:
    check('E: unknown cosine_type refuses loudly', True)
check('E: missing attribute falls back to the factory default (shifted)',
      cosine_affine(object()) == (1.0, 1.0, 'shifted'))

# ---------------- F. standalone entry × special-experiment arm key ----------------
import json  # noqa: E402

from utilities.explanations.breakdown import render_for_results_dir  # noqa: E402

torch.manual_seed(66)
arm_param = _dc05h.user_proto_param(D, K)
arm_param['user_ft_ext_param']['cosine_type'] = 'standard'
arm_model = _dc05h.build_recsys(arm_param, N_USERS, N_ITEMS, use_bias=1)
randomize_biases(arm_model, 600)
arm_model.eval()
arm_rd = os.path.join(tmp, 'results_user_proto_cosstd_bias')
os.makedirs(arm_rd, exist_ok=True)
torch.save(arm_model.state_dict(), os.path.join(arm_rd, 'best_model.pth'))
with open(os.path.join(arm_rd, 'config.json'), 'w') as f:
    json.dump({'rec_sys_param': {'use_bias': 1}, 'loss_func_name': 'bce',
               'loss_func_aggr': 'mean', 'ft_ext_param': arm_param}, f, default=str)
with open(os.path.join(arm_rd, 'metadata.json'), 'w') as f:
    json.dump({'model': 'user_proto_cosstd_bias', 'dataset': 'toy'}, f)
try:
    stem = render_for_results_dir(arm_rd, UID, item_id=IID, data_dir=split)
    md = open(stem + '.md').read()
    check('F: arm key reaches the user_proto slot (no NotImplementedError)', True)
    check('F: artifact pair written',
          os.path.exists(stem + '.png') and os.path.exists(stem + '.md'))
    check('F: label keeps the full arm name', 'user_proto_cosstd_bias' in md)
    check('F: c=0 semantics — no baseline formula in the arm artifact',
          '1ᵀt = ' not in md and 'Σ_k u_k = ' not in md)
    check('F: bias lines rendered in the arm artifact',
          'item bias b_i (popularity dial — rank-relevant)' in md)
except NotImplementedError as e:
    check('F: arm key reaches the user_proto slot (no NotImplementedError)', False, repr(e))

print(f"\n{'ALL PASS' if not FAILS else f'FAILURES: {FAILS}'} ({len(FAILS)} failed)")
sys.exit(1 if FAILS else 0)
