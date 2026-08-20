# dc05 t10 — breakdown/read-out integration on a REAL (toy) split + briefly-trained models.
# Pins, for BOTH stage models (fU and its host user_proto, like-for-like):
#   - render_for_results_dir produces the artifact pair; every internal exactness self-check
#     passes (B(t) + Σ personalized == forward score; zooms sum to their parent bar — the
#     compute functions RAISE otherwise);
#   - 4b C2′ on the real module: per-purchase shares == per-word shares regrouped (both equal
#     u*_l − 1 summed, for EVERY prototype);
#   - the explanations pipeline runs end-to-end for fU (intrinsic user naming, USER prototype
#     cards, naming CSVs, dual post-hoc route) and for the host;
#   - figures are copied to dc_checks/dc05/figures/ for the standing graphical-scrutiny pass.
# Run: python Master/temp/dc_checks/dc05/t10_breakdown_readout.py
import importlib.util
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

tmp = tempfile.mkdtemp(prefix='t10_dc05_')
DATASET = 'toy'
os.environ['PROTOMF_DATA_PATH'] = tmp          # BEFORE utilities.consts is imported
os.environ.setdefault('WANDB_MODE', 'disabled')

import torch

from _harness import (make_check, feature_user_proto_param, user_proto_param, build_recsys,
                      make_check as _mk)  # noqa: F401

# the s0 harness is ALSO named _harness — load it under a distinct module name (dc01 t11 pattern)
_spec = importlib.util.spec_from_file_location(
    's0_harness', os.path.join(os.path.dirname(__file__), '..', 's0', '_harness.py'))
_s0h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s0h)

from feature_extraction.feature_ids import inject_feature_ids
from utilities.explanations.breakdown import render_for_results_dir
from utilities.explanations.feature_readout import per_feature_shares
from utilities.explanations.history_readout import (per_purchase_rows, user_history_rows,
                                                    user_train_items)
from utilities.explanations.pipeline import run_explanations_pipeline

check, FAILS = make_check()
torch.manual_seed(0)

split = os.path.join(tmp, DATASET)
N_USERS, N_ITEMS = _s0h.build_toy_canonical(split, n_users=60, n_items=120)
FIELDS = ['department_name', 'product_type_name', 'section_name',
          'colour_group_name', 'graphical_appearance_name']
D, KU = 12, 6

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def train_briefly(model, seed=1, steps=60):
    import pandas as pd
    g = torch.Generator().manual_seed(seed)
    pairs = pd.read_csv(os.path.join(split, 'listening_history_train.csv'),
                        usecols=['user_id', 'item_id'])
    u_all = torch.as_tensor(pairs['user_id'].to_numpy(), dtype=torch.long)
    i_all = torch.as_tensor(pairs['item_id'].to_numpy(), dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(steps):
        sel = torch.randint(0, len(u_all), (32,), generator=g)
        u = u_all[sel]
        pos = i_all[sel].unsqueeze(1)
        neg = torch.randint(0, N_ITEMS, (32, 4), generator=g)
        i = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(32, 5); labels[:, 0] = 1.0
        opt.zero_grad()
        loss = model.loss_func(model(u, i), labels) \
            + model.user_feature_extractor.get_and_reset_loss() \
            + model.item_feature_extractor.get_and_reset_loss()
        loss.backward()
        opt.step()
    model.eval()
    return model


def save_results_dir(model, model_type, config_name, conf_ft):
    rd = os.path.join(tmp, f'results_{config_name}')
    os.makedirs(rd, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(rd, 'best_model.pth'))
    conf = {'rec_sys_param': {'use_bias': 0}, 'loss_func_name': 'bce',
            'loss_func_aggr': 'mean', 'ft_ext_param': conf_ft}
    with open(os.path.join(rd, 'config.json'), 'w') as f:
        json.dump(conf, f, indent=2, default=str)
    with open(os.path.join(rd, 'metadata.json'), 'w') as f:
        json.dump({'model': model_type, 'config_name': config_name, 'dataset': DATASET,
                   'seed': 0}, f, indent=2)
    return rd


# ---- fU model: build via the REAL injection seam (payload from the split's own train file) ----
fu_spec = {
    'ft_type': 'feature_user_proto', 'embedding_dim': D,
    'user_ft_ext_param': {'ft_type': 'feature_user_proto', 'sim_proto_weight': 1.0,
                          'sim_batch_weight': 1.0, 'use_weight_matrix': False,
                          'n_prototypes': KU, 'cosine_type': 'shifted',
                          'reg_proto_type': 'max', 'reg_batch_type': 'max',
                          'use_id_feature': True, 'feature_fields': FIELDS},
    'item_ft_ext_param': {'ft_type': 'embedding'},
}
fu_injected = inject_feature_ids(fu_spec, split)
check("t10.inject_leaves_caller_clean", 'hist_value_ids' not in fu_spec['user_ft_ext_param'])
fu_model = train_briefly(build_recsys(fu_injected, N_USERS, N_ITEMS))
fu_rd = save_results_dir(fu_model, 'feature_user_proto', 'feature_user_proto_toy', fu_spec)

# ---- host model ----
host_model = train_briefly(build_recsys(user_proto_param(D, KU), N_USERS, N_ITEMS), seed=2)
host_rd = save_results_dir(host_model, 'user_proto', 'user_proto_toy',
                           user_proto_param(D, KU))

USER = 7

# 1. C2′ on the real module: per-purchase == per-word regrouping, EVERY prototype
hist = fu_model.user_feature_extractor.embedding_ext
protos = fu_model.user_feature_extractor.prototypes
rows_w, _codes, _hid = user_history_rows(hist, USER)
purchases = user_train_items(split, USER)
from feature_extraction.feature_ids import build_feature_ids  # noqa: E402
fids, _nf = build_feature_ids(split, FIELDS)
rows_p, _ = per_purchase_rows(hist, USER, purchases, fids)
s_w = per_feature_shares(rows_w, protos).sum(dim=0)
s_p = per_feature_shares(rows_p, protos).sum(dim=0)
with torch.no_grad():
    ustar = fu_model.user_feature_extractor(torch.tensor([USER])).squeeze(0)
fu_model.user_feature_extractor.get_and_reset_loss()
check("t10.C2_word_vs_purchase_regroup",
      torch.allclose(s_w, s_p, atol=1e-5),
      f"max |Δ| {(s_w - s_p).abs().max():.3e}")
check("t10.shares_sum_to_activation_minus_1",
      torch.allclose(s_w, ustar - 1.0, atol=1e-4),
      f"max |Δ| {(s_w - (ustar - 1.0)).abs().max():.3e}")

# 2. fU breakdown artifact (self-checks inside raise on any exactness failure)
out_fu = render_for_results_dir(fu_rd, USER)
check("t10.fu_breakdown_written",
      os.path.exists(out_fu + '.png') and os.path.exists(out_fu + '.md'))
md = open(out_fu + '.md').read()
check("t10.fu_md_has_Bt_line", 'item baseline' in md and '1ᵀt' in md)
check("t10.fu_md_has_purchase_zoom", 'per-purchase' in md)
check("t10.fu_md_has_word_alt_zoom", 'by attribute word' in md)
check("t10.fu_md_declares_intrinsic_route", 'intrinsic' in md)
shutil.copy(out_fu + '.png', os.path.join(FIGDIR, 'breakdown_fu_toy.png'))

# 3. host breakdown artifact
out_host = render_for_results_dir(host_rd, USER)
check("t10.host_breakdown_written",
      os.path.exists(out_host + '.png') and os.path.exists(out_host + '.md'))
md_h = open(out_host + '.md').read()
check("t10.host_md_has_Bt_line", 'item baseline' in md_h and '1ᵀt' in md_h)
check("t10.host_md_declares_posthoc", 'post-hoc' in md_h.lower())
shutil.copy(out_host + '.png', os.path.join(FIGDIR, 'breakdown_host_toy.png'))

# 4. explanations pipeline end-to-end, fU (intrinsic user naming + USER cards + dual route)
out_dir = run_explanations_pipeline(fu_rd, sample_users_for_weight_viz=[USER])
check("t10.fu_pipeline_ran", out_dir is not None and os.path.isdir(out_dir))
cards = os.path.join(out_dir, 'prototype_cards_user.png')
check("t10.fu_user_cards_written", os.path.exists(cards))
if os.path.exists(cards):
    shutil.copy(cards, os.path.join(FIGDIR, 'prototype_cards_user_fu_toy.png'))
check("t10.fu_pipeline_nested_under_cosine",
      os.path.basename(out_dir) == 'cosine', f"out_dir={out_dir}")
posthoc_dir = os.path.join(fu_rd, 'explanations', 'lift')
check("t10.fu_dual_route_user_side_written",
      os.path.isdir(posthoc_dir) and any('user' in f for f in os.listdir(posthoc_dir)),
      f"{os.listdir(posthoc_dir) if os.path.isdir(posthoc_dir) else 'missing'}")

# 5. explanations pipeline end-to-end, host (post-hoc user naming + USER cards)
out_dir_h = run_explanations_pipeline(host_rd, sample_users_for_weight_viz=[USER])
check("t10.host_pipeline_ran", out_dir_h is not None and os.path.isdir(out_dir_h))
cards_h = os.path.join(out_dir_h, 'prototype_cards_user.png')
check("t10.host_user_cards_written", os.path.exists(cards_h))
if os.path.exists(cards_h):
    shutil.copy(cards_h, os.path.join(FIGDIR, 'prototype_cards_user_host_toy.png'))

# 6. F-DC05-14 (SC.4 fix): is_id lines are EXEMPT from _barh's top-N fold — the ID row's
# hatched line must survive on the figure even when its |value| would not make the cut.
from matplotlib import pyplot as plt

from utilities.explanations.breakdown import BreakdownLine, _barh

fig, ax = plt.subplots()
fold_lines = [BreakdownLine(f"word {i}", float(10 - i)) for i in range(9)]
fold_lines.append(BreakdownLine("user-ID row (own profile)", 0.001, is_id=True))
_barh(ax, fold_lines, "F-DC05-14 pin")
ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
check("t10.f_dc05_14_id_line_never_folded",
      any("user-ID row" in lbl for lbl in ytick_labels), f"labels={ytick_labels}")
plt.close(fig)

# 7. F-DC05-18 (SC.5 fix): run_combo auto-explanations include the fU headline model;
# arms stay out per the headline-only convention (source check, dc01 t10's regex style).
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
with open(os.path.join(REPO, 'Master', 'scripts', 'run_combo.py')) as f:
    rc_src = f.read()
# Convention change 2026-08-17 (commit f3a900a, user decision at the explanation deep dive):
# run_combo auto-explanations are DISABLED fleet-wide (EXPLAINABLE_MODELS = set()); the
# standalone pipeline is the live gating layer. Pins updated accordingly (dc05 SC.8, trivial
# fix — the old pins asserted the pre-f3a900a F-DC05-18 wiring).
m_rc = re.search(r"EXPLAINABLE_MODELS\s*=\s*set\(\)", rc_src)
check("t10.run_combo_auto_explanations_disabled",
      m_rc is not None, "EXPLAINABLE_MODELS = set() (f3a900a convention)")
from utilities.explanations.pipeline import EXPLAINABLE_MODELS as PIPE_SET
check("t10.pipeline_explainable_fU",
      'feature_user_proto' in PIPE_SET and 'feature_user_proto_noid' not in PIPE_SET,
      "standalone pipeline: fU in, noid excluded (headline-only convention)")

shutil.rmtree(tmp)
print(f"\nfigures for scrutiny -> {FIGDIR}")
print("\nt10: ALL PASS" if not FAILS else f"\nt10: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
