# dc05 t12 — ml-1m READ-OUT chain (SC.4, F-DC05-13). t11 pinned the second testbed's
# TRAINING chain; this pins the explanation surface on the bags layout:
#   1. code→label single source of truth: build_code_to_field_value == the builders' vocab,
#      verified against an independent raw-CSV recompute AND against n_features (1,061);
#      fixed-layout back-compat: the legacy items_info reconstruction is byte-equal on
#      one-value-per-field data.
#   2. intrinsic naming no longer fires the vocab-size guard on ml-1m (F-DC05-13's loud
#      failure) — names produced for every prototype, descriptors from the token vocab.
#   3. the fU breakdown renders END-TO-END on the real ml-1m split (bags per-purchase
#      regrouping; the artifact's internal exactness self-checks are the deep verification),
#      and every word-zoom label belongs to the builder-owned label set.
#   4. per_purchase_rows (bags) regroups exactly: rows sum to q_u.
#   5. item_feature_rows / item_meta_codes on a bags FeatureEmbedding (the fI/lightfm side):
#      padding filtered, rows sum to the forward exactly, codes align with real slots.
# Skips (with a loud line) if data/ml-1m is not staged locally (gitignored artifacts).
# Run: python Master/temp/dc_checks/dc05/t12_ml1m_readout_chain.py
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import torch

from _harness import make_check, build_recsys, feature_user_proto_param

from feature_extraction.feature_extractors import FeatureEmbedding
from feature_extraction.feature_ids import (build_code_to_field_value, build_feature_bags,
                                            build_user_history_weights,
                                            resolve_canonical_fields)

check, FAILS = make_check()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
ML = os.path.join(REPO_ROOT, 'data', 'ml-1m')

if not os.path.exists(os.path.join(ML, 'item_features.csv')):
    print("[SKIP] data/ml-1m not staged locally (item_features.csv missing) — t12 needs the "
          "real split; nothing checked")
    sys.exit(0)

fields, layout = resolve_canonical_fields('ml-1m')
assert fields == ['genres', 'tags'] and layout == 'bags'

# --- 1. code→label single source of truth ------------------------------------------------
code_to_fv = build_code_to_field_value(ML, fields, layout)
_, _, nf = build_feature_bags(ML, fields)
check("t12.label_count_equals_n_features", len(code_to_fv) == nf == 1061,
      f"labels={len(code_to_fv)} nf={nf}")

# independent raw-CSV recompute of the token vocab (separate pandas path)
feats = pd.read_csv(os.path.join(ML, 'item_features.csv'), keep_default_na=False, dtype=str)
expected = []
for fld in fields:
    toks = set()
    for cell in feats[fld]:
        if cell != '':
            toks.update(cell.split('|'))
    expected.extend((fld, t) for t in sorted(toks))
check("t12.labels_match_independent_recompute", code_to_fv == expected,
      f"first mismatch at {next((i for i, (a, b) in enumerate(zip(code_to_fv, expected)) if a != b), None)}")

# fixed-layout back-compat: legacy items_info recipe == builder-owned helper on H&M-style data
from utilities.explanations.breakdown import feature_code_labels
toy_dir = tempfile.mkdtemp(prefix='t12_fixed_')
try:
    pd.DataFrame({'item_id': [0, 1, 2],
                  'colour_group_name': ['Red', 'Blue', 'Red'],
                  'section_name': ['A', 'B', 'B']}).to_csv(
        os.path.join(toy_dir, 'item_features.csv'), index=False)
    toy_info = pd.read_csv(os.path.join(toy_dir, 'item_features.csv'))
    legacy = feature_code_labels(['colour_group_name', 'section_name'], toy_info)
    owned = feature_code_labels(['colour_group_name', 'section_name'], toy_info,
                                dataset_dir=toy_dir, feature_layout='fixed')
    check("t12.fixed_layout_backcompat", legacy == owned, f"{legacy} vs {owned}")
    # bags layout without dataset_dir must refuse loudly, never mislabel
    try:
        feature_code_labels(['colour_group_name'], toy_info, feature_layout='bags')
        check("t12.bags_without_dir_refuses", False, "no raise")
    except ValueError as e:
        check("t12.bags_without_dir_refuses", 'F-DC05-13' in str(e))
finally:
    shutil.rmtree(toy_dir, ignore_errors=True)

# --- 2. intrinsic naming on ml-1m (the former loud failure) ------------------------------
from utilities.explanations.naming import (get_naming_config, intrinsic_naming_config,
                                           name_prototypes_intrinsic)

hist_ids, hist_w, nf_u = build_user_history_weights(ML, fields, layout=layout)
n_users = hist_ids.shape[0]
n_items = len(feats)
D, KU = 8, 4
torch.manual_seed(7)
param = feature_user_proto_param(D, KU, hist_ids, hist_w, nf_u, use_id_feature=True,
                                 feature_fields=fields)
rec = build_recsys(param, n_users, n_items)
hist_embed = rec.user_feature_extractor.embedding_ext
e_f = hist_embed.embedding_layer.weight[:nf_u].detach().numpy()
protos = rec.user_feature_extractor.prototypes.detach().numpy()

base_cfg = get_naming_config('ml-1m', feats)
cfg = intrinsic_naming_config(base_cfg, fields, min_score=-1.0)  # force descriptors
naming = name_prototypes_intrinsic(e_f, protos, fields, feats, cfg, side='user',
                                   code_to_field_value=code_to_fv)
check("t12.intrinsic_naming_runs_on_ml1m", len(naming.names) == KU,
      f"{len(naming.names)} names")
fv_set = set(code_to_fv)
desc_ok = all((s.column, s.value) in fv_set
              for p in naming.profiles for s in p.stats)
check("t12.naming_descriptors_from_token_vocab", desc_ok)

# --- 3. fU breakdown end-to-end on the real split (bags) ---------------------------------
from utilities.explanations.breakdown import (compute_breakdown_feature_user_proto,
                                              write_breakdown)

pairs = pd.read_csv(os.path.join(ML, 'listening_history_train.csv'),
                    usecols=['user_id', 'item_id'])
pairs = pairs.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
uid = int(pairs.groupby('user_id').size().idxmin())   # smallest basket → fast, still real
iid = int(pairs.loc[pairs['user_id'] == uid, 'item_id'].iloc[0])

bd = compute_breakdown_feature_user_proto(rec, uid, iid, feats, fields, ML,
                                          naming_user=naming, feature_layout='bags')
check("t12.fu_breakdown_computes_on_ml1m", bd is not None and len(bd.lines) >= KU)
label_set = set(feature_code_labels(fields, feats, dataset_dir=ML, feature_layout='bags'))
word_zoom_ok = all(l.label in label_set for l in bd.alt_zoom_lines if not l.is_id)
check("t12.word_zoom_labels_from_builder_vocab", word_zoom_ok,
      f"{[l.label for l in bd.alt_zoom_lines][:3]}...")

out_dir = tempfile.mkdtemp(prefix='t12_bd_')
try:
    path = write_breakdown(bd, out_dir)
    check("t12.fu_breakdown_written", os.path.exists(path + '.png')
          and os.path.exists(path + '.md'))
finally:
    shutil.rmtree(out_dir, ignore_errors=True)

# --- 4. per_purchase_rows (bags) regroups exactly: rows sum to q_u -----------------------
from utilities.explanations.history_readout import per_purchase_rows, user_train_items

purchases = user_train_items(ML, uid)
item_bag_ids, item_bag_w, _ = build_feature_bags(ML, fields)
rows, has_id = per_purchase_rows(hist_embed, uid, purchases, item_bag_ids,
                                 feature_weights=item_bag_w)
q_u = hist_embed(torch.tensor([uid])).squeeze(0)
delta = float((rows.sum(dim=0) - q_u).abs().max())
check("t12.per_purchase_rows_sum_to_qu_bags", delta < 1e-4, f"maxΔ={delta:.2e}")
check("t12.per_purchase_row_count", rows.shape[0] == len(purchases) + int(has_id))

# --- 5. fI/lightfm side: bags-aware item_feature_rows / item_meta_codes ------------------
from utilities.explanations.feature_readout import item_feature_rows, item_meta_codes

for use_id in (True, False):
    fe = FeatureEmbedding(n_items, item_bag_ids, nf, D, use_id_feature=use_id,
                          feature_weights=item_bag_w)
    fe.init_parameters()
    for i in (0, 42, n_items - 1):
        r = item_feature_rows(fe, i)
        q_i = fe(torch.tensor([i])).squeeze(0)
        d_i = float((r.sum(dim=0) - q_i).abs().max())
        codes = item_meta_codes(fe, i)
        n_real = int((item_bag_w[i] > 0).sum())
        ok = (d_i < 1e-5 and len(codes) == n_real
              and r.shape[0] == n_real + int(use_id))
        check(f"t12.item_rows_bags_id{int(use_id)}_item{i}", ok,
              f"maxΔ={d_i:.2e} codes={len(codes)}/{n_real}")

print()
print("t12: ALL PASS" if not FAILS else f"t12: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
