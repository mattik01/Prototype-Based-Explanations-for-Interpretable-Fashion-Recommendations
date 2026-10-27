# dc09 SC.3 stage D — one-off descriptive measurement of the cold ID-row convention (F-DC09-08)
# on trained hm_1_month_cold checkpoints with item ID rows (dc01 fI-ids, S0.4 lightfm_tags_ids).
# Not a check with a pass bar except the integrity rows; writes ONLY to --out (never into the
# checkpoint's results dir — run_cold_eval would overwrite the fleet's cold_eval/report.json).
#
# The translation gauge (words of field φ += v_φ, every item ID row −= Σ_φ v_φ) leaves every
# score of the trained model unchanged; the cold convention is applied AFTER it. Three gauges of
# the same trained model:
#   trained  — the checkpoint as saved (the gauge the optimiser happened to land in);
#   centred  — every field's word rows centred on their (unweighted) vocabulary mean, the means
#              moved into the ID rows (the canonical per-field-centred view of F-DC09-07);
#   random   — a random v_φ per field (scale 3 × the word-row std, seed 4).
# Per gauge × convention ∈ {drop, impute_mean}: cold-vs-cold and cold-vs-all HR@10 / NDCG@10
# (+ tie-bracket expectation), exactly the fleet runner's datasets and seeds.
# Integrity: (i) gauge-shifted models score warm test rows identically (sampled logits);
# (ii) trained × drop reproduces the fleet report's numbers.
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from feature_extraction.feature_ids import build_field_index  # noqa: E402
from rec_sys.protomf_dataset import ColdTestDataset  # noqa: E402
from utilities.cold_eval import (_find_feature_embedding, apply_cold_id_convention,  # noqa: E402
                                 build_model, evaluate_rows, load_checkpoint, load_config,
                                 tie_bracket)
from utilities.consts import SINGLE_SEED  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--results-dir', required=True)
ap.add_argument('--variant-path', default='/scratch/c7031336/protomf_data/hm_1_month_cold')
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--n-neg', type=int, default=99)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
name = os.path.basename(args.results_dir.rstrip('/'))
conf = load_config(args.results_dir)
ip = conf['ft_ext_param']['item_ft_ext_param']
fields, layout = ip['feature_fields'], ip.get('feature_layout', 'fixed')
assert layout == 'fixed', 'the centred gauge below assumes one word per field per item'
field_of = build_field_index(args.variant_path, fields, layout)  # (V,) code -> field position
cold = np.sort(pd.read_csv(os.path.join(args.variant_path, 'cold_items.csv'))['item_id'].to_numpy())


def fresh():
    model, n_users, n_items = build_model(conf, args.variant_path, args.device)
    load_checkpoint(model, args.results_dir, args.device)
    return model


def table(model):
    fe = _find_feature_embedding(model.item_feature_extractor)
    return fe, fe.embedding_layer.weight.data


def gauge_vectors(model, gauge):
    """v_φ per field (F, d) for the requested gauge, from the TRAINED weights."""
    fe, W = table(model)
    V = fe.n_features
    words = W[:V].double()
    F = len(fields)
    if gauge == 'trained':
        return torch.zeros((F, W.shape[1]), dtype=torch.float64, device=W.device)
    if gauge == 'centred':
        return -torch.stack([words[field_of.to(W.device) == j].mean(0) for j in range(F)])
    g = torch.Generator().manual_seed(4)
    return (torch.randn((F, W.shape[1]), generator=g, dtype=torch.float64) * 3 * words.std().item()).to(W.device)


def shift(model, v):
    fe, W = table(model)
    V = fe.n_features
    W[:V] += v[field_of.to(W.device)].to(W.dtype)
    W[V:V + fe.n_objects] -= v.sum(0).to(W.dtype)


@torch.no_grad()
def sampled_logits(model, users, items):
    return torch.cat([model(users[s:s + 2048], items[s:s + 2048]).float().cpu()
                      for s in range(0, len(users), 2048)])


def norms(model):
    fe, W = table(model)
    V, M = fe.n_features, fe.n_objects
    ids = W[V:V + M].double()
    warm = np.setdiff1d(np.arange(M), cold)
    codes = fe.feature_ids if hasattr(fe, 'feature_ids') else None
    wsum = W[:V].double()[codes[torch.as_tensor(cold)].to(W.device)].sum(1) if codes is not None else None
    return {'mean_norm_warm_id_row': ids[warm].norm(dim=1).mean().item(),
            'norm_of_warm_mean_id_row': ids[warm].mean(0).norm().item(),
            'mean_norm_cold_word_sum': None if wsum is None else wsum.norm(dim=1).mean().item()}


report = {'results_dir': args.results_dir, 'variant_path': args.variant_path, 'fields': fields,
          'n_cold_items': int(len(cold)), 'rows': [], 'integrity': {}}

# integrity (i): warm test rows (positive + 99 random items, fixed draw), all three gauges
test = pd.read_csv(os.path.join(args.variant_path, 'listening_history_test.csv'), usecols=['user_id', 'item_id'])
rng = np.random.default_rng(SINGLE_SEED)
sel = rng.choice(len(test), size=min(20000, len(test)), replace=False)
base_model = fresh()
n_items = table(base_model)[0].n_objects
users = torch.as_tensor(test['user_id'].to_numpy()[sel], device=args.device)
items = torch.as_tensor(np.concatenate([test['item_id'].to_numpy()[sel, None],
                                        rng.integers(0, n_items, (len(sel), 99))], 1), device=args.device)
ref = sampled_logits(base_model, users, items)
for gauge in ('trained', 'centred', 'random'):
    m = fresh()
    v = gauge_vectors(m, gauge)
    shift(m, v)
    rel = ((sampled_logits(m, users, items) - ref).abs().max() / ref.abs().max()).item()
    report['integrity'][f'warm_logits_max_rel_diff[{gauge}]'] = rel
    report['integrity'][f'norm_shift_per_id_row[{gauge}]'] = v.sum(0).norm().item()
    report[f'norms[{gauge}]'] = norms(m)
    print(f'[{gauge}] warm logits max rel diff {rel:.1e}; |Σv| {v.sum(0).norm().item():.3f}; {report[f"norms[{gauge}]"]}')
    del m

for gauge in ('trained', 'centred', 'random'):
    for conv in ('drop', 'impute_mean'):
        m = fresh()
        shift(m, gauge_vectors(m, gauge))
        info, _ = apply_cold_id_convention(m, args.variant_path, conv)
        row = {'gauge': gauge, 'convention': conv, 'id_step': info}
        for ranking, key in (('cold', 'cold_vs_cold'), ('all', 'cold_vs_all')):
            ds = ColdTestDataset(args.variant_path, ranking=ranking, n_neg=args.n_neg, seed=SINGLE_SEED)
            agg, _, ties = evaluate_rows(m, ds, 256, args.device)
            tb = tie_bracket(*ties, k=10, point_hr=agg['hit_ratio@10'], point_ndcg=agg['ndcg@10'])
            row[key] = {'n_rows': len(ds), 'hit_ratio@10': agg['hit_ratio@10'], 'ndcg@10': agg['ndcg@10'],
                        'hr10_expected': tb['hit_ratio']['expected'], 'ndcg10_expected': tb['ndcg']['expected']}
        report['rows'].append(row)
        print(f"{gauge:8s} {conv:12s} cold-vs-cold HR@10 {row['cold_vs_cold']['hit_ratio@10']:.4f} "
              f"N@10 {row['cold_vs_cold']['ndcg@10']:.4f} | cold-vs-all HR@10 {row['cold_vs_all']['hit_ratio@10']:.4f} "
              f"N@10 {row['cold_vs_all']['ndcg@10']:.4f}")
        del m

# integrity (ii): trained × drop vs the fleet report (produced by the pre-amendment runner = drop)
fleet_path = os.path.join(args.results_dir, 'cold_eval', 'report.json')
if os.path.exists(fleet_path):
    with open(fleet_path) as f:
        fleet = json.load(f)
    mine = next(r for r in report['rows'] if r['gauge'] == 'trained' and r['convention'] == 'drop')
    report['integrity']['fleet_report_reproduced'] = {
        key: {'fleet': fleet[key]['hit_ratio@10'], 'here': mine[key]['hit_ratio@10'],
              'abs_diff': abs(fleet[key]['hit_ratio@10'] - mine[key]['hit_ratio@10'])}
        for key in ('cold_vs_cold', 'cold_vs_all')}
    print('fleet reproduced:', report['integrity']['fleet_report_reproduced'])

with open(os.path.join(args.out, f'{name}.json'), 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"written {os.path.join(args.out, f'{name}.json')}")
