"""U2a/U3 composition instrument (plan: composition_upgrades_plan_2026-08-24.md).

Post-hoc, checkpoint-only, all four expl_deep_dive fI arms: hm_1_month (fixed layout)
AND ml-1m (bags layout, genres+tags) — ml-1m always included as reference (Matteo,
2026-08-24). Measures: word-norm-vs-frequency, per-field means (weighted/unweighted),
the shared/centring component and its size, centred norms, and geometry raw vs centred
(activation-cloud entropy eff. rank per sc8 definition, omnipresent-token share of
||q||^2, item pairwise cos, prototype-profile correlation).

Layout note: on hm (one value per field) the centring shift C is EXACTLY constant
across items; on ml-1m (variable-size bags, every token weight 1.0) the shift is
item-dependent (sum of the token-wise field means) — "shared in expectation" only,
and bag size itself is a vote-mass channel (S0.5 addendum must-mention).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from feature_extraction.feature_ids import (build_feature_ids, build_feature_bags,  # noqa: E402
                                            build_code_to_field_value)

SP = '/tmp/claude-1000/-home-mattik01-Desktop-githubs-ProtoMF/d0e48ee7-b3dd-4288-a15c-b5a1003934c4/scratchpad'
OUT = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    'hm_1_month': {
        'data': f'{SP}/hm_1_month', 'layout': 'fixed',
        'fields': ['department_name', 'product_type_name', 'section_name',
                   'colour_group_name', 'graphical_appearance_name']},
    'ml-1m': {
        'data': f'{SP}/ml-1m', 'layout': 'bags', 'fields': ['genres', 'tags']},
}
ARMS = {
    'fI_hm': ('hm_1_month', 'Master/experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573'),
    'noid_hm': ('hm_1_month', 'Master/experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573'),
    'fI_ml': ('ml-1m', 'Master/experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573'),
    'noid_ml': ('ml-1m', 'Master/experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573'),
}

def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float('nan')

def eff_rank_entropy(M):
    S = torch.linalg.svdvals(M.double())
    p = (S ** 2 / (S ** 2).sum()).numpy()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))

def pairwise_cos_sample(Q, n=2000, seed=0):
    g = np.random.default_rng(seed)
    idx = g.integers(0, Q.shape[0], size=(n, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    a, b = Q[idx[:, 0]], Q[idx[:, 1]]
    c = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
    return {'mean': float(c.mean()), 'p10': float(c.quantile(.1)), 'p90': float(c.quantile(.9))}

# ---- dataset preps -------------------------------------------------------------------
preps = {}
for ds, cfg in DATASETS.items():
    if cfg['layout'] == 'fixed':
        ids, n_feat = build_feature_ids(cfg['data'], cfg['fields'])
        w = torch.ones_like(ids, dtype=torch.float32)
    else:
        ids, w, n_feat = build_feature_bags(cfg['data'], cfg['fields'])
    c2fv = build_code_to_field_value(cfg['data'], cfg['fields'], layout=cfg['layout'])
    counts = np.bincount(ids.reshape(-1).numpy(), weights=w.reshape(-1).numpy(),
                         minlength=n_feat).astype(int)
    field_of_code = np.array([cfg['fields'].index(f) for f, _ in c2fv])
    preps[ds] = dict(ids=ids, w=w, n_feat=n_feat, c2fv=c2fv, counts=counts,
                     foc=field_of_code, fields=cfg['fields'],
                     bag_sizes=w.sum(1).numpy())

results = {'_meta': {}}
for ds, p in preps.items():
    top = int(np.argmax(p['counts']))
    results['_meta'][ds] = {
        'n_items': int(p['ids'].shape[0]), 'n_features': int(p['n_feat']),
        'omni_token': {'field': p['c2fv'][top][0], 'value': p['c2fv'][top][1],
                       'count': int(p['counts'][top]),
                       'item_frac': float(p['counts'][top] / p['ids'].shape[0])},
        'bag_size': {'mean': float(p['bag_sizes'].mean()), 'p10': float(np.quantile(p['bag_sizes'], .1)),
                     'p90': float(np.quantile(p['bag_sizes'], .9))}}

# ---- per-arm analysis ----------------------------------------------------------------
for arm, (ds, path) in ARMS.items():
    p = preps[ds]
    ids, w, n_feat, counts, foc = p['ids'], p['w'], p['n_feat'], p['counts'], p['foc']
    sd = torch.load(os.path.join(REPO, path, 'best_model.pth'), map_location='cpu')
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    W = sd['item_feature_extractor.embedding_ext.embedding_layer.weight']
    P = sd['item_feature_extractor.prototypes']
    assert W.shape[0] in (n_feat, n_feat + ids.shape[0]), (arm, W.shape, n_feat)
    has_id = W.shape[0] > n_feat
    E = W[:n_feat]
    omni = int(np.argmax(counts))
    R = {'dataset': ds, 'd': int(W.shape[1]), 'K': int(P.shape[0]), 'has_id': bool(has_id)}

    norms = E.norm(dim=1).numpy()
    R['norm_vs_logfreq_spearman_overall'] = spearman(norms, np.log1p(counts))
    R['norm_vs_logfreq_spearman_per_field'] = {
        f: spearman(norms[foc == j], np.log1p(counts[foc == j]))
        for j, f in enumerate(p['fields'])}
    order = np.argsort(-norms)
    R['top10_norm'] = [{'field': p['c2fv'][i][0], 'value': p['c2fv'][i][1],
                        'norm': float(norms[i]), 'count': int(counts[i])} for i in order[:10]]
    R['omni'] = {'value': p['c2fv'][omni][1], 'norm_raw': float(norms[omni]),
                 'norm_rank': int(np.where(order == omni)[0][0]) + 1, 'count': int(counts[omni])}

    Q_meta = (E[ids] * w.unsqueeze(-1)).sum(dim=1)
    R['corr_bagsize_qmeta_norm'] = spearman(p['bag_sizes'], Q_meta.norm(dim=1).numpy())
    if has_id:
        ID = W[n_feat:]
        Q = Q_meta + ID
        r = (ID.norm(dim=1) / Q_meta.norm(dim=1).clamp_min(1e-12)).numpy()
        R['id_meta_norm_ratio_median'] = float(np.median(r))
        R['id_share_qsq_mean'] = float(
            ((ID * Q).sum(1) / (Q.norm(dim=1) ** 2).clamp_min(1e-12)).mean())
    else:
        Q = Q_meta

    for tag in ('weighted', 'unweighted'):
        mu = torch.zeros_like(E)
        for j in range(len(p['fields'])):
            codes = np.where(foc == j)[0]
            Ef = E[codes].double()
            if tag == 'weighted':
                cw = torch.tensor(counts[codes], dtype=torch.float64).clamp_min(0).unsqueeze(1)
                mu[codes] = ((Ef * cw).sum(0) / cw.sum().clamp_min(1e-12)).float()
            else:
                mu[codes] = Ef.mean(0).float()
        shift = (mu[ids] * w.unsqueeze(-1)).sum(dim=1)          # per-item centring shift
        Ec = E - mu
        Qc = Q - shift
        omni_items = ((ids == omni) & (w > 0)).any(dim=1)
        Es, Ecs = E[omni], Ec[omni]
        A_raw = 1 + (Q @ P.T) / (Q.norm(dim=1, keepdim=True) * P.norm(dim=1)).clamp_min(1e-12)
        A_c = 1 + (Qc @ P.T) / (Qc.norm(dim=1, keepdim=True) * P.norm(dim=1)).clamp_min(1e-12)
        prof_raw = (E @ P.T) / (E.norm(dim=1, keepdim=True) * P.norm(dim=1)).clamp_min(1e-12)
        prof_c = (Ec @ P.T) / (Ec.norm(dim=1, keepdim=True) * P.norm(dim=1)).clamp_min(1e-12)
        pc_raw = np.corrcoef(prof_raw.T.numpy())
        pc_c = np.corrcoef(prof_c.T.numpy())
        iu = np.triu_indices(P.shape[0], 1)
        R[f'centre_{tag}'] = {
            'shift_norm_mean': float(shift.norm(dim=1).mean()),
            'shift_norm_std': float(shift.norm(dim=1).std()),   # 0 on hm (exact C), >0 on ml bags
            'mean_q_meta_norm': float(Q_meta.norm(dim=1).mean()),
            'shift_share_of_mean_q_meta': float(shift.norm(dim=1).mean() / Q_meta.norm(dim=1).mean()),
            'word_norm_median_raw': float(np.median(norms)),
            'word_norm_median_centred': float(np.median(Ec.norm(dim=1).numpy())),
            'omni_norm_centred': float(Ecs.norm()),
            'norm_vs_logfreq_spearman_centred': spearman(Ec.norm(dim=1).numpy(), np.log1p(counts)),
            'eff_rank_raw': eff_rank_entropy(A_raw),
            'eff_rank_centred': eff_rank_entropy(A_c),
            'item_cos_raw': pairwise_cos_sample(Q),
            'item_cos_centred': pairwise_cos_sample(Qc),
            'proto_profile_corr_mean_raw': float(pc_raw[iu].mean()),
            'proto_profile_corr_mean_centred': float(pc_c[iu].mean()),
            'omni_share_qsq_mean_raw': float(((Q[omni_items] @ Es) /
                (Q[omni_items].norm(dim=1) ** 2).clamp_min(1e-12)).mean()),
            'omni_share_qsq_mean_centred': float(((Qc[omni_items] @ Ecs) /
                (Qc[omni_items].norm(dim=1) ** 2).clamp_min(1e-12)).mean()),
        }
    results[arm] = R

with open(os.path.join(OUT, 'results.json'), 'w') as fh:
    json.dump(results, fh, indent=2)
print(json.dumps({k: v for k, v in results.items() if k in ('_meta', 'fI_ml', 'noid_ml')}, indent=1))
