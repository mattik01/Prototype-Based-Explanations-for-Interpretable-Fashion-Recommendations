"""ID-row probe — post-hoc instrument on a trained `feature_item_proto` (fI / fI') checkpoint.

Answers three parked questions on existing checkpoints (zero training; tensors read from the
state dict, one composition pass). Working-evidence tier only — never thesis quotes.

  A. Pair-encoding regression (U4 trigger (i), composition plan 2026-08-24 §2):
     does the per-item ID row e_ID(i) systematically encode attribute-value PAIRS beyond what
     the unary value indicators explain? Multi-output ridge of e_ID (and of the ID row's
     prototype contributions) on unary indicators X1 vs unary+pair indicators X1+X2, K-fold
     CV over items, against a permutation null (X2 rows shuffled across items). Reported as
     ΔR² with the null's distribution. Pre-registered reading: HOT iff ΔR² clears the null's
     95th percentile — only then does E3 ("no conjunctions") count as OBSERVED.
  B. Popularity-stratified ID share (taxonomy row 13; vault 2026-07-11_2132 "extremely
     important"): energy-level id_share = e_ID·q/‖q‖², norm ratio ‖e_ID‖/‖q_meta‖ and the
     activation-level ID share, per train-popularity band. Plus the dc06 D2 instrument
     ‖mean e_ID‖ / mean ‖e_ID‖ and its alignment with the metadata common direction C.
  C. Bag-size channel activity (vault 2026-08-24_1229; bags layout only): Spearman of bag
     size vs train popularity / mean activation level / id_share.

Anchors reproduced for cross-checking against composition_analysis_2026-08-24/analyze.py:
id_meta_norm_ratio_median, id_share_qsq_mean, eff_rank_raw.

Output: <results_dir>/id_row_probe/readout.json + readout.md  (override with --out-dir)

Usage (LEO5 login node, protomf venv):
    python Master/scripts/id_row_probe.py \
        --results-dir /scratch/c7031336/protomf_results/feature_item_proto_hm_1_month_s38210573
    (dataset dir resolved from config.json's data_path basename against DATA_PATH;
     override with --dataset; --sections A,B,C; --device cuda speeds up the ridge SVDs)
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from feature_extraction.feature_ids import (build_code_to_field_value, build_feature_bags,
                                            build_field_index, build_item_interaction_counts)
from utilities.consts import DATA_PATH

PCTS = (1, 5, 25, 50, 75, 95, 99)
POP_BUCKETS = ((0, 0, 'unseen'), (1, 10, '1-10'), (11, 100, '11-100'), (101, 10 ** 9, '101+'))


# ── helpers ──────────────────────────────────────────────────────────────────────────────
def _stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
            **{f'p{p}': float(np.percentile(x, p)) for p in PCTS}}


def _rank(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(a.size, dtype=np.float64)
    # average ties
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def eff_rank_entropy(M: torch.Tensor) -> float:
    s = torch.linalg.svdvals(M.double())
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def load_state(results_dir):
    with open(os.path.join(results_dir, 'config.json')) as f:
        config = json.load(f)
    sd = torch.load(os.path.join(results_dir, 'best_model.pth'), map_location='cpu')
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    W = sd['item_feature_extractor.embedding_ext.embedding_layer.weight'].double()
    P = sd['item_feature_extractor.prototypes'].double()
    return config, W, P


# ── ridge machinery (section A) ───────────────────────────────────────────────────────────
def _r2_blocks(Y, Yhat, Ybar, blocks):
    out = {}
    for name, (a, b) in blocks.items():
        num = ((Y[:, a:b] - Yhat[:, a:b]) ** 2).sum()
        den = ((Y[:, a:b] - Ybar[a:b]) ** 2).sum().clamp_min(1e-12)
        out[name] = float(1 - num / den)
    return out


def ridge_cv(X, Y, folds, lambdas, blocks, device):
    """K-fold CV multi-output ridge (intercept via centring). X dense (n,p) float64 torch.
    Returns {lambda: {block: pooled held-out R²}} — pooled = sum over folds of SSE / SST."""
    n = X.shape[0]
    sse = {lam: {b: 0.0 for b in blocks} for lam in lambdas}
    sst = {b: 0.0 for b in blocks}
    for tr, te in folds:
        Xtr, Xte = X[tr].to(device), X[te].to(device)
        Ytr, Yte = Y[tr].to(device), Y[te].to(device)
        xm, ym = Xtr.mean(0), Ytr.mean(0)
        Xtr_c, Ytr_c = Xtr - xm, Ytr - ym
        U, S, Vh = torch.linalg.svd(Xtr_c, full_matrices=False)
        UtY = U.T @ Ytr_c
        for name, (a, b) in blocks.items():
            sst[name] += float(((Yte[:, a:b] - ym[a:b]) ** 2).sum())
        for lam in lambdas:
            shrink = (S / (S ** 2 + lam)).unsqueeze(1)
            beta = Vh.T @ (shrink * UtY)
            Yhat = (Xte - xm) @ beta + ym
            for name, (a, b) in blocks.items():
                sse[lam][name] += float(((Yte[:, a:b] - Yhat[:, a:b]) ** 2).sum())
        del U, S, Vh, UtY
    return {lam: {b: 1 - sse[lam][b] / max(sst[b], 1e-12) for b in blocks} for lam in lambdas}


def ridge_fit(X, Y, lam, device):
    xm, ym = X.mean(0), Y.mean(0)
    U, S, Vh = torch.linalg.svd((X - xm).to(device), full_matrices=False)
    beta = Vh.T @ ((S / (S ** 2 + lam)).unsqueeze(1) * (U.T @ (Y - ym).to(device)))
    return beta.cpu()


def build_pairs(ids, w, foc, cross_field_only, min_support, max_pairs):
    """Enumerate attribute-value pairs per item; keep pairs with support >= min_support,
    capped at max_pairs by support. Returns (pair_list, X2 dense float64 (n, n_pairs))."""
    n = ids.shape[0]
    per_item = []
    support = Counter()
    for i in range(n):
        toks = ids[i][w[i] > 0].tolist()
        toks = sorted(set(toks))
        pairs = []
        for a_i in range(len(toks)):
            for b_i in range(a_i + 1, len(toks)):
                a, b = toks[a_i], toks[b_i]
                if cross_field_only and foc[a] == foc[b]:
                    continue
                pairs.append((a, b))
        per_item.append(pairs)
        support.update(pairs)
    kept = [(p, c) for p, c in support.items() if c >= min_support]
    kept.sort(key=lambda t: (-t[1], t[0]))
    kept = kept[:max_pairs]
    col = {p: j for j, (p, _) in enumerate(kept)}
    X2 = torch.zeros(n, len(kept), dtype=torch.float64)
    for i, pairs in enumerate(per_item):
        for p in pairs:
            j = col.get(p)
            if j is not None:
                X2[i, j] = 1.0
    meta = {'n_pair_instances': int(sum(support.values())), 'n_unique_pairs': len(support),
            'n_pairs_kept': len(kept), 'min_support': min_support,
            'support_kept': _stats([c for _, c in kept]) if kept else {'n': 0}}
    return [p for p, _ in kept], X2, meta


# ── main ─────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--dataset', default=None, help='dataset dir override (abs path or name under DATA_PATH)')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--sections', default='auto', help="comma list of A,B,C or 'auto'")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-support', type=int, default=5)
    ap.add_argument('--max-pairs', type=int, default=5000)
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--n-perm', type=int, default=20)
    ap.add_argument('--lambdas', default='0.1,1,10,100,1000')
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)

    config, W, P = load_state(args.results_dir)
    spec = config['ft_ext_param']['item_ft_ext_param']
    fields, layout = list(spec['feature_fields']), spec.get('feature_layout', 'fixed')
    use_id = bool(spec.get('use_id_feature', True))
    ds_name = os.path.basename(config['data_path'].rstrip('/'))
    if args.dataset is None:
        data_dir = os.path.join(DATA_PATH, ds_name)
    elif os.path.isabs(args.dataset):
        data_dir = args.dataset
    else:
        data_dir = os.path.join(DATA_PATH, args.dataset)
    out_dir = args.out_dir or os.path.join(args.results_dir, 'id_row_probe')
    os.makedirs(out_dir, exist_ok=True)

    ids, w, n_feat = build_feature_bags(data_dir, fields)
    c2fv = build_code_to_field_value(data_dir, fields, layout=layout)
    foc = build_field_index(data_dir, fields, layout=layout).numpy()
    n_items = ids.shape[0]
    assert W.shape[0] in (n_feat, n_feat + n_items), (W.shape, n_feat, n_items)
    has_id = W.shape[0] > n_feat
    assert has_id == use_id, 'config/use_id_feature disagrees with the checkpoint table size'
    E, ID = W[:n_feat], (W[n_feat:] if has_id else None)
    K, d = P.shape[0], P.shape[1]
    counts = np.bincount(ids.reshape(-1).numpy(), weights=w.reshape(-1).numpy(),
                         minlength=n_feat).astype(int)
    bag_sizes = w.sum(1).numpy()
    pop = build_item_interaction_counts(data_dir).numpy()

    wd = w.double()
    Q_meta = (E[ids] * wd.unsqueeze(-1)).sum(dim=1)
    Q = Q_meta + ID if has_id else Q_meta
    Pn = P / P.norm(dim=1, keepdim=True).clamp_min(1e-12)
    qn = Q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    COS = (Q / qn) @ Pn.T                       # (n, K) cos(q_i, p_k)
    A = 1 + COS

    sections = ['A', 'B'] + (['C'] if layout == 'bags' else []) if args.sections == 'auto' \
        else [s.strip().upper() for s in args.sections.split(',')]
    if not has_id:
        sections = [s for s in sections if s == 'C']
    R = {'_meta': {'results_dir': args.results_dir, 'dataset': ds_name, 'data_dir': data_dir,
                   'model': config.get('_wandb_model', spec['ft_type']), 'layout': layout,
                   'fields': fields, 'n_items': n_items, 'n_features': n_feat, 'd': d, 'K': K,
                   'has_id': has_id, 'sections': sections, 'seed': args.seed,
                   'bag_size': _stats(bag_sizes), 'train_pop': _stats(pop),
                   'tier': 'working evidence (single checkpoint, post-hoc) — not thesis numbers'}}

    # anchors (cross-check with analyze.py)
    anchors = {'eff_rank_raw': eff_rank_entropy(A),
               'corr_bagsize_qmeta_norm': spearman(bag_sizes, Q_meta.norm(dim=1).numpy())}
    if has_id:
        ratio = (ID.norm(dim=1) / Q_meta.norm(dim=1).clamp_min(1e-12)).numpy()
        id_share = ((ID * Q).sum(1) / (Q.norm(dim=1) ** 2).clamp_min(1e-12)).numpy()
        anchors['id_meta_norm_ratio_median'] = float(np.median(ratio))
        anchors['id_share_qsq_mean'] = float(id_share.mean())
    R['anchors'] = anchors

    # ── Section B ────────────────────────────────────────────────────────────────────
    if 'B' in sections:
        # activation-level ID share: Σ_k c_ID(i,k) / Σ_k cos(q_i,p_k), c_ID = e_ID·p̂_k/‖q_i‖
        c_id = (ID / qn) @ Pn.T                                   # (n, K)
        act_share = (c_id.sum(1) / COS.sum(1).abs().clamp_min(1e-9)).numpy()
        act_share = np.clip(act_share, -5, 5)                     # guard degenerate rows
        mean_act = A.mean(1).numpy()

        def band_report(mask):
            return {'n': int(mask.sum()), 'id_share_qsq': _stats(id_share[mask]),
                    'id_meta_norm_ratio': _stats(ratio[mask]),
                    'id_share_activation': _stats(act_share[mask]),
                    'mean_activation': _stats(mean_act[mask])}

        B = {'overall': band_report(np.ones(n_items, bool)),
             'spearman_pop': {'id_share_qsq': spearman(pop, id_share),
                              'id_meta_norm_ratio': spearman(pop, ratio),
                              'id_share_activation': spearman(pop, act_share),
                              'id_norm': spearman(pop, ID.norm(dim=1).numpy()),
                              'qmeta_norm': spearman(pop, Q_meta.norm(dim=1).numpy())},
             'buckets': {}, 'deciles': {}}
        for lo, hi, name in POP_BUCKETS:
            m = (pop >= lo) & (pop <= hi)
            if m.any():
                B['buckets'][name] = band_report(m)
        seen = pop > 0
        if seen.sum() >= 10:
            edges = np.quantile(pop[seen], np.linspace(0, 1, 11))
            dec = np.clip(np.searchsorted(edges, pop, side='right') - 1, 0, 9)
            for q in range(10):
                m = seen & (dec == q)
                if m.any():
                    B['deciles'][f'd{q}'] = {**band_report(m),
                                             'pop_range': [float(pop[m].min()), float(pop[m].max())]}
        # D2 instrument + alignment with metadata common direction
        mean_id = ID.mean(0)
        C_vec = Q_meta.mean(0)
        B['D2'] = {'mean_id_norm_over_mean_norm': float(mean_id.norm() / ID.norm(dim=1).mean()),
                   'cos_meanID_meanQmeta': float(torch.dot(mean_id, C_vec) /
                                                 (mean_id.norm() * C_vec.norm()).clamp_min(1e-12)),
                   'id_norm': _stats(ID.norm(dim=1).numpy()),
                   'eff_rank_ID_block': eff_rank_entropy(ID - mean_id),
                   'eff_rank_ID_block_uncentred': eff_rank_entropy(ID)}
        R['B'] = B

    # ── Section C ────────────────────────────────────────────────────────────────────
    if 'C' in sections:
        C = {'spearman_bagsize_pop': spearman(bag_sizes, pop),
             'spearman_bagsize_mean_activation': spearman(bag_sizes, A.mean(1).numpy()),
             'spearman_bagsize_max_activation': spearman(bag_sizes, A.max(1).values.numpy()),
             'spearman_bagsize_qmeta_norm': anchors['corr_bagsize_qmeta_norm']}
        if has_id:
            C['spearman_bagsize_id_share_qsq'] = spearman(bag_sizes, id_share)
            C['spearman_bagsize_id_meta_norm_ratio'] = spearman(bag_sizes, ratio)
            C['spearman_pop_id_share_given_bagsize_partial'] = None  # placeholder, see md
        # bag-size bands
        edges = np.quantile(bag_sizes, [0, .25, .5, .75, 1.0])
        C['bands'] = {}
        for q in range(4):
            m = (bag_sizes >= edges[q]) & ((bag_sizes <= edges[q + 1]) if q == 3 else (bag_sizes < edges[q + 1]))
            if m.any():
                C['bands'][f'q{q}'] = {'n': int(m.sum()), 'bag_range': [float(edges[q]), float(edges[q + 1])],
                                       'train_pop': _stats(pop[m]), 'mean_activation': _stats(A[m].mean(1).numpy()),
                                       **({'id_share_qsq': _stats(id_share[m])} if has_id else {})}
        R['C'] = C

    # ── Section A ────────────────────────────────────────────────────────────────────
    if 'A' in sections:
        cross_only = layout == 'fixed'
        pairs, X2, pmeta = build_pairs(ids, w, foc, cross_only, args.min_support, args.max_pairs)
        X1 = torch.zeros(n_items, n_feat, dtype=torch.float64)
        X1.scatter_add_(1, ids.long(), wd.clamp_max(1.0))
        X1 = X1.clamp_max(1.0)
        c_id = (ID / qn) @ Pn.T
        Y = torch.cat([ID, c_id], dim=1)
        blocks = {'e_ID': (0, d), 'c_ID': (d, d + K)}
        lambdas = [float(x) for x in args.lambdas.split(',')]
        perm_all = rng.permutation(n_items)
        folds = []
        for k in range(args.n_folds):
            te = perm_all[k::args.n_folds]
            tr = np.setdiff1d(perm_all, te)
            folds.append((torch.from_numpy(tr), torch.from_numpy(te)))
        A_out = {'design': {'n_unary_cols': n_feat, 'cross_field_only': cross_only, **pmeta,
                            'n_items': n_items, 'n_folds': args.n_folds, 'lambdas': lambdas,
                            'n_perm': args.n_perm}}
        cv1 = ridge_cv(X1, Y, folds, lambdas, blocks, device)
        X12 = torch.cat([X1, X2], dim=1)
        cv12 = ridge_cv(X12, Y, folds, lambdas, blocks, device) if X2.shape[1] > 0 else None
        A_out['cv_unary'] = {str(l): v for l, v in cv1.items()}
        A_out['cv_unary_plus_pairs'] = {str(l): v for l, v in cv12.items()} if cv12 else None
        A_out['verdict'] = {}
        if cv12:
            lam1 = {blk: max(lambdas, key=lambda l: cv1[l][blk]) for blk in blocks}
            lam12 = {blk: max(lambdas, key=lambda l: cv12[l][blk]) for blk in blocks}
            null_lams = sorted(set(lam12.values()))
            null_runs = []
            for _ in range(args.n_perm):   # one SVD per fold per permutation serves both blocks
                Xp = torch.cat([X1, X2[torch.from_numpy(rng.permutation(n_items))]], dim=1)
                null_runs.append(ridge_cv(Xp, Y, folds, null_lams, blocks, device))
            for blk in blocks:
                r2_1, r2_12 = cv1[lam1[blk]][blk], cv12[lam12[blk]][blk]
                null = np.array([run[lam12[blk]][blk] for run in null_runs])
                delta = r2_12 - r2_1
                null_delta = null - r2_1
                A_out['verdict'][blk] = {
                    'lambda_unary': lam1[blk], 'lambda_pairs': lam12[blk],
                    'r2_unary': r2_1, 'r2_unary_plus_pairs': r2_12, 'delta_r2': delta,
                    'null_delta_r2': _stats(null_delta),
                    'null_p95': float(np.percentile(null_delta, 95)),
                    'perm_p_value': float((np.sum(null_delta >= delta) + 1) / (len(null_delta) + 1)),
                    'hot': bool(delta > np.percentile(null_delta, 95) and delta > 0)}
            # top pairs by coefficient norm on e_ID block (full-data fit)
            lam = A_out['verdict']['e_ID']['lambda_pairs']
            beta = ridge_fit(X12, Y[:, :d], lam, device)
            b2 = beta[n_feat:].norm(dim=1).numpy()
            b1 = beta[:n_feat].norm(dim=1).numpy()
            order = np.argsort(-b2)[:20]
            A_out['top_pairs_by_coef_norm'] = [
                {'pair': [list(c2fv[a]), list(c2fv[b])], 'coef_norm': float(b2[j]),
                 'support': int(X2[:, j].sum())} for j in order for a, b in [pairs[j]]]
            A_out['coef_norm'] = {'unary': _stats(b1), 'pairs': _stats(b2)}
        R['A'] = A_out

    R['_meta']['wall_seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, 'readout.json'), 'w') as f:
        json.dump(R, f, indent=2)
    write_md(R, os.path.join(out_dir, 'readout.md'))
    print(f"id_row_probe: wrote {out_dir}/readout.{{json,md}} in {R['_meta']['wall_seconds']}s")


def write_md(R, path):
    m = R['_meta']
    L = [f"# ID-row probe — `{os.path.basename(m['results_dir'].rstrip('/'))}`", '',
         f"*{m['tier']}.* dataset `{m['dataset']}`, layout `{m['layout']}`, fields {m['fields']}, "
         f"n_items {m['n_items']}, V {m['n_features']}, d {m['d']}, K {m['K']}, has_id {m['has_id']}; "
         f"sections {m['sections']}; wall {m['wall_seconds']}s.", '',
         '## Anchors (cross-check vs composition_analysis_2026-08-24)', '']
    for k, v in R['anchors'].items():
        L.append(f'- {k}: {v:.4f}')
    if 'B' in R:
        B = R['B']
        L += ['', '## B — popularity-stratified ID share', '',
              'Spearman with train popularity: ' + ', '.join(f'{k} {v:.3f}' for k, v in B['spearman_pop'].items()), '',
              '| band | n | id_share_qsq mean | median | norm ratio median | act-level ID share mean | mean activation |',
              '|---|---:|---:|---:|---:|---:|---:|']
        rows = [('overall', B['overall'])] + list(B['buckets'].items()) + list(B['deciles'].items())
        for name, b in rows:
            tag = name if 'pop_range' not in b else f"{name} (pop {b['pop_range'][0]:.0f}-{b['pop_range'][1]:.0f})"
            L.append(f"| {tag} | {b['n']} | {b['id_share_qsq']['mean']:.3f} | {b['id_share_qsq']['p50']:.3f} | "
                     f"{b['id_meta_norm_ratio']['p50']:.3f} | {b['id_share_activation']['mean']:.3f} | "
                     f"{b['mean_activation']['mean']:.3f} |")
        D = B['D2']
        L += ['', f"D2: ‖mean e_ID‖ / mean ‖e_ID‖ = **{D['mean_id_norm_over_mean_norm']:.3f}**; "
                  f"cos(mean e_ID, mean q_meta) = {D['cos_meanID_meanQmeta']:.3f}; "
                  f"eff. rank of ID block {D['eff_rank_ID_block']:.2f} (uncentred {D['eff_rank_ID_block_uncentred']:.2f}); "
                  f"‖e_ID‖ median {D['id_norm']['p50']:.3f}."]
    if 'C' in R:
        C = R['C']
        L += ['', '## C — bag-size channel (Spearman)', '']
        for k, v in C.items():
            if k != 'bands' and v is not None:
                L.append(f'- {k}: {v:.3f}')
        L += ['', '| bag-size quartile | n | bag range | train pop mean | mean activation | id_share mean |', '|---|---:|---|---:|---:|---:|']
        for name, b in C['bands'].items():
            L.append(f"| {name} | {b['n']} | {b['bag_range'][0]:.0f}-{b['bag_range'][1]:.0f} | {b['train_pop']['mean']:.1f} | "
                     f"{b['mean_activation']['mean']:.3f} | {b.get('id_share_qsq', {}).get('mean', float('nan')):.3f} |")
    if 'A' in R:
        A = R['A']
        dsg = A['design']
        L += ['', '## A — pair-encoding regression (U4 trigger i)', '',
              f"Design: {dsg['n_unary_cols']} unary cols; pairs {dsg['n_unique_pairs']} unique / "
              f"{dsg['n_pairs_kept']} kept (support ≥ {dsg['min_support']}, cross-field only {dsg['cross_field_only']}); "
              f"{dsg['n_folds']}-fold CV over {dsg['n_items']} items; null = {dsg['n_perm']} X2-row permutations.", '']
        if A.get('verdict'):
            L += ['| response | λ | R² unary | R² +pairs | ΔR² | null ΔR² mean | null p95 | perm p | HOT |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
            for blk, v in A['verdict'].items():
                L.append(f"| {blk} | {v['lambda_pairs']:g} | {v['r2_unary']:.4f} | {v['r2_unary_plus_pairs']:.4f} | "
                         f"{v['delta_r2']:+.4f} | {v['null_delta_r2']['mean']:+.4f} | {v['null_p95']:+.4f} | "
                         f"{v['perm_p_value']:.3f} | {'**YES**' if v['hot'] else 'no'} |")
            L += ['', 'Top pairs by coefficient norm (e_ID response, full-data fit):', '']
            for t in A['top_pairs_by_coef_norm']:
                (f1, v1), (f2, v2) = t['pair']
                L.append(f"- {f1}=`{v1}` × {f2}=`{v2}` — coef {t['coef_norm']:.3f}, support {t['support']}")
            L.append(f"\nCoefficient norms: unary median {A['coef_norm']['unary']['p50']:.3f}, pairs median {A['coef_norm']['pairs']['p50']:.3f}.")
        else:
            L.append('No pairs met the support threshold — section A not evaluated.')
        L += ['', 'Reading rule (pre-registered): HOT iff ΔR² > null p95 on the e_ID response. '
                  'The permutation null keeps the pair columns\' marginals but breaks their link to the item '
                  '(and to its unary indicators), so it measures pure overfit inflation of equally sparse columns.']
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
