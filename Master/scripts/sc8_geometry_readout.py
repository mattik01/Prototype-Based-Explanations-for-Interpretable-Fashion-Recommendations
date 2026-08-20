"""SC.8 geometry instrument set (dc01 scrutiny commitments F-DC01-05/08/09) — post-hoc
geometry measurements on a trained checkpoint whose ITEM branch is a PrototypeEmbedding.
Zero training cost: tensors are read from the checkpoint; one batched forward composes the
item cloud. Applies GR7-fairly to every arm AND the host (feature-keyed sub-instruments skip
cleanly where no feature table exists). All outputs are threshold-free continua — any
discretization belongs to the dossier reading, not here ("production decides, research
characterizes"). Scrutiny working basis only, never thesis quotes (F-DC01-06 restriction).

Instruments:
  A. Twin-angle distribution (F-DC01-09): cosine between item representations for
     same-signature twin pairs vs matched (same first-field, different signature) pairs vs
     random pairs — at BOTH the composed-embedding level (q) and the activation level (t*,
     the surface the score actually consumes). Plus the ID-row-vs-metadata-mass ratio
     ||e_ID,i|| / ||m_i|| (the angular twin-separation budget) on ids arms.
  B. Gauge-mass / activation-spectrum (F-DC01-08): singular spectrum of the item activation
     matrix T = t*(all items); per-direction mean user-energy share aligned to that spectrum
     (its low-sigma tail is the effective gauge); exact gauge dimension (sigma ~ 0).
  C. M1/M4 geometry set (F-DC01-05): per-prototype activation spread; prototype pairwise
     latent cosine; prototype profile (cos(e_f, p_k)) pairwise overlap; per-value linear
     projection share vs catalog frequency (the omnipresence / value-dominance readout),
     incl. the aggregate ID-row share (feature-explained-fraction complement) on ids arms.

Output: <results_dir>/sc8_geometry/readout.json + readout.md.

Usage:
    python Master/scripts/sc8_geometry_readout.py \
        --results-dir /scratch/.../feature_item_proto_hm_1_month_s38210573
    (dataset dir resolved from config.json's data_path basename against DATA_PATH;
     override with --dataset; --max-pairs bounds the sampled pair sets, default 20000)
"""
import argparse
import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from feature_extraction.feature_extractors import (Embedding, FeatureEmbedding,
                                                   PrototypeEmbedding)
from utilities.cold_eval import build_model, load_checkpoint
from utilities.consts import DATA_PATH, SINGLE_SEED
from utilities.utils import reproducible

PCTS = (1, 5, 25, 50, 75, 95, 99)


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
            **{f'p{p}': float(np.percentile(x, p)) for p in PCTS}}


def _pair_cos(M: torch.Tensor, pairs: np.ndarray) -> np.ndarray:
    a = torch.nn.functional.normalize(M[pairs[:, 0]], dim=-1)
    b = torch.nn.functional.normalize(M[pairs[:, 1]], dim=-1)
    return (a * b).sum(-1).cpu().numpy()


def _sample_pairs(groups, rng, max_pairs):
    """Sample up to max_pairs unordered index pairs from within the given groups
    (list of index arrays, each of size >= 2), weighted by each group's pair count."""
    sizes = np.array([len(g) for g in groups], dtype=np.int64)
    weights = sizes * (sizes - 1) // 2
    total = int(weights.sum())
    if total == 0:
        return np.empty((0, 2), dtype=np.int64)
    n = min(max_pairs, total)
    gi = rng.choice(len(groups), size=n, p=weights / weights.sum())
    out = np.empty((n, 2), dtype=np.int64)
    for r, g in enumerate(gi):
        i, j = rng.choice(sizes[g], size=2, replace=False)
        out[r] = groups[g][i], groups[g][j]
    return out


def run_readout(results_dir: str, dataset: str = None, device: str = 'cpu',
                seed: int = SINGLE_SEED, max_pairs: int = 20000,
                batch_size: int = 4096) -> dict:
    with open(os.path.join(results_dir, 'config.json')) as f:
        conf = json.load(f)
    dataset = dataset or os.path.basename(conf['data_path'].rstrip('/'))
    data_path = dataset if os.path.isabs(dataset) else os.path.join(DATA_PATH, dataset)

    model, _, _ = build_model(conf, data_path, device)
    load_checkpoint(model, results_dir, device)
    model.eval()

    item_ext = model.item_feature_extractor
    assert isinstance(item_ext, PrototypeEmbedding), \
        f'item branch must be PrototypeEmbedding, got {type(item_ext).__name__}'
    inner = item_ext.embedding_ext
    is_feature = isinstance(inner, FeatureEmbedding)
    user_ext = model.user_feature_extractor
    assert isinstance(user_ext, Embedding), \
        f'user branch must be Embedding, got {type(user_ext).__name__}'

    n_items = inner.n_objects if is_feature else inner.embedding_layer.num_embeddings
    reproducible(seed)
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        idx_all = torch.arange(n_items, device=device)
        Q = torch.cat([inner(idx_all[s:s + batch_size])
                       for s in range(0, n_items, batch_size)])       # (n_items, d)
        T = torch.cat([item_ext(idx_all[s:s + batch_size])
                       for s in range(0, n_items, batch_size)])       # (n_items, K)
        U = user_ext.embedding_layer.weight.detach()                  # (n_users, K)
        P = item_ext.prototypes.detach()                              # (K, d)

    out = {'results_dir': results_dir, 'dataset': dataset, 'seed': seed,
           'model': conf['ft_ext_param']['item_ft_ext_param'].get('ft_type',
                    conf['ft_ext_param'].get('ft_type')),
           'item_branch': type(inner).__name__,
           'n_items': int(n_items), 'n_users': int(U.shape[0]),
           'n_prototypes': int(P.shape[0]), 'embedding_dim': int(P.shape[1]),
           'max_pairs': max_pairs}

    # ---- A. Twin-angle distribution (F-DC01-09) ------------------------------------------
    A = {}
    if is_feature and inner.feature_ids.shape[1] > 0:
        F = inner.feature_ids.cpu().numpy()                           # (n_items, F)
        if inner.feature_weights is not None:                         # bag layout: mask padding
            W = inner.feature_weights.cpu().numpy()
            sigs = [tuple(sorted(int(t) for t, w in zip(r, wr) if w > 0))
                    for r, wr in zip(F, W)]
        else:
            sigs = [tuple(int(t) for t in r) for r in F]
        by_sig = defaultdict(list)
        for i, s in enumerate(sigs):
            by_sig[s].append(i)
        twin_groups = [np.array(v) for v in by_sig.values() if len(v) >= 2]
        n_twin_items = int(sum(len(g) for g in twin_groups))
        twin_pairs = _sample_pairs(twin_groups, rng, max_pairs)

        # matched: same first-field value, different signature
        by_f0 = defaultdict(list)
        for i, r in enumerate(F):
            by_f0[int(r[0])].append(i)
        matched = []
        f0_groups = [np.array(v) for v in by_f0.values() if len(v) >= 2]
        cand = _sample_pairs(f0_groups, rng, max_pairs * 2)
        for a, b in cand:
            if sigs[a] != sigs[b]:
                matched.append((a, b))
            if len(matched) >= len(twin_pairs):
                break
        matched = np.array(matched, dtype=np.int64).reshape(-1, 2)

        rand = rng.integers(0, n_items, size=(len(twin_pairs), 2))
        rand = rand[rand[:, 0] != rand[:, 1]]

        A['n_signature_classes_ge2'] = len(twin_groups)
        A['n_twin_items'] = n_twin_items
        A['twin_item_share'] = n_twin_items / n_items
        for label, pairs in (('twin', twin_pairs), ('matched_nontwin', matched),
                             ('random', rand)):
            if len(pairs) == 0:
                A[label] = {'n': 0}
                continue
            A[label] = {'q_cos': _stats(_pair_cos(Q, pairs)),
                        't_cos': _stats(_pair_cos(T, pairs)),
                        't_l2_dist': _stats(torch.linalg.norm(
                            T[pairs[:, 0]] - T[pairs[:, 1]], dim=-1).cpu().numpy())}
        if inner.use_id_feature:
            with torch.no_grad():
                E_id = inner.embedding_layer.weight[inner.n_features:
                                                    inner.n_features + n_items]
                M = Q - E_id                                          # metadata-only sum
                ratio = (torch.linalg.norm(E_id, dim=-1) /
                         torch.linalg.norm(M, dim=-1).clamp_min(1e-12))
            A['id_over_metadata_norm_ratio'] = _stats(ratio.cpu().numpy())
    else:
        A['skipped'] = 'no feature table on this item branch (host/f0 arm) — twin pairs undefined'
        rand = rng.integers(0, n_items, size=(max_pairs, 2))
        rand = rand[rand[:, 0] != rand[:, 1]]
        A['random'] = {'q_cos': _stats(_pair_cos(Q, rand)),
                       't_cos': _stats(_pair_cos(T, rand))}
    out['A_twin_angles'] = A

    # ---- B. Gauge mass / activation spectrum (F-DC01-08) ---------------------------------
    with torch.no_grad():
        Tc = T.double()
        _, S, Vh = torch.linalg.svd(Tc, full_matrices=False)          # S: (K,), Vh: (K, K)
        energy = (S ** 2)
        t_energy_share = (energy / energy.sum()).cpu().numpy()
        Ud = U.double()
        proj = Ud @ Vh.T                                              # (n_users, K)
        u_energy = proj ** 2
        u_share = (u_energy / u_energy.sum(dim=1, keepdim=True).clamp_min(1e-30))
        u_share_mean = u_share.mean(dim=0).cpu().numpy()
        rel = (S / S[0]).cpu().numpy()
        exact_gauge_dim = int((rel < 1e-10).sum())
        p = t_energy_share[t_energy_share > 0]
        eff_rank_entropy = float(np.exp(-(p * np.log(p)).sum()))
    out['B_gauge_spectrum'] = {
        'singular_values_rel': [float(x) for x in rel],
        't_energy_share': [float(x) for x in t_energy_share],
        'mean_user_energy_share_per_direction': [float(x) for x in u_share_mean],
        'exact_gauge_dim': exact_gauge_dim,
        'effective_rank_entropy': eff_rank_entropy,
        'note': 'directions sorted by T-spectrum; the low-sigma tail is the effective gauge — '
                'read the (t_energy_share, mean_user_energy_share) continuum jointly',
    }

    # ---- C. M1/M4 geometry set (F-DC01-05) ------------------------------------------------
    C = {}
    C['t_activation_spread_per_proto'] = _stats(T.std(dim=0).cpu().numpy())
    C['q_direction_mean_pairwise_cos'] = _stats(
        _pair_cos(Q, rng.integers(0, n_items, size=(max_pairs, 2))))
    with torch.no_grad():
        Pn = torch.nn.functional.normalize(P, dim=-1)
        pc = (Pn @ Pn.T).cpu().numpy()
    iu = np.triu_indices(pc.shape[0], k=1)
    C['proto_pairwise_cos'] = _stats(pc[iu])
    if is_feature:
        with torch.no_grad():
            E = inner.embedding_layer.weight[:inner.n_features]       # (V, d)
            PROF = (torch.nn.functional.normalize(E, dim=-1)
                    @ Pn.T).cpu()                                     # (V, K) cos(e_f, p_k)
            PROFn = torch.nn.functional.normalize(PROF.T.double(), dim=-1)
            oc = (PROFn @ PROFn.T).cpu().numpy()
        C['profile_pairwise_overlap_cos'] = _stats(oc[iu])

        # per-value linear projection share vs catalog frequency
        F = inner.feature_ids.cpu().numpy()
        Wb = (inner.feature_weights.cpu().numpy()
              if inner.feature_weights is not None else None)
        qn2 = (Q.double() ** 2).sum(-1).clamp_min(1e-30)
        Ed = inner.embedding_layer.weight.double()
        carriers = defaultdict(list)
        for i, r in enumerate(F):
            vals = (set(int(t) for t, w in zip(r, Wb[i]) if w > 0) if Wb is not None
                    else set(int(t) for t in r))
            for v in vals:
                carriers[v].append(i)
        per_value = {}
        with torch.no_grad():
            for v, items in carriers.items():
                ii = torch.as_tensor(items, device=Q.device)
                sh = (Q.double()[ii] @ Ed[v]) / qn2[ii]
                per_value[int(v)] = {'freq': len(items) / n_items,
                                     'mean_share': float(sh.mean()),
                                     'n_items': len(items)}
        out_labels = None
        try:
            from feature_extraction.feature_ids import build_code_to_field_value
            ip = conf['ft_ext_param']['item_ft_ext_param']
            out_labels = build_code_to_field_value(
                data_path, ip['feature_fields'],
                layout=ip.get('feature_layout', 'fixed'))
        except Exception as e:                                        # labels are decoration
            C['label_note'] = f'labels unavailable: {e}'
        if out_labels is not None:
            for v in per_value:
                f_, val_ = out_labels[v]
                per_value[v]['label'] = f'{f_}={val_}'
        C['per_value_share_vs_freq'] = per_value
        if inner.use_id_feature:
            with torch.no_grad():
                E_id = Ed[inner.n_features:inner.n_features + n_items]
                id_share = ((Q.double() * E_id).sum(-1) / qn2).cpu().numpy()
            C['id_row_share'] = _stats(id_share)
    out['C_geometry'] = C

    out_dir = os.path.join(results_dir, 'sc8_geometry')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'readout.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # ---- markdown companion ----------------------------------------------------------------
    L = [f'# SC.8 geometry readout — {os.path.basename(results_dir)}', '',
         f'model `{out["model"]}` | item branch `{out["item_branch"]}` | '
         f'{n_items} items | {out["n_users"]} users | K={out["n_prototypes"]} | '
         f'd={out["embedding_dim"]} | seed {seed}', '',
         '> Scrutiny working basis only (F-DC01-06 restriction) — no thesis quotes.', '']
    L += ['## A. Twin angles (F-DC01-09)', '']
    if 'skipped' in A:
        L += [f'_{A["skipped"]}_',
              f'random q-cos mean {A["random"]["q_cos"].get("mean", float("nan")):.4f} / '
              f't-cos mean {A["random"]["t_cos"].get("mean", float("nan")):.4f}', '']
    else:
        L += [f'{A["n_signature_classes_ge2"]} twin classes (≥2), {A["n_twin_items"]} items '
              f'({A["twin_item_share"]:.1%} of catalog)', '',
              '| pair set | n | q-cos mean | q-cos p5 | t-cos mean | t-L2 mean |',
              '|---|---:|---:|---:|---:|---:|']
        for label in ('twin', 'matched_nontwin', 'random'):
            s = A[label]
            if s.get('n') == 0 or 'q_cos' not in s:
                L.append(f'| {label} | 0 | — | — | — | — |')
                continue
            L.append(f'| {label} | {s["q_cos"]["n"]} | {s["q_cos"]["mean"]:.4f} | '
                     f'{s["q_cos"]["p5"]:.4f} | {s["t_cos"]["mean"]:.6f} | '
                     f'{s["t_l2_dist"]["mean"]:.4f} |')
        if 'id_over_metadata_norm_ratio' in A:
            r = A['id_over_metadata_norm_ratio']
            L += ['', f'ID/metadata norm ratio ‖e_ID‖/‖m‖: mean {r["mean"]:.3f}, '
                      f'median {r["p50"]:.3f}, p5 {r["p5"]:.3f}, p95 {r["p95"]:.3f}']
        L.append('')
    B = out['B_gauge_spectrum']
    L += ['## B. Activation spectrum & user gauge mass (F-DC01-08)', '',
          f'exact gauge dim: **{B["exact_gauge_dim"]}** | effective rank (entropy): '
          f'{B["effective_rank_entropy"]:.2f} / K={out["n_prototypes"]}', '',
          '| k | rel σ | T-energy share | mean u-energy share |', '|---:|---:|---:|---:|']
    K = len(B['singular_values_rel'])
    show = list(range(min(5, K))) + list(range(max(5, K - 5), K))
    prev = -1
    for k in show:
        if k <= prev:
            continue
        if k > prev + 1:
            L.append('| … | … | … | … |')
        L.append(f'| {k} | {B["singular_values_rel"][k]:.2e} | '
                 f'{B["t_energy_share"][k]:.2e} | '
                 f'{B["mean_user_energy_share_per_direction"][k]:.2e} |')
        prev = k
    L.append('')
    L += ['## C. M1/M4 geometry (F-DC01-05)', '',
          f'- t\\* per-prototype spread: mean {C["t_activation_spread_per_proto"]["mean"]:.4f} '
          f'(p5 {C["t_activation_spread_per_proto"]["p5"]:.4f}, '
          f'p95 {C["t_activation_spread_per_proto"]["p95"]:.4f})',
          f'- q-direction pairwise cos (random): mean '
          f'{C["q_direction_mean_pairwise_cos"]["mean"]:.4f}',
          f'- prototype pairwise cos: mean {C["proto_pairwise_cos"]["mean"]:.4f}, '
          f'p99 {C["proto_pairwise_cos"]["p99"]:.4f}']
    if 'profile_pairwise_overlap_cos' in C:
        L.append(f'- profile pairwise overlap cos: mean '
                 f'{C["profile_pairwise_overlap_cos"]["mean"]:.4f}, '
                 f'p99 {C["profile_pairwise_overlap_cos"]["p99"]:.4f}')
    if 'id_row_share' in C:
        L.append(f'- ID-row linear share of ‖q‖²: mean {C["id_row_share"]["mean"]:.3f} '
                 f'(median {C["id_row_share"]["p50"]:.3f})')
    if 'per_value_share_vs_freq' in C:
        pv = sorted(C['per_value_share_vs_freq'].items(),
                    key=lambda kv: -kv[1]['freq'])[:15]
        L += ['', '### Per-value share vs catalog frequency (top-15 by freq)', '',
              '| value | freq | mean linear share |', '|---|---:|---:|']
        for v, dd in pv:
            L.append(f'| {dd.get("label", v)} | {dd["freq"]:.3f} | {dd["mean_share"]:.4f} |')
    with open(os.path.join(out_dir, 'readout.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'sc8_geometry readout written to {out_dir}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SC.8 geometry instrument set (F-DC01-05/08/09)')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=SINGLE_SEED)
    ap.add_argument('--max-pairs', type=int, default=20000)
    args = ap.parse_args()
    run_readout(args.results_dir, dataset=args.dataset, device=args.device,
                seed=args.seed, max_pairs=args.max_pairs)
