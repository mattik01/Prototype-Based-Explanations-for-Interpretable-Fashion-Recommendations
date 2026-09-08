"""dc07 membership read-out — the three pre-registered instruments of the membership-similarity
candidate (requirements memo Master/temp/dc07_membership_similarity_requirements_2026-09-08.md §8),
run post-hoc on a trained checkpoint whose user and/or item branch is a PrototypeEmbedding.
Login-node workload (no training). Scrutiny working basis only — never thesis quotes.

Instruments (per prototype side found):
  1. Membership entropy H(m) per object: distribution; fraction in the uniform regime
     (H > 0.9·log K — the τ→∞ popularity-channel degeneracy) and in the hard regime
     (H < 0.1·log K); Spearman of H vs train history length (users) / train popularity (items).
     The direct test of P4 (evidence mass → confidence) and the τ-degeneracy guard.
  2. Top-prototype score share: Σ_l t_{i,l} m_l restricted to l* = argmax_l m_l, over the top-10
     recommendations of sampled users — the rediscovered-popularity detector (a dominant
     prototype everyone belongs to reproduces B(t) at the layer level). Plus the mean membership
     vector and the share of objects whose argmax is the modal prototype.
  3. Winning τ from config.json against the search range (DC07_TAU_MIN/MAX in confs.hyper_params),
     flagged when within 10 % of an edge in log-space.

On a COSINE-family checkpoint the script still runs (comparison rows): the "membership" is then
the row-normalised activation act/Σact — labelled as such, never as a membership.

Output: <results_dir>/dc07_membership/readout.json + readout.md

Usage:
    python Master/scripts/dc07_membership_readout.py --results-dir /scratch/.../user_proto_sm_hm_1_month_s38210573
"""
import argparse
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get('PROTOMF_REPO', os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch

from feature_extraction.feature_extractors import Embedding, PrototypeEmbedding
from utilities.cold_eval import build_model, load_checkpoint
from utilities.consts import DATA_PATH, SINGLE_SEED
from utilities.utils import reproducible

EPS = 1e-12
PCTS = (1, 5, 25, 50, 75, 95, 99)


def _stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
            **{f'p{p}': float(np.percentile(x, p)) for p in PCTS}}


def _spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _entropy_rows(m: np.ndarray) -> np.ndarray:
    return -(m * np.log(np.maximum(m, EPS))).sum(axis=1)


def _tau_range():
    try:
        from confs.hyper_params import DC07_TAU_MIN, DC07_TAU_MAX
        return float(DC07_TAU_MIN), float(DC07_TAU_MAX)
    except Exception:  # noqa: BLE001 — range constants absent → report unflagged
        return None, None


def _side_readout(name: str, ext: PrototypeEmbedding, other_vecs: np.ndarray, covariate: np.ndarray,
                  covariate_name: str, n_objects: int, rng, sample_users: int, top_k: int,
                  device: str, batch_size: int, other_is_user: bool) -> dict:
    """One prototype side. ``other_vecs`` is the free side's table (rows in prototype space K).
    ``other_is_user``: True when the prototype side is the ITEM side (scores = U @ Mᵀ)."""
    K = int(ext.n_prototypes)
    ct = getattr(ext, 'cosine_type', 'shifted')
    membership = bool(getattr(ext, 'membership_mode', False))
    with torch.no_grad():
        idx_all = torch.arange(n_objects, device=device)
        act = torch.cat([ext(idx_all[s:s + batch_size]) for s in range(0, n_objects, batch_size)])
        act = act.detach().cpu().double().numpy()                       # (N, K) = the forward output
        Q = torch.cat([ext.embedding_ext(idx_all[s:s + batch_size])
                       for s in range(0, n_objects, batch_size)]).detach().cpu().double().numpy()
    q_norm = np.linalg.norm(Q, axis=1)
    if membership:
        M = act
        unit = 'membership m (forward output)'
    else:
        M = act / np.maximum(act.sum(axis=1, keepdims=True), EPS)
        unit = f'row-normalised activation act/Σact under cosine_type={ct} (NOT a membership; comparison row)'

    R = {'side': name, 'cosine_type': ct, 'membership_family': membership,
         'temperature': float(getattr(ext, 'temperature', 1.0)) if membership else None,
         'n_objects': int(n_objects), 'n_prototypes': K, 'unit': unit}

    # 1. entropy
    H = _entropy_rows(M)
    logK = math.log(K)
    R['entropy'] = {
        'log_K': logK,
        'H_over_logK': _stats(H / logK),
        'frac_uniform_regime_H_gt_0.9logK': float((H > 0.9 * logK).mean()),
        'frac_hard_regime_H_lt_0.1logK': float((H < 0.1 * logK).mean()),
        f'spearman_H_vs_{covariate_name}': _spearman(H, covariate),
        'spearman_H_vs_q_norm': _spearman(H, q_norm),
        f'spearman_q_norm_vs_{covariate_name}': _spearman(q_norm, covariate),
        'q_norm': _stats(q_norm),
    }
    if membership and ct == 'sigmoid':
        R['entropy']['note'] = ('sigmoid memberships do not sum to 1; H is computed on the row-normalised '
                                'm/Σm, and the raw row sum is reported separately')
        R['entropy']['row_sum'] = _stats(act.sum(axis=1))

    # 2. top-prototype score share over top-k recommendations
    mean_m = M.mean(axis=0)
    modal = int(mean_m.argmax())
    R['mean_membership_vector'] = [float(v) for v in mean_m]
    R['mean_membership_max'] = float(mean_m.max())
    R['mean_membership_argmax'] = modal
    R['uniform_value_1_over_K'] = 1.0 / K
    R['modal_prototype_share_of_objects'] = float((M.argmax(axis=1) == modal).mean())
    lstar = M.argmax(axis=1)
    if other_is_user:
        # prototype side = items: scores_u = U[u] @ Mᵀ ; share_i = U[u,l*_i]·M[i,l*_i] / score
        n_users = other_vecs.shape[0]
        su = rng.choice(n_users, size=min(sample_users, n_users), replace=False)
        scores = other_vecs[su] @ M.T                                   # (S, N_items)
        top = np.argsort(-scores, axis=1)[:, :top_k]                    # (S, k)
        l_top = lstar[top]                                              # (S, k) argmax prototype of each top item
        Us = other_vecs[su]                                             # (S, K)
        num = Us[np.arange(len(su))[:, None], l_top] * M[top, l_top]    # U[u,l*_i]·M[i,l*_i]
    else:
        # prototype side = users: scores_u = M[u] @ Tᵀ ; share_i = M[u,l*_u]·T[i,l*_u] / score
        su = rng.choice(n_objects, size=min(sample_users, n_objects), replace=False)
        scores = M[su] @ other_vecs.T                                   # (S, N_items)
        top = np.argsort(-scores, axis=1)[:, :top_k]
        ls = lstar[su]
        num = M[su, ls][:, None] * other_vecs[top, ls[:, None]]
    den = np.take_along_axis(scores, top, axis=1)
    share = num / np.where(np.abs(den) > EPS, den, np.nan)
    R['top_prototype_score_share'] = {'top_k': top_k, 'n_sample_users': int(len(su)),
                                      'stats': _stats(share)}
    # layer-level popularity channel: how much a per-item scalar Σ_l x_{i,l}·mean_m_l explains pop
    if other_is_user:
        chan = M @ other_vecs.mean(axis=0)                              # items: m_i · mean user vector
    else:
        chan = other_vecs @ mean_m                                      # items: t_i · mean membership
    R['spearman_layer_channel_vs_item_pop'] = None  # filled by caller (needs item pop)
    R['_chan'] = chan
    return R


def run_readout(results_dir: str, dataset: str = None, device: str = 'cpu', seed: int = SINGLE_SEED,
                sample_users: int = 2000, top_k: int = 10, batch_size: int = 8192) -> dict:
    with open(os.path.join(results_dir, 'config.json')) as f:
        conf = json.load(f)
    dataset = dataset or os.path.basename(conf['data_path'].rstrip('/'))
    data_path = dataset if os.path.isabs(dataset) else os.path.join(DATA_PATH, dataset)
    model, n_users, n_items = build_model(conf, data_path, device)
    load_checkpoint(model, results_dir, device)
    model.eval()
    reproducible(seed)
    rng = np.random.default_rng(seed)

    train = pd.read_csv(os.path.join(data_path, 'listening_history_train.csv'), usecols=['user_id', 'item_id'])
    hist = train.groupby('user_id').size()
    user_hist = np.zeros(n_users, dtype=np.int64); user_hist[hist.index.to_numpy()] = hist.to_numpy()
    pop = train.groupby('item_id').size()
    item_pop = np.zeros(n_items, dtype=np.int64); item_pop[pop.index.to_numpy()] = pop.to_numpy()

    u_ext, i_ext = model.user_feature_extractor, model.item_feature_extractor
    out = {'results_dir': os.path.abspath(results_dir), 'dataset': dataset, 'seed': seed,
           'model': conf['ft_ext_param'].get('ft_type'), 'n_users': int(n_users), 'n_items': int(n_items),
           'sides': []}

    if isinstance(u_ext, PrototypeEmbedding) and isinstance(i_ext, Embedding):
        T = i_ext.embedding_layer.weight.detach().cpu().double().numpy()
        R = _side_readout('user', u_ext, T, user_hist.astype(np.float64), 'history_length', n_users, rng,
                          sample_users, top_k, device, batch_size, other_is_user=False)
        R['spearman_layer_channel_vs_item_pop'] = _spearman(R.pop('_chan'), item_pop)
        out['sides'].append(R)
    if isinstance(i_ext, PrototypeEmbedding) and isinstance(u_ext, Embedding):
        U = u_ext.embedding_layer.weight.detach().cpu().double().numpy()
        R = _side_readout('item', i_ext, U, item_pop.astype(np.float64), 'train_popularity', n_items, rng,
                          sample_users, top_k, device, batch_size, other_is_user=True)
        R['spearman_layer_channel_vs_item_pop'] = _spearman(R.pop('_chan'), item_pop)
        out['sides'].append(R)
    if not out['sides']:
        raise RuntimeError(f'no (PrototypeEmbedding × Embedding) side pair found: user={type(u_ext).__name__}, '
                           f'item={type(i_ext).__name__} — dc07 stage 1/2 hosts only')

    # 3. winning τ vs range
    lo, hi = _tau_range()
    taus = {}
    for side in ('user', 'item'):
        spec = conf['ft_ext_param'].get(f'{side}_ft_ext_param', {})
        if 'temperature' in spec:
            tau = float(spec['temperature'])
            entry = {'tau': tau, 'range': [lo, hi]}
            if lo and hi:
                pos = math.log(tau / lo) / math.log(hi / lo)
                entry['log_position_in_range'] = pos
                entry['near_edge'] = bool(pos < 0.1 or pos > 0.9)
                entry['edge'] = 'tau_min' if pos < 0.1 else ('tau_max (uniform/popularity regime!)' if pos > 0.9 else None)
            taus[side] = entry
    out['winning_temperature'] = taus

    out_dir = os.path.join(results_dir, 'dc07_membership')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'readout.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)

    L = [f'# dc07 membership readout — {out["model"]} × {dataset}', '',
         f'(scrutiny working basis only — no thesis quotes; seed {seed})', '']
    for R in out['sides']:
        e = R['entropy']; s = R['top_prototype_score_share']
        L += [f'## {R["side"]} side — cosine_type={R["cosine_type"]}'
              + (f', τ={R["temperature"]:.4g}' if R['temperature'] is not None else ''),
              f'- unit: {R["unit"]}', f'- objects {R["n_objects"]:,} | K={R["n_prototypes"]} | log K = {e["log_K"]:.3f}', '',
              '### 1. membership entropy H(m)',
              f'- H/logK: mean {e["H_over_logK"]["mean"]:.4f}, p5 {e["H_over_logK"]["p5"]:.4f}, '
              f'p50 {e["H_over_logK"]["p50"]:.4f}, p95 {e["H_over_logK"]["p95"]:.4f}',
              f'- **uniform regime (H > 0.9 log K): {e["frac_uniform_regime_H_gt_0.9logK"]:.3f}** of objects | '
              f'hard regime (H < 0.1 log K): {e["frac_hard_regime_H_lt_0.1logK"]:.3f}',
              f'- Spearman(H, {[k for k in e if k.startswith("spearman_H_vs_") and "q_norm" not in k][0][14:]}) = '
              f'{[v for k, v in e.items() if k.startswith("spearman_H_vs_") and "q_norm" not in k][0]:.4f}; '
              f'Spearman(H, ‖q‖) = {e["spearman_H_vs_q_norm"]:.4f}',
              f'- ‖q‖: mean {e["q_norm"]["mean"]:.4f}, p5 {e["q_norm"]["p5"]:.4f}, p95 {e["q_norm"]["p95"]:.4f}', '',
              '### 2. top-prototype score share (popularity-rediscovery detector)',
              f'- share over top-{s["top_k"]} recs: mean {s["stats"].get("mean", float("nan")):.4f}, '
              f'p50 {s["stats"].get("p50", float("nan")):.4f} ({s["n_sample_users"]} sampled users)',
              f'- mean membership vector: max {R["mean_membership_max"]:.4f} at prototype {R["mean_membership_argmax"]} '
              f'(uniform = {R["uniform_value_1_over_K"]:.4f}); modal prototype is argmax for '
              f'{R["modal_prototype_share_of_objects"]:.3f} of objects',
              f'- Spearman(layer-level per-item channel, item pop) = {R["spearman_layer_channel_vs_item_pop"]:.4f}', '']
    L += ['### 3. winning temperature', '']
    if not taus:
        L.append('- no `temperature` key in config (cosine-family run)')
    for side, t in taus.items():
        L.append(f'- {side}: τ = {t["tau"]:.4g}, range {t["range"]}'
                 + (f', log-position {t["log_position_in_range"]:.3f}' if 'log_position_in_range' in t else '')
                 + (f' — **FLAG: near {t["edge"]}**' if t.get('near_edge') else ''))
    with open(os.path.join(out_dir, 'readout.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=SINGLE_SEED)
    ap.add_argument('--sample-users', type=int, default=2000)
    ap.add_argument('--top-k', type=int, default=10)
    a = ap.parse_args()
    run_readout(a.results_dir, a.dataset, a.device, a.seed, a.sample_users, a.top_k)


if __name__ == '__main__':
    main()
