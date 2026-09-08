"""SC.8 fU geometry instrument set (dc05 scrutiny commitments F-DC05-01/02/03a/06 +
design doc §3.8 M1'-M6' / 3b) — post-hoc geometry measurements on a trained checkpoint
whose USER branch is a PrototypeEmbedding (fU-ProtoMF arms AND the host ``user_proto``,
GR7-fairly: feature-keyed sub-instruments skip cleanly where no word table exists).
Zero training cost: tensors are read from the checkpoint; one batched forward composes
the user cloud. All outputs are threshold-free continua — any discretization belongs to
the dossier reading, not here ("production decides, research characterizes"). Scrutiny
working basis only, never thesis quotes (F-DC01-06 restriction).

Instruments:
  A. M5' set (F-DC05-01): mean angle of the user representation (metadata part m_u on fU;
     the free vector on the host) to the population mean direction, per train-history-length
     band — the committed M5' precondition figure data — plus metadata-part norm ||m_u|| and
     ID-energy-share ||e_ID||^2/(||m||^2+||e_ID||^2) per band on ids arms.
  B. Cloud spread & coverage in BOTH weightings (F-DC05-02): directional spread (top-eig
     share, effective rank of the q-hat covariance) and the A1 Eq. 4-5 coverage quantities
     (per-user max sim, per-prototype max sim), user-uniform vs interaction-weighted — the
     gap is the implicit-regularizer effect's size.
  C. B(t) channel (F-DC05-06 comparative rule / M6'): Var_i(B(t_i)), the B-vs-personalized
     variance share on sampled users, and the B(t) x item-train-popularity correlation —
     identical computation on fU and host; attribution by measurement, never by fiat.
  D. Gauge / activation spectrum (3b-i, C7'): singular spectrum + effective rank of the
     user activation matrix (u*-1); exact pinned dimension rank([1, P-hat]); per-item
     gauge mass of the trained t vectors (energy outside the pinned subspace).
  E. Profile instruments (M1'/M2'/M4', fU only): intrinsic profile C[f,l] = cos(e_f, p_l)
     entropy + top1-top2 margin; profile pairwise overlap; prototype pairwise latent cos;
     per-value vote-weight share vs basket frequency (the omnipresence readout).
  F. Member-user purchase-lift profiles (F-DC05-03a secondary data-grounded arm, per
     F-DC05-20): per prototype, feature lift over the pooled train purchases of the top-k
     nearest users vs the intrinsic profile — Spearman per prototype + side-by-side top
     descriptors for sample prototypes.

Output: <results_dir>/sc8_fu_geometry/readout.json + readout.md.

Usage:
    python Master/scripts/sc8_fu_geometry_readout.py \
        --results-dir /scratch/.../feature_user_proto_hm_1_month_s38210573
    (dataset dir resolved from config.json's data_path basename against DATA_PATH;
     override with --dataset; --hist-bands "3-3,4-5,6-12,13-" adds explicit bands
     beside the quartiles; --top-members 50; --sample-users 2000 for the B(t) share)
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get('PROTOMF_REPO',
                           os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch

from feature_extraction.feature_extractors import (Embedding, HistoryFeatureEmbedding,
                                                   PrototypeEmbedding)
from feature_extraction.feature_ids import (build_code_to_field_value, build_feature_bags,
                                            build_feature_ids)
from utilities.cold_eval import build_model, load_checkpoint
from utilities.consts import DATA_PATH, SINGLE_SEED
from utilities.utils import reproducible

PCTS = (1, 5, 25, 50, 75, 95, 99)
EPS = 1e-12


def _stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
            **{f'p{p}': float(np.percentile(x, p)) for p in PCTS}}


def _unit(M: np.ndarray) -> np.ndarray:
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), EPS)


def _eff_rank(sv: np.ndarray) -> float:
    """Participation ratio (sum s^2)^2 / sum s^4 of a singular-value vector."""
    s2 = np.asarray(sv, dtype=np.float64) ** 2
    denom = float((s2 ** 2).sum())
    return float(s2.sum() ** 2 / denom) if denom > 0 else 0.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float('nan')


def _weighted_cov_spectrum(U_hat: np.ndarray, w: np.ndarray) -> dict:
    """Eigen-spectrum of the w-weighted covariance of unit vectors (directional spread)."""
    w = w / w.sum()
    mean = (U_hat * w[:, None]).sum(axis=0)
    X = U_hat - mean[None, :]
    C = (X * w[:, None]).T @ X
    ev = np.linalg.eigvalsh(C)[::-1]
    ev = np.maximum(ev, 0.0)
    tot = float(ev.sum())
    return {'top_eig_share': float(ev[0] / tot) if tot > 0 else 0.0,
            'eff_rank': _eff_rank(np.sqrt(ev)),
            'total_var': tot,
            'mean_resultant_length': float(np.linalg.norm(mean))}


def _parse_bands(spec: str):
    bands = []
    for part in spec.split(','):
        lo, hi = part.split('-')
        bands.append((int(lo), int(hi) if hi else np.inf))
    return bands


def run_readout(results_dir: str, dataset: str = None, device: str = 'cpu',
                seed: int = SINGLE_SEED, hist_bands: str = None,
                sample_users: int = 2000, top_members: int = 50,
                batch_size: int = 8192) -> dict:
    with open(os.path.join(results_dir, 'config.json')) as f:
        conf = json.load(f)
    dataset = dataset or os.path.basename(conf['data_path'].rstrip('/'))
    data_path = dataset if os.path.isabs(dataset) else os.path.join(DATA_PATH, dataset)

    model, _, _ = build_model(conf, data_path, device)
    load_checkpoint(model, results_dir, device)
    model.eval()

    user_ext = model.user_feature_extractor
    assert isinstance(user_ext, PrototypeEmbedding), \
        f'user branch must be PrototypeEmbedding, got {type(user_ext).__name__}'
    inner = user_ext.embedding_ext
    is_fu = isinstance(inner, HistoryFeatureEmbedding)
    item_ext = model.item_feature_extractor
    assert isinstance(item_ext, Embedding), \
        f'item branch must be Embedding, got {type(item_ext).__name__}'

    reproducible(seed)
    rng = np.random.default_rng(seed)

    # ---- shared tensors -------------------------------------------------------------------
    with torch.no_grad():
        P = user_ext.prototypes.detach().cpu().double().numpy()            # (K, d)
        T = item_ext.embedding_layer.weight.detach().cpu().double().numpy()  # (M, K)
        n_users = (inner.hist_value_ids.shape[0] if is_fu
                   else inner.embedding_layer.num_embeddings)
        idx_all = torch.arange(n_users, device=device)
        Q = torch.cat([inner(idx_all[s:s + batch_size])
                       for s in range(0, n_users, batch_size)]).cpu().double().numpy()
        if is_fu:
            ids_buf = inner.hist_value_ids.cpu().numpy()                    # (N, D_max)
            w_buf = inner.hist_weights.cpu().double().numpy()               # (N, D_max)
            E_word = inner.embedding_layer.weight.detach().cpu().double().numpy()
            n_feat = int(inner.n_features)
            use_id = bool(inner.use_id_feature)
            # metadata part m_u (composition minus the ID row)
            if use_id:
                id_rows = E_word[n_feat + np.arange(n_users)]               # (N, d)
                M_meta = Q - id_rows
            else:
                id_rows = None
                M_meta = Q
        else:
            M_meta, id_rows, n_feat, use_id = Q, None, 0, False

    K, d = P.shape
    n_items = T.shape[0]

    # ---- similarity read-out of the run (dc07 generalisation, 2026-09-08) -----------------
    # Cosine family: act = a·cos + c under the run's own cosine_type (fixes the former hardcoded
    # 1+cos, which silently misreported 'standard' / 'shifted_and_div' runs). Membership family
    # (softmax / sigmoid): act = the membership itself, recomputed from the checkpoint's P, τ, b.
    # `act_center` is the gauge constant removed before the activation spectrum: c for cosine,
    # 1/K for softmax (rows sum to 1 — the simplex analogue of the +1), 0.5 for sigmoid.
    ct = getattr(user_ext, 'cosine_type', 'shifted')
    membership = bool(getattr(user_ext, 'membership_mode', False))
    tau = float(getattr(user_ext, 'temperature', 1.0))
    _AFFINE = {'shifted': (1.0, 1.0), 'standard': (1.0, 0.0), 'shifted_and_div': (0.5, 0.5)}
    if membership:
        mb = (user_ext.membership_bias.detach().cpu().double().numpy() if ct == 'sigmoid' else None)

        def _act(Qm):
            logits = (Qm @ P.T) / tau
            if ct == 'softmax':
                logits = logits - logits.max(axis=1, keepdims=True)
                e = np.exp(logits)
                return e / e.sum(axis=1, keepdims=True)
            return 1.0 / (1.0 + np.exp(-(logits + mb)))
        a_aff, c_aff = None, None
        act_center = (1.0 / K) if ct == 'softmax' else 0.5
    else:
        if ct not in _AFFINE:
            raise ValueError(f'unknown cosine_type {ct!r}')
        a_aff, c_aff = _AFFINE[ct]

        def _act(Qm):
            return c_aff + a_aff * (_unit(Qm) @ _unit(P).T)
        act_center = c_aff

    train = pd.read_csv(os.path.join(data_path, 'listening_history_train.csv'),
                        usecols=['user_id', 'item_id'])
    hist = train.groupby('user_id').size()
    user_hist = np.zeros(n_users, dtype=np.int64)
    user_hist[hist.index.to_numpy()] = hist.to_numpy()
    pop = train.groupby('item_id').size()
    item_pop = np.zeros(n_items, dtype=np.int64)
    item_pop[pop.index.to_numpy()] = pop.to_numpy()

    out = {'results_dir': os.path.abspath(results_dir), 'dataset': dataset, 'seed': seed,
           'model': conf['ft_ext_param'].get('ft_type'),
           'user_branch': type(inner).__name__, 'is_fu': is_fu, 'use_id_feature': use_id,
           'n_users': int(n_users), 'n_items': int(n_items),
           'n_prototypes': int(K), 'embedding_dim': int(d), 'n_features': int(n_feat),
           'similarity': {'cosine_type': ct, 'membership': membership, 'temperature': tau,
                          'affine': None if membership else [a_aff, c_aff],
                          'act_center': float(act_center)}}

    # band structure: quartiles always; explicit bands when given
    qs = np.percentile(user_hist, [25, 50, 75])
    band_defs = [('q1', 0, qs[0]), ('q2', qs[0], qs[1]),
                 ('q3', qs[1], qs[2]), ('q4', qs[2], np.inf)]
    out['hist_quartile_edges'] = [float(q) for q in qs]
    explicit = _parse_bands(hist_bands) if hist_bands else []

    def _band_masks():
        masks = []
        prev_hi = None
        for name, lo, hi in band_defs:
            m = (user_hist > lo) & (user_hist <= hi) if prev_hi is not None else \
                (user_hist <= hi)
            # quartile bands: (prev, cur] with first = <= q1
            masks.append((name, f'(|H|<={hi:g})' if prev_hi is None else
                          f'({lo:g}<|H|<={hi:g})', m))
            prev_hi = hi
        for lo, hi in explicit:
            m = (user_hist >= lo) & (user_hist <= hi)
            label = f'[{lo},{"inf" if np.isinf(hi) else int(hi)}]'
            masks.append((f'band{label}', label, m))
        return masks

    # ---- A. M5' set -----------------------------------------------------------------------
    A = {}
    mean_dir = M_meta.mean(axis=0)
    mean_dir = mean_dir / max(np.linalg.norm(mean_dir), EPS)
    M_hat = _unit(M_meta)
    ang = np.degrees(np.arccos(np.clip(M_hat @ mean_dir, -1.0, 1.0)))
    m_norm = np.linalg.norm(M_meta, axis=1)
    A['angle_to_mean_deg_overall'] = _stats(ang)
    A['metadata_norm_overall'] = _stats(m_norm)
    if use_id:
        id_norm2 = (id_rows ** 2).sum(axis=1)
        id_share = id_norm2 / np.maximum(id_norm2 + m_norm ** 2, EPS)
        A['id_energy_share_overall'] = _stats(id_share)
    A['by_band'] = []
    for name, label, m in _band_masks():
        if m.sum() == 0:
            continue
        entry = {'band': name, 'range': label, 'n_users': int(m.sum()),
                 'angle_to_mean_deg': _stats(ang[m]),
                 'metadata_norm': _stats(m_norm[m])}
        if use_id:
            entry['id_energy_share'] = _stats(id_share[m])
        A['by_band'].append(entry)
    out['A_m5_set'] = A

    # ---- B. Cloud spread & coverage, two weightings (F-DC05-02) ---------------------------
    B = {}
    Q_hat = _unit(Q)
    w_uni = np.ones(n_users, dtype=np.float64)
    w_int = user_hist.astype(np.float64)
    sims = _act(Q)                                                         # (N, K) activations
    per_user_max = sims.max(axis=1)
    per_proto_max = sims.max(axis=0)
    for wname, w in (('user_uniform', w_uni), ('interaction_weighted', w_int)):
        wn = w / w.sum()
        B[wname] = {
            'spread': _weighted_cov_spectrum(Q_hat, w),
            'coverage_user_to_proto_mean_max_sim': float((per_user_max * wn).sum()),
        }
    B['coverage_proto_to_user_max_sim'] = _stats(per_proto_max)  # user-side max is weighting-free
    # weighting delta headline
    B['spread_top_eig_share_delta'] = (B['interaction_weighted']['spread']['top_eig_share']
                                       - B['user_uniform']['spread']['top_eig_share'])
    out['B_cloud_two_weightings'] = B

    # ---- C. B(t) channel (F-DC05-06 / M6') ------------------------------------------------
    # Cosine family: B(t) = c·1ᵀt, the offset's per-item mass. Membership family: no offset
    # exists, so the block measures the dc07 popularity-rediscovery detector instead — the
    # top-prototype score share over the top-10 recommendations and the mean membership vector
    # (a dominant prototype everyone belongs to reproduces B(t) at the layer level).
    C = {'mode': 'membership' if membership else 'affine_offset'}
    su = rng.choice(n_users, size=min(sample_users, n_users), replace=False)
    Ustar_s = sims[su]                                                     # (S, K)
    C['n_sample_users'] = int(len(su))
    lp = np.log1p(item_pop.astype(np.float64))
    if not membership:
        Bt = c_aff * T.sum(axis=1)                                         # (M,)
        C['var_B_across_items'] = float(Bt.var())
        C['B_stats'] = _stats(Bt)
        C['pearson_B_logpop'] = float(np.corrcoef(Bt, lp)[0, 1]) if Bt.var() > 0 else float('nan')
        C['spearman_B_pop'] = _spearman(Bt, item_pop.astype(np.float64)) if Bt.var() > 0 else float('nan')
        pers = (Ustar_s - c_aff) @ T.T                                     # (S, M)
        var_pers = pers.var(axis=1)                                        # per-user var over items
        C['mean_var_personalized_across_items'] = float(var_pers.mean())
        C['B_variance_share'] = float(Bt.var() / max(Bt.var() + var_pers.mean(), EPS))
    else:
        scores = Ustar_s @ T.T                                             # (S, M)
        top = np.argsort(-scores, axis=1)[:, :10]                          # (S, 10)
        lstar = Ustar_s.argmax(axis=1)                                     # (S,)
        num = Ustar_s[np.arange(len(su)), lstar][:, None] * T[top, lstar[:, None]]
        den = np.take_along_axis(scores, top, axis=1)
        share = num / np.where(np.abs(den) > EPS, den, np.nan)
        C['top_prototype_score_share_top10'] = _stats(share[np.isfinite(share)])
        mean_m = sims.mean(axis=0)                                         # (K,)
        C['mean_membership_vector'] = [float(v) for v in mean_m]
        C['mean_membership_max'] = float(mean_m.max())
        C['mean_membership_argmax'] = int(mean_m.argmax())
        C['modal_prototype_share_of_objects'] = float((sims.argmax(axis=1) == mean_m.argmax()).mean())
        if ct == 'softmax':
            H = -(sims * np.log(np.maximum(sims, EPS))).sum(axis=1)
            C['membership_entropy_over_logK'] = _stats(H / np.log(K))
        # popularity of the layer-level channel: Σ_l t_{i,l}·mean_m_l vs item popularity
        chan = T @ mean_m
        C['spearman_meanmembership_channel_pop'] = _spearman(chan, item_pop.astype(np.float64))
    out['C_bt_channel'] = C

    # ---- D. Gauge / activation spectrum (3b-i) --------------------------------------------
    D = {}
    sv = np.linalg.svd(sims - act_center, compute_uv=False)                # (N, K) activations, gauge removed
    D['activation_sv_top10'] = [float(s) for s in sv[:10]]
    D['activation_eff_rank'] = _eff_rank(sv)
    D['activation_rank_1e-6'] = int((sv > sv[0] * 1e-6).sum()) if sv.size else 0
    # pinned subspace of t-space: the score probes t only via 1^T t and P-hat^T t, i.e. inner
    # products with the K-vectors {1} and the d columns of P-hat (K x d); note K <= d+1 means
    # NO exact gauge (only effective/weakly-pinned directions via activation under-spanning)
    pin = np.concatenate([np.ones((K, 1)), _unit(P)], axis=1)              # (K, d+1)
    Qpin, Rpin = np.linalg.qr(pin)
    diag = np.abs(np.diag(Rpin))
    r_pin = int((diag > diag.max() * 1e-10).sum()) if diag.size else 0
    D['pinned_dim'] = r_pin
    D['exact_gauge_dim'] = int(K - r_pin)
    Qp = Qpin[:, :r_pin]
    T_in = T @ Qp                                                          # (M, r)
    in_energy = (T_in ** 2).sum(axis=1)
    tot_energy = (T ** 2).sum(axis=1)
    gauge_mass = 1.0 - in_energy / np.maximum(tot_energy, EPS)
    D['t_gauge_mass'] = _stats(gauge_mass)
    D['B_is_pinned'] = True   # 1 in pinned basis by construction
    out['D_gauge_spectrum'] = D

    # ---- E. Profile instruments (fU only) -------------------------------------------------
    if is_fu and n_feat > 0:
        E = {}
        E_meta = E_word[:n_feat]                                           # (V, d)
        Cprof = _unit(E_meta) @ _unit(P).T                                 # (V, K) cos(e_f, p_l)
        # entropy of softmax over the vocabulary, per prototype
        Z = np.exp(Cprof - Cprof.max(axis=0, keepdims=True))
        Pr = Z / Z.sum(axis=0, keepdims=True)
        ent = -(Pr * np.log(np.maximum(Pr, EPS))).sum(axis=0)
        E['profile_softmax_entropy'] = _stats(ent)
        E['profile_entropy_uniform_ref'] = float(np.log(n_feat))
        srt = np.sort(Cprof, axis=0)[::-1]
        E['profile_top1'] = _stats(srt[0])
        E['profile_top1_top2_margin'] = _stats(srt[0] - srt[1])
        # profile pairwise overlap + prototype pairwise latent cos (upper triangles)
        pc = _unit(Cprof.T) @ _unit(Cprof.T).T
        iu = np.triu_indices(K, 1)
        E['profile_pairwise_overlap_cos'] = _stats(pc[iu])
        pl = _unit(P) @ _unit(P).T
        E['proto_pairwise_latent_cos'] = _stats(pl[iu])
        # per-value vote-weight share vs basket frequency (M2' omnipresence, vote-weight form)
        word_norm = np.linalg.norm(E_meta, axis=1)                         # ||e_f||
        flat_ids = ids_buf.reshape(-1)
        flat_w = w_buf.reshape(-1)
        keep = flat_w > 0
        flat_ids, flat_w = flat_ids[keep], flat_w[keep]
        row_user = np.repeat(np.arange(n_users), ids_buf.shape[1])[keep]
        vote = flat_w * word_norm[flat_ids]
        denom_u = np.zeros(n_users)
        np.add.at(denom_u, row_user, vote)
        share = vote / np.maximum(denom_u[row_user], EPS)
        share_sum = np.zeros(n_feat)
        np.add.at(share_sum, flat_ids, share)
        carrier_cnt = np.zeros(n_feat)
        np.add.at(carrier_cnt, flat_ids, 1.0)
        mean_share = share_sum / np.maximum(carrier_cnt, 1.0)
        freq = carrier_cnt / n_users
        fields = conf['ft_ext_param']['user_ft_ext_param']['feature_fields']
        layout = conf['ft_ext_param']['user_ft_ext_param'].get('feature_layout', 'fixed')
        code_fv = build_code_to_field_value(data_path, fields, layout=layout)
        E['share_vs_freq_pearson'] = float(np.corrcoef(freq, mean_share)[0, 1])
        top = np.argsort(-freq)[:15]
        E['per_value_share_vs_freq_top15'] = [
            {'code': int(f), 'label': f'{code_fv[f][0]}={code_fv[f][1]}',
             'freq': float(freq[f]), 'mean_vote_share': float(mean_share[f])}
            for f in top]
        out['E_profiles'] = E

        # ---- F. Member-user purchase-lift vs intrinsic (F-DC05-03a/-20) -------------------
        F = {}
        cosUP = Q_hat @ _unit(P).T                                         # (N, K)
        if layout == 'bags':
            bag_ids, bag_w, nf2 = build_feature_bags(data_path, fields)
            bag_ids = bag_ids.numpy(); bag_mask = bag_w.numpy() > 0
            item_codes = [bag_ids[i][bag_mask[i]] for i in range(n_items)]
        else:
            fids, nf2 = build_feature_ids(data_path, fields)
            item_codes = [row for row in fids.numpy()]
        assert nf2 == n_feat, f'vocab mismatch: builder {nf2} vs model {n_feat}'
        # base distribution over train purchases
        base_cnt = np.zeros(n_feat)
        tr_items = train['item_id'].to_numpy()
        for i, c in zip(*np.unique(tr_items, return_counts=True)):
            base_cnt[item_codes[i]] += c
        base_p = base_cnt / max(base_cnt.sum(), EPS)
        user_items = train.groupby('user_id')['item_id'].agg(list)
        spearmans, details = [], []
        for l in range(K):
            members = np.argsort(-cosUP[:, l])[:top_members]
            cnt = np.zeros(n_feat)
            for u in members:
                for it in user_items.get(u, []):
                    cnt[item_codes[it]] += 1
            tot = cnt.sum()
            if tot == 0:
                spearmans.append(float('nan')); continue
            lift = (cnt / tot + EPS) / (base_p + EPS)
            rho = _spearman(np.log(lift), Cprof[:, l])
            spearmans.append(rho)
            top5_int = set(int(f) for f in np.argsort(-Cprof[:, l])[:5])
            masked = np.where(cnt >= 5, lift, -np.inf)
            top10_lift_set = set(int(f) for f in np.argsort(-masked)[:10])
            overlaps5in10 = len(top5_int & top10_lift_set)
            details.append({'prototype': int(l), 'spearman': float(rho),
                            'top5_intrinsic_in_top10_lift': overlaps5in10,
                            'members_mean_hist': float(user_hist[members].mean()),
                            'top5_intrinsic': [f'{code_fv[f][0]}={code_fv[f][1]}'
                                               for f in np.argsort(-Cprof[:, l])[:5]],
                            'top5_lift': [f'{code_fv[f][0]}={code_fv[f][1]}'
                                          for f in np.argsort(-np.where(cnt >= 5, lift, -np.inf))[:5]]})
        sp = np.array([s for s in spearmans if np.isfinite(s)])
        F['spearman_lift_vs_intrinsic'] = _stats(sp)
        F['top5_intrinsic_in_top10_lift'] = _stats(
            [dd['top5_intrinsic_in_top10_lift'] for dd in details])
        order = np.argsort([dd['spearman'] for dd in details])
        picks = sorted({0, len(details) // 2, len(details) - 1,
                        len(details) // 4, 3 * len(details) // 4})
        F['sample_prototypes'] = [details[order[i]] for i in picks if i < len(details)]
        F['top_members_k'] = top_members
        out['F_member_lift'] = F

    # ---- write ---------------------------------------------------------------------------
    out_dir = os.path.join(results_dir, 'sc8_fu_geometry')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'readout.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)

    L = [f'# SC.8 fU geometry readout — {out["model"]} × {dataset}',
         '', f'(scrutiny working basis only — no thesis quotes; seed {seed})', '',
         f'- users {n_users:,} | items {n_items:,} | K={K} | d={d} | '
         f'user branch {out["user_branch"]}{" (ids)" if use_id else (" (noid)" if is_fu else "")}',
         f'- similarity: cosine_type={ct}' + (f', τ={tau:.4g} (membership family; spectrum centred at '
                                             f'{act_center:.4g})' if membership else
                                             f' (act = {a_aff:g}·cos + {c_aff:g})'),
         '', '## A. M5\' set — angle-to-mean / metadata norm / ID share by |H| band', '',
         '| band | range | users | angle° mean | angle° p50 | ‖m‖ mean |'
         + (' ID-share mean |' if use_id else ''),
         '|---|---|---:|---:|---:|---:|' + ('---:|' if use_id else '')]
    for e in A['by_band']:
        row = (f"| {e['band']} | {e['range']} | {e['n_users']:,} | "
               f"{e['angle_to_mean_deg']['mean']:.2f} | {e['angle_to_mean_deg']['p50']:.2f} | "
               f"{e['metadata_norm']['mean']:.3f} |")
        if use_id:
            row += f" {e['id_energy_share']['mean']:.3f} |"
        L.append(row)
    L += ['', '## B. Cloud spread & coverage — two weightings (F-DC05-02)', '']
    for wname in ('user_uniform', 'interaction_weighted'):
        s = B[wname]['spread']
        L.append(f"- {wname}: top-eig share {s['top_eig_share']:.4f}, eff rank "
                 f"{s['eff_rank']:.2f}, mean resultant length {s['mean_resultant_length']:.4f}, "
                 f"coverage U→P mean max-sim {B[wname]['coverage_user_to_proto_mean_max_sim']:.4f}")
    L.append(f"- top-eig-share delta (interaction − uniform): {B['spread_top_eig_share_delta']:+.4f}")
    L.append(f"- coverage P→U max-sim: mean {B['coverage_proto_to_user_max_sim']['mean']:.4f}, "
             f"min p1 {B['coverage_proto_to_user_max_sim']['p1']:.4f}")
    if not membership:
        L += ['', '## C. B(t) channel (comparative instrument, F-DC05-06)', '',
              f"- Var_i(B) = {C['var_B_across_items']:.4f}; mean per-user Var_i(personalized) = "
              f"{C['mean_var_personalized_across_items']:.4f}; **B variance share = "
              f"{C['B_variance_share']:.4f}** ({C['n_sample_users']} sampled users)",
              f"- corr(B, log pop) Pearson {C['pearson_B_logpop']:.4f}; "
              f"Spearman(B, pop) {C['spearman_B_pop']:.4f}"]
    else:
        L += ['', '## C. Layer-level popularity channel (dc07 membership family; B(t) undefined — no offset)', '',
              f"- top-prototype score share over top-10 recs: mean {C['top_prototype_score_share_top10']['mean']:.4f}, "
              f"p50 {C['top_prototype_score_share_top10']['p50']:.4f} ({C['n_sample_users']} sampled users)",
              f"- mean membership vector: max {C['mean_membership_max']:.4f} at prototype "
              f"{C['mean_membership_argmax']} (uniform would be {1.0 / K:.4f}); modal prototype is argmax for "
              f"{C['modal_prototype_share_of_objects']:.3f} of users",
              f"- Spearman(Σ_l t_l·mean_m_l, item pop) = {C['spearman_meanmembership_channel_pop']:.4f}"]
        if 'membership_entropy_over_logK' in C:
            e = C['membership_entropy_over_logK']
            L.append(f"- membership entropy / log K: mean {e['mean']:.4f}, p5 {e['p5']:.4f}, p95 {e['p95']:.4f}")
    L += ['', '## D. Gauge / activation spectrum (3b-i)', '',
          f"- activation eff rank {D['activation_eff_rank']:.2f} "
          f"(numerical rank {D['activation_rank_1e-6']} of {K})",
          f"- pinned dim {D['pinned_dim']} (of d+1={d + 1} max); exact gauge dim {D['exact_gauge_dim']}",
          f"- t gauge mass: mean {D['t_gauge_mass']['mean']:.4f}, p50 {D['t_gauge_mass']['p50']:.4f}, "
          f"p99 {D['t_gauge_mass']['p99']:.4f}"]
    if 'E_profiles' in out:
        E = out['E_profiles']
        L += ['', '## E. Profile instruments (M1\'/M2\'/M4\')', '',
              f"- profile softmax entropy: mean {E['profile_softmax_entropy']['mean']:.3f} "
              f"(uniform ref {E['profile_entropy_uniform_ref']:.3f})",
              f"- top1 cos: mean {E['profile_top1']['mean']:.4f}; top1−top2 margin mean "
              f"{E['profile_top1_top2_margin']['mean']:.4f}",
              f"- profile pairwise overlap cos: mean {E['profile_pairwise_overlap_cos']['mean']:.4f}, "
              f"p99 {E['profile_pairwise_overlap_cos']['p99']:.4f}",
              f"- proto pairwise latent cos: mean {E['proto_pairwise_latent_cos']['mean']:.4f}, "
              f"p99 {E['proto_pairwise_latent_cos']['p99']:.4f}",
              f"- vote-share vs basket-freq Pearson: {E['share_vs_freq_pearson']:.4f}",
              '', '### Per-value vote share vs basket frequency (top-15 by freq)', '',
              '| value | basket freq | mean vote share |', '|---|---:|---:|']
        for v in E['per_value_share_vs_freq_top15']:
            L.append(f"| {v['label']} | {v['freq']:.3f} | {v['mean_vote_share']:.4f} |")
    if 'F_member_lift' in out:
        F = out['F_member_lift']
        L += ['', '## F. Member-user purchase-lift vs intrinsic profiles (F-DC05-03a/-20)', '',
              f"- Spearman(lift, intrinsic) over {K} prototypes: mean "
              f"{F['spearman_lift_vs_intrinsic']['mean']:.3f}, p25 "
              f"{F['spearman_lift_vs_intrinsic']['p25']:.3f}, p75 "
              f"{F['spearman_lift_vs_intrinsic']['p75']:.3f} (k={F['top_members_k']} members)",
              f"- top-5 intrinsic descriptors found in top-10 member-lift: mean "
              f"{F['top5_intrinsic_in_top10_lift']['mean']:.2f} of 5 "
              f"(p25 {F['top5_intrinsic_in_top10_lift']['p25']:.0f}, "
              f"p75 {F['top5_intrinsic_in_top10_lift']['p75']:.0f})",
              '', '| proto | ρ | top-5 intrinsic | top-5 member-lift |', '|---|---:|---|---|']
        for dd in F['sample_prototypes']:
            L.append(f"| {dd['prototype']} | {dd['spearman']:.3f} | "
                     f"{'; '.join(dd['top5_intrinsic'])} | {'; '.join(dd['top5_lift'])} |")
    with open(os.path.join(out_dir, 'readout.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'sc8_fu_geometry readout written to {out_dir}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SC.8 fU geometry instrument set (dc05)')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=SINGLE_SEED)
    ap.add_argument('--hist-bands', default=None,
                    help='explicit |H| bands beside quartiles, e.g. "3-3,4-5,6-12,13-"')
    ap.add_argument('--sample-users', type=int, default=2000)
    ap.add_argument('--top-members', type=int, default=50)
    args = ap.parse_args()
    run_readout(args.results_dir, dataset=args.dataset, device=args.device,
                seed=args.seed, hist_bands=args.hist_bands,
                sample_users=args.sample_users, top_members=args.top_members)
