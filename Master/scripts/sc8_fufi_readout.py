"""SC.8 fUfI one-off measurements — dc09 scrutiny commitments F-DC09-09 / -10 / -11 / -21(ii)
(design doc 3b-xii, M16, M18; dossier SC.7 §3). Read-only on a trained checkpoint of any fUfI
arm (`feature_user_item_proto`, `_noid`, `_sep`, `_noid_sep`); instrument A also runs on the UI
host (`user_item_proto`), instruments C and E on any model whose item side is a word composition
(fI too). Zero training cost. Every output is a descriptive continuum: working evidence for
SC.8, never a thesis quote (F-DC01-06 restriction).

Each instrument says whether a reparametrisation that leaves every score unchanged could move
it (learnings [dc09 SC.1b + SC.5]):

  A. Common-mode decomposition (F-DC09-09, 3b-xii). Every score component is a user factor
     times an item factor, so its exact two-way decomposition over ALL users x ALL items (grand
     mean, item main effect, user main effect, interaction) follows from KxK matrices — no
     sampling. Headline per component: common share = SS_item / (SS_item + SS_int), the part of
     the within-user, item-to-item (rank-relevant) variation that is the same for every user.
     Grand mean + item main effect is exactly the renderer's "common to all users" value.
     Beside it, Spearman(common part, log train popularity) — M8's kept B_item instrument,
     extended to the common parts. Fixed by the scores.
     ids arms with a user-ID row add the 3b-xii comparator: the rank-relevant SS of the user-ID
     line's contribution relative to the half's, in the field-mean read-out. On the user side
     that split is NOT fixed by the scores (F-DC09-13, -21); the output says so.
  B. Same-word vs different-word pairs (F-DC09-10, M16). The personalised part is exactly
     q_u' M_ui q_i with M_ui = a_u P_u' W_t / |q_u| + a_t W_u' P_t / |q_i| (P rows unit
     prototypes, bias-free projections). Expanding both compositions into their rows splits it
     into same-word pairs (a word in the basket and in the candidate), different-word pairs and
     ID-involving pairs, exactly per pair (asserted against the forward score). Pair sets per
     sampled user: the top-10 unseen items, the test item, 10 random unseen items (the overlap
     base rate). ids arms are read in the field-mean read-out (relative to that convention: the
     translation of F-DC09-07 moves the raw split); noid arms are fixed by the scores.
  C. Basket twins (F-DC09-10). Fraction of train positives (u, i) whose other train purchases
     hold an item with the identical word set; the catalogue twin fraction beside it. Data only.
  D. Cross-prototype cosine (F-DC09-11, M18). Row-max distribution of cos(p^u_l, p^t_k) in both
     directions, the within-set off-diagonal row max for context, and a random-direction
     reference with the same K_u, K_t, d. Fixed by the scores (the prototypes are compared with
     q, which no reparametrisation moves). Under `_sep` the two sets live on different tables:
     the control.
  E. Item-side words-vs-ID variance split (F-DC09-21 ii). Across items: the word part minus what
     the per-field masses predict (least squares) vs the ID part plus it — invariant to the
     translation on both layouts; on the fixed layout (every field mass 1) it is the plain
     variance split. In q_i, and through W_t (t_hat) and on B_item = c_u 1't_hat where a linear
     unit exists. The raw split (as weight decay left it) is printed beside it for contrast.
     ids arms only (an item-ID row is needed).

Output: <out-dir>/readout.json + readout.md (default <results-dir>/sc8_fufi_readout/).

Usage:
    python Master/scripts/sc8_fufi_readout.py --results-dir <run dir> [--out-dir DIR]
        [--dataset NAME_OR_PATH] [--sample-users 2000] [--seed 38210573]
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

from feature_extraction.feature_extractors import (FeatureEmbedding, HistoryFeatureEmbedding,
                                                   PrototypeEmbedding)
from feature_extraction.feature_ids import build_field_index
from utilities.explanations.breakdown import cosine_affine
from utilities.explanations.gauge import field_means

PCTS = (1, 5, 25, 50, 75, 95, 99)
EPS = 1e-12
TOP_N = 10
SEED = 38210573


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {'n': 0}
    return {'n': int(x.size), 'mean': float(x.mean()), 'std': float(x.std()),
            **{f'p{p}': float(np.percentile(x, p)) for p in PCTS}}


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float('nan')


def _unit(M: torch.Tensor) -> torch.Tensor:
    return M / M.norm(dim=-1, keepdim=True).clamp_min(EPS)


@torch.no_grad()
def _batched(fn, n: int, batch: int = 8192) -> torch.Tensor:
    return torch.cat([fn(torch.arange(s, min(s + batch, n))) for s in range(0, n, batch)]).double()


def _reset(*fes):
    for fe in fes:
        if hasattr(fe, 'get_and_reset_loss'):
            fe.get_and_reset_loss()


def proto_layers(model):
    """(user PrototypeEmbedding, item PrototypeEmbedding, kind) — kind 'fufi' | 'host' | None."""
    ufe, ife = model.user_feature_extractor, model.item_feature_extractor
    if hasattr(ufe, 'proto') and hasattr(ife, 'proto'):
        return ufe.proto, ife.proto, 'fufi'

    def find(fe):
        return next((m for m in (getattr(fe, 'model_1', None), getattr(fe, 'model_2', None))
                     if isinstance(m, PrototypeEmbedding)), None)
    up, ip = find(ufe), find(ife)
    return (up, ip, 'host') if up is not None and ip is not None else (None, None, None)


def item_composition(model):
    """The item side's word composition (FeatureEmbedding), or None."""
    ife = model.item_feature_extractor
    for cand in (getattr(ife, 'base', None), getattr(ife, 'embedding_ext', None), ife):
        if isinstance(cand, FeatureEmbedding):
            return cand
    return None


def item_code_weights(ib: FeatureEmbedding) -> torch.Tensor:
    """(n_items, V) dense per-item word weights (fixed layout: 1 per field word)."""
    ids = ib.feature_ids
    w = ib.feature_weights if getattr(ib, 'feature_weights', None) is not None \
        else torch.ones(ids.shape)
    M = torch.zeros(ids.shape[0], ib.n_features, dtype=torch.float64)
    M.scatter_add_(1, ids.long(), w.double())
    return M


def user_code_weights(ub: HistoryFeatureEmbedding, users) -> torch.Tensor:
    """(len(users), V) dense basket word weights w̄_{u,f} (word codes only)."""
    V = ub.n_features
    M = torch.zeros(len(users), V, dtype=torch.float64)
    for r, u in enumerate(users):
        codes, w = ub.user_slots(int(u))
        keep = codes < V
        M[r].index_add_(0, codes[keep].long(), w[keep].double())
    return M


def user_field_mass(ub: HistoryFeatureEmbedding, foc, n_fields: int) -> torch.Tensor:
    """(n_users, n_fields): summed basket weight of each field's words, per user."""
    n, V = ub.n_objects, ub.n_features
    if ub.pooling == 'bag':
        counts = ub.row_ptr[1:] - ub.row_ptr[:-1]
        users = torch.repeat_interleave(torch.arange(n), counts)
        ids, w = ub.flat_ids, ub.flat_w
    else:
        ids, w = ub.hist_value_ids.reshape(-1), ub.hist_weights.reshape(-1)
        users = torch.arange(n).repeat_interleave(ub.hist_value_ids.shape[1])
    keep = (w != 0) & (ids < V)
    mass = torch.zeros(n, n_fields, dtype=torch.float64)
    mass.index_put_((users[keep], foc[ids[keep]]), w[keep].double(), accumulate=True)
    return mass


def item_field_mass(ib: FeatureEmbedding, foc) -> torch.Tensor:
    return item_code_weights(ib) @ torch.nn.functional.one_hot(
        foc, int(foc.max()) + 1).double()


# ---------------------------------------------------------------------------
# A — common-mode two-way decomposition (exact, all users x all items)
# ---------------------------------------------------------------------------

def two_way(X: torch.Tensor, Y: torch.Tensor) -> dict:
    """Exact two-way decomposition of H = X Y' over every row of X (users) and Y (items):
    H = m + a_i + b_u + r. Sums of squares from KxK matrices (r_ui = x~_u . y~_i)."""
    nu, ni = X.shape[0], Y.shape[0]
    xb, yb = X.mean(0), Y.mean(0)
    m = float(xb @ yb)
    a = Y @ xb - m
    b = X @ yb - m
    Xc, Yc = X - xb, Y - yb
    ss_int = float(((Xc.T @ Xc) * (Yc.T @ Yc)).sum())
    ss_item, ss_user = float(nu * (a @ a)), float(ni * (b @ b))
    rank = ss_item + ss_int
    return {'grand_mean': m, 'ss_item': ss_item, 'ss_user': ss_user, 'ss_interaction': ss_int,
            'common_share': ss_item / rank if rank > 0 else float('nan'),
            'sd_item_main': float(np.sqrt(ss_item / (nu * ni))),
            'sd_user_main': float(np.sqrt(ss_user / (nu * ni))),
            'sd_interaction': float(np.sqrt(ss_int / (nu * ni))),
            '_common_part': (m + a)}


def instrument_common(Uo, Io, k_u, aff, popularity, id_parts=None) -> dict:
    a_u, c_u, a_t, c_t = aff
    XU, YU = Uo[:, :k_u] - c_u, Io[:, :k_u]
    XI, YI = Uo[:, k_u:], Io[:, k_u:] - c_t
    b_item, b_user = c_u * YU.sum(1), c_t * XI.sum(1)
    one_u, one_i = torch.ones(len(Uo), 1, dtype=Uo.dtype), torch.ones(len(Io), 1, dtype=Io.dtype)
    comps = {'U_half': (XU, YU), 'I_half': (XI, YI),
             'personalised': (torch.cat([XU, XI], 1), torch.cat([YU, YI], 1)),
             'score': (torch.cat([XU, XI, one_u, b_user[:, None]], 1),
                       torch.cat([YU, YI, b_item[:, None], one_i], 1))}
    logpop = np.log1p(popularity)
    out = {'note': 'exact over all users x all items; common share = SS_item/(SS_item+SS_int); '
                   'fixed by the scores'}
    for name, (X, Y) in comps.items():
        tw = two_way(X, Y)
        common = tw.pop('_common_part').numpy()
        tw['spearman_common_part_vs_log_popularity'] = _spearman(common, logpop)
        out[name] = tw
    out['B_item'] = {'sd': float(b_item.std()), 'mean': float(b_item.mean()),
                     'spearman_vs_log_popularity': _spearman(b_item.numpy(), logpop)}
    if id_parts is not None:
        XU_id, XI_id = id_parts
        cmp = {}
        for name, (Xid, X, Y) in {'U_half': (XU_id, XU, YU), 'I_half': (XI_id, XI, YI),
                                  'personalised': (torch.cat([XU_id, XI_id], 1),
                                                   torch.cat([XU, XI], 1),
                                                   torch.cat([YU, YI], 1))}.items():
            full, idp = two_way(X, Y), two_way(Xid, Y)
            r_full = full['ss_item'] + full['ss_interaction']
            cmp[name] = {'id_line_rank_ss_ratio': (idp['ss_item'] + idp['ss_interaction']) / r_full
                         if r_full > 0 else float('nan'),
                         'id_line_common_share': idp['common_share']}
        cmp['note'] = ('3b-xii comparator, field-mean read-out. NOT fixed by the scores: on the user '
                       'side the words/ID split moves under a scale reparametrisation pinned only by '
                       'leave-one-out training (F-DC09-13, -21). Read as where training left the mass.')
        out['user_id_line'] = cmp
    return out


def user_id_parts(model, Q_u, Uo, k_u, aff, foc, n_fields):
    """The user-ID line's share of the two user factors (u*-c, u_hat), field-mean read-out."""
    ufe = model.user_feature_extractor
    ub = ufe.base
    a_u, c_u = aff[0], aff[1]
    T = ub.embedding_layer.weight.detach().double()
    V = ub.n_features
    e_id = T[ub.id_row_offset + torch.arange(ub.n_objects)]
    id_line = e_id + user_field_mass(ub, foc, n_fields) @ field_means(T[:V], foc, n_fields)
    Pu = _unit(ufe.proto.prototypes.detach().double())
    norms = Q_u.norm(dim=1, keepdim=True).clamp_min(EPS)
    xu_lin = a_u * (Q_u / norms) @ Pu.T
    err = float((xu_lin - (Uo[:, :k_u] - c_u)).abs().max())
    assert err < 1e-3, f'cosine attribution does not reproduce u*-c (max err {err:.2e})'
    W_u = ufe.linear_layer.weight.detach().double()
    return a_u * (id_line / norms) @ Pu.T, id_line @ W_u.T


# ---------------------------------------------------------------------------
# B — same-word / different-word / ID-involving pairs (exact per pair)
# ---------------------------------------------------------------------------

def readout_rows(base, foc, n_fields):
    """(word rows (V,d), field means (F,d)) of one side's table in the read-out: centred where
    the side has an ID row (field-mean read-out), raw otherwise (means returned as zeros)."""
    T = base.embedding_layer.weight.detach().double()
    V = base.n_features
    if base.use_id_feature:
        m = field_means(T[:V], foc, n_fields)
        return T[:V] - m[foc], m
    return T[:V], torch.zeros(n_fields, T.shape[1], dtype=T.dtype)


def instrument_pairs(model, Uo, Io, Q_u, Q_i, k_u, aff, foc, n_fields, pair_sets) -> dict:
    ufe, ife = model.user_feature_extractor, model.item_feature_extractor
    ub, ib = ufe.base, ife.base
    a_u, c_u, a_t, c_t = aff
    EU, _ = readout_rows(ub, foc, n_fields)
    EI, _ = readout_rows(ib, foc, n_fields)
    Pu = _unit(ufe.proto.prototypes.detach().double())
    Pt = _unit(ife.proto.prototypes.detach().double())
    Wt = ife.linear_layer.weight.detach().double()           # (K_u, d)
    Wu = ufe.linear_layer.weight.detach().double()           # (K_t, d)
    A = a_u * Pu.T @ Wt                                       # q_u' A q_i / |q_u| = U-half
    Bm = a_t * Wu.T @ Pt                                      # q_u' B q_i / |q_i| = I-half
    dA = torch.einsum('fd,de,fe->f', EU, A, EI)
    dB = torch.einsum('fd,de,fe->f', EU, Bm, EI)
    Mi = item_code_weights(ib)                                # (n_i, V)
    Wi = Mi @ EI                                              # item word parts (read-out)
    b_item = c_u * Io[:, :k_u].sum(1)
    b_user = c_t * Uo[:, k_u:].sum(1)
    n_qi = Q_i.norm(dim=1).clamp_min(EPS)
    out, max_err = {}, 0.0
    for set_name, pairs in pair_sets.items():
        users = sorted(pairs)
        Wd = user_code_weights(ub, users)
        Wu_word = Wd @ EU
        acc = {k: [] for k in ('P', 'same', 'diff', 'id', 'overlap')}
        for r, u in enumerate(users):
            c = torch.as_tensor(pairs[u], dtype=torch.long)
            qu, nu, ni = Q_u[u], Q_u[u].norm().clamp_min(EPS), n_qi[c]
            P = (qu @ A @ Q_i[c].T) / nu + (qu @ Bm @ Q_i[c].T) / ni
            fwd = (Uo[u] @ Io[c].T) - b_item[c] - b_user[u]
            max_err = max(max_err, float(((P - fwd).abs() / (1.0 + fwd.abs())).max()))
            ww = (Wu_word[r] @ A @ Wi[c].T) / nu + (Wu_word[r] @ Bm @ Wi[c].T) / ni
            ov = Mi[c] * Wd[r]                                 # (n_c, V) same-word weights
            same = (ov @ dA) / nu + (ov @ dB) / ni
            acc['P'].append(P); acc['same'].append(same); acc['diff'].append(ww - same)
            acc['id'].append(P - ww); acc['overlap'].append((ov.sum(1) > 0).double())
        cat = {k: torch.cat(v).numpy() for k, v in acc.items()}
        mP = float(cat['P'].mean())
        out[set_name] = {'n_pairs': int(cat['P'].size), 'mean_personalised': mP,
                         'mean_same_word': float(cat['same'].mean()),
                         'mean_different_word': float(cat['diff'].mean()),
                         'mean_id_involving': float(cat['id'].mean()),
                         'share_same_word_of_mean': float(cat['same'].mean() / mP) if mP else float('nan'),
                         'frac_pairs_with_overlap': float(cat['overlap'].mean()),
                         'same_word_per_pair': _stats(cat['same'])}
    assert max_err < 1e-3, f'pair expansion does not reproduce the forward (rel err {max_err:.2e})'
    out['max_rel_err_vs_forward'] = max_err
    out['note'] = ('exact per pair; ids arms in the field-mean read-out (relative to that '
                   'convention), noid arms fixed by the scores' if ub.use_id_feature or ib.use_id_feature
                   else 'exact per pair; noid arm: fixed by the scores')
    return out


# ---------------------------------------------------------------------------
# C — basket twins (data only)
# ---------------------------------------------------------------------------

def instrument_twins(ib: FeatureEmbedding, train: pd.DataFrame) -> dict:
    Mi = item_code_weights(ib)
    sigs = [tuple(np.nonzero(row)[0].tolist()) for row in Mi.numpy()]
    sig_id = pd.Series(sigs).astype(str).factorize()[0]
    cat_counts = np.bincount(sig_id)
    df = train[['user_id', 'item_id']].drop_duplicates().copy()
    df['sig'] = sig_id[df['item_id'].to_numpy()]
    df['n_same_sig'] = df.groupby(['user_id', 'sig'])['item_id'].transform('size')
    return {'frac_train_positives_with_basket_twin': float((df['n_same_sig'] >= 2).mean()),
            'frac_catalogue_items_with_twin': float((cat_counts[sig_id] >= 2).mean()),
            'n_train_positives': int(len(df)), 'note': 'data only (word sets of the item composition)'}


# ---------------------------------------------------------------------------
# D — cross-prototype cosine
# ---------------------------------------------------------------------------

def instrument_cross_cos(user_proto, item_proto, seed: int, n_draws: int = 200) -> dict:
    Pu = _unit(user_proto.prototypes.detach().double())
    Pt = _unit(item_proto.prototypes.detach().double())
    C = Pu @ Pt.T
    Cuu, Ctt = Pu @ Pu.T, Pt @ Pt.T
    Cuu.fill_diagonal_(-np.inf); Ctt.fill_diagonal_(-np.inf)
    g = torch.Generator().manual_seed(seed)
    ref = []
    for _ in range(n_draws):
        Ru = _unit(torch.randn(Pu.shape, generator=g, dtype=torch.float64))
        Rt = _unit(torch.randn(Pt.shape, generator=g, dtype=torch.float64))
        ref.append((Ru @ Rt.T).max(1).values)
    return {'K_u': int(Pu.shape[0]), 'K_t': int(Pt.shape[0]), 'd': int(Pu.shape[1]),
            'user_proto_rowmax_vs_item_protos': _stats(C.max(1).values.numpy()),
            'item_proto_rowmax_vs_user_protos': _stats(C.max(0).values.numpy()),
            'within_user_protos_rowmax': _stats(Cuu.max(1).values.numpy()),
            'within_item_protos_rowmax': _stats(Ctt.max(1).values.numpy()),
            'random_reference_user_rowmax': _stats(torch.cat(ref).numpy()),
            'note': 'fixed by the scores; random reference = isotropic directions in R^d (the '
                    'trained space is lower-dimensional, so the reference is loose)'}


# ---------------------------------------------------------------------------
# E — item-side words-vs-ID variance split
# ---------------------------------------------------------------------------

def _split(Wp: torch.Tensor, Ip: torch.Tensor) -> dict:
    if Wp.dim() == 1:
        Wp, Ip = Wp[:, None], Ip[:, None]
    Wc, Ic = Wp - Wp.mean(0), Ip - Ip.mean(0)
    tw, ti, tc = float((Wc ** 2).sum()), float((Ic ** 2).sum()), float(2 * (Wc * Ic).sum())
    tot = tw + ti + tc
    return {'words': tw / tot, 'id': ti / tot, 'cross': tc / tot} if tot > 0 else {}


def instrument_item_split(model, ib: FeatureEmbedding, Q_i, foc, c_u=None) -> dict:
    if not ib.use_id_feature:
        return {'applies': False, 'note': 'no item-ID row'}
    T = ib.embedding_layer.weight.detach().double()
    V, n = ib.n_features, ib.n_objects
    ID = T[V + torch.arange(n)]
    words = Q_i - ID
    Mf = item_field_mass(ib, foc)
    beta = torch.linalg.pinv(Mf) @ words                    # least squares on the field masses
    w_res = words - Mf @ beta
    id_p = Q_i - w_res
    out = {'applies': True, 'q': _split(w_res, id_p), 'q_raw': _split(words, ID),
           'field_mass_constant': bool(torch.allclose(Mf, Mf[:1].expand_as(Mf))),
           'note': ('convention-free: the word part minus what the field masses predict vs the ID '
                    'part plus it (invariant to the translation on both layouts); *_raw = as weight '
                    'decay left it, for contrast')}
    lin = getattr(model.item_feature_extractor, 'linear_layer', None)
    if lin is not None:
        Wt = lin.weight.detach().double()
        out['t_hat'] = _split(w_res @ Wt.T, id_p @ Wt.T)
        out['t_hat_raw'] = _split(words @ Wt.T, ID @ Wt.T)
        if c_u:
            out['B_item'] = _split(c_u * (w_res @ Wt.T).sum(1), c_u * (id_p @ Wt.T).sum(1))
            out['B_item_raw'] = _split(c_u * (words @ Wt.T).sum(1), c_u * (ID @ Wt.T).sum(1))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _pair_sets(scores_fn, n_items, train, test, users, rng):
    seen = train.groupby('user_id')['item_id'].apply(lambda s: set(s.tolist())).to_dict()
    test_item = test.drop_duplicates('user_id').set_index('user_id')['item_id'].to_dict()
    top, tst, rnd = {}, {}, {}
    for u in users:
        s = scores_fn(u).clone()
        sv = seen.get(u, set())
        if sv:
            s[torch.as_tensor(sorted(sv))] = -np.inf
        top[u] = torch.topk(s, TOP_N).indices.tolist()
        if u in test_item:
            tst[u] = [int(test_item[u])]
        pool = np.setdiff1d(np.arange(n_items), np.fromiter(sv, dtype=np.int64, count=len(sv)))
        rnd[u] = rng.choice(pool, size=min(TOP_N, len(pool)), replace=False).tolist()
    return {'top10_unseen': top, 'test_item': tst, 'random10_unseen': rnd}


@torch.no_grad()
def run_on_model(model, data_path: str, fields=None, layout: str = 'fixed',
                 sample_users: int = 2000, seed: int = SEED) -> dict:
    model.eval()
    ufe, ife = model.user_feature_extractor, model.item_feature_extractor
    up, ip, kind = proto_layers(model)
    ib = item_composition(model)
    train = pd.read_csv(os.path.join(data_path, 'listening_history_train.csv'),
                        usecols=['user_id', 'item_id'])
    test = pd.read_csv(os.path.join(data_path, 'listening_history_test.csv'),
                       usecols=['user_id', 'item_id'])
    pop = np.bincount(train.drop_duplicates()['item_id'].to_numpy(), minlength=model.n_items)
    foc = n_fields = None
    if fields and ib is not None:
        foc = build_field_index(data_path, fields, layout)
        n_fields = len(fields)
    out = {'kind': kind, 'n_users': int(model.n_users), 'n_items': int(model.n_items)}
    if kind is not None:
        k_u = up.n_prototypes
        a_u, c_u, _ = cosine_affine(up)
        a_t, c_t, _ = cosine_affine(ip)
        aff = (a_u, c_u, a_t, c_t)
        Uo = _batched(ufe, model.n_users); Io = _batched(ife, model.n_items)
        _reset(ufe, ife)
        id_parts = Q_u = Q_i = None
        if kind == 'fufi':
            Q_u = _batched(ufe.base, model.n_users); Q_i = _batched(ife.base, model.n_items)
            if ufe.base.use_id_feature and foc is not None:
                id_parts = user_id_parts(model, Q_u, Uo, k_u, aff, foc, n_fields)
        out['A_common_mode'] = instrument_common(Uo, Io, k_u, aff, pop, id_parts)
        if kind == 'fufi' and foc is not None:
            rng = np.random.default_rng(seed)
            has_hist = np.flatnonzero(np.bincount(train['user_id'], minlength=model.n_users) > 0)
            users = np.sort(rng.choice(has_hist, size=min(sample_users, len(has_hist)), replace=False))
            sets = _pair_sets(lambda u: Uo[u] @ Io.T, model.n_items, train, test, users.tolist(), rng)
            out['B_word_pairs'] = instrument_pairs(model, Uo, Io, Q_u, Q_i, k_u, aff, foc, n_fields, sets)
            out['B_word_pairs']['n_sampled_users'] = int(len(users))
        if kind == 'fufi':
            out['D_cross_prototype_cosine'] = instrument_cross_cos(up, ip, seed)
            out['D_cross_prototype_cosine']['shared_table'] = bool(
                ufe.base.embedding_layer.weight is ife.base.embedding_layer.weight)
    if ib is not None:
        out['C_basket_twins'] = instrument_twins(ib, train)
        if foc is not None:
            Q_items = _batched(ib, model.n_items)
            out['E_item_words_vs_id'] = instrument_item_split(
                model, ib, Q_items, foc, c_u=cosine_affine(up)[1] if kind == 'fufi' else None)
    return out


def _fmt(x):
    return f'{x:+.4f}' if isinstance(x, float) else str(x)


def to_markdown(r: dict, title: str) -> str:
    L = [f'# SC.8 fUfI readout — {title}', '',
         '*Working evidence for SC.8 (F-DC01-06 restriction): not thesis numbers.*', '']
    A = r.get('A_common_mode')
    if A:
        L += ['## A. Common-mode decomposition (F-DC09-09)', '', A['note'], '',
              '| component | grand mean | sd item main | sd user main | sd interaction | common share | ρ(common, log pop) |',
              '|---|---|---|---|---|---|---|']
        for c in ('U_half', 'I_half', 'personalised', 'score'):
            t = A[c]
            L.append(f"| {c} | {_fmt(t['grand_mean'])} | {t['sd_item_main']:.4f} | {t['sd_user_main']:.4f} | "
                     f"{t['sd_interaction']:.4f} | {t['common_share']:.3f} | "
                     f"{t['spearman_common_part_vs_log_popularity']:+.3f} |")
        L += ['', f"B_item: mean {A['B_item']['mean']:+.4f}, sd {A['B_item']['sd']:.4f}, "
                  f"ρ(B_item, log pop) {A['B_item']['spearman_vs_log_popularity']:+.3f}", '']
        if 'user_id_line' in A:
            u = A['user_id_line']
            L += ['**User-ID line (3b-xii comparator).** ' + u['note'], '',
                  '| half | ID-line rank SS / half rank SS | ID line common share |', '|---|---|---|']
            for c in ('U_half', 'I_half', 'personalised'):
                L.append(f"| {c} | {u[c]['id_line_rank_ss_ratio']:.3f} | {u[c]['id_line_common_share']:.3f} |")
            L.append('')
    B = r.get('B_word_pairs')
    if B:
        L += ['## B. Same-word vs different-word pairs (F-DC09-10)', '', B['note'] +
              f" (max rel. err vs forward {B['max_rel_err_vs_forward']:.1e}; {B['n_sampled_users']} users).", '',
              '| pairs | n | mean personalised | same-word | different-word | ID-involving | same-word / mean | pairs with overlap |',
              '|---|---|---|---|---|---|---|---|']
        for s in ('top10_unseen', 'test_item', 'random10_unseen'):
            t = B[s]
            L.append(f"| {s} | {t['n_pairs']} | {_fmt(t['mean_personalised'])} | {_fmt(t['mean_same_word'])} | "
                     f"{_fmt(t['mean_different_word'])} | {_fmt(t['mean_id_involving'])} | "
                     f"{t['share_same_word_of_mean']:.3f} | {t['frac_pairs_with_overlap']:.3f} |")
        L.append('')
    C = r.get('C_basket_twins')
    if C:
        L += ['## C. Basket twins (F-DC09-10)', '',
              f"Train positives with a basket twin: **{C['frac_train_positives_with_basket_twin']:.3f}** "
              f"(of {C['n_train_positives']}); catalogue items with a twin: "
              f"{C['frac_catalogue_items_with_twin']:.3f}. {C['note']}.", '']
    D = r.get('D_cross_prototype_cosine')
    if D:
        L += ['## D. Cross-prototype cosine (F-DC09-11)', '',
              f"K_u {D['K_u']}, K_t {D['K_t']}, d {D['d']}, shared table: {D['shared_table']}. {D['note']}.", '',
              '| row max of | p5 | p25 | median | p75 | p95 |', '|---|---|---|---|---|---|']
        for k, lab in (('user_proto_rowmax_vs_item_protos', 'each user proto vs item protos'),
                       ('item_proto_rowmax_vs_user_protos', 'each item proto vs user protos'),
                       ('within_user_protos_rowmax', 'user protos vs each other'),
                       ('within_item_protos_rowmax', 'item protos vs each other'),
                       ('random_reference_user_rowmax', 'random reference')):
            s = D[k]
            L.append(f"| {lab} | {s['p5']:.3f} | {s['p25']:.3f} | {s['p50']:.3f} | {s['p75']:.3f} | {s['p95']:.3f} |")
        L.append('')
    E = r.get('E_item_words_vs_id')
    if E:
        L += ['## E. Item-side words vs ID, variance split (F-DC09-21 ii)', '']
        if not E['applies']:
            L += [E['note'], '']
        else:
            L += [E['note'] + f"; field masses constant across items: {E['field_mass_constant']}.", '',
                  '| quantity | words | ID | 2·cov |', '|---|---|---|---|']
            for k in ('q', 'q_raw', 't_hat', 't_hat_raw', 'B_item', 'B_item_raw'):
                if k in E and E[k]:
                    L.append(f"| {k} | {E[k]['words']:.3f} | {E[k]['id']:.3f} | {E[k]['cross']:+.3f} |")
            L.append('')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--sample-users', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    from utilities.cold_eval import build_model, load_checkpoint
    from utilities.consts import DATA_PATH
    with open(os.path.join(args.results_dir, 'config.json')) as f:
        conf = json.load(f)
    dataset = args.dataset or os.path.basename(conf['data_path'].rstrip('/'))
    data_path = dataset if os.path.isabs(dataset) else os.path.join(DATA_PATH, dataset)
    model, _, _ = build_model(conf, data_path, 'cpu')
    load_checkpoint(model, args.results_dir, 'cpu')
    item_conf = conf['ft_ext_param'].get('item_ft_ext_param', {})
    r = run_on_model(model, data_path, item_conf.get('feature_fields'),
                     item_conf.get('feature_layout', 'fixed'), args.sample_users, args.seed)
    r['results_dir'] = os.path.abspath(args.results_dir)
    out_dir = args.out_dir or os.path.join(args.results_dir, 'sc8_fufi_readout')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'readout.json'), 'w') as f:
        json.dump(r, f, indent=2)
    with open(os.path.join(out_dir, 'readout.md'), 'w') as f:
        f.write(to_markdown(r, os.path.basename(args.results_dir.rstrip('/'))))
    print(f'readout written to {out_dir}')


if __name__ == '__main__':
    main()
