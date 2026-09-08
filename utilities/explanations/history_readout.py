"""dc05 intrinsic read-outs for ``feature_user_proto`` (fU-ProtoMF).

Every quantity here is a function of the model's *own* parameters / forward-pass tensors —
nothing is recomputed post-hoc. The user-prototype activation decomposes exactly (design doc
§3.4, 4b C1′/C2′): with ``q_u = Σ_j w̄_j·e_j (+ e_ID)`` the rows sum to q_u, the shares sum to
``cos(q_u, p_l)`` (under the shifted default ``u*_l = 1 + cos`` that is ``u*_l − 1``; in
general ``u*_l = a·cos + c`` — the breakdown renderer applies the run's own affine,
cosine-offset generalization cosbias2x2), and the SAME sum regroups exactly two ways:

1. **per basket word** — rows ``w̄_j·e_j`` (+ the ID row); shares via
   :func:`feature_readout.per_feature_shares` (generic: any rows summing to q).
2. **per purchase** — rows ``(1/|H_u|)·Σ_{f∈f_i} e_f`` per purchased item i (+ the ID row);
   the two readings are alternative zooms of ONE sum and are never added together.

Honesty notes carried from dc01 (rendered, never hidden): the offset's mass is
``B(t) = c·1ᵀt`` on this host — a per-ITEM scalar that is NOT rank-inert (design doc §3.4;
absent entirely when c = 0, i.e. cosine_type='standard');
shares are jointly normalized through ``‖q_u‖`` (exact summands, not counterfactual effects).
dc07 (2026-09-08): none of the above applies to the membership family (cosine_type softmax /
sigmoid) — there the exact additive unit is the log-odds between two prototypes, not a share
of the activation; the affine renderer refuses those runs (breakdown.cosine_affine).
"""
import os

import pandas as pd
import torch


def user_history_rows(hist_embed, user_idx: int):
    """Gather the weighted embedding rows composing ONE user's representation q_u.

    Mirrors ``HistoryFeatureEmbedding.forward`` for one user: the non-padding basket slots'
    rows ``w̄_j·e_j`` (weights applied, so the rows SUM to the metadata part), plus the ID row
    iff ``use_id_feature``. Row sum == q_u exactly.

    :param hist_embed: a built ``HistoryFeatureEmbedding``.
    :param user_idx: the user index.
    :return: ``(rows (R, d), word_codes list[int], has_id bool)`` — rows[j] belongs to global
        word code word_codes[j]; the ID row (when present) is rows[-1] and has no code.
    """
    with torch.no_grad():
        ids = hist_embed.hist_value_ids[user_idx]           # (D_max,)
        w = hist_embed.hist_weights[user_idx]               # (D_max,)
        nz = w != 0
        codes = ids[nz]
        rows = hist_embed.embedding_layer(codes) * w[nz].unsqueeze(-1)   # (R_meta, d)
        if hist_embed.use_id_feature:
            id_idx = torch.tensor([hist_embed.n_features + user_idx], dtype=torch.long,
                                  device=hist_embed.embedding_layer.weight.device)
            rows = torch.cat([rows, hist_embed.embedding_layer(id_idx)], dim=0)
    return rows, codes.tolist(), hist_embed.use_id_feature


def per_purchase_rows(hist_embed, user_idx: int, purchased_item_ids,
                      feature_ids: torch.LongTensor,
                      feature_weights: torch.Tensor = None):
    """The per-PURCHASE regrouping of the same composition (4b C2′): one row per purchased
    item, ``r_i = (1/|H_u|)·Σ_{f∈f_i} e_f``, plus the ID row iff ``use_id_feature``. The rows
    SUM to q_u exactly (same summands as :func:`user_history_rows`, grouped by purchase), so
    they feed :func:`feature_readout.per_feature_shares` unchanged.

    :param hist_embed: a built ``HistoryFeatureEmbedding``.
    :param user_idx: the user index (for the ID row).
    :param purchased_item_ids: the user's deduplicated train-window item ids (H_u) — use
        :func:`user_train_items` so the set matches the builder's exactly.
    :param feature_ids: per-item global-code tensor on the SAME split dir + field set the
        model was built with — fixed layout: ``(n_items, F)`` from ``build_feature_ids``;
        bags layout: the padded ``(n_items, D_max)`` ids from ``build_feature_bags`` (pass
        ``feature_weights`` alongside, F-DC05-13).
    :param feature_weights: bags layout only — the matching ``(n_items, D_max)`` weight
        tensor (1.0 real / 0.0 padding) from ``build_feature_bags``; padding then contributes
        exactly nothing, mirroring the builder's aggregation
        (``r_i = (1/|H_u|)·Σ_{tokens of i} e_t``). None (default) = fixed layout, unchanged.
    :return: ``(rows (P(+1), d), has_id bool)`` — rows[i] belongs to purchased_item_ids[i];
        the ID row (when present) is rows[-1]. NOTE: this regroups the METADATA mass only per
        purchase — the ID row stays its own line, exactly as in the word-level reading.
    """
    if len(purchased_item_ids) == 0:
        raise ValueError('per_purchase_rows needs a non-empty purchase set (H_u); an empty '
                         'basket has no per-purchase reading (q_u = e_ID or 0 — 4b C3)')
    with torch.no_grad():
        device = hist_embed.embedding_layer.weight.device
        items_t = torch.as_tensor(list(purchased_item_ids), dtype=torch.long)
        codes = feature_ids[items_t]  # (P, F) fixed | (P, D_max) bags
        emb = hist_embed.embedding_layer(codes.to(device))  # (P, F|D_max, d)
        if feature_weights is not None:
            w = feature_weights[items_t].to(device)  # (P, D_max) — 0.0 padding kills its rows
            rows = (emb * w.unsqueeze(-1)).sum(dim=1)  # (P, d)
        else:
            rows = emb.sum(dim=1)  # (P, d)
        rows = rows / float(len(purchased_item_ids))
        if hist_embed.use_id_feature:
            id_idx = torch.tensor([hist_embed.n_features + user_idx], dtype=torch.long,
                                  device=device)
            rows = torch.cat([rows, hist_embed.embedding_layer(id_idx)], dim=0)
    return rows, hist_embed.use_id_feature


def user_train_items(dataset_dir: str, user_id: int):
    """The user's deduplicated train-window purchase set H_u, exactly as
    ``build_user_history_weights`` counts it (dedup on (user_id, item_id) keep-first).

    :param dataset_dir: the split directory (its OWN train file — leakage rule).
    :param user_id: the user index.
    :return: list[int] item ids, in first-purchase order.
    """
    df = pd.read_csv(os.path.join(dataset_dir, 'listening_history_train.csv'),
                     usecols=['user_id', 'item_id'])
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    return df.loc[df['user_id'] == user_id, 'item_id'].astype(int).tolist()
