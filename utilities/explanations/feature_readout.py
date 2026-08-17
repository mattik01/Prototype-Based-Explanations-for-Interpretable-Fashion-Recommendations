"""dc01 intrinsic per-feature read-out for ``feature_item_proto``.

Every quantity here is a function of the model's *own* parameters / forward-pass tensors — nothing
is recomputed post-hoc. Two read-outs (dc01 §3.4):

1. ``per_feature_shares`` — the EXACT additive decomposition of each item-prototype activation
   ``t*_k`` over the item's composing feature rows. With ``q_i = Σ_r e_r``, each row contributes
   ``c_{r,k} = e_r·p_k / (‖q_i‖‖p_k‖)`` and ``Σ_r c_{r,k} = cos(q_i, p_k)`` exactly — under the
   shifted default (``t*_k = 1 + cos``) that is ``t*_k − 1``; in general (cosine-offset
   generalization, cosbias2x2) ``t*_k = a·cos + c`` and the shares cover ``(t*_k − c)/a``.
   Honesty notes (rendered, never hidden): the offset ``c`` is a feature-independent baseline
   that the shares do NOT cover (absent entirely under 'standard'), and the shares are JOINTLY
   normalized through ``‖q_i‖`` — they are exact additive shares of the cosine part, not
   counterfactual effects (removing a row rescales the others).

2. ``global_prototype_profile`` — a prototype's learned attribute profile: ``cos(e_f, p_k)`` for
   every feature-value row in the vocabulary. Read from parameters alone, this is the structured
   signal an LLM-naming layer (dc01 S4) would consume in place of post-hoc top-k item profiles.

Integration into the run_combo explanations pipeline (and adding feature_item_proto to
EXPLAINABLE_MODELS) is the deferred P9 — these functions are standalone and unit-tested for now.
"""
import torch


def per_feature_shares(feature_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Exact additive shares of each item-prototype activation over the item's composing rows.

    :param feature_embeddings: (R, d) the embedding rows composing the item (their sum is q_i);
        include the per-item ID row iff the model used ``use_id_feature=True``.
    :param prototypes: (K, d) the item prototypes p_k.
    :return: c (R, K) with ``c[r, k] = e_r·p_k / (‖q_i‖‖p_k‖)``; ``c.sum(dim=0) == cos(q_i, p_k)``
        exactly (= ``t*_k − 1`` under the shifted default; ``(t*_k − c)/a`` in general).
    """
    assert feature_embeddings.dim() == 2 and prototypes.dim() == 2, "expect (R,d) and (K,d)"
    q = feature_embeddings.sum(dim=0)                    # (d,)
    q_norm = q.norm()
    p_norm = prototypes.norm(dim=1)                      # (K,)
    return (feature_embeddings @ prototypes.T) / (q_norm * p_norm)   # (R, K)


def activation_from_shares(shares: torch.Tensor, scale: float = 1.0,
                           offset: float = 1.0) -> torch.Tensor:
    """Reconstruct the activations ``t*_k = offset + scale·Σ_r c_{r,k}`` from the shares.

    Defaults reproduce the shifted cosine (``1 + Σ_r c_{r,k}``); pass the run's own affine
    (scale a, offset c) for other cosine_types — cosine-offset generalization, 2026-08-17
    special experiments (cosbias2x2). The offset is added back explicitly so callers never
    silently absorb the baseline.
    :param shares: (R, K) from :func:`per_feature_shares`.
    :return: (K,) the activations t*_k.
    """
    return offset + scale * shares.sum(dim=0)


def global_prototype_profile(emb_table: torch.Tensor, prototypes: torch.Tensor,
                             eps: float = 1e-8) -> torch.Tensor:
    """Per-prototype attribute profile: ``cos(e_f, p_k)`` over the feature-value vocabulary.

    :param emb_table: (V, d) the feature-value embedding rows (e.g. ``embedding_layer.weight[:n_features]``).
    :param prototypes: (K, d).
    :param eps: floor on row norms so zero/cold rows give a cosine of 0 instead of NaN.
    :return: (V, K) cosine similarities in [-1, 1].
    """
    assert emb_table.dim() == 2 and prototypes.dim() == 2, "expect (V,d) and (K,d)"
    e = emb_table / emb_table.norm(dim=1, keepdim=True).clamp_min(eps)
    p = prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(eps)
    return e @ p.T


def item_meta_codes(feature_embedding, item_idx: int) -> list:
    """The REAL metadata codes composing item ``item_idx`` — fixed layout: the full row;
    bags layout: padding slots (weight 0.0) filtered out (F-DC05-13). Row order matches
    :func:`item_feature_rows`' metadata rows, so codes and rows zip 1:1."""
    codes = feature_embedding.feature_ids[item_idx]
    if getattr(feature_embedding, 'feature_weights', None) is not None:
        codes = codes[feature_embedding.feature_weights[item_idx] > 0]
    return codes.tolist()


def item_feature_rows(feature_embedding, item_idx: int) -> torch.Tensor:
    """Gather the embedding rows that compose a single item's representation q_i.

    Mirrors ``FeatureEmbedding.forward`` for one item (metadata rows, plus the ID row iff
    ``use_id_feature``), returning the (R, d) rows so they can be fed to :func:`per_feature_shares`.
    Bags layout (F-DC05-13): only REAL slots enter (padding weight 0.0 filtered — under weights
    ∈ {0, 1} filtering equals the forward's multiply), so the rows still sum to q_i exactly and
    align 1:1 with :func:`item_meta_codes`.

    :param feature_embedding: a built ``feature_extraction.feature_extractors.FeatureEmbedding``.
    :param item_idx: the item index.
    :return: (R, d) the composing embedding rows (no grad).
    """
    with torch.no_grad():
        codes = feature_embedding.feature_ids[item_idx]
        weights = None
        if getattr(feature_embedding, 'feature_weights', None) is not None:
            w = feature_embedding.feature_weights[item_idx]
            real = w > 0
            codes, weights = codes[real], w[real]
        rows_idx = codes.tolist()
        if feature_embedding.use_id_feature:
            rows_idx = rows_idx + [feature_embedding.n_features + item_idx]
        idx = torch.tensor(rows_idx, dtype=torch.long,
                           device=feature_embedding.embedding_layer.weight.device)
        rows = feature_embedding.embedding_layer(idx)
        if weights is not None:
            # mirror the forward's weighting exactly (real weights are 1.0 today; kept
            # general so the identity Σ rows == q_i holds under any weight scheme)
            w_full = torch.cat([weights, weights.new_ones(1)]) \
                if feature_embedding.use_id_feature else weights
            rows = rows * w_full.unsqueeze(-1)
        return rows
