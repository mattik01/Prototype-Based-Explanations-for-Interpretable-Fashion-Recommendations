"""dc02 (attr_item_proto) intrinsic read-out — exact, forward-pass-faithful quantities.

Everything here is either a model parameter read directly (A rows, W_t rows) or exact
arithmetic on forward-pass quantities — nothing is reconstructed post-hoc (design doc §3.4).

Conventions (all torch, float32):
- ``A``           : effective prototype matrix (K, V) — softplus(Ã) when K1 is on; callers must
                    pass ``AttributePrototypeEmbedding.effective_prototypes()`` /
                    ``AttrItemProtoAccessor.item_prototypes()``, never the raw parameter.
- ``x_row``       : one item's multi-hot (V,) with exactly F ones (‖x‖₂ = √F — the property
                    behind the EXACT decomposition; guaranteed by build_attr_multi_hot).
- ``field_offsets``: cumulative block bounds, length F+1.

The §3.4 amendment (post-5b critique pt. 4): the shifted cosine's +1 injects a USER-CONSTANT
Σ_k û_k into every item's score — constant across items for a given user, hence
ranking-irrelevant, but misleading if rendered as part of an item's "activation contribution".
Rendered explanations therefore report the ITEM-DISCRIMINATING contribution û_k·(t*_k − 1),
which the per-attribute shares decompose exactly and completely, with the user constant
disclosed once (``user_constant``), not folded into per-item numbers.
"""
import math
from typing import List, Tuple

import torch

_EPS = 1e-8


def per_attribute_shares(x_row: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Exact additive per-attribute shares of an item's prototype activations.

    share[f, k] = A[k, v_f] / (√F · ‖a_k‖) for the F attribute values v_f set in ``x_row``.
    Because ‖x_row‖₂ = √F exactly, ``shares.sum(dim=0) == t*_k − 1`` to float precision
    (claim 3 — verified in t08 against the module's own forward output).

    :param x_row: (V,) multi-hot of ONE item.
    :param prototypes: effective A (K, V).
    :return: (F, K) shares, rows ordered by ascending value index (= field order).
    """
    v_idx = (x_row > 0).nonzero(as_tuple=False).squeeze(1)          # (F,)
    n_fields = int(v_idx.shape[0])
    norms = prototypes.norm(dim=1).clamp_min(_EPS)                   # (K,)
    return prototypes[:, v_idx].T / (math.sqrt(n_fields) * norms)    # (F, K)


def activation_from_shares(shares: torch.Tensor) -> torch.Tensor:
    """Reassemble t* from shares: t*_k = 1 + Σ_f share[f, k]. Exact inverse of the decomposition."""
    return 1.0 + shares.sum(dim=0)


def item_discriminating_contributions(u_hat: torch.Tensor, t_star: torch.Tensor) -> torch.Tensor:
    """The rendered per-prototype quantity û_k·(t*_k − 1) (§3.4 amendment): the part of the
    ûᵀt* score half that actually discriminates between items for this user."""
    return u_hat * (t_star - 1.0)


def user_constant(u_hat: torch.Tensor) -> torch.Tensor:
    """The ranking-irrelevant baseline Σ_k û_k the shifted cosine adds to EVERY item's score
    for this user. Disclosed once per explanation, never folded into per-item numbers."""
    return u_hat.sum()


def score_decomposition(u_star: torch.Tensor, u_hat: torch.Tensor,
                        t_hat: torch.Tensor, t_star: torch.Tensor) -> dict:
    """Exact decomposition of one (user, item) score into its L_u + K + 1 addends (claim 4 +
    §3.4 amendment). All inputs are 1-dim vectors for a single user/item pair.

    total == u*ᵀt̂ + ûᵀt* == Σ_l proj_terms + Σ_k proto_terms_discriminating + user_constant,
    reproducing ``RecSys.forward``'s logit exactly (verified in t08).
    """
    proj_terms = u_star * t_hat                                    # (L_u,) — u*_l · t̂_l
    proto_terms = item_discriminating_contributions(u_hat, t_star)  # (K,)  — û_k · (t*_k − 1)
    const = user_constant(u_hat)                                    # scalar — Σ_k û_k
    total = proj_terms.sum() + proto_terms.sum() + const
    return {
        'proj_terms': proj_terms,
        'proto_terms_discriminating': proto_terms,
        'user_constant': const,
        'total': total,
    }


def prototype_attr_profile(prototypes: torch.Tensor, field_offsets: List[int],
                           code_to_field_value: List[Tuple[str, str]],
                           top_m: int = 3) -> List[List[dict]]:
    """Render each prototype as its top-weight attribute values PER FIELD — the "prototype IS a
    readable attribute profile" read-out (§3.4 pt. 2). Reads A directly; no naming procedure
    is needed to *ground* the prototype (verbalisation on top is the naming layer's job).

    :return: profiles[k][f] = {'field', 'top': [(value, weight), ...]} (top_m per field,
        by descending weight).
    """
    profiles = []
    for k in range(prototypes.shape[0]):
        row = prototypes[k]
        fields = []
        for f in range(len(field_offsets) - 1):
            lo, hi = field_offsets[f], field_offsets[f + 1]
            block = row[lo:hi]
            m = min(top_m, hi - lo)
            top_w, top_i = torch.topk(block, m)
            field_name = code_to_field_value[lo][0]
            fields.append({
                'field': field_name,
                'top': [(code_to_field_value[lo + int(i)][1], float(w))
                        for w, i in zip(top_w, top_i)],
            })
        profiles.append(fields)
    return profiles


def per_field_mass(prototypes: torch.Tensor, field_offsets: List[int]) -> torch.Tensor:
    """Per-prototype fraction of squared-L2 weight mass per field — the free single-field-
    collapse diagnostic (§3.5 mode (c)): a row concentrating ~all mass in one field block means
    that prototype differentiates on internal taxonomy only.

    :return: (K, F), rows sum to 1.
    """
    masses = []
    for f in range(len(field_offsets) - 1):
        block = prototypes[:, field_offsets[f]:field_offsets[f + 1]]
        masses.append((block ** 2).sum(dim=1))
    mass = torch.stack(masses, dim=1)                               # (K, F)
    return mass / mass.sum(dim=1, keepdim=True).clamp_min(_EPS)
