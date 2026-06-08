"""Build a ProtoFeatureProfile — the stable intermediate between a feature *source*
and the namer.

A profile is, for one prototype, the full list of (feature, value) statistics over its
top-k representative items. The derived source here computes top-k from a prototype
weight matrix; a future intrinsic source would populate the same structure straight from
learned per-prototype feature weights, leaving the namer and plots untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from utilities.explanations.naming.config import NamingConfig
from utilities.explanations.naming.describers import (
    DESCRIBERS,
    FeatureValueStat,
    PreparedFeature,
)


@dataclass
class ProtoFeatureProfile:
    proto_idx: int
    side: str                       # "item" | "user"
    topk_item_ids: List[int]
    topk_size: int
    stats: List[FeatureValueStat]   # every (feature, value) seen in the top-k set


def prepare_features(items_info: pd.DataFrame, cfg: NamingConfig) -> List[PreparedFeature]:
    """One global pass per configured feature that exists in items_info."""
    prepared: List[PreparedFeature] = []
    for spec in cfg.features:
        if spec.column not in items_info.columns:
            continue
        describer = DESCRIBERS[spec.kind]
        prepared.append(describer.prepare(items_info, spec))
    return prepared


def build_profile(
    proto_idx: int,
    side: str,
    topk_item_ids: List[int],
    items_info_indexed: pd.DataFrame,
    prepared: List[PreparedFeature],
    cfg: NamingConfig,
) -> ProtoFeatureProfile:
    """Aggregate one prototype's top-k items into per-(feature, value) statistics."""
    rows = items_info_indexed.loc[topk_item_ids]
    stats: List[FeatureValueStat] = []
    for pf in prepared:
        counts, denom = pf.subset_counts(rows)
        for value, count in counts.items():
            global_count = pf.global_counts.get(value, 0)
            global_freq = global_count / pf.global_denom if pf.global_denom else 0.0
            topk_freq = count / denom if denom else 0.0
            # top-k is a subset of all items, so global_freq > 0 whenever count > 0.
            lift = topk_freq / global_freq if global_freq > 0 else 0.0
            score = lift if cfg.scoring == "lift" else topk_freq
            stats.append(
                FeatureValueStat(
                    column=pf.spec.column,
                    value=value,
                    count=count,
                    topk_freq=topk_freq,
                    global_freq=global_freq,
                    lift=lift,
                    score=score,
                )
            )
    return ProtoFeatureProfile(
        proto_idx=proto_idx,
        side=side,
        topk_item_ids=list(topk_item_ids),
        topk_size=len(rows),
        stats=stats,
    )


def top_k_ids_for_prototype(item_weights: np.ndarray, proto_idx: int, k: int) -> List[int]:
    """Row indices of the k items with the highest weight on this prototype.

    Row index equals item_id by construction (embeddings are ordered by remapped id),
    matching utilities.explanations_utils.get_top_k_items.
    """
    order = np.argsort(-item_weights[:, proto_idx])[:k]
    return order.tolist()
