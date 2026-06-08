"""Turn ProtoFeatureProfiles into human-readable names + a transparent result object.

The namer is the second, kind-agnostic half of the pipeline: it filters and ranks the
(feature, value) statistics of a profile and formats the survivors into a length-bounded
string. It keeps the chosen descriptors so every name is fully reconstructable from the
dumped stats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from utilities.explanations.naming.config import NamingConfig
from utilities.explanations.naming.describers import FeatureValueStat
from utilities.explanations.naming.profile import (
    ProtoFeatureProfile,
    build_profile,
    prepare_features,
    top_k_ids_for_prototype,
)


@dataclass
class PrototypeName:
    proto_idx: int
    side: str
    name: str
    descriptors: List[FeatureValueStat]   # chosen, in display order


@dataclass
class NamingResult:
    side: str                              # "item" | "user"
    names: Dict[int, PrototypeName]
    profiles: List[ProtoFeatureProfile]
    config: NamingConfig

    def labels_in_order(self, n: int) -> List[str]:
        """Names indexed 0..n-1 (empty string for any gap), for plot annotation."""
        return [self.names[i].name if i in self.names else "" for i in range(n)]


def name_from_profile(profile: ProtoFeatureProfile, cfg: NamingConfig) -> PrototypeName:
    def keep(s: FeatureValueStat) -> bool:
        if s.count < cfg.min_count:
            return False
        if cfg.scoring == "lift" and s.lift < cfg.min_lift:
            return False
        return s.score >= cfg.min_score

    candidates = sorted(
        (s for s in profile.stats if keep(s)),
        key=lambda s: (-s.score, -s.count, s.column, s.value),
    )

    chosen: List[FeatureValueStat] = []
    per_column: Dict[str, int] = {}
    for s in candidates:
        if cfg.max_per_column is not None and per_column.get(s.column, 0) >= cfg.max_per_column:
            continue
        chosen.append(s)
        per_column[s.column] = per_column.get(s.column, 0) + 1
        if len(chosen) >= cfg.max_descriptors:
            break

    if chosen:
        parts = [cfg.descriptor_format.format(value=s.value, column=s.column) for s in chosen]
        name = cfg.separator.join(parts)
    else:
        name = cfg.fallback_name.format(idx=profile.proto_idx)

    return PrototypeName(profile.proto_idx, profile.side, name, chosen)


def name_prototypes_from_weights(
    item_weights: np.ndarray,
    items_info: pd.DataFrame,
    cfg: NamingConfig,
    side: str,
) -> NamingResult:
    """Derived-source entrypoint: name every prototype from a (n_items, n_protos) weight
    matrix (item->item-proto similarity, or items-in-user-proto-space projections)."""
    items_info_indexed = items_info.set_index(cfg.id_column)
    prepared = prepare_features(items_info, cfg)

    names: Dict[int, PrototypeName] = {}
    profiles: List[ProtoFeatureProfile] = []
    n_protos = item_weights.shape[1]
    for p in range(n_protos):
        topk_ids = top_k_ids_for_prototype(item_weights, p, cfg.top_k_for_naming)
        profile = build_profile(p, side, topk_ids, items_info_indexed, prepared, cfg)
        profiles.append(profile)
        names[p] = name_from_profile(profile, cfg)

    return NamingResult(side=side, names=names, profiles=profiles, config=cfg)
