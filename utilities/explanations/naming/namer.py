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


def name_prototypes_intrinsic(
    feature_value_embeddings: np.ndarray,   # (n_features, d) — the learned e_f rows
    prototypes: np.ndarray,                 # (n_protos, d)
    feature_fields: List[str],              # model's item feature fields, in feature_ids code order
    items_info: pd.DataFrame,
    cfg: NamingConfig,
    side: str = "item",
    eps: float = 1e-8,
) -> NamingResult:
    """Intrinsic-source entrypoint (dc01): score every (feature-value, prototype) pair by
    ``cos(e_f, p_k)`` read straight from the learned parameters, and populate the SAME
    ProtoFeatureProfile / FeatureValueStat structures the post-hoc source uses — so the namer
    and every plot work unchanged. ``score`` carries the cosine; the frequency/lift/count fields
    are 0 (they have no meaning intrinsically), so use a cfg with ``scoring != 'lift'`` and a
    cosine ``min_score`` floor.

    The (feature-value -> code) order is reconstructed deterministically from ``feature_fields``
    + ``items_info`` exactly as ``feature_extraction.feature_ids.build_feature_ids`` builds it
    (per field: lexicographically sorted unique values, concatenated with running offsets), so
    row f of ``feature_value_embeddings`` corresponds to the f-th (column, value) below.
    """
    e = feature_value_embeddings
    p = prototypes
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + eps)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + eps)
    cos = e @ p.T  # (n_features, n_protos)

    code_to_cv: List = []
    for field in feature_fields:
        for value in sorted(items_info[field].astype(str).unique().tolist()):
            code_to_cv.append((field, value))
    if len(code_to_cv) != cos.shape[0]:
        raise ValueError(
            f"intrinsic naming: reconstructed feature vocab size {len(code_to_cv)} != "
            f"feature_value_embeddings rows {cos.shape[0]} — feature_fields/items_info mismatch")

    names: Dict[int, PrototypeName] = {}
    profiles: List[ProtoFeatureProfile] = []
    n_protos = cos.shape[1]
    for k in range(n_protos):
        stats = [
            FeatureValueStat(column=col, value=val, count=0, topk_freq=0.0,
                             global_freq=0.0, lift=0.0, score=float(cos[f, k]))
            for f, (col, val) in enumerate(code_to_cv)
        ]
        profile = ProtoFeatureProfile(proto_idx=k, side=side, topk_item_ids=[], topk_size=0, stats=stats)
        profiles.append(profile)
        names[k] = name_from_profile(profile, cfg)

    return NamingResult(side=side, names=names, profiles=profiles, config=cfg)
