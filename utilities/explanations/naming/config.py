"""Configuration for prototype naming: which features, how to score, how to format.

A :class:`NamingConfig` is pure data (no model logic), resolved per-dataset from
``NAMING_DEFAULTS`` with a generic fallback that infers categorical features from the
``items_info`` columns. Because it is decoupled from *how* a prototype's feature
association is obtained, the same config drives the derived (top-k) namer today and the
future intrinsic namer unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class FeatureSpec:
    """One item-feature column and how to read its values."""
    column: str
    kind: str = "categorical"               # "categorical" | "numeric"
    multi_value_sep: Optional[str] = None   # e.g. "|" for ml-1m genres; None = single value
    # --- numeric seam (only consulted when kind == "numeric") ---
    bin_quantiles: Optional[List[float]] = None   # e.g. [0, .25, .5, .75, 1.0]
    bin_labels: Optional[List[str]] = None        # e.g. ["cheap", "average", "expensive", "luxury"]


@dataclass
class NamingConfig:
    """All knobs for turning a prototype's top-k items into a name."""
    features: List[FeatureSpec] = field(default_factory=list)
    id_column: str = "item_id"

    # --- selection / scoring ---
    normalize: str = "lift"        # "lift" -> topk_freq / global_freq ; "raw" -> topk_freq
    top_k_for_naming: int = 20     # how many top items define a prototype (may differ from display top_k)
    min_count: int = 2             # a descriptor needs >= this many supporting top-k items
    min_lift: float = 1.0          # in lift mode, keep only values more frequent than global
    min_score: float = 0.0         # dynamic-length floor on the selection score

    # --- formatting ---
    max_descriptors: int = 3           # primary name-length knob
    max_per_column: Optional[int] = None
    descriptor_format: str = "{value}"  # or "{column}={value}"
    separator: str = ", "
    fallback_name: str = "proto {idx}"  # used when nothing passes the filters


# Curated feature order for H&M (most semantically distinctive first). lift-based
# selection + max_descriptors keeps the final name short regardless of list length.
_HM_FEATURES: List[FeatureSpec] = [
    FeatureSpec("product_type_name"),
    FeatureSpec("product_group_name"),
    FeatureSpec("colour_group_name"),
    FeatureSpec("perceived_colour_master_name"),
    FeatureSpec("graphical_appearance_name"),
    FeatureSpec("garment_group_name"),
    FeatureSpec("index_group_name"),
    FeatureSpec("section_name"),
    FeatureSpec("department_name"),
]

NAMING_DEFAULTS: Dict[str, NamingConfig] = {
    # ml-1m: genres is the only semantic categorical column; it is pipe-separated.
    # (year is numeric -> future seam; title is free text -> excluded.)
    "ml-1m": NamingConfig(features=[FeatureSpec("genres", multi_value_sep="|")]),
    "hm_1_month": NamingConfig(features=list(_HM_FEATURES)),
    "hm_3_month": NamingConfig(features=list(_HM_FEATURES)),
    "hm_full": NamingConfig(features=list(_HM_FEATURES)),
}

# Columns never useful as descriptors regardless of dataset (ids / free text).
_GENERIC_EXCLUDE = {"item", "title"}


def _infer_config(items_info: Optional[pd.DataFrame]) -> NamingConfig:
    """Generic fallback for an unregistered dataset: treat every non-id, non-text column
    as categorical, auto-detecting pipe-separated multi-value columns."""
    if items_info is None:
        return NamingConfig(features=[])
    specs: List[FeatureSpec] = []
    for col in items_info.columns:
        if col in _GENERIC_EXCLUDE or col == "item_id":
            continue
        sample = items_info[col].dropna().astype(str)
        sep = "|" if sample.str.contains(r"\|").any() else None
        specs.append(FeatureSpec(col, multi_value_sep=sep))
    return NamingConfig(features=specs)


def get_naming_config(
    dataset: str,
    items_info: Optional[pd.DataFrame] = None,
    overrides: Optional[dict] = None,
) -> NamingConfig:
    """Resolve the naming config for a dataset.

    :param dataset: dataset key (e.g. "ml-1m", "hm_1_month")
    :param items_info: used to (a) drive the generic fallback and (b) drop configured
        features that are absent from this dataset's columns.
    :param overrides: shallow overrides of scalar knobs (e.g. {"top_k_for_naming": 50}).
    """
    cfg = NAMING_DEFAULTS.get(dataset)
    if cfg is None:
        cfg = _infer_config(items_info)

    if items_info is not None:
        present = set(items_info.columns)
        cfg = replace(cfg, features=[f for f in cfg.features if f.column in present])

    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
