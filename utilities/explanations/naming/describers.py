"""Feature describers: extract per-item "tokens" from a feature column and count them.

A describer is the only component that knows how a given *kind* of feature is read.
Everything downstream (lift scoring, name formatting, plotting) is kind-agnostic and
operates on the resulting counts.

Flow per feature:
    describer.prepare(items_info, spec)  -> PreparedFeature   (one global pass)
    prepared.subset_counts(topk_rows)    -> (counts, denom)   (per prototype)

``prepare`` does the whole-universe pass once (and, for numeric features, fixes the
quantile bin edges) so that a prototype's top-k counts are scored against stable global
statistics computed with the *same* tokenization.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureValueStat:
    """Transparent record for one (feature, value) in one prototype's top-k set."""
    column: str
    value: str
    count: int          # # top-k items carrying this value
    topk_freq: float    # count / top-k size
    global_freq: float  # fraction of all items carrying this value
    lift: float         # topk_freq / global_freq
    score: float        # selection score (= lift or topk_freq, per NamingConfig.scoring)


@dataclass
class PreparedFeature:
    """A feature after its one global pass; knows how to count any item subset."""
    spec: "object"                       # FeatureSpec (avoid import cycle with config)
    describer: "FeatureDescriber"
    global_counts: Dict[str, int]
    global_denom: int
    state: dict = field(default_factory=dict)   # describer-private (e.g. numeric bin edges)

    def subset_counts(self, rows: pd.DataFrame) -> Tuple[Dict[str, int], int]:
        counts: Dict[str, int] = {}
        for token_set in self.describer.tokens(rows[self.spec.column], self):
            for tok in token_set:
                counts[tok] = counts.get(tok, 0) + 1
        return counts, len(rows)


class FeatureDescriber(ABC):
    kind: str = ""

    @abstractmethod
    def tokens(self, series: pd.Series, prepared: PreparedFeature) -> Iterable[Set[str]]:
        """Yield, per row in ``series``, the set of value-tokens that row carries."""

    def prepare(self, items_info: pd.DataFrame, spec) -> PreparedFeature:
        prepared = PreparedFeature(spec=spec, describer=self, global_counts={}, global_denom=0)
        self._fit(items_info, spec, prepared)   # hook for describers that need a global fit
        counts: Dict[str, int] = {}
        for token_set in self.tokens(items_info[spec.column], prepared):
            for tok in token_set:
                counts[tok] = counts.get(tok, 0) + 1
        prepared.global_counts = counts
        prepared.global_denom = len(items_info)
        return prepared

    def _fit(self, items_info: pd.DataFrame, spec, prepared: PreparedFeature) -> None:
        """Optional global fit (no-op by default)."""


def _clean_cell(cell) -> Optional[str]:
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


class CategoricalDescriber(FeatureDescriber):
    kind = "categorical"

    def tokens(self, series: pd.Series, prepared: PreparedFeature) -> Iterable[Set[str]]:
        sep = prepared.spec.multi_value_sep
        for cell in series:
            s = _clean_cell(cell)
            if s is None:
                yield set()
            elif sep:
                yield {t.strip() for t in s.split(sep) if t.strip()}
            else:
                yield {s}


class NumericDescriber(FeatureDescriber):
    """Quantile-bins a numeric column into labelled bands (e.g. cheap/average/luxury).

    Functional seam: works if a FeatureSpec sets kind="numeric", but no current dataset
    configures one (H&M price is not in item_features yet). Global bin edges are fixed in
    ``_fit`` so prototype subsets are binned consistently with the whole-item universe.
    """
    kind = "numeric"
    _DEFAULT_QUANTILES = [0.0, 0.25, 0.5, 0.75, 1.0]
    _DEFAULT_LABELS = ["very low", "low", "high", "very high"]

    def _fit(self, items_info: pd.DataFrame, spec, prepared: PreparedFeature) -> None:
        quantiles = spec.bin_quantiles or self._DEFAULT_QUANTILES
        labels = spec.bin_labels or self._DEFAULT_LABELS
        if len(labels) != len(quantiles) - 1:
            raise ValueError(
                f"numeric feature '{spec.column}': need len(bin_labels) == len(bin_quantiles) - 1"
            )
        values = pd.to_numeric(items_info[spec.column], errors="coerce").dropna()
        edges = np.unique(np.quantile(values, quantiles))
        prepared.state["edges"] = edges
        prepared.state["labels"] = labels

    def tokens(self, series: pd.Series, prepared: PreparedFeature) -> Iterable[Set[str]]:
        edges = prepared.state["edges"]
        labels = prepared.state["labels"]
        values = pd.to_numeric(series, errors="coerce")
        for v in values:
            if pd.isna(v):
                yield set()
                continue
            # rightmost edge inclusive; clamp index into the labels range
            idx = int(np.searchsorted(edges, v, side="right") - 1)
            idx = max(0, min(idx, len(labels) - 1))
            yield {labels[idx]}


DESCRIBERS: Dict[str, FeatureDescriber] = {
    "categorical": CategoricalDescriber(),
    "numeric": NumericDescriber(),
}
