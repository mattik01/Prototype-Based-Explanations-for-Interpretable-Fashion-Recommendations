"""Explainer ABC + shared context passed to each explainer."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from utilities.explanations.accessor.base import ProtoAccessor
from utilities.explanations.naming import NamingConfig, NamingResult


@dataclass
class ExplainCtx:
    accessor: ProtoAccessor
    items_info: pd.DataFrame
    output_dir: str
    tsne_sample_size: int
    top_k: int
    sample_users_for_weight_viz: List[int]
    # Prototype names, computed once in the pipeline and shared by all explainers.
    naming_cfg: Optional[NamingConfig] = None
    naming_item: Optional[NamingResult] = None   # item-prototype names (closeness route)
    naming_user: Optional[NamingResult] = None   # user-prototype names (activation route)
    # Split directory (dc05: the fU breakdown's per-purchase zoom reads the train file).
    dataset_dir: Optional[str] = None
    # Feature layout of the model's grounded side ('fixed' | 'bags') — the breakdown slots
    # need it to gather purchase rows and code→labels from the right builder (F-DC05-13).
    feature_layout: str = "fixed"


class Explainer(ABC):
    name: str = "Explainer"

    @abstractmethod
    def supports(self, model_type: str) -> bool:
        ...

    @abstractmethod
    def run(self, ctx: ExplainCtx) -> None:
        """Write artifacts to ctx.output_dir. Should degrade gracefully on missing data."""
        ...
