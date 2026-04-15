"""Explainer ABC + shared context passed to each explainer."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import pandas as pd

from utilities.explanations.accessor.base import ProtoAccessor


@dataclass
class ExplainCtx:
    accessor: ProtoAccessor
    items_info: pd.DataFrame
    output_dir: str
    tsne_sample_size: int
    top_k: int
    sample_users_for_weight_viz: List[int]


class Explainer(ABC):
    name: str = "Explainer"

    @abstractmethod
    def supports(self, model_type: str) -> bool:
        ...

    @abstractmethod
    def run(self, ctx: ExplainCtx) -> None:
        """Write artifacts to ctx.output_dir. Should degrade gracefully on missing data."""
        ...
