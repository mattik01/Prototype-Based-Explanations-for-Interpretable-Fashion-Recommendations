"""Registry of explainers. To add a new one: create a file here, append an instance below."""
from typing import List

from utilities.explanations.explainers.base import ExplainCtx, Explainer
from utilities.explanations.explainers.breakdown import BreakdownExplainer
from utilities.explanations.explainers.feature_small_multiples import FeatureSmallMultiplesExplainer
from utilities.explanations.explainers.naming import NamingExplainer
from utilities.explanations.explainers.proto_cards import ProtoCardsExplainer
from utilities.explanations.explainers.top_k_items import TopKItemsExplainer
from utilities.explanations.explainers.tsne import TSNEExplainer
from utilities.explanations.explainers.weight_viz import WeightVizExplainer

REGISTERED_EXPLAINERS: List[Explainer] = [
    NamingExplainer(),
    TSNEExplainer(),
    TopKItemsExplainer(),
    WeightVizExplainer(),
    FeatureSmallMultiplesExplainer(),
    ProtoCardsExplainer(),
    BreakdownExplainer(),
]

__all__ = ["REGISTERED_EXPLAINERS", "Explainer", "ExplainCtx"]
