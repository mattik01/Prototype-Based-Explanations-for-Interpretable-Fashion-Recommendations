"""TSNE plot explainer — delegates to utilities.explanations_utils.tsne_plot (unchanged)."""
import os

import numpy as np
from matplotlib import pyplot as plt

from utilities.explanations.explainers.base import ExplainCtx, Explainer
from utilities.explanations_utils import tsne_plot


class TSNEExplainer(Explainer):
    name = "tsne"

    def supports(self, model_type: str) -> bool:
        return model_type in {"item_proto", "user_proto", "user_item_proto"}

    def run(self, ctx: ExplainCtx) -> None:
        rng = np.random.default_rng(42)

        if ctx.accessor.has_item_prototypes:
            items = ctx.accessor.item_embeddings()
            protos = ctx.accessor.item_prototypes()
            sample = _sample(items, ctx.tsne_sample_size, rng)
            labels = ctx.naming_item.labels_in_order(protos.shape[0]) if ctx.naming_item else None
            tsne_plot(
                sample, protos,
                object_legend_text="Items",
                perplexity=30,
                path_save_fig=os.path.join(ctx.output_dir, "tsne_item_prototypes.png"),
                prototype_labels=labels,
            )
            plt.close("all")

        if ctx.accessor.has_user_prototypes:
            users = ctx.accessor.user_embeddings()
            protos = ctx.accessor.user_prototypes()
            sample = _sample(users, ctx.tsne_sample_size, rng)
            labels = ctx.naming_user.labels_in_order(protos.shape[0]) if ctx.naming_user else None
            tsne_plot(
                sample, protos,
                object_legend_text="Users",
                perplexity=30,
                path_save_fig=os.path.join(ctx.output_dir, "tsne_user_prototypes.png"),
                prototype_labels=labels,
            )
            plt.close("all")


def _sample(embeddings: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    size = min(n, embeddings.shape[0])
    idx = rng.choice(embeddings.shape[0], size=size, replace=False)
    return embeddings[idx]
