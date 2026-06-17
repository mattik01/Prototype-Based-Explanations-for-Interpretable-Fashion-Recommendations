"""Top-k items per prototype — delegates to utilities.explanations_utils.get_top_k_items.

For item prototypes: item_weights = item-to-item-prototype similarity matrix.
For user prototypes: item_weights = all item embeddings (per get_top_k_items docstring).

When items_info has only (item_id, item), the output CSV is still useful: it has
the prototype column plus the raw item id and weight.
"""
import os

import pandas as pd

from utilities.explanations.explainers.base import ExplainCtx, Explainer
from utilities.explanations_utils import get_top_k_items


class TopKItemsExplainer(Explainer):
    name = "top_k_items"

    def supports(self, model_type: str) -> bool:
        return model_type in {"item_proto", "user_proto", "user_item_proto", "feature_item_proto"}

    def run(self, ctx: ExplainCtx) -> None:
        if ctx.accessor.has_item_prototypes:
            sim = ctx.accessor.item_to_item_proto_sim()
            self._dump_per_prototype(
                sim, ctx,
                out_name="top_k_items_per_item_prototype.csv",
            )

        if ctx.accessor.has_user_prototypes:
            # Interpret each user prototype by the items most aligned with it.
            # The accessor returns items in user-proto space (raw embedding for user_proto,
            # EmbeddingW projection for user_item_proto).
            items_in_upspace = ctx.accessor.items_in_user_proto_space()
            if items_in_upspace is not None:
                self._dump_per_prototype(
                    items_in_upspace, ctx,
                    out_name="top_k_items_per_user_prototype.csv",
                )

    @staticmethod
    def _dump_per_prototype(item_weights, ctx: ExplainCtx, out_name: str) -> None:
        n_protos = item_weights.shape[1]
        frames = []
        for p in range(n_protos):
            df = get_top_k_items(item_weights, ctx.items_info, proto_idx=p, top_k=ctx.top_k)
            df = df.copy()
            df["prototype"] = p
            frames.append(df)
        pd.concat(frames).to_csv(os.path.join(ctx.output_dir, out_name))
