"""Feature small-multiples — item-prototype side only.

For a categorical feature (e.g. ml-1m `genres`), draw one TSNE panel per feature value
on a *single shared layout*:
  - items carrying that value are highlighted (blue); the rest are faded grey,
  - item prototypes are coloured on a black -> red scale by how strongly they activate
    that value (their naming score for it),
  - a prototype's name is annotated only when the name actually contains that value.

This is a separate figure from the standard TSNE plot, and reuses the per-prototype
per-value scores already computed by the naming step (ctx.naming_item).
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.manifold import TSNE

from utilities.explanations.explainers.base import ExplainCtx, Explainer
from utilities.explanations.naming.describers import DESCRIBERS

# black -> red: a prototype that strongly activates a value glows red in that value's
# panel and stays near-black everywhere else (shared vmax makes panels comparable).
_BLACK_RED = LinearSegmentedColormap.from_list("black_red", ["black", "red"])


class FeatureSmallMultiplesExplainer(Explainer):
    name = "feature_small_multiples"

    # max distinct values to render as panels (guard against high-cardinality features)
    max_values = 40

    def supports(self, model_type: str) -> bool:
        # Needs item prototypes (genre membership is an item property).
        return model_type in {"item_proto", "user_item_proto"}

    def run(self, ctx: ExplainCtx) -> None:
        if ctx.naming_item is None or ctx.naming_cfg is None:
            return
        if not ctx.accessor.has_item_prototypes:
            return

        rng = np.random.default_rng(42)
        item_emb = ctx.accessor.item_embeddings()
        protos = ctx.accessor.item_prototypes()

        # One shared TSNE layout (prototypes first, then a sample of items).
        n_items = item_emb.shape[0]
        sample_size = min(ctx.tsne_sample_size, n_items)
        sample_ids = rng.choice(n_items, size=sample_size, replace=False)
        coords = TSNE(perplexity=30, metric="cosine", init="pca",
                      learning_rate="auto", random_state=42).fit_transform(
            np.vstack([protos, item_emb[sample_ids]])
        )
        proto_xy = coords[:len(protos)]
        item_xy = coords[len(protos):]

        items_indexed = ctx.items_info.set_index(ctx.naming_cfg.id_column)
        sample_rows = items_indexed.loc[sample_ids]

        for spec in ctx.naming_cfg.features:
            if spec.kind != "categorical" or spec.column not in ctx.items_info.columns:
                continue
            self._plot_feature(ctx, spec, sample_rows, item_xy, proto_xy)

    def _plot_feature(self, ctx, spec, sample_rows, item_xy, proto_xy):
        describer = DESCRIBERS[spec.kind]
        prepared = describer.prepare(ctx.items_info, spec)

        # Panels: the most frequent values (capped), so the grid stays sane.
        values = sorted(prepared.global_counts, key=prepared.global_counts.get, reverse=True)
        values = values[:self.max_values]
        if not values:
            return

        # Per-(value, prototype) activation scores from the naming profiles.
        score_by_value = {v: {} for v in values}
        for profile in ctx.naming_item.profiles:
            for s in profile.stats:
                if s.column == spec.column and s.value in score_by_value:
                    score_by_value[s.value][profile.proto_idx] = s.score
        vmax = max((sc for d in score_by_value.values() for sc in d.values()), default=1.0) or 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)

        # Item membership per panel (tokenize the sampled items once).
        token_sets = list(describer.tokens(sample_rows[spec.column], prepared))

        # Which values are present in each prototype's *name* (for selective annotation).
        name_values = {
            idx: {d.value for d in pn.descriptors}
            for idx, pn in ctx.naming_item.names.items()
        }

        n = len(values)
        ncols = min(6, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6 + 0.6),
                                 squeeze=False)

        n_protos = proto_xy.shape[0]
        for k, value in enumerate(values):
            ax = axes[k // ncols][k % ncols]
            member = np.array([value in ts for ts in token_sets])

            ax.scatter(item_xy[~member, 0], item_xy[~member, 1], s=4, c="0.8", alpha=0.4)
            ax.scatter(item_xy[member, 0], item_xy[member, 1], s=6, c="#3b6fb6", alpha=0.7)

            scores = np.array([score_by_value[value].get(p, 0.0) for p in range(n_protos)])
            ax.scatter(proto_xy[:, 0], proto_xy[:, 1], s=28, c=scores, cmap=_BLACK_RED,
                       norm=norm, edgecolors="white", linewidths=0.3, zorder=3)

            for p in range(n_protos):
                if value in name_values.get(p, ()):
                    ax.annotate(ctx.naming_item.names[p].name, proto_xy[p],
                                fontsize=5, fontweight="bold", color="darkred",
                                xytext=(2, 2), textcoords="offset points")

            ax.set_title(f"{value}  (n={int(member.sum())})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        # Blank any unused grid cells.
        for k in range(n, nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")

        sm = ScalarMappable(norm=norm, cmap=_BLACK_RED)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, shrink=0.35, aspect=30)
        cbar.set_label(f"prototype score ({ctx.naming_cfg.scoring})", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        fig.suptitle(f"Item prototypes by '{spec.column}' — small multiples", fontsize=11)
        out = os.path.join(ctx.output_dir, f"feature_small_multiples_{spec.column}_item.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
