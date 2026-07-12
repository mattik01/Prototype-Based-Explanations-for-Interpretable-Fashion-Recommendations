"""Prototype cards — the standing "what do the prototypes look like" figure (SC.5.3).

One compact card per item prototype: its name + its top profile entries as horizontal
bars, in the SAME card geometry for every naming route — so the host's post-hoc reading
and the variant's intrinsic reading compare like-for-like (side-by-side composition of
the two grids happens at SC.8/SC.9). The mechanism difference is declared in the figure
subtitle, and post-hoc cards additionally list their top exemplar items (the route's
evidence — an inherent, declared asymmetry, not a styling one).

Consumes the NamingResult already computed by the pipeline (ctx.naming_item), so the
cards are guaranteed consistent with the names every other artifact shows.
"""
import os

import numpy as np
from matplotlib import pyplot as plt

from utilities.explanations.explainers.base import ExplainCtx, Explainer

_POS = "#2a78d6"
_NEG = "#e34948"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_GRID = "#d9d8d4"

_ROUTE_LABEL = {
    "cosine": "intrinsic profile — cos(e_f, p_k), read from the model's parameters",
    "lift": "post-hoc profile — lift over each prototype's top-k nearest items",
    "raw": "post-hoc profile — frequency over each prototype's top-k nearest items",
}


class ProtoCardsExplainer(Explainer):
    name = "proto_cards"

    top_entries = 6      # profile entries per card
    ncols = 4

    def supports(self, model_type: str) -> bool:
        # attr_item_proto joins at dc02's SC.5 (no cross-candidate work at dc01's).
        # user_proto / feature_user_proto joined at the dc05 build (fU stage, like-for-like:
        # fU cards = intrinsic word-profile bars, host cards = post-hoc descriptors — same
        # card geometry per the SC.5 fairness contract).
        return model_type in {"item_proto", "user_item_proto", "feature_item_proto",
                              "user_proto", "feature_user_proto"}

    def run(self, ctx: ExplainCtx) -> None:
        # One grid per available side, same card geometry (the mechanism difference is
        # declared in the subtitle, never styled).
        self._render_side(ctx, ctx.naming_item, "Item prototypes — profile cards",
                          "prototype_cards_item.png")
        self._render_side(ctx, ctx.naming_user, "User prototypes — profile cards",
                          "prototype_cards_user.png")

    def _render_side(self, ctx: ExplainCtx, result, suptitle: str, out_name: str) -> None:
        if result is None or not result.profiles:
            return
        scoring = result.config.scoring
        route = _ROUTE_LABEL.get(scoring, f"profile scoring: {scoring}")

        profiles = sorted(result.profiles, key=lambda p: p.proto_idx)
        n = len(profiles)
        ncols = min(self.ncols, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(ncols * 2.7, nrows * 1.9 + 0.7), dpi=200)
        # shared x-scale across cards, but over the SHOWN entries only — a single
        # off-card outlier (e.g. a rare value with huge lift) must not flatten the grid
        tops = {p.proto_idx: sorted(p.stats, key=lambda s: -s.score)[:self.top_entries]
                for p in profiles}
        vmax = max((abs(s.score) for t in tops.values() for s in t), default=1.0) or 1.0

        for j, profile in enumerate(profiles):
            ax = axes[j // ncols][j % ncols]
            top = list(reversed(tops[profile.proto_idx]))
            y = np.arange(len(top))
            vals = [s.score for s in top]
            ax.barh(y, vals, height=0.62,
                    color=[_POS if v >= 0 else _NEG for v in vals], zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels([_trunc(s.value, 20) for s in top],
                               fontsize=5.5, color=_INK)
            ax.set_xlim(-vmax * 0.05, vmax * 1.05)
            ax.axvline(0, color=_INK2, linewidth=0.6)
            name = result.names[profile.proto_idx].name
            title = f"p{profile.proto_idx} · {_trunc(name, 26)}"
            # post-hoc cards carry their evidence: the top exemplar items
            if profile.topk_item_ids:
                exemplars = ", ".join(str(i) for i in profile.topk_item_ids[:3])
                title += f"\n(top items: {exemplars})"
            ax.set_title(title, fontsize=6, color=_INK, loc="left")
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(_GRID)
            ax.tick_params(axis="x", colors=_INK2, labelsize=5)
            ax.xaxis.grid(True, color=_GRID, linewidth=0.5)
            ax.set_axisbelow(True)

        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(suptitle, fontsize=10, color=_INK)
        fig.text(0.5, 0.955 - 0.03, route, ha="center", fontsize=7, color=_INK2)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(ctx.output_dir, out_name)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _trunc(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"
