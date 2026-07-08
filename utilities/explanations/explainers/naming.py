"""Prototype-name tables + transparent naming stats.

Consumes the NamingResults already computed by the pipeline (ctx.naming_item /
ctx.naming_user) and writes, per side:
  - prototype_names_<side>.csv        : prototype -> name (+ chosen descriptors)
  - prototype_naming_stats_<side>.csv : every (feature, value) candidate with its raw
                                        count, top-k freq, global freq, lift, score, and
                                        whether it was selected into the name.

Every name is fully reconstructable from the stats dump (transparency requirement).
"""
import os

import pandas as pd

from utilities.explanations.explainers.base import ExplainCtx, Explainer
from utilities.explanations.naming import NamingResult


class NamingExplainer(Explainer):
    name = "naming"

    def supports(self, model_type: str) -> bool:
        return model_type in {"item_proto", "user_proto", "user_item_proto", "feature_item_proto",
                              "attr_item_proto"}

    def run(self, ctx: ExplainCtx) -> None:
        for result in (ctx.naming_item, ctx.naming_user):
            if result is not None:
                self._dump(result, ctx.output_dir)

    @staticmethod
    def _dump(result: NamingResult, output_dir: str) -> None:
        side = result.side

        name_rows = []
        for idx in sorted(result.names):
            pn = result.names[idx]
            name_rows.append({
                "prototype": idx,
                "name": pn.name,
                "n_descriptors": len(pn.descriptors),
                "descriptors": "; ".join(
                    f"{d.column}={d.value}(score={d.score:.3f})" for d in pn.descriptors
                ),
            })
        pd.DataFrame(name_rows).to_csv(
            os.path.join(output_dir, f"prototype_names_{side}.csv"), index=False
        )

        stat_rows = []
        for profile in result.profiles:
            selected = {
                (d.column, d.value) for d in result.names[profile.proto_idx].descriptors
            }
            for s in profile.stats:
                stat_rows.append({
                    "prototype": profile.proto_idx,
                    "feature": s.column,
                    "value": s.value,
                    "count": s.count,
                    "topk_size": profile.topk_size,
                    "topk_freq": s.topk_freq,
                    "global_freq": s.global_freq,
                    "lift": s.lift,
                    "score": s.score,
                    "selected": (s.column, s.value) in selected,
                })
        stats_df = pd.DataFrame(stat_rows)
        # Most informative ordering: by prototype, then by descending score.
        if not stats_df.empty:
            stats_df = stats_df.sort_values(
                ["prototype", "score", "count"], ascending=[True, False, False]
            )
        stats_df.to_csv(
            os.path.join(output_dir, f"prototype_naming_stats_{side}.csv"), index=False
        )
