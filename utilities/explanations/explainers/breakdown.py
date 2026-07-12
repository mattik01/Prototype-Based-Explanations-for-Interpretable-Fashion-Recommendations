"""Per-recommendation breakdown explainer — the SC.5.4 shared renderer, auto-invoked.

For each sample user, renders the "why did I get this recommendation?" artifact for the
user's top-1 item via utilities.explanations.breakdown. Supported here: the dc01 stage
pair (feature_item_proto, item_proto). Other slots (lightfm/mf/popularity) are reached
via the breakdown CLI — they are not in the auto-explanations pipeline; user_proto /
user_item_proto slots are deferred to their lineage stages (SC.5 gate, 2026-07-12).
"""
from utilities.explanations.breakdown import (
    compute_breakdown_feature_item_proto,
    compute_breakdown_item_proto,
    top1_item_for_user,
    write_breakdown,
)
from utilities.explanations.explainers.base import ExplainCtx, Explainer


class BreakdownExplainer(Explainer):
    name = "breakdown"

    def supports(self, model_type: str) -> bool:
        return model_type in {"feature_item_proto", "item_proto"}

    def run(self, ctx: ExplainCtx) -> None:
        accessor = ctx.accessor
        model = accessor.model
        model_type = accessor.model_type
        for uid in ctx.sample_users_for_weight_viz:
            if uid < 0 or uid >= accessor.n_users:
                continue
            item_id = top1_item_for_user(model, uid, accessor.n_items)
            if model_type == "feature_item_proto":
                # in the intrinsic pass the naming cfg's features ARE the model's fields,
                # in code order (set from config in the pipeline).
                feature_fields = [s.column for s in ctx.naming_cfg.features]
                bd = compute_breakdown_feature_item_proto(
                    model, uid, item_id, ctx.items_info, feature_fields,
                    naming_item=ctx.naming_item)
            else:
                bd = compute_breakdown_item_proto(
                    model, uid, item_id, ctx.items_info, naming_item=ctx.naming_item)
            write_breakdown(bd, ctx.output_dir)
