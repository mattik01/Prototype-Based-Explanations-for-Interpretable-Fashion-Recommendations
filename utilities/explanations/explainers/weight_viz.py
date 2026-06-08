"""Per-recommendation weight visualization — user_item_proto only.

Delegates to utilities.explanations_utils.weight_visualization (unchanged).
"""
import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utilities.explanations.explainers.base import ExplainCtx, Explainer
from utilities.explanations_utils import weight_visualization


class WeightVizExplainer(Explainer):
    name = "weight_viz"

    def supports(self, model_type: str) -> bool:
        # Projections only exist for the weight-tied double branch (user_item_proto).
        return model_type == "user_item_proto"

    def run(self, ctx: ExplainCtx) -> None:
        accessor = ctx.accessor
        model = accessor.model

        # Top-scoring item for each sample user
        with torch.no_grad():
            all_item_ids = torch.arange(accessor.n_items)
            item_out = model.item_feature_extractor(all_item_ids)  # (n_items, out_dim)

            user_protos = torch.as_tensor(accessor.user_prototypes(), dtype=torch.float32)
            item_protos = torch.as_tensor(accessor.item_prototypes(), dtype=torch.float32)
            user_emb_all = torch.as_tensor(accessor.user_embeddings(), dtype=torch.float32)
            item_emb_all = torch.as_tensor(accessor.item_embeddings(), dtype=torch.float32)

            u_proto_labels = (ctx.naming_user.labels_in_order(user_protos.shape[0])
                              if ctx.naming_user else None)
            i_proto_labels = (ctx.naming_item.labels_in_order(item_protos.shape[0])
                              if ctx.naming_item else None)

            for uid in ctx.sample_users_for_weight_viz:
                if uid < 0 or uid >= accessor.n_users:
                    continue
                u_ids = torch.tensor([uid])
                user_out = model.user_feature_extractor(u_ids)  # (1, out_dim)

                scores = (user_out @ item_out.T).squeeze(0)
                top_item = int(scores.argmax().item())

                u_sim_mtx = _shifted_cos(user_emb_all[uid:uid + 1], user_protos).squeeze(0).numpy()
                i_sim_mtx = _shifted_cos(item_emb_all[top_item:top_item + 1], item_protos).squeeze(0).numpy()

                u_proj = accessor.user_proj_to_item_proto_space(u_ids).squeeze()
                i_proj = accessor.item_proj_to_user_proto_space(torch.tensor([top_item])).squeeze()

                weight_visualization(u_sim_mtx, u_proj, i_sim_mtx, i_proj, annotate_top_k=3,
                                     u_proto_labels=u_proto_labels, i_proto_labels=i_proto_labels)

                # weight_visualization creates two figures (user side, item side); save both.
                figs = [plt.figure(n) for n in plt.get_fignums()]
                if len(figs) >= 2:
                    user_fig, item_fig = figs[-2], figs[-1]
                    user_fig.savefig(
                        os.path.join(ctx.output_dir, f"weight_viz_user{uid}_user_side.png"),
                        format="png", dpi=200, bbox_inches="tight",
                    )
                    item_fig.savefig(
                        os.path.join(ctx.output_dir, f"weight_viz_user{uid}_item_side.png"),
                        format="png", dpi=200, bbox_inches="tight",
                    )
                plt.close("all")


def _shifted_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 + F.normalize(a, dim=1) @ F.normalize(b, dim=1).T
