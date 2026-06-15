"""Assemble per-variant prototype t-SNE PNGs into one grid per dataset.

Rows = regularization level (off .. x100), columns = item | user prototypes.
Each panel is titled with its regularization level and test accuracy (HR@10, NDCG@10).

Output: Master/experiments/reg_sensitivity/tsne_grid_<dataset>.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.join(REPO, "Master", "experiments", "reg_sensitivity")
SEED = 38210573

VARIANTS = [
    ("reg_off", "off  (×0)"),
    ("reg_x0.01", "×0.01"),
    ("reg_x0.1", "×0.1"),
    ("reg_x1", "×1  (baseline / hyperopt-selected)"),
    ("reg_x10", "×10"),
    ("reg_x100", "×100"),
]
DATASETS = [
    ("ml-1m", "MovieLens-1M"),
    ("hm_1_month", "H&M (1 month)"),
]


def acc(results_dir):
    with open(os.path.join(results_dir, "test_metrics.json")) as f:
        m = json.load(f)
    return m["hit_ratio@10"], m["ndcg@10"]


def arch_info(dataset):
    """Read architecture + base (x1) reg weights from the x1 variant config
    (x1 leaves the baseline weights unscaled), so the header is exact."""
    cfg_path = os.path.join(
        ROOT, f"user_item_proto_{dataset}_s{SEED}_reg_x1", "config.json")
    with open(cfg_path) as f:
        c = json.load(f)
    ft = c["ft_ext_param"]
    u, i = ft["user_ft_ext_param"], ft["item_ft_ext_param"]
    return {
        "emb_dim": ft["embedding_dim"],
        "item_protos": i["n_prototypes"],
        "user_protos": u["n_prototypes"],
        "u_sim_proto": u["sim_proto_weight"], "u_sim_batch": u["sim_batch_weight"],
        "i_sim_proto": i["sim_proto_weight"], "i_sim_batch": i["sim_batch_weight"],
    }


def build(dataset, pretty):
    nrows = len(VARIANTS)
    a = arch_info(dataset)
    fig, axes = plt.subplots(nrows, 2, figsize=(9.5, 4.4 * nrows))
    header = (
        f"{pretty} — user_item_proto prototype t-SNE vs prototype regularization (seed {SEED})\n"
        f"embedding_dim = {a['emb_dim']}   |   item prototypes = {a['item_protos']}   |   "
        f"user prototypes = {a['user_protos']}\n"
        f"Base reg weights (×1, jointly scaled by the multiplier) — 2 regularizers × 2 sides = 4:\n"
        f"item: sim_proto={a['i_sim_proto']:.4g}, sim_batch={a['i_sim_batch']:.4g}   |   "
        f"user: sim_proto={a['u_sim_proto']:.4g}, sim_batch={a['u_sim_batch']:.4g}\n"
        f"baseline config: Master/experiments/results/user_item_proto_{dataset}_s{SEED}/config.json"
    )
    fig.suptitle(header, fontsize=12, y=0.998)

    for r, (vkey, vlabel) in enumerate(VARIANTS):
        rdir = os.path.join(ROOT, f"user_item_proto_{dataset}_s{SEED}_{vkey}")
        lift = os.path.join(rdir, "explanations", "lift")
        hr, ndcg = acc(rdir)
        for c, (side, side_label) in enumerate([("item", "Item prototypes"),
                                                ("user", "User prototypes")]):
            ax = axes[r, c]
            png = os.path.join(lift, f"tsne_{side}_prototypes.png")
            ax.imshow(mpimg.imread(png))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(
                f"{side_label}  |  reg {vlabel}\n"
                f"HR@10 = {hr:.4f}   NDCG@10 = {ndcg:.4f}",
                fontsize=10,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(ROOT, f"tsne_grid_{dataset}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for ds, pretty in DATASETS:
        build(ds, pretty)
