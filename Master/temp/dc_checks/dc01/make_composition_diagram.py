"""Schematic: how feature_item_proto builds an item vector q_i.

Contrasts the standard ProtoMF free per-item embedding (one looked-up vector) with dc01's
feature-composed embedding q_i = Σ_f e_f (+ e_ID): the SUM of the learned embedding VECTORS of
the item's feature VALUES. Pure illustration (no model/data); the displayed sum strip is the
true element-wise sum of the displayed feature strips. Run: python make_composition_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

rng = np.random.default_rng(7)
D = 8
feats = ["department = Bags", "product_type = Bag", "colour = Black",
         "section = Womens Big acc.", "appearance = Solid"]
fvecs = rng.normal(0, 1, size=(len(feats), D))
id_vec = rng.normal(0, 1, size=D) * 0.3        # small per-item ID contribution
q_feat = fvecs.sum(0)
q_full = q_feat + id_vec
cf_vec = rng.normal(0, 1, size=D)              # CF free embedding (illustrative)

cmap = plt.cm.coolwarm
vmax = float(np.abs(np.concatenate([fvecs.ravel(), q_full, cf_vec, id_vec])).max())
CW, CH = 0.020, 0.055


def strip(ax, x, y, vec, label=None, label_dx=0.006, fs=8.5, bold=False):
    for j, v in enumerate(vec):
        ax.add_patch(Rectangle((x + j * CW, y), CW, CH, facecolor=cmap(0.5 + 0.5 * v / vmax),
                               edgecolor="white", lw=0.6))
    if label:
        ax.text(x + len(vec) * CW + label_dx, y + CH / 2, label, va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")


def box(ax, x, y, w, h, text, fc="#eef2f7", ec="#5b6b7b", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color="#5b6b7b", lw=1.3))


fig, ax = plt.subplots(figsize=(12, 6.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.axvline(0.485, color="0.85", lw=1)

# ---------------- LEFT: standard ProtoMF (free per-item embedding) ----------------
ax.text(0.03, 0.95, "Standard ProtoMF  (user_item_proto)", fontsize=12, fontweight="bold")
box(ax, 0.05, 0.74, 0.30, 0.07, "item i  (a black bag)")
arrow(ax, 0.20, 0.74, 0.20, 0.66)
ax.text(0.215, 0.70, "ID lookup", fontsize=8.5, style="italic", color="#444")
box(ax, 0.04, 0.50, 0.33, 0.14,
    "per-item embedding table\n(N items × d)  —  one row per item", fc="#f5f0e8", ec="#b08948", fs=9)
arrow(ax, 0.20, 0.50, 0.20, 0.40)
strip(ax, 0.115, 0.32, cf_vec)
ax.text(0.115 + D * CW + 0.006, 0.32 + CH / 2, r"$=\ q_i$", va="center", fontsize=12, fontweight="bold")
ax.text(0.05, 0.20, "One learned vector per item (looked up by ID).\n"
                    "Not feature-derived → items independent → spread out in t-SNE.",
        fontsize=9, color="#333", va="top")

# ---------------- RIGHT: feature_item_proto (sum of feature-value embeddings) ----------------
x0 = 0.52
ax.text(x0 + 0.01, 0.95, "feature_item_proto  (dc01, intrinsic)", fontsize=12, fontweight="bold")
box(ax, x0, 0.86, 0.46, 0.06, "item i  (a black bag)  →  its feature VALUES", fs=9)

y = 0.78
for k, (name, vec) in enumerate(zip(feats, fvecs)):
    ax.text(x0 + 0.005, y + CH / 2, name, va="center", ha="left", fontsize=8.5)
    strip(ax, x0 + 0.215, y, vec, label=f"e[{name.split(' = ')[1]}]" if k == 0 else None, fs=7.5)
    if k < len(feats):
        ax.text(x0 + 0.20, y - 0.012, "+", fontsize=12, ha="center", color="#444")
    y -= 0.082
# ID row
ax.text(x0 + 0.005, y + CH / 2, "item-ID row  (small)", va="center", ha="left", fontsize=8.5, color="#777")
strip(ax, x0 + 0.215, y, id_vec)

# sum bar
y_sum = y - 0.10
ax.add_patch(FancyArrowPatch((x0 + 0.215 + D * CW / 2, y - 0.01),
                             (x0 + 0.215 + D * CW / 2, y_sum + CH + 0.012),
                             arrowstyle="-|>", mutation_scale=13, color="#3b7dd8", lw=1.6))
ax.text(x0 + 0.215 + D * CW + 0.02, (y + y_sum) / 2 + 0.02, r"$\sum$", fontsize=15,
        color="#3b7dd8", fontweight="bold")
strip(ax, x0 + 0.215, y_sum, q_full)
ax.text(x0 + 0.215 + D * CW + 0.006, y_sum + CH / 2, r"$=\ q_i = \sum_{f}\, e_f\ (+\,e_{ID})$",
        va="center", fontsize=12, fontweight="bold", color="#1f4e8c")

ax.text(x0, 0.075, "q_i is the SUM of the learned feature-value embedding VECTORS (not the labels).\n"
                   "Two items with the same feature values share the same e_f → nearly identical q_i\n"
                   "→ they collapse together in t-SNE (feature-coherent clusters).\n"
                   "(Either q_i then feeds the double-tie — see item_vector_double_tie.png.)",
        fontsize=9, color="#1f4e8c", va="top")

fig.suptitle("How the item vector q_i is built:  free per-item embedding  vs  sum of feature-value embeddings",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "Master/temp/dc_checks/dc01/figures/item_vector_composition.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print(f"wrote {out}")
