"""Schematic: the item factor q_i under the DOUBLE-TIED prototype layer (UI-ProtoMF), and the
single thing dc01 changes (how q_i is obtained: free per-item row -> sum of feature embeddings).

Corrects the earlier 'free embedding' framing: the base vector is per-item learned but NOT
single-purpose — the same q_i feeds both halves of the score (t* = sim to item-prototypes, and
t̂ = W^t q_i projected into user-prototype space). dc01 leaves the tie + both roles untouched.
Pure illustration. Run: python make_doubletie_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
BLUE, GREY, ORANGE = "#1f4e8c", "#5b6b7b", "#b08948"


def box(x, y, w, h, text, fc="#eef2f7", ec=GREY, fs=10, weight="normal", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=tc)


def arr(x0, y0, x1, y1, color=GREY, lw=1.6, label=None, lx=0, ly=0, fs=9, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=fs, color=color,
                ha="center", va="center", fontweight="bold")


# --- base embeddings (top) ---
box(0.16, 0.83, 0.20, 0.09, "user base  u", fc="#eaf0e6", ec="#5a7d3a")
box(0.64, 0.83, 0.20, 0.09, "item base  q$_i$", fc="#dde9f7", ec=BLUE, fs=11, weight="bold")
ax.text(0.74, 0.785, "= free per-item row  (user_item_proto)\n"
                     "   or  Σ$_f$ e$_f$  (feature_item_proto, dc01)",
        ha="center", va="top", fontsize=8.5, color=BLUE, style="italic")

# --- two prototype spaces (mid) ---
box(0.04, 0.40, 0.42, 0.20, "", fc="#fbf7f0", ec=ORANGE)
ax.text(0.25, 0.575, "USER-prototype space  (R$^{K_u}$)", ha="center", fontsize=10,
        color=ORANGE, fontweight="bold")
box(0.10, 0.46, 0.13, 0.07, "u*", fc="#eaf0e6", ec="#5a7d3a", fs=11, weight="bold")
box(0.27, 0.46, 0.13, 0.07, "t̂", fc="#dde9f7", ec=BLUE, fs=11, weight="bold")
ax.text(0.25, 0.435, r"$s_{user} = u^* \cdot \hat{t}$", ha="center", fontsize=9.5)

box(0.54, 0.40, 0.42, 0.20, "", fc="#fbf7f0", ec=ORANGE)
ax.text(0.75, 0.575, "ITEM-prototype space  (R$^{K_t}$)", ha="center", fontsize=10,
        color=ORANGE, fontweight="bold")
box(0.60, 0.46, 0.13, 0.07, "û", fc="#eaf0e6", ec="#5a7d3a", fs=11, weight="bold")
box(0.77, 0.46, 0.13, 0.07, "t*", fc="#dde9f7", ec=BLUE, fs=11, weight="bold")
ax.text(0.75, 0.435, r"$s_{item} = \hat{u} \cdot t^*$", ha="center", fontsize=9.5)

# --- arrows: each base feeds BOTH spaces (the double tie) ---
# user u -> u* (cosine, same space)   and  -> û (W^u, into item space)
arr(0.24, 0.83, 0.17, 0.53, color="#5a7d3a", label="cos vs P$^u$", lx=-0.055, ly=0.02, fs=8)
arr(0.28, 0.83, 0.65, 0.53, color="#5a7d3a", label="W$^u$", lx=0.0, ly=0.06, fs=9)
# item q_i -> t* (cosine, same space)  and  -> t̂ (W^t, into user space)  [highlighted: the tie]
arr(0.76, 0.83, 0.835, 0.53, color=BLUE, lw=2.4, label="cos vs P$^t$", lx=0.055, ly=0.02, fs=8)
arr(0.72, 0.83, 0.335, 0.53, color=BLUE, lw=2.4, label="W$^t$", lx=0.0, ly=0.08, fs=9)

# --- score (bottom) ---
box(0.32, 0.13, 0.36, 0.10, r"score(u,i) = $u^*\!\cdot\hat{t}\ +\ \hat{u}\cdot t^*$",
    fc="#f0eef7", ec="#5b4b8a", fs=12, weight="bold")
arr(0.25, 0.40, 0.42, 0.23, color="#5b4b8a")
arr(0.75, 0.40, 0.58, 0.23, color="#5b4b8a")

ax.text(0.5, 0.965,
        "The item factor q$_i$ under the double-tied prototype layer (UI-ProtoMF)",
        ha="center", fontsize=13, fontweight="bold")
ax.text(0.5, 0.045,
        "The SAME q$_i$ feeds BOTH halves: its own item-prototype similarity (t*) AND its projection "
        "into user-prototype space (t̂).  dc01 changes only how q$_i$ is built (free row → Σ$_f$ e$_f$); "
        "the tie and both roles are unchanged, and feature embeddings get gradient from both halves.",
        ha="center", va="center", fontsize=9, color=BLUE,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f8ff", ec=BLUE, lw=0.8))

out = "Master/temp/dc_checks/dc01/figures/item_vector_double_tie.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print(f"wrote {out}")
