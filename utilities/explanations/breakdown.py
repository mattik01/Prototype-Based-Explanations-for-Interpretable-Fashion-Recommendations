"""Shared user-perspective recommendation breakdown (dc01 SC.5.4, scrutiny protocol v1.2).

One renderer, one layout, one visual language for every model in the comparison; each
model plugs its mechanism into declared slots — advantages may come from the algorithm
only, never from rendering effort (the SC.5 fairness contract). Produces, per
(user, recommended item, model), a self-contained "why did I get this recommendation?"
artifact: a figure (PNG) plus a text companion (markdown), thesis-figure quality,
non-interactive, compact for repeated use.

Slots built at dc01's SC.5 (2026-07-12):
  - ``feature_item_proto`` (fI, incl. ``_noid``/``_f0`` arms) — per-prototype contribution
    bars + intrinsic per-feature-share zoom of the top prototype,
  - ``item_proto`` (the fI host) — identical contribution math (same shifted-cosine item
    branch, same free user vector), post-hoc top-k exemplar zoom (declared as post-hoc),
  - ``lightfm_tags`` / ``lightfm_tags_ids`` — exact per-feature bars u·e_r (no prototype
    layer; that absence is the comparison),
  - ``mf`` — the honest-asymmetry row: total score, "no decomposable explanation surface",
  - popularity — non-personalized rank/percentile statement (dataset-derived, no checkpoint).
``user_proto`` / ``user_item_proto`` slots are deliberately deferred to the lineage stages
that compare them (fU / fUfI); the frame is model-agnostic, so they are slot work only.

Honesty scope (SC.5 gate decision, 2026-07-12): the artifacts carry ARITHMETIC honesty
only — the +1 rank-inert baseline shown (never silently absorbed), signed contributions
(negative = legitimate anti-affinity), the item-ID row as its own line, and the
feature-explained fraction. Effect-epistemics (share determinacy, user-coefficient gauge)
are deliberately NOT rendered here; they belong to the thesis's dedicated hidden-effects
section (design doc §3.4 / §I.1 re-scope notes, same date).

Every artifact SELF-CHECKS before it is written: the sum of its rendered parts is
asserted against the model's own forward score (and the zoom against its parent bar).
"""
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Ensure repo root is on sys.path when invoked as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from utilities.explanations.feature_readout import item_feature_rows, per_feature_shares

# --- shared visual language (identical for every model — the fairness contract) ---
_POS = "#2a78d6"        # positive contribution (diverging cool pole)
_NEG = "#e34948"        # negative contribution (diverging warm pole)
_INK = "#0b0b0b"        # primary text
_INK2 = "#52514e"       # secondary text
_GRID = "#d9d8d4"       # recessive grid
_FIGSIZE = (7.2, 4.6)   # compact, repeated-use target
_TOP_LINES = 6          # bars shown per panel (rest folded into a stated remainder)

_SELF_CHECK_TOL = 5e-4  # relative; float32 sums over K·R terms


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class BreakdownLine:
    label: str
    value: float
    is_id: bool = False          # the per-item ID row (rendered set off: hatch + label)


@dataclass
class Breakdown:
    """Everything the renderer needs; produced by one compute_* slot per model."""
    model_label: str             # display name, e.g. "fI-ProtoMF (feature_item_proto)"
    mechanism: str               # 'proto' | 'flat' | 'opaque' | 'popularity'
    user_id: int
    item_id: int
    item_desc: str               # metadata one-liner for the header
    total_score: float           # the model's own forward score (popularity: count)
    naming_route: str            # declared naming mechanism (or 'n/a')
    baseline: Optional[float] = None          # proto slot: Σ_k u_k (rank-inert, disclosed)
    lines: List[BreakdownLine] = field(default_factory=list)   # left panel
    lines_title: str = ""
    zoom_kind: str = "none"      # 'shares' | 'exemplars' | 'statement' | 'none'
    zoom_title: str = ""
    zoom_lines: List[BreakdownLine] = field(default_factory=list)
    zoom_parent_value: Optional[float] = None  # the bar the zoom decomposes (self-check)
    statement: str = ""          # opaque/popularity center statement
    feature_explained: Optional[str] = None    # the M3 fraction line (fI / lightfm_ids)
    notes: List[str] = field(default_factory=list)  # method lines (md companion)


def _assert_close(got: float, expected: float, what: str, tol: float = _SELF_CHECK_TOL):
    """Loud exactness self-check — a breakdown that does not sum to the model's own
    score must never be written."""
    scale = max(1.0, abs(expected))
    if not math.isfinite(got) or abs(got - expected) > tol * scale:
        raise AssertionError(
            f"breakdown self-check FAILED ({what}): rendered parts sum to {got!r}, "
            f"model says {expected!r} — refusing to write a wrong explanation")


def _forward_score(model, user_id: int, item_id: int) -> float:
    with torch.no_grad():
        return float(model(torch.tensor([user_id]),
                           torch.tensor([[item_id]])).squeeze())


def _bias_lines(model, user_id: int, item_id: int) -> List[BreakdownLine]:
    """Disclosed bias terms when a model carries them (the scrutiny fleet is bias-free;
    handled anyway so the renderer never silently absorbs a score component)."""
    if not getattr(model, "use_bias", False):
        return []
    with torch.no_grad():
        u_b = float(model.user_bias(torch.tensor([user_id])).squeeze())
        i_b = float(model.item_bias(torch.tensor([item_id])).squeeze())
        g_b = float(model.global_bias.squeeze())
    return [BreakdownLine("user bias", u_b), BreakdownLine("item bias", i_b),
            BreakdownLine("global bias", g_b)]


# ---------------------------------------------------------------------------
# shared label helpers
# ---------------------------------------------------------------------------

def feature_code_labels(feature_fields: List[str], items_info: pd.DataFrame) -> List[str]:
    """Global-code -> human label, reconstructed exactly as build_feature_ids orders the
    vocabulary (per field: lexicographically sorted unique values, concatenated)."""
    labels = []
    for fld in feature_fields:
        short = fld[:-5] if fld.endswith("_name") else fld
        for value in sorted(items_info[fld].astype(str).unique().tolist()):
            labels.append(f"{short} = {value}")
    return labels


def item_description(items_info: pd.DataFrame, item_id: int,
                     max_fields: int = 3) -> str:
    """Short metadata line for the header (falls back to the raw id)."""
    df = items_info.set_index("item_id") if items_info.index.name != "item_id" else items_info
    if item_id not in df.index:
        return f"item {item_id}"
    row = df.loc[item_id]
    parts = []
    for col in ("product_type_name", "colour_group_name", "department_name",
                "section_name", "item", "genres", "title"):
        if col in row.index and pd.notna(row[col]) and len(parts) < max_fields:
            parts.append(str(row[col]))
    desc = " · ".join(parts) if parts else ""
    return f"item {item_id}" + (f" ({desc})" if desc else "")


def _proto_labels(naming_item, n_protos: int) -> List[str]:
    if naming_item is not None:
        return [f"p{k} · {n}" if n else f"p{k}"
                for k, n in enumerate(naming_item.labels_in_order(n_protos))]
    return [f"p{k}" for k in range(n_protos)]


# ---------------------------------------------------------------------------
# slots — one compute function per mechanism
# ---------------------------------------------------------------------------

def compute_breakdown_feature_item_proto(model, user_id: int, item_id: int,
                                         items_info: pd.DataFrame,
                                         feature_fields: List[str],
                                         naming_item=None,
                                         model_label: str = "fI-ProtoMF (feature_item_proto)",
                                         ) -> Breakdown:
    """fI slot: s_k = u_k·t*_k with the rank-inert Σ_k u_k baseline disclosed; zoom =
    exact per-row shares u_k*·c_{r,k*} of the top prototype (metadata rows + ID row)."""
    proto_fe = model.item_feature_extractor          # PrototypeEmbedding
    feat_embed = proto_fe.embedding_ext              # FeatureEmbedding
    with torch.no_grad():
        u = model.user_feature_extractor(torch.tensor([user_id])).squeeze(0)  # (K,)
        tstar = proto_fe(torch.tensor([item_id])).squeeze(0)                  # (K,)
    total = _forward_score(model, user_id, item_id)
    bias = _bias_lines(model, user_id, item_id)

    baseline = float(u.sum())
    s_disc = (u * (tstar - 1.0)).numpy()             # item-discriminating contributions
    _assert_close(baseline + float(s_disc.sum()) + sum(b.value for b in bias),
                  total, "Σ_k u_k + Σ_k u_k(t*_k−1) [+bias] vs forward score")

    labels = _proto_labels(naming_item, len(s_disc))
    lines = [BreakdownLine(labels[k], float(s_disc[k])) for k in range(len(s_disc))]

    # zoom: exact share decomposition of the top |contribution| prototype
    k_star = int(np.argmax(np.abs(s_disc)))
    rows = item_feature_rows(feat_embed, item_id)                    # (R, d)
    shares = per_feature_shares(rows, proto_fe.prototypes)           # (R, K)
    zoom_vals = (u[k_star] * shares[:, k_star]).detach().numpy()    # score units
    _assert_close(float(zoom_vals.sum()), float(s_disc[k_star]),
                  "Σ_r u_k*·c_{r,k*} vs the top prototype's bar")

    code_labels = feature_code_labels(feature_fields, items_info)
    item_codes = feat_embed.feature_ids[item_id].tolist()
    zoom_lines = [BreakdownLine(code_labels[c], float(zoom_vals[j]))
                  for j, c in enumerate(item_codes)]
    if feat_embed.use_id_feature:
        zoom_lines.append(BreakdownLine("item-ID row (own memory)",
                                        float(zoom_vals[-1]), is_id=True))

    # M3 feature-explained fraction over the WHOLE item-discriminating score
    with torch.no_grad():
        all_shares = per_feature_shares(rows, proto_fe.prototypes)   # (R, K)
        per_row_total = (all_shares * u.unsqueeze(0)).sum(dim=1)     # Σ_k u_k c_{r,k}
    disc_total = float(per_row_total.sum())
    id_part = float(per_row_total[-1]) if feat_embed.use_id_feature else 0.0
    feat_part = disc_total - id_part
    if abs(disc_total) > 1e-12:
        pct = 100.0 * feat_part / disc_total
        pct = abs(pct) if pct == 0 else pct   # normalize -0.0
        fe_line = (f"feature-explained {feat_part:+.4f} vs item-ID {id_part:+.4f} "
                   f"of the item-discriminating score {disc_total:+.4f} "
                   f"(features: {pct:.1f}%)")
    else:
        fe_line = "item-discriminating score ≈ 0 — fraction undefined"

    bd = Breakdown(
        model_label=model_label, mechanism="proto",
        user_id=user_id, item_id=item_id,
        item_desc=item_description(items_info, item_id),
        total_score=total,
        naming_route="intrinsic — cos(e_f, p_k), read from the model's parameters",
        baseline=baseline,
        lines=lines, lines_title="per-prototype contribution u_k·(t*_k − 1)",
        zoom_kind="shares",
        zoom_title=f"inside {_trunc(labels[k_star], 22)}: exact shares u_k·c(r,k)",
        zoom_lines=zoom_lines, zoom_parent_value=float(s_disc[k_star]),
        feature_explained=fe_line,
        notes=[
            "Shares are EXACT additive summands of the computed score (not counterfactual "
            "effects; rows are jointly normalized through ‖q_i‖ — removing one rescales the others).",
            "Negative values are legitimate anti-affinities and are rendered as such.",
            f"The rank-inert baseline Σ_k u_k = {baseline:+.4f} is identical for every item "
            "this user ranks; bars show the item-discriminating remainder.",
        ])
    bd.lines.extend(bias)
    return bd


def compute_breakdown_item_proto(model, user_id: int, item_id: int,
                                 items_info: pd.DataFrame,
                                 naming_item=None,
                                 top_k_exemplars: int = 8,
                                 model_label: str = "I-ProtoMF (item_proto)",
                                 ) -> Breakdown:
    """Host slot: identical contribution math (same shifted-cosine branch, same free u);
    zoom = the prototype's nearest catalog items — declared post-hoc."""
    proto_fe = model.item_feature_extractor          # PrototypeEmbedding (free ext)
    with torch.no_grad():
        u = model.user_feature_extractor(torch.tensor([user_id])).squeeze(0)
        tstar = proto_fe(torch.tensor([item_id])).squeeze(0)
    total = _forward_score(model, user_id, item_id)
    bias = _bias_lines(model, user_id, item_id)

    baseline = float(u.sum())
    s_disc = (u * (tstar - 1.0)).numpy()
    _assert_close(baseline + float(s_disc.sum()) + sum(b.value for b in bias),
                  total, "Σ_k u_k + Σ_k u_k(t*_k−1) [+bias] vs forward score")

    labels = _proto_labels(naming_item, len(s_disc))
    lines = [BreakdownLine(labels[k], float(s_disc[k])) for k in range(len(s_disc))]

    # zoom (post-hoc, declared): nearest catalog items to the top prototype
    k_star = int(np.argmax(np.abs(s_disc)))
    with torch.no_grad():
        emb = proto_fe.embedding_ext.embedding_layer.weight              # (n_items, d)
        p = proto_fe.prototypes[k_star:k_star + 1]                       # (1, d)
        sim = (1.0 + torch.nn.functional.cosine_similarity(
            emb, p.expand_as(emb), dim=1)).numpy()
    order = np.argsort(-sim)[:top_k_exemplars]
    zoom_lines = [BreakdownLine(item_description(items_info, int(i)), float(sim[i]))
                  for i in order]

    bd = Breakdown(
        model_label=model_label, mechanism="proto",
        user_id=user_id, item_id=item_id,
        item_desc=item_description(items_info, item_id),
        total_score=total,
        naming_route="post-hoc — lift over the prototype's top-k nearest items",
        baseline=baseline,
        lines=lines, lines_title="per-prototype contribution u_k·(t*_k − 1)",
        zoom_kind="exemplars",
        zoom_title=f"inside {_trunc(labels[k_star], 22)}: nearest items (post-hoc)",
        zoom_lines=zoom_lines, zoom_parent_value=None,   # post-hoc list, not a sum
        notes=[
            "The zoom panel is a POST-HOC interpretation (nearest items after training) — "
            "the model computes no feature-level quantity for it; this asymmetry vs the "
            "feature-grounded variant is the object of comparison, and is declared, not styled away.",
            "Negative values are legitimate anti-affinities and are rendered as such.",
            f"The rank-inert baseline Σ_k u_k = {baseline:+.4f} is identical for every item "
            "this user ranks; bars show the item-discriminating remainder.",
        ])
    bd.lines.extend(bias)
    return bd


def compute_breakdown_lightfm(model, user_id: int, item_id: int,
                              items_info: pd.DataFrame,
                              feature_fields: List[str],
                              model_label: str = "LightFM-style CBF (lightfm)",
                              ) -> Breakdown:
    """CBF slot: S = u·q_i = Σ_r u·e_r — exact per-feature bars, no prototype layer."""
    feat_embed = model.item_feature_extractor        # FeatureEmbedding
    with torch.no_grad():
        u = model.user_feature_extractor(torch.tensor([user_id])).squeeze(0)   # (d,)
        rows = item_feature_rows(feat_embed, item_id)                          # (R, d)
        contribs = (rows @ u).numpy()
    total = _forward_score(model, user_id, item_id)
    bias = _bias_lines(model, user_id, item_id)
    _assert_close(float(contribs.sum()) + sum(b.value for b in bias),
                  total, "Σ_r u·e_r [+bias] vs forward score")

    code_labels = feature_code_labels(feature_fields, items_info)
    item_codes = feat_embed.feature_ids[item_id].tolist()
    lines = [BreakdownLine(code_labels[c], float(contribs[j]))
             for j, c in enumerate(item_codes)]
    fe_line = None
    if feat_embed.use_id_feature:
        lines.append(BreakdownLine("item-ID row (own memory)",
                                   float(contribs[-1]), is_id=True))
        id_part = float(contribs[-1])
        feat_part = float(contribs.sum()) - id_part
        if abs(contribs.sum()) > 1e-12:
            pct = 100.0 * feat_part / float(contribs.sum())
            pct = abs(pct) if pct == 0 else pct   # normalize -0.0
            fe_line = (f"feature-explained {feat_part:+.4f} vs item-ID {id_part:+.4f} "
                       f"(features: {pct:.1f}%)")

    bd = Breakdown(
        model_label=model_label, mechanism="flat",
        user_id=user_id, item_id=item_id,
        item_desc=item_description(items_info, item_id),
        total_score=total, naming_route="n/a (no prototype layer)",
        lines=lines, lines_title="exact per-feature contribution u·e_r",
        zoom_kind="statement",
        statement=("No prototype layer: feature-level attribution is this model's entire "
                   "explanation mechanism. The absence of a prototype-shaped reading is "
                   "part of the comparison."),
        feature_explained=fe_line,
        notes=["Contributions are exact additive summands of the score S = u·(Σ_r e_r).",
               "Negative values are legitimate anti-affinities and are rendered as such."])
    bd.lines.extend(bias)
    return bd


def compute_breakdown_mf(model, user_id: int, item_id: int,
                         items_info: pd.DataFrame,
                         model_label: str = "MF (mf)") -> Breakdown:
    """The honest-asymmetry row: total score only."""
    total = _forward_score(model, user_id, item_id)
    return Breakdown(
        model_label=model_label, mechanism="opaque",
        user_id=user_id, item_id=item_id,
        item_desc=item_description(items_info, item_id),
        total_score=total, naming_route="n/a",
        zoom_kind="statement",
        statement=("Latent-factor dot product: this model computes no prototype- or "
                   "feature-level quantity — there is no decomposable explanation "
                   "surface. This row exists so the comparison states that honestly "
                   "rather than omitting the model."),
        notes=["Total score verified against the model forward pass."])


def compute_breakdown_popularity(dataset_dir: str, user_id: int,
                                 item_id: Optional[int] = None) -> Breakdown:
    """Non-personalized reference: global popularity rank/percentile from the train split."""
    train = pd.read_csv(os.path.join(dataset_dir, "listening_history_train.csv"))
    counts = train["item_id"].value_counts()
    if item_id is None:
        item_id = int(counts.index[0])
    count = int(counts.get(item_id, 0))
    n_items_ranked = len(counts)
    rank = int((counts > count).sum()) + 1
    top_share = 100.0 * rank / max(1, n_items_ranked)
    items_info_path = os.path.join(dataset_dir, "item_features.csv")
    items_info = (pd.read_csv(items_info_path) if os.path.exists(items_info_path)
                  else pd.read_csv(os.path.join(dataset_dir, "item_ids.csv")))
    return Breakdown(
        model_label="Popularity reference", mechanism="popularity",
        user_id=user_id, item_id=int(item_id),
        item_desc=item_description(items_info, int(item_id)),
        total_score=float(count), naming_route="n/a",
        zoom_kind="statement",
        statement=(f"Non-personalized: every user receives the same ranking. This item was "
                   f"purchased {count}× in the train window — popularity rank {rank} of "
                   f"{n_items_ranked} (top {top_share:.1f}%). Your history played no role."),
        notes=["Score = raw train-split interaction count (rank-based scorer in eval, F-S0-08)."])


# ---------------------------------------------------------------------------
# rendering — one figure + one markdown, identical frame for every model
# ---------------------------------------------------------------------------

def _style_axis(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=7)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _barh(ax, lines: List[BreakdownLine], title: str):
    """Signed horizontal bars, top _TOP_LINES by |value|, remainder folded and stated."""
    shown = sorted(lines, key=lambda l: -abs(l.value))[:_TOP_LINES]
    shown = list(reversed(shown))                      # largest on top
    rest = [l for l in lines if l not in shown]
    rest_sum = sum(l.value for l in rest)
    if rest:
        plural = "s" if len(rest) != 1 else ""
        shown.insert(0, BreakdownLine(f"(+ {len(rest)} smaller line{plural})", rest_sum))
    y = np.arange(len(shown))
    vals = [l.value for l in shown]
    colors = [_POS if v >= 0 else _NEG for v in vals]
    bars = ax.barh(y, vals, height=0.62, color=colors, zorder=3)
    for bar, line in zip(bars, shown):
        if line.is_id:
            bar.set_hatch("///")
            bar.set_edgecolor("white")
            bar.set_linewidth(0.5)
    ax.axvline(0, color=_INK2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_trunc(l.label, 34) for l in shown], fontsize=7, color=_INK)
    ax.margins(x=0.28, y=0.04)   # headroom so end labels never collide with tick labels
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:+.3f}", (v, yi), fontsize=6, color=_INK2,
                    xytext=(3 if v >= 0 else -3, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center")
    ax.set_title(title, fontsize=8, color=_INK, loc="left")
    _style_axis(ax)


def _text_panel(ax, title: str, body_lines: List[str], wrap_width: int = 44):
    import textwrap
    ax.axis("off")
    ax.set_title(title, fontsize=8, color=_INK, loc="left")
    wrapped = []
    for line in body_lines:
        wrapped.extend(textwrap.wrap(line, wrap_width) or [""])
    ax.text(0.0, 0.95, "\n".join(wrapped), transform=ax.transAxes,
            fontsize=7, color=_INK, va="top", ha="left")


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def render_breakdown_figure(bd: Breakdown, out_path: str) -> str:
    # statement-only mechanisms get a compact frame (same visual language, no blank canvas)
    statement_only = bd.mechanism in ("opaque", "popularity")
    fig = plt.figure(figsize=(_FIGSIZE[0], 2.0) if statement_only else _FIGSIZE, dpi=200)
    header = f"Why {bd.item_desc} for user {bd.user_id}?  —  {bd.model_label}"
    if bd.mechanism == "popularity":
        score_line = f"train-split popularity count = {bd.total_score:.0f}"
    elif bd.baseline is not None:
        score_line = (f"score S = {bd.total_score:+.4f}  =  rank-inert baseline "
                      f"Σ_k u_k = {bd.baseline:+.4f} (same for every item)  +  "
                      f"item-discriminating {bd.total_score - bd.baseline:+.4f}")
    else:
        score_line = f"score S = {bd.total_score:+.4f}"
    fig.text(0.03, 0.92 if statement_only else 0.965, header,
             fontsize=9.5, fontweight="bold", color=_INK)
    fig.text(0.03, 0.83 if statement_only else 0.925, score_line,
             fontsize=7.5, color=_INK2)

    if statement_only:
        ax = fig.add_axes([0.04, 0.24, 0.92, 0.44])
        _text_panel(ax, "", [bd.statement], wrap_width=88)
    elif bd.mechanism == "flat":
        axl = fig.add_axes([0.30, 0.16, 0.36, 0.70])
        _barh(axl, bd.lines, bd.lines_title)
        axr = fig.add_axes([0.72, 0.16, 0.26, 0.70])
        _text_panel(axr, "mechanism", [bd.statement], wrap_width=33)
    else:  # proto
        axl = fig.add_axes([0.24, 0.16, 0.30, 0.70])
        _barh(axl, bd.lines, bd.lines_title)
        if bd.zoom_kind == "shares":
            axr = fig.add_axes([0.72, 0.16, 0.26, 0.70])
            _barh(axr, bd.zoom_lines, _trunc(bd.zoom_title, 60))
        else:  # exemplars
            axr = fig.add_axes([0.60, 0.16, 0.38, 0.70])
            body = [f"{_trunc(l.label, 40)}  (1+cos = {l.value:.3f})"
                    for l in bd.zoom_lines]
            _text_panel(axr, _trunc(bd.zoom_title, 70), body, wrap_width=52)

    footer = f"prototype naming: {bd.naming_route}"
    if bd.feature_explained:
        footer += f"\n{bd.feature_explained}"
    fig.text(0.03, 0.015, footer, fontsize=6.5, color=_INK2, va="bottom")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_breakdown_text(bd: Breakdown) -> str:
    md = [f"# Why {bd.item_desc} for user {bd.user_id}? — {bd.model_label}", ""]
    if bd.mechanism == "popularity":
        md.append(f"Train-split popularity count: **{bd.total_score:.0f}**")
    elif bd.baseline is not None:
        md.append(f"Score **S = {bd.total_score:+.4f}** = rank-inert baseline "
                  f"Σ_k u_k = **{bd.baseline:+.4f}** (identical for every item this user "
                  f"ranks) + item-discriminating **{bd.total_score - bd.baseline:+.4f}**.")
    else:
        md.append(f"Score **S = {bd.total_score:+.4f}**.")
    md.append("")
    if bd.lines:
        md += [f"## {bd.lines_title}", "", "| line | contribution |", "|---|---|"]
        for l in sorted(bd.lines, key=lambda l: -abs(l.value)):
            tag = " *(ID row)*" if l.is_id else ""
            md.append(f"| {l.label}{tag} | {l.value:+.4f} |")
        md.append("")
    if bd.zoom_kind == "shares" and bd.zoom_lines:
        md += [f"## {bd.zoom_title}", "", "| row | share (score units) |", "|---|---|"]
        for l in bd.zoom_lines:
            tag = " *(ID row)*" if l.is_id else ""
            md.append(f"| {l.label}{tag} | {l.value:+.4f} |")
        md.append(f"\nΣ = {sum(l.value for l in bd.zoom_lines):+.4f} "
                  f"(= that prototype's bar, exactly)")
        md.append("")
    elif bd.zoom_kind == "exemplars" and bd.zoom_lines:
        md += [f"## {bd.zoom_title}", ""]
        for l in bd.zoom_lines:
            md.append(f"- {l.label} (1+cos = {l.value:.3f})")
        md.append("")
    if bd.statement:
        md += ["## Mechanism", "", bd.statement, ""]
    if bd.feature_explained:
        md += [f"**{bd.feature_explained}**", ""]
    md += ["## Method notes", ""]
    md += [f"- {n}" for n in bd.notes]
    md.append(f"- Prototype naming route: {bd.naming_route}.")
    md.append("- Self-check passed: the rendered parts sum to the model's own forward "
              "score (asserted before writing).")
    return "\n".join(md) + "\n"


def write_breakdown(bd: Breakdown, out_dir: str, stem: Optional[str] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = stem or f"breakdown_user{bd.user_id}_item{bd.item_id}"
    render_breakdown_figure(bd, os.path.join(out_dir, f"{stem}.png"))
    with open(os.path.join(out_dir, f"{stem}.md"), "w") as f:
        f.write(render_breakdown_text(bd))
    return os.path.join(out_dir, stem)


# ---------------------------------------------------------------------------
# orchestration (results-dir entry point + CLI)
# ---------------------------------------------------------------------------

def top1_item_for_user(model, user_id: int, n_items: int,
                       batch: int = 8192) -> int:
    """The model's top-scoring item for a user (full catalog, no exclusion — parity with
    the weight_viz convention)."""
    best_val, best_idx = -float("inf"), 0
    with torch.no_grad():
        u = torch.tensor([user_id])
        for start in range(0, n_items, batch):
            idx = torch.arange(start, min(start + batch, n_items))
            scores = model(u, idx.unsqueeze(0)).squeeze(0)
            m = int(torch.argmax(scores))
            if float(scores[m]) > best_val:
                best_val, best_idx = float(scores[m]), int(idx[m])
    return best_idx


def render_for_results_dir(results_dir: str, user_id: int,
                           item_id: Optional[int] = None,
                           out_dir: Optional[str] = None,
                           data_dir: Optional[str] = None) -> str:
    """Load a trained combo and render its breakdown for (user, item). ``item_id`` defaults
    to the model's top-1 recommendation. ``data_dir`` overrides DATA_PATH/<dataset>."""
    from utilities.explanations.loader import load_recsys_from_results_dir
    from utilities.explanations.items_info import load_items_info
    from utilities.explanations.naming import (get_naming_config,
                                               intrinsic_naming_config,
                                               name_prototypes_from_weights,
                                               name_prototypes_intrinsic)
    from utilities.explanations.accessor import get_accessor

    model, config, metadata = load_recsys_from_results_dir(results_dir, data_dir=data_dir)
    model_type = metadata["model"]
    dataset = metadata["dataset"]
    items_info = load_items_info(dataset, dataset_dir=data_dir)
    if item_id is None:
        item_id = top1_item_for_user(model, user_id, model.n_items)
    out_dir = out_dir or os.path.join(results_dir, "explanations", "breakdown")

    base_cfg = get_naming_config(dataset, items_info)
    if model_type.startswith("feature_item_proto"):
        feature_fields = config["ft_ext_param"]["item_ft_ext_param"]["feature_fields"]
        accessor = get_accessor("feature_item_proto", model)
        naming = None
        try:
            naming = name_prototypes_intrinsic(
                accessor.feature_value_embeddings(), accessor.item_prototypes(),
                feature_fields, items_info,
                intrinsic_naming_config(base_cfg, feature_fields), side="item")
        except Exception as e:
            print(f"[breakdown] ⚠ intrinsic naming failed: {e!r}")
        bd = compute_breakdown_feature_item_proto(
            model, user_id, item_id, items_info, feature_fields, naming,
            model_label=f"fI-ProtoMF ({model_type})")
    elif model_type == "item_proto":
        accessor = get_accessor("item_proto", model)
        naming = None
        try:
            naming = name_prototypes_from_weights(
                accessor.item_to_item_proto_sim(), items_info, base_cfg, side="item")
        except Exception as e:
            print(f"[breakdown] ⚠ post-hoc naming failed: {e!r}")
        bd = compute_breakdown_item_proto(model, user_id, item_id, items_info, naming)
    elif model_type.startswith("lightfm"):
        feature_fields = config["ft_ext_param"]["item_ft_ext_param"]["feature_fields"]
        bd = compute_breakdown_lightfm(model, user_id, item_id, items_info,
                                       feature_fields,
                                       model_label=f"LightFM-style CBF ({model_type})")
    elif model_type == "mf":
        bd = compute_breakdown_mf(model, user_id, item_id, items_info)
    elif model_type in ("user_proto", "user_item_proto"):
        raise NotImplementedError(
            f"breakdown slot for {model_type} is deliberately deferred to the lineage "
            "stage that compares it (fU / fUfI) — dc01 SC.5 gate decision, 2026-07-12")
    else:
        raise NotImplementedError(f"no breakdown slot for model {model_type!r}")

    return write_breakdown(bd, out_dir)


def main():
    p = argparse.ArgumentParser(description="Render a user-perspective recommendation "
                                            "breakdown for a trained combo (or the "
                                            "popularity reference).")
    p.add_argument("--results-dir", help="combo folder (best_model.pth/config/metadata)")
    p.add_argument("--user", type=int, required=True)
    p.add_argument("--item", type=int, default=None,
                   help="default: the model's top-1 recommendation for the user")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--data", default=None, help="dataset dir override (default: DATA_PATH/<dataset>)")
    p.add_argument("--popularity", action="store_true",
                   help="render the popularity-reference breakdown (needs --data, no results dir)")
    args = p.parse_args()

    if args.popularity:
        if not args.data:
            p.error("--popularity requires --data <dataset dir>")
        bd = compute_breakdown_popularity(args.data, args.user, args.item)
        out = write_breakdown(bd, args.out_dir or os.path.join(args.data, "breakdown"))
    else:
        if not args.results_dir:
            p.error("--results-dir required (or use --popularity)")
        out = render_for_results_dir(args.results_dir, args.user, args.item,
                                     args.out_dir, args.data)
    print(f"[breakdown] written → {out}.png / .md")


if __name__ == "__main__":
    main()
