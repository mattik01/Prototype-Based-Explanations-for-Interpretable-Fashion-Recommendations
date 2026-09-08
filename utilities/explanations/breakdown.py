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
Slots built at the dc05 build (2026-07-12, the fU stage this frame was waiting for):
  - ``feature_user_proto`` (fU, incl. ``_noid`` — the SLOT handles the noid arm; the standalone
    pipeline keys on the headline model name, so ablation-arm artifacts come via this module's
    CLI, the host convention since dc01's F-DC01-11 gate) — per-community personalized bars t_l·(u*_l−1)
    with the **item baseline B(t) = 1ᵀt disclosed as its own non-personalized line** (on the U
    host the +1 mass is a per-ITEM scalar and is NOT rank-inert — design doc §3.4; the wording
    differs from fI's rank-inert line on purpose), zoom = exact per-PURCHASE shares of the top
    community (word-level zoom in the md companion — two exact regroupings of one sum),
  - ``user_proto`` (the fU host) — identical contribution math + the same B(t) line (GR7-fair),
    zoom = the community's top-k nearest users' consumed-item lift — declared post-hoc.
The ``user_item_proto`` slot stays deferred to the fUfI merge stage.

Honesty scope (SC.5 gate decision, 2026-07-12): the artifacts carry ARITHMETIC honesty
only — the +1 rank-inert baseline shown (never silently absorbed), signed contributions
(negative = legitimate anti-affinity), the item-ID row as its own line, and the
feature-explained fraction. Effect-epistemics (share determinacy, user-coefficient gauge)
are deliberately NOT rendered here; they belong to the thesis's dedicated hidden-effects
section (design doc §3.4 / §I.1 re-scope notes, same date).

Every artifact SELF-CHECKS before it is written: the sum of its rendered parts is
asserted against the model's own forward score (and the zoom against its parent bar).

Cosine-offset generalization (2026-08-17 special experiments, cosbias2x2): the split is no
longer hardcoded to the shifted cosine. Every supported ``cosine_type`` is AFFINE in the
cosine, act = a·cos + c, and the baseline/personalized split generalizes to
baseline = c·1ᵀ(other side) and personalized = (other side)·(act − c). CRITICAL: the
arithmetic self-check is c-INVARIANT — t·(u*−c) + c·1ᵀt ≡ t·u* for ANY c — so the offset
is read from the run's config (via the PrototypeEmbedding the loader built from it), never
assumed and never "verified" by the sum. With c = 0 (standard cosine) there IS no baseline
term, and no baseline line is rendered. ``use_bias`` runs additionally disclose
b_u + b_i + b_g as their own lines (b_i rank-relevant, b_u/b_g rank-inert).
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

from utilities.explanations.feature_readout import (item_feature_rows, item_meta_codes,
                                                    per_feature_shares)

# --- shared visual language (identical for every model — the fairness contract) ---
_POS = "#2a78d6"        # positive contribution (diverging cool pole)
_NEG = "#e34948"        # negative contribution (diverging warm pole)
_INK = "#0b0b0b"        # primary text
_INK2 = "#52514e"       # secondary text
_GRID = "#d9d8d4"       # recessive grid
_FIGSIZE = (7.2, 4.6)   # compact, repeated-use target
_TOP_LINES = 6          # bars shown per panel (rest folded into a stated remainder)

_SELF_CHECK_TOL = 5e-4  # relative; float32 sums over K·R terms

# --- cosine-offset generalization (2026-08-17 special experiments, cosbias2x2) ---
# act = a·cos + c per cosine_type. per_feature_shares rows sum to the RAW cosine, so every
# intrinsic zoom scales its shares by a (act − c = a·cos); the disclosed baseline mass is
# c·1ᵀ(other side) and vanishes (line absent, not rendered as 0) when c = 0.
_COSINE_AFFINE = {
    "shifted": (1.0, 1.0),          # act = 1 + cos ∈ [0, 2] — the fleet default
    "standard": (1.0, 0.0),         # act = cos ∈ [−1, 1] — NO baseline mass exists
    "shifted_and_div": (0.5, 0.5),  # act = (1 + cos)/2 ∈ [0, 1] — half-scale baseline mass
}
_COSINE_UNIT_LABEL = {"shifted": "1+cos", "standard": "cos", "shifted_and_div": "(1+cos)/2"}
# dc07 (2026-09-08): the membership family (cosine_type 'softmax' / 'sigmoid') is deliberately
# ABSENT from this table — softmax(⟨q,P⟩/τ) is not an affine map of cos, so there is no share
# zoom on m; cosine_affine() raises for it (stage-1 contract). The membership-family analogue
# is a LOG-ODDS decomposition (log m_l − log m_k = Σ_f c_f⟨e_f, p_l−p_k⟩/τ) rendered by a
# separate path (dc07 stage-2 deliverable D3). Never add a fake affine entry here.


def cosine_affine(proto_fe):
    """(scale a, offset c, cosine_type) of a side's similarity act = a·cos + c.

    Read from the ``PrototypeEmbedding`` the loader built FROM THE RUN'S CONFIG — the
    only honest source: with a wrong c the sum self-check still passes identically
    (t·(u*−c) + c·1ᵀt ≡ t·u* for any c), so the offset is a config fact, not a checkable
    one. The 'shifted' fallback mirrors the factory's own default for a missing key."""
    ct = getattr(proto_fe, "cosine_type", "shifted")
    if ct not in _COSINE_AFFINE:
        raise ValueError(f"unknown cosine_type {ct!r} — refusing to split the score")
    a, c = _COSINE_AFFINE[ct]
    return a, c, ct


def _coef(c: float) -> str:
    """Baseline-formula coefficient: '' for c=1 (the historical 1ᵀt / Σ_k u_k wording),
    '0.5·' for shifted_and_div."""
    return "" if c == 1.0 else f"{c:g}·"


def _act_minus_c(var: str, c: float) -> str:
    """Contribution-term formula: '(t*_k − 1)' under shifted, 't*_k' under standard (c=0)."""
    return var if c == 0.0 else f"({var} − {c:g})"


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
    baseline: Optional[float] = None          # proto slot: the c-shift mass (disclosed);
    # None when c = 0 (standard cosine — no baseline exists, the line is entirely absent)
    # 'rank_inert' (I host: c·Σ_k u_k, item-independent) | 'item_intercept' (U host: B(t) = c·1ᵀt,
    # a per-item scalar that participates in item ranking — dc05 §3.4 mandatory wording change)
    baseline_kind: str = "rank_inert"
    cos_offset: float = 1.0      # c of the side's affine sim (renders the baseline formula)
    bias_total: Optional[float] = None   # Σ of disclosed bias lines (use_bias runs only)
    lines: List[BreakdownLine] = field(default_factory=list)   # left panel
    lines_title: str = ""
    zoom_kind: str = "none"      # 'shares' | 'exemplars' | 'statement' | 'none'
    zoom_title: str = ""
    zoom_lines: List[BreakdownLine] = field(default_factory=list)
    zoom_parent_value: Optional[float] = None  # the bar the zoom decomposes (self-check)
    zoom_value_label: str = "1+cos"            # unit label for 'exemplars' zoom values
    # optional SECOND exact reading of the same zoomed bar (md companion only — the two
    # regroupings are alternative zooms of ONE sum, never added; dc05 word-level zoom)
    alt_zoom_title: str = ""
    alt_zoom_lines: List[BreakdownLine] = field(default_factory=list)
    statement: str = ""          # opaque/popularity center statement
    feature_explained: Optional[str] = None    # the M3 fraction line (fI / lightfm_ids / fU)
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


# Rendered whenever bias lines exist (cosbias2x2 arms): the three terms are NOT epistemic
# equals — b_i is an explicit popularity dial and participates in item ranking; b_u and b_g
# are per-user/global constants, identical for every item this user ranks.
_BIAS_NOTE = ("Bias terms are disclosed as their own lines: the item bias b_i is an explicit "
              "per-item popularity dial and IS rank-relevant; the user bias b_u and the "
              "global bias are per-user/global constants — rank-inert, identical for every "
              "item this user ranks.")


def _bias_lines(model, user_id: int, item_id: int) -> List[BreakdownLine]:
    """Disclosed bias terms when a model carries them (the scrutiny fleet is bias-free; the
    cosbias2x2 special arms retrain with use_bias=1) — the renderer never silently absorbs
    a score component. Labels carry the rank-relevance split (see _BIAS_NOTE)."""
    if not getattr(model, "use_bias", False):
        return []
    with torch.no_grad():
        u_b = float(model.user_bias(torch.tensor([user_id])).squeeze())
        i_b = float(model.item_bias(torch.tensor([item_id])).squeeze())
        g_b = float(model.global_bias.squeeze())
    return [BreakdownLine("user bias b_u (rank-inert)", u_b),
            BreakdownLine("item bias b_i (popularity dial — rank-relevant)", i_b),
            BreakdownLine("global bias (rank-inert)", g_b)]


def _attach_bias(bd: "Breakdown", bias: List[BreakdownLine]) -> "Breakdown":
    """Uniform bias attachment for every decomposing slot: lines into the bars panel,
    Σ into the header split, the rank-relevance note into the md companion."""
    if bias:
        bd.lines.extend(bias)
        bd.bias_total = sum(b.value for b in bias)
        bd.notes.append(_BIAS_NOTE)
    return bd


# ---------------------------------------------------------------------------
# shared label helpers
# ---------------------------------------------------------------------------

def feature_code_labels(feature_fields: List[str], items_info: pd.DataFrame,
                        dataset_dir: str = None,
                        feature_layout: str = "fixed") -> List[str]:
    """Global-code -> human label. With ``dataset_dir`` given (the in-repo callers), labels
    come from the builder-owned ``build_code_to_field_value`` — the single source of truth,
    correct on BOTH layouts (F-DC05-13). The legacy path (no dataset_dir) reconstructs the
    FIXED-layout vocabulary from ``items_info`` (identical recipe by construction) and
    refuses the bags layout rather than mislabel."""
    if dataset_dir is not None:
        from feature_extraction.feature_ids import build_code_to_field_value
        try:
            code_to_fv = build_code_to_field_value(dataset_dir, feature_fields, feature_layout)
            return [f"{fld[:-5] if fld.endswith('_name') else fld} = {value}"
                    for fld, value in code_to_fv]
        except Exception:
            # Fixed layout: the items_info reconstruction below is the identical recipe —
            # degrade gracefully (toy results dirs in tests). Bags: fall through to the
            # refusal below — a fixed-recipe fallback would silently mislabel (F-DC05-13).
            if feature_layout != "fixed":
                raise
    if feature_layout != "fixed":
        raise ValueError(
            "feature_code_labels needs dataset_dir for the bags layout — the items_info "
            "reconstruction is fixed-layout-only (F-DC05-13)")
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
                                         dataset_dir: str = None,
                                         feature_layout: str = "fixed",
                                         ) -> Breakdown:
    """fI slot: s_k = u_k·t*_k with the rank-inert c·Σ_k u_k baseline disclosed (offset c
    from the run's cosine_type — line absent when c = 0, cosbias2x2); zoom =
    exact per-row shares a·u_k*·c_{r,k*} of the top prototype (metadata rows + ID row).
    ``dataset_dir``/``feature_layout`` route code→labels through the builder-owned vocab
    (F-DC05-13); dataset_dir=None keeps the legacy fixed-layout items_info reconstruction."""
    proto_fe = model.item_feature_extractor          # PrototypeEmbedding
    feat_embed = proto_fe.embedding_ext              # FeatureEmbedding
    with torch.no_grad():
        u = model.user_feature_extractor(torch.tensor([user_id])).squeeze(0)  # (K,)
        tstar = proto_fe(torch.tensor([item_id])).squeeze(0)                  # (K,)
    total = _forward_score(model, user_id, item_id)
    bias = _bias_lines(model, user_id, item_id)

    # cosine-offset generalization, 2026-08-17 special experiments (cosbias2x2):
    # t*_k = a·cos + c from the run's config — baseline c·Σ_k u_k, bars u_k·(t*_k − c).
    a_sc, c_off, _ct = cosine_affine(proto_fe)
    baseline = c_off * float(u.sum()) if c_off != 0.0 else None
    s_disc = (u * (tstar - c_off)).numpy()           # item-discriminating contributions
    _assert_close((baseline or 0.0) + float(s_disc.sum()) + sum(b.value for b in bias),
                  total, f"{_coef(c_off)}Σ_k u_k + Σ_k u_k·{_act_minus_c('t*_k', c_off)} "
                  "[+bias] vs forward score")

    labels = _proto_labels(naming_item, len(s_disc))
    lines = [BreakdownLine(labels[k], float(s_disc[k])) for k in range(len(s_disc))]

    # zoom: exact share decomposition of the top |contribution| prototype. Shares sum to the
    # RAW cosine, so the affine scale a applies: t*_k − c = a·cos = a·Σ_r c_{r,k}.
    k_star = int(np.argmax(np.abs(s_disc)))
    rows = item_feature_rows(feat_embed, item_id)                    # (R, d)
    shares = per_feature_shares(rows, proto_fe.prototypes)           # (R, K)
    zoom_vals = (a_sc * u[k_star] * shares[:, k_star]).detach().numpy()   # score units
    _assert_close(float(zoom_vals.sum()), float(s_disc[k_star]),
                  "Σ_r a·u_k*·c_{r,k*} vs the top prototype's bar")

    code_labels = feature_code_labels(feature_fields, items_info,
                                      dataset_dir=dataset_dir,
                                      feature_layout=feature_layout)
    item_codes = item_meta_codes(feat_embed, item_id)   # bags: padding filtered (F-DC05-13)
    zoom_lines = [BreakdownLine(code_labels[c], float(zoom_vals[j]))
                  for j, c in enumerate(item_codes)]
    if feat_embed.use_id_feature:
        zoom_lines.append(BreakdownLine("item-ID row (own memory)",
                                        float(zoom_vals[-1]), is_id=True))

    # M3 feature-explained fraction over the WHOLE item-discriminating score
    with torch.no_grad():
        all_shares = per_feature_shares(rows, proto_fe.prototypes)   # (R, K)
        per_row_total = a_sc * (all_shares * u.unsqueeze(0)).sum(dim=1)  # Σ_k a·u_k·c_{r,k}
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
        baseline=baseline, cos_offset=c_off,
        lines=lines,
        lines_title=f"per-prototype contribution u_k·{_act_minus_c('t*_k', c_off)}",
        zoom_kind="shares",
        zoom_title=f"inside {_trunc(labels[k_star], 22)}: exact shares "
                   f"{'' if a_sc == 1.0 else f'{a_sc:g}·'}u_k·c(r,k)",
        zoom_lines=zoom_lines, zoom_parent_value=float(s_disc[k_star]),
        feature_explained=fe_line,
        notes=[
            "Shares are EXACT additive summands of the computed score (not counterfactual "
            "effects; rows are jointly normalized through ‖q_i‖ — removing one rescales the others).",
            "Negative values are legitimate anti-affinities and are rendered as such.",
        ])
    if baseline is not None:
        bd.notes.append(
            f"The rank-inert baseline {_coef(c_off)}Σ_k u_k = {baseline:+.4f} is identical "
            "for every item this user ranks; bars show the item-discriminating remainder.")
    else:
        bd.notes.append(
            "Standard cosine (offset c = 0): the similarity carries no baseline mass — "
            "the bars ARE the whole dot-product score; no baseline line exists.")
    return _attach_bias(bd, bias)


def compute_breakdown_item_proto(model, user_id: int, item_id: int,
                                 items_info: pd.DataFrame,
                                 naming_item=None,
                                 top_k_exemplars: int = 8,
                                 model_label: str = "I-ProtoMF (item_proto)",
                                 ) -> Breakdown:
    """Host slot: identical contribution math (same cosine branch — offset per the run's
    config, cosbias2x2 — same free u); zoom = the prototype's nearest catalog items —
    declared post-hoc, in the model's own similarity units."""
    proto_fe = model.item_feature_extractor          # PrototypeEmbedding (free ext)
    with torch.no_grad():
        u = model.user_feature_extractor(torch.tensor([user_id])).squeeze(0)
        tstar = proto_fe(torch.tensor([item_id])).squeeze(0)
    total = _forward_score(model, user_id, item_id)
    bias = _bias_lines(model, user_id, item_id)

    # cosine-offset generalization, 2026-08-17 special experiments (cosbias2x2):
    # identical split math to the fI slot, offset read from the run's config.
    a_sc, c_off, ct = cosine_affine(proto_fe)
    baseline = c_off * float(u.sum()) if c_off != 0.0 else None
    s_disc = (u * (tstar - c_off)).numpy()
    _assert_close((baseline or 0.0) + float(s_disc.sum()) + sum(b.value for b in bias),
                  total, f"{_coef(c_off)}Σ_k u_k + Σ_k u_k·{_act_minus_c('t*_k', c_off)} "
                  "[+bias] vs forward score")

    labels = _proto_labels(naming_item, len(s_disc))
    lines = [BreakdownLine(labels[k], float(s_disc[k])) for k in range(len(s_disc))]

    # zoom (post-hoc, declared): nearest catalog items to the top prototype — displayed in
    # the model's OWN similarity units (a·cos + c; ordering is c/a-invariant)
    k_star = int(np.argmax(np.abs(s_disc)))
    with torch.no_grad():
        emb = proto_fe.embedding_ext.embedding_layer.weight              # (n_items, d)
        p = proto_fe.prototypes[k_star:k_star + 1]                       # (1, d)
        sim = (c_off + a_sc * torch.nn.functional.cosine_similarity(
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
        baseline=baseline, cos_offset=c_off,
        lines=lines,
        lines_title=f"per-prototype contribution u_k·{_act_minus_c('t*_k', c_off)}",
        zoom_kind="exemplars",
        zoom_title=f"inside {_trunc(labels[k_star], 22)}: nearest items (post-hoc)",
        zoom_lines=zoom_lines, zoom_parent_value=None,   # post-hoc list, not a sum
        zoom_value_label=_COSINE_UNIT_LABEL[ct],
        notes=[
            "The zoom panel is a POST-HOC interpretation (nearest items after training) — "
            "the model computes no feature-level quantity for it; this asymmetry vs the "
            "feature-grounded variant is the object of comparison, and is declared, not styled away.",
            "Negative values are legitimate anti-affinities and are rendered as such.",
        ])
    if baseline is not None:
        bd.notes.append(
            f"The rank-inert baseline {_coef(c_off)}Σ_k u_k = {baseline:+.4f} is identical "
            "for every item this user ranks; bars show the item-discriminating remainder.")
    else:
        bd.notes.append(
            "Standard cosine (offset c = 0): the similarity carries no baseline mass — "
            "the bars ARE the whole dot-product score; no baseline line exists.")
    return _attach_bias(bd, bias)


def compute_breakdown_feature_user_proto(model, user_id: int, item_id: int,
                                         items_info: pd.DataFrame,
                                         feature_fields: List[str],
                                         dataset_dir: str,
                                         naming_user=None,
                                         model_label: str = "fU-ProtoMF (feature_user_proto)",
                                         feature_layout: str = "fixed",
                                         ) -> Breakdown:
    """fU slot (dc05 §3.4, ratified defaults): s_l = t_l·u*_l rendered as the personalized part
    t_l·(u*_l−c) per taste community + the disclosed item baseline B(t) = c·1ᵀt (NOT rank-inert
    on this host; offset c from the run's cosine_type — line absent when c = 0, cosbias2x2);
    zoom = exact per-PURCHASE shares of the top community (figure default), with the
    word-level regrouping of the same sum in the md companion (two exact readings, never added).
    ``feature_layout`` (from the saved config) selects the item builder feeding the per-purchase
    regrouping and the code→label source (F-DC05-13; 'bags' = ml-1m)."""
    from feature_extraction.feature_ids import build_feature_bags, build_feature_ids
    from utilities.explanations.history_readout import (per_purchase_rows, user_history_rows,
                                                        user_train_items)

    proto_fe = model.user_feature_extractor          # PrototypeEmbedding
    hist_embed = proto_fe.embedding_ext              # HistoryFeatureEmbedding
    with torch.no_grad():
        ustar = proto_fe(torch.tensor([user_id])).squeeze(0)                   # (K_u,)
        t = model.item_feature_extractor(torch.tensor([item_id])).squeeze(0)   # (K_u,)
    total = _forward_score(model, user_id, item_id)
    bias = _bias_lines(model, user_id, item_id)

    # cosine-offset generalization, 2026-08-17 special experiments (cosbias2x2):
    # u*_l = a·cos + c from the run's config — B(t) = c·1ᵀt, bars t_l·(u*_l − c).
    a_sc, c_off, _ct = cosine_affine(proto_fe)
    baseline = c_off * float(t.sum()) if c_off != 0.0 else None   # B(t) — NOT rank-inert
    s_pers = (t * (ustar - c_off)).numpy()           # personalized contributions
    _assert_close((baseline or 0.0) + float(s_pers.sum()) + sum(b.value for b in bias),
                  total, f"B(t) = {_coef(c_off)}1ᵀt + Σ_l t_l·{_act_minus_c('u*_l', c_off)} "
                  "[+bias] vs forward score")

    labels = _proto_labels(naming_user, len(s_pers))
    lines = [BreakdownLine(labels[l], float(s_pers[l])) for l in range(len(s_pers))]
    l_star = int(np.argmax(np.abs(s_pers)))

    # word-level reading (md companion zoom + the feature-explained footer); shares sum to
    # the RAW cosine, so the affine scale a applies (u*_l − c = a·Σ_r c_{r,l})
    word_rows, word_codes, has_id = user_history_rows(hist_embed, user_id)      # (R, d)
    shares_w = per_feature_shares(word_rows, proto_fe.prototypes)               # (R, K_u)
    code_labels = feature_code_labels(feature_fields, items_info,
                                      dataset_dir=dataset_dir,
                                      feature_layout=feature_layout)
    alt_zoom_vals = (a_sc * t[l_star] * shares_w[:, l_star]).detach().numpy()
    alt_zoom_lines = [BreakdownLine(code_labels[c], float(alt_zoom_vals[j]))
                      for j, c in enumerate(word_codes)]
    if has_id:
        alt_zoom_lines.append(BreakdownLine("user-ID row (own profile)",
                                            float(alt_zoom_vals[-1]), is_id=True))
    _assert_close(float(alt_zoom_vals.sum()), float(s_pers[l_star]),
                  "Σ_r a·t_l*·c_{r,l*} (word rows) vs the top community's bar")

    # per-purchase reading (figure default zoom) — the same sum regrouped by purchase (4b C2′);
    # the item tensors come from the layout's own builder (F-DC05-13)
    purchases = user_train_items(dataset_dir, user_id)
    if purchases:
        if feature_layout == "bags":
            feature_ids, bag_w, _nf = build_feature_bags(dataset_dir, feature_fields)
        else:
            feature_ids, _nf = build_feature_ids(dataset_dir, feature_fields)
            bag_w = None
        purch_rows, _ = per_purchase_rows(hist_embed, user_id, purchases, feature_ids,
                                          feature_weights=bag_w)
        shares_p = per_feature_shares(purch_rows, proto_fe.prototypes)          # (P(+1), K_u)
        zoom_vals = (a_sc * t[l_star] * shares_p[:, l_star]).detach().numpy()
        zoom_lines = [BreakdownLine(item_description(items_info, int(i)),
                                    float(zoom_vals[j]))
                      for j, i in enumerate(purchases)]
        if has_id:
            zoom_lines.append(BreakdownLine("user-ID row (own profile)",
                                            float(zoom_vals[-1]), is_id=True))
        _assert_close(float(zoom_vals.sum()), float(s_pers[l_star]),
                      "Σ_i a·t_l*·c_{i,l*} (per-purchase rows) vs the top community's bar")
        zoom_kind = "shares"
        zoom_title = (f"inside {_trunc(labels[l_star], 22)}: exact per-purchase shares "
                      f"{'' if a_sc == 1.0 else f'{a_sc:g}·'}t_l·c(i,l)")
        statement = ""
    else:
        # verified degenerate path (4b C3): empty train basket → q_u = e_ID(u) (or 0 without
        # the ID row, where u* ≡ c and the personalized part vanishes — the honest
        # non-personalized fallback: S = B(t) = c·1ᵀt, which is 0 under standard cosine)
        zoom_kind, zoom_title, zoom_lines = "statement", "", []
        statement = ("Empty train-window basket: the composition has no purchase to attribute "
                     "to — the representation is the user-ID row alone (or, without it, the "
                     "personalized score vanishes and only the non-personalized baseline mass "
                     "B(t) — zero under standard cosine — remains).")

    # feature-explained fraction of the PERSONALIZED score (dc05 re-scope: B(t) sits outside
    # the grounded narrative by construction; fidelity claims are about S − B(t))
    with torch.no_grad():
        per_row_total = a_sc * (shares_w * t.unsqueeze(0)).sum(dim=1)  # Σ_l a·t_l·c_{r,l}
    pers_total = float(per_row_total.sum())
    id_part = float(per_row_total[-1]) if has_id else 0.0
    feat_part = pers_total - id_part
    if abs(pers_total) > 1e-12:
        pct = 100.0 * feat_part / pers_total
        pct = abs(pct) if pct == 0 else pct   # normalize -0.0
        fe_line = (f"feature-explained {feat_part:+.4f} vs user-ID {id_part:+.4f} "
                   f"of the personalized score {pers_total:+.4f} "
                   f"(features: {pct:.1f}%)")
    else:
        fe_line = "personalized score ≈ 0 — fraction undefined"

    bd = Breakdown(
        model_label=model_label, mechanism="proto",
        user_id=user_id, item_id=item_id,
        item_desc=item_description(items_info, item_id),
        total_score=total,
        naming_route="intrinsic — cos(e_f, p^u_l), read from the model's parameters",
        baseline=baseline, baseline_kind="item_intercept", cos_offset=c_off,
        lines=lines,
        lines_title=f"personalized part t_l·{_act_minus_c('u*_l', c_off)}",
        zoom_kind=zoom_kind,
        zoom_title=zoom_title,
        zoom_lines=zoom_lines,
        zoom_parent_value=float(s_pers[l_star]) if purchases else None,
        alt_zoom_title=(f"inside {_trunc(labels[l_star], 22)}: the same bar by attribute word "
                        f"{'' if a_sc == 1.0 else f'{a_sc:g}·'}t_l·c(r,l)"),
        alt_zoom_lines=alt_zoom_lines,
        statement=statement,
        feature_explained=fe_line,
        notes=[
            "Shares are EXACT additive summands of the computed score (not counterfactual "
            "effects; rows are jointly normalized through ‖q_u‖ — removing one purchase "
            "rescales the others).",
            "Purchase-level and word-level zooms are two exact regroupings of ONE sum — "
            "alternative readings, never added together.",
            "Negative values are legitimate anti-affinities and are rendered as such.",
        ])
    if baseline is not None:
        bd.notes.append(
            f"The item baseline B(t) = {_coef(c_off)}1ᵀt = {baseline:+.4f} is "
            "non-personalized (identical for every user) but NOT rank-inert — it is this "
            "item's own scalar and participates in item ranking; bars show the personalized "
            "remainder.")
    else:
        bd.notes.append(
            "Standard cosine (offset c = 0): the similarity carries no baseline mass — "
            "there is no item baseline B(t); the bars ARE the whole dot-product score.")
    return _attach_bias(bd, bias)


def compute_breakdown_user_proto(model, user_id: int, item_id: int,
                                 items_info: pd.DataFrame,
                                 dataset_dir: str,
                                 naming_user=None,
                                 top_k_users: int = 50,
                                 top_exemplars: int = 8,
                                 min_support: int = 3,
                                 model_label: str = "U-ProtoMF (user_proto)",
                                 ) -> Breakdown:
    """Host slot (fU stage bar, like-for-like): identical contribution math on t_l·(u*_l−c) +
    the same disclosed B(t) line (the host owes the same disclosure — GR7-fair; offset per the
    run's config, cosbias2x2); zoom = the top
    community's nearest USERS' consumed-item lift — declared post-hoc (the ratified steel-man
    reading of the host's §5.2-style profiling)."""
    proto_fe = model.user_feature_extractor          # PrototypeEmbedding (free ext)
    with torch.no_grad():
        ustar = proto_fe(torch.tensor([user_id])).squeeze(0)
        t = model.item_feature_extractor(torch.tensor([item_id])).squeeze(0)
    total = _forward_score(model, user_id, item_id)
    bias = _bias_lines(model, user_id, item_id)

    # cosine-offset generalization, 2026-08-17 special experiments (cosbias2x2):
    # identical split math to the fU slot, offset read from the run's config.
    a_sc, c_off, _ct = cosine_affine(proto_fe)
    baseline = c_off * float(t.sum()) if c_off != 0.0 else None
    s_pers = (t * (ustar - c_off)).numpy()
    _assert_close((baseline or 0.0) + float(s_pers.sum()) + sum(b.value for b in bias),
                  total, f"B(t) = {_coef(c_off)}1ᵀt + Σ_l t_l·{_act_minus_c('u*_l', c_off)} "
                  "[+bias] vs forward score")

    labels = _proto_labels(naming_user, len(s_pers))
    lines = [BreakdownLine(labels[l], float(s_pers[l])) for l in range(len(s_pers))]
    l_star = int(np.argmax(np.abs(s_pers)))

    # zoom (post-hoc, declared): the community's top-k nearest users' consumed-item lift
    with torch.no_grad():
        emb = proto_fe.embedding_ext.embedding_layer.weight                 # (n_users, d)
        p = proto_fe.prototypes[l_star:l_star + 1]                          # (1, d)
        sim = torch.nn.functional.cosine_similarity(emb, p.expand_as(emb), dim=1).numpy()
    nearest_users = set(np.argsort(-sim)[:top_k_users].tolist())

    train = pd.read_csv(os.path.join(dataset_dir, "listening_history_train.csv"),
                        usecols=["user_id", "item_id"])
    train = train.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    nb = train[train["user_id"].isin(nearest_users)]
    zoom_lines = []
    if len(nb) > 0:
        nb_counts = nb["item_id"].value_counts()
        glob_counts = train["item_id"].value_counts()
        lift = ((nb_counts / len(nb)) / (glob_counts.loc[nb_counts.index] / len(train)))
        lift = lift[nb_counts >= min_support].sort_values(ascending=False)
        zoom_lines = [BreakdownLine(item_description(items_info, int(i)), float(v))
                      for i, v in lift.head(top_exemplars).items()]

    bd = Breakdown(
        model_label=model_label, mechanism="proto",
        user_id=user_id, item_id=item_id,
        item_desc=item_description(items_info, item_id),
        total_score=total,
        naming_route="post-hoc — lift over the community's top-k aligned items",
        baseline=baseline, baseline_kind="item_intercept", cos_offset=c_off,
        lines=lines,
        lines_title=f"personalized part t_l·{_act_minus_c('u*_l', c_off)}",
        zoom_kind="exemplars",
        zoom_title=(f"inside {_trunc(labels[l_star], 22)}: what its {top_k_users} nearest "
                    f"users buy (post-hoc)"),
        zoom_lines=zoom_lines, zoom_parent_value=None,   # post-hoc list, not a sum
        zoom_value_label="lift",
        notes=[
            "The zoom panel is a POST-HOC profile (the community's nearest users' consumed "
            "items after training) — the model computes no purchase- or feature-level "
            "quantity for it; this asymmetry vs the history-grounded variant is the object "
            "of comparison, and is declared, not styled away.",
            "Negative values are legitimate anti-affinities and are rendered as such.",
        ])
    if baseline is not None:
        bd.notes.append(
            f"The item baseline B(t) = {_coef(c_off)}1ᵀt = {baseline:+.4f} is "
            "non-personalized (identical for every user) but NOT rank-inert — it is this "
            "item's own scalar and participates in item ranking; bars show the personalized "
            "remainder.")
    else:
        bd.notes.append(
            "Standard cosine (offset c = 0): the similarity carries no baseline mass — "
            "there is no item baseline B(t); the bars ARE the whole dot-product score.")
    return _attach_bias(bd, bias)


def compute_breakdown_lightfm(model, user_id: int, item_id: int,
                              items_info: pd.DataFrame,
                              feature_fields: List[str],
                              model_label: str = "LightFM-style CBF (lightfm)",
                              dataset_dir: str = None,
                              feature_layout: str = "fixed",
                              ) -> Breakdown:
    """CBF slot: S = u·q_i = Σ_r u·e_r — exact per-feature bars, no prototype layer.
    ``dataset_dir``/``feature_layout`` route code→labels through the builder-owned vocab
    (F-DC05-13); dataset_dir=None keeps the legacy fixed-layout items_info reconstruction."""
    feat_embed = model.item_feature_extractor        # FeatureEmbedding
    with torch.no_grad():
        u = model.user_feature_extractor(torch.tensor([user_id])).squeeze(0)   # (d,)
        rows = item_feature_rows(feat_embed, item_id)                          # (R, d)
        contribs = (rows @ u).numpy()
    total = _forward_score(model, user_id, item_id)
    bias = _bias_lines(model, user_id, item_id)
    _assert_close(float(contribs.sum()) + sum(b.value for b in bias),
                  total, "Σ_r u·e_r [+bias] vs forward score")

    code_labels = feature_code_labels(feature_fields, items_info,
                                      dataset_dir=dataset_dir,
                                      feature_layout=feature_layout)
    item_codes = item_meta_codes(feat_embed, item_id)   # bags: padding filtered (F-DC05-13)
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
    return _attach_bias(bd, bias)


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
    """Signed horizontal bars, top _TOP_LINES by |value|, remainder folded and stated.
    ``is_id`` lines are EXEMPT from the fold (F-DC05-14): the ID row is the ratified
    honesty device ("its own is_id line") and is always rendered, however small."""
    id_lines = [l for l in lines if l.is_id]
    rest_pool = [l for l in lines if not l.is_id]
    shown = sorted(rest_pool, key=lambda l: -abs(l.value))[:_TOP_LINES] + id_lines
    shown = list(reversed(sorted(shown, key=lambda l: -abs(l.value))))  # largest on top
    rest = [l for l in rest_pool if l not in shown]
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
    # cosine-offset generalization (cosbias2x2): baseline formulas carry the config's own
    # offset coefficient; disclosed bias terms are split out of the personalized/
    # item-discriminating remainder (they are not part of the mechanism's sum).
    bt = bd.bias_total if bd.bias_total is not None else 0.0
    bias_part = (f"  +  bias terms {bd.bias_total:+.4f} (see bars)"
                 if bd.bias_total is not None else "")
    if bd.mechanism == "popularity":
        score_line = f"train-split popularity count = {bd.total_score:.0f}"
    elif bd.baseline is not None and bd.baseline_kind == "item_intercept":
        # dc05 §3.4 mandatory wording: on the U host the c-shift mass B(t) = c·1ᵀt is a
        # per-ITEM scalar — non-personalized but NOT rank-inert. Never narrated as personalized.
        score_line = (f"score S = {bd.total_score:+.4f}  =  item baseline "
                      f"B(t) = {_coef(bd.cos_offset)}1ᵀt = {bd.baseline:+.4f} "
                      f"(non-personalized — this item's own "
                      f"scalar, identical for every user)  +  "
                      f"personalized {bd.total_score - bd.baseline - bt:+.4f}{bias_part}")
    elif bd.baseline is not None:
        score_line = (f"score S = {bd.total_score:+.4f}  =  rank-inert baseline "
                      f"{_coef(bd.cos_offset)}Σ_k u_k = {bd.baseline:+.4f} "
                      f"(same for every item)  +  "
                      f"item-discriminating {bd.total_score - bd.baseline - bt:+.4f}"
                      f"{bias_part}")
    elif bd.bias_total is not None:
        # no baseline mass (standard cosine, or a flat slot) but disclosed bias terms exist
        score_line = (f"score S = {bd.total_score:+.4f}  =  contribution bars "
                      f"{bd.total_score - bt:+.4f}{bias_part}")
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
            body = [f"{_trunc(l.label, 40)}  ({bd.zoom_value_label} = {l.value:.3f})"
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
    # same generalized split as the figure header (cosbias2x2): config-owned offset
    # coefficient in the baseline formula, bias terms split out of the remainder
    bt = bd.bias_total if bd.bias_total is not None else 0.0
    bias_part = (f" + bias terms **{bd.bias_total:+.4f}** (their own lines below)"
                 if bd.bias_total is not None else "")
    if bd.mechanism == "popularity":
        md.append(f"Train-split popularity count: **{bd.total_score:.0f}**")
    elif bd.baseline is not None and bd.baseline_kind == "item_intercept":
        md.append(f"Score **S = {bd.total_score:+.4f}** = item baseline "
                  f"B(t) = {_coef(bd.cos_offset)}1ᵀt = **{bd.baseline:+.4f}** "
                  f"(non-personalized — this item's own "
                  f"scalar, identical for every user; it participates in item ranking, unlike "
                  f"the item-prototype hosts' rank-inert baseline) + personalized "
                  f"**{bd.total_score - bd.baseline - bt:+.4f}**{bias_part}.")
    elif bd.baseline is not None:
        md.append(f"Score **S = {bd.total_score:+.4f}** = rank-inert baseline "
                  f"{_coef(bd.cos_offset)}Σ_k u_k = **{bd.baseline:+.4f}** (identical for "
                  f"every item this user ranks) + item-discriminating "
                  f"**{bd.total_score - bd.baseline - bt:+.4f}**{bias_part}.")
    elif bd.bias_total is not None:
        md.append(f"Score **S = {bd.total_score:+.4f}** = contribution bars "
                  f"**{bd.total_score - bt:+.4f}**{bias_part}.")
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
            md.append(f"- {l.label} ({bd.zoom_value_label} = {l.value:.3f})")
        md.append("")
    if bd.alt_zoom_lines:
        md += [f"## {bd.alt_zoom_title}", "",
               "*Alternative exact reading of the SAME bar — the two zooms regroup one sum "
               "and are never added together.*", "",
               "| row | share (score units) |", "|---|---|"]
        for l in sorted(bd.alt_zoom_lines, key=lambda l: -abs(l.value)):
            tag = " *(ID row)*" if l.is_id else ""
            md.append(f"| {l.label}{tag} | {l.value:+.4f} |")
        md.append(f"\nΣ = {sum(l.value for l in bd.alt_zoom_lines):+.4f} "
                  f"(= that community's bar, exactly)")
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
    from feature_extraction.feature_ids import build_code_to_field_value
    from utilities.consts import DATA_PATH
    from utilities.explanations.loader import load_recsys_from_results_dir
    from utilities.explanations.items_info import load_items_info
    from utilities.explanations.naming import (get_naming_config,
                                               intrinsic_naming_config,
                                               name_prototypes_from_weights,
                                               name_prototypes_intrinsic)
    from utilities.explanations.accessor import base_model_type, get_accessor

    model, config, metadata = load_recsys_from_results_dir(results_dir, data_dir=data_dir)
    # Arm keys (…_noid/_f0 ablations; …_bias/_cosstd/_cosstd_bias special-experiment arms,
    # cosbias2x2 2026-08-17) normalize to their base slot — the slots themselves read
    # cosine_type/use_bias from the loaded model, so every arm renders correctly; the
    # figure label keeps the FULL arm name so artifacts stay distinguishable.
    model_name = metadata["model"]
    model_type = base_model_type(model_name)
    dataset = metadata["dataset"]
    items_info = load_items_info(dataset, dataset_dir=data_dir)
    dataset_dir = data_dir if data_dir is not None else os.path.join(DATA_PATH, dataset)
    if item_id is None:
        item_id = top1_item_for_user(model, user_id, model.n_items)
    out_dir = out_dir or os.path.join(results_dir, "explanations", "breakdown")

    base_cfg = get_naming_config(dataset, items_info)
    if model_type.startswith("feature_item_proto"):
        item_spec = config["ft_ext_param"]["item_ft_ext_param"]
        feature_fields = item_spec["feature_fields"]
        feature_layout = item_spec.get("feature_layout", "fixed")
        code_to_fv = build_code_to_field_value(dataset_dir, feature_fields, feature_layout)
        accessor = get_accessor("feature_item_proto", model)
        naming = None
        try:
            naming = name_prototypes_intrinsic(
                accessor.feature_value_embeddings(), accessor.item_prototypes(),
                feature_fields, items_info,
                intrinsic_naming_config(base_cfg, feature_fields), side="item",
                code_to_field_value=code_to_fv)
        except Exception as e:
            print(f"[breakdown] ⚠ intrinsic naming failed: {e!r}")
        bd = compute_breakdown_feature_item_proto(
            model, user_id, item_id, items_info, feature_fields, naming,
            model_label=f"fI-ProtoMF ({model_name})",
            dataset_dir=dataset_dir, feature_layout=feature_layout)
    elif model_type == "item_proto":
        accessor = get_accessor("item_proto", model)
        naming = None
        try:
            naming = name_prototypes_from_weights(
                accessor.item_to_item_proto_sim(), items_info, base_cfg, side="item")
        except Exception as e:
            print(f"[breakdown] ⚠ post-hoc naming failed: {e!r}")
        bd = compute_breakdown_item_proto(model, user_id, item_id, items_info, naming,
                                          model_label=f"I-ProtoMF ({model_name})")
    elif model_type.startswith("lightfm"):
        item_spec = config["ft_ext_param"]["item_ft_ext_param"]
        feature_fields = item_spec["feature_fields"]
        feature_layout = item_spec.get("feature_layout", "fixed")
        bd = compute_breakdown_lightfm(model, user_id, item_id, items_info,
                                       feature_fields,
                                       model_label=f"LightFM-style CBF ({model_name})",
                                       dataset_dir=dataset_dir, feature_layout=feature_layout)
    elif model_type == "mf":
        bd = compute_breakdown_mf(model, user_id, item_id, items_info)
    elif model_type.startswith("feature_user_proto"):
        user_spec = config["ft_ext_param"]["user_ft_ext_param"]
        feature_fields = user_spec["feature_fields"]
        feature_layout = user_spec.get("feature_layout", "fixed")
        code_to_fv = build_code_to_field_value(dataset_dir, feature_fields, feature_layout)
        accessor = get_accessor("feature_user_proto", model)
        naming = None
        try:
            naming = name_prototypes_intrinsic(
                accessor.feature_value_embeddings(), accessor.user_prototypes(),
                feature_fields, items_info,
                intrinsic_naming_config(base_cfg, feature_fields), side="user",
                code_to_field_value=code_to_fv)
        except Exception as e:
            print(f"[breakdown] ⚠ intrinsic naming failed: {e!r}")
        bd = compute_breakdown_feature_user_proto(
            model, user_id, item_id, items_info, feature_fields, dataset_dir, naming,
            model_label=f"fU-ProtoMF ({model_name})", feature_layout=feature_layout)
    elif model_type == "user_proto":
        accessor = get_accessor("user_proto", model)
        naming = None
        try:
            naming = name_prototypes_from_weights(
                accessor.items_in_user_proto_space(), items_info, base_cfg, side="user")
        except Exception as e:
            print(f"[breakdown] ⚠ post-hoc naming failed: {e!r}")
        bd = compute_breakdown_user_proto(model, user_id, item_id, items_info,
                                          dataset_dir, naming,
                                          model_label=f"U-ProtoMF ({model_name})")
    elif model_type == "user_item_proto":
        raise NotImplementedError(
            "breakdown slot for user_item_proto is deliberately deferred to the fUfI merge "
            "stage that compares it — dc01 SC.5 gate decision, 2026-07-12")
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
