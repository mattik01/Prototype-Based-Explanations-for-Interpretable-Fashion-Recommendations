# Phase 4.0 — H&M Feature Study: Plan

> Draft 2026-06-11 (Claude, while user works through the reading list).
> Scope per user direction: **methodical, beyond the 9 extracted columns**; reuse what
> Kaggle competitors found matters (already mined in `hm_dataset_notes.md` §2/§6.4);
> add engineered-feature candidates; **runs locally** (cluster only if a step would take
> excessive hours — none should). Framing: this feeds the **feature-aware prototype
> architecture (4.3)** — it is *not* about maximizing competitive metrics.

## A. Feature inventory & extraction gap

`articles.csv` has 25 columns; `hm_splitter.py` extracts **9** name-columns into
`item_features.csv`. Currently **not** extracted:

| Missing column | Why it matters (Kaggle evidence) |
|---|---|
| `product_code` | Variant-agnostic group ("other colors" channel, repeat-purchase at product level) — `[03] [04] [09] [10]` |
| `perceived_colour_value_name` | Light/dark tone, used by `[09]` |
| `prod_name` | Mostly redundant with product_code grouping; low priority |
| `detail_desc` (free text) | Text features used by 4/10 sources; compressed (SVD/UMAP) only; optional later |
| `*_no` code columns | Redundant with name columns — skip |

From `transactions_train.csv` (not item-static, must be derived):
**price** (multi-stat per item — 4/10 sources), **temporal counts/recency** (the single
most consistent FE pattern, ≥4 sources), **repeat-purchase rate** (universal).
From `customers.csv`: `age` (universal), `postal_code` (2nd-place insight) — user-side,
relevant only if user features ever enter the architecture.

## B. Per-feature empirical analysis (on the hm_1_month item set, 13,651 items)

For each of the 9 + missing categorical columns, joined to V1 items:
cardinality, missingness, value-distribution entropy (effective #values), top-10 values
with shares, long-tail shape. → Establishes which features have usable resolution at
V1 scale (a 131-value colour scheme behaves differently on 13.7K items than 106K).

## C. Redundancy / hierarchy structure

Pairwise **normalized mutual information** between all categorical columns + explicit
hierarchy checks (is `product_type` a refinement of `product_group`? `department` ⊂
`section` ⊂ `index_group`?). → Avoids feeding the architecture near-duplicate features;
informs whether prototypes should see a few orthogonal features or one taxonomy level.

## D. Discriminative-power probes (cheap, local, on the V1 train split, 623K rows)

1. **Within-user feature homogeneity:** for each feature, how much more often do two
   items bought by the same user share a value vs. two random items (lift over base
   rate)? High lift ⇒ the feature carries preference signal worth grounding prototypes in.
2. **Popularity skew per value:** does the feature separate high/low-velocity items?
3. **Repeat-purchase concentration by `product_code`** (validates the variant channel).

## E. Engineered-feature candidates (ranked by Kaggle ROI, `hm_dataset_notes.md` §6.4)

1. `product_code` variant-group id (+ #variants per group) — interpretable, prototype-friendly.
2. Per-item price stats (mean/median/min/max within window) + price-band bucket.
3. Temporal: item age in window, recency-weighted count (power-law decay won in `[08]`).
4. Repeat-purchase rate per item / product_code (smoothed, `[09]` recipe).
5. (Later/optional) compressed `detail_desc` text embedding — only if a design candidate
   needs it; image features explicitly deprioritized (+0.0007 CV only, `[03]`).

## F. Deliverables

- `Master/experiments/hm_features/feature_analysis/` — CSVs + figures (B–D outputs)
- `Master/docs/hm_feature_analysis.md` — findings write-up, feeds 4.3
- Likely splitter follow-up: add `product_code` (+ price stats?) to `item_features.csv`
  — would touch `data/hm/hm_splitter.py` and require regenerating V1 (cheap, ~minutes).

## Compute envelope

All steps minutes-scale locally (623K-row splits, 105K-row articles). Only full-window
transaction scans (31M rows) need chunked reads — still local, ~minutes. No cluster needed.
