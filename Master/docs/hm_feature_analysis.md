# H&M Feature Analysis (Phase 4.0)

> Status 2026-06-11: parts **B–E done** (per-feature stats, redundancy structure,
> discriminative-power probes, engineered-feature probes) on the V1 `hm_1_month` set
> (13,651 items / 623K train interactions; part E on 1.14M raw window transactions).
> Plan: `Master/temp/feature_study_4_0_plan.md`. Raw outputs:
> `Master/experiments/hm_features/feature_analysis/` (CSVs).
> Scripts: `Master/scripts/hm_feature_analysis.py`, `hm_feature_analysis_e.py` (local, minutes).
> Kaggle grounding: `hm_dataset_notes.md` §2 / §6.4.

## Headline findings

1. **`product_code` is by far the strongest signal — and we don't extract it yet.**
   Within-user homogeneity lift **35.9×** (two purchases by the same user share a
   product_code 35.9× more often than chance) and popularity η² 0.75. This is the
   Kaggle "other colors / variants / repeat-purchase" channel showing up natively in
   V1. It is *not* in `item_features.csv` (9 columns extracted; splitter follow-up
   recommended). Caveat for architecture design: 7,174 values over 13,651 items
   (~1.9 items/value) — it is an **identity-like grouping**, not a generalizing
   attribute; useful for explanation ("variant of X you bought") and candidate logic,
   not as a one-hot prototype feature.

2. **`department_name` (197 values) is the best *generalizing* feature** — lift 5.2,
   η² 0.18, effective cardinality ~79. Then `section_name` (3.3), `product_type_name`
   (2.8), `garment_group_name` (2.8). These mid-granularity taxonomy levels are the
   natural candidates for grounding feature-aware prototypes.

3. **Colour and pattern features are weak preference signals** (lift 1.13–1.34) —
   despite their prominence in Kaggle write-ups. Interpretation: the Kaggle "colour"
   value comes through the *variant channel* (same product, different colour =
   product_code group), **not** through a global "this user likes black" effect.
   Colour features remain valuable for *explanations* (human-legible) even though
   their standalone discriminative power is low — a relevant tension for R4.

4. **The taxonomy is hierarchical and redundant — pick one level per branch.**
   Near-functional dependencies (FD > 0.95):
   `product_type → product_group` (1.00), `department → garment_group` (0.999),
   `section → index_group` (0.995), and `product_code →` essentially everything.
   Strong redundancy: `colour_group ↔ perceived_colour_master` (NMI 0.81),
   `section ↔ department` (0.76). Feeding all 9–11 columns into a model would
   double-count; the structure suggests ~4 quasi-orthogonal groups:
   **product-kind** (product_type or department), **garment/section context**,
   **colour** (one of the three colour columns), **pattern**
   (graphical_appearance).

## Per-feature reference (V1 item set)

| Feature | #values | eff. #values | lift | η²(pop) | Note |
|---|---:|---:|---:|---:|---|
| product_code | 7,174 | 5,542 | **35.91** | 0.751 | identity-like; not extracted yet |
| department_name | 197 | 79.0 | **5.16** | 0.180 | best generalizing feature |
| section_name | 51 | 28.2 | 3.33 | 0.143 | ⊂ index_group |
| product_type_name | 102 | 29.9 | 2.83 | 0.076 | ⊂ product_group; 28% of values <5 items |
| garment_group_name | 21 | 14.7 | 2.79 | 0.053 | ← department (FD 0.999) |
| product_group_name | 12 | 6.0 | 1.44 | 0.038 | coarse |
| index_group_name | 5 | 3.2 | 1.36 | 0.065 | coarsest (Ladies/Men/Kids/…) |
| perceived_colour_master_name | 19 | 10.8 | 1.34 | 0.025 | ↔ colour_group NMI 0.81 |
| colour_group_name | 48 | 19.0 | 1.33 | 0.029 | |
| perceived_colour_value_name | 7 | 4.5 | 1.16 | 0.011 | light/dark; not extracted yet |
| graphical_appearance_name | 28 | 5.4 | 1.13 | 0.020 | pattern; top1 = "Solid" 58% |

(lift = within-user same-value match probability ÷ interaction-weighted chance;
η² = share of log-popularity variance across items explained by the feature.)

## Implications for 4.3 (architecture candidates)

- Prototype-feature grounding should lean on **department/section/product_type-level
  attributes** — they have both resolution and signal.
- **product_code belongs in the data** (extract it) but enters as *explanation/eval
  machinery* (variant-aware analysis, repeat detection), not as a model feature —
  or via an embedding if a design candidate wants it.
- A feature-aware model that *only* gets colour+pattern will likely show little
  quantitative gain — manage expectations for ablations; colour's role is legibility.
- No missingness anywhere (0% across all 11 columns on V1) — no imputation machinery
  needed for categorical item features.

## Part E — engineered-feature probes (raw window transactions, 1.14M rows)

5. **Price is a mid-strength signal — stronger than colour/pattern, weaker than
   taxonomy.** Mean-price decile (10 bins): homogeneity lift **2.36**, η² 0.044 —
   slots between `product_type` (2.83) and `product_group` (1.44) in the part-D
   table. Users have a price level; a price-band feature is a legitimate
   prototype-grounding candidate and is *not* derivable from `item_features.csv`
   (needs transactions). Per-item stats in `price_stats.csv`.

6. **Repeat purchases are ~12% of raw transactions — and our pipeline deletes them.**
   In the raw window: 11.2% of (user, article) pairs repeat (11.9% of all
   transactions are article re-buys); at the variant level, 17.8% of
   (user, product_code) pairs repeat and **8.2% buy multiple variants** of the same
   product. The splitter dedups (customer, article, date) and the leave-one-out
   protocol is set-based, so this entire channel — the single strongest Kaggle
   signal — is **invisible to our models by design**. Worth one honest paragraph in
   the thesis (protocol choice, paper-faithful), and it strengthens the variant
   story: `product_code` groups capture much of what repeat-heuristics exploit.

7. **The 1-month window is effectively an "alive items only" catalog** — ~83% of the
   13,651 V1 items are active in any given week, and 98.9% of last-week transactions
   are on items already active the prior week. The Kaggle 30-day-whitelist finding
   (`[08]`, 20% of full catalog → 90% coverage) is *built into* V1 by construction —
   empirical backing for the short-window design (masterplan H&M strategy).

## Open follow-ups

- Splitter: add `product_code` (+ `perceived_colour_value_name`? + per-item mean
  price?) to `item_features.csv` and regenerate V1 (cheap; touches
  `data/hm/hm_splitter.py`).
- Optional: quick web/forum sweep for feature-importance findings beyond the 10 mined
  Kaggle sources (low expected yield — §6.4 already quantifies the ladder).
- Optional later: time-decay weighting probes (power-law won in `[08]`) — only
  relevant if a design candidate uses temporal weighting.
