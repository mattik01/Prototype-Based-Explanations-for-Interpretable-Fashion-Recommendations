---
date: 2026-06-11
time: "08:43"
phase: 4
tags: [thesis/data]
placed: 2026-08-20
---
# Phase 4.0 feature analysis — what H&M's features are worth, and a protocol caveat

Ran the 4.0 feature study locally on V1 `hm_1_month` (13,651 items / 623K train interactions; probes: within-user homogeneity lift vs. chance, popularity η², NMI/functional-dependency redundancy — `Master/docs/hm_feature_analysis.md`, outputs in `Master/experiments/hm_features/feature_analysis/`). **Signal ranking:** `product_code` (variant group) dominates at 35.9× lift but is identity-like (~1.9 items/value) → explanation/eval machinery, not a generalizing model feature; **mid-level taxonomy** (`department` 5.2×, `section` 3.3×, `product_type` 2.8×) are the natural prototype-grounding candidates; **price decile 2.36×** (needs transactions, not in `item_features.csv`); **colour/pattern are weak (~1.3×)** — Kaggle's colour emphasis works via the variant channel, not global colour taste, so colour's value is explanation legibility, an accuracy-vs-legibility tension for the feature-aware design. The taxonomy is hierarchical/redundant (product_type→product_group FD 1.0; colour↔colour_master NMI 0.81) → feed ~4 quasi-orthogonal feature groups, not all 9 columns.

**Dataset-suitability insight (→ thesis Dataset chapter, suitability/limitations paragraph):** ~12% of raw window transactions are article re-buys (17.8% of user×product pairs repeat; 8.2% buy multiple variants), but the paper-faithful dedup + set-based leave-one-out protocol makes this — the single strongest signal in the Kaggle competition — structurally invisible to all our models. This is a deliberate protocol choice (comparability with ProtoMF), and it means our metrics measure *discovery* recommendation only; it also strengthens the variant/`product_code` story for explanations. Conversely, the 1-month window is an "alive catalog" by construction (98.9% of last-week transactions are on items active the prior week), so the Kaggle 30-day-whitelist finding empirically backs the V1 short-window design.

[[data]] [[evaluation]] [[decisions]] [[phase-4]]
