---
date: 2026-06-07
time: "23:09"
phase: 3
tags: [thesis/eval-framework]
placed: 2026-08-20
---
# Masterplan 3.5.2 reduced to lightweight contextual MAP@12 + baselines

Dropped the planned switchable dual-mode (competition vs paper) evaluation pipeline as over-engineered for a metric we can't fairly compete on. Replaced with a lightweight temporal-week MAP@12 for positioning only, plus cheap interpretable reference baselines (popularity, repeat-purchase) so the number is meaningful. HR@10 / NDCG@10 leave-one-out stays the primary evaluation and model-selection metric.

[[evaluation]] [[decisions]] [[phase-3]]
