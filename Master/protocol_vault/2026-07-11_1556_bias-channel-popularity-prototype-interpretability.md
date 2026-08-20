---
date: 2026-07-11
time: "15:56"
phase: 4
tags: [thesis/eval-framework, thesis/hidden-effects]
placed: 2026-08-20
---
# Bias channel: cancels except item-popularity; fleet stays bias-free; evidence-triggered interpretability ablation

Refined during S0.4 gate discussion. **Structural fact:** our eval ranks 100 candidates within one user's row, so user bias and global bias cancel exactly (also b_u in BPR pairwise training) — the entire `use_bias` question reduces to **item bias = an explicit popularity prior**. **Decision:** the fleet keeps `use_bias=0` for the scrutiny phase — (1) paper/replication and all frozen references are bias-free (comparability), (2) dc02 structurally *requires* bias-off (ID-keyed side channel would break its bottleneck guarantee), (3) dc01's bias-gap defect (`dc01_feature_composed_bias_gap.md`) stays inert with bias off, (4) popularity effects are already observable via the popularity reference row and popularity-sampled negatives readout (S0.1 D3). **Interpretability stake (why this may become necessary):** without a bias channel, popularity has nowhere to live *except* the prototype geometry — prototypes may partly encode "popular" instead of taste structure, degrading prototype interpretability (extends [[2026-06-16_2014_popularity-bias-prototype-interpretability-ablation]]). **Trigger:** evidence, not speculation — if SC.8 rendered-explanation spot-checks show popularity contamination (e.g. bestsellers dominating top-k profiles across prototypes), promote a bias-on ablation ("does an explicit popularity channel clean up the prototypes?") for the affected candidate; dc01 would then also need the feature-composed-bias fix. To be listed in the S0.6 charter as a named, evidence-triggered option.

[[explanations]] [[evaluation]] [[decisions]] [[experiments]] [[phase-4]]
