---
date: 2026-06-16
time: "20:14"
phase: 2
tags: [thesis/hidden-effects, thesis/interpretability-tuning]
placed: 2026-08-20
---
# Ablation idea: popularity (item) bias × prototype interpretability

In ProtoMF's item-ranking / implicit-feedback regime, only the **item (popularity) bias** is live — global bias and user bias are no-ops for per-user ranking (a constant shift, same across all items for a user, does not change the order). Worth investigating as an **ablation specifically for its effect on prototype interpretability**: does an explicit popularity-bias term **purify** the prototypes (offload the popularity signal so prototypes encode genuine taste/content structure instead of one uninformative "popular items" prototype), or **dilute** them (siphon explanatory share away from prototypes → hollow "recommended because popular" explanations)? Empirical, could go either way. Note ProtoMF already partly handles popularity via popularity-based negative sampling, so the explicit term is partly redundant. Surfaced from an S7 annotation on Biased MF.

[[experiments]] [[explanations]] [[phase-4]]
