---
date: 2026-06-16
time: "18:37"
phase: 2
tags: [thesis/conclusion]
placed: 2026-08-20
---
# FM×prototype: richer explainable interaction over the prototype layer (experiment idea)

A scoring-head variant — Axis B (readout), NOT feature grounding. Feed prototype-similarities as the "features" into a 2nd-order interaction (free-weight bilinear / plain 2nd-order Generalized Factorization — not FM proper, since prototype-sims are few and dense, so FM's sparsity factorization is unneeded and free weights are more readable). Generalizes ProtoMF's essentially-diagonal prototype interaction to **full cross-prototype** interaction, and stays explainable **only** with a plain additive+pairwise readout (NO neural NFM/DeepFM). Bonus: adds interaction-level explanations (which prototype-pairs drive a recommendation). Logged as an **experiment idea / potential additive contribution**, explicitly **orthogonal to the feature-integration core** (does not ground prototypes). Risks: expressiveness can erode prototype grounding (R_proto/R_batch must hold or interaction capacity be capped); transparent ≠ simple (m×n terms need top-k pruning, Rudin); only worth it if accuracy gain justifies the interpretability cost. See design caution on the two axes ([[explanations]]).

[[experiments]] [[explanations]] [[decisions]] [[phase-4]]
