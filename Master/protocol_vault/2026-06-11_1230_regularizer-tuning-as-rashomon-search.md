---
date: 2026-06-11
time: "12:30"
phase: 4
tags: [thesis/interpretability-objective]
placed: 2026-08-20
---
# Regularizer tuning as Rashomon-set search — theoretical grounding for the 4.6 experiment

Insight building on today's regularizer/hyperopt-mismatch entry: selecting `sim_proto_weight`/`sim_batch_weight` by an **explainability metric subject to an accuracy tolerance ε** is exactly *searching the Rashomon set for its most interpretable member* (Breiman's Rashomon effect; Rudin et al. 2022 Grand Challenge #9 — the set of near-optimal models is typically large and diverse enough to contain interpretable ones). This upgrades the planned regularizer×explainability ablation (masterplan 4.6, ideas.md §5) from an ad-hoc tuning fix to a theoretically grounded method: ε defines the Rashomon set, the explainability metric is the secondary selection criterion within it. Same argument also justifies ProtoMF's premise itself (prototype constraint stays inside MF's Rashomon set — our replication numbers are a data point).

[[phase-4]] [[explanations]] [[evaluation]] [[experiments]]
