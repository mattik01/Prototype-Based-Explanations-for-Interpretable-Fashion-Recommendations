---
date: 2026-06-11
time: "11:44"
phase: 4
tags: [thesis/background, thesis/interpretability-objective]
placed: 2026-08-20
---
# Prototype regularizers are interpretability penalties — hyperopt tunes them on accuracy

Insight (from placing ProtoMF in Rudin's constraint framing): the two prototype regularizers are *pure interpretability penalties* — R_proto anchors every prototype near real entities (no dead/undepictable prototypes), R_batch keeps every entity close to some prototype (no unexplainable entities); neither serves ranking accuracy directly. Yet their weights (`sim_proto_weight`, `sim_batch_weight`) are currently selected by hyperopt against `hit_ratio@10`, i.e. an accuracy metric decides how strongly interpretability is enforced — a conceptual mismatch. Future experiment idea (recorded in ideas.md §5, masterplan 4.6): tune/select these weights against an explainability metric (e.g. prototype coherence/nameability) instead of, or alongside, accuracy.

[[phase-4]] [[explanations]] [[evaluation]] [[experiments]]
