---
date: 2026-06-07
time: "23:11"
phase: 3
---
# Rationale for H&M: sparse + feature-rich is the rigorous justification

Clarified *why* H&M rather than just the paper datasets (which also have latent features — ml-1m genres, amazon category/brand). The rigorous reason: H&M is **sparse** (sparsest by density of all variants: 0.033–0.062% at 5-core vs ml-1m 3.05% and amazon 0.131%), the regime where pure CF leaves value on the table, so feature-aware extensions can *demonstrably* help. **Correction (avoid oversimplifying):** it is *not* "dense paper datasets vs sparse H&M" — only **ml-1m is dense** (the control where features should add little); **amazon is also sparse and notably item-cold** (~9.1 int/item, colder items than H&M). The honest framing is a dense control (ml-1m) plus *two flavors* of sparsity — item-cold amazon vs user-thin, large-catalog, feature-rich H&M — which makes a richer "where do features help" ablation than a dense/sparse binary. Secondary reason for H&M: interpretable fashion domain as an explanation showcase. Optional future axis: run feature-prototypes on ml-1m/amazon latent features to sharpen that story.

[[data]] [[decisions]] [[phase-3]]
