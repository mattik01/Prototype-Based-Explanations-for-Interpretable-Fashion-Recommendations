---
date: 2026-06-16
time: "18:08"
phase: 2
tags: [thesis/background]
placed: 2026-08-20
---
# Design principle: prefer intrinsic interpretability over post-hoc

Grounded in reading `utilities/explanations_utils.py`. ProtoMF's per-recommendation explanation (`weight_visualization`) is **intrinsic** — it decomposes the model's actual prototype-similarity score into per-prototype contributions. But **prototype grounding** (what each prototype *means*) is currently **post-hoc**: `get_top_k_items` assigns meaning after training via nearest items, and **user** prototypes are doubly indirect (read through the items a maximally-close user would be recommended). The feature-aware extension's core interpretability value is to move prototype grounding from post-hoc → **intrinsic** — item prototypes defined in feature space carry inherent meaning, no after-the-fact item lookup. Design principle for candidate selection: **avoid post-hoc-only additions; prefer by-design integration** that makes prototype meaning intrinsic. Directly serves the Rudin (S1) interpretable-over-explainable thesis angle. Pairs with the attribute-source positioning entry.

[[decisions]] [[explanations]] [[paper-understanding]] [[phase-2]]
