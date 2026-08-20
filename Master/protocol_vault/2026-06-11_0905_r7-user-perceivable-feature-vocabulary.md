---
date: 2026-06-11
time: "09:05"
phase: 4
tags: [thesis/data, thesis/interpretability-tuning]
placed: 2026-08-20
---
# R7 — "the model should roughly see what the user sees" (balanced)

Added **R7** to `feature_aware_solution_requirements.md` (+ a matching §4 design tension). **The thought:** intrinsic explanations are ultimately *for the user*, so they are most transparent when stated in features the user actually perceives (colour, pattern, garment kind, price, the image itself) rather than retailer-internal taxonomy — i.e. the model should roughly see what the user sees. **Why now:** Kaggle competitors skipped images mainly for compute and tiny MAP gain, but we don't compete — at V1 scale (13.7K items) perceptual/image features are cheap, and the 4.0 analysis showed the user-visible features are exactly the *weak* accuracy signals (colour/pattern ~1.3× lift vs. department 5.2×), so without an explicit requirement raw signal strength would silently squeeze them out. **The caveat (decision):** recommendations come first — transparency means the *user has insight*, not that the internals cater to the most obvious properties; it's a balancing act. The bar: with compute available, features (raw or engineered) should still be the/among the best they can be for recommendation quality *and also* be expressed in (or attributable to) user-intuitive vocabulary.

[[decisions]] [[explanations]] [[phase-4]]
