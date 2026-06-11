---
date: 2026-06-11
time: "16:42"
phase: 4
---
# ProtoMF in the x→c→y schema: solves c→y, open on c-semantics (inverse of Koh)

Positioning insight for the interpretable-AI thesis section. **Source of the notation:** x→c (input to concepts) and c→y (concepts to output) comes from Koh et al. 2020 (Concept Bottleneck Models), as adopted in Rudin et al. 2022's disentangled-neural-network challenge (GC#5/#6); that passage's stated gap is that c→y "often remains a black box" (patched post-hoc à la Chen et al. 2020, or made a possibly-too-weak linear layer). **ProtoMF solves exactly that c→y half:** with c = prototype-similarity vector, the score is additive over prototypes (transparent by architecture, no post-hoc attribution); the `use_weight_matrix` / `prototypes_double_tie` variants put a linear projection on the similarities = literally Koh's linear c→y. **ProtoMF's open half is the *semantics* of c** — prototype meaning is not intrinsic because inputs are IDs (nothing to ground a name in), so naming is post-hoc (our lift-based layer). This makes ProtoMF the **complementary inverse** of concept-bottleneck models: they supervise c to be named (interpretable c) but struggle with c→y; ProtoMF gets c→y for free but leaves c unsupervised. The feature-aware extension targets ProtoMF's open half (intrinsically nameable c via features) → moves toward "both halves interpretable." **Honesty caveat for writing:** the full model is a bilinear user×item dot product, so phrase c→y as *additive over prototypes* (each term a user-side × item-side product), not a single linear readout. Builds on [[posthoc-naming-vs-intrinsic-model-distinction]] and the sleep-paper disentanglement analysis.

[[phase-4]] [[explanations]] [[paper-understanding]] [[decisions]]
