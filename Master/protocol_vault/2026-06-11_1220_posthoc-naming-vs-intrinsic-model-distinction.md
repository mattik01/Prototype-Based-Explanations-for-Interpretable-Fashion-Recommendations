---
date: 2026-06-11
time: "12:20"
phase: 4
---
# Post-hoc applies to prototype *naming*, not the model — finer axis than XAI vs. IML

Positioning nuance for the thesis: the classic post-hoc-XAI vs. interpretable-by-design debate (Rudin: post-hoc explanations of black boxes are unreliable in high-stakes settings) does **not** separate base ProtoMF from our extension — both are intrinsically interpretable (the reasoning *is* prototype similarities). What is post hoc in base ProtoMF is only the prototype **semantics/naming**: prototypes are learned from interactions alone, so names must be inferred afterwards (our derived lift-based naming layer), grounding them in a vocabulary that is only moderately concrete for users. The feature-aware extension moves along a *finer* axis: it makes the naming **intrinsic** by building prototypes from explicit features, shifting them from numbers to user-perceivable vocabulary. The thesis must explicitly differentiate this naming axis (plus the related question of how far the model's input features cover the *user's* vocabulary) from the coarser XAI-vs-IML separation — otherwise we'd wrongly imply base ProtoMF is a black box, or that only our extension is "interpretable".

[[phase-4]] [[explanations]] [[decisions]]
