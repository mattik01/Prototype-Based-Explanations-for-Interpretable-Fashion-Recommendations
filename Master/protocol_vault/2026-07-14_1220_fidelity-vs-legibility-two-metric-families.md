---
date: 2026-07-14
time: "12:20"
phase: 4
tags: [thesis/knobs-metrics]
placed: 2026-08-20
---
# Fidelity vs legibility: the two families of interpretability metrics — an organizing axis for the taxonomy

Companion/extract of [[2026-07-14_1218_descriptor-gap-always-a-gap-fidelity-vs-legibility]], split out for direct placement in the knobs-and-metrics taxonomy section (Matteo: "I like the fidelity and legibility metrics — protocol that and save it for the taxonomy section").

Every interpretability metric in the taxonomy answers one of two different questions, and the two are orthogonal by construction:

- **Fidelity metrics — "is the explanation TRUE?"** They measure the descriptor gap: feature-explained fraction (coverage — ID mass is content with no descriptor at all), prototype-item purity (extension fit), determinacy / share identifiability (is the claimed per-feature contribution even well-posed?), cosine-readout reliability (Steck), seed-stability of interpretations (descriptor drift variance).
- **Legibility metrics — "is the explanation SIMPLE enough to hold?"** They measure explanation complexity: profile sharpness/entropy (activation sparsity), prototype separation/distinctness, model-size sparsity.

Orthogonality proof-by-example: a 100-term explanation with every term perfectly descriptored has zero gap and is unusable (faithful, illegible); a sharp 3-term explanation can be mislabeled throughout (legible, unfaithful). "Explained well" = small gap AND legible — the transparency-window/messy-room image predicted exactly this split (an honestly-windowed room can be messy; a tidy room can have a lying window).

**Consequences:** (1) every metric row in the taxonomy gets a fidelity/legibility tag (some rows both — e.g. separation is mainly legibility, gap-adjacent only via descriptor distinguishability); (2) the split is citable, not invented — it matches the faithfulness-vs-comprehensibility axis of the XAI evaluation literature (S4/G1 grounding pass); (3) tuning goals can now be stated precisely: Part I's grounding attacks the fidelity family, Part II's knobs mostly steer the legibility family — a knob that improves legibility while fidelity is unmeasured is exactly the blind-tuning mismatch that motivates the part.

**Legibility caveat (Matteo, added same day, near-literal):** legibility must NOT be misunderstood as *what the user sees* — there is a bunch of post-processing one could do to make a model's output sensible for application systems (grouping, truncation, verbalization, UI). **Legibility refers to the honest model output** — the raw explanation as the computation emits it. Post-processing itself is explicitly NOT a concern of this thesis (Matteo). One can argue that tuning legibility at the model (a) decreases the amount of post-processing necessary, or removes it; and (b) simplifies the model's *internal reasoning mechanism* itself — "careful with this one, but you get the point" — which could even increase performance or generalization. Point (b) is the "tuning is reasoning-change" claim ([[2026-07-13_2108_part-two-motivation-explained-well-tuning-is-reasoning-change]]) and connects to the accuracy-cost flip and the expert-infusion observation; carry it with its hedge.

[[explanations]] [[decisions]] [[thesis-writing]] [[phase-4]]
