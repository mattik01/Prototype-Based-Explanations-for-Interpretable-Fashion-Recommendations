---
date: 2026-07-12
time: "12:51"
phase: 4
tags: [thesis/conclusion]
placed: 2026-08-20
---
# Bias & fairness — explicitly parked out of scope (candidate third thesis block, or future work)

**Decision (user, 2026-07-12):** bias and fairness concerns — and the interpretability *opportunities* attached to that field — are **explicitly out of scope for now**, but this is a *park*, not a discard. It must not be forgotten.

**Why it's a natural fit (not a random add-on):** interpretable / prototype-based recommendation and fairness are tightly linked — a model whose scoring surface is transparent (named prototypes, additive attribution) is exactly the kind of model where bias can be *inspected and localized* rather than only measured at the output. Popularity bias in particular already surfaces in our own work (the bias-channel discussion, [[2026-07-11_1556_bias-channel-popularity-prototype-interpretability]]; the id-twins/popularity braid, [[2026-07-12_0006_id-twins-popularity-braid-cold-eval]]). So grounded prototypes plausibly offer a *handle* on fairness — an opportunity, not just a risk to disclose.

**Placement options (to decide later, with the supervisor):**
1. A **third, smaller thesis block** *after* part two (part one = the feature-grounding lineage; part two = Interpretability Quantified). A compact segment: what bias/fairness questions the grounded-prototype design opens, possibly with a small demonstration.
2. **Future Work** only — flagged as a direction the architecture enables, left for others / a later project.

Either way: keep it visible in the outline as an explicit, deliberate omission (reviewers should see it was *chosen*, not overlooked), and let ongoing literature reading collect fairness/interpretability pointers in parallel so the option stays cheap to pick up.

**Status:** parked. Not in scope for the current lineage or the quantified-interpretability section. Revisit when part two is in hand.

[[decisions]] [[fairness]] [[explanations]] [[phase-4]]
