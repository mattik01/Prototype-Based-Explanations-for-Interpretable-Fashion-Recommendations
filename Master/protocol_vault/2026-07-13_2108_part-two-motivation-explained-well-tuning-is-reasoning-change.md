---
date: 2026-07-13
time: "21:08"
phase: 4
tags: [thesis/intro, thesis/interpretability-objective]
placed: 2026-08-20
---
# Part-II motivation: transparent AND explained well — and tuning for interpretability = changing the reasoning itself

*(Verbatim-capture entry — thesis-framing discussion during the outline session, user requested protocol. This is the cleanest statement of Part II's "why" so far; destined for §1.4 and Ch 10's motivation section, where it now sits as bullets. Belongs beside [[2026-07-12_1409_two-route-rashomon-methodology]] and [[2026-06-11_1230_regularizer-tuning-as-rashomon-search]].)*

## The motivation (Matteo, near-literal)

"We don't want just transparent recommendations — we want recommendations that are transparent **and explained well**. And for an interpretable model, that means in a sense that **the reasoning needs to be good/simple**."

## The sharpening (worked through together)

For a *post-hoc* explained black box, "explained well" is about the **explanation's relationship to the model** — does the explanation approximate the model faithfully? For an *intrinsically interpretable* model, the explanation **is** the model's reasoning — there is nothing to approximate, no fidelity gap. So "explained well" **collapses into "reasons well"**: simply, sharply, in few terms a human can hold.

Image (drafting-layer, may not survive into prose): transparency is a **window**; if the room behind it is a mess, you have gained honesty but not understanding. Part I builds the window (the reasoning becomes readable); Part II measures — and then optimizes — the tidiness of the room.

Consequence: once reasoning quality is measurable, it can become training pressure — hence the working title "**Interpretability as a Model Objective**". This also completes the transparent-through-interpretable lane from §1.1: transparency is the end, but transparency alone under-delivers; good reasoning is what makes it useful.

## The addition (Matteo, near-literal — the deeper point)

"It is very clear that in an interpretable model, **'tuning for interpretability' is changing the model's reasoning process** — and probably **simplifying** it. Which also ties in with the Rashomon-set notion."

Unpacked: in a black box, interpretability tuning changes only the *explanation layer*; the model computes as before. In an intrinsic model there is no separate layer — an interpretability knob (crispness, sparsity, separation, understandability-weighting) **reaches into the computation itself** and reshapes *how the model decides*, typically toward simpler reasoning. That is both the power and the responsibility of Route 2: the knobs are not cosmetic.

The Rashomon tie-in: the set of near-equally-accurate models contains many different *reasoning processes* for the same task. Tuning for interpretability is **navigating within the Rashomon set** toward the members whose reasoning is simplest/most legible — accuracy tolerance ε is the budget, the knobs are the steering ([[2026-06-11_1230_regularizer-tuning-as-rashomon-search]]). The knob does not merely re-decorate one model; it selects a *different reasoner* of (nearly) equal competence.

[[phase-4]] [[explanations]] [[thesis-writing]]
