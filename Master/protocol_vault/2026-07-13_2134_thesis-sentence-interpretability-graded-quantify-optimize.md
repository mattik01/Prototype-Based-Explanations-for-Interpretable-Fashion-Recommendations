---
date: 2026-07-13
time: "21:34"
phase: 4
---
# Thesis sentence (Matteo's voice): interpretability graded, not binary → quantify it, then optimize for it

*(Literal thesis-candidate sentence, user-authored during the outline session and refined together over several rounds. Directive: preserve Matteo's voice — the "One can" is deliberate (see notes). **Placement: either §1.4 of the Introduction (slightly adjusted) OR the opening of Part II / Ch 10 — it is the bridge sentence and works at either end of the bridge.**)*

## Final version (agreed 2026-07-13)

> If we take interpretability seriously as a quality with many benefits for a recommender system (see §1.1), and understand it not as a binary property but as something a model can have more or less of — collaborative filtering, ProtoMF, our extensions — then it is reasonable for us to attempt to quantify it. One can then even optimize for it — not just in model design, but in training and evaluation.

**Home-dependent adjustment (Matteo):** the em-dash list presumes the graded comparison as *established* — true in Ch 10, but in the Introduction it is still a novel claim. **Preferred variant (Matteo, final): intro form, WITHOUT naming the models** — the demonstration promise alone:

> …but as something a model can have more or less of — as we will demonstrate in Chapter~X — then…

(Active "as we will demonstrate" over passive, keeping "we" as actor until the deliberate "one" takeover. This also removes the "extensions vs explanations" word question from the sentence entirely — the enumeration is gone; the chain lives in §1.1.)

**Final refinement (Matteo): BOTH homes use a chapter reference, neither names the models.** Only the reference's direction differs: Introduction → forward, "as we will demonstrate in Chapter~X"; Ch 10 → self/backward, "as we demonstrate in this chapter" or "as demonstrated in Part~I".

## Original (Matteo, verbatim)

"If we take Interpretability as a quality that improves a RecSys in many ways [see 1.1] serious, and understand it not as a binary property but something that a Model can have more or less of (CF, vs. ProtoMF; vs our explanations), then it is reasonable fur us to attempt to quantifiy it and use it as a Metric we want to consider, if not optimize for even beyond Model design, even in Training."

In-session amendments (Matteo): "with many benefits" preferred over "improves in many ways"; inherent-value clause cut — "then it is reasonable to attempt" with the prior context already carries that the quantification is valuable in itself; last sentence rebuilt by Matteo as "One might then even optimize for it, not just in model design, but even in training/evaluation" → hedge fix might→can, slash spelled out.

## Notes from the refinement

- **"One can then even…" is deliberate voice, kept on purpose:** the shift from "we" (our project quantifies) to "one" (anyone can now optimize) universalizes the contribution — it claims a general capability, not a private trick. Classic formal register (math's "one can show"); Rudin-style voiced scientific prose as the model. Caution recorded: use "one" at load-bearing moments only.
- **The sentence quietly encodes the two-route methodology:** optimize in *training* = Route 2 (knobs); optimize in *evaluation* = Route 1 (measure and select); *model design* = the architectural route. The intro would preview Ch 10's architecture in one line.
- **Open word choice:** "our extensions" (models compared) vs Matteo's original "our explanations" (grading reaches into the explanation layer) — currently "extensions"; restore "explanations" if the broader reading is intended.

[[phase-4]] [[explanations]] [[thesis-writing]]
