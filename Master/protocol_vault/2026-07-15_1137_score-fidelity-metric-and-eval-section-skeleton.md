---
date: 2026-07-15
time: "11:37"
phase: 4
tags: [thesis/eval-framework, thesis/knobs-metrics]
placed: 2026-08-20
---
# Score Fidelity (named) + the offline-explainability-evaluation section skeleton — the S2 §4.3 mine

*(From the S2 read, §4.3 pp61–62, worked through interactively over two sessions. Three products: a named metric, a thesis-section skeleton, and the cite/can't-use/can-use inventory. Captured fully — this is the blueprint for the eval chapter's explainability part.)*

## The naming decision: **Score Fidelity** (Matteo's name, adopted)

Our graded metric — the fraction of a recommendation score's mass attributable to grounded (feature-worded) components, i.e. the feature-explained score fraction / S0.2 score-share made into a first-class eval metric — is presented as **Score Fidelity**, positioned as the graded generalization of Peake & Wang 2018's Model Fidelity (reproduce their formula verbatim, then ours): **Model Fidelity counts the items the explanation covers; Score Fidelity measures the score mass it covers.** Item-level → score-level, binary → graded.

- It answers "fidelity of *what* to *what*": fidelity of **the explanation to the score** — how much of the score the grounded explanation accounts for. (This was the test that killed the earlier candidate name "explanation fidelity", which collides with the mechanism-faithfulness sense — see [[2026-07-14_1504_fidelity-term-disambiguation-relocated-not-escaped]].)
- **Guard sentence required at first use:** an XAI reader may parse "score fidelity" as surrogate-agreement (how well a separate explanation model *reproduces* the black-box's scores — the LIME family). One defining sentence dispatches it: ours decomposes the actual score; nothing is approximated.
- Bonus taxonomy for the disambiguation paragraph — three fidelities, one paragraph, each dispatched: **mechanism fidelity** (attention-debate sense; ours = 1 by construction), **Model Fidelity** (P&W item counting; trivial for intrinsic models), **Score Fidelity** (ours; the quantity that stays informative).

## The section skeleton (positioning paragraph + inventory — Matteo's directive)

Offline explainability evaluation is a field most readers won't know; the thesis section opens with a primer and walks exists → can't use (justified) → can use:

1. **Why a primer:** the field is young and unstandardized — two license quotes: S2 §4.3 "more offline evaluation measures/protocols are yet to be proposed" + §6.2.1 "there still lacks a usable offline explainability measure." (Same two quotes license us to *propose* metrics.)
2. **What exists (per S2):** two families — *coverage-counting* (MEP/MER, Abdollahi & Nasraoui 2017; Model Fidelity, Peake & Wang 2018) and *ground-truth matching* (BLEU/ROUGE against the user's own review as ground-truth explanation; readability indices as a footnote).
3. **What we cannot use, and why:** ground-truth matching — no reviews, no ground-truth explanations at any scale (the generality trade — pointer to [[2026-07-13_1817_core-level-modality-light-positioning]]); **MER** — its denominator ("all explainable items for this user") collapses for intrinsic models where every item is explainable; readability indices — they measure the template, not the model (one contrast sentence available: sentence readability ≠ R7 vocabulary perceivability).
4. **What we can use / adapt:** **MEP@θ** — threshold our graded share (item is *feature-explained at level θ* iff feature-attributable share ≥ θ), sweep θ for a curve; anchors our metric in the established coverage-counting framework instead of inventing ex nihilo. **Score Fidelity** as above. Plus the property-based family (profile entropy, purity, sharpness — the knobs/metrics taxonomy), licensed by the two quotes in (1) as proposed metrics for the gap S2 concedes.

## Standing caveat

All S2 §4.3 machinery is bookkeeping over *whatever* the explainer emits — none of it verifies semantics. Score Fidelity says how much of the score speaks feature vocabulary, not whether the vocabulary is truthful about the item; semantic truth stays with the grounding work and the relocated-fidelity argument (1504).

[[decisions]] [[evaluation]] [[explanations]] [[paper-understanding]] [[phase-4]]
