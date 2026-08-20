---
date: 2026-07-14
time: "15:04"
phase: 4
tags: [thesis/eval-framework, thesis/background]
placed: 2026-08-20
---
# Fidelity: disambiguate the term at introduction — it exists in the attention debate too, but bites us at a different point

*(From the S2 read, §3.5 p48 (the attention-fidelity passage), worked through interactively. Directive for thesis writing: whenever the thesis introduces "fidelity", do the disambiguation below. Captured fully so the argument regenerates at writing time.)*

## The terminological hazard

"Explanation fidelity" is an established term with a specific home: the deep-explainable-models debate, canonically the attention fight — Jain & Wallace 2019 ("attention is not explanation") vs Wiegreffe & Pinter 2019 ("depends on your definition; not disproven") — with S2 §3.5 p48 as the survey-level anchor certifying fidelity as a recognized open problem. There, fidelity measures **the gap between an explanation read-out and the mechanism that actually produced the decision** (do the attention weights causally reflect why the model scored this item?). If the thesis uses "fidelity" without disambiguation, a reader from that literature will assume we mean that gap — and for us that gap is trivially closed, which would make the discussion look confused or evasive.

## Our position: fidelity is exact by construction at the contribution level

In ProtoMF (and the fI lineage) there is no mechanism/explanation gap to measure: the score *is* the sum of per-prototype contributions — the decomposition we show is the computation itself, not a post-hoc readout of it (`weight_visualization`; [[2026-06-16_1808_intrinsic-over-posthoc-principle]], [[2026-07-12_1200_linearity-interpretability-additive-attribution]]). Contribution-level fidelity = 1 by construction. The question the attention debate agonizes over doesn't get *answered* by our design — it doesn't *arise*.

## But we relocate the problem, we don't escape it

The honest continuation that makes the claim credible: what is contested for us is not "do the weights reflect the mechanism" but **"does the *name* attached to a prototype reflect what the prototype does"** — fidelity of the *semantics layer*, exactly the battleground the feature-grounding work fights on (base ProtoMF: post-hoc naming; dc01: naming made intrinsic — [[2026-06-11_1220_posthoc-naming-vs-intrinsic-model-distinction]]). One-liner for the thesis: *attention debates whether the weights are faithful; we get faithful weights for free and spend our effort making them mean something.*

## Framing decisions bundled in (same discussion)

- **Retired framing:** "we provide an attention-free approach, probably not worse, possibly not better" — rejected because (a) "not worse than attention" is an empirical claim with no attention variant of our model to compare against (the AFM/B8 road not taken), and (b) it frames us as lacking something we in fact don't need.
- **Adopted framing:** exact-by-construction attribution, with the fidelity problem honestly relocated to prototype semantics.
- **Attention itself: contested ≠ refuted.** W&P's pushback stands; the thesis should not caricature attention as a debunked vehicle — only as one whose faithfulness needs arguing, whereas ours doesn't at the contribution level.
- **Citation map:** the *debate* → Jain & Wallace 2019 + Wiegreffe & Pinter 2019 (E4a/E4b in the library) + S2 §3.5 p48 as survey anchor; the fidelity *metric* → Peake & Wang 2018 (Model Fidelity, S2 §4.3) and G1 (Barkan 2026 stochastic path integration).

[[decisions]] [[explanations]] [[paper-understanding]] [[phase-4]]
