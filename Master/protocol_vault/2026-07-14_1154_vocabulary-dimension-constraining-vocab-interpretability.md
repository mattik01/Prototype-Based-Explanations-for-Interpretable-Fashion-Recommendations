---
date: 2026-07-14
time: "11:54"
phase: 4
tags: [thesis/knobs-metrics]
placed: 2026-08-20
---
# The vocabulary dimension: constraining a model's vocabulary as an under-discussed interpretability dimension (and free expert-knowledge infusion)

*(Verbatim-capture entry — Matteo's conceptual claim from the Part II outline session, worked through while placing the understandability-weighted knob in the taxonomy. Captured close to literally per the capture rule; this is the freshest conceptual move of the 2026-07-14 session and a candidate thesis-level claim for the knobs-and-metrics chapter.)*

## The claim (Matteo, near-literal)

The understandability-weighted mechanism is connected to **vocabulary** — "an aspect that was not often highlighted in research in our field."

1. **In a truly interpretable model, the vocabulary of the model's explanations is closely connected to its internal vocabulary** — possibly on a slightly different abstraction level ("see prototypes — but not really"). A post-hoc-explained black box can explain itself in whatever vocabulary the explainer chooses; an intrinsic model *speaks the language it thinks in*. That identity is what makes vocabulary a property OF the model rather than of the explanation layer.
2. **Constraining a model's vocabulary — through whichever means — might in fact be an interpretability metric/dimension that is rarely discussed as such.** The constraint can enter anywhere (feature selection, weighted penalties, architectural grounding); what matters is that the *set of terms the model can reason in* is itself a measurable, steerable interpretability quantity.
3. **Citation anchor:** interpretable sleep scoring — Niknazar & Mednick. Cite the **original academic publication** (the C6 PDF in the literature folder is Matteo's own poster made *for* that publication, not the citable object; pull full bib data at writing time).
4. **The added observation: constraining a model's vocabulary can act as a sort of expert-knowledge infusion — and can even INCREASE model performance.** The expert vocabulary is already proven for the problem, so the constraint guides learning instead of forcing the model to rediscover structure unconstrained (this is the C6 result, previously protocolled as [[2026-06-11_1555_interpretability-constraints-as-expert-knowledge-infusion]]). This feeds the accuracy-cost inoculation passage directly: vocabulary constraint is a case where the interpretability move can flip the accuracy trade to positive.

## Placement and connections

- Lives in the **knobs-and-metrics chapter's taxonomy** as its novelty flag; the **understandability-weighted feature-use row** ([[2026-07-12_1413_understandability-weighted-feature-use-and-sparsity-axes]]) is the vocabulary dimension's concrete instantiation — Rudin cited for the principle ("preferences among the choice of variables"), the `(1−s_f)`-weighted group-lasso mechanism claimed as ours.
- **Retroactive reading of Part I** (noted in session, sentence-worthy at writing time): grounding *is* vocabulary alignment — fI/fU align the model's internal vocabulary with the catalogue's human vocabulary; the naming gap was a vocabulary mismatch all along.

[[explanations]] [[decisions]] [[thesis-writing]] [[phase-4]]
