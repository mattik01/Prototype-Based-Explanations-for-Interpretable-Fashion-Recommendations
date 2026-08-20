---
date: 2026-07-12
time: "14:13"
phase: 4
tags: [thesis/interpretability-tuning]
placed: 2026-08-20
---
# Understandability-weighted feature use — a candidate novel Route-2 knob; and sparsity is four distinct axes

*(Candidate-contribution entry, worked through interactively (Matteo raised both points directly). Records a concrete novel mechanism for the block-two Route-2 track, plus a refinement that "sparsity" is not one thing. Parent: [[2026-07-12_1409_two-route-rashomon-methodology]]; novelty boundary vs Rudin in [[2026-07-12_1411_interpretability-knobs-map-to-rudin-families]]; user-vocabulary desideratum in [[2026-06-11_0905_r7-user-perceivable-feature-vocabulary]]. Taxonomy artifact: `Master/docs/interpretability_knobs_metrics_taxonomy.md`.)*

## Understandability-weighted feature use (the candidate contribution)

The idea, in plain terms: not all input features are equally understandable to the end user. Colour and pattern are things a shopper can see and name; some retailer taxonomy codes are opaque. So attach to each feature `f` an **understandability score** `s_f ∈ [0,1]` (high for user-perceivable features, low for opaque ones — the R7 "user-perceivable vocabulary" ranking already orders these). We *also* have each feature's **predictive value** `v_f`, which we've measured on H&M (e.g. colour ≈ 1.3× lift, department ≈ 5.2× lift — the user-legible features are, awkwardly, the weaker accuracy signals). Then build the feature set / feature usage by trading `s_f` against `v_f` instead of chasing `v_f` alone.

The mechanism can live in either route, and that's the appealing part:
- **Route 1 version:** pick the feature set up front by the `(s_f, v_f)` trade-off, train, measure interpretability after. Trivial and robust.
- **Route 2 version (the interesting one):** don't hard-select — apply an **understandability-weighted group-lasso**: penalise each feature block's usage in proportion to `(1 − s_f)`. Understandable features are cheap to use; opaque features are expensive. Fully differentiable. The result is **sparsity that is biased toward user-legible features**, not blind sparsity. This is a clean way to *operationalise the R7 desideratum* ("the model should roughly see what the user sees") as an actual training objective, and it directly answers the R7 tension that user-visible features get squeezed out because they're the weaker signals — the weighting deliberately protects them.

**Novelty boundary (carry this exactly):** this maps to Rudin's "preferences among the choice of variables," but Rudin never formalises a per-feature understandability weight — she only gestures at "expert importance via regularization." So cite Rudin for the *principle*, claim the *mechanism* (the `(1−s_f)`-weighted group-lasso) as our own. Do not mis-attribute the formula to Rudin.

**One open sub-decision (parked):** where does `s_f` come from — a heuristic derived from the R7 vocabulary ranking, or a small human elicitation? Decide at build time.

## Sparsity is four distinct axes (not one)

Prompted by Matteo pointing out "sparsity in features used and prototypes used and such." The single "activation sparsity" row was hiding four genuinely different things, each with its own metric and its own knob:

1. **Activation sparsity** — each *entity* expresses few prototypes. Metric: activation entropy per entity. Knob: the `max` regularizer / low-entropy penalty. (This was the one already captured.)
2. **Prototype-feature sparsity** — each *prototype* is defined by few *features/fields* (a prototype = "red + dress", not a dense blend over all fields). Metric: number of active fields in the profile row `A[k,·]`. Knob: L1 / group-lasso on `A`. **New.**
3. **Model-size sparsity** — few prototypes *in total* (small K). Metric: number of effective (non-dead) prototypes. Knob: `n_prototypes`, or prune dead prototypes.
4. **Feature-set sparsity** — few *features used by the whole model* (feature selection). Metric: number of features retained. Knob: feature selection / group-lasso on feature blocks. This is the axis the understandability weighting acts on. **New.**

**Critical non-overlap to keep straight:** crispness (the dc02 K2 penalty) is **not** prototype-feature sparsity. Crispness = "one *value* per field" (sharp *within* a field). Feature-sparsity = "few fields active *at all*." A prototype can be perfectly crisp on every one of its 8 fields yet still use all 8 — crisp but dense. We likely want both, and they are separate penalties. Rudin corroborates the general point that sparsity is model-class-specific countables (see the Rudin-families note).

[[explanations]] [[decisions]] [[phase-4]]
