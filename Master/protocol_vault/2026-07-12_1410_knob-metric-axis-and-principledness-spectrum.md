---
date: 2026-07-12
time: "14:10"
phase: 4
tags: [thesis/interpretability-tuning]
placed: 2026-08-20
---
# "Knob vs metric" IS the Route-1/Route-2 axis; and the principledness spectrum for interpretability-in-the-loss

*(Methodology entry, worked through interactively. Two tightly linked ideas that make the two-route block operational; merged because they're used together. Parent: [[2026-07-12_1409_two-route-rashomon-methodology]]. Full worked list of properties/metrics/knobs: `Master/docs/interpretability_knobs_metrics_taxonomy.md`.)*

## Idea 1 — a knob and a metric are the same property on opposite sides, and which side it can live on decides its route

The reason "knobs" and "metrics" felt tangled (Matteo: "not yet clearly separated in my head") is that **most interpretability properties can appear as both**, and it's easy not to notice you've switched sides:

- A **knob** is an *input* to training — a lever you set (a diversity penalty weight, `n_prototypes`, a sparsity target). It lives on the *cause* side.
- A **metric** is an *output* you measure on a *trained* model (measured prototype spread, measured activation sparsity, purity, stability). It lives on the *effect* side.

"Diversity" is the classic trap: it is a knob (a penalty weight you tune) *and* a metric (the spread you measure afterward). The bridge between the two sides is the whole point of the thesis block: **a knob is usually the differentiable, training-time *surrogate* of a metric you actually care about at evaluation time.**

That bridge maps one-to-one onto the two routes:
- **Route 2 = turn a metric into a knob** — build a differentiable surrogate loss; the property moves to the cause side and is *trained in*.
- **Route 1 = leave the metric as a metric** — measure it (non-differentiable is fine), tune *other* knobs, select on the measured value.

**Operational consequence:** separating the property list into "has a plausible differentiable surrogate" vs "doesn't" is not bookkeeping — it *decides which route each property can even live in*. A property with no differentiable surrogate is **Route-1-only, full stop.** When we sorted the actual list this held up empirically: prototype-item purity, cosine-readout reliability (Steck), and seed-stability-of-interpretations have blank surrogate columns → they are structurally Route-1-only. That's the thesis claim ("no surrogate ⇒ measure-and-select, don't train") confirmed against our own material rather than merely asserted. The three-column artifact (Property → Metric → Surrogate/knob) is the organised form of this.

## Idea 2 — the principledness spectrum makes "just another regularizer" a real position, not a failure

Every property that *does* become a loss term (Route 2) sits somewhere on a spectrum of how principled its derivation is:

1. **Ad-hoc penalty** — a weighted term that "works." (This is what ProtoMF's `sim_*_weight` regularizers are today.)
2. **Penalty with a probabilistic reading** — the term equals a known prior/likelihood, so training becomes MAP estimation. (E.g. sparsity ↔ Laplace prior; prototype diversity ↔ a determinantal point process (DPP), which is literally a diversity prior with a clean probabilistic form; entropy penalties ↔ categorical/softmax distributions.)
3. **Fully derived** — the loss falls out of a generative model.

This spectrum is what makes the block-two decision ("a bare regularizer is an acceptable landing") intellectually honest instead of a cop-out. The thesis contribution can be *placing each interpretability property on this spectrum* — which ones we can lift to level 2/3, which stay at level 1 — so that **"just another regularizer" reads as "this property is documented at level 1," and "future work" becomes concrete: future work = lifting properties up the spectrum.**

Two facts from sorting the real list:
- **ProtoMF's existing entropy-based regularizers are already ~level 2**, not ad-hoc: the soft (entropy) variants of the proto/batch regularizers, inclusiveness, crispness, and activation-sparsity all have a categorical/softmax-entropy reading. So the block does *not* start from zero principledness.
- **The one clear level-1 outlier is the dc02 K3 separation penalty** (a plain cosine-margin hinge, `relu(cos−m)`). That makes it the obvious **first move**: lifting separation to a DPP gives it a genuine probabilistic prior and demonstrates the "climb the spectrum" contribution on a concrete knob that already exists in the code.

[[explanations]] [[decisions]] [[phase-4]]
