---
date: 2026-07-12
time: "14:09"
phase: 4
tags: [thesis/interpretability-tuning]
placed: 2026-08-20
---
# Thesis block two (the Rashomon block) — the two-route methodology, and the decision to carry both

*(Design/decision entry, worked through interactively. This is the founding methodology note for the SECOND thesis block — the "Rashomon set" block that sits on top of the feature-aware ProtoMF work (block one, the fI-ProtoMF lineage). It fixes the shape of the block before any of it is built. Supporting artifact with the full property/metric/knob taxonomy: `Master/docs/interpretability_knobs_metrics_taxonomy.md`. Companion notes: [[2026-06-11_1722_rashomon-set-recsys-positioning]], [[2026-06-11_1230_regularizer-tuning-as-rashomon-search]], [[2026-07-12_1208_giving-up-predictive-power-sparsity-flips-the-trade]].)*

## The motivating framing (why a Rashomon block belongs in recsys at all)

Recommendation is **simultaneously low-stakes and high-stakes**, and that contradiction is the licence for the whole block. Low-stakes: there is no inherently correct answer — many different ranked lists are all "fine," and simple models already sit near the achievable ceiling (Dacrema-2019 flat leaderboards; our own ProtoMF≈MF replication). High-stakes: the *aggregate* decision moves millions in sales, so which of the many near-equal models you ship actually matters. The low-stakes half is exactly the empirical signature of a **large Rashomon set** (many diverse models tie near the top); the high-stakes half is why you should care *which* member you pick. This double-stakes framing is also the direct rebuttal to Rudin's scoping — her interpretability-by-design argument is written for *high-stakes classification* (recidivism, credit, healthcare), and the reflex objection is "recsys is low-stakes, the apparatus doesn't apply." The answer is: it's both at once, and the Rashomon apparatus is precisely the tool for a domain where the set is large *and* the choice has consequences. Treat this paragraph as load-bearing motivation, not throwaway framing.

## The two routes

The block presents two ways to get an interpretable-yet-accurate model, and — the key realisation — **they are not rival philosophies; they are the same two-axis problem, differing only in *when* you collapse the two axes** (accuracy and interpretability).

**Route 1 — explore and select.** Keep accuracy and interpretability as separate axes. Argue recsys has a large Rashomon set, define/borrow a principled notion of "which models count as in the set" (an accuracy threshold ε, or the agreement-proxy — see the measurability correction note), then *search within that set* for the most interpretable member. Interpretability enters as extra knobs in a hyperopt-style search, conditioned on staying above an accuracy floor. Crucially, in Route 1 the interpretability quantity you select on **never needs to be differentiable** — it's a measured number used for selection/early-stopping, and early stopping can watch two numbers (accuracy metric + interpretability metric) without either being a training loss.

**Route 2 — infuse and optimise.** Collapse the two axes up front into a single scalar training objective: take a tangible interpretability property, build a **differentiable** surrogate for it, and add it to the recommendation loss so interpretability is *trained in* rather than *selected for*. Ideally the added term has a probabilistic/MAP reading (a prior), not just an ad-hoc weight. Validation can still be a two-number, non-differentiable early-stopping check — the differentiability requirement lives only in the training loss, not in validation.

**Why they unify:** scalarising the two axes (Route 2) is a *special case* of exploring the front (Route 1) — pick weights and you land on one point of the Pareto/Rashomon front. So the natural thesis structure is Route 1 as the *framework* and Route 2 as the *differentiable instantiation inside it*, not two silos. (Matteo's own vault already half-drew this loop in [[2026-06-11_1230_regularizer-tuning-as-rashomon-search]]: selecting ProtoMF's regularizer weights on an interpretability metric under an accuracy tolerance *is* Route 1 operating on Route 2's knobs.)

## The decision

- **Carry BOTH routes in the thesis.** They reinforce each other and share the taxonomy.
- **Emphasise Route 2.** The differentiable-interpretability-objective is the higher-ceiling, more distinctive contribution.
- **A bare regularizer is an acceptable landing.** If Route 2 ends up "just another weighted regularizer term" without the full probabilistic derivation, that is *not* a failure — it is a documented position (see the principledness-spectrum note), and **full reconciliation of interpretability-as-a-principled-training-objective is framed as future work.** This keeps the ambition honest: aim high, but the fallback is still a legitimate result.

## Continuity with block one

Block one already does Route-2-lite without naming it: ProtoMF's `sim_proto_weight`/`sim_batch_weight` regularizers are interpretability-shaping loss terms bolted onto the recommendation loss, chosen ad hoc. So Route 2 is literally "formalise and generalise what the codebase already does by hand," and Route 1 treats those same weights as the interpretability knobs to search over. The block is a natural outgrowth of block one, not a fresh start.

[[decisions]] [[explanations]] [[evaluation]] [[phase-4]]
