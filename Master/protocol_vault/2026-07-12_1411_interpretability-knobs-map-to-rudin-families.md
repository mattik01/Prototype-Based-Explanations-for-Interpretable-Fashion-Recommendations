---
date: 2026-07-12
time: "14:11"
phase: 4
---
# Our interpretability knobs map onto Rudin's constraint families — the citable spine for the taxonomy

*(Literature-grounding entry, from a close read of the two Rudin PDFs: C3 = Rudin 2019 "Stop Explaining Black Boxes", S1 = Rudin et al. 2022 "Interpretable ML: 10 Grand Challenges". This gives the block-two property list a canonical, citable backbone instead of an invented ad-hoc list. Parent: [[2026-07-12_1409_two-route-rashomon-methodology]]; taxonomy artifact: `Master/docs/interpretability_knobs_metrics_taxonomy.md`. See also [[2026-06-11_1555_interpretability-constraints-as-expert-knowledge-infusion]].)*

## The spine: Rudin's master enumeration of interpretability constraints

R22 (§General Principles, p.4) gives a canonical list we can hang the whole taxonomy off, verbatim:
> "sparsity of the model, monotonicity with respect to a variable, decomposability into sub-models, an ability to perform case-based reasoning …, disentanglement …, generative constraints (e.g., laws of physics), preferences among the choice of variables, or any other type of constraint that is relevant to the domain."

Rudin explicitly frames this as **open-ended and domain-specific** ("any list of interpretability metrics would be similarly fated" — no complete list is possible), which is exactly the licence to add our own recsys-specific rows while still organising them under her families. This is a much stronger thesis framing than an invented list: we present Rudin's families, then show which slots block-one already fills, which block two adds, and which we deliberately leave open.

## Rudin's optimisation form = Route 2's formula, plus a third organising axis

R22 Eq.1 is literally the Route-2 scalarised objective, with a citation, and it names a distinction we should carry:
> `min_f (1/n)Σ Loss(f, z_i) + C · InterpretabilityPenalty(f)  subject to  InterpretabilityConstraint(f)`

The **soft penalty vs. hard constraint** split is a genuine third axis (alongside route, and alongside pipeline-stage). Every knob in our taxonomy is either a soft penalty term (added to the loss) or a hard constraint (enforced on the model form) — worth tagging each row accordingly.

## Where our existing knobs already sit in Rudin's families (with citations)

Three of our knobs turn out to *be* Rudin constraints, which hands us free citations and a conceptual home:

- **dc02 K3 "separation" ≈ disentanglement.** R22 §6 (p.34), Rudin's own words: *"a separation cost is applied to the learned prototypes to encourage diversity of the prototypical parts, which is very similar to the idea of disentanglement."* So the separation penalty (and its DPP-lift) belongs in the **disentanglement** family, not off on its own.
- **Popularity-decorrelation candidate ≈ concept whitening.** R22 §5 describes decorrelating latent concepts and aligning them to axes (Chen et al. 2020, "concept whitening," a "PCA for neural networks"). That's the citation home for the "decorrelate prototypes from popularity so they encode taste, not popularity" idea.
- **dc02 K1 non-negativity ≈ half of Rudin's monotonicity machinery.** R22 Eq.4 shows monotone GAM components are built from one-directional indicators **plus non-negative coefficients**. Our softplus non-negativity reparam is exactly the non-negative-coefficient half — so monotonicity is closer than it looked.

## Sparsity is model-class-specific countables (confirms the 4-axis split)

Rudin never treats sparsity as one thing; she counts different things per model class: number of features/variables, number of nonzero coefficients, number of terms/conditions, number of leaves, number of rules, number of component functions, **number of prototypes** — plus a **local (per-prediction) vs global** distinction. This validates our own 4-axis decomposition (see the understandability/sparsity note). Carry her caveat too: *"sparsity ≠ interpretability … humans are mentally opposed to too simplistic representations"* — do not equate the two.

## Honesty boundary on the understandability weight (important)

Rudin names "preferences among the choice of variables" and says expert-judged feature importance can be *"incorporate[d] through regularization,"* but **she never formalises a differentiable per-feature understandability weight.** So our understandability-weighted feature penalty is *our* mechanism — cite Rudin for the *principle* (variable preferences), claim the *mechanism* as a contribution, and do not mis-attribute the formula to her. Clean novelty line.

[[explanations]] [[paper-understanding]] [[phase-4]]
