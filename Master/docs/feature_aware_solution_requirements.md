# Feature-Aware Prototype Architecture — Solution Requirements

> Status: living document. First structured iteration, 2026-06-08.
> Scope: the Phase 4 extension that makes ProtoMF prototypes **feature-aware** and
> **intrinsically self-explaining**, integrated into training in a *tied* way
> (analogous to the `prototypes_double_tie` variant of `user_item_proto`).

## 1. Motivation

ProtoMF represents users/items purely by similarity to learned **collaborative**
prototypes. Two limitations matter for this thesis:

1. **Sparse-interaction / feature-rich regime.** On datasets like H&M, many items
   have few or no interactions but rich metadata (garment group, colour, department,
   section, product type, plus images). CF-only prototypes cannot place such items
   meaningfully — there is no learned embedding for an item nobody clicked yet.
   See [[ideas]] §1 (cold-start) and §3 (user-side features).

2. **Explanations are currently post-hoc.** The present explanation path
   (`utilities/explanations/explainers/top_k_items.py` + the `naming/` layer)
   profiles each prototype *after training* by its top-k nearest items. This is a
   read-out of a CF space, not a model-intrinsic, feature-grounded reason. We want
   explanations that the model *computes as part of its forward pass*, not a
   separate interpretation system bolted on top.

## 2. Problem framing

> Design an extension of the (double-tied) user/item prototype architecture in which
> **item features are first-class citizens of the prototype space**, such that the
> model (a) degrades gracefully under interaction sparsity by leaning on features,
> and (b) produces explanations that are intrinsic, prototype-shaped, and
> feature-grounded.

## 3. Requirements

> **Nothing here is absolutely hard.** *(Reframed 2026-06-11.)* All requirements sit
> on a spectrum of importance: some are close to non-negotiable (~99% hard), others
> are weaker preferences — but as a baseline, none is a categorical veto. Whether a
> deviation is acceptable depends on how valuable the design or finding that demands
> it turns out to be. Read every "must" below as a *strong default*, not an absolute.

### Core requirements (strong defaults — high on the spectrum)

- **R1 — Tied integration, not a bolt-on.** Feature-awareness must be wired into the
  architecture and the training objective in a *tied* way, mirroring the
  `prototypes_double_tie` philosophy where the user-embedding and item-prototype
  branches share weights. Features influence the *same* prototype representation that
  drives the recommendation score; they are not a parallel reranker or a separate
  post-hoc model. (The two original bullets that seeded this doc: "integrated into
  the architecture (like double tie) tied training" and "tie features to prototypes,
  making the previous prototypes self-explaining.")

- **R2 — Intrinsic, not post-hoc, explanations.** The explanation for a
  recommendation must be derivable from quantities the model computes at inference
  time (prototype activations and their feature grounding), not reconstructed
  afterwards by a separate procedure like top-k nearest items. The current
  `top_k_items` explainer is explicitly the baseline we are moving *beyond* (it stays
  as a comparison point).

- **R3 — Stays within the prototype paradigm.** Explanations must remain
  prototype-shaped: a recommendation is explained by *which prototype(s)* it / the
  user activates and *what those prototypes mean*. The novelty is that prototype
  meaning is now tied to **features/attributes** (e.g. "dark trousers, menswear")
  rather than only to a cloud of representative items.

- **R4 — Feature-grounded prototypes.** There must be an explicit, learned association
  between prototypes and item features, such that a prototype can be read as a
  human-understandable attribute profile, and an item's prototype activation can be
  attributed back to its features.

- **R5 — Sparsity robustness.** The design must let an item (or user) be represented
  from features alone when interaction signal is weak or absent — i.e. a meaningful
  prototype activation for an unseen / cold item. This is the concrete mechanism
  behind the cold-start contribution in [[ideas]] §1.

- **R6 — No accuracy regression on the dense regime (target).** The feature-aware
  variant should match or beat CF-only ProtoMF on the standard (dense) benchmarks,
  not just on the cold subset. Beating the paper baseline is the primary thesis bar
  per the thesis-focus memory; explanations are the second pillar.

- **R7 — User-perceivable explanation vocabulary (balanced).** *(Added 2026-06-11,
  out of the Phase 4.0 feature-analysis discussion.)* Guiding thought: for intrinsic
  explanations to be maximally transparent, **the model should roughly see what the
  user sees** — explanations are ultimately *for the user*, so the features they are
  stated in should be intuitively understandable (colour, pattern, garment kind,
  price, the image itself), not retailer-internal taxonomy codes. **Caveat — the
  balancing act:** a recommender system must first serve *good recommendations* and
  be driven mainly by what helps there; transparency means the **user has insight**,
  not that the internals cater to the most obvious properties. The bar therefore is:
  when computing power is available, the features used — engineered or raw — should
  still be **the (or among the) best they can be** for recommendation quality, *and
  also* be expressed in (or attributable to) a vocabulary the user intuitively
  understands. Empirical context (`hm_feature_analysis.md`): the most user-visible
  features are the weaker accuracy signals (colour/pattern ~1.3× lift) while internal
  taxonomy is strong (department 5.2×) — R7 is what keeps that tension an explicit
  design dimension instead of letting raw signal strength decide alone. Also
  strengthens the case for S3 (images are literally what the user sees; compute is a
  non-issue at V1 scale — the Kaggle teams' reason to skip images does not apply to
  us since we do not compete).

- **R8 — Conceptual parsimony (ProtoMF granularity).** *(Added 2026-06-11.)* A
  candidate design must be **one clean, isolated, distinct idea** at roughly the same
  conceptual granularity as ProtoMF itself (`user_item_proto` is the reference for
  "how big one idea is"). No over-engineered ensembles, no stacking of multiple
  mechanisms, no competition-style component piles — this is research, not a Kaggle
  contest. If a design needs three tricks to work, it is three candidates (or none).

### Soft / experimental requirements (lower on the spectrum — nice to have, or to compare against)

- **S1 — A deliberately decoupled variant.** Although R1 mandates a tied design as the
  headline contribution, we *may* also build a more "on-top" / loosely-coupled
  feature-explanation variant as an ablation, to demonstrate *why* tying is better
  (or to quantify the trade-off). The current post-hoc explainer is the degenerate
  end of this spectrum.

- **S2 — User-side feature-aware prototypes.** Symmetric extension to user metadata
  (age, club membership, …). Deferred per [[ideas]] §3; architecture should not
  preclude it.

- **S3 — Image features.** Garment images as an additional feature channel feeding the
  same prototype space (model input *or* explanation visualisation). Open per
  [[ideas]] open questions; note the known partial-image-set caveat in CLAUDE.md.

- **S4 — LLM-assisted prototype naming.** *(Added 2026-06-11; directional, details
  TBD.)* Light experimentation with an LLM as a *renaming/verbalisation layer* on top
  of prototype profiles: given a prototype's feature grounding (or its post-hoc top-k
  profile), the LLM produces a sensible human-readable name. This applies regardless
  of whether the underlying explanation is intrinsic or post-hoc — even with an
  intrinsically explainable model, the *naming* of prototypes may benefit from LLM
  rewording. Slots into the existing `naming/` layer as the natural insertion point.
  The LLM names, never scores: it must not become part of the recommendation or
  explanation *mechanism* (that stays the non-goal below).

- **S5 — Training-pipeline compatibility (minor, but assessed).** *(Added 2026-06-11.)*
  The current pipeline is deliberately efficient: per training step only
  `(batch, 1 + n_neg)` sampled items are scored (no full-catalog pass), item/user
  branches are cheap indexed lookups, sparse CSR/COO negative sampling, and extractor
  regularisation losses ride the `get_and_reset_loss()` accumulator. A candidate
  design should state concretely whether it preserves this profile (categorical
  feature lookups: essentially free) or breaks it (e.g. per-batch content encoders →
  must specify a precompute/cache strategy). Explicitly a **small factor, not a
  driving one** — recommendation/explanation improvement is the goal; this is a
  feasibility sanity check, not an optimisation contest.

### Non-goals (for the core thesis)

- LLM-generated natural-language explanations as the *primary explanation mechanism*
  (may be mentioned as related/future work). Note this does **not** exclude S4 —
  LLM-assisted *naming* of prototypes is in scope as a light experiment.
- A full human user study for explanation quality (out of scope unless the supervisor
  requests it; see [[ideas]] §2).

## 4. Design tensions to resolve

- **Where do features enter?** Candidate insertion points (verify against code, see
  `Master/docs/modification_map.md`): a feature encoder feeding the
  `PrototypeEmbedding` branch in `feature_extraction/feature_extractors.py`; a new
  `ft_type` in the factory; or prototypes living *in feature space* rather than CF
  space.
- **Fidelity vs. accuracy.** The tighter features are tied to the score (good for
  intrinsic explanation fidelity), the more we constrain the model — quantify this.
- **Feature attribution granularity.** Per-attribute vs. per-attribute-group vs.
  learned concepts. Concept-bottleneck-style designs make the attribution explicit but
  may cost accuracy.
- **Perceptual alignment vs. predictive strength (R7).** The 4.0 analysis shows the
  most user-visible features are the weakest predictors and vice versa. One candidate
  resolution: accuracy-bearing features (taxonomy, price, CF) drive the score, while
  prototype *grounding / read-out* is constrained to user-perceivable vocabulary —
  quantify what that constraint costs (links to the fidelity-vs-accuracy tension
  above).
- **Identifiability of prototypes.** A prototype must map to a *stable, distinct*
  feature profile to be explainable — regularisation (cf. ProtoMF's
  `sim_proto`/`sim_batch` losses) likely needs a feature-aware analogue. The relevant
  mechanism is **disentanglement** (MacridVAE-style per-factor regularisation); but
  note the *validity threat*: disentanglement does not automatically yield
  interpretability (see the critical-review entry, reading-list Theme E). Known
  prototype failure modes (prototype quality/quantity, collapse) are catalogued in the
  part-prototype survey (reading-list S2) — treat them as a pitfalls checklist.

## 5. How the current explanation path falls short (baseline to beat)

- `top_k_items.py`: post-hoc, reads CF similarity after training (violates R2).
- `naming/`: assigns names by feature lift over top-k items — already *uses* features,
  but still post-hoc and detached from the score (informative prior art for R4, but
  not the intrinsic mechanism we want).
- `tsne.py`, `weight_viz.py`, `feature_small_multiples.py`: visualisation only.

These remain valuable as **comparison baselines** and as evidence of the post-hoc gap
the intrinsic design is meant to close.

## 6. Open questions (carried from [[ideas]])

- Primary thesis framing: accuracy-first vs. explainability-first? (Leaning
  explainability-first.)
- Explanation-quality evaluation method (coherence/purity, fidelity, qualitative).
- H&M preprocessing specifics (k-core, temporal window) that affect the
  sparse-regime evaluation.
- Which interpretable-AI principle framework to adopt for the thesis section
  "ProtoMF within interpretable AI" (one framework or a small justified set —
  S1 Rudin vs. S2 Zhang & Chen vs. trustworthy-AI catalogues; see ideas §4).
  Once fixed, design candidates get placed against it in protocol Step 5.

---
*Reading list to ground these designs: `Master/literature/feature_aware_reading_list.md`.*
