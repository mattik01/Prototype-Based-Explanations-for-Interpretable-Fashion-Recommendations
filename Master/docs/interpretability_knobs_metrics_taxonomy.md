# Interpretability: properties → metrics → knobs (harvest draft)

Draft taxonomy for the Rashomon-set methodology block (thesis part two).
Harvested from the protocol vault + `feature_extraction/feature_extractors.py` + the 5 paper summaries + the Rudin C3/S1 PDFs, on 2026-07-12.
**Status: living design doc for thesis block two (the Rashomon block). Protocolled 2026-07-12 — see the `2026-07-12_14xx` entries in `protocol_vault/`. Still open: merge Matteo's own knob/metric list, which has items not in the vault.**

## Reading the columns

- **Property** — the interpretability desideratum, plain language.
- **Metric** — what you measure on a *trained* model (eval-time; need NOT be differentiable). Route-1 material.
- **Surrogate / knob** — the differentiable training lever, if one exists (blank ⇒ **Route-1-only**). Route-2 material.
- **Level** — principledness of the surrogate: **1** ad-hoc penalty · **2** has a probabilistic/prior reading (entropy, categorical, DPP…) · **3** fully derived from a generative model · **—** structural (architecture, not a loss).
- **Route** — where the property can live: R2 (trainable), R1 (measure/select only), or both.

## Table

| # | Property (desideratum) | Metric (measured, eval-time) | Surrogate / knob (differentiable) | Level | Route | Source |
|---|---|---|---|---|---|---|
| 1 | **Coverage** — every entity is near some prototype (no unexplainable entity) | frac. entities within τ of nearest prototype | `reg_batch` (max / soft-entropy) · weight `sim_batch_weight` | 2 (soft) / 1 (max) | R2+R1 | code L361–367,433; 6-11_1144 |
| 2 | **No dead prototype** — each prototype sits in a dense region of real objects | min prototype→item distance; # unused prototypes | `reg_proto` (max / soft / **incl**) · `sim_proto_weight`; dc02 **K4 push** `L_push` | 2 | R2+R1 | code L369–377,398–407,660–673 |
| 3 | **Balanced usage / inclusiveness** — all prototypes used, none idle | entropy of prototype load dist. `q_k` | `reg_proto='incl'` (−entropy of `q_k`); ACF inclusiveness | 2 | R2+R1 | code L398–407,225–291 |
| 4 | **Activation sparsity / commitment** — each entity expresses few prototypes | mean activation entropy per entity | `max` regularizer; ACF exclusiveness entropy; (L1/Laplace candidate) | 2 | R2+R1 | code; 6-15_1238; 7-12_1208 |
| 5 | **Prototype diversity / separation** — no near-duplicate prototypes | pairwise cosine of prototype vectors | dc02 **K3 sep** hinge `relu(cos−m)` · `sep_weight`,`sep_margin` | **1** → DPP lifts to 2 | R2+R1 | code L651–658 |
| 6 | **Field crispness / nameability** — each prototype commits to one value per attr. field | mean Shannon field-entropy; "coherence/nameability" | dc02 **K2 crisp** `L_crisp = mean H(softmax(A[k,block_f]))` · `crisp_weight` | 2 | R2+R1 | code L642–649; 6-11_1144 |
| 7 | **Prototype-item purity / coherence** — top-k items of a prototype are semantically pure | genre-purity of top-k items (ml-1m 0.70→0.87 w/ reg; collapses x100) | *(none direct; reg_proto/push are indirect drivers)* | — | **R1** | 6-15_1238 |
| 8 | **Additivity / linearity of attribution** — score = Σ named contributions | interaction-term magnitude vs additive part | *architectural:* `use_weight_matrix` linear readout, keep last mile additive; K1 nonneg | — (structural) | R2 (design) | 7-12_1200; 6-11_1642,1504 |
| 9 | **Non-cancelling profiles** — attribute profile reads additively | count of negative profile entries | dc02 **K1 nonneg** softplus reparam `A=softplus(Ã)` | — (structural) | R2 (design) | code L601–606 |
| 10 | **Cosine-readout semantic reliability** (Steck) — cosine ⇒ genuine similarity | metric-swap ablation; rescaling perturbation; feature-overlap proxy | `cosine_type` variants (angle-only); no diff. surrogate for *reliability itself* | — | **R1** | 6-16_2102 |
| 11 | **Seed stability of interpretations** — same prototypes across seeds | nearest-items overlap / rank-corr across seeds | *(none)* | — | **R1** | 6-16_2102 |
| 12 | **User-perceivability of vocabulary (R7)** — model sees what the user sees | coverage of user vocabulary; lift of user-visible features | *candidate:* importance floor / reweight on user-visible features | 1 (if built) | R1 (+R2 cand.) | 6-11_0905,1220 |
| 13 | **Faithfulness / feature-explained fraction** — explanation covers the real computation (low ID residual) | **feature-explained fraction** / ID share (headline); stratify by popularity | *candidate:* penalize ID-branch norm / residual magnitude | 1–2 (if built) | R1 (+R2 cand.) | 7-11_2132 |
| 14 | **Popularity disentanglement** — prototypes encode taste, not popularity | corr(prototype, popularity); rendered-explanation spot-checks | *candidate:* explicit bias channel (arch); decorrelation penalty | 1 (if built) | R1 (+R2 cand.) | 6-16_2014; 7-11_1556 |
| 15 | **Rotation-invariant read-outs** — quote directions (cosines), never coordinates | n/a (design constraint on all metrics) | *architectural constraint* | — | (constraint) | 7-11_2201 |
| 16 | **Intrinsic (vs post-hoc) grounding** — prototype meaning is inherent | share of naming that is by-design vs lift-based post-hoc | *architectural:* feature-built prototypes | — | (positioning) | 6-11_1220; 6-16_1808 |

## The knob inventory (Route-2 levers that already exist in code)

Base ProtoMF: `sim_proto_weight`, `sim_batch_weight` × {`reg_proto_type`, `reg_batch_type` ∈ max/soft/incl} · `cosine_type` · `n_prototypes` · `use_weight_matrix`.
dc02 (`attr_item_proto`): **K1** nonneg · **K2** `crisp_weight` · **K3** `sep_weight`/`sep_margin` · **K4** `push_weight`/`push_n_items`.
Baseline comparator: ACF `delta_exc`/`delta_inc` (same entropy family — the "prior art" row).

## What the table says

1. **Route 2 is already ~6 knobs deep, and most are level 2.** Every entropy-based penalty (coverage-soft, inclusiveness, sparsity, crispness) already has a categorical/softmax-entropy reading — they are *not* ad-hoc. The one clear level-1 outlier is **K3 separation** (plain cosine hinge) → the obvious **DPP-lift** target for the "raise it to a probabilistic prior" story.
2. **The knob/metric split sorts the routes cleanly.** Rows 7, 10, 11 have blank surrogate columns → structurally **Route-1-only** (purity, cosine-reliability, seed-stability). Confirms the thesis claim that "no differentiable surrogate ⇒ measure-and-select, don't train."
3. **The loop is already half-drawn in your notes.** 2026-06-11_1230 proposes selecting the Route-2 weights on a Route-1 metric (coherence/purity) under accuracy tolerance ε — that *is* Route 1 operating on Route 2's knobs. Your two routes are already entangled; the thesis can present that as the unifying frame, not two silos.
4. **Three new Route-2 knob candidates fall out of existing metrics** (rows 12–14): user-visible-feature importance floor; ID-residual penalty (push explanation onto features); popularity-decorrelation penalty. None built yet — natural contribution slots.

## Net-new from paper summaries (S1, S5, C3, S7, B5)

**New properties / knob families not in the vault harvest:**

| # | Property | Metric | Surrogate / knob | Level | Route | Source |
|---|---|---|---|---|---|---|
| 17 | **Monotonicity** — score moves consistently with a chosen feature | check monotone response | monotonicity constraint (well-studied, differentiable) | 2 | R2 (new) | S1 Ch.list; C3 |
| 18 | **Variable preferences / feature-importance constraints** — prefer some features over others | measured feature importance | importance floor / preference penalty (= row 12, now with a cite) | 1–2 | R2 (new) | S1; S7 feature-injection "regularize" fork |
| 19 | **Human–model similarity alignment** — latent similarity matches human judgment | similarity-perception gap (human study, e.g. HIVE); prototype-item purity is a cheap proxy | *(none — needs humans)* | — | **R1** | S5 §3.1.2 |
| 20 | **2D-plot faithfulness** — prototype-space plot preserves real structure | DR-algorithm comparison (t-SNE vs UMAP/PaCMAP) | DR-algorithm choice (knob, non-diff.) | — | **R1** | S1 Ch.#7 |
| 21 | **Additive feature transparency** (LightFM shallow end) — justify by feature similarity/distance | feature-similarity attribution | sum-of-features composition (architectural; ancestor/foil, not target) | — (struct) | R2 (foil) | B5 §6.3 |
| 22 | **# prototypes = interpretability ↔ meaningfulness trade** | K vs coherence vs coverage | `n_prototypes` (already row-listed; S5 frames the trade) | — | both | S5 §3.1.1 |

**Rudin's canonical constraint families** (S1/C3) — a *citable spine* for the whole property list: sparsity · monotonicity · decomposability(=additivity) · case-based reasoning · disentanglement · generative constraints · variable preferences. The thesis property list can be organized under these with a citation, rather than invented ad hoc. We currently have knobs touching sparsity, decomposability, case-based; **monotonicity, disentanglement, variable-preferences are open Route-2 slots**.

**Variable-importance measurement taxonomy** (S1): per-prediction / per-model / model-independent. Interpretable models don't need the per-prediction type — the reasoning shows importance directly. Relevant to *how* rows 7/13/19 get measured.

**Evaluation-metric note** (S7 §3.5.2): AUC penalizes all rank positions equally → position-aware metrics (NDCG/MRR/MAP) better for top-N. Bears on the user's "merged validation metric for early stopping" idea — the accuracy half should be position-aware.

## Sparsity is four distinct axes (was collapsed into row 4)

| Axis | Meaning | Metric | Knob | Status |
|---|---|---|---|---|
| **Activation sparsity** | entity expresses few prototypes | activation entropy per entity | `max` reg / low-entropy | = row 4 |
| **Prototype-feature sparsity** | each prototype uses few *fields* | # active fields in profile `A[k,·]` | L1 / group-lasso on `A` | **new — NOT crispness** |
| **Model-size sparsity** | few prototypes total (small K) | # effective / non-dead prototypes | `n_prototypes`; prune dead | partial (n_prototypes) |
| **Feature-set sparsity** | few features used model-wide | # features retained | feature selection / group-lasso on feature blocks | **gap** |

Key non-overlap: **crispness (row 6) ≠ prototype-feature sparsity.** Crispness = "one value *per* field" (sharp within a field); feature-sparsity = "few fields active *at all*." A prototype can be crisp on all 8 fields yet use all 8 — crisp but dense. Likely want both; separate penalties.

## Understandability-weighted feature use (new mechanism — sharper than row 12)

Per-feature **understandability score** `s_f ∈ [0,1]` (R7 vocabulary already ranks these: colour/pattern high, retailer taxonomy code low) × per-feature **predictive value** `v_f` (measured: colour ~1.3× lift, department ~5.2×). Select/weight features by the `(s_f, v_f)` trade-off.

- **Route 1:** pick feature set by the trade-off up front, train, measure after (robust).
- **Route 2:** **understandability-weighted group-lasso** — penalize each feature block's usage ∝ `(1 − s_f)`. Understandable features cheap, opaque features expensive → sparsity *biased toward understandable features*, not blind. Differentiable. **Clean novel Route-2 knob; operationalizes R7.**
- Maps to Rudin's **"variable preferences"** constraint → citable.
- Sub-decision (park): source of `s_f` — heuristic from R7 ranking vs. small human elicitation.

## ⚠ Two tensions (updated after reading the Rudin PDFs)

1. **Rashomon measurability — the summary oversold "can't be measured."** The full R22 §9 actually offers a *measurement toolkit*: Rashomon **volume** & **ratio** (Semenova et al. 2019), **pattern Rashomon ratio**, **ε-flatness/sharpness**, a **closed form for linear regression** (via data-matrix singular values), sampling/brute-force otherwise, plus **MCR / variable-importance-cloud** as measurable projections. But: no universal simple measure; volume is parameterization-dependent; the closed form is *linear-regression only*; deep-MF volume needs sampling/brute force. So the honest picture in our ranking/deep-MF regime is:
   - **The tractable instrument is Rudin's practical proxy** (R22 p.45): *"run many different types of ML algorithms … if they generally perform similarly, it correlates with the existence of a large Rashomon set."* Directly executable in recsys, and ties to the Dacrema-2019 flat-leaderboards evidence already in the vault.
   - The vault's "can't be measured here" (`2026-06-11_1722`) is right about *exact volume/ratio in deep-MF parameter space*, wrong as a blanket claim. Route 1 option (b) — approximate the Rashomon set in the ranking regime — is **more viable than I said last turn**: the multi-model-agreement proxy is honest, cheap, and citable. The fork is now: (a) selection within a restricted family, vs (b) the agreement-proxy Rashomon estimate. (b) is recommended.

2. **"Recsys is low-stakes."** Rudin's whole argument is scoped to *high-stakes classification* (C3 caveat). The user's Route-1 motivation — recommendation as simultaneously low- and high-stakes (millions in sales) — is precisely the rebuttal to that caveat, and is doing real load-bearing work. Worth developing deliberately as *the* justification for importing the Rudin/Rashomon apparatus into recsys, not treating it as throwaway framing.

## Rudin constraint catalog (from C3 2019 + S1 2022 PDFs — citable spine)

**Master enumeration** (R22 §General Principles, p.4, direct quote) — hang the whole property list off this:
> "sparsity of the model, monotonicity w.r.t. a variable, decomposability into sub-models, an ability to perform case-based reasoning …, disentanglement …, generative constraints (e.g., laws of physics), preferences among the choice of variables, or any other type of constraint relevant to the domain."

**Optimization form** (R22 Eq.1) = the Route-2 scalarization, with a citation, and it names a *third axis*:
> `min_f (1/n)Σ Loss(f,z_i) + C·InterpretabilityPenalty(f)  s.t. InterpretabilityConstraint(f)` — explicit **soft penalty vs. hard constraint** split. Every row's knob is either a soft penalty term or a hard constraint; worth tagging.

**Sparsity is model-class-specific countables** (confirms the 4-axis split above): # features/variables · # nonzero coefficients · # terms/conditions · # leaves · # rules · # component functions · **# prototypes** · **local (per-prediction) vs global** sparsity. Caveat Rudin insists on: *"sparsity ≠ interpretability … humans are mentally opposed to too simplistic representations"* — do not equate the two.

**Monotonicity machinery** (R22 Eq.4): one-directional indicators + **nonnegative coefficients** → monotone component. Note: dc02's **K1 nonneg is already half of this**. Plus "falling constraints" (decreasing along a rule list) and nice/integer coefficients.

**Separation ≈ disentanglement** (R22 §6, p.34, direct): *"a separation cost is applied to the learned prototypes to encourage diversity … very similar to the idea of disentanglement."* → dc02 **K3 sep gets a citation** and moves into the disentanglement family; DPP-lift lands there too.

**Concept whitening / decorrelation** (R22 §5, Chen et al. 2020): decorrelate concepts + align to axes = *"PCA for neural networks."* → the citation home for the **popularity-decorrelation** candidate (row 14).

**Variable preferences — honesty note:** Rudin names "preferences among the choice of variables" and *"prefer variables domain experts identified as important"* and *"incorporate expert importance through regularization"* — but **never formalizes a differentiable per-feature understandability weight.** So the understandability-weighted group-lasso is **our novel formalization**; cite Rudin for the *principle* (variable preferences), not the mechanism. Good for novelty; don't mis-attribute.

**Domain-specific, open-ended:** *"any list of interpretability metrics would be similarly fated"* (R22 p.4) — the taxonomy is explicitly open by Rudin's own framing; the fully- vs partially-interpretable distinction (R22 p.4) is a useful grading axis.

## Open cleanups before protocolling

- Rows 12–14 mix "metric we have" with "knob we could build" — decide which become Route-2 experiments vs Route-1 diagnostics.
- Confirm the level tags (esp. which entropy penalties you're willing to call level-2 "probabilistic" vs level-1 "ad-hoc").
- Matteo's own list (mentioned but not yet handed over) needs merging in — likely adds properties not in the vault.
