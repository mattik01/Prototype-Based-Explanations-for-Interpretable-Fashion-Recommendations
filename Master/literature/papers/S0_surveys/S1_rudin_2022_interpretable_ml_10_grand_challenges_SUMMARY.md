# S1 — Rudin et al. 2022, "Interpretable ML: Fundamental Principles and 10 Grand Challenges"

Mini-summary, thesis-relevant only. Position survey (Statistics Surveys). PDF in this folder. Insights logged in `protocol_vault/` (all 2026-06-11).

## One line
Field-defining argument for **interpretable-by-design** over post-hoc explanation of black boxes, organized as 10 grand challenges + a taxonomy (Table 1) matching data types to interpretable model families.

## Why it matters to us
- Locates ProtoMF: prototype / case-based reasoning = **Challenge #4**. Table 1 "case-based reasoning" row is our family; its "any data type" clause covers interaction + attribute data (no other row fits ID/interaction data).
- Supplies the vocabulary for the "ProtoMF within interpretable AI" thesis section (S1 is one candidate principle framework — see [[2026-06-11_1111_protomf-within-interpretable-ai]]).

## Most citable for (★ = strongest)
- ★ **Disadvantages of black-box + post-hoc explanation in high-stakes settings** — arguments are named and well-justified; the canonical cite for "why interpretable-by-design, not post-hoc XAI." *(My read while annotating: this paper is highly re-citable specifically here. The dedicated companion is Rudin 2019 = C3.)*
- Accuracy–interpretability trade-off is largely a **myth** → **Rashomon set** (Ch.#9): a large near-optimal set likely contains an interpretable model.
- Interpretability as **constraints/penalties**: sparsity, monotonicity, decomposability, case-based reasoning, disentanglement, generative constraints, variable preferences.
- **Variable-importance** taxonomy (per-prediction / per-model / model-independent); interpretable models don't need the per-prediction type — reasoning shows it directly.
- **Dimension reduction** (Ch.#7): DR choice (t-SNE vs UMAP/PaCMAP) decides which structure is preserved → affects faithfulness of 2D explanations.
- **Concept-bottleneck / disentangled NN** x→c→y framing (Ch.#5/#6; notation from Koh et al. 2020): the c→y mapping usually stays a black box.

## The 10 challenges (relevance flag)
1 sparse logical models · 2 scoring systems · 3 GAMs · **4 case-based reasoning ← US** · 5 supervised disentanglement · 6 unsupervised disentanglement · **7 dimension reduction (our prototype plots)** · 8 physics/causal constraints · **9 Rashomon set (our trade-off arg)** · 10 interpretable RL.

## Our insights from reading it (protocol links)
- ProtoMF = case-based reasoning (Ch.#4), **not a GAM** (additive only in prototype space) — [[2026-06-11_1504_protomf-not-a-gam-additive-only-in-proto-space]]
- Prototype regularizers are **interpretability penalties** mis-tuned on accuracy; tuning them on an explainability metric = **Rashomon-set search** — [[2026-06-11_1144_proto-regularizers-interpretability-not-accuracy]], [[2026-06-11_1230_regularizer-tuning-as-rashomon-search]]
- "Post-hoc" applies to prototype **naming**, not the model — finer axis than XAI-vs-IML — [[2026-06-11_1220_posthoc-naming-vs-intrinsic-model-distinction]]
- ProtoMF **solves c→y**, open on c-semantics; complementary inverse of Koh concept bottlenecks — [[2026-06-11_1642_protomf-solves-c-to-y-open-on-c-semantics]]
- Naming problem = **tax on abstraction**; memory-based CF avoids it only by not abstracting — [[2026-06-11_1806_why-protomf-over-cf-the-positioning-argument]]
- **Rashomon & RecSys**: low accuracy ceiling vs. large Rashomon set are independent; cite Dacrema 2019 flat leaderboards; use as light lens only — [[2026-06-11_1722_rashomon-set-recsys-positioning]]
- Interpretability constraints double as **expert-knowledge infusion** channels (sleep paper C6) — [[2026-06-11_1555_interpretability-constraints-as-expert-knowledge-infusion]]
- **Part-based prototypes** (Kim et al. 2014) = refinement unlocked once features enter; minor — [[2026-06-11_1522_part-based-prototypes-unlocked-by-features]]
- DR-algorithm comparison for prototype plots (Ch.#7) → masterplan 4.5

## Caveats when citing
- Survey assumes **supervised classification**; we transfer the framing to top-N ranking — say so.
- **Don't overclaim Rashomon** (it can't be measured here) — conceptual lens, not a method or measured quantity.
