# kuto5046 — 23rd place silver (SpeakerDeck post-mortem)

**Source ID:** `10_kuto5046_23rd_silver`
**Type:** video / slides (SpeakerDeck post-mortem, Japanese)
**URL:** https://speakerdeck.com/kuto5046/kaggle-h-and-mkonpezhen-rifan-ri
**Author / Team:** kuto5046
**Date posted:** 2022 (post-competition)
**Audience signal:** Public SpeakerDeck post-mortem
**Final score:** Public LB 14th / Private LB **23/2952 — Silver**.
**Thesis tags:** `image-features` `text-features` `ensemble` `temporal-split` `cf-only` `repeat-purchase`

---

## Architecture summary

Two-stage pipeline. **Multi-strategy candidate generation** narrows ~100k items to 100–300 candidates per user. **LightGBM Ranker with LambdaRank objective** scores candidates. Final result is a **6-model ensemble** with rank-position weighting to avoid over-emphasizing top predictions.

## Features used

- **User static features**: age, postal code.
- **User dynamic features**: purchase count, days since last purchase, category purchase rates, reorder rate, discount statistics.
- **Item static features**: product category, **BERT text embeddings** compressed via **UMAP to 6 dimensions**.
- **Item dynamic features**: sales growth, recent ranking by age/category, online purchase rate.
- **CF features**: similarity scores and ranks from **LightGCN** graph embeddings (via RecBole library).
- **Image / text features**: **YES — BERT text embeddings + UMAP reduction**. Image features not used.

## Feature engineering specifics

- **BERT → UMAP-6D** text compression for items — extreme dimensionality reduction (probably from 768 → 6) to feed into GBDT.
- **LightGCN** via RecBole library — graph-CF embeddings as similarity features for ranking.
- Time-series leak-prevention: candidate/feature generation uses only **pre-training-window historical data**.

## Candidate generation

Multiple strategies feeding the ranker (~100–300 candidates/user):
1. **Recent popular items** (globally + segmented by category/age)
2. **Recently purchased items** (repeat-purchase channel — explicitly called out as dominant)
3. **Related items via ItemCF** (same color variants + similar products)
4. **LightGCN embeddings** (graph-CF retrieval)

## Ranking / model

- **LightGBM Ranker** with **LambdaRank** loss.
- **Grouped by customer purchase counts per week** — non-standard grouping to optimize ranking quality across user activity levels.
- **6-model ensemble** at the end, with different candidates and features per model. Weighting by item rank position (not equal weight).

## Validation strategy

- Weekly time-series split, 3–5 week training windows.
- Strict leak-proofing: candidate generation and feature engineering use only data prior to the training window.

## Stated pitfalls / lessons

- Cold-start was a non-factor: **only 0.7% cold-start users and 0.9% cold-start items**. Rule-based age-based-popularity post-processing for new users gave minor lift.
- Implicit-feedback nature requires careful **negative sampling** for ranker training.
- Fashion's **repeat-purchase pattern dominates** — recently purchased items is *the* signal.

## Notable insights / quotes (from kuto5046's summary of higher-ranked solutions)

> **1st place:** Graph embeddings (ProNE), feature store implementation, CatBoost-GPU optimization. **"candidate generation strategy was the key"**

> **3rd place:** BPR matrix factorization for implicit feedback similarity features.

> **4th place:** Two-tower neural networks with **gating mechanisms for user activity-weighted learning** (NEW pattern not seen in other sources).

> **6th place:** **User embeddings derived from concatenated text of all previously purchased items (TF-IDF + SVD).** (Item-text content used to *build user representations*, not just item features — novel pattern.)

## Reported metrics

- Public LB: 14th place
- Private LB: 23rd / 2952 — Silver
- Specific MAP@12 numbers not in summary.
- 0.7% cold-start users, 0.9% cold-start items (dataset stats).

## Code / notebooks

Not directly linked in summary. SpeakerDeck slides only.

## Thesis relevance

- **Phase 4.0 / 4.3 (multi-modal):** **clean evidence that BERT text embeddings + UMAP-6D are competitive on this dataset** — directly relevant if our feature-aware prototype variants need text features.
- **Phase 4.3 (graph embeddings recur):** ProNE (1st), DeepWalk/Node2Vec/LINE/ProNE (1st, per jingsailian), LightGCN (this source), DeepWalk (fujikawa) — graph embeddings are ubiquitous in top solutions. Our feature-aware prototype design should consider graph-derived features as a benchmark to beat or absorb.
- **Phase 4.4 (novel patterns to consider):**
  - **6th place pattern:** *user embedding from text of historical items* — could ProtoMF prototypes carry text content naturally? Prototype as "what kind of items this user buys" semantically?
  - **4th place pattern:** *gating mechanism for user activity weighting* — relevant for handling H&M's heavy-tail customer purchase distribution.
- **Phase 4.5 (explanations):** BERT-UMAP item embeddings give a **continuous text-similarity axis** for prototype interpretation — "this prototype represents 'cotton sweaters' because its UMAP-6D centroid is near these representative items."
- **Cross-source pattern confirmation:** *"candidate generation strategy was the key"* — now confirmed by 1st-place quote (via kuto5046), Fujikawa's 12th-place observation, and FLAIRS paper's emphasis on retrieval. Strongest single recurring lesson across this entire extraction sweep.
