# Mike Chesnokov — 142nd place bronze (GitHub repo)

**Source ID:** `05_mike_chesnokov_142nd_bronze`
**Type:** writeup (GitHub repo)
**URL:** https://github.com/mike-chesnokov/kaggle_hm_recs_2022
**Author / Team:** mike-chesnokov (solo)
**Date posted:** 2022 (post-competition)
**Audience signal:** Public GitHub repo with notebooks
**Final score:** 142/2952 — Bronze. **Public LB 0.02530, private LB 0.02554.**
**Thesis tags:** `ensemble` `repeat-purchase` `cf-only` `temporal-split`

---

## Architecture summary

Equal-weight blend of 4 models: 1 baseline + 3 second-level models. Each 2nd-level model combines an implicit KNN retriever with a LightGBM ranker.

## Features used

- **Item features**: product metadata (not enumerated in README).
- **User features**: demographics + behavioral attributes (age, etc.).
- **Engineered (user × item)**:
  - Time-decayed transaction counts
  - Price weighting
  - Frequent item-pair co-purchase signals
- **Image / text features**: **NOT used** — explicit omission despite availability in competition data.

## Feature engineering specifics

- Time decay applied to transaction counts to capture recency.
- Price-weighted history (heavier weight to items in user's typical price band).
- Interaction counts computed across multiple training windows + recent periods.

## Candidate generation

Multiple retrieval strategies:
- **User-to-user KNN** with TF-IDF similarity weighted by item importance — 100 candidates/user.
- **Item-to-item KNN** with TF-IDF similarity — 50–70 candidates/user (depending on weighting scheme).
- **Cold-start fallback** — gender-based personalized recommendations + popular items with temporal decay.

## Ranking / model

- **Baseline** (alone reaches LB 0.02359): time-decayed personal history + frequent item pairs for incomplete baskets + gender-based cold-start.
- **3 × 2nd-level models** (each LB 0.02422–0.02432): each pairs an implicit KNN retriever with a LightGBM ranker.
- **Final ensemble**: equal-weight blend of 4 models → LB 0.02530.

## Validation strategy

Not fully detailed in README. Multiple training windows used for feature computation suggests temporal holdout.

## Stated pitfalls / lessons

- Ensemble diversity (4 conceptually different models) drove the final lift over any individual model (best single: 0.02432 → ensemble 0.02530, +0.001 absolute).
- Time decay weighting critical for capturing purchase recency — fashion has strong recency signal (matches aji_samudra & fujikawa finding).
- Explicit choice to **not use image/text features** — yet still cracked top 5%.

## Notable insights / quotes

No standout direct quotes — repo is light on prose.

## Reported metrics

- MAP@12 public: **0.02530** (ensemble); 0.02359 (baseline alone); 0.02422–0.02432 (2nd-level singles).
- MAP@12 private: **0.02554**.
- Ablation: baseline → ensemble = +0.00171 (+7.2% relative).
- No HR@K / NDCG@K reported.

## Code / notebooks

Repo organized in Jupyter notebooks (92.3%) + Python utilities (7.7%). Notebooks cover validation, feature gen, and per-model training.

## Thesis relevance

- **Phase 4.0:** confirms even competitive bronze-tier solutions (top 5%) skip image/text features — **structured transaction features alone can clear ~LB 0.025**.
- **Phase 3.7:** the +7.2% ensemble lift over best single is comparable to wp_zhang's +2.1% — ensemble gains are real but diminish vs. the candidate-strategy gains at higher ranks.
- **Phase 4.4 (baselines):** this is a clean lower-bound reference point — "no images, no graph embeddings, no NN, just CF + LightGBM ranker" → LB 0.0253. Anything our feature-aware prototype scores below this would be a problem.
- **Cold-start as a strategy:** gender-based fallback matches aji_samudra — recurring pattern that gender is a powerful demographic signal in fashion.
