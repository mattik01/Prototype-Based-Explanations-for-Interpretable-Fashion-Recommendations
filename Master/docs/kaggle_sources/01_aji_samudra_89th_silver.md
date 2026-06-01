# Aji Samudra & Hervind Philipe — 89th place silver (Medium writeup)

**Source ID:** `01_aji_samudra_89th_silver`
**Type:** blog
**URL:** https://ajisamudra.medium.com/silver-medal-solution-on-kaggle-h-m-personalized-fashion-recommendations-a0878e1eae63
**Author / Team:** Aji Samudra, Hervind Philipe
**Date posted:** 2022 (post-competition)
**Audience signal:** Medium article, public
**Final score:** 89/2952 — Silver. MAP@12 not explicitly stated in article body.
**Thesis tags:** `cold-start` `feature-importance` `repeat-purchase` `temporal-split` `ensemble`

---

## Architecture summary

Funnel architecture: 106k products → ~30 candidates per user via 6 retrieval strategies → 7 diverse ranking models (LGBMRanker, LGBMClassifier, CatBoostClassifier) → ensemble. Stage performance: ~7% recall in retrieval; ensembling delivers diversity at ranking.

## Features used

- **Item features**: color, category, department, price; `num_bought_last_7d` (purchase frequency).
- **User features**: age, `num_trx_last_90d`, `mean_price_last_90d`.
- **Engineered (user × item)**: price difference (user_mean − product_price); co-purchase history by color, category, attributes.
- **Image / text features**: not used — computational constraints explicitly cited.

## Feature engineering specifics

Temporal aggregations computed across **multiple windows**: all-time, 8 weeks, 4 weeks, 1 week. The authors restrict candidate pool to articles purchased in **last 6 weeks** based on EDA showing "majority of transactions made by articles still bought last 1 week" — i.e., a strong recency signal in fashion.

## Candidate generation

Six strategies feeding the funnel:
1. Cold-start: 12 popular products from last week
2. Popular products segmented by customer group (last 1 week)
3. Historical purchases by the customer (repeat-purchase signal)
4. Co-purchase patterns (items bought together)
5. Price similarity (price-band matching)
6. TensorFlow Recommenders Retrieval (neural CF)

## Ranking / model

7 ranking models ensembled. Algorithms: LightGBM Ranker, LightGBM Classifier, CatBoost Classifier. Diversity engineered through varied negative-sample generation ratios and techniques (not architectural variety).

## Validation strategy

Single-fold holdout: **last 7 days of training data** held as validation. Authors acknowledge single fold is suboptimal but report it "still giving us good correlation with leaderboard score." No CV across multiple weeks.

## Stated pitfalls / lessons

- Cold-start needs a **separate retrieval strategy** — popular-last-week for users with no history.
- Without filtering, candidate matrix = 145 billion rows — must prune aggressively.
- Model diversity through negative-sampling variety beats stacking architectural variations.
- Compute-constrained: had to skip image/text features entirely (a known opportunity left on the table).

## Notable insights / quotes

> "It's important to have good user-item feature to get well performing model"

> The authors stated their goal was having "diverse models so that later we could ensemble the predictions"

> "Ranking ML not only for sorting products customer bought but also products customer not bought" — rationale for mixing positive/negative samples from both retrieval and random.

## Reported metrics

- MAP@12 public / private: **not explicitly reported in article**.
- Combined retrieval recall: ~7%.
- No HR@K / NDCG@K reported.

## Code / notebooks

Not linked in article. Authors don't appear to have open-sourced the full pipeline.

## Thesis relevance

- **Phase 4.0 (Feature analysis):** confirms colour, category, department, price as the standard item-feature set. User features used minimally (age + 2 derived).
- **Phase 4.3 (Architecture):** explicit "compute-constrained → no images" — common pattern. Our prototype-based approach should aim to be lighter than image-embedding solutions.
- **Phase 3.7 (Competition context):** the 6-strategy funnel + 7-model ensemble is *typical* of mid-silver tier; we should report this as the "industry-standard" approach to compare against.
- **Cold-start (ideas.md #1):** validates that competition-grade pipelines treat cold-start as a separate retrieval lane, not a property of the ranker. Our feature-aware prototypes can address it inside the ranker — that's a cleanerSTATE
￼
 story.
