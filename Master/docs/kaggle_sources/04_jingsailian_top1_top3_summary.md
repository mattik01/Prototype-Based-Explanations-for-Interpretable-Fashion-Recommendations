# jingsailian.com — Chinese summary of 1st & 3rd place solutions

**Source ID:** `04_jingsailian_top1_top3_summary`
**Type:** writeup (third-party Chinese summary)
**URL:** https://www.jingsailian.com/news/110624.html
**Author / Team:** uncredited summary author (paraphrases 1st place senkin13 & 30CrMnSiA, and 3rd place writeups)
**Date posted:** 2022 (post-competition)
**Audience signal:** Chinese-language Kaggle community summary post
**Final score:** Top 1: public 0.0372 / private **0.0379** (cited via Fujikawa); Top 3: public 0.03510 (cited internal CV)
**Thesis tags:** `ensemble` `repeat-purchase` `cf-only` `temporal-split` `feature-importance`

---

## Architecture summary

**1st place:** Multi-strategy retrieval funnel → LightGBM ranking with carefully engineered recall-side features. **3rd place:** ItemCF-style retrieval with BPR matrix factorization similarity injected as a ranking feature.

## Features used (1st place)

- **Item / interaction features**:
  - Purchase counts in multiple windows (week / month / quarter, time-weighted)
  - Last-transaction gap (recency)
  - Similarity scores from each recall channel (used as ranker features)
- **User features**:
  - Aggregations: mean/max/min user purchase prices, mean/max/min buyer ages per item
- **Engineered (user × item)**:
  - Age differences (user age vs. item buyer-age statistics)
  - Category proportion ratios (share of user's purchases in given category)
- **Image / text features**: **Not emphasized** in this summary — focus is on transaction + count features + graph embeddings.

## Feature engineering specifics

Five distinct feature categories explicitly enumerated:
1. **Count features** — purchases by timeframe (week/month/quarter), time-weighted
2. **Temporal features** — recency of purchase, last-transaction gap
3. **Aggregation features** — mean/max/min over user purchase prices, item buyer ages
4. **Difference / ratio features** — age difference, category proportion ratios
5. **Similarity scores from recall methods** — each retrieval channel's score becomes a ranker feature

## Candidate generation (1st place)

Six retrieval channels:
1. **Repeat purchases** — user's prior purchase history
2. **ItemCF** — item-item collaborative filtering
3. **Same product different variants** — color/size variant matching (matches Fujikawa's "Other colors")
4. **Popularity topN** — by sales volume
5. **Graph embeddings** — DeepWalk, Node2Vec, LINE, ProNE (four graph-embedding algorithms!)
6. **Logistic regression coarse-ranking** from top-1,000 popular items

## Ranking / model

**LightGBM ranker** with the 5-category feature stack above.

## Validation strategy

Not detailed in this summary. The CV numbers cited (0.02855 / 0.03262) suggest standard temporal holdout consistent with other top solutions.

## Stated pitfalls / lessons

> "In addition, if I only increase the recall num and don't adding the recall features, the CV score is very very poor"

**Translation:** Adding more candidates without injecting the recall channel's similarity scores as ranking features hurts performance. Recall-feature injection lifts CV from **0.02855 → 0.03262** (+14% relative).

This is a critical insight: **recall-stage similarity scores must be propagated to the ranker as features**, not just used to define the candidate pool.

## Notable insights / quotes

> 1st place — recall-side features boost CV from 0.02855 to 0.03262 (paraphrased).

> 3rd place — BPR Matrix Factorization similarity feature delivered "obvious improvement" 0.03363 → 0.03510 (+4.4% relative).

## Reported metrics

- 1st place MAP@12 private: **0.0379** (per cross-source citation)
- 1st place ablation: recall-features-only CV = 0.02855 → with-recall-features CV = 0.03262 (+0.00407 absolute)
- 3rd place ablation: pre-BPR CV = 0.03363 → with-BPR-similarity = 0.03510 (+0.00147 absolute)

## Code / notebooks

Not linked in this summary article. Original Kaggle writeups (reCAPTCHA-protected) presumably linked notebooks.

## Thesis relevance

- **Phase 3.7 / 3.8 (CF context):** **Strongest single piece of evidence** that recall-channel signals as features are doing most of the work — not the raw candidate pool. Validates the "candidate strategy is key" lesson from Fujikawa with hard ablation numbers.
- **Phase 4.0 / 4.3 (feature ROI):** the 1st place 5-category feature taxonomy is a clean blueprint for what to engineer when building feature-aware prototypes. Notably **no explicit reliance on image features** — graph embeddings and structured transaction features dominate.
- **Phase 4.5 (explanations):** "Same product different variants" + "repeat purchases" recur in 1st place — these are *the* two interpretable candidate channels and map directly onto ProtoMF explanation prototypes ("we recommend X because it's a color variant of something you bought" or "you previously bought this").
- **Citable to thesis Ch. 6:** the recall-feature ablation (+0.004 absolute CV) is a strong number to anchor the argument that prototype-based explanations *should reflect the retrieval rationale*.
