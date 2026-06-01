# Altimetrik Poland — claimed top 1.6% (Medium blog)

**Source ID:** `06_altimetrik_poland_top1.6pct`
**Type:** blog (corporate tech blog)
**URL:** https://altimetrikpoland.medium.com/retrieval-ranking-recommendation-system-for-h-m-dataset-728fdb32caa6
**Author / Team:** Altimetrik Poland Tech Blog (no individual author cited)
**Date posted:** 2022
**Audience signal:** Corporate engineering blog post (Medium)
**Final score:** **Claimed "top 1.6% of leaderboard"** (would correspond to ~rank 50/2952). Exact MAP@12 not stated.
**Thesis tags:** `cold-start` `image-features` `cf-only` `temporal-split` `repeat-purchase`

---

## Architecture summary

Three-stage pipeline: **Retrieval → business-rule filtering → ranking**. Retrieval narrows 100k products to a few hundred candidates per customer; filtering applies business constraints (out-of-stock, etc.); ranking selects top 12 via LightGBM Ranker.

## Features used

- **Item features**: color, design, graphics (visual/style metadata).
- **User features**: age, postal code, membership status.
- **Engineered (user × item)**:
  - Median price per user
  - Days since user's last purchase
  - "Custom preference patterns" — percentage of user's transactions per product category
- **Image / text features**: **YES** — transformer embeddings of product images used as one of three similar-product retrieval signals.

## Feature engineering specifics

User × category interaction features (proportion of user's spend / activity per product category) explicitly highlighted. Demographic features (age + postal code) used for cold-start retrieval.

## Candidate generation

Four retrieval methods:
1. **Most popular products with demographic conditioning** — ML model predicts top 100 from a 600-item popular pool, conditioned on customer age + postal code (cold-start lane).
2. **Similar products** — three retrieval signals fused:
   - Image embeddings (transformer)
   - Product attribute (tabular metadata) similarity
   - Customer-article interaction matrix similarity
3. **Similar customers** — "other customers also bought" co-purchase patterns.
4. **Repurchasing** — customer's purchase history.

## Ranking / model

**LightGBM Ranker** (tree-based learning-to-rank). Stated rationale: handles mixed categorical + continuous features without preprocessing and scales to candidate-set inference.

## Validation strategy

MAP@12 against held-out actual purchases. Specific split details not stated in blog post.

## Stated pitfalls / lessons

- Two-stage architecture enables effective systems under tight compute budgets (vs. ranking 100k items end-to-end).
- Recency matters — restricting to recent training data plus periodic retraining is non-negotiable.
- Cold-start handled by demographic conditioning (age + postal code) — direct match with aji_samudra's segmentation logic.
- Session-based behaviour requires different modeling than historical pattern matching (open issue acknowledged).

## Notable insights / quotes

> "Hybrid retrieval-ranking enables effective systems with limited computing resources."

> "Custom preference patterns showing percentage of customer transactions per product category."

## Reported metrics

- MAP@12 public/private: **not stated explicitly**, only "top 1.6%" claim.
- No HR@K / NDCG@K reported.

## Code / notebooks

Not linked from blog post. Internal Altimetrik project — code may not be public.

## Thesis relevance

- **Phase 4.0 / 4.3:** validates the postal-code + age cold-start retrieval lane (2nd place insight from Fujikawa) and shows it works in practice in a non-medal solution.
- **Phase 4.3 (image features):** image transformer embeddings used as **just one of three** similar-product signals — not the dominant one — consistent with fujikawa's "+0.0007 CV from images" finding. **Image features are useful but rarely the headline gain.**
- **Phase 4.5 (explanations):** the explicit decomposition of "similar products" into image + tabular + interaction lanes is exactly the kind of multi-source signal that ProtoMF prototypes could *unify* — each prototype could carry interpretation from one signal type.
- **Caveat:** "top 1.6%" is unverified (anonymous corporate post), so weight evidence lower than named writeups. Useful as a *pattern* corroborator, not as a primary citation.
