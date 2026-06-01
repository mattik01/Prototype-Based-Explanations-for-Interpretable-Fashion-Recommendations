# Kazuki Fujikawa — 12th place gold (SpeakerDeck post-mortem)

**Source ID:** `03_fujikawa_12th_gold`
**Type:** video / slides
**URL:** https://speakerdeck.com/kfujikawa/h-and-m-personalized-fashion-recommendation
**Author / Team:** Kazuki Fujikawa
**Date posted:** 2022 (post-competition)
**Audience signal:** Conference-style post-mortem presentation
**Final score:** 12/2952 — Gold. **Public LB 0.0339.** Private LB not stated.
**Thesis tags:** `feature-importance` `repeat-purchase` `image-features` `ensemble` `temporal-split` `cf-only`

---

## Architecture summary

Two-track pipeline: (1) candidate-group-aware GBDT and (2) embedding-based NN, fused via NN→GBDT stacking and final weighted rank averaging (weights optimized via Optuna). Three candidate groups feed the GBDT track; ~30k active items (past 5 weeks) feed the NN track.

## Features used

- **Item features (engineered embeddings)**:
  - EfficientNet image embeddings (+ PCA)
  - Word2Vec embeddings
  - DeepWalk embeddings (graph-based)
- **Customer/transaction features**:
  - Weeks since final purchase
  - Weeks purchased intervals
  - Total cumulative quantity purchased
  - Cumulative quantity per item
- **Engineered (user × item)**:
  - Cosine similarity (top-1 → top-2 to 5 similar items per past purchase) — extended similarity rankings
- **Color variants**: explicitly engineered as a candidate-generation channel.
- **Image / text features**: **YES** — EfficientNet image embeddings, with PCA dim-reduction, contributed +0.0007 CV (0.0345 → 0.0352).

## Feature engineering specifics

- **Metadata embeddings introduced** lifted CV from 0.0254 → 0.0298 (substantial — +17% relative).
- **Likelihood dropout** (anti-overfitting on per-item params): 0.0298 → 0.0306.
- **DiceLoss scaling (1e-2)** balancing BCE/Dice magnitudes: 0.0306 → 0.0327.
- **Top2-5 similar-item features** (extended cosine-rank features): 0.0327 → 0.0345.
- **Image+graph embeddings (EfficientNet+PCA+DeepWalk)**: 0.0345 → 0.0352.

## Candidate generation

Three primary GBDT-track groups:
1. **Re-order** — past purchases by same customer (median: 8 items)
2. **Other colors** — color variants of past purchases (median: 8 × variations)
3. **Last week top-1k** — top 1,000 trending items minus re-orders

NN track: ~30,000 active items from past 5 weeks (customer-agnostic, no per-user retrieval).

## Ranking / model

- **LightGBM Ranker** per candidate group (3 separate models)
- **Neural Network** with embedding-based similarity matching
- **Stacking**: NN predictions fed as features into GBDT
- **Final ensemble**: weighted rank averaging, weights tuned via Optuna

Stage-by-stage scores:
| Stage | CV | LB |
|---|---|---|
| cdeotte baseline | 0.0264 | 0.021 |
| GBDT candidate groups | 0.0390 | 0.0324 |
| NN (fully tuned) | 0.0348–0.0352 | 0.0298–0.0303 |
| NN + GBDT stacking | 0.0374 | 0.0313 |
| Final ensemble | 0.0405 | **0.0339** |

## Validation strategy

Weekly holdout CV across **3 folds**. Fold-0 designated primary. Test week = **2020-09-22**; training lookback = 2018-09-20 onward.

## Stated pitfalls / lessons

- **"Retrieve → Ranking insufficient recall"** in the retrieval phase — flagged as their biggest weakness.
- **ItemCF was explored but never integrated** — regret point.
- **NN was confined to final ranking** — should have preceded GBDT (architectural ordering matters).
- **Actionable regret:** two-stage embedding-retrieval → transformer-ranking approach (used by 46th place) showed **+0.007 gain** — Fujikawa proposed it early, deprioritized it.

## Notable insights / quotes

Fujikawa explicitly summarizes lessons from the leaderboard:

> 1st place: **"candidate strategy is key"** — achieved 0.0379 private. "Feature engineering contribution limited; candidate selection critical."

> 2nd place: **Postal code / age-based popularity grouping effective**.

> Cross-team: **"Simple 600-item retrieval beats complex ItemCF approaches"**.

## Reported metrics

- MAP@12 public final: **0.0339**
- MAP@12 private final: not stated
- 1st place private MAP@12 (cited): **0.0379**
- Baseline cdeotte: CV 0.0264, LB 0.021
- Full ablation ladder (above) — extremely useful for thesis context
- No HR@K / NDCG@K reported

## Code / notebooks

No public repo linked in the slides.

## Thesis relevance

- **Phase 4.0/4.3 (feature ROI):** crystal-clear quantified ablations — metadata embeddings biggest single boost (+0.0044 CV); image+graph embeddings smallest (+0.0007). **Suggests rich metadata > rich image features in this domain at this rank.** Strong evidence for prioritizing structured features in feature-aware prototypes.
- **Phase 4.5 (explanations):** "Other colors" and "Re-order" as candidate channels are *human-interpretable* — they map almost directly to plausible explanation prototypes (e.g., "Recommended because it's the same product in a different color you've bought"). Direct narrative for our explainability story.
- **Phase 3.7 (CF context):** **The 1st place insight — "candidate strategy is key, feature engineering contribution limited"** — is a *direct challenge* to our feature-aware thesis premise. Need to address head-on: either (a) CF-only candidates with feature-aware ranking still wins, or (b) features change the candidate frontier itself. **Important framing for thesis introduction.**
- **CF-only ceiling:** GBDT-candidate-groups (basically CF + heuristic retrieval) hit LB 0.0324 — about 85% of the 1st-place private score (0.0379). Useful upper-bound estimate for a CF-style approach.
- **Thesis-citable:** Fujikawa's quantified ablation table is gold for our Ch. 6 results discussion.
