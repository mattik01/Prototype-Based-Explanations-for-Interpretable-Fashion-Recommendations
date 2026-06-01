# Singha Roy, D'Amico, Lawlor, Hurley — FLAIRS 2023 academic paper

**Source ID:** `08_singharoy_damico_flairs2023`
**Type:** paper (peer-reviewed academic publication)
**URL:** https://journals.flvc.org/FLAIRS/article/view/133307 (PDF: download/133307/137921)
**Author / Team:** Aayush Singha Roy, Edoardo D'Amico, Aonghus Lawlor, Neil Hurley (UCD + Insight Centre for Data Analytics, Dublin)
**Date posted:** 2023 (FLAIRS Conference Proceedings, Volume 36)
**Audience signal:** Peer-reviewed academic publication
**Final score:** **Top 2% final ranking** (~rank 60/2954 — gold/high-silver). MAP@12 0.0328 (Cosine+WI+TW+R), 0.0343 (RP3β+WI+TW+R).
**Thesis tags:** `temporal-split` `cf-only` `seasonality` `repeat-purchase` `feature-importance` `ensemble`

---

## Architecture summary

Two-stage pipeline formalized: **ItemKNN candidate generator → LightGBM reranker**. Authors contribute two domain-specific modifications: **whitelisting (WI)** and **time-weighted interactions (TW)**, both motivated by fashion-specific temporal characteristics.

## Features used

- **Item features**: 24 columns from `articles.csv` (used by both retriever and reranker).
- **User features**: 6 columns from `customers.csv`.
- **Context features**: 3 per-transaction features.
- **Engineered features**: 65 total in the reranker (item + user + context concatenated, plus engineered additions). Specific feature names not enumerated.
- **Image / text features**: **Not used** — pure CF + tabular reranker.

## Feature engineering specifics

- **Time-weighted interactions (TW):** the user×item interaction matrix is weighted by a decay function `f(Δt)` before computing similarity. Tested: constant, linear, exponential, power-law. **Power-law wins** — gives more importance to recent interactions but preserves long-tail history (exponential decays too aggressively).
- **Whitelisting items (WI):** at prediction time, restrict the candidate space to items transacted within a recent window (best: **30 days**). Yields 90% coverage of holdout-week purchases from only 20% of the catalog.

## Candidate generation

**Item-item KNN collaborative filtering** with two similarity measures tested:
- **Cosine** similarity
- **RP3β** (random walk-based, Christoffel et al. 2015) — slightly better

Output: top-k items per user (k varied by experiment, MAP@12/@100/@200 reported).

## Ranking / model

**LightGBM** reranker. Inputs: 65 features (user + item + context). Loss: listwise LTR objective. Validation: held out last week of training data for hyperparameter selection.

## Validation strategy

- Temporal holdout: last week of training = validation; full training data used for final fit after hyperparameter selection.
- Test week = competition holdout (2020-09-23 → 2020-09-29).
- Evaluation: MAP@K and Recall@K for K ∈ {12, 100, 200}.

## Stated pitfalls / lessons

Two fashion-domain observations, both validated empirically:
1. **Items have lives** — k-means clustering on sales profiles yields two groups: *seasonal items* (sales drop to 0 within 4 months) and *long-lived items* (slow decay). Only ~35K of 106K items are "alive" at any time.
2. **Tastes drift fast** — user interaction recency carries more signal than older interactions. Power-law decay outperforms constant/linear/exponential.

## Notable insights / quotes

> "Going back 30 days from the start of a given holdout week and selecting the whitelisted items as the ones transacted within this timeframe leads to 90% coverage over the items transacted in the holdout week."

> "Only 20% of the total items in the catalogue are responsible for 90% of the transactions in the holdout week."

> "The findings indicate that applying the modifications together leads to a significant increase in the overall performance of the pipeline, resulting in a minimum gain of 35.46% and a maximum gain of 54.71%."

## Reported metrics

Full ablation table (their main result — extremely thesis-citable):

| Setup | MAP@12 | MAP@100 | Recall@12 | Recall@100 |
|---|---|---|---|---|
| Cosine (plain) | 0.0115 | 0.0132 | 0.0286 | 0.0881 |
| Cosine + WI | 0.0136 | 0.0160 | 0.0346 | 0.0992 |
| Cosine + WI + TW | 0.0271 | 0.0290 | 0.0536 | 0.1121 |
| **Cosine + WI + TW + Reranker** | **0.0328** | 0.0351 | 0.0657 | 0.0853 |
| RP3β (plain) | 0.0120 | 0.0140 | 0.0297 | 0.0890 |
| RP3β + WI + TW | 0.0275 | 0.0298 | 0.0548 | 0.1173 |
| **RP3β + WI + TW + Reranker** | **0.0343** | 0.0369 | 0.0733 | 0.0889 |

**Time-weight ablation (power-law wins):**

| Weight function | Cosine MAP@12 | RP3β MAP@12 |
|---|---|---|
| Constant | 0.0115 | 0.0120 |
| Linear | 0.0161 | 0.0167 |
| Exponential | 0.0262 | 0.0270 |
| **Power-Law** | **0.0268** | **0.0272** |

## Code / notebooks

**Public GitHub:** https://github.com/damicoedoardo/HnMChallenge — actual challenge code from the team.

## Thesis relevance

- **Phase 2.5 / Ch. 2 (related work):** **First and only peer-reviewed academic paper directly on the H&M Kaggle dataset** found in this sweep. Highest-quality citation for thesis. Cleanly motivates "fashion is different from generic recsys."
- **Phase 4.3 (architecture):** validates the ItemKNN-retriever + LightGBM-reranker baseline as a strong-but-simple architecture. **Top 2% rank with no NN, no graph embeddings, no images** — proves the value of well-engineered temporal CF.
- **Phase 4.4 (ablations):** the WI + TW ablation table is a model for *how* to present feature-aware ablations in our own thesis. Two independent modifications, each measured alone and together — exactly the structure our prototype-vs-feature-aware-prototype comparisons should use.
- **Phase 4.0 (feature ROI):** **20% of items carry 90% of holdout transactions** is a direct fashion-domain statistic we can cite in thesis introduction. Strong motivation for prototype-based methods (a small number of representative item-clusters genuinely span most of the demand).
- **Phase 4.5 (explanations):** the "items have lives" finding (seasonal vs. long-lived) is a natural axis for *prototype interpretability* — feature-aware prototypes could explicitly model item-life-stage as an explanation dimension.
- **Caveat / opportunity:** authors used 65 features in the reranker but did *not* use image/text features. **Our feature-aware prototype work has clean room to layer on multimodal features atop their CF+temporal baseline.**
