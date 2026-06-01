# Wp-Zhang — 45th place silver (GitHub repo)

**Source ID:** `02_wp_zhang_45th_silver`
**Type:** writeup (GitHub README)
**URL:** https://github.com/Wp-Zhang/H-M-Fashion-RecSys
**Author / Team:** Wp-Zhang (solo or small team — not specified)
**Date posted:** 2022 (post-competition)
**Audience signal:** Public GitHub repo with full pipeline code
**Final score:** 45/3006 — Silver. **Public LB 0.0292, private LB 0.02996.**
**Thesis tags:** `ensemble` `image-features` `text-features` `cf-only` `temporal-split`

---

## Architecture summary

Two-stage with two distinct candidate generation strategies → three ranking models per strategy (LGBMRanker, LGBMClassifier, DNN) → ensemble across both strategies. Ensembling the two strategies is presented as the key driver: "Candidates from the two strategies are quite different so that ensembling the ranking results can help to improve the score."

## Features used

- **Item features**: not enumerated in README.
- **User features**: not enumerated in README.
- **Engineered**: extensive embedding-based features (see below).
- **Image / text features**: **YES** — uses image embeddings as part of the engineered feature stack.

## Feature engineering specifics

Multiple pre-trained embedding families used as features for both retrieval and ranking:
- DSSM (user & item)
- YouTube-inspired (user & item)
- Word2Vec standard (user, item, product)
- Word2Vec skip-gram (user, item, product)
- Image embeddings

Note: actual column lists for `articles.csv` / `customers.csv` not detailed in README — would require diving into the source code.

## Candidate generation

Two distinct strategies (named "LGB Recall 1" and a second). README doesn't fully document them but emphasizes they produce different candidate sets — the diversity is the point.

## Ranking / model

Per candidate strategy: 3 models — LightGBM Ranker, LightGBM Classifier, Deep Neural Network. Total: 6 ranking models. Final score is ensemble across all 6.

## Validation strategy

Trained on **4 weeks of data** only (hardware-constrained — 50G RAM). Average 50 candidates per user generated. Not detailed beyond this in README.

## Stated pitfalls / lessons

- Hardware limits (50G RAM) → had to cap to 4 weeks of training data and ~50 candidates/user.
- Feature engineering is centralized in "LGB Recall 1.ipynb" → must be run first (engineering pattern: features computed once, reused across all models).

## Notable insights / quotes

> "Candidates from the two strategies are quite different so that ensembling the ranking results can help to improve the score."

> Single recall strategy: LB 0.0286 → ensembled approach: LB 0.0292 — **+0.0006 improvement attributed purely to retrieval diversity**.

## Reported metrics

- MAP@12 public: **0.0292**
- MAP@12 private: **0.02996**
- Ablation: single strategy 0.0286 → ensemble 0.0292 (+2.1% relative)
- No HR@K / NDCG@K reported.

## Code / notebooks

Full pipeline available at https://github.com/Wp-Zhang/H-M-Fashion-RecSys — cookiecutter-data-science structure with `src/data/`, `src/features/`, `src/retrieval/`, `notebooks/`, `models/`.

## Thesis relevance

- **Phase 4.0/4.3:** image embeddings DID make it into a silver-medal solution — image features are usable but not mandatory at this rank.
- **Phase 3.7:** the +2.1% delta from candidate diversity quantifies retrieval-stage importance vs. ranking — supports the "candidate strategy is key" insight that recurs in higher-ranked solutions.
- **Phase 4.4 (ablations):** this is a clean reference point — 4-week training window, ~50 candidates, ensemble of 6 → 0.0292 public.
- **Investigate further:** source code may reveal which articles.csv columns mattered; worth a follow-up extraction pass if Phase 4.0 needs more depth.
