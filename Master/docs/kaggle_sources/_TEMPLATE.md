# <Source title — author / rank>

**Source ID:** `<short-slug-matching-filename>`
**Type:** writeup | notebook | discussion | video | blog | paper
**URL:** <link>
**Author / Team:** <name(s)>
**Date posted:** YYYY-MM-DD
**Audience signal:** upvotes / views / replies / final leaderboard rank
**Final score:** public LB MAP@12 = X.XXXX, private LB MAP@12 = X.XXXX (if applicable)
**Thesis tags:** `cold-start` `feature-importance` `explainability` `seasonality` `repeat-purchase` `dedup` `cf-only` `ensemble` `image-features` `text-features` `temporal-split` (delete unused)

---

## Architecture summary

<1 paragraph: candidate generation → ranking → ensembling. What's the overall pipeline?>

## Features used

- **Item features** (`articles.csv` columns): ...
- **User features** (`customers.csv` columns): ...
- **Engineered features**: ...
- **Image / text features**: yes / no — how?

## Feature engineering specifics

<Temporal windows, popularity decay, repeat-purchase signals, co-occurrence stats, target encoding, etc.>

## Candidate generation

<How candidates per user were generated before ranking. Strategy mix and sizes.>

## Ranking / model

<Model class, key hyperparameters, training objective.>

## Validation strategy

<How they split, leak-proofing, CV scheme, public-LB vs. local CV gap.>

## Stated pitfalls / lessons

<What they explicitly warn about or learned the hard way.>

## Notable insights / quotes

> "..." — direct quotes for thesis citation.

## Reported metrics

- MAP@12 public / private: ...
- Any HR@K / NDCG@K / recall@K / precision@K reported: ...
- Ablation numbers (feature importance, leave-one-out): ...

## Code / notebooks

<Links to any public code, notebooks, GitHub.>

## Thesis relevance

<1-3 bullets explicitly tying findings to thesis chapters or phase decisions.>
