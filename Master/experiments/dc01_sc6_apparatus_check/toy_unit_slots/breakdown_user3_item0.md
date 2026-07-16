# Why item 0 (Top · Black · Blouse) for user 3? — Popularity reference

Train-split popularity count: **65**

## Mechanism

Non-personalized: every user receives the same ranking. This item was purchased 65× in the train window — popularity rank 1 of 202 (top 0.5%). Your history played no role.

## Method notes

- Score = raw train-split interaction count (rank-based scorer in eval, F-S0-08).
- Prototype naming route: n/a.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
