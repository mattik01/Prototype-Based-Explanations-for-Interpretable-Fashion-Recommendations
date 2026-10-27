# Why item 21 (Action|Romance · m21) for user 3? — fUfI-ProtoMF (feature_user_item_proto)

Score **S = +2.3597** = item baseline 1ᵀt̂ = **-0.9327** (this item's own scalar, identical for every user; it participates in item ranking) + your baseline 1ᵀû (rank-inert) = **-0.1269** + personalised **+3.4194**.

Of the bars' total **+3.4194**, **+2.3496** is common to all users (its average over every user, for this item) and **+1.0698** is specific to you.

## C: (u*−1)·t̂    A: û·(t*−1)

| line | contribution |
|---|---|
| A5 · Action, slow, dark | +1.1790 |
| A0 · Romance, dark, slow | +0.7274 |
| A2 · slow | +0.7111 |
| C1 · Drama, long | +0.3739 |
| C3 · dark, Romance, violent | +0.3577 |
| A1 · dark, Action, Romance | +0.3168 |
| C2 · Action, violent, witty | -0.2515 |
| A3 · Drama, long, witty | -0.2000 |
| A4 · slow, Romance, dark | +0.1232 |
| C4 · Romance, dark, slow | +0.0879 |
| C0 · Romance, long, Drama | -0.0061 |

## inside A5 · Action, slow, da…: shares by item word

| row | share (score units) |
|---|---|
| genre: Action | +3.1215 |
| genre: Romance | +0.0299 |
| item ID + field means *(ID row)* | -1.9723 |

Σ = +1.1790 (= that prototype's bar, exactly)

## inside A5 · Action, slow, da…: the same bar by your purchases — rows of û_k (exact for this half)

*Alternative exact reading of the SAME bar — the two zooms regroup one sum and are never added together.*

| row | share (score units) |
|---|---|
| your row + field means *(ID row)* | +1.4577 |
| item 40 (Drama|Horror|Romance · m40) | +0.1655 |
| item 84 (Action|Comedy · m84) | -0.1652 |
| item 10 (Action · m10) | -0.1181 |
| item 18 (Comedy|Horror|Romance · m18) | -0.0821 |
| item 26 (Action|Horror|Romance · m26) | -0.0768 |
| item 73 (Comedy · m73) | -0.0020 |

Σ = +1.1790 (= the zoomed bar, exactly)

## Inside the item baseline: this item's words (baseline offset per word)

*Exact summands of 1ᵀt̂ — the same for every user.*

| row | offset |
|---|---|
| genres = Romance | -0.9187 |
| item ID + field means | -0.1521 |
| genres = Action | +0.1381 |

## Each bar split: common to all users vs specific to you

*Common to all users = the bar's average over every user the model scores, for this item (the user-average of the user factor times this item's factor). It is a per-item quantity every user gets; 'specific to you' is your deviation from it (F-DC09-09).*

| bar | contribution | common to all users | specific to you |
|---|---|---|---|
| A5 · Action, slow, dark | +1.1790 | +0.8026 | +0.3764 |
| A0 · Romance, dark, slow | +0.7274 | +0.0718 | +0.6556 |
| A2 · slow | +0.7111 | +1.4746 | -0.7635 |
| C1 · Drama, long | +0.3739 | +0.3847 | -0.0108 |
| C3 · dark, Romance, violent | +0.3577 | +0.3762 | -0.0186 |
| A1 · dark, Action, Romance | +0.3168 | +0.2126 | +0.1042 |
| C2 · Action, violent, witty | -0.2515 | -0.1263 | -0.1252 |
| A3 · Drama, long, witty | -0.2000 | -0.2920 | +0.0920 |
| A4 · slow, Romance, dark | +0.1232 | -0.6374 | +0.7606 |
| C4 · Romance, dark, slow | +0.0879 | +0.0899 | -0.0020 |
| (+ 1 smaller bar) | -0.0061 | -0.0072 | +0.0010 |

## Named coverage per component

*Each personalised half is a product of a user factor and an item factor, both exact sums over rows, so it splits exactly into word×word mass (fully named) and mass that involves an ID line.*

| component | total | your words × item words | your words × item-ID line | your personal row × item words | both ID lines |
|---|---|---|---|---|---|
| U-half (communities) | +0.5618 | -0.1698 | +0.2711 | +0.8949 | -0.4345 |
| I-half (archetypes) | +2.8576 | +0.8099 | -1.9948 | -1.9305 | +5.9730 |
| item baseline | -0.9327 | -0.7806 | — | — | -0.1521 (item-ID line) |

## What if this item differed in one word? (exact re-score, item side)

*Each row re-scores a hypothetical item that differs from this one only in that word, replaced by another value of the same field (same item-ID row; both halves move).*

| this item's word | lowest-scoring substitute (ΔS) | highest-scoring substitute (ΔS) |
|---|---|---|
| genres = Action | → genres = Comedy: -1.7448 | → genres = Drama: -0.5279 |
| genres = Romance | → genres = Comedy: -0.9603 | → genres = Drama: -0.2520 |

**personalised +3.4194 = common to all users +2.3496 + specific to you +1.0698**

**named (word × word): U-half -30.2% · I-half 28.3% of each half; the rest involves an ID line**

## Method notes

- Every number is an exact summand of the model's own forward score (self-checked). Shares are not counterfactual effects: the cosine factors are jointly normalised through ‖q‖, and removing a row rescales the others. The what-if table is the counterfactual reading (exact re-score).
- Word rows are read relative to their field: each word's row minus the mean of its field's vocabulary rows, with the ID line carrying the field means. The rows still sum to the composed vector exactly, so every total is unchanged.
- The item baseline 1ᵀt̂ is this item's own scalar: the same for every user, and it participates in item ranking. Your baseline 1ᵀû is the same for every item you rank (rank-inert). Bars show the personalised remainder.
- Bars pool both units: C_l = taste community (user prototype), contribution (u*_l − 1)·t̂_l; A_k = item archetype, contribution û_k·(t*_k − 1).
- Negative values are legitimate anti-affinities and are rendered as such.
- Prototype naming route: intrinsic — cos(e_f, p) on both sides, words relative to their field mean.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
