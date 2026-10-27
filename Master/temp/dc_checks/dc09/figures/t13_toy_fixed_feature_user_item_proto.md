# Why item 6 (Dress · Black · Trouser) for user 3? — fUfI-ProtoMF (feature_user_item_proto)

Score **S = +6.5734** = item baseline 1ᵀt̂ = **+3.1414** (this item's own scalar, identical for every user; it participates in item ranking) + your baseline 1ᵀû (rank-inert) = **+0.6252** + personalised **+2.8069**.

Of the bars' total **+2.8069**, **-1.4324** is common to all users (its average over every user, for this item) and **+4.2392** is specific to you.

## C: (u*−1)·t̂    A: û·(t*−1)

| line | contribution |
|---|---|
| A0 · Jersey, Beige, Top | +0.7851 |
| C0 · White, Blue, Mens | +0.6305 |
| A5 · Top, White, Divided | +0.6146 |
| C4 · Bottom, Trouser, Knitwear | +0.5441 |
| A3 · Solid, Shoe, Top | +0.2985 |
| A2 · Shoe, Red, All over pattern | +0.2207 |
| C2 · Blue, Womens, Green | -0.1953 |
| A1 · Blue, Shoe, Sneaker | -0.0849 |
| A4 · Womens, Solid, Beige | +0.0634 |
| C1 · Top, White, Jersey | -0.0550 |
| C3 · Jersey, White, Bottom | -0.0148 |

## inside A0 · Jersey, Beige, T…: shares by item word

| row | share (score units) |
|---|---|
| dept: Trouser | +0.3366 |
| type: Dress | +0.2915 |
| section: Womens | +0.2913 |
| colour: Black | -0.3107 |
| pattern: Solid | -0.2282 |
| item ID + field means *(ID row)* | +0.4046 |

Σ = +0.7851 (= that prototype's bar, exactly)

## inside A0 · Jersey, Beige, T…: the same bar by your purchases — rows of û_k (exact for this half)

*Alternative exact reading of the SAME bar — the two zooms regroup one sum and are never added together.*

| row | share (score units) |
|---|---|
| your row + field means *(ID row)* | +0.5177 |
| item 4 (Top · Green · Jersey) | +0.1933 |
| item 5 (Bottom · Beige · Blouse) | +0.1065 |
| item 18 (Dress · Black · Shoe) | -0.0528 |
| item 7 (Sneaker · White · Knitwear) | +0.0340 |
| item 11 (Sneaker · Beige · Trouser) | -0.0136 |

Σ = +0.7851 (= the zoomed bar, exactly)

## Inside the item baseline: this item's words (baseline offset per word)

*Exact summands of 1ᵀt̂ — the same for every user.*

| row | offset |
|---|---|
| graphical_appearance = Solid | +2.3091 |
| item ID + field means | +1.1318 |
| section = Womens | +1.1289 |
| colour_group = Black | -0.9907 |
| product_type = Dress | -0.4349 |
| department = Trouser | -0.0028 |

## Each bar split: common to all users vs specific to you

*Common to all users = the bar's average over every user the model scores, for this item (the user-average of the user factor times this item's factor). It is a per-item quantity every user gets; 'specific to you' is your deviation from it (F-DC09-09).*

| bar | contribution | common to all users | specific to you |
|---|---|---|---|
| A0 · Jersey, Beige, Top | +0.7851 | +0.4205 | +0.3645 |
| C0 · White, Blue, Mens | +0.6305 | -0.7257 | +1.3562 |
| A5 · Top, White, Divided | +0.6146 | -0.0551 | +0.6697 |
| C4 · Bottom, Trouser, Knitwear | +0.5441 | +0.1021 | +0.4420 |
| A3 · Solid, Shoe, Top | +0.2985 | -1.5789 | +1.8774 |
| A2 · Shoe, Red, All over pattern | +0.2207 | +0.1598 | +0.0610 |
| C2 · Blue, Womens, Green | -0.1953 | -0.0353 | -0.1600 |
| A1 · Blue, Shoe, Sneaker | -0.0849 | -0.0339 | -0.0510 |
| A4 · Womens, Solid, Beige | +0.0634 | +0.0858 | -0.0224 |
| C1 · Top, White, Jersey | -0.0550 | -0.0592 | +0.0042 |
| (+ 1 smaller bar) | -0.0148 | +0.2875 | -0.3023 |

## Named coverage per component

*Each personalised half is a product of a user factor and an item factor, both exact sums over rows, so it splits exactly into word×word mass (fully named) and mass that involves an ID line.*

| component | total | your words × item words | your words × item-ID line | your personal row × item words | both ID lines |
|---|---|---|---|---|---|
| U-half (communities) | +0.9094 | +0.4908 | -0.1814 | +0.5158 | +0.0841 |
| I-half (archetypes) | +1.8975 | +0.4320 | +0.7982 | -1.1950 | +1.8622 |
| item baseline | +3.1414 | +2.0096 | — | — | +1.1318 (item-ID line) |

## What if this item differed in one word? (exact re-score, item side)

*Each row re-scores a hypothetical item that differs from this one only in that word, replaced by another value of the same field (same item-ID row; both halves move).*

| this item's word | lowest-scoring substitute (ΔS) | highest-scoring substitute (ΔS) |
|---|---|---|
| graphical_appearance = Solid | → graphical_appearance = Stripe: -2.7388 | → graphical_appearance = All over pattern: -1.2086 |
| colour_group = Black | → colour_group = Red: -1.5355 | → colour_group = Green: +1.2430 |
| department = Trouser | → department = Blouse: -1.3116 | → department = Knitwear: -0.1207 |
| product_type = Dress | → product_type = Sneaker: -0.9020 | → product_type = Bottom: -0.3665 |
| section = Womens | → section = Divided: -0.5503 | → section = Mens: +0.0795 |

**personalised +2.8069 = common to all users -1.4324 + specific to you +4.2392**

**named (word × word): U-half 54.0% · I-half 22.8% of each half; the rest involves an ID line**

## Method notes

- Every number is an exact summand of the model's own forward score (self-checked). Shares are not counterfactual effects: the cosine factors are jointly normalised through ‖q‖, and removing a row rescales the others. The what-if table is the counterfactual reading (exact re-score).
- Word rows are read relative to their field: each word's row minus the mean of its field's vocabulary rows, with the ID line carrying the field means. The rows still sum to the composed vector exactly, so every total is unchanged.
- The item baseline 1ᵀt̂ is this item's own scalar: the same for every user, and it participates in item ranking. Your baseline 1ᵀû is the same for every item you rank (rank-inert). Bars show the personalised remainder.
- Bars pool both units: C_l = taste community (user prototype), contribution (u*_l − 1)·t̂_l; A_k = item archetype, contribution û_k·(t*_k − 1).
- Negative values are legitimate anti-affinities and are rendered as such.
- Prototype naming route: intrinsic — cos(e_f, p) on both sides, words relative to their field mean.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
