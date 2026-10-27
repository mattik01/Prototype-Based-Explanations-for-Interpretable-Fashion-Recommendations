# Why item 1 (Bottom · White · Trouser) for user 3? — fUfI-ProtoMF (feature_user_item_proto_noid_sep)

Score **S = +2.2242** = item baseline 1ᵀt̂ = **-1.8790** (this item's own scalar, identical for every user; it participates in item ranking) + your baseline 1ᵀû (rank-inert) = **+3.1685** + personalised **+0.9347**.

Of the bars' total **+0.9347**, **+0.9162** is common to all users (its average over every user, for this item) and **+0.0185** is specific to you.

## C: (u*−1)·t̂    A: û·(t*−1)

| line | contribution |
|---|---|
| A1 · Mens, Red, Green | -1.4173 |
| C2 · Divided, White, Knitwear | +1.3786 |
| A2 · Red, All over pattern, Bottom | +0.4603 |
| C0 · Dress, White, Black | -0.4346 |
| A5 · Mens, Blue, Divided | +0.3837 |
| A4 · Bottom, All over pattern, Knitwear | +0.2319 |
| A3 · Shoe, Beige, Divided | +0.1881 |
| C1 · Dress, Red, Blouse | +0.1550 |
| C4 · Bottom, Trouser, Shoe | +0.1276 |
| A0 · Jersey, Trouser, Bottom | -0.1271 |
| C3 · Red, Dress, Beige | -0.0114 |

## inside A1 · Mens, Red, Green: shares by item word

| row | share (score units) |
|---|---|
| dept: Trouser | +0.3856 |
| type: Bottom | -0.3258 |
| section: Mens | -0.4988 |
| colour: White | -0.4305 |
| pattern: Stripe | -0.5478 |

Σ = -1.4173 (= that prototype's bar, exactly)

## inside A1 · Mens, Red, Green: the same bar by your purchases — rows of û_k (exact for this half)

*Alternative exact reading of the SAME bar — the two zooms regroup one sum and are never added together.*

| row | share (score units) |
|---|---|
| item 7 (Sneaker · White · Knitwear) | -0.5768 |
| item 4 (Top · Green · Jersey) | -0.3682 |
| item 11 (Sneaker · Beige · Trouser) | -0.3631 |
| item 5 (Bottom · Beige · Blouse) | -0.1964 |
| item 18 (Dress · Black · Shoe) | +0.0872 |

Σ = -1.4173 (= the zoomed bar, exactly)

## Inside the item baseline: this item's words (baseline offset per word)

*Exact summands of 1ᵀt̂ — the same for every user.*

| row | offset |
|---|---|
| department = Trouser | -1.9158 |
| colour_group = White | +0.8030 |
| graphical_appearance = Stripe | -0.8007 |
| product_type = Bottom | +0.6542 |
| section = Mens | -0.6198 |

## Each bar split: common to all users vs specific to you

*Common to all users = the bar's average over every user the model scores, for this item (the user-average of the user factor times this item's factor). It is a per-item quantity every user gets; 'specific to you' is your deviation from it (F-DC09-09).*

| bar | contribution | common to all users | specific to you |
|---|---|---|---|
| A1 · Mens, Red, Green | -1.4173 | -1.0954 | -0.3219 |
| C2 · Divided, White, Knitwear | +1.3786 | +1.2055 | +0.1731 |
| A2 · Red, All over pattern, Bottom | +0.4603 | +0.3572 | +0.1031 |
| C0 · Dress, White, Black | -0.4346 | -0.2685 | -0.1661 |
| A5 · Mens, Blue, Divided | +0.3837 | +0.4782 | -0.0945 |
| A4 · Bottom, All over pattern, Knitwear | +0.2319 | +0.1720 | +0.0599 |
| A3 · Shoe, Beige, Divided | +0.1881 | +0.1088 | +0.0793 |
| C1 · Dress, Red, Blouse | +0.1550 | +0.2950 | -0.1400 |
| C4 · Bottom, Trouser, Shoe | +0.1276 | +0.1265 | +0.0010 |
| A0 · Jersey, Trouser, Bottom | -0.1271 | -0.4477 | +0.3205 |
| (+ 1 smaller bar) | -0.0114 | -0.0155 | +0.0041 |

## Named coverage per component

*Each personalised half is a product of a user factor and an item factor, both exact sums over rows, so it splits exactly into word×word mass (fully named) and mass that involves an ID line.*

| component | total | your words × item words | your words × item-ID line | your personal row × item words | both ID lines |
|---|---|---|---|---|---|
| U-half (communities) | +1.2151 | +1.2151 | +0.0000 | +0.0000 | +0.0000 |
| I-half (archetypes) | -0.2804 | -0.2804 | +0.0000 | +0.0000 | +0.0000 |
| item baseline | -1.8790 | -1.8790 | — | — | +0.0000 (item-ID line) |

## What if this item differed in one word? (exact re-score, item side)

*Each row re-scores a hypothetical item that differs from this one only in that word, replaced by another value of the same field (same item-ID row; both halves move).*

| this item's word | lowest-scoring substitute (ΔS) | highest-scoring substitute (ΔS) |
|---|---|---|
| graphical_appearance = Stripe | → graphical_appearance = All over pattern: -1.4338 | → graphical_appearance = Solid: -0.8038 |
| colour_group = White | → colour_group = Green: -0.8083 | → colour_group = Black: -0.2203 |
| department = Trouser | → department = Blouse: -0.6370 | → department = Jersey: -0.2512 |
| product_type = Bottom | → product_type = Dress: -0.3834 | → product_type = Top: +0.4245 |
| section = Mens | → section = Womens: +0.1732 | → section = Divided: +1.3620 |

**personalised +0.9347 = common to all users +0.9162 + specific to you +0.0185**

**noid arm: no ID rows — every component is a sum over words only**

## Method notes

- Every number is an exact summand of the model's own forward score (self-checked). Shares are not counterfactual effects: the cosine factors are jointly normalised through ‖q‖, and removing a row rescales the others. The what-if table is the counterfactual reading (exact re-score).
- No ID rows: the word rows are read as trained.
- The item baseline 1ᵀt̂ is this item's own scalar: the same for every user, and it participates in item ranking. Your baseline 1ᵀû is the same for every item you rank (rank-inert). Bars show the personalised remainder.
- Bars pool both units: C_l = taste community (user prototype), contribution (u*_l − 1)·t̂_l; A_k = item archetype, contribution û_k·(t*_k − 1).
- Negative values are legitimate anti-affinities and are rendered as such.
- Prototype naming route: intrinsic — cos(e_f, p) on both sides.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
