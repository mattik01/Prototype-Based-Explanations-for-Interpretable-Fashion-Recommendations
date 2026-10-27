# Why item 66 (Dress · Black · Trouser) for user 3? — UI-ProtoMF (user_item_proto)

Score **S = +4.5621** = item baseline 1ᵀt̂ = **+1.5317** (this item's own scalar, identical for every user; it participates in item ranking) + your baseline 1ᵀû (rank-inert) = **+3.1896** + personalised **-0.1591**.

Of the bars' total **-0.1591**, **-0.4712** is common to all users (its average over every user, for this item) and **+0.3121** is specific to you.

## C: (u*−1)·t̂    A: û·(t*−1)

| line | contribution |
|---|---|
| A1 · Knitwear, Blue, Sneaker | -1.7189 |
| A4 · Green, Stripe, Mens | +1.0483 |
| A5 · Black, Jersey, Dress | +0.9487 |
| C3 · Sneaker, White, Blue | -0.2343 |
| A2 · White, Stripe, Mens | -0.1618 |
| C0 · Green, Top, Blouse | +0.1430 |
| C1 · Top, Black, Red | -0.1266 |
| C4 · White, Bottom, Stripe | -0.0641 |
| A0 · Blue, Trouser, Blouse | -0.0492 |
| A3 · Beige, Bottom, Jersey | +0.0487 |
| C2 · Knitwear, Shoe, Top | +0.0071 |

## inside A1 · Knitwear, Blue, …: nearest items (post-hoc)

- item 32 (Top · Red · Knitwear) (1+cos = 1.921)
- item 114 (Dress · Black · Jersey) (1+cos = 1.912)
- item 103 (Sneaker · White · Shoe) (1+cos = 1.804)
- item 109 (Bottom · White · Jersey) (1+cos = 1.791)
- item 67 (Sneaker · White · Knitwear) (1+cos = 1.779)
- item 87 (Sneaker · Blue · Knitwear) (1+cos = 1.773)
- item 113 (Bottom · Beige · Shoe) (1+cos = 1.766)
- item 95 (Sneaker · Beige · Blouse) (1+cos = 1.758)

## Each bar split: common to all users vs specific to you

*Common to all users = the bar's average over every user the model scores, for this item (the user-average of the user factor times this item's factor). It is a per-item quantity every user gets; 'specific to you' is your deviation from it (F-DC09-09).*

| bar | contribution | common to all users | specific to you |
|---|---|---|---|
| A1 · Knitwear, Blue, Sneaker | -1.7189 | -0.2131 | -1.5058 |
| A4 · Green, Stripe, Mens | +1.0483 | +0.0008 | +1.0475 |
| A5 · Black, Jersey, Dress | +0.9487 | -0.0527 | +1.0014 |
| C3 · Sneaker, White, Blue | -0.2343 | +0.0062 | -0.2405 |
| A2 · White, Stripe, Mens | -0.1618 | -0.0316 | -0.1301 |
| C0 · Green, Top, Blouse | +0.1430 | -0.0508 | +0.1938 |
| C1 · Top, Black, Red | -0.1266 | -0.0644 | -0.0623 |
| C4 · White, Bottom, Stripe | -0.0641 | +0.0087 | -0.0727 |
| A0 · Blue, Trouser, Blouse | -0.0492 | +0.0157 | -0.0649 |
| A3 · Beige, Bottom, Jersey | +0.0487 | -0.0885 | +0.1372 |
| (+ 1 smaller bar) | +0.0071 | -0.0013 | +0.0085 |

**personalised -0.1591 = common to all users -0.4712 + specific to you +0.3121**

## Method notes

- The zoom panel is a POST-HOC interpretation — the model computes no word- or purchase-level quantity beneath t̂_l or û_k; this asymmetry vs the feature-grounded variant is the object of comparison, declared, not styled away.
- The item baseline 1ᵀt̂ is this item's own scalar: the same for every user, and it participates in item ranking. Your baseline 1ᵀû is the same for every item you rank (rank-inert). Bars show the personalised remainder.
- Bars pool both units: C_l = taste community (user prototype), contribution (u*_l − 1)·t̂_l; A_k = item archetype, contribution û_k·(t*_k − 1).
- Negative values are legitimate anti-affinities and are rendered as such.
- Prototype naming route: post-hoc — lift over each prototype's top-k items, both sides.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
