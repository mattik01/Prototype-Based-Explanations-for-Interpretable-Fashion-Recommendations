# Why item 12 (Comedy · m12) for user 3? — UI-ProtoMF (user_item_proto)

Score **S = +2.4586** = item baseline 1ᵀt̂ = **-0.0601** (this item's own scalar, identical for every user; it participates in item ranking) + your baseline 1ᵀû (rank-inert) = **+1.3075** + personalised **+1.2113**.

Of the bars' total **+1.2113**, **-0.0068** is common to all users (its average over every user, for this item) and **+1.2181** is specific to you.

## C: (u*−1)·t̂    A: û·(t*−1)

| line | contribution |
|---|---|
| A3 · slow, quirky, witty | +0.6743 |
| C0 · funny, Romance, Comedy | +0.3510 |
| C1 · quirky, witty, funny | -0.2489 |
| C2 · dark, witty, violent | -0.2000 |
| A1 · witty, dark, Comedy | +0.1882 |
| C4 · funny, slow, long | +0.1834 |
| A5 · quirky, Romance, Action | +0.1256 |
| C3 · long, Drama, violent | +0.0561 |
| A2 · dark, Comedy, Drama | +0.0509 |
| A4 · funny, Drama, Horror | +0.0403 |
| A0 · Action, slow, Horror | -0.0096 |

## inside A3 · slow, quirky, wi…: nearest items (post-hoc)

- item 12 (Comedy · m12) (1+cos = 1.985)
- item 63 (Romance · m63) (1+cos = 1.881)
- item 19 (Comedy|Drama|Romance · m19) (1+cos = 1.682)
- item 33 (Action|Horror|Romance · m33) (1+cos = 1.577)
- item 4 (Romance · m4) (1+cos = 1.562)
- item 79 (Horror|Romance · m79) (1+cos = 1.511)
- item 15 (Action|Drama · m15) (1+cos = 1.505)
- item 72 (Action|Drama|Horror · m72) (1+cos = 1.500)

## Each bar split: common to all users vs specific to you

*Common to all users = the bar's average over every user the model scores, for this item (the user-average of the user factor times this item's factor). It is a per-item quantity every user gets; 'specific to you' is your deviation from it (F-DC09-09).*

| bar | contribution | common to all users | specific to you |
|---|---|---|---|
| A3 · slow, quirky, witty | +0.6743 | +0.0642 | +0.6102 |
| C0 · funny, Romance, Comedy | +0.3510 | -0.0466 | +0.3976 |
| C1 · quirky, witty, funny | -0.2489 | +0.0518 | -0.3007 |
| C2 · dark, witty, violent | -0.2000 | +0.0046 | -0.2046 |
| A1 · witty, dark, Comedy | +0.1882 | -0.0441 | +0.2323 |
| C4 · funny, slow, long | +0.1834 | -0.0277 | +0.2111 |
| A5 · quirky, Romance, Action | +0.1256 | +0.0092 | +0.1164 |
| C3 · long, Drama, violent | +0.0561 | -0.0123 | +0.0684 |
| A2 · dark, Comedy, Drama | +0.0509 | +0.0233 | +0.0276 |
| A4 · funny, Drama, Horror | +0.0403 | -0.0205 | +0.0608 |
| (+ 1 smaller bar) | -0.0096 | -0.0087 | -0.0009 |

**personalised +1.2113 = common to all users -0.0068 + specific to you +1.2181**

## Method notes

- The zoom panel is a POST-HOC interpretation — the model computes no word- or purchase-level quantity beneath t̂_l or û_k; this asymmetry vs the feature-grounded variant is the object of comparison, declared, not styled away.
- The item baseline 1ᵀt̂ is this item's own scalar: the same for every user, and it participates in item ranking. Your baseline 1ᵀû is the same for every item you rank (rank-inert). Bars show the personalised remainder.
- Bars pool both units: C_l = taste community (user prototype), contribution (u*_l − 1)·t̂_l; A_k = item archetype, contribution û_k·(t*_k − 1).
- Negative values are legitimate anti-affinities and are rendered as such.
- Prototype naming route: post-hoc — lift over each prototype's top-k items, both sides.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
