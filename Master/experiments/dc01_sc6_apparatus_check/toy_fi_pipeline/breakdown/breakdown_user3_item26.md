# Why item 26 (Dress · Red · Trouser) for user 3? — fI-ProtoMF (feature_item_proto)

Score **S = +0.5292** = rank-inert baseline Σ_k u_k = **-2.2313** (identical for every item this user ranks) + item-discriminating **+2.7605**.

## per-prototype contribution u_k·(t*_k − 1)

| line | contribution |
|---|---|
| p2 · Jersey, White, Bottom | +1.3486 |
| p0 · White, Bottom, Shoe | +1.0462 |
| p4 · Divided, Mens, Blouse | +0.3027 |
| p1 · Sneaker | +0.2443 |
| p3 · Dress, Green, All over pattern | -0.1753 |
| p6 · Trouser, Top, Beige | -0.0411 |
| p5 · Mens | +0.0350 |

## inside p2 · Jersey, White, B…: exact shares u_k·c(r,k)

| row | share (score units) |
|---|---|
| department = Trouser | +0.0211 |
| product_type = Dress | +0.2622 |
| section = Divided | +0.6918 |
| colour_group = Red | -0.1077 |
| graphical_appearance = All over pattern | +0.0416 |
| item-ID row (own memory) *(ID row)* | +0.4396 |

Σ = +1.3486 (= that prototype's bar, exactly)

**feature-explained +1.4696 vs item-ID +1.2909 of the item-discriminating score +2.7605 (features: 53.2%)**

## Method notes

- Shares are EXACT additive summands of the computed score (not counterfactual effects; rows are jointly normalized through ‖q_i‖ — removing one rescales the others).
- Negative values are legitimate anti-affinities and are rendered as such.
- The rank-inert baseline Σ_k u_k = -2.2313 is identical for every item this user ranks; bars show the item-discriminating remainder.
- Prototype naming route: intrinsic — cos(e_f, p_k), read from the model's parameters.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
