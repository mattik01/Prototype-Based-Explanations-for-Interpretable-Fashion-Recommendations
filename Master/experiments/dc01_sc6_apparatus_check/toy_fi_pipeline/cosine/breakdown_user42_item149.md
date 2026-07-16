# Why item 149 (Bottom · Beige · Jersey) for user 42? — fI-ProtoMF (feature_item_proto)

Score **S = +4.2219** = rank-inert baseline Σ_k u_k = **+2.8621** (identical for every item this user ranks) + item-discriminating **+1.3598**.

## per-prototype contribution u_k·(t*_k − 1)

| line | contribution |
|---|---|
| p3 · Dress, Green, All over pattern | +1.0773 |
| p2 · Jersey, White, Bottom | +0.1432 |
| p4 · Divided, Mens, Blouse | +0.1248 |
| p5 · Mens | +0.0302 |
| p6 · Trouser, Top, Beige | -0.0153 |
| p1 · Sneaker | -0.0031 |
| p0 · White, Bottom, Shoe | +0.0027 |

## inside p3 · Dress, Green, Al…: exact shares u_k·c(r,k)

| row | share (score units) |
|---|---|
| department = Jersey | +0.2412 |
| product_type = Bottom | +0.0702 |
| section = Divided | +0.2126 |
| colour_group = Beige | +0.1732 |
| graphical_appearance = All over pattern | +0.2039 |
| item-ID row (own memory) *(ID row)* | +0.1763 |

Σ = +1.0773 (= that prototype's bar, exactly)

**feature-explained +1.1264 vs item-ID +0.2334 of the item-discriminating score +1.3598 (features: 82.8%)**

## Method notes

- Shares are EXACT additive summands of the computed score (not counterfactual effects; rows are jointly normalized through ‖q_i‖ — removing one rescales the others).
- Negative values are legitimate anti-affinities and are rendered as such.
- The rank-inert baseline Σ_k u_k = +2.8621 is identical for every item this user ranks; bars show the item-discriminating remainder.
- Prototype naming route: intrinsic — cos(e_f, p_k), read from the model's parameters.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
