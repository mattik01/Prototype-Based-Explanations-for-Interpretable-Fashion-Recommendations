# Why item 2390 (Trousers · Light Orange · Basic 1) for user 42? — fI-ProtoMF (feature_item_proto)

Score **S = +0.2545** = rank-inert baseline Σ_k u_k = **-0.5352** (identical for every item this user ranks) + item-discriminating **+0.7897**.

## per-prototype contribution u_k·(t*_k − 1)

| line | contribution |
|---|---|
| p6 · proto 6 | +0.0107 |
| p25 · proto 25 | +0.0106 |
| p7 · proto 7 | +0.0106 |
| p30 · proto 30 | +0.0106 |
| p60 · proto 60 | +0.0106 |
| p18 · proto 18 | +0.0106 |
| p0 · proto 0 | +0.0106 |
| p48 · proto 48 | +0.0106 |
| p5 · proto 5 | +0.0106 |
| p65 · proto 65 | +0.0106 |
| p66 · proto 66 | +0.0106 |
| p64 · proto 64 | +0.0106 |
| p8 · proto 8 | +0.0106 |
| p27 · proto 27 | +0.0106 |
| p43 · proto 43 | +0.0106 |
| p54 · proto 54 | +0.0106 |
| p39 · proto 39 | +0.0106 |
| p53 · proto 53 | +0.0106 |
| p24 · proto 24 | +0.0106 |
| p13 · proto 13 | +0.0106 |
| p57 · proto 57 | +0.0106 |
| p21 · proto 21 | +0.0106 |
| p46 · proto 46 | +0.0106 |
| p45 · proto 45 | +0.0106 |
| p42 · proto 42 | +0.0106 |
| p69 · proto 69 | +0.0106 |
| p37 · proto 37 | +0.0106 |
| p20 · proto 20 | +0.0106 |
| p44 · proto 44 | +0.0106 |
| p16 · proto 16 | +0.0106 |
| p4 · proto 4 | +0.0106 |
| p15 · proto 15 | +0.0106 |
| p75 · proto 75 | +0.0106 |
| p11 · proto 11 | +0.0105 |
| p55 · proto 55 | +0.0105 |
| p22 · proto 22 | +0.0105 |
| p50 · proto 50 | +0.0105 |
| p12 · proto 12 | +0.0105 |
| p1 · proto 1 | +0.0105 |
| p35 · proto 35 | +0.0105 |
| p49 · proto 49 | +0.0105 |
| p51 · proto 51 | +0.0104 |
| p14 · proto 14 | +0.0104 |
| p58 · proto 58 | +0.0104 |
| p29 · proto 29 | +0.0104 |
| p56 · proto 56 | +0.0104 |
| p33 · proto 33 | +0.0103 |
| p10 · proto 10 | +0.0103 |
| p28 · proto 28 | +0.0103 |
| p17 · proto 17 | +0.0103 |
| p23 · proto 23 | +0.0103 |
| p34 · proto 34 | +0.0103 |
| p73 · proto 73 | +0.0103 |
| p38 · proto 38 | +0.0103 |
| p26 · proto 26 | +0.0102 |
| p41 · proto 41 | +0.0102 |
| p47 · proto 47 | +0.0102 |
| p31 · proto 31 | +0.0102 |
| p70 · proto 70 | +0.0102 |
| p52 · proto 52 | +0.0102 |
| p62 · proto 62 | +0.0102 |
| p3 · proto 3 | +0.0102 |
| p68 · proto 68 | +0.0101 |
| p2 · proto 2 | +0.0101 |
| p40 · proto 40 | +0.0101 |
| p19 · proto 19 | +0.0101 |
| p74 · proto 74 | +0.0101 |
| p71 · proto 71 | +0.0101 |
| p61 · proto 61 | +0.0101 |
| p36 · proto 36 | +0.0100 |
| p9 · proto 9 | +0.0100 |
| p63 · proto 63 | +0.0100 |
| p67 · proto 67 | +0.0100 |
| p72 · proto 72 | +0.0100 |
| p32 · proto 32 | +0.0099 |
| p59 · proto 59 | +0.0094 |

## inside p6 · proto 6: exact shares u_k·c(r,k)

| row | share (score units) |
|---|---|
| item-ID row (own memory) *(ID row)* | +0.0107 |

Σ = +0.0107 (= that prototype's bar, exactly)

**feature-explained +0.0000 vs item-ID +0.7897 of the item-discriminating score +0.7897 (features: 0.0%)**

## Method notes

- Shares are EXACT additive summands of the computed score (not counterfactual effects; rows are jointly normalized through ‖q_i‖ — removing one rescales the others).
- Negative values are legitimate anti-affinities and are rendered as such.
- The rank-inert baseline Σ_k u_k = -0.5352 is identical for every item this user ranks; bars show the item-discriminating remainder.
- Prototype naming route: intrinsic — cos(e_f, p_k), read from the model's parameters.
- Self-check passed: the rendered parts sum to the model's own forward score (asserted before writing).
