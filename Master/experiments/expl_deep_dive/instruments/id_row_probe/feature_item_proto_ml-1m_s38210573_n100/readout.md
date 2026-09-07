# ID-row probe — `feature_item_proto_ml-1m_s38210573_n100`

*working evidence (single checkpoint, post-hoc) — not thesis numbers.* dataset `ml-1m`, layout `bags`, fields ['genres', 'tags'], n_items 3125, V 1061, d 78, K 77, has_id True; sections ['A', 'B', 'C']; wall 196.4s.

## Anchors (cross-check vs composition_analysis_2026-08-24)

- eff_rank_raw: 1.6123
- corr_bagsize_qmeta_norm: 0.8495
- id_meta_norm_ratio_median: 0.2986
- id_share_qsq_mean: 0.1315

## B — popularity-stratified ID share

Spearman with train popularity: id_share_qsq 0.139, id_meta_norm_ratio -0.080, id_share_activation -0.193, id_norm 0.591, qmeta_norm 0.534

| band | n | id_share_qsq mean | median | norm ratio median | act-level ID share mean | mean activation |
|---|---:|---:|---:|---:|---:|---:|
| overall | 3125 | 0.131 | 0.092 | 0.299 | 0.028 | 1.023 |
| 1-10 | 401 | 0.117 | 0.067 | 0.320 | 0.333 | 1.036 |
| 11-100 | 1442 | 0.129 | 0.086 | 0.314 | 0.094 | 1.031 |
| 101+ | 1282 | 0.138 | 0.102 | 0.280 | -0.142 | 1.010 |
| d0 (pop 4-8) | 283 | 0.116 | 0.067 | 0.320 | 0.362 | 1.036 |
| d1 (pop 9-14) | 301 | 0.111 | 0.069 | 0.319 | 0.247 | 1.035 |
| d2 (pop 15-24) | 329 | 0.132 | 0.083 | 0.328 | 0.200 | 1.035 |
| d3 (pop 25-40) | 334 | 0.132 | 0.092 | 0.323 | 0.037 | 1.033 |
| d4 (pop 41-65) | 314 | 0.131 | 0.086 | 0.301 | 0.019 | 1.027 |
| d5 (pop 66-105) | 309 | 0.135 | 0.094 | 0.295 | 0.014 | 1.028 |
| d6 (pop 106-163) | 315 | 0.137 | 0.098 | 0.290 | -0.012 | 1.020 |
| d7 (pop 164-267) | 315 | 0.157 | 0.118 | 0.323 | -0.067 | 1.017 |
| d8 (pop 268-486) | 312 | 0.131 | 0.100 | 0.272 | -0.197 | 1.008 |
| d9 (pop 487-2804) | 313 | 0.130 | 0.099 | 0.239 | -0.295 | 0.993 |

D2: ‖mean e_ID‖ / mean ‖e_ID‖ = **0.069**; cos(mean e_ID, mean q_meta) = 0.973; eff. rank of ID block 6.90 (uncentred 6.89); ‖e_ID‖ median 0.151.

## C — bag-size channel (Spearman)

- spearman_bagsize_pop: 0.672
- spearman_bagsize_mean_activation: -0.276
- spearman_bagsize_max_activation: 0.044
- spearman_bagsize_qmeta_norm: 0.850
- spearman_bagsize_id_share_qsq: -0.374
- spearman_bagsize_id_meta_norm_ratio: -0.502

| bag-size quartile | n | bag range | train pop mean | mean activation | id_share mean |
|---|---:|---|---:|---:|---:|
| q0 | 668 | 1-6 | 34.8 | 1.036 | 0.219 |
| q1 | 805 | 6-10 | 69.4 | 1.028 | 0.133 |
| q2 | 822 | 10-16 | 146.7 | 1.021 | 0.110 |
| q3 | 830 | 16-72 | 436.9 | 1.010 | 0.081 |

## A — pair-encoding regression (U4 trigger i)

Design: 1061 unary cols; pairs 91216 unique / 5000 kept (support ≥ 5, cross-field only False); 5-fold CV over 3125 items; null = 20 X2-row permutations.

| response | λ | R² unary | R² +pairs | ΔR² | null ΔR² mean | null p95 | perm p | HOT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| e_ID | 1000 | -0.0077 | -0.0184 | -0.0108 | -0.0043 | -0.0025 | 1.000 | no |
| c_ID | 1000 | 0.0078 | 0.0114 | +0.0036 | -0.0046 | -0.0028 | 0.048 | **YES** |

Top pairs by coefficient norm (e_ID response, full-data fit):

- genres=`Comedy` × genres=`Drama` — coef 0.008, support 186
- genres=`Comedy` × tags=`original` — coef 0.005, support 240
- genres=`Action` × genres=`Drama` — coef 0.004, support 85
- genres=`Drama` × genres=`Thriller` — coef 0.003, support 93
- genres=`Comedy` × genres=`Romance` — coef 0.003, support 183
- genres=`Action` × genres=`Comedy` — coef 0.003, support 62
- genres=`Drama` × genres=`Romance` — coef 0.003, support 186
- genres=`Children's` × genres=`Comedy` — coef 0.003, support 89
- genres=`Crime` × genres=`Drama` — coef 0.003, support 74
- genres=`Action` × tags=`original` — coef 0.003, support 114
- genres=`Thriller` × tags=`original` — coef 0.002, support 136
- genres=`Action` × genres=`Sci-Fi` — coef 0.002, support 102
- genres=`Thriller` × tags=`action` — coef 0.002, support 96
- genres=`Comedy` × tags=`love story` — coef 0.002, support 68
- genres=`Drama` × tags=`original` — coef 0.002, support 272
- genres=`Comedy` × tags=`comedy` — coef 0.002, support 333
- tags=`action` × tags=`original` — coef 0.002, support 47
- genres=`Comedy` × tags=`mentor` — coef 0.002, support 43
- genres=`Comedy` × tags=`drama` — coef 0.002, support 39
- tags=`comedy` × tags=`original` — coef 0.002, support 84

Coefficient norms: unary median 0.000, pairs median 0.001.

Reading rule (pre-registered): HOT iff ΔR² > null p95 on the e_ID response. The permutation null keeps the pair columns' marginals but breaks their link to the item (and to its unary indicators), so it measures pure overfit inflation of equally sparse columns.
