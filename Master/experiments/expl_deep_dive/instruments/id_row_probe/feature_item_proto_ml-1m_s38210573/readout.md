# ID-row probe — `feature_item_proto_ml-1m_s38210573`

*working evidence (single checkpoint, post-hoc) — not thesis numbers.* dataset `ml-1m`, layout `bags`, fields ['genres', 'tags'], n_items 3125, V 1061, d 80, K 77, has_id True; sections ['A', 'B', 'C']; wall 196.7s.

## Anchors (cross-check vs composition_analysis_2026-08-24)

- eff_rank_raw: 1.7754
- corr_bagsize_qmeta_norm: 0.8032
- id_meta_norm_ratio_median: 0.2471
- id_share_qsq_mean: 0.1256

## B — popularity-stratified ID share

Spearman with train popularity: id_share_qsq 0.115, id_meta_norm_ratio -0.027, id_share_activation -0.113, id_norm 0.547, qmeta_norm 0.485

| band | n | id_share_qsq mean | median | norm ratio median | act-level ID share mean | mean activation |
|---|---:|---:|---:|---:|---:|---:|
| overall | 3125 | 0.126 | 0.076 | 0.247 | -0.074 | 1.020 |
| 1-10 | 401 | 0.111 | 0.055 | 0.241 | 0.046 | 1.033 |
| 11-100 | 1442 | 0.126 | 0.073 | 0.257 | -0.064 | 1.030 |
| 101+ | 1282 | 0.130 | 0.084 | 0.238 | -0.123 | 1.006 |
| d0 (pop 4-8) | 283 | 0.110 | 0.052 | 0.238 | 0.054 | 1.032 |
| d1 (pop 9-14) | 301 | 0.105 | 0.056 | 0.240 | 0.078 | 1.034 |
| d2 (pop 15-24) | 329 | 0.129 | 0.076 | 0.271 | -0.115 | 1.032 |
| d3 (pop 25-40) | 334 | 0.128 | 0.074 | 0.255 | -0.007 | 1.031 |
| d4 (pop 41-65) | 314 | 0.129 | 0.075 | 0.257 | -0.109 | 1.030 |
| d5 (pop 66-105) | 309 | 0.131 | 0.079 | 0.251 | -0.128 | 1.023 |
| d6 (pop 106-163) | 315 | 0.128 | 0.084 | 0.255 | -0.146 | 1.019 |
| d7 (pop 164-267) | 315 | 0.155 | 0.107 | 0.284 | -0.095 | 1.011 |
| d8 (pop 268-486) | 312 | 0.125 | 0.078 | 0.230 | 0.007 | 1.005 |
| d9 (pop 487-2804) | 313 | 0.113 | 0.074 | 0.193 | -0.265 | 0.986 |

D2: ‖mean e_ID‖ / mean ‖e_ID‖ = **0.086**; cos(mean e_ID, mean q_meta) = 0.984; eff. rank of ID block 4.96 (uncentred 4.94); ‖e_ID‖ median 0.074.

## C — bag-size channel (Spearman)

- spearman_bagsize_pop: 0.672
- spearman_bagsize_mean_activation: -0.357
- spearman_bagsize_max_activation: -0.002
- spearman_bagsize_qmeta_norm: 0.803
- spearman_bagsize_id_share_qsq: -0.368
- spearman_bagsize_id_meta_norm_ratio: -0.440

| bag-size quartile | n | bag range | train pop mean | mean activation | id_share mean |
|---|---:|---|---:|---:|---:|
| q0 | 668 | 1-6 | 34.8 | 1.035 | 0.214 |
| q1 | 805 | 6-10 | 69.4 | 1.032 | 0.130 |
| q2 | 822 | 10-16 | 146.7 | 1.024 | 0.106 |
| q3 | 830 | 16-72 | 436.9 | 0.994 | 0.069 |

## A — pair-encoding regression (U4 trigger i)

Design: 1061 unary cols; pairs 91216 unique / 5000 kept (support ≥ 5, cross-field only False); 5-fold CV over 3125 items; null = 20 X2-row permutations.

| response | λ | R² unary | R² +pairs | ΔR² | null ΔR² mean | null p95 | perm p | HOT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| e_ID | 1000 | -0.0040 | -0.0112 | -0.0072 | -0.0041 | -0.0009 | 0.905 | no |
| c_ID | 1000 | 0.0095 | 0.0143 | +0.0048 | -0.0042 | -0.0020 | 0.048 | **YES** |

Top pairs by coefficient norm (e_ID response, full-data fit):

- genres=`Comedy` × genres=`Drama` — coef 0.005, support 186
- genres=`Comedy` × tags=`original` — coef 0.003, support 240
- genres=`Action` × genres=`Drama` — coef 0.002, support 85
- genres=`Drama` × genres=`Thriller` — coef 0.002, support 93
- genres=`Comedy` × genres=`Romance` — coef 0.002, support 183
- genres=`Drama` × genres=`Romance` — coef 0.002, support 186
- genres=`Action` × genres=`Comedy` — coef 0.002, support 62
- genres=`Thriller` × tags=`original` — coef 0.002, support 136
- genres=`Drama` × tags=`original` — coef 0.002, support 272
- genres=`Crime` × genres=`Drama` — coef 0.002, support 74
- genres=`Action` × genres=`Sci-Fi` — coef 0.001, support 102
- genres=`Comedy` × tags=`comedy` — coef 0.001, support 333
- genres=`Drama` × tags=`based on a book` — coef 0.001, support 113
- genres=`Drama` × tags=`comedy` — coef 0.001, support 22
- genres=`Adventure` × genres=`Drama` — coef 0.001, support 31
- genres=`Comedy` × tags=`mentor` — coef 0.001, support 43
- genres=`Comedy` × tags=`relationships` — coef 0.001, support 69
- genres=`Drama` × genres=`War` — coef 0.001, support 71
- genres=`Children's` × genres=`Comedy` — coef 0.001, support 89
- tags=`action` × tags=`original` — coef 0.001, support 47

Coefficient norms: unary median 0.000, pairs median 0.000.

Reading rule (pre-registered): HOT iff ΔR² > null p95 on the e_ID response. The permutation null keeps the pair columns' marginals but breaks their link to the item (and to its unary indicators), so it measures pure overfit inflation of equally sparse columns.
