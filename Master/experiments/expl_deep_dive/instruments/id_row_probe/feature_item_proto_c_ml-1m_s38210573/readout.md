# ID-row probe — `feature_item_proto_c_ml-1m_s38210573`

*working evidence (single checkpoint, post-hoc) — not thesis numbers.* dataset `ml-1m`, layout `bags`, fields ['genres', 'tags'], n_items 3125, V 1061, d 81, K 76, has_id True; sections ['A', 'B', 'C']; wall 196.4s.

## Anchors (cross-check vs composition_analysis_2026-08-24)

- eff_rank_raw: 1.5065
- corr_bagsize_qmeta_norm: 0.8729
- id_meta_norm_ratio_median: 0.5306
- id_share_qsq_mean: 0.3241

## B — popularity-stratified ID share

Spearman with train popularity: id_share_qsq -0.476, id_meta_norm_ratio -0.519, id_share_activation -0.258, id_norm 0.231, qmeta_norm 0.705

| band | n | id_share_qsq mean | median | norm ratio median | act-level ID share mean | mean activation |
|---|---:|---:|---:|---:|---:|---:|
| overall | 3125 | 0.324 | 0.242 | 0.531 | 0.516 | 1.140 |
| 1-10 | 401 | 0.519 | 0.442 | 0.896 | 0.568 | 1.286 |
| 11-100 | 1442 | 0.389 | 0.325 | 0.666 | 0.561 | 1.234 |
| 101+ | 1282 | 0.190 | 0.136 | 0.327 | 0.450 | 0.988 |
| d0 (pop 4-8) | 283 | 0.538 | 0.489 | 0.974 | 0.582 | 1.286 |
| d1 (pop 9-14) | 301 | 0.461 | 0.383 | 0.769 | 0.519 | 1.284 |
| d2 (pop 15-24) | 329 | 0.448 | 0.361 | 0.734 | 0.552 | 1.273 |
| d3 (pop 25-40) | 334 | 0.410 | 0.353 | 0.710 | 0.581 | 1.252 |
| d4 (pop 41-65) | 314 | 0.347 | 0.302 | 0.626 | 0.564 | 1.213 |
| d5 (pop 66-105) | 309 | 0.292 | 0.249 | 0.526 | 0.565 | 1.157 |
| d6 (pop 106-163) | 315 | 0.240 | 0.203 | 0.436 | 0.726 | 1.104 |
| d7 (pop 164-267) | 315 | 0.236 | 0.180 | 0.422 | 0.912 | 1.037 |
| d8 (pop 268-486) | 312 | 0.173 | 0.119 | 0.290 | 0.296 | 0.962 |
| d9 (pop 487-2804) | 313 | 0.109 | 0.068 | 0.202 | -0.138 | 0.835 |

D2: ‖mean e_ID‖ / mean ‖e_ID‖ = **0.494**; cos(mean e_ID, mean q_meta) = 0.672; eff. rank of ID block 4.99 (uncentred 4.21); ‖e_ID‖ median 0.136.

## C — bag-size channel (Spearman)

- spearman_bagsize_pop: 0.672
- spearman_bagsize_mean_activation: -0.689
- spearman_bagsize_max_activation: -0.430
- spearman_bagsize_qmeta_norm: 0.873
- spearman_bagsize_id_share_qsq: -0.717
- spearman_bagsize_id_meta_norm_ratio: -0.705

| bag-size quartile | n | bag range | train pop mean | mean activation | id_share mean |
|---|---:|---|---:|---:|---:|
| q0 | 668 | 1-6 | 34.8 | 1.254 | 0.619 |
| q1 | 805 | 6-10 | 69.4 | 1.212 | 0.367 |
| q2 | 822 | 10-16 | 146.7 | 1.140 | 0.250 |
| q3 | 830 | 16-72 | 436.9 | 0.978 | 0.119 |

## A — pair-encoding regression (U4 trigger i)

Design: 1061 unary cols; pairs 91216 unique / 5000 kept (support ≥ 5, cross-field only False); 5-fold CV over 3125 items; null = 20 X2-row permutations.

| response | λ | R² unary | R² +pairs | ΔR² | null ΔR² mean | null p95 | perm p | HOT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| e_ID | 1000 | -0.0065 | -0.0153 | -0.0088 | -0.0048 | -0.0011 | 0.952 | no |
| c_ID | 100 | 0.1633 | 0.1753 | +0.0120 | -0.0420 | -0.0329 | 0.048 | **YES** |

Top pairs by coefficient norm (e_ID response, full-data fit):

- genres=`Comedy` × tags=`original` — coef 0.005, support 240
- genres=`Comedy` × genres=`Drama` — coef 0.003, support 186
- genres=`Comedy` × tags=`comedy` — coef 0.003, support 333
- genres=`Horror` × tags=`horror` — coef 0.002, support 189
- genres=`Action` × tags=`original` — coef 0.002, support 114
- genres=`Drama` × tags=`adapted from:book` — coef 0.002, support 106
- tags=`comedy` × tags=`original` — coef 0.002, support 84
- genres=`Drama` × tags=`based on a book` — coef 0.002, support 113
- genres=`Comedy` × tags=`family` — coef 0.002, support 78
- genres=`Drama` × tags=`original` — coef 0.002, support 272
- tags=`action` × tags=`original` — coef 0.002, support 47
- genres=`Thriller` × tags=`suspense` — coef 0.002, support 125
- genres=`Comedy` × tags=`mentor` — coef 0.002, support 43
- genres=`Drama` × genres=`Thriller` — coef 0.002, support 93
- genres=`Crime` × genres=`Drama` — coef 0.002, support 74
- genres=`Action` × tags=`great` — coef 0.002, support 23
- genres=`Drama` × tags=`dramatic` — coef 0.002, support 185
- genres=`Children's` × genres=`Comedy` — coef 0.002, support 89
- genres=`Action` × genres=`Drama` — coef 0.002, support 85
- genres=`Action` × tags=`chase` — coef 0.002, support 97

Coefficient norms: unary median 0.000, pairs median 0.000.

Reading rule (pre-registered): HOT iff ΔR² > null p95 on the e_ID response. The permutation null keeps the pair columns' marginals but breaks their link to the item (and to its unary indicators), so it measures pure overfit inflation of equally sparse columns.
