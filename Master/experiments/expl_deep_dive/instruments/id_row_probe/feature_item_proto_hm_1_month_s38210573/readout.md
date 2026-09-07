# ID-row probe — `feature_item_proto_hm_1_month_s38210573`

*working evidence (single checkpoint, post-hoc) — not thesis numbers.* dataset `hm_1_month`, layout `fixed`, fields ['department_name', 'product_type_name', 'section_name', 'colour_group_name', 'graphical_appearance_name'], n_items 13651, V 426, d 81, K 76, has_id True; sections ['A', 'B']; wall 53.7s.

## Anchors (cross-check vs composition_analysis_2026-08-24)

- eff_rank_raw: 1.2178
- corr_bagsize_qmeta_norm: nan
- id_meta_norm_ratio_median: 0.4604
- id_share_qsq_mean: 0.3096

## B — popularity-stratified ID share

Spearman with train popularity: id_share_qsq 0.493, id_meta_norm_ratio 0.501, id_share_activation -0.448, id_norm 0.528, qmeta_norm -0.360

| band | n | id_share_qsq mean | median | norm ratio median | act-level ID share mean | mean activation |
|---|---:|---:|---:|---:|---:|---:|
| overall | 13651 | 0.310 | 0.215 | 0.460 | -0.144 | 1.334 |
| unseen | 3 | 0.218 | 0.281 | 0.465 | 0.224 | 1.520 |
| 1-10 | 5768 | 0.190 | 0.078 | 0.214 | 0.234 | 1.506 |
| 11-100 | 6818 | 0.329 | 0.304 | 0.597 | -0.305 | 1.304 |
| 101+ | 1062 | 0.840 | 0.820 | 1.905 | -1.166 | 0.599 |
| d0 (pop 1-4) | 1330 | 0.179 | 0.051 | 0.186 | 0.221 | 1.512 |
| d1 (pop 5-5) | 1072 | 0.184 | 0.066 | 0.204 | 0.229 | 1.509 |
| d2 (pop 6-7) | 1655 | 0.182 | 0.078 | 0.216 | 0.222 | 1.507 |
| d3 (pop 8-9) | 1195 | 0.203 | 0.099 | 0.242 | 0.251 | 1.501 |
| d4 (pop 10-12) | 1405 | 0.230 | 0.121 | 0.265 | 0.287 | 1.486 |
| d5 (pop 13-17) | 1387 | 0.243 | 0.165 | 0.319 | 0.303 | 1.468 |
| d6 (pop 18-26) | 1452 | 0.266 | 0.227 | 0.439 | 0.282 | 1.403 |
| d7 (pop 27-41) | 1416 | 0.325 | 0.327 | 0.642 | -0.121 | 1.273 |
| d8 (pop 42-79) | 1352 | 0.472 | 0.475 | 0.951 | -1.755 | 1.055 |
| d9 (pop 80-1303) | 1384 | 0.797 | 0.769 | 1.748 | -1.346 | 0.655 |

D2: ‖mean e_ID‖ / mean ‖e_ID‖ = **0.007**; cos(mean e_ID, mean q_meta) = -0.708; eff. rank of ID block 9.24 (uncentred 9.24); ‖e_ID‖ median 0.061.

## A — pair-encoding regression (U4 trigger i)

Design: 426 unary cols; pairs 8745 unique / 3635 kept (support ≥ 5, cross-field only True); 5-fold CV over 13651 items; null = 20 X2-row permutations.

| response | λ | R² unary | R² +pairs | ΔR² | null ΔR² mean | null p95 | perm p | HOT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| e_ID | 1000 | -0.0040 | -0.0034 | +0.0006 | -0.0012 | -0.0003 | 0.048 | **YES** |
| c_ID | 100 | 0.0082 | 0.0203 | +0.0121 | -0.0166 | -0.0127 | 0.048 | **YES** |

Top pairs by coefficient norm (e_ID response, full-data fit):

- section_name=`Womens Everyday Collection` × colour_group_name=`Beige` — coef 0.003, support 111
- product_type_name=`Dress` × graphical_appearance_name=`All over pattern` — coef 0.003, support 399
- product_type_name=`Sweater` × graphical_appearance_name=`Melange` — coef 0.003, support 236
- product_type_name=`Cardigan` × section_name=`Womens Everyday Collection` — coef 0.003, support 28
- product_type_name=`Sweater` × graphical_appearance_name=`Solid` — coef 0.003, support 573
- department_name=`Knitwear` × section_name=`Womens Tailoring` — coef 0.002, support 96
- product_type_name=`Dress` × section_name=`Womens Tailoring` — coef 0.002, support 235
- section_name=`Divided Collection` × colour_group_name=`Black` — coef 0.002, support 477
- department_name=`Knitwear` × product_type_name=`Sweater` — coef 0.002, support 369
- department_name=`Blouse` × section_name=`Womens Everyday Collection` — coef 0.002, support 319
- department_name=`Blouse` × section_name=`Womens Tailoring` — coef 0.002, support 170
- product_type_name=`Sweater` × section_name=`Womens Tailoring` — coef 0.002, support 54
- product_type_name=`Dress` × colour_group_name=`Black` — coef 0.002, support 540
- department_name=`Knitwear` × section_name=`Womens Everyday Collection` — coef 0.002, support 180
- department_name=`Jersey Basic` × product_type_name=`Dress` — coef 0.002, support 57
- product_type_name=`Dress` × section_name=`Womens Everyday Basics` — coef 0.002, support 57
- product_type_name=`Skirt` × colour_group_name=`Black` — coef 0.002, support 162
- department_name=`Jersey` × product_type_name=`Top` — coef 0.002, support 218
- department_name=`Knitwear` × section_name=`Womens Casual` — coef 0.002, support 54
- product_type_name=`Sweater` × section_name=`Contemporary Smart` — coef 0.002, support 61

Coefficient norms: unary median 0.000, pairs median 0.000.

Reading rule (pre-registered): HOT iff ΔR² > null p95 on the e_ID response. The permutation null keeps the pair columns' marginals but breaks their link to the item (and to its unary indicators), so it measures pure overfit inflation of equally sparse columns.
