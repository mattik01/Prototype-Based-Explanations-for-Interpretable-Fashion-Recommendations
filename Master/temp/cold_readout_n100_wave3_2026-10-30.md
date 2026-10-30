# 100-trial cold rows incl. wave 3 (dc09 fUfI arms + two-sided controls) — 2026-10-30

Cold evals of the 12 wave-3 retrains (8167433–8167452) on the LEO5 login node, fleet checkout at 71c8394
(= retrain hash; cold_eval code identical to 921d74a). Launchers `/scratch/c7031336/smoke_runs/cold_n100_wave3/gpu{0,1}.sh`
(loop over `run_cold_evals.sh <m> <ds> _n100`), collector `collect_cold_n100.py`. The two ml-1m `_sep` control retrains
(8181862 / 8181864, `sep_controls_cold_submissions_2026-10-30.md`) finished later the same day (COMPLETED, 18 / 34 min);
their rows come from a second collection, cold-evaluated with the scratch copy of the current runner
(`/scratch/c7031336/smoke_runs/cold_n100_wave3/run_cold_evals.sh`; the fleet checkout's 71c8394 copy predates the dataset/tier arguments).
Bars to read fUfI against: patched UI host (attr-kNN cold-vs-cold) 0.417 hm / 0.533 ml-1m; LightFM 0.50–0.52 hm / 0.62 ml-1m.

# 100-trial cold rows — collected

Single seed 38210573; single-config cold retrains of the 100-trial winners (S0.3 §6); 99 sampled negatives; mean-ID imputation for cold ID rows. Quoting: F-S0-14 / -15 / -17 (see the collector docstring).

## hm_1_month

| model | warm ref (n100 test) | warm on variant | native cold-vs-cold | native cold-vs-all | attr-kNN cold-vs-cold | attr-kNN cold-vs-all | cold-vs-cold N@10 (native / kNN) |
|---|---|---|---|---|---|---|---|
| mf | 0.4667 | 0.5020 | 0.0910 | 0.0092 | 0.2528 | 0.3671 | 0.0416 / 0.1324 |
| user_proto | 0.6218 | 0.6666 | 0.1029 | 0.0000 | 0.3524 | 0.2776 | 0.0454 / 0.1899 [0.1887, 0.1912] |
| item_proto | 0.5536 | 0.6129 | 0.0954 [0.0931, 0.1710] | 0.0133 [0.0104, 0.0356] | 0.2624 | 0.3395 | 0.0440 [0.0432, 0.0670] / 0.1305 |
| user_item_proto | 0.6713 | 0.6889 | chance | 0.0000 | 0.4167 | 0.3157 | chance / 0.2313 [0.2297, 0.2328] |
| feature_item_proto | 0.5585 | 0.5632 | 0.3189 | 0.2657 | — | — | 0.1588 [0.1577, 0.1600] / — |
| feature_item_proto_noid | 0.4294 | 0.4200 | 0.4360 | 0.4370 | — | — | 0.2393 [0.2377, 0.2410] / — |
| feature_user_proto | 0.6238 | 0.6702 | 0.0855 | 0.0000 | 0.3934 | 0.3059 | 0.0374 / 0.2168 [0.2154, 0.2182] |
| feature_user_proto_noid | 0.6126 | 0.6600 | 0.0937 | 0.0000 | 0.3712 | 0.3038 | 0.0429 / 0.2007 [0.1995, 0.2019] |
| lightfm_tags | 0.4947 | 0.4690 | 0.5030 | 0.5030 | — | — | 0.3028 [0.3002, 0.3054] / — |
| lightfm_tags_ids | 0.5308 | 0.5044 | 0.5195 | 0.5201 | — | — | 0.3130 [0.3102, 0.3158] / — |
| feature_user_item_proto | 0.7138 | 0.7012 | 0.5081 | 0.3989 | — | — | 0.2974 [0.2963, 0.2985] / — |
| feature_user_item_proto_noid | 0.5694 | 0.5610 | 0.5138 | 0.5108 | — | — | 0.3035 [0.3009, 0.3061] / — |
| feature_user_item_proto_sep | 0.6872 | 0.6874 | 0.5293 | 0.4247 | — | — | 0.3080 [0.3054, 0.3106] / — |
| feature_user_item_proto_noid_sep | 0.5664 | 0.5574 | 0.4936 | 0.4906 | — | — | 0.2839 [0.2813, 0.2865] / — |
| lightfm_tags_hist_ids | 0.6927 | 0.6957 | 0.5598 | 0.4726 | — | — | 0.3432 [0.3406, 0.3460] / — |
| lightfm_tags_hist | 0.5624 | 0.5443 | 0.5510 | 0.5556 | — | — | 0.3371 [0.3345, 0.3398] / — |
| popularity | | | 0.10 (chance by construction, F-S0-17) | 0.0000 | | | |

**Dev-tier hm cold rows (2026-07/08; reference only, other tier):**

| model | native cold-vs-cold | native cold-vs-all | attr-kNN cold-vs-cold | attr-kNN cold-vs-all | convention |
|---|---|---|---|---|---|
| mf | 0.0953 | 0.0001 | 0.2453 | 0.3147 | drop (pre-2026-10-26) |
| user_proto | 0.1081 | 0.0000 | 0.3000 | 0.2217 | drop (pre-2026-10-26) |
| item_proto | chance | 0.0091 [0.0000, 0.0987] | 0.2359 | 0.2737 | drop (pre-2026-10-26) |
| user_item_proto | chance | 0.0000 | 0.4156 | 0.3137 | drop (pre-2026-10-26) |
| feature_item_proto | 0.2876 | 0.2179 | — | — | drop (pre-2026-10-26) |
| feature_item_proto_noid | 0.3884 | 0.3937 | — | — | drop (pre-2026-10-26) |
| feature_user_proto | 0.0955 | 0.0000 | 0.3266 | 0.2829 | drop (pre-2026-10-26) |
| feature_user_proto_noid | 0.1133 | 0.0000 | 0.3419 | 0.2998 | drop (pre-2026-10-26) |
| lightfm_tags | 0.4706 | 0.4702 | — | — | drop (pre-2026-10-26) |
| lightfm_tags_ids | 0.5081 | 0.5082 | — | — | drop (pre-2026-10-26) |

## ml-1m

| model | warm ref (n100 test) | warm on variant | native cold-vs-cold | native cold-vs-all | attr-kNN cold-vs-cold | attr-kNN cold-vs-all | cold-vs-cold N@10 (native / kNN) |
|---|---|---|---|---|---|---|---|
| mf | 0.4897 | 0.5413 | 0.0999 | 0.0153 | 0.4355 | 0.3880 | 0.0457 / 0.2539 |
| user_proto | 0.6142 | 0.6714 | 0.1162 | 0.0001 | 0.5313 | 0.2905 | 0.0486 / 0.3201 |
| item_proto | 0.5723 | 0.6133 | chance | 0.0418 [0.0000, 0.2890] | 0.5145 | 0.3783 | chance / 0.3036 |
| user_item_proto | 0.6246 | 0.6864 | chance | 0.0000 | 0.5330 | 0.3030 | chance / 0.3192 |
| feature_item_proto | 0.5628 | 0.5546 | 0.4667 | 0.4972 | — | — | 0.2670 / — |
| feature_item_proto_noid | 0.5156 | 0.5168 | 0.4375 | 0.4576 | — | — | 0.2576 / — |
| feature_user_proto | 0.5613 | 0.5934 | 0.0932 | 0.0758 | 0.4843 | 0.3053 | 0.0409 / 0.2872 |
| feature_user_proto_noid | 0.6082 | 0.6609 | 0.1120 | 0.0000 | 0.5262 | 0.2873 | 0.0599 / 0.3171 |
| lightfm_tags | 0.5905 | 0.5998 | 0.6165 | 0.6148 | — | — | 0.3862 / — |
| lightfm_tags_ids | 0.6155 | 0.6220 | 0.6176 | 0.6138 | — | — | 0.3860 / — |
| feature_user_item_proto | 0.6308 | 0.6488 | 0.5692 | 0.5168 | — | — | 0.3436 / — |
| feature_user_item_proto_noid | 0.6172 | 0.6121 | 0.5345 | 0.5491 | — | — | 0.3123 / — |
| feature_user_item_proto_sep | 0.6465 | 0.6604 | 0.5761 | 0.5356 | — | — | 0.3492 / — |
| feature_user_item_proto_noid_sep | 0.6263 | 0.6272 | 0.5343 | 0.5556 | — | — | 0.3082 / — |
| lightfm_tags_hist_ids | 0.6530 | 0.6631 | 0.5888 | 0.5771 | — | — | 0.3562 / — |
| lightfm_tags_hist | 0.6097 | 0.6173 | 0.5679 | 0.5806 | — | — | 0.3427 / — |
| lightfm_tags_hist_ids_sep | 0.6485 | 0.6544 | 0.5923 | 0.5537 | — | — | 0.3585 / — |
| lightfm_tags_hist_sep | 0.6145 | 0.6117 | 0.5328 | 0.5630 | — | — | 0.3143 / — |
| popularity | | | 0.10 (chance by construction, F-S0-17) | 0.0000 | | | |

