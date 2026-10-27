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
| popularity | | | 0.10 (chance by construction, F-S0-17) | 0.0000 | | | |


## Generation and reading notes (2026-10-27)

**Generator.** `Master/temp/run_cold_evals.sh <model> <dataset> _n100` per row (LEO5 login node, fleet checkout `71c8394`, mean-ID imputation; GPU 0 hm, GPU 1 ml-1m), collected by `Master/temp/collect_cold_n100.py`. The retrains are SC.6 wave group C (job IDs in `Master/temp/sc6_wave_submissions_2026-10-27.md`), all 20 completed OK.

**Per-item cluster-bootstrap 95% intervals, native cold-vs-cold HR@10** (the runner reports them for the native rows only; there are none for the attr-kNN rows):

| | fI (ids) | fI-noid | lightfm_tags | lightfm_tags_ids |
|---|---|---|---|---|
| hm (2,730 cold items, 121,024 rows) | 0.319 [0.286, 0.352] | 0.436 [0.425, 0.447] | 0.503 [0.492, 0.513] | 0.519 [0.505, 0.532] |
| ml-1m (625 cold items, 112,979 rows) | 0.467 [0.428, 0.503] | 0.438 [0.371, 0.497] | 0.617 [0.576, 0.656] | 0.618 [0.578, 0.655] |

**Do not misread:**
- **Native rows of the CF models (mf, user_proto, fU, fU-noid ≈ 0.09–0.12) are chance.** A cold item's row there is an untrained random vector, so its scores do not tie and the bracket looks tight. The value is still a chance draw (cf. F-S0-17's 0.084–0.15 spread for the same per-item randomness).
- **fU ml-1m native cold-vs-all 0.076 is a BCE calibration artefact.** It is the only BCE winner: warm items are pushed below zero, and untrained cold rows near zero outrank some of them.
- **The attr-kNN rows' cold-vs-all can exceed their cold-vs-cold** (mf hm 0.367 vs 0.253): patched cold items inherit warm neighbours' popularity mass. Cold-vs-all is calibration-confounded by design; it is not a quality reading.
- **ml-1m cold rows are optimistic.** Tag-Genome relevance is learned from user tagging and rating (learnings [dc09 SC.1a], F-DC09-05), so a "cold" movie's tags already carry collaborative signal. This lifts every feature-based route there (the native feature rows and the attr-kNN patch alike). hm's static taxonomy is the strict cold test.
- **The dev-tier hm rows are another tier** (30 trials, drop convention — equal to imputation within ≤ 0.001 per F-DC09-08). Shown for stability only, never as comparison rows.
