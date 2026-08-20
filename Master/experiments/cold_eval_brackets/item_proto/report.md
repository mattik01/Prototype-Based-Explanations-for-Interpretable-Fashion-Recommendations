# Cold eval — item_proto_hm_1_month_cold_s38210573

variant: `/scratch/c7031336/protomf_data/hm_1_month_cold` | n_neg=99 | seed=38210573 | use_bias=0

| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |
|---|---:|---:|---:|---|
| canonical warm reference | | 0.4938 | 0.2631 | warm Δhr@10 +0.0440 |
| warm test (variant) | 59048 | 0.5378 | 0.2966 |  |
| cold-vs-cold (PRIMARY) | 121024 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| cold-vs-all (secondary) | 121024 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| attr-kNN cold_vs_cold | | 0.2358 | 0.1183 | |
| attr-kNN cold_vs_all | | 0.2737 | 0.1049 | |
| popularity cold_vs_cold | | 0.1202 | 0.0562 | |
| popularity cold_vs_all | | 0.0000 | 0.0000 | |

## F-S0-14 tie brackets @10 (worst / expected / best)

| block | HR@10 w/e/b | NDCG@10 w/e/b | straddle share | any-tie share | block size mean/max | point in bracket |
|---|---|---|---:|---:|---:|---|
| cold_vs_cold | 0.0000 / 0.1000 / 1.0000 | 0.0000 / 0.0454 / 1.0000 | 1.000 | 1.000 | 100.00 / 100 | yes |
| cold_vs_all | 0.0000 / 0.0091 / 0.0987 | 0.0000 / 0.0027 / 0.0299 | 0.099 | 1.000 | 20.77 / 40 | yes |
| attr-kNN cold_vs_cold | 0.2352 / 0.2359 / 0.2366 | 0.1174 / 0.1181 / 0.1188 | 0.001 | 0.059 | 1.07 / 6 | yes |
| attr-kNN cold_vs_all | 0.2735 / 0.2737 / 0.2739 | 0.1047 / 0.1049 / 0.1050 | 0.000 | 0.012 | 1.01 / 3 | yes |
| popularity cold_vs_cold | 0.1202 / 0.1202 / 0.1202 | 0.0562 / 0.0562 / 0.0562 | 0.000 | 0.000 | 1.00 / 1 | NO |
| popularity cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | yes |

tie-block (top-10, 5000 users): mean distinct sigs 10.00, twin share 0.000, mean max class 1.00

## cold-vs-cold by original-popularity decile

| decile | rows | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0 | 1564 | 0.0000 | 0.0000 |
| 1 | 1864 | 0.0000 | 0.0000 |
| 2 | 2323 | 0.0000 | 0.0000 |
| 3 | 3097 | 0.0000 | 0.0000 |
| 4 | 3994 | 0.0000 | 0.0000 |
| 5 | 5461 | 0.0000 | 0.0000 |
| 6 | 7843 | 0.0000 | 0.0000 |
| 7 | 12021 | 0.0000 | 0.0000 |
| 8 | 20222 | 0.0000 | 0.0000 |
| 9 | 62635 | 0.0000 | 0.0000 |
