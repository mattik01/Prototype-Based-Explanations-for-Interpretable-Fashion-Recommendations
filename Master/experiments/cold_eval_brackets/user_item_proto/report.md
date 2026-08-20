# Cold eval — user_item_proto_hm_1_month_cold_s38210573

variant: `/scratch/c7031336/protomf_data/hm_1_month_cold` | n_neg=99 | seed=38210573 | use_bias=0

| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |
|---|---:|---:|---:|---|
| canonical warm reference | | 0.6587 | 0.4164 | warm Δhr@10 +0.0350 |
| warm test (variant) | 59048 | 0.6937 | 0.4429 |  |
| cold-vs-cold (PRIMARY) | 121024 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| cold-vs-all (secondary) | 121024 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| attr-kNN cold_vs_cold | | 0.4155 | 0.2283 | |
| attr-kNN cold_vs_all | | 0.3137 | 0.1330 | |
| popularity cold_vs_cold | | 0.1202 | 0.0562 | |
| popularity cold_vs_all | | 0.0000 | 0.0000 | |

## F-S0-14 tie brackets @10 (worst / expected / best)

| block | HR@10 w/e/b | NDCG@10 w/e/b | straddle share | any-tie share | block size mean/max | point in bracket |
|---|---|---|---:|---:|---:|---|
| cold_vs_cold | 0.0000 / 0.1000 / 1.0000 | 0.0000 / 0.0454 / 1.0000 | 1.000 | 1.000 | 100.00 / 100 | yes |
| cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 1.000 | 20.77 / 40 | yes |
| attr-kNN cold_vs_cold | 0.4148 / 0.4156 / 0.4163 | 0.2260 / 0.2276 / 0.2292 | 0.002 | 0.059 | 1.06 / 6 | yes |
| attr-kNN cold_vs_all | 0.3135 / 0.3137 / 0.3139 | 0.1329 / 0.1330 / 0.1332 | 0.000 | 0.012 | 1.01 / 3 | yes |
| popularity cold_vs_cold | 0.1202 / 0.1202 / 0.1202 | 0.0562 / 0.0562 / 0.0562 | 0.000 | 0.000 | 1.00 / 1 | NO |
| popularity cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | yes |

tie-block (top-10, 5000 users): mean distinct sigs 9.28, twin share 0.512, mean max class 1.60

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
