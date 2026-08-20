# Cold eval — feature_item_proto_hm_1_month_cold_s38210573

variant: `/scratch/c7031336/protomf_data/hm_1_month_cold` | n_neg=99 | seed=38210573 | use_bias=0

| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |
|---|---:|---:|---:|---|
| canonical warm reference | | 0.4984 | 0.2757 | warm Δhr@10 +0.0165 |
| warm test (variant) | 59048 | 0.5149 | 0.2933 |  |
| cold-vs-cold (PRIMARY) | 121024 | 0.2876 | 0.1434 | [0.2519, 0.3226] |
| cold-vs-all (secondary) | 121024 | 0.2179 | 0.0853 | [0.1856, 0.2498] |
| popularity cold_vs_cold | | 0.1202 | 0.0562 | |
| popularity cold_vs_all | | 0.0000 | 0.0000 | |

## F-S0-14 tie brackets @10 (worst / expected / best)

| block | HR@10 w/e/b | NDCG@10 w/e/b | straddle share | any-tie share | block size mean/max | point in bracket |
|---|---|---|---:|---:|---:|---|
| cold_vs_cold | 0.2869 / 0.2876 / 0.2882 | 0.1420 / 0.1430 / 0.1440 | 0.001 | 0.058 | 1.06 / 6 | yes |
| cold_vs_all | 0.2178 / 0.2179 / 0.2181 | 0.0852 / 0.0853 / 0.0854 | 0.000 | 0.012 | 1.01 / 3 | yes |
| popularity cold_vs_cold | 0.1202 / 0.1202 / 0.1202 | 0.0562 / 0.0562 / 0.0562 | 0.000 | 0.000 | 1.00 / 1 | NO |
| popularity cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | yes |

tie-block (top-10, 5000 users): mean distinct sigs 9.67, twin share 0.129, mean max class 1.24

## cold-vs-cold by original-popularity decile

| decile | rows | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0 | 1564 | 0.1151 | 0.0546 |
| 1 | 1864 | 0.1202 | 0.0578 |
| 2 | 2323 | 0.1214 | 0.0525 |
| 3 | 3097 | 0.1108 | 0.0503 |
| 4 | 3994 | 0.1067 | 0.0531 |
| 5 | 5461 | 0.1364 | 0.0612 |
| 6 | 7843 | 0.1464 | 0.0635 |
| 7 | 12021 | 0.1602 | 0.0714 |
| 8 | 20222 | 0.1961 | 0.0916 |
| 9 | 62635 | 0.4081 | 0.2097 |
