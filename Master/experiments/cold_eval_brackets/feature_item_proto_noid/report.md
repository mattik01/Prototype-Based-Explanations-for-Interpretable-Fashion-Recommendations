# Cold eval — feature_item_proto_noid_hm_1_month_cold_s38210573

variant: `/scratch/c7031336/protomf_data/hm_1_month_cold` | n_neg=99 | seed=38210573 | use_bias=0

| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |
|---|---:|---:|---:|---|
| canonical warm reference | | 0.4229 | 0.2317 | warm Δhr@10 -0.0005 |
| warm test (variant) | 59048 | 0.4224 | 0.2320 |  |
| cold-vs-cold (PRIMARY) | 121024 | 0.3882 | 0.2057 | [0.3635, 0.4113] |
| cold-vs-all (secondary) | 121024 | 0.3935 | 0.2069 | [0.3691, 0.4164] |
| popularity cold_vs_cold | | 0.1202 | 0.0562 | |
| popularity cold_vs_all | | 0.0000 | 0.0000 | |

## F-S0-14 tie brackets @10 (worst / expected / best)

| block | HR@10 w/e/b | NDCG@10 w/e/b | straddle share | any-tie share | block size mean/max | point in bracket |
|---|---|---|---:|---:|---:|---|
| cold_vs_cold | 0.3875 / 0.3884 / 0.3892 | 0.2032 / 0.2049 / 0.2067 | 0.002 | 0.058 | 1.06 / 6 | yes |
| cold_vs_all | 0.3930 / 0.3937 / 0.3944 | 0.2047 / 0.2062 / 0.2078 | 0.001 | 0.056 | 1.06 / 7 | yes |
| popularity cold_vs_cold | 0.1202 / 0.1202 / 0.1202 | 0.0562 / 0.0562 / 0.0562 | 0.000 | 0.000 | 1.00 / 1 | NO |
| popularity cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | yes |

tie-block (top-10, 5000 users): mean distinct sigs 4.75, twin share 0.989, mean max class 5.03

## cold-vs-cold by original-popularity decile

| decile | rows | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0 | 1564 | 0.2334 | 0.1158 |
| 1 | 1864 | 0.2259 | 0.1135 |
| 2 | 2323 | 0.2462 | 0.1205 |
| 3 | 3097 | 0.2247 | 0.1080 |
| 4 | 3994 | 0.2356 | 0.1142 |
| 5 | 5461 | 0.2970 | 0.1442 |
| 6 | 7843 | 0.3001 | 0.1528 |
| 7 | 12021 | 0.3220 | 0.1599 |
| 8 | 20222 | 0.3606 | 0.1820 |
| 9 | 62635 | 0.4605 | 0.2530 |
