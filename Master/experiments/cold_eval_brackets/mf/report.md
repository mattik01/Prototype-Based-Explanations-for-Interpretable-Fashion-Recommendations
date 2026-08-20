# Cold eval — mf_hm_1_month_cold_s38210573

variant: `/scratch/c7031336/protomf_data/hm_1_month_cold` | n_neg=99 | seed=38210573 | use_bias=0

| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |
|---|---:|---:|---:|---|
| canonical warm reference | | 0.3652 | 0.2233 | warm Δhr@10 +0.0508 |
| warm test (variant) | 59048 | 0.4160 | 0.2667 |  |
| cold-vs-cold (PRIMARY) | 121024 | 0.0010 | 0.0010 | [0.0003, 0.0023] |
| cold-vs-all (secondary) | 121024 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| attr-kNN cold_vs_cold | | 0.2412 | 0.1304 | |
| attr-kNN cold_vs_all | | 0.3098 | 0.1696 | |
| popularity cold_vs_cold | | 0.1202 | 0.0562 | |
| popularity cold_vs_all | | 0.0000 | 0.0000 | |

## F-S0-14 tie brackets @10 (worst / expected / best)

| block | HR@10 w/e/b | NDCG@10 w/e/b | straddle share | any-tie share | block size mean/max | point in bracket |
|---|---|---|---:|---:|---:|---|
| cold_vs_cold | 0.0953 / 0.0953 / 0.0953 | 0.0424 / 0.0424 / 0.0424 | 0.000 | 0.000 | 1.00 / 1 | NO |
| cold_vs_all | 0.0001 / 0.0001 / 0.0001 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | NO |
| attr-kNN cold_vs_cold | 0.2449 / 0.2453 / 0.2458 | 0.1302 / 0.1311 / 0.1319 | 0.001 | 0.058 | 1.06 / 6 | NO |
| attr-kNN cold_vs_all | 0.3146 / 0.3147 / 0.3148 | 0.1708 / 0.1710 / 0.1712 | 0.000 | 0.012 | 1.01 / 3 | NO |
| popularity cold_vs_cold | 0.1202 / 0.1202 / 0.1202 | 0.0562 / 0.0562 / 0.0562 | 0.000 | 0.000 | 1.00 / 1 | NO |
| popularity cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | yes |

tie-block (top-10, 5000 users): mean distinct sigs 9.54, twin share 0.365, mean max class 1.39

## cold-vs-cold by original-popularity decile

| decile | rows | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0 | 1564 | 0.0006 | 0.0006 |
| 1 | 1864 | 0.0011 | 0.0011 |
| 2 | 2323 | 0.0000 | 0.0000 |
| 3 | 3097 | 0.0010 | 0.0010 |
| 4 | 3994 | 0.0005 | 0.0004 |
| 5 | 5461 | 0.0011 | 0.0011 |
| 6 | 7843 | 0.0015 | 0.0015 |
| 7 | 12021 | 0.0013 | 0.0013 |
| 8 | 20222 | 0.0005 | 0.0005 |
| 9 | 62635 | 0.0011 | 0.0011 |
