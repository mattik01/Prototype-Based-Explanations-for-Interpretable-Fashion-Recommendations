# Cold eval — acf_hm_1_month_cold_s38210573

variant: `/scratch/c7031336/protomf_data/hm_1_month_cold` | n_neg=99 | seed=38210573 | use_bias=0

| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |
|---|---:|---:|---:|---|
| canonical warm reference | | 0.5764 | 0.3307 | warm Δhr@10 +0.0543 |
| warm test (variant) | 59048 | 0.6307 | 0.3653 |  |
| cold-vs-cold (PRIMARY) | 121024 | 0.0718 | 0.0334 | [0.0604, 0.0851] |
| cold-vs-all (secondary) | 121024 | 0.0040 | 0.0012 | [0.0032, 0.0050] |
| attr-kNN cold_vs_cold | | 0.3094 | 0.1644 | |
| attr-kNN cold_vs_all | | 0.2334 | 0.0974 | |
| popularity cold_vs_cold | | 0.1202 | 0.0562 | |
| popularity cold_vs_all | | 0.0000 | 0.0000 | |

## F-S0-14 tie brackets @10 (worst / expected / best)

| block | HR@10 w/e/b | NDCG@10 w/e/b | straddle share | any-tie share | block size mean/max | point in bracket |
|---|---|---|---:|---:|---:|---|
| cold_vs_cold | 0.0815 / 0.0864 / 0.1410 | 0.0364 / 0.0380 / 0.0547 | 0.059 | 0.827 | 38.07 / 83 | NO |
| cold_vs_all | 0.0047 / 0.0063 / 0.0158 | 0.0014 / 0.0019 / 0.0048 | 0.011 | 0.730 | 8.39 / 29 | NO |
| attr-kNN cold_vs_cold | 0.3090 / 0.3096 / 0.3103 | 0.1629 / 0.1640 / 0.1650 | 0.001 | 0.059 | 1.06 / 6 | yes |
| attr-kNN cold_vs_all | 0.2333 / 0.2334 / 0.2336 | 0.0973 / 0.0974 / 0.0975 | 0.000 | 0.012 | 1.01 / 3 | yes |
| popularity cold_vs_cold | 0.1202 / 0.1202 / 0.1202 | 0.0562 / 0.0562 / 0.0562 | 0.000 | 0.000 | 1.00 / 1 | NO |
| popularity cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | yes |

tie-block (top-10, 5000 users): mean distinct sigs 9.44, twin share 0.539, mean max class 1.54

## cold-vs-cold by original-popularity decile

| decile | rows | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0 | 1564 | 0.0767 | 0.0369 |
| 1 | 1864 | 0.0960 | 0.0460 |
| 2 | 2323 | 0.0895 | 0.0410 |
| 3 | 3097 | 0.0888 | 0.0426 |
| 4 | 3994 | 0.0904 | 0.0525 |
| 5 | 5461 | 0.0894 | 0.0467 |
| 6 | 7843 | 0.0524 | 0.0232 |
| 7 | 12021 | 0.1217 | 0.0565 |
| 8 | 20222 | 0.0838 | 0.0378 |
| 9 | 62635 | 0.0557 | 0.0253 |
