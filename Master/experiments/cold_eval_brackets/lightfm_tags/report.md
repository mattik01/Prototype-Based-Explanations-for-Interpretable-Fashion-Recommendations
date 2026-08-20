# Cold eval — lightfm_tags_hm_1_month_cold_s38210573

variant: `/scratch/c7031336/protomf_data/hm_1_month_cold` | n_neg=99 | seed=38210573 | use_bias=0

| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |
|---|---:|---:|---:|---|
| canonical warm reference | | 0.4582 | 0.2709 | warm Δhr@10 -0.0191 |
| warm test (variant) | 59048 | 0.4391 | 0.2610 |  |
| cold-vs-cold (PRIMARY) | 121024 | 0.4705 | 0.2781 | [0.4594, 0.4816] |
| cold-vs-all (secondary) | 121024 | 0.4700 | 0.2782 | [0.4585, 0.4813] |
| popularity cold_vs_cold | | 0.1202 | 0.0562 | |
| popularity cold_vs_all | | 0.0000 | 0.0000 | |

## F-S0-14 tie brackets @10 (worst / expected / best)

| block | HR@10 w/e/b | NDCG@10 w/e/b | straddle share | any-tie share | block size mean/max | point in bracket |
|---|---|---|---:|---:|---:|---|
| cold_vs_cold | 0.4700 / 0.4706 / 0.4711 | 0.2742 / 0.2766 / 0.2791 | 0.001 | 0.058 | 1.06 / 6 | yes |
| cold_vs_all | 0.4695 / 0.4702 / 0.4708 | 0.2745 / 0.2769 / 0.2794 | 0.001 | 0.056 | 1.06 / 7 | yes |
| popularity cold_vs_cold | 0.1202 / 0.1202 / 0.1202 | 0.0562 / 0.0562 / 0.0562 | 0.000 | 0.000 | 1.00 / 1 | NO |
| popularity cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | yes |

tie-block (top-10, 5000 users): mean distinct sigs 3.30, twin share 0.992, mean max class 6.56

## cold-vs-cold by original-popularity decile

| decile | rows | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0 | 1564 | 0.3523 | 0.1945 |
| 1 | 1864 | 0.3863 | 0.2182 |
| 2 | 2323 | 0.3810 | 0.2073 |
| 3 | 3097 | 0.3781 | 0.2057 |
| 4 | 3994 | 0.3971 | 0.2228 |
| 5 | 5461 | 0.4609 | 0.2688 |
| 6 | 7843 | 0.4534 | 0.2661 |
| 7 | 12021 | 0.4855 | 0.2849 |
| 8 | 20222 | 0.4809 | 0.2851 |
| 9 | 62635 | 0.4853 | 0.2905 |
