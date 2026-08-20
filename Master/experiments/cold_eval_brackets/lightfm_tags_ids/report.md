# Cold eval — lightfm_tags_ids_hm_1_month_cold_s38210573

variant: `/scratch/c7031336/protomf_data/hm_1_month_cold` | n_neg=99 | seed=38210573 | use_bias=0

| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |
|---|---:|---:|---:|---|
| canonical warm reference | | 0.5087 | 0.3063 | warm Δhr@10 -0.0208 |
| warm test (variant) | 59048 | 0.4879 | 0.2948 |  |
| cold-vs-cold (PRIMARY) | 121024 | 0.5080 | 0.3012 | [0.4936, 0.5203] |
| cold-vs-all (secondary) | 121024 | 0.5082 | 0.3004 | [0.4940, 0.5200] |
| popularity cold_vs_cold | | 0.1202 | 0.0562 | |
| popularity cold_vs_all | | 0.0000 | 0.0000 | |

## F-S0-14 tie brackets @10 (worst / expected / best)

| block | HR@10 w/e/b | NDCG@10 w/e/b | straddle share | any-tie share | block size mean/max | point in bracket |
|---|---|---|---:|---:|---:|---|
| cold_vs_cold | 0.5074 / 0.5081 / 0.5089 | 0.2972 / 0.2998 / 0.3024 | 0.001 | 0.058 | 1.06 / 6 | yes |
| cold_vs_all | 0.5081 / 0.5082 / 0.5084 | 0.2995 / 0.3000 / 0.3006 | 0.000 | 0.012 | 1.01 / 3 | yes |
| popularity cold_vs_cold | 0.1202 / 0.1202 / 0.1202 | 0.0562 / 0.0562 / 0.0562 | 0.000 | 0.000 | 1.00 / 1 | NO |
| popularity cold_vs_all | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.000 | 0.000 | 1.00 / 1 | yes |

tie-block (top-10, 5000 users): mean distinct sigs 3.94, twin share 0.989, mean max class 5.89

## cold-vs-cold by original-popularity decile

| decile | rows | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0 | 1564 | 0.3402 | 0.1898 |
| 1 | 1864 | 0.3943 | 0.2203 |
| 2 | 2323 | 0.3805 | 0.2093 |
| 3 | 3097 | 0.3694 | 0.2030 |
| 4 | 3994 | 0.3911 | 0.2180 |
| 5 | 5461 | 0.4576 | 0.2654 |
| 6 | 7843 | 0.4605 | 0.2663 |
| 7 | 12021 | 0.4913 | 0.2872 |
| 8 | 20222 | 0.5032 | 0.2954 |
| 9 | 62635 | 0.5497 | 0.3321 |
