# Stratified readout — user_proto_hm_1_month_s38210573

dataset: hm_1_month | rows: 73418 | overall HR@10 0.5729 / NDCG@10 0.3319

## By user train-history quartile (D4, facet B warm-thin)

| quartile | history range | rows | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Q1 | [3, 3] | 18442 | 0.5718 | 0.3350 |
| Q2 | [4, 5] | 23124 | 0.5798 | 0.3378 |
| Q3 | [6, 8] | 16658 | 0.5710 | 0.3305 |
| Q4 | [9, 89] | 15194 | 0.5657 | 0.3205 |

## By positive-item train popularity (stratum TW)

| bucket | rows | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| 0 (train-unseen, F-S0-04) | 9 | 0.0000 | 0.0000 |
| 1-5 | 2301 | 0.0000 | 0.0000 |
| 6-20 | 10072 | 0.0039 | 0.0012 |
| 21-100 | 27160 | 0.4198 | 0.1700 |
| 101-inf | 33876 | 0.9038 | 0.5826 |
