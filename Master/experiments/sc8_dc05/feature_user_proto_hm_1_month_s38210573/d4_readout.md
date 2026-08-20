# Stratified readout — feature_user_proto_hm_1_month_s38210573

dataset: hm_1_month | rows: 73418 | overall HR@10 0.5824 / NDCG@10 0.3492

## By user train-history quartile (D4, facet B warm-thin)

| quartile | history range | rows | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Q1 | [3, 3] | 18442 | 0.5819 | 0.3529 |
| Q2 | [4, 5] | 23124 | 0.5922 | 0.3565 |
| Q3 | [6, 8] | 16658 | 0.5778 | 0.3463 |
| Q4 | [9, 89] | 15194 | 0.5733 | 0.3368 |

## By positive-item train popularity (stratum TW)

| bucket | rows | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| 0 (train-unseen, F-S0-04) | 9 | 0.0000 | 0.0000 |
| 1-5 | 2301 | 0.0000 | 0.0000 |
| 6-20 | 10072 | 0.0052 | 0.0016 |
| 21-100 | 27160 | 0.4458 | 0.1890 |
| 101-inf | 33876 | 0.9033 | 0.6049 |
