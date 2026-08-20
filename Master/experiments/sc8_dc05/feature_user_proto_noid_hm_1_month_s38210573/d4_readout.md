# Stratified readout — feature_user_proto_noid_hm_1_month_s38210573

dataset: hm_1_month | rows: 73418 | overall HR@10 0.5852 / NDCG@10 0.3509

## By user train-history quartile (D4, facet B warm-thin)

| quartile | history range | rows | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Q1 | [3, 3] | 18442 | 0.5846 | 0.3562 |
| Q2 | [4, 5] | 23124 | 0.5941 | 0.3584 |
| Q3 | [6, 8] | 16658 | 0.5822 | 0.3483 |
| Q4 | [9, 89] | 15194 | 0.5759 | 0.3360 |

## By positive-item train popularity (stratum TW)

| bucket | rows | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| 0 (train-unseen, F-S0-04) | 9 | 0.0000 | 0.0000 |
| 1-5 | 2301 | 0.0000 | 0.0000 |
| 6-20 | 10072 | 0.0103 | 0.0033 |
| 21-100 | 27160 | 0.4696 | 0.2035 |
| 101-inf | 33876 | 0.8888 | 0.5964 |
