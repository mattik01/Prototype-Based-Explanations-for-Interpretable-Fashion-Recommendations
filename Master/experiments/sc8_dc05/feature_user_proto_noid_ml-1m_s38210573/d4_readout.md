# Stratified readout — feature_user_proto_noid_ml-1m_s38210573

dataset: ml-1m | rows: 6034 | overall HR@10 0.5317 / NDCG@10 0.2998

## By user train-history quartile (D4, facet B warm-thin)

| quartile | history range | rows | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Q1 | [3, 25] | 1518 | 0.6133 | 0.3641 |
| Q2 | [26, 56] | 1513 | 0.5790 | 0.3266 |
| Q3 | [57, 121] | 1495 | 0.5211 | 0.2852 |
| Q4 | [122, 1415] | 1508 | 0.4125 | 0.2227 |

## By positive-item train popularity (stratum TW)

| bucket | rows | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| 1-5 | 17 | 0.0000 | 0.0000 |
| 6-20 | 192 | 0.0000 | 0.0000 |
| 21-100 | 828 | 0.0109 | 0.0034 |
| 101-inf | 4997 | 0.6402 | 0.3615 |
