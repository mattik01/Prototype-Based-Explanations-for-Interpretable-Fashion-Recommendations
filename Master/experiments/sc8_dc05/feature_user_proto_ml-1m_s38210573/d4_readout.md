# Stratified readout — feature_user_proto_ml-1m_s38210573

dataset: ml-1m | rows: 6034 | overall HR@10 0.5436 / NDCG@10 0.3099

## By user train-history quartile (D4, facet B warm-thin)

| quartile | history range | rows | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Q1 | [3, 25] | 1518 | 0.6212 | 0.3786 |
| Q2 | [26, 56] | 1513 | 0.5863 | 0.3367 |
| Q3 | [57, 121] | 1495 | 0.5418 | 0.3001 |
| Q4 | [122, 1415] | 1508 | 0.4244 | 0.2236 |

## By positive-item train popularity (stratum TW)

| bucket | rows | HR@10 | NDCG@10 |
|---|---:|---:|---:|
| 1-5 | 17 | 0.0000 | 0.0000 |
| 6-20 | 192 | 0.0000 | 0.0000 |
| 21-100 | 828 | 0.0060 | 0.0018 |
| 101-inf | 4997 | 0.6554 | 0.3739 |
