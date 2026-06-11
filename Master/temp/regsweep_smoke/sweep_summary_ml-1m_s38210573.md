# Regularizer sensitivity sweep: user_item_proto x ml-1m (seed=38210573, profile=smoke)

All hyperparameters fixed at the hyperopt-selected baseline; only the four
prototype regularizer weights are jointly scaled by `reg_multiplier`.

| Variant | Multiplier | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 | Wall | Status |
|---------|:----------:|:---------:|:-----------:|:----------:|:------------:|:----:|:------:|
| off | 0 | 0.5956 | 0.3395 | 0.5746 | 0.3250 | 10m | OK |
| x100 | 100 | 0.4592 | 0.2541 | 0.4402 | 0.2425 | 10m | OK |
