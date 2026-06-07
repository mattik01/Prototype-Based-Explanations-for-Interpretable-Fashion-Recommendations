# Experiment Summary: user_item_proto x ml-1m (seed=38210573)

- **Machine:** n052.intern.leo5.intra.uibk.ac.at
- **Completed:** 100/100 trials (configured: 100)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 6h10m25s
- **Run name:** user_item_proto_ml-1m_n0_38210573_2026-5-21_15-30-27.283036

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| ml-1m | user_item_proto | 5 | yes | 12,472 | 95.6 | 72.8 | 26,215 | 6h10m25s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| ml-1m | user_item_proto | single | 0.6543 | 0.3790 | **0.6246** | 0.3573 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| ml-1m | user_item_proto | 73 | 256 | 76(u)/11(i) | 1.798(u)/1.626(i) | 0.001765(u)/0.009407(i) | s_sm | adagrad | 0.09874 | 3.73e-04 | 39 |

## All Test Metrics

- **hit_ratio@1:** 0.1485
- **hit_ratio@10:** 0.6246
- **hit_ratio@3:** 0.3286
- **hit_ratio@5:** 0.4412
- **hit_ratio@50:** 0.9624
- **ndcg@1:** 0.1485
- **ndcg@10:** 0.3573
- **ndcg@3:** 0.2519
- **ndcg@5:** 0.2981
- **ndcg@50:** 0.4344
- **test_loss:** -3.3064

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1601
- **hit_ratio@10:** 0.6543
- **hit_ratio@3:** 0.3528
- **hit_ratio@5:** 0.4690
- **hit_ratio@50:** 0.9683
- **ndcg@1:** 0.1601
- **ndcg@10:** 0.3790
- **ndcg@3:** 0.2714
- **ndcg@5:** 0.3191
- **ndcg@50:** 0.4511
- **val_loss:** -3.4240
