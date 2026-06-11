# Experiment Summary: user_proto x hm_1_month (seed=38210573)

- **Machine:** n047.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 3h35m08s
- **Run name:** user_proto_hm_1_month_n0_38210573_2026-6-8_15-41-40.742945

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_1_month | user_proto | 5 | yes | 2,124 | 20.7 | 62.1 | 22,308 | 3h35m08s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_1_month | user_proto | single | 0.5824 | 0.3408 | **0.5725** | 0.3322 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_1_month | user_proto | 81 | 256 | 76 (u) | 1.798 | 0.001765 | s_sm | adagrad | 0.09874 | 3.73e-04 | 39 |

## All Test Metrics

- **hit_ratio@1:** 0.1438
- **hit_ratio@10:** 0.5725
- **hit_ratio@3:** 0.3062
- **hit_ratio@5:** 0.4104
- **hit_ratio@50:** 0.9068
- **ndcg@1:** 0.1438
- **ndcg@10:** 0.3322
- **ndcg@3:** 0.2371
- **ndcg@5:** 0.2799
- **ndcg@50:** 0.4077
- **test_loss:** 0.0946

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1492
- **hit_ratio@10:** 0.5824
- **hit_ratio@3:** 0.3177
- **hit_ratio@5:** 0.4229
- **hit_ratio@50:** 0.9103
- **ndcg@1:** 0.1492
- **ndcg@10:** 0.3408
- **ndcg@3:** 0.2460
- **ndcg@5:** 0.2892
- **ndcg@50:** 0.4150
- **val_loss:** 0.0532
