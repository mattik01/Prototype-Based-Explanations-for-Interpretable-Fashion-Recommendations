# Experiment Summary: item_proto x hm_1_month (seed=38210573)

- **Machine:** n034.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 2h54m19s
- **Run name:** item_proto_hm_1_month_n0_38210573_2026-6-8_15-29-29.161883

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_1_month | item_proto | 5 | yes | 11,100 | 30.7 | 68.4 | 21,700 | 2h54m19s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_1_month | item_proto | single | 0.5015 | 0.2653 | **0.4937** | 0.2630 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_1_month | item_proto | 81 | 256 | 76 (i) | 1.798 | 0.001765 | s_sm | adagrad | 0.09874 | 3.73e-04 | 39 |

## All Test Metrics

- **hit_ratio@1:** 0.0871
- **hit_ratio@10:** 0.4937
- **hit_ratio@3:** 0.2299
- **hit_ratio@5:** 0.3344
- **hit_ratio@50:** 0.8420
- **ndcg@1:** 0.0871
- **ndcg@10:** 0.2630
- **ndcg@3:** 0.1686
- **ndcg@5:** 0.2115
- **ndcg@50:** 0.3415
- **test_loss:** 0.3831

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.0858
- **hit_ratio@10:** 0.5015
- **hit_ratio@3:** 0.2298
- **hit_ratio@5:** 0.3381
- **hit_ratio@50:** 0.8502
- **ndcg@1:** 0.0858
- **ndcg@10:** 0.2653
- **ndcg@3:** 0.1680
- **ndcg@5:** 0.2124
- **ndcg@50:** 0.3439
- **val_loss:** 0.3628
