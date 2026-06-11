# Experiment Summary: mf x hm_1_month (seed=38210573)

- **Machine:** n049.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 3h05m32s
- **Run name:** mf_hm_1_month_n0_38210573_2026-6-8_15-6-43.107252

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_1_month | mf | 5 | yes | 2,110 | 11.2 | 55.2 | 21,931 | 3h05m32s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_1_month | mf | single | 0.3149 | 0.2174 | **0.3652** | 0.2233 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_1_month | mf | 44 | 512 |  |  |  | s_sm | adagrad | 0.05613 | 1.02e-04 | 15 |

## All Test Metrics

- **hit_ratio@1:** 0.1047
- **hit_ratio@10:** 0.3652
- **hit_ratio@3:** 0.2148
- **hit_ratio@5:** 0.2790
- **hit_ratio@50:** 0.5657
- **ndcg@1:** 0.1047
- **ndcg@10:** 0.2233
- **ndcg@3:** 0.1684
- **ndcg@5:** 0.1950
- **ndcg@50:** 0.2687
- **test_loss:** 4.6051

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1153
- **hit_ratio@10:** 0.3149
- **hit_ratio@3:** 0.2097
- **hit_ratio@5:** 0.2557
- **hit_ratio@50:** 0.4399
- **ndcg@1:** 0.1153
- **ndcg@10:** 0.2174
- **ndcg@3:** 0.1719
- **ndcg@5:** 0.1932
- **ndcg@50:** 0.2642
- **val_loss:** 4.6052
