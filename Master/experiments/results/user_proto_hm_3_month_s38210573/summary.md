# Experiment Summary: user_proto x hm_3_month (seed=38210573)

- **Machine:** n035.intern.leo5.intra.uibk.ac.at
- **Completed:** 5/5 trials (configured: 5)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 8h46m10s
- **Run name:** user_proto_hm_3_month_n0_38210573_2026-4-15_10-20-28.830989

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_3_month | user_proto | 5 | yes | 2,590 | 3.5 | 61.2 | 18,841 | 8h46m10s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_3_month | user_proto | single | 0.5092 | 0.2814 | **0.4981** | 0.2740 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_3_month | user_proto | 46 | 256 | 58 (u) | 3.02 | 5.71 | s_sm | adagrad | 8.05e-03 | 5.53e-04 | 17 |

## All Test Metrics

- **hit_ratio@1:** 0.1075
- **hit_ratio@10:** 0.4981
- **hit_ratio@3:** 0.2371
- **hit_ratio@5:** 0.3304
- **hit_ratio@50:** 0.9206
- **ndcg@1:** 0.1075
- **ndcg@10:** 0.2740
- **ndcg@3:** 0.1817
- **ndcg@5:** 0.2200
- **ndcg@50:** 0.3689
- **test_loss:** -13.5354

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1112
- **hit_ratio@10:** 0.5092
- **hit_ratio@3:** 0.2454
- **hit_ratio@5:** 0.3409
- **hit_ratio@50:** 0.9246
- **ndcg@1:** 0.1112
- **ndcg@10:** 0.2814
- **ndcg@3:** 0.1881
- **ndcg@5:** 0.2272
- **ndcg@50:** 0.3749
- **val_loss:** -13.6050
