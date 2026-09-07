# Experiment Summary: lightfm_hist_ids x hm_1_month (seed=38210573)

- **Machine:** n038.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 2h20m36s
- **Run name:** lightfm_hist_ids_hm_1_month_n0_38210573_2026-8-21_6-10-45.712772

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_1_month | lightfm_hist_ids | 5 | yes | 2,740 | 21.5 | 61.2 | 24,306 | 2h20m36s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_1_month | lightfm_hist_ids | single | 0.6700 | 0.4437 | **0.6464** | 0.4223 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_1_month | lightfm_hist_ids | 69 | 256 |  |  |  | s_sm | adagrad | 0.08854 | 1.05e-04 | 48 |

## All Test Metrics

- **hit_ratio@1:** 0.2331
- **hit_ratio@10:** 0.6464
- **hit_ratio@3:** 0.4174
- **hit_ratio@5:** 0.5133
- **hit_ratio@50:** 0.9219
- **ndcg@1:** 0.2331
- **ndcg@10:** 0.4223
- **ndcg@3:** 0.3398
- **ndcg@5:** 0.3792
- **ndcg@50:** 0.4845
- **test_loss:** 3.4728

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.2519
- **hit_ratio@10:** 0.6700
- **hit_ratio@3:** 0.4389
- **hit_ratio@5:** 0.5368
- **hit_ratio@50:** 0.9306
- **ndcg@1:** 0.2519
- **ndcg@10:** 0.4437
- **ndcg@3:** 0.3603
- **ndcg@5:** 0.4006
- **ndcg@50:** 0.5026
- **val_loss:** 3.3589
