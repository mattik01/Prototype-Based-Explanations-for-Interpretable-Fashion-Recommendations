# Experiment Summary: lightfm_hist x hm_1_month (seed=38210573)

- **Machine:** n051.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 3h31m30s
- **Run name:** lightfm_hist_hm_1_month_n0_38210573_2026-8-21_20-37-56.203055

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_1_month | lightfm_hist | 5 | yes | 2,078 | 13.6 | 58.6 | 22,890 | 3h31m30s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_1_month | lightfm_hist | single | 0.6743 | 0.4472 | **0.6511** | 0.4269 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_1_month | lightfm_hist | 69 | 64 |  |  |  | s_sm | adagrad | 0.03212 | 1.05e-04 | 48 |

## All Test Metrics

- **hit_ratio@1:** 0.2382
- **hit_ratio@10:** 0.6511
- **hit_ratio@3:** 0.4206
- **hit_ratio@5:** 0.5162
- **hit_ratio@50:** 0.9242
- **ndcg@1:** 0.2382
- **ndcg@10:** 0.4269
- **ndcg@3:** 0.3439
- **ndcg@5:** 0.3832
- **ndcg@50:** 0.4885
- **test_loss:** 3.4140

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.2542
- **hit_ratio@10:** 0.6743
- **hit_ratio@3:** 0.4444
- **hit_ratio@5:** 0.5403
- **hit_ratio@50:** 0.9340
- **ndcg@1:** 0.2542
- **ndcg@10:** 0.4472
- **ndcg@3:** 0.3645
- **ndcg@5:** 0.4039
- **ndcg@50:** 0.5059
- **val_loss:** 3.3038
