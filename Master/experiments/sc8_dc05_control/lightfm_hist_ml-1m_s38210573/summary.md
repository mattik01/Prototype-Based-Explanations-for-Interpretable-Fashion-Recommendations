# Experiment Summary: lightfm_hist x ml-1m (seed=38210573)

- **Machine:** n038.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 1h50m07s
- **Run name:** lightfm_hist_ml-1m_n0_38210573_2026-8-21_21-46-50.236886

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| ml-1m | lightfm_hist | 5 | yes | 3,062 | 46.3 | 76.7 | 23,025 | 1h50m07s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| ml-1m | lightfm_hist | single | 0.6425 | 0.3789 | **0.6107** | 0.3551 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| ml-1m | lightfm_hist | 89 | 256 |  |  |  | s_sm | adagrad | 0.02522 | 2.43e-04 | 26 |

## All Test Metrics

- **hit_ratio@1:** 0.1535
- **hit_ratio@10:** 0.6107
- **hit_ratio@3:** 0.3301
- **hit_ratio@5:** 0.4417
- **hit_ratio@50:** 0.9506
- **ndcg@1:** 0.1535
- **ndcg@10:** 0.3551
- **ndcg@3:** 0.2548
- **ndcg@5:** 0.3006
- **ndcg@50:** 0.4324
- **test_loss:** 3.5627

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1661
- **hit_ratio@10:** 0.6425
- **hit_ratio@3:** 0.3598
- **hit_ratio@5:** 0.4735
- **hit_ratio@50:** 0.9592
- **ndcg@1:** 0.1661
- **ndcg@10:** 0.3789
- **ndcg@3:** 0.2774
- **ndcg@5:** 0.3240
- **ndcg@50:** 0.4511
- **val_loss:** 3.4058
