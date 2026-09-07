# Experiment Summary: lightfm_hist_ids x ml-1m (seed=38210573)

- **Machine:** n049.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 1h59m06s
- **Run name:** lightfm_hist_ids_ml-1m_n0_38210573_2026-8-21_20-56-43.121780

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| ml-1m | lightfm_hist_ids | 5 | yes | 3,044 | 47.2 | 63.2 | 22,914 | 1h59m06s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| ml-1m | lightfm_hist_ids | single | 0.6722 | 0.4027 | **0.6374** | 0.3767 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| ml-1m | lightfm_hist_ids | 69 | 64 |  |  |  | s_sm | adagrad | 0.03212 | 1.05e-04 | 48 |

## All Test Metrics

- **hit_ratio@1:** 0.1671
- **hit_ratio@10:** 0.6374
- **hit_ratio@3:** 0.3513
- **hit_ratio@5:** 0.4702
- **hit_ratio@50:** 0.9624
- **ndcg@1:** 0.1671
- **ndcg@10:** 0.3767
- **ndcg@3:** 0.2735
- **ndcg@5:** 0.3224
- **ndcg@50:** 0.4510
- **test_loss:** 3.5173

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1833
- **hit_ratio@10:** 0.6722
- **hit_ratio@3:** 0.3840
- **hit_ratio@5:** 0.5020
- **hit_ratio@50:** 0.9687
- **ndcg@1:** 0.1833
- **ndcg@10:** 0.4027
- **ndcg@3:** 0.2992
- **ndcg@5:** 0.3477
- **ndcg@50:** 0.4709
- **val_loss:** 3.3440
