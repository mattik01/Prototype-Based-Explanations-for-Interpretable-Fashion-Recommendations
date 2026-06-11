# Experiment Summary: acf x hm_1_month (seed=38210573)

- **Machine:** n043.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 2h55m55s
- **Run name:** acf_hm_1_month_n0_38210573_2026-6-8_15-20-27.108622

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_1_month | acf | 5 | yes | 2,426 | 26.5 | 52.0 | 23,441 | 2h55m55s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_1_month | acf | single | 0.5884 | 0.3415 | **0.5764** | 0.3307 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_1_month | acf | 81 | 64 | 14 (n_a) |  |  | bpr | adam | 5.79e-03 | 3.38e-03 | 49 |

## All Test Metrics

- **hit_ratio@1:** 0.1379
- **hit_ratio@10:** 0.5764
- **hit_ratio@3:** 0.3034
- **hit_ratio@5:** 0.4110
- **hit_ratio@50:** 0.8977
- **ndcg@1:** 0.1379
- **ndcg@10:** 0.3307
- **ndcg@3:** 0.2330
- **ndcg@5:** 0.2773
- **ndcg@50:** 0.4039
- **test_loss:** 9836.7020

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1468
- **hit_ratio@10:** 0.5884
- **hit_ratio@3:** 0.3158
- **hit_ratio@5:** 0.4232
- **hit_ratio@50:** 0.9054
- **ndcg@1:** 0.1468
- **ndcg@10:** 0.3415
- **ndcg@3:** 0.2440
- **ndcg@5:** 0.2882
- **ndcg@50:** 0.4136
- **val_loss:** 9605.1519
