# Experiment Summary: acf x hm_3_month (seed=38210573)

- **Machine:** n039.intern.leo5.intra.uibk.ac.at
- **Completed:** 5/5 trials (configured: 5)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 4h04m20s
- **Run name:** acf_hm_3_month_n0_38210573_2026-4-16_17-12-39.249239

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_3_month | acf | 5 | yes | 3,306 | 18.0 | 61.6 | 22,887 | 4h04m20s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_3_month | acf | single | 0.3686 | 0.1926 | **0.3623** | 0.1885 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_3_month | acf | 55 | 128 | 92 (n_a) |  |  | bpr | adam | 0.0102 | 4.70e-03 | 46 |

## All Test Metrics

- **hit_ratio@1:** 0.0653
- **hit_ratio@10:** 0.3623
- **hit_ratio@3:** 0.1519
- **hit_ratio@5:** 0.2227
- **hit_ratio@50:** 0.8324
- **ndcg@1:** 0.0653
- **ndcg@10:** 0.1885
- **ndcg@3:** 0.1147
- **ndcg@5:** 0.1437
- **ndcg@50:** 0.2921
- **test_loss:** 15101.8094

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.0656
- **hit_ratio@10:** 0.3686
- **hit_ratio@3:** 0.1577
- **hit_ratio@5:** 0.2325
- **hit_ratio@50:** 0.8287
- **ndcg@1:** 0.0656
- **ndcg@10:** 0.1926
- **ndcg@3:** 0.1181
- **ndcg@5:** 0.1488
- **ndcg@50:** 0.2938
- **val_loss:** 15222.0969
