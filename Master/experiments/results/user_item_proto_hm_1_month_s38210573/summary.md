# Experiment Summary: user_item_proto x hm_1_month (seed=38210573)

- **Machine:** n029.intern.leo5.intra.uibk.ac.at
- **Completed:** 30/30 trials (configured: 30)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 3h45m45s
- **Run name:** user_item_proto_hm_1_month_n0_38210573_2026-6-8_15-58-48.168290

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_1_month | user_item_proto | 5 | yes | 9,984 | 30.5 | 54.0 | 21,963 | 3h45m45s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_1_month | user_item_proto | single | 0.6813 | 0.4342 | **0.6606** | 0.4174 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_1_month | user_item_proto | 73 | 256 | 76(u)/11(i) | 1.798(u)/1.626(i) | 0.001765(u)/0.009407(i) | s_sm | adagrad | 0.09874 | 3.73e-04 | 39 |

## All Test Metrics

- **hit_ratio@1:** 0.2145
- **hit_ratio@10:** 0.6606
- **hit_ratio@3:** 0.4083
- **hit_ratio@5:** 0.5131
- **hit_ratio@50:** 0.9329
- **ndcg@1:** 0.2145
- **ndcg@10:** 0.4174
- **ndcg@3:** 0.3265
- **ndcg@5:** 0.3696
- **ndcg@50:** 0.4790
- **test_loss:** -3.4337

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.2266
- **hit_ratio@10:** 0.6813
- **hit_ratio@3:** 0.4271
- **hit_ratio@5:** 0.5346
- **hit_ratio@50:** 0.9415
- **ndcg@1:** 0.2266
- **ndcg@10:** 0.4342
- **ndcg@3:** 0.3425
- **ndcg@5:** 0.3867
- **ndcg@50:** 0.4933
- **val_loss:** -3.5464
