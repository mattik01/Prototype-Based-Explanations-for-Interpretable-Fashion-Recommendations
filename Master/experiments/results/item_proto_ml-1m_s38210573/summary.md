# Experiment Summary: item_proto x ml-1m (seed=38210573)

- **Machine:** n033.intern.leo5.intra.uibk.ac.at
- **Completed:** 100/100 trials (configured: 100)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 9h45m35s
- **Run name:** item_proto_ml-1m_n0_38210573_2026-6-6_2-26-13.704524

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| ml-1m | item_proto | 5 | yes | 12,704 | 88.8 | 78.1 | 25,964 | 9h45m35s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| ml-1m | item_proto | single | 0.5877 | 0.3323 | **0.5723** | 0.3191 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| ml-1m | item_proto | 99 | 64 | 99 (i) | 0.448 | 0.4781 | s_sm | adagrad | 0.03212 | 1.05e-04 | 48 |

## All Test Metrics

- **hit_ratio@1:** 0.1251
- **hit_ratio@10:** 0.5723
- **hit_ratio@3:** 0.2847
- **hit_ratio@5:** 0.3931
- **hit_ratio@50:** 0.9450
- **ndcg@1:** 0.1251
- **ndcg@10:** 0.3191
- **ndcg@3:** 0.2166
- **ndcg@5:** 0.2611
- **ndcg@50:** 0.4034
- **test_loss:** 1.7852

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1369
- **hit_ratio@10:** 0.5877
- **hit_ratio@3:** 0.3001
- **hit_ratio@5:** 0.4074
- **hit_ratio@50:** 0.9562
- **ndcg@1:** 0.1369
- **ndcg@10:** 0.3323
- **ndcg@3:** 0.2300
- **ndcg@5:** 0.2740
- **ndcg@50:** 0.4163
- **val_loss:** 1.7138
