# Experiment Summary: mf x ml-1m (seed=38210573)

- **Machine:** n039.intern.leo5.intra.uibk.ac.at
- **Completed:** 5/5 trials (configured: 5)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 13m37s
- **Run name:** mf_ml-1m_n0_38210573_2026-4-15_1-38-47.777716

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| ml-1m | mf | 5 | yes | 1,516 | 13.2 | 96.4 | 20,021 | 13m37s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| ml-1m | mf | single | 0.4400 | 0.2431 | **0.4307** | 0.2357 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| ml-1m | mf | 82 | 256 |  |  |  | s_sm | adagrad | 0.09874 | 3.73e-04 | 39 |

## All Test Metrics

- **hit_ratio@1:** 0.0903
- **hit_ratio@10:** 0.4307
- **hit_ratio@3:** 0.2050
- **hit_ratio@5:** 0.2864
- **hit_ratio@50:** 0.8774
- **ndcg@1:** 0.0903
- **ndcg@10:** 0.2357
- **ndcg@3:** 0.1558
- **ndcg@5:** 0.1892
- **ndcg@50:** 0.3344
- **test_loss:** 4.4226

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.0970
- **hit_ratio@10:** 0.4400
- **hit_ratio@3:** 0.2135
- **hit_ratio@5:** 0.2922
- **hit_ratio@50:** 0.8913
- **ndcg@1:** 0.0970
- **ndcg@10:** 0.2431
- **ndcg@3:** 0.1632
- **ndcg@5:** 0.1955
- **ndcg@50:** 0.3431
- **val_loss:** 4.4102
