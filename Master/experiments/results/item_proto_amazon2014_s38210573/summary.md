# Experiment Summary: item_proto x amazon2014 (seed=38210573)

- **Machine:** n033.intern.leo5.intra.uibk.ac.at
- **Completed:** 100/100 trials (configured: 100)
- **ASHA:** yes
- **Concurrency:** 5
- **Wall time:** 2h08m02s
- **Run name:** item_proto_amazon2014_n0_38210573_2026-6-6_0-16-32.283054

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| amazon2014 | item_proto | 5 | yes | 11,262 | 40.8 | 61.3 | 24,888 | 2h08m02s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| amazon2014 | item_proto | single | 0.4935 | 0.2829 | **0.3950** | 0.2193 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| amazon2014 | item_proto | 58 | 128 | 94 (i) | 1.364 | 0.01531 | s_sm | adagrad | 0.05949 | 1.80e-04 | 32 |

## All Test Metrics

- **hit_ratio@1:** 0.0853
- **hit_ratio@10:** 0.3950
- **hit_ratio@3:** 0.1983
- **hit_ratio@5:** 0.2689
- **hit_ratio@50:** 0.8278
- **ndcg@1:** 0.0853
- **ndcg@10:** 0.2193
- **ndcg@3:** 0.1499
- **ndcg@5:** 0.1789
- **ndcg@50:** 0.3134
- **test_loss:** 1.4534

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.1199
- **hit_ratio@10:** 0.4935
- **hit_ratio@3:** 0.2584
- **hit_ratio@5:** 0.3437
- **hit_ratio@50:** 0.8754
- **ndcg@1:** 0.1199
- **ndcg@10:** 0.2829
- **ndcg@3:** 0.1994
- **ndcg@5:** 0.2345
- **ndcg@50:** 0.3677
- **val_loss:** 1.2606
