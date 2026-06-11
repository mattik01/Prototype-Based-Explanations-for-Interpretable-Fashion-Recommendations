# Experiment Summary: item_proto x hm_3_month (seed=38210573)

- **Machine:** n033.intern.leo5.intra.uibk.ac.at
- **Completed:** 5/5 trials (configured: 5)
- **ASHA:** yes
- **Concurrency:** 4
- **Wall time:** 5h37m08s
- **Run name:** item_proto_hm_3_month_n0_38210573_2026-4-16_17-12-50.331739

## GPU Benchmark Row

| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|
| hm_3_month | item_proto | 4 | yes | 5,820 | 18.8 | 67.9 | 21,186 | 5h37m08s |

## Results Row

| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|
| hm_3_month | item_proto | single | 0.4419 | 0.2283 | **0.4313** | 0.2191 |

## Hyperparameters Row

| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |
|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|
| hm_3_month | item_proto | 81 | 256 | 76 (i) | 1.798 | 0.001765 | s_sm | adagrad | 0.09874 | 3.73e-04 | 39 |

## All Test Metrics

- **hit_ratio@1:** 0.0633
- **hit_ratio@10:** 0.4313
- **hit_ratio@3:** 0.1714
- **hit_ratio@5:** 0.2612
- **hit_ratio@50:** 0.7930
- **ndcg@1:** 0.0633
- **ndcg@10:** 0.2191
- **ndcg@3:** 0.1256
- **ndcg@5:** 0.1631
- **ndcg@50:** 0.3045
- **test_loss:** 0.7703

## All Validation Metrics (at best checkpoint)

- **hit_ratio@1:** 0.0668
- **hit_ratio@10:** 0.4419
- **hit_ratio@3:** 0.1835
- **hit_ratio@5:** 0.2742
- **hit_ratio@50:** 0.8035
- **ndcg@1:** 0.0668
- **ndcg@10:** 0.2283
- **ndcg@3:** 0.1345
- **ndcg@5:** 0.1727
- **ndcg@50:** 0.3139
- **val_loss:** 0.7375
