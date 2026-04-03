# Replication Report

## Dataset Comparison

Pre-filter = raw interactions before processing. Post-filter = after dedup + k-core + leave-1-out split.

| Dataset | K-core | Pre-filter | Users | Items | **Interactions** | Density | Avg/User | Avg/Item |
|---------|:------:|-----------:|------:|------:|-----------------:|--------:|---------:|---------:|
| amazon2014 | 5 | 1,324,753 | 6,950 | 14,494 | **132,209** | 0.131% | 19.0 | 9.1 |
| ml-1m | 5 | 1,000,209 | 6,034 | 3,125 | **574,376** | 3.047% | 95.2 | 183.8 |
| hm_3_month | 5 | 4,056,792 | 256,701 | 26,963 | **2,898,804** | 0.042% | 11.3 | 107.5 |
| hm_full | 5 | 31,788,324 | 889,062 | 90,690 | **26,215,294** | 0.033% | 29.5 | 289.1 |

## GPU Benchmark

**Machine:** DESKTOP-715IF4P — 16 GB VRAM, 16 CPU cores

- **Worstcase VRAM (MB):** single-trial stress test with max params, 512 MB baseline subtracted
- **SAFE-ISH VRAM (MB):** `(peak_vram_during_run - 512) / concurrency` — per-trial estimate under real multi-trial load

### VRAM-relevant hyperparameter ranges

- **mf:** emb_dim:[10,100], batch:[64,512]
- **acf:** emb_dim:[10,100], n_anchors:[10,100], batch:[64,512]
- **user_proto:** emb_dim:[10,100], u_protos:[10,100], batch:[64,512]
- **item_proto:** emb_dim:[10,100], i_protos:[10,100], batch:[64,512]
- **user_item_proto:** emb_dim:[10,100], u_protos:[10,100], i_protos:[10,100], batch:[64,512]

| Dataset    | Model           | Concurrency | ASHA | Worstcase VRAM (MB) | SAFE-ISH VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|------------|-----------------|:-----------:|:----:|:-------------------:|:-------------------:|:--------:|:--------:|:------------:|:---------:|
| amazon2014 | mf              |          16 |  no  |               1,062 |                 297 |     22.7 |     85.5 |       19,401 |   2h 43m  |
| amazon2014 | acf             |          16 |  no  |               1,198 |                 352 |     36.8 |     88.6 |       19,674 |   3h 07m  |
| amazon2014 | user_proto      |          16 |  no  |               1,266 |                 311 |     36.8 |     92.9 |       20,508 |   4h 11m  |
| amazon2014 | item_proto      |             |  no  |              10,072 |                     |          |          |              |           |
| amazon2014 | user_item_proto |           8 | yes  |              13,024 |               1,914 |     96.1 |     70.1 |       18,767 |   2h 26m  |
| ml-1m      | mf              |          16 |  no  |               4,925 |                 276 |     48.4 |     94.6 |       19,563 |   5h 14m  |
| ml-1m      | acf             |          16 |  no  |               4,138 |                 313 |     84.8 |     94.4 |       17,248 |   9h 43m  |
| ml-1m      | user_proto      |           8 |  no  |               4,180 |                 249 |     41.1 |     64.5 |       14,078 |  ~14h16m  |
| ml-1m      | item_proto      |             |  no  |              13,002 |                     |          |          |              |           |
| ml-1m      | user_item_proto |             |  no  |              13,020 |                     |          |          |              |           |
| hm_3_month | mf              |          16 | yes  |                     |                 855 |     59.6 |     99.9 |       23,664 |  ~12h45m† |
| hm_3_month | acf             |             |      |                     |                     |          |          |              |           |
| hm_3_month | user_proto      |             |      |                     |                     |          |          |              |           |
| hm_3_month | item_proto      |             |      |                     |                     |          |          |              |           |
| hm_3_month | user_item_proto |           4 | yes  |                     |                     |          |          |              |       †   |

### Notes

- **item_proto / user_item_proto:** Prototype similarity matrices scale with n_items × n_prototypes. These eat 10–13 GB worst-case — must run 1 trial at a time.
- **Lighter models on amazon2014:** VRAM is so low (~1–1.3 GB) that CPU cores become the bottleneck at 16 concurrent trials.
- **`NUM_WORKERS`:** Stays 0 on Windows (multiprocessing limitation). Not a bottleneck since GPU util is modest for the lighter models anyway.
- **ml-1m user_proto:** SSH session drop killed the process at 99/100 trials. Best trial selected manually from 99 completed, test run executed separately. Concurrency reduced to 8 (from previous 16-trial OOM).
- **† hm_3_month user_item_proto:** Run via `start.py` (no GPU sampler attached), so no VRAM/GPU stats available. 19/30 trials completed (13 finished, rest killed on interruption). Wall time not recorded.
- **† hm_3_month mf:** 65/100 trials completed with ASHA (16 running + 1 error + 1 pending when interrupted at ~12h 45m). Best trial from incomplete search, test run extracted manually. Peak VRAM 14,185 MB across 16 concurrent trials.

## Replication Results

100-sample hyperopt, single seed (38210573). Paper = original 3-seed mean from Table 2 (RecSys '22). Val = best validation (trial selection). Test = held-out evaluation.

\* = incomplete hyperopt with ASHA early stopping. Results are preliminary — best trial from incomplete search.

| Dataset    | Model           | Seeds  | Paper HR@10 | Paper NDCG@10 | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|------------|-----------------|:------:|:-----------:|:-------------:|:---------:|:-----------:|:----------:|:------------:|
| amazon2014 | mf              | single |   **0.255** |         0.140 |    0.2426 |      0.1351 | **0.2082** |       0.1112 |
| amazon2014 | acf             | single |   **0.392** |         0.202 |    0.4649 |      0.2403 | **0.3776** |       0.1896 |
| amazon2014 | user_proto      | single |   **0.276** |         0.152 |    0.3866 |      0.2132 | **0.2859** |       0.1525 |
| amazon2014 | item_proto      | single |   **0.371** |         0.194 |           |             |            |              |
| amazon2014 | user_item_proto | single |   **0.401** |         0.220 |    0.4931 |      0.2881 | **0.3788** |       0.2116 |
| ml-1m      | mf              | single |   **0.571** |         0.326 |    0.6013 |      0.3503 | **0.5676** |       0.3246 |
| ml-1m      | acf             | single |   **0.597** |         0.335 |    0.6294 |      0.3530 | **0.6006** |       0.3364 |
| ml-1m      | user_proto      | single |   **0.583** |         0.333 |    0.6268 |      0.3720 | **0.5933** |       0.3465 |
| ml-1m      | item_proto      | single |   **0.544** |         0.303 |           |             |            |              |
| ml-1m      | user_item_proto | single |   **0.657** |         0.383 |           |             |            |              |
| hm_3_month | mf*             | single |         N/A |           N/A |    0.2488 |      0.1318 | **0.2448** |       0.1291 |
| hm_3_month | acf             | single |         N/A |           N/A |           |             |            |              |
| hm_3_month | user_proto      | single |         N/A |           N/A |           |             |            |              |
| hm_3_month | item_proto      | single |         N/A |           N/A |           |             |            |              |
| hm_3_month | user_item_proto*| single |         N/A |           N/A |    0.5548 |      0.3166 | **0.5343** |       0.3016 |

## Best Trial Hyperparameters

Selected by Ray Tune (best val HR@10). Proto-specific columns left blank for non-prototype models.

| Dataset    | Model           | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim   | lr     | wd       | neg_train |
|------------|-----------------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-------:|:------:|:--------:|:---------:|
| amazon2014 | mf              |      61 |   512 |                |                    |                    | bpr  | adagrad | 0.0448 | 1.15e-4  |        47 |
| amazon2014 | acf             |      13 |   128 |       50 (n_a) |                    |                    | bpr  | adam    | 0.0152 | 6.62e-3  |        49 |
| amazon2014 | user_proto      |      78 |   128 |         68 (u) |             0.5985 |             0.5579 | s_sm | adagrad | 0.0446 | 1.12e-4  |        21 |
| amazon2014 | item_proto      |         |       |                |                    |                    |      |         |        |          |           |
| amazon2014 | user_item_proto |      89 |   256 |    57(u)/12(i) | 0.0123(u)/3.937(i) | 0.0093(u)/0.023(i) | s_sm | adagrad | 0.0648 | 3.00e-4  |        36 |
| ml-1m      | mf              |      34 |    64 |                |                    |                    | s_sm | adam    | 1.00e-4 | 1.02e-4 |        37 |
| ml-1m      | acf             |      60 |   128 |       72 (n_a) |                    |                    | bce  | adam    | 3.33e-3 | 9.85e-3 |        23 |
| ml-1m      | user_proto      |      77 |   128 |         76 (u) |             0.2121 |             0.0045 | s_sm | adagrad | 0.0890 | 1.43e-4  |        45 |
| ml-1m      | item_proto      |         |       |                |                    |                    |      |         |        |          |           |
| ml-1m      | user_item_proto |         |       |                |                    |                    |      |         |        |          |           |
| hm_3_month | mf*             |      88 |   256 |                |                    |                    | bce  | adagrad | 0.0038 | 1.03e-4  |        45 |
| hm_3_month | acf             |         |       |                |                    |                    |      |         |        |          |           |
| hm_3_month | user_proto      |         |       |                |                    |                    |      |         |        |          |           |
| hm_3_month | item_proto      |         |       |                |                    |                    |      |         |        |          |           |
| hm_3_month | user_item_proto*|      73 |   256 |    76(u)/11(i) | 1.798(u)/1.626(i)  | 0.0018(u)/0.009(i) | s_sm | adagrad | 0.0987 | 3.73e-4  |        39 |
