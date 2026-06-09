# Replication Report

The LEO5 cluster is the **primary processing machine** going forward; all production runs land
there and feed the [Cluster hardware table](#cluster-leo5--primary). The home GPU machine (16 GB)
is kept only for smoke/dev sizing reference.

In-scope datasets: **amazon2014**, **ml-1m** (paper replication) and **hm_1_month** (H&M small).
Out-of-scope rows that still carry useful numbers (hm_3_month, hm_full, lfm2b) are parked in the
[Appendix](#appendix--out-of-scope-retained-for-reference).

## Dataset Comparison

Pre-filter = raw interactions before processing. Post-filter (= **Interactions** column) = after
dedup + k-core + leave-1-out split.

| Dataset | K-core | Pre-filter | Users | Items | **Interactions** | Density | Avg/User | Avg/Item |
|---------|:------:|-----------:|------:|------:|-----------------:|--------:|---------:|---------:|
| amazon2014 | 5 | 1,324,753 | 6,950 | 14,494 | **132,209** | 0.131% | 19.0 | 9.1 |
| ml-1m | 5 | 1,000,209 | 6,034 | 3,125 | **574,376** | 3.047% | 95.2 | 183.8 |
| hm_1_month | 5 | — | 73,418 | 13,651 | **623,230** | 0.062% | 8.5 | 45.7 |

## Run Results

Our headline held-out test numbers across all in-scope datasets (single seed 38210573). H&M has no
paper baseline, so it lives only here. amazon2014/ml-1m additionally appear in
[Replication Results](#replication-results) with the paper comparison.

‡ = **hm_1_month** = `dev` profile (30 samples / 60 epochs); amazon2014/ml-1m = `prod` (100 / 100).
See [Hardware Benchmarks](#hardware-benchmarks) for profile definitions. Any smoke-only cell would be
marked and replaced once a full run lands (none currently — all rows are full runs).

| Dataset    | Model           | Test HR@10 | Test NDCG@10 |
|------------|-----------------|:----------:|:------------:|
| amazon2014 | mf              |     0.2082 |       0.1112 |
| amazon2014 | acf             |     0.3776 |       0.1896 |
| amazon2014 | user_proto      |     0.2859 |       0.1525 |
| amazon2014 | item_proto      |     0.3950 |       0.2193 |
| amazon2014 | user_item_proto |     0.3788 |       0.2116 |
| ml-1m      | mf              |     0.5676 |       0.3246 |
| ml-1m      | acf             |     0.6006 |       0.3364 |
| ml-1m      | user_proto      |     0.5933 |       0.3465 |
| ml-1m      | item_proto      |     0.5723 |       0.3191 |
| ml-1m      | user_item_proto |     0.6246 |       0.3573 |
| hm_1_month | mf‡             |     0.3652 |       0.2233 |
| hm_1_month | acf‡            |     0.5764 |       0.3307 |
| hm_1_month | user_proto‡     |     0.5725 |       0.3322 |
| hm_1_month | item_proto‡     |     0.4937 |       0.2630 |
| hm_1_month | user_item_proto‡|     0.6606 |       0.4174 |

## Replication Results

100-sample hyperopt, single seed (38210573), paper datasets only. Paper = original 3-seed mean from
Table 2 (RecSys '22). Val = best validation (trial selection). Test = held-out evaluation.

| Dataset    | Model           | Paper HR@10 | Paper NDCG@10 | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|------------|-----------------|:-----------:|:-------------:|:---------:|:-----------:|:----------:|:------------:|
| amazon2014 | mf              |   **0.255** |         0.140 |    0.2426 |      0.1351 | **0.2082** |       0.1112 |
| amazon2014 | acf             |   **0.392** |         0.202 |    0.4649 |      0.2403 | **0.3776** |       0.1896 |
| amazon2014 | user_proto      |   **0.276** |         0.152 |    0.3866 |      0.2132 | **0.2859** |       0.1525 |
| amazon2014 | item_proto      |   **0.371** |         0.194 |    0.4935 |      0.2829 | **0.3950** |       0.2193 |
| amazon2014 | user_item_proto |   **0.401** |         0.220 |    0.4931 |      0.2881 | **0.3788** |       0.2116 |
| ml-1m      | mf              |   **0.571** |         0.326 |    0.6013 |      0.3503 | **0.5676** |       0.3246 |
| ml-1m      | acf             |   **0.597** |         0.335 |    0.6294 |      0.3530 | **0.6006** |       0.3364 |
| ml-1m      | user_proto      |   **0.583** |         0.333 |    0.6268 |      0.3720 | **0.5933** |       0.3465 |
| ml-1m      | item_proto      |   **0.544** |         0.303 |    0.5877 |      0.3323 | **0.5723** |       0.3191 |
| ml-1m      | user_item_proto |   **0.657** |         0.383 |    0.6543 |      0.3790 | **0.6246** |       0.3573 |

## Best Trial Hyperparameters

Selected by Ray Tune (best val HR@10). Proto-specific columns blank for non-prototype models.

| Dataset    | Model           | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim   | lr     | wd       | neg_train |
|------------|-----------------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-------:|:------:|:--------:|:---------:|
| amazon2014 | mf              |      61 |   512 |                |                    |                    | bpr  | adagrad | 0.0448 | 1.15e-4  |        47 |
| amazon2014 | acf             |      13 |   128 |       50 (n_a) |                    |                    | bpr  | adam    | 0.0152 | 6.62e-3  |        49 |
| amazon2014 | user_proto      |      78 |   128 |         68 (u) |             0.5985 |             0.5579 | s_sm | adagrad | 0.0446 | 1.12e-4  |        21 |
| amazon2014 | item_proto      |      58 |   128 |         94 (i) |          1.364 (i) |        0.01531 (i) | s_sm | adagrad | 0.0595 | 1.80e-4  |        32 |
| amazon2014 | user_item_proto |      89 |   256 |    57(u)/12(i) | 0.0123(u)/3.937(i) | 0.0093(u)/0.023(i) | s_sm | adagrad | 0.0648 | 3.00e-4  |        36 |
| ml-1m      | mf              |      34 |    64 |                |                    |                    | s_sm | adam    | 1.00e-4 | 1.02e-4 |        37 |
| ml-1m      | acf             |      60 |   128 |       72 (n_a) |                    |                    | bce  | adam    | 3.33e-3 | 9.85e-3 |        23 |
| ml-1m      | user_proto      |      77 |   128 |         76 (u) |             0.2121 |             0.0045 | s_sm | adagrad | 0.0890 | 1.43e-4  |        45 |
| ml-1m      | item_proto      |      99 |    64 |         99 (i) |          0.448 (i) |        0.4781 (i) | s_sm | adagrad | 0.0321 | 1.05e-4  |        48 |
| ml-1m      | user_item_proto |      73 |   256 |    76(u)/11(i) | 1.798(u)/1.626(i)  | 0.0018(u)/0.0094(i) | s_sm | adagrad | 0.0987 | 3.73e-4  |        39 |
| hm_1_month | mf‡             |      44 |   512 |                |                    |                    | s_sm | adagrad | 0.0561 | 1.02e-4  |        15 |
| hm_1_month | acf‡            |      81 |    64 |       14 (n_a) |                    |                    | bpr  | adam    | 5.79e-3 | 3.38e-3 |        49 |
| hm_1_month | user_proto‡     |      81 |   256 |         76 (u) |             1.798  |           0.001765 | s_sm | adagrad | 0.0987 | 3.73e-4  |        39 |
| hm_1_month | item_proto‡     |      81 |   256 |         76 (i) |          1.798 (i) |       0.001765 (i) | s_sm | adagrad | 0.0987 | 3.73e-4  |        39 |
| hm_1_month | user_item_proto‡|      73 |   256 |    76(u)/11(i) | 1.798(u)/1.626(i)  | 0.001765(u)/0.009407(i) | s_sm | adagrad | 0.0987 | 3.73e-4 |     39 |

> **‡ hm_1_month best-trial provenance:** The proto models' best configs are near-identical across
> models (`user_proto`/`item_proto` share emb=81, batch=256, n_protos=76, sim_proto_w=1.798,
> sim_batch_w=0.001765, lr=0.0987, wd=3.73e-4, neg=39) and the `user_item_proto` config matches the
> ml-1m best trial. This is genuine: the values were read directly from each run's authoritative
> `config.json` on LEO5 (`/scratch/c7031336/protomf_results/<model>_hm_1_month_s38210573/config.json`).
> The similarity stems from the shared Ray Tune seed (38210573) — `user_proto` and `item_proto` have
> structurally identical search spaces, so the sampled trial sequence is identical and the same trial
> index won. Test metrics are dataset-specific and genuine (e.g. `user_item_proto` HR@10=0.6606 vs
> ml-1m 0.6246).

## Hardware Benchmarks

**Profile** (run configuration, in the tables below): `smoke` = 5 samples / 15 epochs · `dev` = 30
samples / 60 epochs, patience 7 · `prod` = 100 samples / 100 epochs, patience 10 · `stress` =
single-trial VRAM probe (no full search). Wall time scales strongly with profile; VRAM/RAM scale
with concurrency, not sample count.

### Cluster (LEO5) — primary

Observed per-run resource usage from completed cluster runs (full hyperopt). Schema matches the
pipeline's auto-generated `summary.md` "GPU Benchmark Row" — **paste new rows here directly**.
`GPU` = which LEO5 GPU the job landed on (A100 / A30 / A40); `Peak VRAM` = `max_vram_mib`.
*(This is the table the `leo5-submit` skill reads for VRAM / RAM / wall-time estimates.)*

| Dataset    | Model           | GPU  | Concurrency | ASHA | Profile | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time | Job ID  |
|------------|-----------------|:----:|:-----------:|:----:|:-------:|:--------------:|:--------:|:--------:|:------------:|:---------:|:-------:|
| amazon2014 | item_proto      | A100 |           5 | yes  |  prod   |         11,262 |     40.8 |     61.3 |       24,888 |  2h 08m   | 6600278 |
| ml-1m      | item_proto      | A100 |           5 | yes  |  prod   |         12,704 |     88.8 |     78.1 |       25,964 |  9h 46m   | 6600279 |
| ml-1m      | user_item_proto | A30  |           5 | yes  |  prod   |         12,472 |     95.6 |     72.8 |       26,215 |  6h 10m   | 6535752 |
| hm_1_month | mf              | A30  |           5 | yes  |  dev    |          2,110 |     11.2 |     55.2 |       21,931 | 3h05m32s  | 6610328 |
| hm_1_month | acf             | A30  |           5 | yes  |  dev    |          2,426 |     26.5 |     52.0 |       23,441 | 2h55m55s  | 6610329 |
| hm_1_month | user_proto      | A30  |           5 | yes  |  dev    |          2,124 |     20.7 |     62.1 |       22,308 | 3h35m08s  | 6610330 |
| hm_1_month | item_proto      | A100 |           5 | yes  |  dev    |         11,100 |     30.7 |     68.4 |       21,700 | 2h54m19s  | 6610332 |
| hm_1_month | user_item_proto | A100 |           5 | yes  |  dev    |          9,984 |     30.5 |     54.0 |       21,963 | 3h45m45s  | 6610333 |

**Sizing guidance (concurrency 5):**
- **RAM is the binding constraint** and is dataset-dominated (~uniform across models): hm_1_month
  peaks ~26–27 GB (avg ~22 GB) at 5 concurrent trials (each loads the full sparse matrices). The
  skill's ~8 GB formula is ~3× too low → request **`--mem 40G`**.
- **VRAM:** CF/light models ~2 GB; item-prototype models ~10–12 GB peak. A30 (24 GB) fits all at
  concurrency 5; A100 used for the heavy item-proto models but not strictly required on VRAM grounds.
- **Wall time:** hm_1_month 30-sample dev ≈ 3–4 h/model. Heavier 100-sample runs scale up
  (ml-1m item_proto hit ~10 h); budget several days for full hm production runs.

### GPU Machine (16 GB) — smoke/dev reference

Stress-test VRAM planning on the home machine (DESKTOP-715IF4P, 16 GB VRAM, 16 cores). All rows are
`gpu-machine`, non-smoke. **Worstcase VRAM** = single-trial stress test, max params, 512 MB baseline
subtracted. **SAFE-ISH VRAM** = `(peak_vram − 512) / concurrency` (per-trial under real multi-trial
load). VRAM hyperparameter ranges: emb_dim [10,100], protos/anchors [10,100], batch [64,512].

| Dataset    | Model           | Conc | ASHA | Profile | Worstcase VRAM (MB) | SAFE-ISH VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|------------|-----------------|:----:|:----:|:-------:|:-------------------:|:------------------:|:--------:|:--------:|:------------:|:---------:|
| amazon2014 | mf              |   16 |  no  |  prod   |               1,062 |                297 |     22.7 |     85.5 |       19,401 |   2h 43m  |
| amazon2014 | acf             |   16 |  no  |  prod   |               1,198 |                352 |     36.8 |     88.6 |       19,674 |   3h 07m  |
| amazon2014 | user_proto      |   16 |  no  |  prod   |               1,266 |                311 |     36.8 |     92.9 |       20,508 |   4h 11m  |
| amazon2014 | item_proto      |      |  no  | stress  |              10,072 |                    |          |          |              |           |
| amazon2014 | user_item_proto |    8 | yes  |  prod   |              13,024 |              1,914 |     96.1 |     70.1 |       18,767 |   2h 26m  |
| ml-1m      | mf              |   16 |  no  |  prod   |               4,925 |                276 |     48.4 |     94.6 |       19,563 |   5h 14m  |
| ml-1m      | acf             |   16 |  no  |  prod   |               4,138 |                313 |     84.8 |     94.4 |       17,248 |   9h 43m  |
| ml-1m      | user_proto      |    8 |  no  |  prod   |               4,180 |                249 |     41.1 |     64.5 |       14,078 |  ~14h16m  |
| ml-1m      | item_proto      |      |  no  | stress  |              13,002 |                    |          |          |              |           |
| ml-1m      | user_item_proto |      |  no  | stress  |              13,020 |                    |          |          |              |           |

- **item_proto / user_item_proto:** prototype similarity matrices scale with n_items × n_prototypes
  → 10–13 GB worst-case; run **1 trial at a time** on the 16 GB machine.
- **Lighter models on amazon2014:** VRAM so low (~1–1.3 GB) that CPU cores bottleneck at 16 trials.
- **ml-1m user_proto:** SSH drop killed the process at 99/100 trials; best of 99 selected manually,
  test run executed separately. Concurrency reduced to 8 (from a 16-trial OOM).

---

## Appendix — out-of-scope, retained for reference

Datasets/runs no longer in active scope (hm_3_month, hm_full, lfm2b), kept only for their measured
numbers. Not maintained; supersede or delete when no longer useful.

### Dataset stats (out-of-scope)

| Dataset | K-core | Pre-filter | Users | Items | **Interactions** | Density | Avg/User | Avg/Item |
|---------|:------:|-----------:|------:|------:|-----------------:|--------:|---------:|---------:|
| hm_3_month | 5 | 4,056,792 | 256,701 | 26,963 | **2,898,804** | 0.042% | 11.3 | 107.5 |
| hm_full | 5 | 31,788,324 | 889,062 | 90,690 | **26,215,294** | 0.033% | 29.5 | 289.1 |

### hm_3_month preliminary results (* = incomplete hyperopt, ASHA early stopping)

| Dataset    | Model           | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |
|------------|-----------------|:---------:|:-----------:|:----------:|:------------:|
| hm_3_month | mf*             |    0.2488 |      0.1318 | **0.2448** |       0.1291 |
| hm_3_month | user_item_proto*|    0.5548 |      0.3166 | **0.5343** |       0.3016 |

hm_3_month `mf` best-trial hyperparameters: emb 88, batch 256, bce, adagrad, lr 0.0038, wd 1.03e-4,
neg 45.

> **hm_3_month user_item_proto hyperparameters were cleared (2026-06-05):** the previously recorded
> values were byte-identical to the ml-1m user_item_proto run and could not be verified against a
> source `config.json` (run was on the Windows GPU machine, not LEO5/repo). Re-extract from the GPU
> machine's `protomf_results` or W&B before re-filling. (Contrast with the in-scope hm_1_month
> provenance note above, where the LEO5 `config.json` *was* verified.)

### hm_3_month smoke results (LEO5, single seed 38210573, ASHA, patience 10)

† = TIMEOUT (hit SLURM time limit before all trials finished); best-so-far val shown, no test run.

| Batch | Model      | Machine   | Conc | Trials | Status   | Wall    | Val HR@10 | Test HR@10 | Test NDCG@10 | Peak VRAM (MiB) | Avg RAM (MiB) | Avg GPU% | Avg CPU% |
|-------|------------|-----------|:----:|:------:|----------|:-------:|:---------:|:----------:|:------------:|:---------------:|:-------------:|:--------:|:--------:|
| 1 (100ep) | acf†       | leo5 A30  |  5 | 3/5 | TIMEOUT  | 12h00m | 0.3879 |        |        |        |        |      |      |
| 1 (100ep) | user_proto | leo5 A30  |  5 | 5/5 | COMPLETE |  8h46m | 0.5092 | 0.4980 | 0.2740 |  2,590 | 18,841 |  3.5 | 61.2 |
| 1 (100ep) | item_proto†| leo5 A100 |  4 | 4/5 | TIMEOUT  | 12h00m | 0.4489 |        |        |        |        |      |      |
| 2 (10ep)  | acf        | leo5 A30  |  5 | 5/5 | COMPLETE |  4h04m | 0.3686 | 0.3620 | 0.1890 |  3,306 | 22,887 | 18.0 | 61.6 |
| 2 (10ep)  | item_proto | leo5 A100 |  4 | 5/5 | COMPLETE |  5h37m | 0.4419 | 0.4310 | 0.2190 |  5,820 | 21,186 | 18.8 | 67.9 |

Notes: item_proto VRAM on A100 only 5.8–7.8 GB peak (well under the 16 GB-machine worstcase); A100s
run 5 concurrent item_proto trials comfortably. hm_3_month per-trial times ~45 min–2 h even at 10
epochs → 100-sample/100-epoch production runs should budget 3–5 days.

### hm_3_month GPU-machine hardware (16 GB)

| Dataset    | Model           | Conc | ASHA | SAFE-ISH VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |
|------------|-----------------|:----:|:----:|:------------------:|:--------:|:--------:|:------------:|:---------:|
| hm_3_month | mf              |   16 | yes  |                855 |     59.6 |     99.9 |       23,664 |  ~12h45m† |
| hm_3_month | user_item_proto |    4 | yes  |                    |          |          |              |       †   |

† hm_3_month mf: 65/100 trials completed with ASHA (16 running + 1 error + 1 pending at interrupt
~12h45m); best trial from incomplete search, test run extracted manually; peak VRAM 14,185 MB across
16 concurrent trials. † hm_3_month user_item_proto: run via `start.py` (no GPU sampler) → no
VRAM/GPU stats; 19/30 trials completed (13 finished, rest killed on interruption); wall time not
recorded.

### lfm2b-1mon (paper reference numbers)

Dataset removed due to licensing issues (zipped copy in the data folder; not used). Paper HR@10 /
NDCG@10 from RecSys '22 Table 2: mf 0.215 / 0.118 · acf 0.517 / 0.291 · user_proto 0.322 / 0.179 ·
item_proto 0.457 / 0.251 · user_item_proto 0.579 / 0.347. Reported post-filter (10-core): 3,555
users, 77,985 items, 877,365 interactions.
