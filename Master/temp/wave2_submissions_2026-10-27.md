# Wave 2 — cold single-trial retrains of the 100-trial configs, submitted 2026-10-27

All: `--retrain-config /scratch/c7031336/protomf_results/<model>_<dataset>_s38210573_n100/config.json`
`--profile production` (one trial — the saved config's `num_samples = 1` wins — ≤ 100 epochs, patience 10,
matching the tier the configs were selected under), seed 38210573, `--gpu-per-trial 1.0` (concurrency 1),
`--num-workers 1`, `--cpus 4` (the F-S0-14 floor; a lone trial keeps ~1.8 threads busy), `--mem 24G`
(2026-08 cold retrains peaked at 16–17.5 GB), `--gpu-type any`, `--wandb-tag thesis-grade`. Cluster checkout
at `c0c6ac1` (feat/scrutiny; same model code as wave 1's `3ce563d` — the dc09 changes leave every existing
path bit-identical, golden-guarded). Results land in
`/scratch/c7031336/protomf_results/<model>_<cold dataset>_s38210573_n100/` (the retrain inherits its source's
tier since `c0c6ac1`; the 2026-08 dev-tier cold folders keep their bare names and stay untouched).

`--time` (measured rule, `/leo5-submit` STEP 3e, 2026-10-27): `ceil_hour(min(100, 1.4 × E_src) × t_epoch + 0.25 h)`,
with `E_src` / `t_epoch` the source winner's epochs and median shared-GPU epoch time from its Ray
`progress.csv`. Scope (user, 2026-10-27): ten models per cold dataset; acf dropped, lightfm_hist(_ids) not run.
Cold evaluation is a post-hoc login-node step (`utilities/cold_eval.py`, mean-ID imputation default).

Pre-batch check (login node, 2026-10-27, `/scratch/c7031336/smoke_runs/wave2_check/`): 2-epoch retrains of the `_n100` configs (fI hm, fI ml-1m, item_proto ml-1m) all RUN_OK into `..._cold_s38210573_n100`; lone epoch times 0.38–0.57× the shared ones (fI hm 92–95 s vs 164, fI ml-1m 66 s vs 176) — every limit below has room; item_proto ml-1m measured at 73 s. Cold eval: hm complete (mean-ID imputation on 2730 cold items). ml-1m: numbers written, but the CF attr-kNN patch and the tie-block diagnostic are skipped (H&M fields hardwired) and the markdown writer crashes on the skipped diagnostic — evaluation-side gaps, fixed before the ml-1m cold evals run; the retrains are unaffected.

Submitted 2026-10-27 one wrapper call per run (user-authorised wave); all 20 PENDING (Priority). Cluster state at submission: no GPU placeable; 2 A30 free within 1 day, most within 2.5–3 days. **Checkout lock** on `~/UIFProtoMF/ProtoMF` until all 20 have started.

| # | job id | model | cold dataset | E_src | t_epoch | --time |
|---|---|---|---|---|---|---|
| 1 | 8159117 | user_item_proto | hm_1_month_cold | 99 | 163 s | 05:00:00 |
| 2 | 8159122 | user_proto | hm_1_month_cold | 70 | 149 s | 05:00:00 |
| 3 | 8159123 | item_proto | hm_1_month_cold | 49 | 154 s | 04:00:00 |
| 4 | 8159124 | mf | hm_1_month_cold | 99 | 141 s | 05:00:00 |
| 5 | 8159125 | feature_item_proto | hm_1_month_cold | 19 | 164 s | 02:00:00 |
| 6 | 8159126 | feature_item_proto_noid | hm_1_month_cold | 35 | 227 s | 04:00:00 |
| 7 | 8159127 | feature_user_proto | hm_1_month_cold | 33 | 132 s | 02:00:00 |
| 8 | 8159128 | feature_user_proto_noid | hm_1_month_cold | 59 | 130 s | 04:00:00 |
| 9 | 8159129 | lightfm_tags | hm_1_month_cold | 29 | 176 s | 03:00:00 |
| 10 | 8159130 | lightfm_tags_ids | hm_1_month_cold | 23 | 148 s | 02:00:00 |
| 11 | 8159134 | user_item_proto | ml-1m_cold | 47 | 83 s | 02:00:00 |
| 12 | 8159135 | user_proto | ml-1m_cold | 40 | 145 s | 03:00:00 |
| 13 | 8159136 | item_proto | ml-1m_cold | 75 | 73 s (lone, measured; 374 s shared, GPU-bound) | 03:00:00 |
| 14 | 8159137 | mf | ml-1m_cold | 17 | 96 s | 01:00:00 |
| 15 | 8159138 | feature_item_proto | ml-1m_cold | 55 | 176 s | 05:00:00 |
| 16 | 8159139 | feature_item_proto_noid | ml-1m_cold | 50 | 247 s | 06:00:00 |
| 17 | 8159140 | feature_user_proto | ml-1m_cold | 51 | 90 s | 03:00:00 |
| 18 | 8159141 | feature_user_proto_noid | ml-1m_cold | 44 | 151 s | 03:00:00 |
| 19 | 8159142 | lightfm_tags | ml-1m_cold | 39 | 110 s | 02:00:00 |
| 20 | 8159143 | lightfm_tags_ids | ml-1m_cold | 35 | 138 s | 03:00:00 |
