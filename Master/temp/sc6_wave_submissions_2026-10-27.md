# dc09 SC.6 wave — 32 jobs from one sync, submitted 2026-10-27

Authorised at the dc09 SC.6 gate (user, 2026-10-27: "yes pelase thank you"; plan and sizing in
`Master/docs/scrutiny/sc09_fufi_double_tie_merge.md` §SC.6). Replaces the cancelled wave 2
(`wave2_submissions_2026-10-27.md`; same 20 retrains, new `--time`).

- **Code:** fleet checkout `~/UIFProtoMF/ProtoMF` fast-forwarded once, `c0c6ac1 → 71c8394`
  (feat/scrutiny), nothing pending at the pull. One commit hash for all 32 runs. It carries the
  exact fast negative sampler (`aa7184d`; bit-identical negatives, dc_checks/s0 t15), the
  two-sided control `lightfm_tags_hist` (`e18e911`; `_sep` twins registered, not submitted), the
  SC.4 factory guards, the results-folder guard (`c0c6ac1`) and the ml-1m cold eval (`aa85862`).
- **Common flags:** `--profile production --num-workers 1 --gpu-type any --wandb-tag
  thesis-grade`, seed 38210573, one wrapper call (`slurm/run_combo.slurm`) per run.
- **Pre-checks:** data dirs complete for hm_1_month, ml-1m, hm_1_month_cold, ml-1m_cold; all 20
  retrain source configs present; none of the 32 target folders existed. Dry-runs of the three
  line shapes (fUfI ml-1m, control hm at c=7, a retrain) matched the plan.
- **Fractional GPU:** Ray stores resources in 1/10000 units, so `0.1429 × 7` = 1.0003 GPU would
  run only 6 trials. The c=7 and c=6 lines use `0.1428` / `0.1666`; the wrapper confirmed
  "Concurrent trials: 7" / "6". (The `/leo5-submit` class table's `0.1667` for c=6 has the
  same flaw: `floor(1/0.1667)` = 5; fixed in the skill the same day.)
- **Cluster at submission (13:52):** nothing placeable; A100 ×6 idle but their nodes RAM-blocked
  (7–12 GB unallocated); A30: 2 free within 18 h, 4 within 2d06h, 36 within 3 d. Earliest
  realistic start: a couple of jobs tomorrow morning (Wed 10-28), most by Fri 10-30.
- **Checkout lock** on `~/UIFProtoMF/ProtoMF` until all 32 have started. Smokes and cold evals
  run from the dev worktree `~/UIFProtoMF/ProtoMF_dev` (detached; move it to 71c8394 first).

## A — dc09 fUfI hyperopts (100 trials)

| # | job id | model | dataset | conc (`--gpu-per-trial`) | `--cpus` / `--mem` | est. wall | `--time` |
|---|---|---|---|---|---|---|---|
| 1 | 8160164 | feature_user_item_proto | ml-1m | 4 (0.25) | 5 / 28G | ~15–17 h | 1-12:00:00 |
| 2 | 8160165 | feature_user_item_proto_noid | ml-1m | 4 (0.25) | 5 / 28G | ~15–17 h | 1-12:00:00 |
| 3 | 8160166 | feature_user_item_proto_sep | ml-1m | 4 (0.25) | 5 / 28G | ~15–17 h | 1-12:00:00 |
| 4 | 8160167 | feature_user_item_proto_noid_sep | ml-1m | 4 (0.25) | 5 / 28G | ~15–17 h | 1-12:00:00 |
| 5 | 8160168 | feature_user_item_proto | hm_1_month | 5 (0.2) | 6 / 36G | ~12 h | 1-12:00:00 |
| 6 | 8160169 | feature_user_item_proto_noid | hm_1_month | 5 (0.2) | 6 / 36G | ~12 h | 1-12:00:00 |
| 7 | 8160170 | feature_user_item_proto_sep | hm_1_month | 5 (0.2) | 6 / 36G | ~12 h | 1-12:00:00 |
| 8 | 8160171 | feature_user_item_proto_noid_sep | hm_1_month | 5 (0.2) | 6 / 36G | ~12 h | 1-12:00:00 |

## B — two-sided decoupled control (100 trials)

| # | job id | model | dataset | conc (`--gpu-per-trial`) | `--cpus` / `--mem` | est. wall | `--time` |
|---|---|---|---|---|---|---|---|
| 9 | 8160173 | lightfm_tags_hist_ids | ml-1m | 6 (0.1666) | 7 / 36G | ~8 h | 1-00:00:00 |
| 10 | 8160174 | lightfm_tags_hist | ml-1m | 6 (0.1666) | 7 / 36G | ~8 h | 1-00:00:00 |
| 11 | 8160175 | lightfm_tags_hist_ids | hm_1_month | 7 (0.1428) | 8 / 40G | ~4.5 h | 12:00:00 |
| 12 | 8160176 | lightfm_tags_hist | hm_1_month | 7 (0.1428) | 8 / 40G | ~4.5 h | 12:00:00 |

Results → `/scratch/c7031336/protomf_results/<model>_<dataset>_s38210573_n100/`.

## C — cold single-trial retrains of the 100-trial configs

All `--retrain-config /scratch/c7031336/protomf_results/<model>_<warm dataset>_s38210573_n100/config.json
--gpu-per-trial 1.0 --cpus 4 --mem 24G`. `--time` = ceil_hour(min(100, 1.4 × E_src) × t_epoch ×
class ratio × 1.15 + 0.25 h) (E_src / t_epoch in the wave-2 record; class ratio = the SC.6
measured new/old epoch time). Results → `<model>_<cold dataset>_s38210573_n100/`.

| # | job id | model | cold dataset | `--time` |
|---|---|---|---|---|
| 13 | 8160178 | user_item_proto | hm_1_month_cold | 04:00:00 |
| 14 | 8160179 | user_proto | hm_1_month_cold | 03:00:00 |
| 15 | 8160180 | item_proto | hm_1_month_cold | 03:00:00 |
| 16 | 8160181 | mf | hm_1_month_cold | 03:00:00 |
| 17 | 8160182 | feature_item_proto | hm_1_month_cold | 02:00:00 |
| 18 | 8160183 | feature_item_proto_noid | hm_1_month_cold | 03:00:00 |
| 19 | 8160184 | feature_user_proto | hm_1_month_cold | 02:00:00 |
| 20 | 8160185 | feature_user_proto_noid | hm_1_month_cold | 02:00:00 |
| 21 | 8160186 | lightfm_tags | hm_1_month_cold | 02:00:00 |
| 22 | 8160187 | lightfm_tags_ids | hm_1_month_cold | 01:00:00 |
| 23 | 8160218 | user_item_proto | ml-1m_cold | 02:00:00 |
| 24 | 8160219 | user_proto | ml-1m_cold | 03:00:00 |
| 25 | 8160220 | item_proto | ml-1m_cold | 03:00:00 |
| 26 | 8160221 | mf | ml-1m_cold | 01:00:00 |
| 27 | 8160222 | feature_item_proto | ml-1m_cold | 05:00:00 |
| 28 | 8160224 | feature_item_proto_noid | ml-1m_cold | 06:00:00 |
| 29 | 8160225 | feature_user_proto | ml-1m_cold | 03:00:00 |
| 30 | 8160226 | feature_user_proto_noid | ml-1m_cold | 03:00:00 |
| 31 | 8160227 | lightfm_tags | ml-1m_cold | 03:00:00 |
| 32 | 8160228 | lightfm_tags_ids | ml-1m_cold | 03:00:00 |

Totals ≈ 155 GPU-hours (A ≈ 108, B ≈ 25, C ≤ 25), ≈ 1.9k billed thread-hours.

## After they land
- **C:** cold evals on the login node (`utilities/cold_eval.py`, mean-ID imputation, code ≥ aa85862);
  quote the cold-vs-cold popularity row as chance (0.10, F-S0-17).
- **A + B:** warm readout by the §4 rule of SC.6, fUfI − UI = (fUfI − control) + (control − mf) − (UI − mf),
  paired by ID setting. If a `_sep` arm is the best fUfI arm on a dataset, submit
  `lightfm_tags_hist(_ids)_sep` on that dataset with group-B sizing.
- **Wave 3:** cold retrains of the dc09 arms and the controls from their selected configs.
- Logs: `/scratch/c7031336/protomf_logs/<jobname>_<jobid>.out`; exit 0 ⇔ run dir complete.
