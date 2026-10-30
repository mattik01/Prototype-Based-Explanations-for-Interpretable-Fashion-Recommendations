# `_sep` controls — cold single-trial retrains, submitted 2026-10-30

Authorised by the user 2026-10-30 ("please do everything"). Completes wave 3
(`wave3_submissions_2026-10-28.md`) with the two ml-1m `_sep` controls whose warm hyperopts
landed 2026-10-28 (8165191 / 8165193, `sep_controls_submissions_2026-10-28.md`). One `/leo5-submit` per run.

- **Code:** fleet checkout `~/UIFProtoMF/ProtoMF` at **71c8394** — the warm runs' hash. Local HEAD 921d74a
  (pushed 2026-10-30) differs only in explanation code / docs; `utilities/cold_eval.py`, `run_combo.py`,
  `rec_sys/`, `feature_extraction/`, `confs/` identical (`git diff --stat 71c8394 921d74a`). Nothing pulled.
- **Common flags:** `--retrain-config /scratch/c7031336/protomf_results/<model>_ml-1m_s38210573_n100/config.json
  --profile production --gpu-per-trial 1.0 --cpus 4 --mem 24G --gpu-type any --num-workers 1
  --wandb-tag thesis-grade`, seed 38210573. Results → `<model>_ml-1m_cold_s38210573_n100/`.
- **Pre-submit check (login node, `/scratch/c7031336/smoke_runs/wave3_check/check.sh`, the wave-3 launcher):**
  2-epoch retrain + `utilities.cold_eval` on ml-1m_cold for both models — 2/2 retrain exit 0, 2/2 cold-eval
  exit 0; ids arm: ID-row imputation on 625 cold rows (FeatureEmbedding), noid arm has no ID-row step.
- **`--time`** = ceil_hour(min(100, 1.4 × E_src) × t_epoch + 0.25 h). E_src / shared t_epoch from the
  winners' Ray progress.csv (`wave3_esrc.py`). Sources are GPU-bound (99.2 / 97.6% avg GPU), so t_epoch
  = lone epoch measured in the check above (upper bound: the card was shared with the cold evals).
- **Cluster at submission (13:02):** 5 A30 slots of this shape placeable (n042/046/047/048/050).

| # | job id | model | cold dataset | E_src | t_epoch shared → lone | --time |
|---|---|---|---|---|---|---|
| 1 | 8181862 | lightfm_tags_hist_ids_sep | ml-1m_cold | 38 | 68.9 s → 36.1 s | 01:00:00 |
| 2 | 8181864 | lightfm_tags_hist_sep | ml-1m_cold | 74 | 269.2 s → 44.7 s | 02:00:00 |

After landing: cold eval `run_cold_evals.sh <model> ml-1m _n100` (fleet checkout, ≥ aa85862, mean-ID
imputation). Checkout lock on `~/UIFProtoMF/ProtoMF` until both have started.

## Outcome (2026-10-30)

Both COMPLETED, exit 0: 8181862 in 18:20 (32 epochs, best @22; source 38), 8181864 in 34:05 (43 epochs, best @33; source 74). Cold-evaluated on the login node with the scratch runner copy; rows in `cold_readout_n100_wave3_2026-10-30.md` and dc09 SC.8 §3. Fleet checkout lock released (no compared job pending).
