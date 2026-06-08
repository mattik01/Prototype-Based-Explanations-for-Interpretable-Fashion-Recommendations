# P5 — Concurrency Audit (cluster execution path)

Date: 2026-06-08. Scope: code that runs concurrently on LEO5 — multiple SLURM jobs
(one per combo) and multiple Ray Tune trials per job, sharing networked storage.

## Unifying anti-pattern
Three recurring root causes, present in combination:
1. **Wall-clock timestamps used as uniqueness keys** instead of a real job identity
   (`SLURM_JOB_ID` / PID). Collides when two things start in the same time-tick.
2. **Shared networked directories** written by all jobs (home/repo `Master/temp/...`,
   scratch `ray_results` / `wandb_logs`), so a non-unique name = cross-job collision.
3. **Peripheral subsystems are fatal to the core pipeline** — GPU telemetry and wandb
   can abort training or result-persistence.

All three only bite **under concurrency**, which is why single runs "work 80% of the time."

## HIGH — caused today's failures (confirmed)
- **H1 — GpuSampler session_id collision → silent result loss.**
  `gpu_sampler.py:93` `session_id = datetime.now().strftime("%Y%m%d_%H%M%S")` (second
  resolution). All jobs share `Master/temp/gpu_logs/` (run_combo.py:73) on networked home.
  Two jobs constructing a sampler in the same second (mf + acf today, both 09:36:43) →
  same `gpu_<ts>.csv`. First to finish renames it (`rename_log`, run_combo.py:304, inside
  the `finally`); the second's `os.rename` throws `FileNotFoundError`, which aborts the
  process **before `_save_combo_results` (run_combo.py:327)**. acf trained fine (5/5
  trials, HR@10 0.376) and its results were discarded by a GPU-log rename.
- **H2 — wandb.init unguarded + online network dependency → trial death.**
  `experiment_helper.py:101` `wandb.init(...)` has no try/except and runs online. Under
  concurrent-init load (item_proto + user_item_proto, 10 inits at once today), it exceeded
  the 90 s `init_timeout` → `CommError` → trial crashed. Observability killed the experiment.

## MEDIUM — latent (not yet triggered, but real)
- **M1 — Results dir not job-unique.** `run_combo.py:228`
  `folder_name = f"{model}_{dataset}_s{seed}"`. Two concurrent runs of the *same* combo
  (e.g. a rerun while the first is finishing) write to the same dir and clobber artifacts.
  No job-id/timestamp in the path.
- **M2 — GPU CSV rename target not job-unique.** `run_combo.py:303`
  `gpu_{model}_{dataset}_s{seed}.csv` — same key as M1; concurrent same-combo runs collide
  on the destination too.
- **M3 — Ray experiment name relies on microsecond timestamp.** `experiment_helper.py:223`
  `name=generate_id(prefix=group_name)`; `generate_id` (utils.py:38) uses
  `datetime.now()` to microsecond. `storage_path=RAY_RESULTS_DIR` is shared across all jobs.
  Practically safe (microsecond), but same anti-pattern — uniqueness by luck, not identity.

## LOW — smells / minor
- **L1 — gpu_sampler module globals.** `_prev_cpu_times`, `_prev_wall_time` (gpu_sampler.py
  40-41) are module-level mutables; in run_combo chaining multiple combos in one process they
  bleed across combos (minor CPU% inaccuracy at handoff).
- **L2 — wandb test/aggr run names not unique.** `experiment_helper.py:261` `_test` and `:300`
  `_aggr` names collide for duplicate combos (wandb dedups by run-id, so cosmetic).
- **L3 — false uniqueness in group_name.** `host_name = platform.node()[:2]` = `"n0"` for
  *every* LEO5 node (n0xx). `group_name = {model}_{dataset}_n0_{seed}` is not node-unique;
  harmless now (Ray registry is process-local) but a trap if ever used for a path.

## Fix themes (for the eventual hardening commit — P1–P3 + this)
1. **Identity, not timestamps:** key all per-run filenames/dirs on
   `SLURM_JOB_ID` (+ PID fallback). Fixes H1, M1, M2, M3.
2. **Per-job paths:** write telemetry CSV and Ray/wandb dirs under the job's own scratch
   space, not shared repo storage.
3. **Peripheral = non-fatal:** wrap sampler stop/rename/summary and `wandb.init` in
   try/except; `_save_combo_results` must run first/independently of telemetry. Fixes H1, H2.
4. **Honest exit codes:** nonzero if 0 trials complete or `test_metrics.json` not persisted.

## Decision 1 — W&B strategy (agreed 2026-06-08; default revised to online same day)
- **`wandb.init` / per-epoch `wandb.log` / `wandb.finish` all made non-fatal** in any mode
  (`safe_wandb_*` in `utilities/utils.py`; `trainer._report` broadened to `except Exception`).
  This is the load-bearing fix: W&B can no longer terminate a run regardless of mode.
- **Default `WANDB_MODE=online`** (live dashboard). Acceptable *because* of the wrapper —
  online is now termination-safe; the only residual cost is possible logging gaps under heavy
  concurrent load (a metrics-completeness issue, never a dead run).
- **`--wandb-mode offline` opt-in** to fully decouple training from the network (zero network
  calls during training; `wandb sync` runs as the last SLURM step). Recommended when network
  flakiness or full reproducibility matters more than a live dashboard.
- *(Initial decision was offline-default; revised to online-default per user once the non-fatal
  wrapper removed the termination risk that motivated offline.)*

## Status — fixes APPLIED 2026-06-08 (commit pending verification)
- **H1 fixed:** `GpuSampler.session_id` now keyed on `SLURM_JOB_ID`+PID; `rename_log`
  defensive; `run_combo` telemetry teardown wrapped non-fatal so `_save_combo_results`
  always runs; `GPU_LOG_DIR` per-job on scratch.
- **H2 fixed:** `safe_wandb_*` wrappers + offline default + post-run sync.
- **M1/M2 partially:** GPU CSV + W&B + status now per-job (`SLURM_JOB_ID`); results dir
  `{model}_{dataset}_s{seed}` still not job-unique — acceptable (a *rerun* of the same combo
  is intentional overwrite), revisit only if we ever run identical combos concurrently.
- **Honest exit:** `run_combo.py` writes a status sentinel (`RUN_STATUS_FILE`) = OK only on
  >0 completed trials AND persisted `test_metrics.json`; the SLURM wrapper now captures the
  python exit code and treats the sentinel as authoritative (the generated job script
  previously ended on `rm -rf`, masking every failure as exit 0 → `COMPLETED`).
- **M3/L1–L3:** left as-is (microsecond Ray id is safe-enough; the rest cosmetic).

## Verification gate
After hardening, rerun the 5-model smoke sweep **twice** (cluster — local Ray is too old to
import `experiment_helper`); require all 5 to persist results both times before trusting the
pipeline for baselines.
