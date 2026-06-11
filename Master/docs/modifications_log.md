# Modifications Log

Tracks all original repo files modified from their upstream state.

## CLAUDE.md
- Added RULES section: temp file rule, protocol entry prompting rule, modifications log rule
- Added Branch Strategy, Naming Conventions, Reserved ft_type Values, Results Storage, Modification Map, H&M Data sections

## utilities/consts.py
- Changed `DATA_PATH` from hardcoded absolute path to relative `os.path.join` (machine-independent)
- Replaced hardcoded `WANDB_API_KEY` with file-based loading from `Master/sensitive/api_keys/wandb_key.txt`
- Added `import os`

## .gitignore
- Added (already existed with `__pycache__` rules; no new entries needed at root level)
- Added ignore rule for downloaded literature PDFs (`Master/literature/papers/**/*.pdf`, ~182 MB, re-fetchable)
- Added ignore rule for mirrored run checkpoints (`Master/experiments/results/**/*.pth`, ~326 MB; archived on LEO5 + local disk — run metadata stays in git, the pre-existing tracked canonical checkpoint is unaffected)

## confs/hyper_params.py
- Added `debug_hyper_params` config with fixed tiny values for fast CPU smoke tests (embedding_dim=8, n_prototypes=3, 3 epochs, num_samples=1)

## start.py
- Added `debug` model choice that uses `debug_hyper_params`
- Fixed typo: `--multipl2erfve` → `--multiple` (accidental corruption of argparse flag name)
- Added `hm_full` and `hm_3_month` to dataset choices
- Added `hm_1_month` to dataset choices (V1 dev/judgment testbed)

## experiment_helper.py
- `start_hyper()` now respects optional `num_samples` key in config dict (falls back to global `NUM_SAMPLES`)
- Removed legacy `_metric/` prefix from `metric` and `metric_name` — newer Ray Tune no longer auto-prefixes reported metrics, causing silent hangs on metric validation
- Enhanced `WandbLoggerCallback` with console capture (`console: auto`), and `group` for grouping trials by model/dataset/seed

## experiment_helper.py (continued)
- Moved W&B metadata (model, dataset, group, tags) from environment variables to config dict (`_wandb_*` keys) — fixes incorrect tagging when `run_combo.py` chains multiple combos in the same process

## rec_sys/trainer.py
- Added `use_ray` flag (default=True) to `Trainer.__init__()` to allow standalone PyTorch training without Ray Tune
- Extracted `_report()` method that conditionally uses `ray.train.report()` or prints metrics locally
- `run()` now saves checkpoints via plain `torch.save()` when `use_ray=False`
- Ray import moved from top-level to inside `_report()` (lazy) so the module loads without Ray when not needed
- Added timing metrics to `run()`: epoch duration, elapsed time, ETA, patience counter, LR, and epoch number — all reported to W&B per epoch

## README.md
- Replaced original paper README with thesis-specific README that credits the upstream repo, describes thesis goals, and documents the extended repository structure

## Master/scripts/gpu_graph.py
- Removed CSV logging logic (csv import, LOG_DIR, csv_file, csv_writer, all CSV writes); now purely a visual terminal tool with no file I/O

## utilities/explanations_utils.py
- Removed deprecated `square_distances=True` parameter from TSNE call (removed in scikit-learn 1.6)
- `tsne_plot` now infers save format from the file extension and saves at dpi=200 with tight bbox (was hardcoded `format='pdf'`) — explanation figures are now emitted as PNG
- `tsne_plot` gained `prototype_labels` (annotate prototype markers with derived names) and a `point_styles` seam (per-point feature-encoded glyphs; default unchanged uniform dots)
- `weight_visualization` gained `u_proto_labels`/`i_proto_labels` (prototype-name captions), a plain-language symbol legend for s/t̂/u* etc., and a minimum panel-width floor so the figure with fewer prototypes no longer collapses to an unreadable sliver
- `tsne_plot` prototype-label annotation now repels overlapping names via `adjustText` (optional dependency; graceful fallback to fixed-offset annotation when absent) with thin leader lines back to markers

## utilities/consts.py (LEO5 migration)
- `DATA_PATH` now supports `PROTOMF_DATA_PATH` env var override (falls back to relative path)
- `WANDB_API_KEY` loading wrapped in try/except with `WANDB_API_KEY` env var fallback

## experiment_helper.py (LEO5 migration)
- Added explicit `ray.init()` with `RAY_TMPDIR` env var support before `tune.run()`
- Added `storage_path` to `tune.run()` via `RAY_RESULTS_DIR` env var (directs results to scratch on HPC)

## rec_sys/tester.py (LEO5 migration)
- Added `weights_only=True` to `torch.load()` for PyTorch 2.x compatibility

## utilities/consts.py (experiment results)
- Added `EXPERIMENT_RESULTS_PATH` with `PROTOMF_RESULTS_PATH` env var override (defaults to `Master/experiments/results/`)

## experiment_helper.py (experiment results)
- `start_hyper()` now returns a rich dict (test_metrics, val_metrics, best_config, checkpoint path, trial stats) instead of just test metrics
- `start_multiple_hyper()` updated to use `result['test_metrics']` from new return format

## Master/scripts/run_combo.py (experiment results)
- Added `_save_combo_results()`: creates structured results folder per combo with checkpoint, config, metrics JSONs, hardware log, and summary.md
- Added `_generate_summary()`: produces pre-formatted replication report table rows
- `run_single_combo()` now saves results automatically after each experiment

## confs/hyper_params.py (per-execution resource config)
- Removed `num_samples: 30` override from `proto_double_tie_chose_original_hyper_params` — all models now use the same default (100)

## experiment_helper.py (per-execution resource config)
- `start_hyper()` now accepts optional `resource_cfg` dict for per-execution overrides of gpu_per_trial, cpu_per_trial, num_samples, num_workers, grace_period, patience, optimizing_metric, and n_epochs
- `load_data()` now accepts `num_workers` parameter instead of reading global constant
- Execution-level overrides injected into config dict with `_` prefix keys for passthrough to trials

## rec_sys/trainer.py (per-execution resource config)
- `Trainer` now reads `_max_patience` and `_optimizing_metric` from config namespace (falls back to consts)

## Master/scripts/run_combo.py (per-execution resource config)
- Added CLI args: `--num-samples`, `--gpu-per-trial`, `--cpu-per-trial`, `--num-workers`, `--n-epochs`, `--grace-period`, `--patience`, `--wandb-mode`, `--optimizing-metric`
- Builds `resource_cfg` dict from CLI and passes through to `start_hyper()`

## experiment_helper.py (ASHA toggle + W&B tags)
- ASHA scheduler now conditional: `ASHAScheduler(grace_period) if asha else None`
- Extra W&B tags from `resource_cfg['wandb_tags']` merged into train and test wandb.init calls

## Master/scripts/run_combo.py (ASHA toggle + W&B tags)
- Added `--no-asha` flag to disable ASHA early stopping scheduler
- Added `--wandb-tag` repeatable flag for extra W&B tags

## slurm/run_combo.slurm (ASHA toggle + W&B tags + CPU override)
- Added `--no-asha`, `--wandb-tag`, `--cpus` flags with passthrough

## Master/docs/replication_report.md (Machine column)
- Added Machine column to GPU Benchmark table for time estimation across devices

## Master/scripts/gpu_sampler.py (process-scoped metrics)
- `query_system()` now measures only the current process tree (RSS + CPU% of self + children) instead of node-wide `psutil.virtual_memory()` / `psutil.cpu_percent()` — fixes inflated RAM/CPU numbers on shared HPC nodes

## rec_sys/protomf_dataset.py (PyTorch 2.x compat)
- `T_co` import: fallback to `typing.Any` when `torch.utils.data.dataset.T_co` is removed (PyTorch ≥ 2.5)

## Master/scripts/run_combo.py (explanations auto-invocation)
- Auto-invokes `utilities.explanations.pipeline.run_explanations_pipeline(results_dir)` after `_save_combo_results()` for explainable models (`item_proto`, `user_proto`, `user_item_proto`); wrapped in try/except so failures do not invalidate the training run
- Added `--skip-explanations` CLI flag (default off) to disable the auto-invocation

## utilities/consts.py (early-stopping min_delta)
- Added `MIN_DELTA_DEFAULTS` dict (per-metric defaults, 1e-4 for all `hit_ratio@K` / `ndcg@K`) and `MIN_DELTA_FALLBACK` (1e-4) for unknown metrics

## rec_sys/trainer.py (early-stopping min_delta)
- `Trainer` now reads `_min_delta` from config (default 0.0 → preserves original "any improvement" behavior when unset)
- Early-stopping comparison changed from `curr > best` to `curr > best + min_delta`; logs `min_delta` at init

## experiment_helper.py (early-stopping min_delta)
- `start_hyper()` resolves `min_delta` from `resource_cfg`; when omitted, falls back to `MIN_DELTA_DEFAULTS[optimizing_metric]` then `MIN_DELTA_FALLBACK`
- Injects `_min_delta` into config for trial passthrough; logs the resolved value

## utilities/utils.py (W&B hardening)
- Added `safe_wandb_init`, `safe_wandb_finish`, `safe_wandb_login` — observability must never abort a run; init/finish/login failures are caught and logged, training continues without W&B

## experiment_helper.py (concurrency/W&B hardening)
- All `wandb.init`/`finish`/`login` calls (train, test, aggregate phases) routed through `safe_wandb_*` so a W&B failure can never terminate a trial
- `start_hyper()` now raises a clear `RuntimeError` if 0 trials completed (all errored), instead of failing cryptically in `get_best_checkpoint` — makes total failure explicit

## rec_sys/trainer.py (W&B hardening)
- `_report()` W&B log guard broadened from `except ImportError` to `except Exception` — a mid-training online `wandb.log` network/comm error can no longer abort training

## Master/docs/replication_report.md (restructure)
- Restructured around LEO5 as the primary machine: scoped to amazon2014/ml-1m/hm_1_month, added a paper-free Run Results table, merged cluster runs into one hardware table, dropped the Smoke Test section, and parked out-of-scope rows (hm_3_month/hm_full/lfm2b) with their numbers in a bottom Appendix
