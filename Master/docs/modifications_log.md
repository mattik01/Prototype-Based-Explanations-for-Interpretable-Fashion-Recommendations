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

## confs/hyper_params.py
- Added `debug_hyper_params` config with fixed tiny values for fast CPU smoke tests (embedding_dim=8, n_prototypes=3, 3 epochs, num_samples=1)

## start.py
- Added `debug` model choice that uses `debug_hyper_params`
- Fixed typo: `--multipl2erfve` → `--multiple` (accidental corruption of argparse flag name)
- Added `hm_full` and `hm_3_month` to dataset choices

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
