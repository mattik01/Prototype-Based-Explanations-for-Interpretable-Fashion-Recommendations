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
