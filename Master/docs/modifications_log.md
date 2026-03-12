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

## experiment_helper.py
- `start_hyper()` now respects optional `num_samples` key in config dict (falls back to global `NUM_SAMPLES`)

## README.md
- Replaced original paper README with thesis-specific README that credits the upstream repo, describes thesis goals, and documents the extended repository structure
