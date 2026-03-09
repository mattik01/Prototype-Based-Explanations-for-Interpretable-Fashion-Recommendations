---
date: 2026-03-09
time: "16:41"
phase: 2
---
# Phase 2.1–2.2: Dataset preprocessing and configuration

Downloaded and preprocessed ML-1M (574K interactions, 6K users, 3K items) and Amazon2014 Video Games (132K interactions, 7K users, 14K items). LFM-2b skipped — dataset removed from official source (cp.jku.at) due to license issues; replication proceeds with 2/3 datasets. Configured `DATA_PATH` in `consts.py` and externalized `WANDB_API_KEY` to `api_keys/wandb_key.txt` (gitignored) with a new `api_keys/` folder convention for all secrets.

[[phase-2]] [[data]] [[setup]] [[decisions]]
