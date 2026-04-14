Help the user submit a job to the LEO5 HPC cluster. Follow these steps exactly:

## Step 1: Check cluster status
Run `python Master/scripts/leo5_status.py` to get current free resources.
**If SSH fails**, tell the user they likely need to connect to VPN first and stop.

## Step 2: Show what's missing and ask what to run
Read `Master/docs/replication_report.md`. Identify combos with empty Test HR@10 (need running) and combos marked `*` (incomplete, may want rerun). Note that `lfm2b-1mon` data is NOT on LEO5 yet.

Present a compact list of missing combos, then ask:
> "What would you like to run? Pick from above or type a custom combo (e.g. `mf ml-1m`, `item_proto hm_3_month 3seed`, `all missing amazon2014`)."

Accept freeform input — the user may type exact combos, groups, or custom requests.

## Step 3: Suggest SLURM configuration
Based on the chosen combo(s) and current cluster state, suggest `sbatch` command(s). Use these rules:

**GPU assignment (from VRAM benchmarks in the report):**
- `mf`, `acf`, `user_proto` → A30 (lightweight, <5 GB VRAM)
- `item_proto`, `user_item_proto` → A100 (10-13 GB VRAM for 16 concurrent trials)
- If the suggested GPU type has 0 free, suggest the next best alternative or note the wait

**Time limits (with 2x safety margin for LEO5):**
- `mf` / `acf` / `user_proto` on amazon2014/ml-1m → `1-00:00:00` (1 day)
- `item_proto` / `user_item_proto` on amazon2014/ml-1m → `1-00:00:00`
- Any model on hm_3_month → `2-00:00:00` (2 days)
- Any model on hm_full → `5-00:00:00` (5 days), heavy models `7-00:00:00`

**Seeds:** Default single seed `38210573`. For 3-seed runs: `38210573 9491758 2931009`.

**Format:**
```
sbatch --gres=gpu:a30:1 --time=1-00:00:00 --job-name=pmf_MODEL_DATASET_sSEED slurm/run_combo.sbatch MODEL DATASET SEED
```

Present the suggestion(s) and **wait for the user to confirm or adjust** before executing.

## Step 4: Execute (only after user confirms)
Run the confirmed `sbatch` command(s) via SSH:
```
ssh leo5 "cd /home/c703/c7031336/UIFProtoMF/ProtoMF && sbatch ..."
```
Then run `python Master/scripts/leo5_status.py` to show the job in the queue.
