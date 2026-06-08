Suggest a SLURM job configuration for a model×dataset combo on LEO5, optimized for quick scheduling based on current cluster load. Follow this reasoning pipeline exactly.

> ⚠️ **CAVEAT — code sync before submit (non-negotiable).** The cluster runs whatever code its
> checkout currently has. **Before submitting, verify the cluster is on the same commit as local
> HEAD** (laptop is the source of truth). A stale cluster checkout silently runs old code and wastes
> the slot. The full verification + remediation lives in STEP 7; never skip it. (Code syncs via git;
> **data does not** — dataset CSVs are gitignored and must be scp'd separately, so also confirm the
> target dataset dir actually exists on the cluster before submitting.)

---

## STEP 0: CLUSTER STATE

If `/leo5-load` output is NOT already in this conversation context, ask the user to run `/leo5-load` first, then stop.

Read `Master/docs/replication_report.md` — you need the GPU Benchmark table for VRAM, RAM, and wall-time estimates.

## STEP 0.5: IDENTIFY CANDIDATE NODES

From the `/leo5-load` output, filter to **candidate nodes** — GPU nodes with:
- At least 1 free GPU
- At least 6 free CPUs
- At least 8 GB free memory

If no candidates exist, tell the user and suggest waiting or trying off-peak hours. Stop.

## STEP 1: IDENTIFY COMBO + EXPERIMENT PARAMS

Extract from conversation context: **model**, **dataset**, **seed**.
If any is ambiguous → ask.

### 1a. Derive the search-budget profile

`run_combo.py` exposes `--profile`, which bundles `num_samples / n_epochs / patience / grace_period`. Work out which one this run wants from context, then **state your choice and the resolved values** so the user can correct:

| Profile | num_samples / n_epochs / patience / grace | When to use |
|---|---|---|
| `production` | 100 / 100 / 10 / 4 | Final, **paper-comparable / reportable** numbers. Default when in doubt for a result that goes in the thesis. |
| `dev` | 30 / 60 / 7 / 4 | Iterating on code/features, exploring a new combo, sanity-checking a change. ~4× faster, lands ~90% of the full-100 best val (back-tested on 729 trials). **Not** for final reporting. |
| `smoke` | 5 / 15 / 10 / 4 | Just confirming the pipeline runs end-to-end on this combo. |

Derivation cues: words like *final / paper / report / baseline / replication* → `production`. *Try / test / iterate / quick / feature / debug / does it run* → `dev` or `smoke`. If genuinely ambiguous → ask, defaulting to `production` for safety.

**Caveat to surface:** `dev` under-serves slow-converging combos — `user_proto` and `item_proto` on `amazon2014` (winners peak late / near the epoch cap). If the target is one of those and the result matters, recommend `production` or `dev` with `--num-samples 50 --n-epochs 100`.

### 1b. Confirm the rest

- **profile** — resolved above; any explicit flag (`--num-samples`, `--n-epochs`, `--patience`, `--grace-period`) overrides it.
- **ASHA** — on/off (default: on)
- **optimizing_metric** — default `hit_ratio@10`; ask if not clear from context
- **wandb_tags** — default `[]`; `dev`/`smoke` profiles auto-tag runs with the profile name, so they stay out of paper-comparable analysis.

## STEP 2: RANK CANDIDATE NODES

Rank candidates by:
1. Free CPU slots (descending)
2. GPU size (descending) — A100 (80 GB) > A40 (48 GB) > A30 (24 GB)

Rationale: more free CPUs → higher concurrency potential → more VRAM needed → bigger GPU helps.

For each candidate node (best first), evaluate Steps 3–4. Stop after finding 2–3 viable configs.

## STEP 3: DETERMINE CONCURRENCY FOR THIS NODE

Inputs: node's GPU type + VRAM, free CPUs, free memory, and the model's VRAM profile from the replication report GPU Benchmark table.

### 3a. VRAM-driven max concurrency

- Look up the model's **SAFE-ISH VRAM (MB)** per trial from the replication report.
- If no benchmark exists for this exact model×dataset, estimate from the closest available entry (same model on a different dataset, scaled by interaction count ratio).
- `max_concurrent_vram = floor(gpu_total_vram_mb / safe_vram_per_trial)`
- Apply 80% safety margin: `max_concurrent_vram = floor(0.8 × max_concurrent_vram)`
- Cap at `num_samples`.

### 3b. CPU-driven max concurrency

Three modes (DataLoader workers per trial). `cpu_per_trial` is always 1.

| Mode | num_workers | Real cores/trial | Formula |
|------|:-----------:|:----------------:|---------|
| Lean | 0 | 1 | `floor(free_cpus / 1)` |
| **Standard** (default) | 1 | 2 | `floor(free_cpus / 2)` |
| Full | 2 | 3 | `floor(free_cpus / 3)` |

Choose mode:
1. Start with **Standard**. Compute its max concurrency.
2. Check **Full**: if Full concurrency ≥ 80% of VRAM-driven max → upgrade to Full (more data prefetch is worth the extra CPU).
3. If Standard concurrency is already below VRAM-driven max → fall back to **Lean** to recover concurrency.

### 3c. Final concurrency

`final_concurrency = min(max_concurrent_vram, max_concurrent_cpu_for_chosen_mode, num_samples)`

Derive:
- `gpu_per_trial = 1 / final_concurrency`
- `cpus_to_request = final_concurrency × cores_per_trial`

If any estimation is uncertain, ask the user.

## STEP 4: MEMORY CHECK

- Look up **Avg RAM (MB)** from the replication report for the model. Scale proportionally to `final_concurrency / benchmark_concurrency`.
- If no benchmark: estimate `~2 GB base + (dataset_interactions × 50 bytes) + (1 GB × concurrency)`.
- Check: estimated RAM ≤ node's free memory?
  - **Yes** → this node is viable, record the config.
  - **No** → reduce concurrency by 1, go back to Step 3c. If concurrency drops below 2 → skip this node.

## STEP 5: DETERMINE TIME

**CRITICAL — Getting the time wrong wastes cluster budget (too long blocks the slot) or kills jobs mid-run (too short). Show your math explicitly so the user can verify.**

### 5a. Find the closest benchmark

Look up wall time from the GPU Benchmark table. Use the **Machine** column to match hardware. If no direct benchmark exists for this model×dataset, find the closest available entry (same model on a different dataset, or same dataset on a similar model) and state the extrapolation.

### 5b. Scale the benchmark wall time

Apply **all** of the following scaling factors — none are optional:

```
estimated_time = benchmark_wall_time
    × (num_samples / benchmark_num_samples)          # trial count (benchmarks are production = 100)
    × epoch_cap_factor                                # profile n_epochs/patience vs production: dev ≈ 0.85×, smoke ≈ 0.5×, production 1.0×
    × (benchmark_concurrency / my_concurrency)       # IMPORTANT: fewer concurrent trials = proportionally longer wall time
    × machine_factor                                  # gpu-machine (RTX 4070 Ti) → A30 ≈ 1.0×, → A100 ≈ 0.6×
    × dataset_factor                                  # ratio of interaction counts if different dataset
    × asha_factor                                     # 1.0 if same ASHA setting, ~2× if turning ASHA off vs benchmark that had it on
```

**Profile reminder:** all replication-report benchmarks are `production` (100 samples / 100 epochs / ASHA on). For `dev`, `num_samples/benchmark_num_samples = 30/100 = 0.30×` and `epoch_cap_factor ≈ 0.85×` (lower n_epochs + patience trims the surviving trials) → roughly **0.25× of the production wall time**. For `smoke`, ~0.04× (but never blindly cap large datasets at 1h — see below).

If the benchmark model differs from the target model, also apply a **model scaling factor** derived from known ratios on other datasets (e.g., acf/mf ratio on amazon2014 and ml-1m).

### 5c. Safety margin and rounding

Apply **2× safety margin** on top of the estimate.

Round up to clean SLURM values: `00:30:00`, `1:00:00`, `2:00:00`, `4:00:00`, `8:00:00`, `12:00:00`, `1-00:00:00`, `2-00:00:00`, `3-00:00:00`, `5-00:00:00`, `7-00:00:00`.

**Do NOT blindly cap smoke tests at 1h.** Large datasets (hm_3_month, hm_full) have long per-trial times regardless of num_samples. The math must drive the estimate.

## STEP 6: PRESENT OPTIONS

Present 2–3 options to the user:

**Option A (greedy):** Targets the single best node.
- Maximum concurrency, fastest completion.
- Show the full `run_combo.slurm` command with all parameters.
- Note: if that node is taken before scheduling, job waits.

**Option B (balanced):** Slightly relaxed config (fewer CPUs/memory) that fits on **N nodes**.
- State how many candidate nodes could serve this config.
- Pro: more likely to schedule quickly via backfill.

**Option C (conservative):** Only if the gap between A and B is large. Even more relaxed.

For each option, show:
- The resolved **profile** (and any explicit overrides).
- Estimated wall time, concurrency, GPU type, num_workers mode, CPUs, memory.
- The full `run_combo.slurm` command ready to copy — include `--profile <profile>` (plus any override flags). Omit `--profile` only for an explicit `production` run, since it's the default.

**Wait for user to confirm or adjust before executing.**

## STEP 7: EXECUTE

Only after user confirms:

1. **Verify code sync (cluster checkout == local HEAD).** Do not trust "I pushed earlier" — check.
   1. Local: `git status --short` (uncommitted?) and `git rev-parse HEAD`.
   2. Cluster: `ssh leo5 "cd ~/UIFProtoMF/ProtoMF && git rev-parse HEAD"`.
   3. Compare:
      - Uncommitted local changes affecting cluster code → ask to commit+push locally first.
      - Local HEAD ≠ cluster HEAD → ask to push locally (if unpushed), then `git pull` on LEO5 (per
        CLAUDE.md cluster sync rule). Laptop is the source of truth; never edit on the cluster.
      - Hashes match and tree clean → proceed.
2. **Verify data presence (separate channel — git will NOT have carried it).** Datasets live on
   **scratch**, not in the repo: jobs read `PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data` (home
   has a 5 GB quota — data must not live there). Confirm the dataset dir exists:
   `ssh leo5 "ls /scratch/c7031336/protomf_data/<dataset>/"`. If missing → stop and scp it to scratch
   (`scp -r data/<dataset> leo5:/scratch/c7031336/protomf_data/`) before submitting; a job against a
   missing dataset dir fails immediately.
3. Run via SSH:
   ```
   ssh leo5 "cd ~/UIFProtoMF/ProtoMF && bash slurm/run_combo.slurm [args...]"
   ```
4. Run `/leo5-load` to show the job in the queue.

---

## Notes

- Surface your reasoning transparently at each step so the user can correct estimates.
- At any point during the pipeline, you may ask the user for input on uncertain decisions.
- The replication report is the primary source for VRAM/RAM/time estimates. When it lacks data for a combo, state your extrapolation logic explicitly.
