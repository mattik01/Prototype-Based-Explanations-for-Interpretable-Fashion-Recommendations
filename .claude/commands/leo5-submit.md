Suggest a SLURM job configuration for a model×dataset combo on LEO5, optimized for quick scheduling based on current cluster load. Follow this reasoning pipeline exactly.

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

Always confirm these (suggest defaults, but use values already stated in context):
- **num_samples** — 100 for production, 5–10 for smoke tests
- **ASHA** — on/off (default: on)
- **optimizing_metric** — default `hit_ratio@10`; ask if not clear from context
- **wandb_tags** — default `[]`; smoke tests should get `["smoke"]`

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
- `cpus_to_request = final_concurrency × cores_per_trial + overhead`
- Overhead: +2 if `free_cpus - (final_concurrency × cores_per_trial) >= 2`, else 0

If any estimation is uncertain, ask the user.

## STEP 4: MEMORY CHECK

- Look up **Avg RAM (MB)** from the replication report for the model. Scale proportionally to `final_concurrency / benchmark_concurrency`.
- If no benchmark: estimate `~2 GB base + (dataset_interactions × 50 bytes) + (1 GB × concurrency)`.
- Check: estimated RAM ≤ node's free memory?
  - **Yes** → this node is viable, record the config.
  - **No** → reduce concurrency by 1, go back to Step 3c. If concurrency drops below 2 → skip this node.

## STEP 5: DETERMINE TIME

Look up wall time from the GPU Benchmark table. Use the **Machine** column to match hardware.

Scale the benchmark wall time by:
- **Dataset size**: ratio of interaction counts if no direct benchmark exists.
- **Machine**: `gpu-machine` (RTX 4070 Ti) → `leo5 A30` ≈ 1:1; → `leo5 A100` ≈ 0.6× (A100 is ~1.5× faster).
- **ASHA**: if `asha=off` and benchmark used `asha=on` → multiply by ~2×.
- **num_samples**: scale linearly relative to benchmark trial count.
- **Concurrency**: if fewer concurrent trials than benchmark → scale up proportionally.

Apply **2× safety margin** on top.

Smoke tests (`num_samples ≤ 10`): cap at `1:00:00`.

Round up to clean SLURM values: `00:30:00`, `1-00:00:00`, `2-00:00:00`, `3-00:00:00`, `5-00:00:00`, `7-00:00:00`.

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
- Estimated wall time, concurrency, GPU type, num_workers mode, CPUs, memory.
- The full `run_combo.slurm` command ready to copy.

**Wait for user to confirm or adjust before executing.**

## STEP 7: EXECUTE

Only after user confirms:

1. Check if local repo has uncommitted changes affecting cluster code.
   - If yes → ask to commit+push locally, then `git pull` on LEO5 (per CLAUDE.md cluster sync rule).
   - If no → proceed.
2. Run via SSH:
   ```
   ssh leo5 "cd ~/UIFProtoMF/ProtoMF && bash slurm/run_combo.slurm [args...]"
   ```
3. Run `/leo5-load` to show the job in the queue.

---

## Notes

- Surface your reasoning transparently at each step so the user can correct estimates.
- At any point during the pipeline, you may ask the user for input on uncertain decisions.
- The replication report is the primary source for VRAM/RAM/time estimates. When it lacks data for a combo, state your extrapolation logic explicitly.
