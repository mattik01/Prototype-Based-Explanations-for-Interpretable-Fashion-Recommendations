Size and submit a `run_combo.py` job (model × dataset) on LEO5. The cluster-state check is
built in (STEP 0) — never ask the user to run `/leo5-load` first. Follow the steps in order
and show the numbers behind every recommendation so the user can correct them.

Rewritten 2026-10-26 at the wave-1 (100-trial fleet) sitting from measured data: SLURM
accounting (`sacct`), the run dirs' `summary.md` / `hardware_summary.json`, the job logs'
`nvidia-smi` lines, and `scontrol show node`. The earlier version sized jobs from the
replication-report benchmark table and the load snapshot's free-GPU count; both misled.

> ⚠️ **Three standing caveats**
> 1. **Code sync.** Jobs `cd` into `~/UIFProtoMF/ProtoMF` *when they start*, not when they
>    are submitted. So (a) the cluster checkout must equal local HEAD before submitting
>    (STEP 6), and (b) **never `git pull` into that checkout while compared jobs are still
>    pending** — a pull would change the code the pending jobs run. Smoke tests and other
>    branches use a separate worktree: `ssh leo5 "cd ~/UIFProtoMF/ProtoMF && git fetch &&
>    git worktree add ../ProtoMF_<branch> <branch>"` (fetch touches refs only). The user
>    granted standing permission to pull on LEO5 (2026-10-26); the pending-jobs lock still
>    applies.
> 2. **Data is not in git.** Datasets live on scratch (`PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data`,
>    exported by the wrapper). Confirm the dataset dir exists before submitting.
> 3. **Reportable numbers go through `std` via this skill, one `sbatch` per run.** The user
>    may authorize a whole wave in one sitting (as on 2026-10-26); then present the full
>    per-job table first, wait for the OK, and submit the runs one after another.

---

## STEP -1: IS THIS A BATCH JOB AT ALL? (login-node escape hatch)

The login node has **2× A30 that SLURM does not manage**, provided for interactive tests
(`Master/sensitive/leo5_cluster.md`). Pipeline checks — does the model build on CUDA, does
the config load, does an epoch run, a genuine `--profile smoke`, a post-hoc readout on a
checkpoint — run **there**, directly, not through the queue. Don't be stingy; just don't park
a long multi-process load on a node ~30 people share. Anything whose numbers get reported or
compared goes through `std`.

## STEP 0: CLUSTER STATE (built in — run it yourself)

```
python Master/scripts/leo5_status.py            # add --need-cpus / --need-mem-gb for the job shape you plan
```
If SSH fails, the VPN is down: ask the user to bring it up and stop. Do not fall back to
guessing.

Read the output as follows:

- **"idle" ≠ "placeable".** GPU nodes sit in the shared `std` partition; CPU-only jobs reserve
  their RAM and cores. A job places only where a free GPU, ≥ 2×`--cpus` idle CPUs and
  ≥ `--mem` *SLURM-unallocated* RAM coincide. The script applies those thresholds and lists
  GPU-idle-but-blocked nodes with the reason. (2026-10-26: all six A100s idle, their nodes
  1–13 GB unallocated for 1.5–9 days → nothing ≥ 36G could start there.)
- **Release horizon.** For each GPU type: how many are held and in which time bins they come
  free. State the **earliest realistic start** for the job in plain words ("nothing before
  Tue morning") — it changes whether sizing for speed or for placeability matters.
- **Free cores arrive in small chunks.** When another GPU job ends, typically ~8 CPUs free with
  it; the rest of the node stays CPU-job territory. Big `--cpus` requests lose those slots to
  smaller jobs.
- Pending GPU jobs of other users compete for every release; your own pending jobs are listed
  with their reason.

**Hardware facts (verified 2026-10-26, job logs + `scontrol`):**

| GPU type (`--gpu-type`) | Nodes | VRAM | Count |
|---|---|---|---|
| `a100` | n029–n034 | **40 GB** (A100-PCIe-40GB — not 80) | 6 |
| `a100_80` | n064 | 80 GB (separate gres type; `a100` does *not* match it) | 2 |
| `a30` | n035–n055 | 24 GB | 42 |
| `a40` | n056–n057 | 48 GB | 4 |
| `any` | all of the above | untyped `gpu:1`, first free GPU of any type | 54 |

Every GPU node: 128 CPUs = 64 cores × 2 threads, 256 GB RAM. `SelectTypeParameters=CR_CORE_MEMORY`:
**`--cpus N` allocates and bills 2N CPUs** (whole cores). Max walltime 10 days.

## STEP 1: COMBO + SEARCH PROFILE

From context: **model**, **dataset**, **seed** (default 38210573). Ask if ambiguous.

| Profile | num_samples / n_epochs / patience / grace | Use |
|---|---|---|
| `production` (default) | 100 / 100 / 10 / 4 | anything reported or compared; results dir gets the `_n100` suffix |
| `dev` | 30 / 60 / 7 / 4 | iterating; ~90% of the 100-trial best val; **never reportable** |
| `smoke` | 5 / 15 / 10 / 4 | pipeline check — prefer the login node (STEP -1) |

Cues: *final / paper / thesis / baseline / fleet* → production; *try / test / iterate / does it
run* → dev or smoke. Fixed for every reportable run: `--num-workers 1` (never 0 — it flips the
eval-negative regime, F-S0-02), ASHA on, `--optimizing-metric hit_ratio@10`, `--wandb-tag
thesis-grade`. `--retrain-config <config.json>` makes a single-trial retrain (cold rows); it
needs `--profile dev` explicitly (F-S0-10) and `--gpu-per-trial 1.0 --cpus 4`.

## STEP 2: MEASURED FOOTPRINT OF THE CLOSEST EARLIER RUN

Use measurements, not the benchmark table. Pick the closest run: same model+dataset (any
trial count) > same model, other dataset > same class (STEP 3b). Pull three things:

```
# per-run summary (concurrency, wall, peak VRAM, avg GPU%) — results dir on scratch
ssh leo5 "cd /scratch/c7031336/protomf_results && grep -hE 'Concurrency|Wall time' <model>_<dataset>_s38210573*/summary.md; \
  python3 -c \"import json;d=json.load(open('<dir>/hardware_summary.json'));print(d['max_vram_mib'],d['avg_gpu_util'])\""
# SLURM accounting: wall, allocated CPUs (= 2 x --cpus), peak RSS, CPU time
ssh leo5 "sacct -S 2026-07-01 --format=JobID%14,JobName%48,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU,State%10 | grep -E 'pmf_<model>_<dataset>'"
```
Trust `sacct` for RAM and CPU: the run dir's own `hardware_log.csv` sampler undercounts CPU
(misses exited Ray workers) and double-counts RAM (shared pages). The replication report's
benchmark table is the fallback only when no run of the model class exists.

**Reference: 30-trial `dev` runs, concurrency 5, A30 (the wave-1 basis, 2026-10-26)**

| combo | wall 30-trial | peak VRAM (5 trials) | avg GPU% | class |
|---|---|---|---|---|
| user_item_proto hm_1_month | 3h19 | 8.9 GB | 51 | heavy |
| feature_item_proto hm_1_month | 2h35 | 9.4 GB | 61 | heavy |
| feature_item_proto_noid hm_1_month | 2h21 | 8.5 GB | 64 | heavy |
| item_proto hm_1_month | 2h07 | 10.5 GB | 59 | heavy |
| item_proto ml-1m | 3h11 | 8.8 GB | 95 | heavy (GPU-bound) |
| user_proto hm_1_month | 3h21 | 2.1 GB | 21 | light |
| mf hm_1_month | 2h46 | 2.1 GB | 13 | light |
| feature_user_proto(_noid) hm_1_month | 2h51 / 2h45 | 2.7 GB | 22 | light |
| lightfm_tags(_ids) hm_1_month | 3h16 / 2h47 | 2.2 GB | 9 | light |
| lightfm_hist(_ids) hm_1_month | 3h32 / 2h21 | 2.1 GB | 14 | light |
| mf ml-1m | 1h46 | — | — | light |
| lightfm_hist(_ids) ml-1m | 1h50 / 1h59 | — | — | light |

Scaling facts: 30 → 100 trials at the same concurrency took **4.8×** the wall (lightfm_hist
ml-1m: 1h50 → 8h51); peak VRAM at 100 trials is **1.4–1.6×** the 30-trial peak (larger configs
get sampled); peak RSS at 100 trials: **~27 GB at concurrency 5, 36–38.5 GB at 8–10**.

## STEP 3: SIZE THE JOB

### 3a GPU type: `any` is the default
Submit `--gpu-type any` unless the VRAM projection (3b) exceeds an A30. Pin `a100` only when
the projection needs 24–32 GB **and** STEP 0 shows an A100 node placeable (its 40 GB is the
cap, never 80). `any` can land on an A100 too, so there is no speed penalty for being untyped,
only a placeability gain.

### 3b Concurrency from the GPU-use class
`concurrency = 1 / --gpu-per-trial`. Class by avg GPU% at concurrency 5 (STEP 2):

| class | avg GPU% @5 | concurrency on `any` | notes |
|---|---|---|---|
| light | ≤ 25 | **8** (`--gpu-per-trial 0.125`) | CPU-bound trials; 12 only if STEP 0 shows ≥ 26 idle CPUs beside a free GPU |
| medium | 25–50 | **6** (`0.1667`) | |
| heavy | > 50 | **5** (`0.2`) | 7–8 only on a placeable A100 with the VRAM check below |
| GPU-bound | ≥ 90 | **5** | already saturates a card; more concurrency buys nothing |

VRAM check: `peak30 × (c / 5) × 1.8 ≤ 0.8 × card` (19.2 GB on `any`, 32 GB on `a100`). A trial
that OOMs is lost from the search, so keep the margin.

### 3c CPUs — always explicit
`--cpus = concurrency + 1`. A trial keeps ~1.35 threads busy (main process + one DataLoader
worker, `OMP_NUM_THREADS=1`), and each requested CPU is a 2-thread core, so this already has
headroom. Never leave `--cpus` to the wrapper's auto formula (it asks for
`concurrency × (1 + num_workers)`, i.e. double). Floor 4 (F-S0-14: 2 requested hangs Ray).

### 3d RAM
`--mem = ceil((27 + 3 × (concurrency − 5)) × 1.15)` GB, rounded up to 4G:

| concurrency | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| `--mem` | 36G | 36G | 44G | 50G | 56G |

Lower `--mem` is also better placement (STEP 0), so do not pad beyond this.

### 3e Time
```
wall_100(c) = wall_30(5) × 5 × f(c)      f(c) = (5/c) / 0.9  for light/medium (trials are CPU-bound, slots ~90% busy)
                                          f(c) = (5/c)^0.6    for heavy (the GPU is already the bottleneck)
--time     = 2 × wall_100, rounded UP to 12:00:00 | 1-00:00:00 | 1-12:00:00 | 2-00:00:00 | 3-00:00:00
```
Show the arithmetic. Shorter limits backfill better, but a timeout loses the whole run; the 2×
is not optional. Retrains: 1–2 h measured → `04:00:00`.

Worked example (wave 1, 2026-10-26): light hm combos with wall_30 ≥ 3 h at concurrency 8 →
11–12 h → `1-12:00:00`; the others → `1-00:00:00`; heavy at 5 → 11–17 h → `1-12:00:00`
(`2-00:00:00` for user_item_proto hm and item_proto ml-1m).

## STEP 4: FLEET MODE (several runs that will be compared)

- **Size each job individually** (user decision 2026-10-26; replaces the 2026-07-11 rule that
  pinned concurrency fleet-wide). What stays fixed across compared rows: profile, ASHA,
  metric, seed, `--num-workers 1`, the W&B tag.
- **Concurrency is a disclosed noise source, not a bias.** After the 20 random TPE start-up
  trials, how many trials are in flight changes which configs get suggested, and ASHA's rung
  decisions depend on arrival order; it never changes how a config is scored. The existing
  ml-1m `_n100` rows already ran at concurrency 5, 8 and 10 (sacct AllocCPUS 20/32/40), so the
  fleet was never uniform. Each run records its concurrency in `summary.md` and in the job
  `.out` ("Concurrent trials"); Ch 4 discloses it. Optional, free: keep matched pairs
  (fI/fI_noid vs lightfm_tags_ids/lightfm_tags; fU/fU_noid vs lightfm_hist_ids/lightfm_hist) at
  the same concurrency.
- **Submit the whole wave at once** when STEP 0 shows nothing can start immediately; order by
  priority (hosts first). Pacing buys nothing while GPUs are held.
- **Lock the checkout** from the first submit until the last compared job has *started*
  (caveat 1). Plan other branches' smoke tests in a worktree before submitting.
- Totals to show: GPU-hours (Σ est. wall) and billed thread-hours (Σ 2 × `--cpus` × wall).

## STEP 5: PRESENT, THEN WAIT

One table, one row per job: model, dataset, class, GPU type, concurrency (`--gpu-per-trial`),
`--cpus`, `--mem`, est. wall, `--time`, results dir
(`<model>_<dataset>_s<seed>_n100/` for production, no suffix for dev/smoke/retrain), earliest
realistic start from STEP 0. Then the exact command(s). **Wait for the user's OK.**

## STEP 6: EXECUTE

1. **Code sync.** Local: `git status --short` (uncommitted code? → ask to commit+push) and
   `git rev-parse --short HEAD`. Cluster: `ssh leo5 "cd ~/UIFProtoMF/ProtoMF && git status --short | grep -v '^??'; git rev-parse --short HEAD"`.
   Mismatch → push locally if needed, then `ssh leo5 "cd ~/UIFProtoMF/ProtoMF && git fetch -q && git pull --ff-only -q origin <branch>"`
   — **only if no compared job of yours is pending** (`squeue -u c7031336 -t PD`). Laptop is the
   source of truth; never edit on the cluster.
2. **Data.** `ssh leo5 "ls /scratch/c7031336/protomf_data/<dataset>/"` must list the split
   files (+ `item_features.csv` for feature models). Missing → stop; `scp -r data/<dataset>
   leo5:/scratch/c7031336/protomf_data/` first.
3. **Dry-run the first job** with `--dry-run` and check the generated script: `--gres=gpu:1`
   (or the typed form), `--cpus-per-task`, `--mem`, `--time`, `--profile production
   --num-samples 100 --num-workers 1`.
4. **Submit**, one wrapper call per run:
   ```
   ssh leo5 "cd ~/UIFProtoMF/ProtoMF && bash slurm/run_combo.slurm -m <model> -d <dataset> \
     --gpu-per-trial <1/c> --num-workers 1 --cpus <c+1> --mem <X>G --gpu-type any --time <T> --wandb-tag thesis-grade"
   ```
   Capture the `Submitted batch job <id>` line for every run.
5. **Confirm and record.** `squeue -u c7031336` (or STEP 0 again). Write the submission table
   with job IDs to `Master/temp/<wave>_submissions_<date>.md` and tell the user. Suggest a
   `/protocol` entry if the wave is a thesis milestone. Logs: `/scratch/c7031336/protomf_logs/<jobname>_<jobid>.out`;
   the job exits non-zero unless `run_combo.py` wrote `OK` to its status file, so a 0 exit means
   the run dir is complete (`metadata.json` → `num_completed`).

---

## Notes

- `--gpu-type a100_80` is accepted by the wrapper and targets n064 only.
- `summary.md` in each run dir, plus `.out`, is the record of concurrency and card; `config.json`
  does not store them.
- The old `/leo5-load` command still exists as a thin standalone wrapper of the same script;
  this skill does not depend on it.
