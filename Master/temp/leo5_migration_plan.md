# LEO5 HPC Cluster Migration Plan

## Context
Moving training from the Windows GPU machine (single RTX 4070 Ti SUPER) to the UIBK LEO5 HPC cluster for significantly faster hyperopt runs. Claude Code can't be installed on the cluster — will run from Linux laptop via SSH.

## Credentials (to save to `Master/sensitive/`)
- Account: `c7031336`
- Host: `login.leo5.uibk.ac.at`
- Status page: `https://login.leo5.uibk.ac.at/cgi-bin/slurm.pl` (VPN required)

---

## Phase 1: Cleanup & Transfer Checklist

### 1a. Gitignored files on GPU machine → inventory & transfer to laptop via Google Drive

**Files to inventory and transfer:**
- [ ] `Master/sensitive/` — API keys, infrastructure docs, wake/shutdown scripts
  - `api_keys/wandb_key.txt` (86 bytes)
  - `infrastructure.md` (6.3 KB)
  - `wake_gpu.sh`, `shutdown_gpu.sh`
  - `.claude/settings.local.json`
- [ ] `data/ml-1m/` — processed splits (~41 MB)
- [ ] `data/amazon2014/` — processed splits (~59 MB)
- [ ] `data/hm_full/` — processed splits (~4.4 GB)
- [ ] `data/hm_3_month/` — processed splits (~508 MB)
- [ ] `data/hm/raw/` — raw Kaggle files (~3.5 GB) — **decide: keep or skip?**
- [ ] `data/hm/` processed — (~850 MB) — **likely redundant with hm_full/hm_3_month**
- [ ] `wandb/` — local W&B logs (192 KB) — **can delete, not needed**
- [ ] `Master/temp/gpu_logs/` — CSV monitoring logs (~12 MB) — **transfer if you want history**
- [ ] `Master/temp/checkpoints/` — single debug checkpoint (292 KB) — **can delete**
- [ ] `__pycache__/` dirs — **delete, not transfer**

### 1b. Cleanup actions on GPU machine
- Delete all `__pycache__/` directories
- Delete `wandb/` local logs
- Delete `Master/temp/checkpoints/`
- Optionally clean large GPU log files in `Master/temp/gpu_logs/`

### 1c. Transfer method
- Upload to Google Drive: `Master/sensitive/`, all processed datasets
- Download on laptop, place in correct paths
- Verify with `ls` / checksums

---

## Phase 2: Save cluster credentials to `Master/sensitive/`

- Create `Master/sensitive/leo5_cluster.md` with:
  - SSH login, hostname, account
  - Partition info (once discovered)
  - Storage paths, module load commands
  - SLURM job template

---

## Phase 3: Cluster Setup & SLURM Integration

### 3a. Discover cluster capabilities (from laptop via SSH, or user logs in manually)
- `sinfo -o '%P %G %D %c %m %l'` — partitions, GPUs, node count, cores, memory, time limits
- `module avail` — find CUDA, Python, PyTorch, conda modules
- `quota` / `df -h` — storage limits

### 3b. Environment setup on LEO5
```bash
# On login node:
module load python/3.x cuda/12.x  # whatever's available
python -m venv ~/protomf_env       # or use conda if module exists
source ~/protomf_env/bin/activate
pip install torch ray[tune] hyperopt wandb bottleneck numpy pandas scikit-learn
```

### 3c. SLURM job design

**Key insight:** Ray Tune runs INSIDE a single SLURM job. Each SLURM job = one `start.py -m X -d Y -s Z` call. SLURM handles node allocation; Ray handles trial scheduling within the job.

**Parallelization strategy:**
- Submit **one SLURM job per (model × dataset × seed)** combination
- Each job requests 1 GPU + N CPUs
- Ray Tune runs 16 concurrent trials on that GPU (GPU_PER_TRIAL=0.0625)
- For multi-seed: submit 3 jobs instead of sequential `start_multiple_hyper()`

**Example SLURM script (`slurm_run.sh`):**
```bash
#!/bin/bash
#SBATCH --job-name=protomf_${MODEL}_${DATASET}_s${SEED}
#SBATCH --partition=<gpu_partition>   # TBD after discovery
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

module load python/3.x cuda/12.x
source ~/protomf_env/bin/activate
export WANDB_API_KEY=$(cat ~/wandb_key.txt)

cd ~/protomf_repo
python start.py -m $MODEL -d $DATASET -s $SEED
```

**Batch submission script:**
```bash
for model in mf acf user_proto item_proto user_item_proto; do
  for dataset in ml-1m amazon2014 hm_3_month; do
    for seed in 38210573 9491758 2931009; do
      sbatch --export=MODEL=$model,DATASET=$dataset,SEED=$seed slurm_run.sh
    done
  done
done
# = 45 jobs, can run in parallel depending on allocation
```

### 3d. Code changes needed
- `utilities/consts.py`: detect LEO5 hostname for `NUM_WORKERS` (Linux, not Windows)
- `experiment_helper.py`: possibly skip aggregation step when seeds run as separate jobs
- Create `Master/scripts/slurm_run.sh` and `Master/scripts/slurm_submit_all.sh`
- W&B key: copy to `~/wandb_key.txt` on cluster (or set env var)

### 3e. W&B on cluster
- W&B works fine with pip install + API key
- Compute nodes need internet access for W&B sync (most HPC clusters allow this)
- Fallback: `wandb offline` mode, then `wandb sync` from login node

---

## Phase 4: Claude Code via SSH from Laptop

**Setup:** Run Claude Code on mattik01 (Linux laptop), wrap commands with SSH.

**Approach:**
- Set up SSH key auth from laptop → LEO5 (avoid password prompts)
- Claude runs locally, uses `ssh c7031336@login.leo5.uibk.ac.at "<command>"` for remote execution
- Can check job status: `ssh leo5 "squeue -u c7031336"`
- Can read logs: `ssh leo5 "cat ~/slurm_12345.out"`
- Can submit jobs: `ssh leo5 "cd protomf_repo && sbatch slurm_run.sh"`

**Limitations:**
- Requires VPN to be active on laptop
- No file editing on cluster (use git push/pull to sync code changes)
- Latency on each SSH command

**Workflow:**
1. Develop/edit code on laptop with Claude
2. `git push` to sync
3. SSH to cluster: `git pull && sbatch ...`
4. Monitor via SSH: `squeue`, `tail -f slurm_*.out`
5. Results sync back via W&B (automatic) or `git push` from cluster

---

## Phase 5: Update Documentation

- Update `CLAUDE.md` — add LEO5 as third machine environment
- Update `Master/sensitive/leo5_cluster.md` — full cluster reference
- Update `Master/docs/masterplan.md` — Phase 1.3.1 HPC section → active, not "if/when needed"
- Add LEO5 to the machines table in memory

---

## Execution Order
1. **Now (GPU machine):** Inventory gitignored files, prepare transfer checklist
2. **Now (GPU machine):** Save cluster credentials to sensitive
3. **Transfer:** Upload to Google Drive, download on laptop
4. **Laptop (with VPN):** SSH into LEO5, discover partitions/modules/storage
5. **Laptop:** Create SLURM scripts, test with debug job
6. **Laptop:** Transfer datasets to cluster, set up env
7. **Laptop:** Submit real jobs, monitor
8. **Ongoing:** Update docs as setup stabilizes
