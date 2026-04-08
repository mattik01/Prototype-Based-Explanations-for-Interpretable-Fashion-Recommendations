# LEO5 HPC Migration Plan

## Context

Moving training from the Windows GPU machine (single RTX 4070 Ti SUPER, 16 GB VRAM) to the UIBK LEO5 HPC cluster. Motivation: massively parallel hyperopt — up to 75 independent SLURM jobs running simultaneously on A100/A30/A40 GPUs. Claude Code runs from the laptop via SSH.

**Constraints:** VPN required for all LEO5 access. Data folder is pending changes (splitter rework). Claude Code cannot be installed on the cluster.

---

## 1. Trial Distribution Strategy

### Decision: Ray Tune inside single SLURM jobs (one job per model × dataset × seed)

**Why not SLURM array jobs (1 trial each)?**
- HyperOpt is Bayesian (TPE) — it needs prior trial results to sample intelligently. Pre-sampling all configs degrades to random search.
- ASHA (`grace_period=4`) kills underperforming trials after 4 epochs. Without a central scheduler, every trial runs to completion (up to 100 epochs + patience=10). This could 10x GPU-hours.
- Both ASHA and HyperOpt are critical to making 100-trial search feasible.

**Why not multi-node Ray cluster?**
- Fragile on HPC (port conflicts, node scheduling, firewalls).
- ProtoMF models are small — no benefit to multi-node training.
- Unnecessary complexity for a thesis project.

**What we get:**
- SLURM-level parallelism at the (model × dataset × seed) granularity = up to 75 independent jobs
- Within each job: Ray Tune manages 16 concurrent trials on 1 GPU with ASHA + HyperOpt intact
- On A100 (80 GB VRAM): even `item_proto` / `user_item_proto` (10-13 GB peak) can run comfortably with 16 concurrent trials
- On A30 (24 GB VRAM): lightweight models (`mf`, `acf`, `user_proto`) fit easily at 16 concurrent

### ASHA: Fully preserved

No changes needed. `ASHAScheduler(grace_period=4)` works within each Ray Tune session exactly as before.

---

## 2. Environment Setup

### Strategy: LEO5 PyTorch module + venv on scratch

```bash
# On LEO5 login node:
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02

# Create venv on scratch (home = 5 GB, too small)
python -m venv /scratch/c7031336/venvs/protomf --system-site-packages
source /scratch/c7031336/venvs/protomf/bin/activate

# Install remaining deps
pip install "ray[tune]" wandb hyperopt scipy pandas bottleneck
```

`--system-site-packages` inherits PyTorch/numpy/CUDA from the module — no reinstall needed.

**Fallback (if module has issues):**
```bash
module load miniconda3
conda create -p /scratch/c7031336/conda_envs/protomf python=3.11
conda activate /scratch/c7031336/conda_envs/protomf
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install "ray[tune]" wandb hyperopt scipy pandas bottleneck
```

### Validation test:
```bash
srun --partition=std --gres=gpu:a30:1 --time=00:10:00 --cpus-per-task=4 bash -c '
    module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
    source /scratch/c7031336/venvs/protomf/bin/activate
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    python -c "import ray; print(ray.__version__)"
    python -c "import wandb; print(wandb.__version__)"
'
```

---

## 3. Data Transfer & Path Strategy

### Problem: `data/` has tracked files

`data/` contains tracked files (splitters, README) alongside gitignored CSV data. Cannot symlink the entire folder.

### Solution: Per-dataset symlinks

```bash
# Create data directories on scratch
mkdir -p /scratch/c7031336/protomf_data/{ml-1m,amazon2014,hm,hm_full,hm_3_month,lfm2b-1mon}

# Transfer processed data from laptop to scratch
rsync -avz --progress data/ml-1m/*.csv leo5:/scratch/c7031336/protomf_data/ml-1m/
rsync -avz --progress data/amazon2014/*.csv leo5:/scratch/c7031336/protomf_data/amazon2014/
# hm data: pending splitter changes, transfer once finalized
# rsync -avz --progress data/hm_full/*.csv leo5:/scratch/c7031336/protomf_data/hm_full/
# rsync -avz --progress data/hm_3_month/*.csv leo5:/scratch/c7031336/protomf_data/hm_3_month/

# On LEO5: create symlinks for the CSV files within the repo's data dirs
# The splitter .py files are tracked by git and stay in home
cd /home/c703/c7031336/UIFProtoMF/ProtoMF/data/ml-1m
ln -s /scratch/c7031336/protomf_data/ml-1m/*.csv .
# Repeat for other datasets
```

**Alternative (simpler, preferred): Override DATA_PATH via env var.**

Add to `utilities/consts.py`:
```python
DATA_PATH = os.environ.get('PROTOMF_DATA_PATH',
                           os.path.join(os.path.dirname(__file__), '..', 'data'))
```

Then in SLURM: `export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data`

This is cleaner — no symlinks, no fragile glob expansions. The scratch data dir mirrors the structure of `data/` but only contains the CSV files.

**Decision: Use the env var approach.** One-line code change, zero symlink maintenance.

---

## 4. W&B Integration

### Online mode (preferred)

Most HPC clusters allow outbound internet from compute nodes. Test first:
```bash
srun --partition=std --time=00:05:00 bash -c '
    source /scratch/c7031336/venvs/protomf/bin/activate
    python -c "import requests; r = requests.get(\"https://api.wandb.ai/healthz\"); print(r.status_code)"
'
```

If 200: use `WANDB_MODE=online` (default). All existing W&B integration works as-is.

### Offline fallback

If compute nodes lack internet:
```bash
export WANDB_MODE=offline
export WANDB_DIR=/scratch/c7031336/wandb_logs
```

After jobs finish, sync from login node:
```bash
source /scratch/c7031336/venvs/protomf/bin/activate
wandb sync /scratch/c7031336/wandb_logs/wandb/offline-run-*
```

### W&B API key: env var fallback

Current code (`consts.py:19-21`) crashes if the key file is missing. Fix:
```python
_wandb_key_path = os.path.join(os.path.dirname(__file__), '..', 'Master', 'sensitive', 'api_keys', 'wandb_key.txt')
try:
    with open(_wandb_key_path, 'r') as _f:
        WANDB_API_KEY = _f.read().strip()
except FileNotFoundError:
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY', '')
```

SLURM script sets `WANDB_API_KEY` via env var, so the file doesn't need to exist on LEO5 (avoids putting secrets in home).

---

## 5. Code Changes Required

### 5a. `utilities/consts.py` — 2 changes

1. **DATA_PATH env var override** (see section 3)
2. **WANDB_API_KEY resilience** (see section 4)

### 5b. `experiment_helper.py` — 2 changes

1. **Ray temp directory on scratch:**
   ```python
   # Before tune.run() in start_hyper():
   import ray
   ray_tmpdir = os.environ.get('RAY_TMPDIR', None)
   if not ray.is_initialized():
       ray.init(_temp_dir=ray_tmpdir) if ray_tmpdir else ray.init()
   ```

2. **Ray results storage on scratch:**
   ```python
   analysis = tune.run(
       ...
       storage_path=os.environ.get('RAY_RESULTS_DIR', None),  # ADD
       ...
   )
   ```

### 5c. `rec_sys/tester.py:54` — PyTorch 2.x compat

```python
# Old:
params = torch.load(self.model_load_path, map_location=self.device)
# New:
params = torch.load(self.model_load_path, map_location=self.device, weights_only=True)
```

### 5d. No other changes needed

- `nn.DataParallel` works fine in PyTorch 2.x (no-op on single GPU)
- `tune.run()` still works in Ray 2.x (deprecated but functional)
- `NUM_WORKERS = 2` on Linux is correct for LEO5
- Device detection (`torch.cuda.is_available()`) is automatic
- `prefetch_factor` handling already correct

---

## 6. SLURM Scripts

### Directory: `slurm/` (repo root)

### `slurm/run_combo.sbatch` — Main job script

```bash
#!/bin/bash
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/c7031336/protomf_logs/%x_%j.out
#SBATCH --error=/scratch/c7031336/protomf_logs/%x_%j.err

# Usage: sbatch --gres=gpu:a30:1 --job-name=pmf_mf_ml-1m_s42 run_combo.sbatch mf ml-1m 42
MODEL=${1:?Usage: sbatch run_combo.sbatch MODEL DATASET SEED}
DATASET=${2:?}
SEED=${3:?}

# --- Environment ---
module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate

# --- Directories (all on scratch) ---
export RAY_TMPDIR=/scratch/c7031336/ray_tmp/${SLURM_JOB_ID}
export RAY_RESULTS_DIR=/scratch/c7031336/ray_results
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data
export WANDB_DIR=/scratch/c7031336/wandb_logs
export WANDB_CACHE_DIR=/scratch/c7031336/wandb_cache
export WANDB_API_KEY=$(cat /home/c703/c7031336/.wandb_key 2>/dev/null || echo "")
export WANDB_MODE=${WANDB_MODE:-online}
export OMP_NUM_THREADS=1
export WANDB_START_METHOD=thread

mkdir -p $RAY_TMPDIR $RAY_RESULTS_DIR $WANDB_DIR $WANDB_CACHE_DIR

# --- Run ---
cd /home/c703/c7031336/UIFProtoMF/ProtoMF
echo "Starting: model=$MODEL dataset=$DATASET seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python start.py -m $MODEL -d $DATASET -s $SEED

echo "Finished: $(date)"

# Cleanup Ray temp
rm -rf $RAY_TMPDIR
```

### `slurm/submit_all.sh` — Batch launcher

```bash
#!/bin/bash
# Submit all experiment jobs. Edit COMBOS to control what runs.
# Usage: bash slurm/submit_all.sh

SCRIPT="slurm/run_combo.sbatch"
SEEDS=(38210573 9491758 2931009)

submit() {
    local model=$1 dataset=$2 gpu=$3 time=$4
    for seed in "${SEEDS[@]}"; do
        echo "Submitting: $model × $dataset × seed=$seed → $gpu (${time})"
        sbatch --gres=gpu:${gpu}:1 \
               --time=$time \
               --job-name="pmf_${model}_${dataset}_s${seed}" \
               $SCRIPT $model $dataset $seed
    done
}

# --- GPU assignment ---
# A30 (24 GB): mf, acf, user_proto (lightweight, < 5 GB VRAM)
# A100 (80 GB): item_proto, user_item_proto (10-13 GB VRAM, safe headroom for 16 concurrent)

# --- Replication: incomplete combos ---
submit item_proto   amazon2014   a100  1-00:00:00
submit item_proto   ml-1m        a100  1-00:00:00
submit user_item_proto ml-1m     a100  1-00:00:00

# --- H&M 3-month ---
submit mf           hm_3_month   a30   2-00:00:00
submit acf          hm_3_month   a30   2-00:00:00
submit user_proto   hm_3_month   a30   2-00:00:00
submit item_proto   hm_3_month   a100  3-00:00:00
submit user_item_proto hm_3_month a100 3-00:00:00

# --- H&M full ---
submit mf           hm_full      a30   5-00:00:00
submit acf          hm_full      a30   5-00:00:00
submit user_proto   hm_full      a30   5-00:00:00
submit item_proto   hm_full      a100  7-00:00:00
submit user_item_proto hm_full   a100  7-00:00:00
```

### `slurm/setup_env.sh` — One-time setup (run manually)

```bash
#!/bin/bash
# One-time environment setup on LEO5. Run from login node.
set -e

echo "=== LEO5 ProtoMF Environment Setup ==="

# 1. Scratch directories
echo "Creating scratch directories..."
mkdir -p /scratch/c7031336/{protomf_data,protomf_logs,ray_tmp,ray_results,wandb_logs,wandb_cache,venvs}

# 2. Python venv
echo "Creating Python venv..."
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
python -m venv /scratch/c7031336/venvs/protomf --system-site-packages
source /scratch/c7031336/venvs/protomf/bin/activate

# 3. Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install "ray[tune]" wandb hyperopt scipy pandas bottleneck

# 4. W&B key
echo "Place your W&B API key in /home/c703/c7031336/.wandb_key"
echo "(or export WANDB_API_KEY in your environment)"

# 5. Verify
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import ray; print(f'Ray {ray.__version__}')"
python -c "import wandb; print(f'W&B {wandb.__version__}')"
python -c "import hyperopt; print(f'Hyperopt {hyperopt.__version__}')"

echo ""
echo "=== Done ==="
echo "Next steps:"
echo "  1. Transfer data: rsync -avz data/ leo5:/scratch/c7031336/protomf_data/"
echo "  2. Place W&B key: echo 'YOUR_KEY' > /home/c703/c7031336/.wandb_key"
echo "  3. Test: sbatch --gres=gpu:a30:1 --time=00:30:00 slurm/run_combo.sbatch debug ml-1m 38210573"
```

---

## 7. Scratch Directory Layout

```
/scratch/c7031336/
├── protomf_data/          # Mirrors data/ structure (CSVs only)
│   ├── ml-1m/             # listening_history_{train,val,test}.csv, user_ids.csv, item_ids.csv
│   ├── amazon2014/
│   ├── hm_full/
│   ├── hm_3_month/
│   └── lfm2b-1mon/
├── protomf_logs/          # SLURM stdout/stderr
├── ray_tmp/               # Per-job Ray temp (auto-cleaned)
├── ray_results/           # Ray Tune trial results + checkpoints
├── wandb_logs/            # W&B run data
├── wandb_cache/           # W&B cache
└── venvs/
    └── protomf/           # Python venv
```

---

## 8. GPU Assignment Strategy

| Model | Peak VRAM (observed) | GPU | Concurrent trials | Notes |
|-------|---------------------|-----|-------------------|-------|
| `mf` | ~1 GB | A30 (24 GB) | 16 | CPU-bound at 16 concurrent |
| `acf` | ~1.2 GB | A30 (24 GB) | 16 | |
| `user_proto` | ~1.3 GB | A30 (24 GB) | 16 | |
| `item_proto` | ~10-13 GB | A100 (80 GB) | 16 | A30 too tight for 16 concurrent |
| `user_item_proto` | ~13 GB | A100 (80 GB) | 16 | Was 1-trial on RTX, now 16 on A100 |

**Key win:** `item_proto` and `user_item_proto` previously ran 1-8 trials at a time on the RTX. On A100 they can run 16 concurrent trials — massive speedup.

---

## 9. Execution Sequence

### Phase A: Code changes (laptop, no VPN needed)
- [ ] Modify `utilities/consts.py` (DATA_PATH override, WANDB_API_KEY resilience)
- [ ] Modify `experiment_helper.py` (Ray tmpdir, storage_path)
- [ ] Modify `rec_sys/tester.py` (weights_only=True)
- [ ] Create `slurm/` directory with scripts
- [ ] Commit and push to `dev`

### Phase B: Cluster setup (VPN required)
- [ ] SSH into LEO5
- [ ] `git pull` the repo
- [ ] Run `bash slurm/setup_env.sh`
- [ ] Place W&B key: `echo 'KEY' > ~/.wandb_key`
- [ ] Test W&B connectivity from compute node (see section 4)

### Phase C: Data transfer (VPN required)
- [ ] Transfer processed CSVs: `rsync -avz data/ml-1m/*.csv leo5:/scratch/c7031336/protomf_data/ml-1m/`
- [ ] Repeat for amazon2014
- [ ] H&M data: transfer once splitter changes are finalized

### Phase D: Smoke test (VPN required)
- [ ] Submit debug job: `sbatch --gres=gpu:a30:1 --time=00:30:00 slurm/run_combo.sbatch debug ml-1m 38210573`
- [ ] Check logs: `tail -f /scratch/c7031336/protomf_logs/pmf_debug_ml-1m_s38210573_*.out`
- [ ] Verify W&B dashboard shows the run
- [ ] Verify Ray temp and results on scratch

### Phase E: Production runs
- [ ] Submit incomplete replication jobs first (item_proto, user_item_proto)
- [ ] Then H&M jobs
- [ ] Monitor: `squeue -u c7031336`, `sacct -u c7031336 --format=JobID,JobName,State,Elapsed,MaxRSS`

---

## 10. Monitoring & Recovery

### From laptop (Claude Code via SSH):
```bash
ssh leo5 "squeue -u c7031336"                    # job status
ssh leo5 "tail -20 /scratch/c7031336/protomf_logs/pmf_mf_ml-1m_*.out"  # latest output
ssh leo5 "sacct -u c7031336 --format=JobID,JobName%30,State,Elapsed"   # history
```

### If a job fails:
1. Check error log: `/scratch/c7031336/protomf_logs/*_JOBID.err`
2. Ray results persist on scratch — can inspect partial results
3. Re-submit with same command (Ray won't resume, starts fresh)

### Result extraction:
- W&B: automatic (if online mode works)
- Manual: `ssh leo5 "cat /scratch/c7031336/ray_results/<run_name>/best_trial_config.json"`
- Best model checkpoints: in Ray results dir on scratch

---

## 11. Open Questions

1. **Internet from compute nodes?** Test W&B connectivity. Determines online vs offline mode.
2. **Ray version compatibility?** `tune.run()` should work in Ray 2.x but may emit deprecation warnings. Test in smoke run.
3. **PyTorch 2.x model compat?** Simple codebase, likely fine. Smoke test will confirm.
4. **H&M data transfer:** Blocked until splitter changes are done. Plan the rsync once data/ is finalized.
5. **`num_samples` for hm_full:** 100 trials × 100 epochs on 889K users — estimate wall time from hm_3_month runs.
