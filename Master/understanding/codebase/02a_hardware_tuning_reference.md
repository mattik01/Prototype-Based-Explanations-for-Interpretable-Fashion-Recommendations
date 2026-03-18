# 02a — Hardware-Aware Parameter Tuning Reference

*Companion to [02_config_entry.md](02_config_entry.md)*

## Parameters to Adjust

| Parameter | File | Default | What it controls |
|---|---|---|---|
| `GPU_PER_TRIAL` | `consts.py:8` | `0.2` | Ray scheduling hint: `1/value` = max concurrent GPU trials. Not a memory limit. |
| `CPU_PER_TRIAL` | `consts.py:9` | `1` | Ray scheduling hint: `total_cores/value` = max concurrent CPU trials. Should account for workers. |
| `NUM_WORKERS` | `consts.py:12` | `2` | DataLoader subprocesses per trial. Each runs negative sampling in parallel with training. |
| `NUM_SAMPLES` | `consts.py:6` | `100` | Total hyperparameter configs to try. Wall-clock = `NUM_SAMPLES / concurrent_trials × avg_trial_time`. |
| `prefetch_factor` | `experiment_helper.py:29` | `5` | Batches pre-loaded per worker. `NUM_WORKERS × prefetch_factor` batches buffered in RAM. |

Leave these alone: `OMP_NUM_THREADS` (keep `1`), `WANDB_START_METHOD` (keep `thread`), `nn.DataParallel` (no-op on single GPU), `n_epochs`/`batch_size`/`val_batch_size` (model concerns, not hardware).

**Key relationship:** Ray runs `min(GPU_allows, CPU_allows)` concurrent trials. Each trial spawns `1 + NUM_WORKERS` processes. Total processes = `concurrent_trials × (1 + NUM_WORKERS)`.

**Model size note:** These models are tiny (~500-600 MB per trial including CUDA overhead). GPU memory is never the bottleneck.

---

## Diagnostic Cheat Sheet

Run `nvidia-smi` (GPU) and `htop` (CPU) during training, then:

| You see | Cause | Fix |
|---|---|---|
| **GPU utilization low** (< 80%) | GPU waiting for data | Increase `NUM_WORKERS` (2→4) or `prefetch_factor` (5→10) |
| **GPU memory mostly empty** | Too few concurrent trials | Decrease `GPU_PER_TRIAL` (0.2→0.1) to allow more trials |
| **CPU cores idle** | Too few concurrent trials | Decrease `CPU_PER_TRIAL` to allow more trials |
| **CPU cores all maxed / thrashing** | Too many trials × workers | Increase `CPU_PER_TRIAL` (1→2) or decrease `NUM_WORKERS` |
| **RAM filling up** | Too many workers holding data copies | Decrease `NUM_WORKERS` or `prefetch_factor` |
| **Trials finishing but many queued** | Resource limits too conservative | Decrease `CPU_PER_TRIAL` and/or `GPU_PER_TRIAL` |
| **Everything looks good but slow** | Not enough total trials | Trials are just slow — enable ASHA scheduler to kill bad ones early |

---

## Quick Setup Recipes

**GPU machine** (RTX 4070 Ti SUPER, 16 GB VRAM, check cores with `nproc`):
```
GPU_PER_TRIAL = 0.1     # 10 concurrent GPU slots
CPU_PER_TRIAL = 2       # cores/2 concurrent CPU slots
NUM_WORKERS   = 2       # start here, bump to 4 if GPU underutilized
NUM_SAMPLES   = 100
```

**Laptop** (8 CPU cores, no GPU, smoke tests only):
```
CPU_PER_TRIAL = 3       # ~2 concurrent trials
NUM_WORKERS   = 0       # CPU training is the bottleneck, not data loading
NUM_SAMPLES   = 1-5
```
