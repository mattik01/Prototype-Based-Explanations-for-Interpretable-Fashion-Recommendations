#!/bin/bash
# dc09 SC.3 stage D smoke: one fUfI arm x dataset through the hyperopt path (login-node A30).
# Usage: smoke_dc09.sh <model> <dataset> <cuda_device_index> [num_samples] [n_epochs]
# Runs from the detached worktree ~/UIFProtoMF/ProtoMF_dev (the fleet checkout is untouched).
# Measures: GPU memory (nvidia-smi, 5s sampling) + summed user RSS (5s sampling).
set -u
M=$1
DS=$2
GPU=$3
NS=${4:-2}
NE=${5:-3}
SMOKE=/scratch/c7031336/smoke_runs/dc09
TAG=${M}_${DS}
mkdir -p "$SMOKE/logs"

module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate

export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data
export RAY_RESULTS_DIR=$SMOKE/ray_results
export PROTOMF_RESULTS_PATH=$SMOKE/results
# Short path — Ray AF_UNIX sockets cap at 107 bytes
export RAY_TMPDIR=/tmp/ray_d9_$GPU
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RAY_RESULTS_DIR" "$PROTOMF_RESULTS_PATH" "$RAY_TMPDIR"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPU" -l 5 \
  > "$SMOKE/logs/gpu_mem_${TAG}.log" 2>&1 &
NVPID=$!
( while true; do ps -u "$USER" -o rss= | awk '{s+=$1} END {print s}'; sleep 5; done ) \
  > "$SMOKE/logs/ram_${TAG}.log" 2>&1 &
RAMPID=$!

cd /home/c703/c7031336/UIFProtoMF/ProtoMF_dev
echo "=== dc09 smoke: $M x $DS on GPU $GPU ($NS trials x $NE epochs), $(git rev-parse --short HEAD), $(date) ==="
START=$(date +%s)
python Master/scripts/run_combo.py -m "$M" -d "$DS" -s 38210573 \
  --profile smoke --num-samples "$NS" --n-epochs "$NE" --gpu-per-trial 0.5 \
  --num-workers 1 --skip-explanations --wandb-mode disabled
PYEXIT=$?

kill $NVPID $RAMPID 2>/dev/null
rm -rf "$RAY_TMPDIR"
echo "SMOKE_EXIT=$PYEXIT model=$M dataset=$DS wall=$(( $(date +%s) - START ))s date=$(date)"
