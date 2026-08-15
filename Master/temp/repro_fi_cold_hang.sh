#!/bin/bash
# Control experiment for the dc01 cold-retrain hang (F-S0-14 sitting, 2026-08-15).
# Kept as the reproduction record cited in sc01_feature_item_proto.md.
#
# Runs the EXACT command that timed out four times under SLURM (jobs 7121151,
# 7121152, 7148306, 7148307) on the LEO5 login node instead, where 128 CPUs are
# visible rather than the 4 threads SLURM allocated. Result: RUN_OK 1/1, exit 0,
# full artifact set, ~6 minutes — establishing that the code path is healthy and
# the failure was the allocation, not the model, the dataset, wandb, or the node.
#
# Faithful to slurm/run_combo.slurm's job body EXCEPT:
#   - PYTHONUNBUFFERED=1   (the batch logs lost everything to block buffering;
#                           now set permanently in the wrapper)
#   - --n-epochs 2         (only needs to reach the first epoch line)
#   - results/ray dirs redirected to *_repro so nothing canonical is touched;
#     its metrics are a 2-epoch toy budget and are DISCARDED, never a row.
set -u
module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate

export RAY_TMPDIR=/scratch/c7031336/ray_tmp/repro_fi_cold
export RAY_RESULTS_DIR=/scratch/c7031336/ray_results_repro
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data
export PROTOMF_RESULTS_PATH=/scratch/c7031336/protomf_results_repro
export GPU_LOG_DIR=/scratch/c7031336/gpu_logs/repro_fi_cold
export RUN_STATUS_FILE=/scratch/c7031336/protomf_status/repro_fi_cold.status
export OMP_NUM_THREADS=1
export WANDB_START_METHOD=thread
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$RAY_TMPDIR" "$RAY_RESULTS_DIR" "$PROTOMF_RESULTS_PATH" "$GPU_LOG_DIR" "$(dirname "$RUN_STATUS_FILE")"
echo "FAIL: not started" > "$RUN_STATUS_FILE"

cd /home/c703/c7031336/UIFProtoMF/ProtoMF
echo "Node: $(hostname) | $(date)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

python Master/scripts/run_combo.py \
  -m feature_item_proto -d hm_1_month_cold -s 38210573 \
  --profile dev --num-samples 30 --gpu-per-trial 1.0 --cpu-per-trial 1 \
  --num-workers 1 --n-epochs 2 --grace-period 4 --patience 7 \
  --wandb-mode disabled --optimizing-metric hit_ratio@10 --wandb-tag dc01-repro \
  --retrain-config /scratch/c7031336/protomf_results/feature_item_proto_hm_1_month_s38210573/config.json
echo "REPRO_EXIT $? $(date)"
