#!/bin/bash
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/c7031336/protomf_logs/%x_%j.out
#SBATCH --error=/scratch/c7031336/protomf_logs/%x_%j.err

# Regularizer sensitivity sweep — 6 fixed-config single trainings, no Ray.
# Usage: sbatch --gres=gpu:a30:1 --job-name=pmf_regsweep_ml-1m run_reg_sweep.sbatch ml-1m [SEED]
DATASET=${1:?Usage: sbatch run_reg_sweep.sbatch DATASET [SEED]}
SEED=${2:-38210573}

# --- Environment ---
module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate

# --- Directories (all on scratch) ---
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data
export PROTOMF_REGSWEEP_PATH=/scratch/c7031336/protomf_results/reg_sensitivity
export WANDB_DIR=/scratch/c7031336/wandb_logs
export WANDB_CACHE_DIR=/scratch/c7031336/wandb_cache
export WANDB_API_KEY=$(cat /home/c703/c7031336/.wandb_key 2>/dev/null || echo "")
export WANDB_MODE=${WANDB_MODE:-online}
export OMP_NUM_THREADS=1
export WANDB_START_METHOD=thread

mkdir -p $PROTOMF_REGSWEEP_PATH $WANDB_DIR $WANDB_CACHE_DIR

# --- Run ---
cd /home/c703/c7031336/UIFProtoMF/ProtoMF
echo "Starting: reg sweep dataset=$DATASET seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python Master/scripts/run_reg_sweep.py -d $DATASET -s $SEED --num-workers 4

echo "Finished: $(date)"
