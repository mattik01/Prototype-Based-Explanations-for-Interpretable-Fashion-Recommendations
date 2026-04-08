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
export PROTOMF_RESULTS_PATH=/scratch/c7031336/protomf_results
export OMP_NUM_THREADS=1
export WANDB_START_METHOD=thread

mkdir -p $RAY_TMPDIR $RAY_RESULTS_DIR $WANDB_DIR $WANDB_CACHE_DIR $PROTOMF_RESULTS_PATH

# --- Run ---
cd /home/c703/c7031336/UIFProtoMF/ProtoMF
echo "Starting: model=$MODEL dataset=$DATASET seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python Master/scripts/run_combo.py -m $MODEL -d $DATASET -s $SEED

echo "Finished: $(date)"

# Cleanup Ray temp
rm -rf $RAY_TMPDIR
