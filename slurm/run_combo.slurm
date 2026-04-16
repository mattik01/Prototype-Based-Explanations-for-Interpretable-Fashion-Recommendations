#!/bin/bash
# ============================================================================
# run_combo.slurm — SLURM submission wrapper for run_combo.py
#
# Accepts all run_combo.py args + resource config, auto-computes matching
# SLURM directives, and submits via sbatch.
#
# Usage:
#   ./slurm/run_combo.slurm -m item_proto -d ml-1m -s 42
#   ./slurm/run_combo.slurm -m mf -d lfm2b-1mon -s 42 --num-samples 50 --gpu-per-trial 0.125
#   ./slurm/run_combo.slurm -m user_item_proto -d ml-1m -s 42 --gpu-type a100 --time 3-00:00:00
# ============================================================================

set -euo pipefail

# ---- Defaults (match run_combo.py defaults) ----
MODEL=""
DATASET=""
SEED="38210573"
DELAY=""
NUM_SAMPLES=100
GPU_PER_TRIAL="0.0625"
CPU_PER_TRIAL=1
NUM_WORKERS=2
N_EPOCHS=100
GRACE_PERIOD=4
PATIENCE=10
MIN_DELTA=""  # empty = use python-side default (per optimizing metric)
WANDB_MODE="online"
OPTIMIZING_METRIC="hit_ratio@10"
ASHA=1
WANDB_TAGS=()

# SLURM-specific defaults
PARTITION="std"
TIME="2-00:00:00"
MEM="64G"
GPU_TYPE="a30"
NUM_GPUS=1
CPUS=""  # empty = auto-compute from formula

usage() {
    cat <<'USAGE'
Usage: ./slurm/run_combo.slurm -m MODEL -d DATASET [OPTIONS]

Required:
  -m, --model MODEL          Model to run (mf, acf, user_proto, item_proto, user_item_proto)
  -d, --dataset DATASET      Dataset (amazon2014, ml-1m, lfm2b-1mon, hm_full, hm_3_month)

Optional (run_combo):
  -s, --seed SEED            Random seed (default: 38210573)
  --delay HOURS              Hours to wait before starting
  --num-samples N            Hyperopt trials (default: 100)
  --gpu-per-trial F          Fractional GPU per trial (default: 0.0625 → 16 concurrent)
  --cpu-per-trial N          CPUs per trial (default: 1)
  --num-workers N            DataLoader workers per trial (default: 2)
  --n-epochs N               Max epochs per trial (default: 100)
  --grace-period N           ASHA min epochs (default: 4)
  --patience N               Early stopping patience (default: 10)
  --min-delta F              Min improvement to reset patience (default: per-metric, ~1e-4)
  --wandb-mode MODE          online|offline|disabled (default: online)
  --optimizing-metric M      Metric for selection (default: hit_ratio@10)
  --no-asha                  Disable ASHA early stopping scheduler (default: enabled)
  --wandb-tag TAG            Extra W&B tag (repeatable)

Optional (SLURM):
  --partition P              SLURM partition (default: std)
  --time T                   Wall-clock limit (default: 2-00:00:00)
  --mem M                    Memory allocation (default: 64G)
  --gpu-type T               GPU type (default: a30)
  --num-gpus N               Number of GPUs (default: 1)
  --cpus N                   CPUs to request (default: auto = concurrent*(cpu_per_trial+num_workers)+2)
  --dry-run                  Print sbatch script without submitting
USAGE
    exit 1
}

DRY_RUN=0

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)         MODEL="$2"; shift 2 ;;
        -d|--dataset)       DATASET="$2"; shift 2 ;;
        -s|--seed)          SEED="$2"; shift 2 ;;
        --delay)            DELAY="$2"; shift 2 ;;
        --num-samples)      NUM_SAMPLES="$2"; shift 2 ;;
        --gpu-per-trial)    GPU_PER_TRIAL="$2"; shift 2 ;;
        --cpu-per-trial)    CPU_PER_TRIAL="$2"; shift 2 ;;
        --num-workers)      NUM_WORKERS="$2"; shift 2 ;;
        --n-epochs)         N_EPOCHS="$2"; shift 2 ;;
        --grace-period)     GRACE_PERIOD="$2"; shift 2 ;;
        --patience)         PATIENCE="$2"; shift 2 ;;
        --min-delta)        MIN_DELTA="$2"; shift 2 ;;
        --wandb-mode)       WANDB_MODE="$2"; shift 2 ;;
        --optimizing-metric) OPTIMIZING_METRIC="$2"; shift 2 ;;
        --no-asha)          ASHA=0; shift ;;
        --wandb-tag)        WANDB_TAGS+=("$2"); shift 2 ;;
        --partition)        PARTITION="$2"; shift 2 ;;
        --time)             TIME="$2"; shift 2 ;;
        --mem)              MEM="$2"; shift 2 ;;
        --gpu-type)         GPU_TYPE="$2"; shift 2 ;;
        --num-gpus)         NUM_GPUS="$2"; shift 2 ;;
        --cpus)             CPUS="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "Error: --model is required"; usage; }
[[ -z "$DATASET" ]] && { echo "Error: --dataset is required"; usage; }

# ---- Auto-compute SLURM resources ----
# Concurrent trials = num_gpus / gpu_per_trial
CONCURRENT=$(python3 -c "import math; print(math.floor($NUM_GPUS / $GPU_PER_TRIAL))")

# CPUs: use explicit --cpus if given, otherwise auto = concurrent*(cpu_per_trial+num_workers)
if [[ -n "$CPUS" ]]; then
    CPUS_NEEDED=$CPUS
else
    CPUS_NEEDED=$(python3 -c "print($CONCURRENT * ($CPU_PER_TRIAL + $NUM_WORKERS))")
fi

JOB_NAME="pmf_${MODEL}_${DATASET}_s${SEED}"

# Build run_combo.py arguments
COMBO_ARGS="-m $MODEL -d $DATASET -s $SEED"
COMBO_ARGS+=" --num-samples $NUM_SAMPLES"
COMBO_ARGS+=" --gpu-per-trial $GPU_PER_TRIAL"
COMBO_ARGS+=" --cpu-per-trial $CPU_PER_TRIAL"
COMBO_ARGS+=" --num-workers $NUM_WORKERS"
COMBO_ARGS+=" --n-epochs $N_EPOCHS"
COMBO_ARGS+=" --grace-period $GRACE_PERIOD"
COMBO_ARGS+=" --patience $PATIENCE"
[[ -n "$MIN_DELTA" ]] && COMBO_ARGS+=" --min-delta $MIN_DELTA"
COMBO_ARGS+=" --wandb-mode $WANDB_MODE"
COMBO_ARGS+=" --optimizing-metric $OPTIMIZING_METRIC"
[[ $ASHA -eq 0 ]] && COMBO_ARGS+=" --no-asha"
for _tag in "${WANDB_TAGS[@]}"; do COMBO_ARGS+=" --wandb-tag $_tag"; done
[[ -n "$DELAY" ]] && COMBO_ARGS+=" --delay $DELAY"

# ---- Print config summary ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  run_combo.slurm — submission summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model:              $MODEL"
echo "  Dataset:            $DATASET"
echo "  Seed:               $SEED"
echo "  Num samples:        $NUM_SAMPLES"
echo "  GPU per trial:      $GPU_PER_TRIAL"
echo "  CPU per trial:      $CPU_PER_TRIAL"
echo "  Num workers:        $NUM_WORKERS"
echo "  Concurrent trials:  $CONCURRENT"
echo "  N epochs:           $N_EPOCHS"
echo "  Grace period:       $GRACE_PERIOD"
echo "  Patience:           $PATIENCE"
echo "  Min delta:          ${MIN_DELTA:-auto (per metric)}"
echo "  Optimizing metric:  $OPTIMIZING_METRIC"
echo "  ASHA:               $([[ $ASHA -eq 1 ]] && echo 'on' || echo 'off')"
echo "  W&B mode:           $WANDB_MODE"
[[ ${#WANDB_TAGS[@]} -gt 0 ]] && echo "  W&B tags:           ${WANDB_TAGS[*]}"
echo "  ──────────────────────────────────────────────────────────"
echo "  SLURM partition:    $PARTITION"
echo "  SLURM GPUs:         ${NUM_GPUS}x ${GPU_TYPE}"
CPUS_LABEL=$([[ -n "$CPUS" ]] && echo "manual" || echo "auto: concurrent*(cpu+workers)")
echo "  SLURM CPUs:         $CPUS_NEEDED ($CPUS_LABEL)"
echo "  SLURM memory:       $MEM"
echo "  SLURM time:         $TIME"
echo "  Job name:           $JOB_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ---- Generate sbatch script ----
SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_NEEDED}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPUS}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=/scratch/c7031336/protomf_logs/%x_%j.out
#SBATCH --error=/scratch/c7031336/protomf_logs/%x_%j.err

# --- Environment ---
module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate

# --- Directories (all on scratch) ---
export RAY_TMPDIR=/scratch/c7031336/ray_tmp/\${SLURM_JOB_ID}
export RAY_RESULTS_DIR=/scratch/c7031336/ray_results
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data
export WANDB_DIR=/scratch/c7031336/wandb_logs
export WANDB_CACHE_DIR=/scratch/c7031336/wandb_cache
export WANDB_API_KEY=\$(cat /home/c703/c7031336/.wandb_key 2>/dev/null || echo "")
export WANDB_MODE=${WANDB_MODE}
export PROTOMF_RESULTS_PATH=/scratch/c7031336/protomf_results
export OMP_NUM_THREADS=1
export WANDB_START_METHOD=thread

mkdir -p \$RAY_TMPDIR \$RAY_RESULTS_DIR \$WANDB_DIR \$WANDB_CACHE_DIR \$PROTOMF_RESULTS_PATH

# --- Run ---
cd /home/c703/c7031336/UIFProtoMF/ProtoMF
echo "Starting: ${COMBO_ARGS}"
echo "Node: \$(hostname) | Job: \$SLURM_JOB_ID | \$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python Master/scripts/run_combo.py ${COMBO_ARGS}

echo "Finished: \$(date)"

# Cleanup Ray temp
rm -rf \$RAY_TMPDIR
EOF
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "=== DRY RUN — sbatch script: ==="
    echo "$SBATCH_SCRIPT"
    exit 0
fi

# ---- Submit ----
TMPFILE=$(mktemp /tmp/run_combo_XXXXXX.sbatch)
echo "$SBATCH_SCRIPT" > "$TMPFILE"
SUBMIT_OUTPUT=$(sbatch "$TMPFILE" 2>&1)
rm -f "$TMPFILE"

echo ""
echo "  $SUBMIT_OUTPUT"
echo ""
