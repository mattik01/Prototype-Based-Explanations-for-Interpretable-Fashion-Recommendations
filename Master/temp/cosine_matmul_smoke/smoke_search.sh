#!/bin/bash
# Bigger smoke (2026-11-12), login node: ONE SHORTENED CELL of the planned §2.3.3 grid — an
# I-ProtoMF ML-1M search with the two regulariser weights held fixed (--fix) and the
# cosine_matmul switch on. 20 trials, at most 6 epochs each, ASHA on. Not a compared row:
# it measures how trials share one A30 with the switch on (epoch time, GPU memory, wall).
# usage: smoke_search.sh <gpu index> <trials at a time: 5|10>
set -u
GPU=$1; CONC=$2
case $CONC in 5) GPT=0.2 ;; 10) GPT=0.1 ;; *) echo "trials at a time must be 5 or 10"; exit 2 ;; esac
W=/scratch/c7031336/smoke_runs/cosine_matmul_2026-11-12
CODE=${CODE:-/home/c703/c7031336/UIFProtoMF/ProtoMF_dev}
TAG=mm_c$CONC

module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data PROTOMF_RESULTS_PATH=$W/results \
       RAY_RESULTS_DIR=$W/ray_results_$TAG RAY_TMPDIR=/tmp/ray_$TAG WANDB_MODE=disabled \
       OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU GPU_LOG_DIR=$W/gpu_logs_$TAG
mkdir -p "$RAY_TMPDIR" "$RAY_RESULTS_DIR" "$W/logs" "$W/results"

nvidia-smi --id=$GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 \
  > "$W/logs/gpu_$TAG.log" 2>&1 &
NVPID=$!

cd "$CODE" || exit 2
echo "code: $(git log --oneline -1)   trials at a time: $CONC   gpu: $GPU"
START=$(date +%s)
python Master/scripts/run_combo.py -m item_proto -d ml-1m -s 38210573 \
  --profile smoke --num-samples 20 --n-epochs 6 --gpu-per-trial $GPT --cpu-per-trial 1 --num-workers 1 \
  --wandb-mode disabled --skip-explanations --results-suffix $TAG \
  --fix ft_ext_param.item_ft_ext_param.cosine_matmul=true \
  --fix ft_ext_param.item_ft_ext_param.sim_proto_weight=0.5 \
  --fix ft_ext_param.item_ft_ext_param.sim_batch_weight=0.5 > "$W/logs/search_$TAG.log" 2>&1
echo "search exit $?   wall $(( $(date +%s) - START )) s"
kill $NVPID 2>/dev/null
rm -rf "$RAY_TMPDIR"

echo "peak GPU memory (MiB): $(cut -d, -f1 "$W/logs/gpu_$TAG.log" | sort -n | tail -1)   mean GPU util (%): $(awk -F, '{s+=$2; n++} END {if (n) printf "%.0f", s/n}' "$W/logs/gpu_$TAG.log")"
grep -a -o "Epoch [0-9]* - Train Loss [0-9.]* - [0-9]*s" "$W/logs/search_$TAG.log" \
  | awk '{t=$NF; sub("s","",t); s+=t; n++; if (t+0>m) m=t+0} END {if (n) printf "epochs trained: %d   mean epoch: %.0f s   longest epoch: %d s\n", n, s/n, m}'
grep -a "fixed (not searched)\|RUN_OK\|RUN_FAILED\|Total runtime\|Avg CPU\|Peak RAM" "$W/logs/search_$TAG.log" | head -8
