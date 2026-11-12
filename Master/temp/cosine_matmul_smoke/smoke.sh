#!/bin/bash
# PREPARED, NOT YET RUN (2026-11-12). End-to-end smoke of the cosine_matmul switch on the LEO5
# login node: 2-epoch fixed-config retrain of the tuned I-ProtoMF ML-1M configuration (_n100),
# switch off / on, same seed, W&B disabled, explanations skipped. Not a compared row.
# Needs the switch code on the cluster first (see SYNC below). Modelled on
# /scratch/c7031336/smoke_runs/wave2_check/check.sh.
#
# SYNC (after commit + push from the laptop; dev worktree, main checkout untouched):
#   ssh leo5 'cd ~/UIFProtoMF/ProtoMF && git fetch origin && \
#             git -C ../ProtoMF_dev checkout --detach origin/feat/scrutiny && \
#             git -C ../ProtoMF_dev log --oneline -1'
# INSTALL + RUN (one arm per A30, each alone on its GPU):
#   scp Master/temp/cosine_matmul_smoke/smoke.sh leo5:/scratch/c7031336/smoke_runs/cosine_matmul_2026-11-12/
#   ssh leo5 'cd /scratch/c7031336/smoke_runs/cosine_matmul_2026-11-12 && chmod +x smoke.sh && \
#             nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv && \
#             (./smoke.sh 0 off > off.out 2>&1 &) ; ./smoke.sh 1 on > on.out 2>&1 ; sleep 20; cat off.out on.out'
# CHECKS on the new code path (cluster platform, where the dc01 golden is strict):
#   ssh leo5 'cd ~/UIFProtoMF/ProtoMF_dev && module purge && \
#             module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02 && \
#             source /scratch/c7031336/venvs/protomf/bin/activate && \
#             python Master/temp/dc_checks/cosine_matmul/t01_cosine_matmul_switch.py | grep -v "^Built\|^- \|^$" | tail -8'
#
# usage: smoke.sh <gpu index> <off|on>
set -u
GPU=$1; ARM=$2
W=/scratch/c7031336/smoke_runs/cosine_matmul_2026-11-12
CODE=${CODE:-/home/c703/c7031336/UIFProtoMF/ProtoMF_dev}
SRC=/scratch/c7031336/protomf_results/item_proto_ml-1m_s38210573_n100/config.json

module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data PROTOMF_RESULTS_PATH=$W/results \
       RAY_RESULTS_DIR=$W/ray_results RAY_TMPDIR=/tmp/ray_cm_$GPU WANDB_MODE=disabled \
       OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU GPU_LOG_DIR=$W/gpu_logs_$ARM
mkdir -p "$RAY_TMPDIR" "$RAY_RESULTS_DIR" "$W/logs" "$W/results"

if [ "$ARM" = on ]; then
  # switch-on copy of the tuned config lives in the scratch smoke folder, never in the repo
  CFG=$W/config_matmul.json
  python - "$SRC" "$CFG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["ft_ext_param"]["item_ft_ext_param"]["cosine_matmul"] = True
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
else
  CFG=$SRC
fi

# run_combo's own hardware_summary.json records 0 MiB VRAM on the login node (wave2_check), so
# sample nvidia-smi directly; --id is the physical index, independent of CUDA_VISIBLE_DEVICES.
nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$W/logs/gpu_mem_$ARM.log" 2>&1 &
NVPID=$!

cd "$CODE" || exit 2
echo "code: $(git log --oneline -1)   arm: $ARM   gpu: $GPU   config: $CFG"
START=$(date +%s)
python Master/scripts/run_combo.py -m item_proto -d ml-1m -s 38210573 --retrain-config "$CFG" \
  --profile production --n-epochs 2 --gpu-per-trial 1.0 --num-workers 1 --wandb-mode disabled \
  --skip-explanations --results-suffix cos_$ARM > "$W/logs/retrain_$ARM.log" 2>&1
echo "$ARM retrain exit $? wall $(( $(date +%s) - START ))s"
kill $NVPID 2>/dev/null
rm -rf "$RAY_TMPDIR"

echo "peak GPU memory (MiB): $(sort -n "$W/logs/gpu_mem_$ARM.log" | tail -1)"
grep -a -o "cosine_matmul: True.*\|Epoch [0-9]* - Train Loss.*\|Epoch [0-9]* - Avg Val Value.*" "$W/logs/retrain_$ARM.log"
# results: off -> $W/results/item_proto_ml-1m_s38210573_n100_cos_off (source folder carries the _n100 tier)
#          on  -> $W/results/item_proto_ml-1m_s38210573_cos_on       (config copy has no source run folder)
for f in "$W"/results/item_proto_ml-1m_s38210573*_cos_$ARM/val_metrics.json; do
  echo "$f"; python -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: round(v, 6) for k, v in d.items() if k in ('hit_ratio@10', 'ndcg@10')})" "$f"
done
