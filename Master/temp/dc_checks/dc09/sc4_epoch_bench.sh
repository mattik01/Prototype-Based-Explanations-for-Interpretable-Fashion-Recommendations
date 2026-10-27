#!/bin/bash
# dc09 SC.4 optimisation-parity bench (login-node A30, dev worktree): 2 epochs per model under
# the matched configs written by `sc4_epoch_bench.py make`, one after another on one GPU.
B=/scratch/c7031336/smoke_runs/dc09/sc4_bench
GPU=${1:-0}
module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data PROTOMF_RESULTS_PATH=$B/results \
       RAY_RESULTS_DIR=$B/ray_results RAY_TMPDIR=/tmp/ray_b9_$GPU WANDB_MODE=disabled \
       OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $B/logs $RAY_TMPDIR $RAY_RESULTS_DIR
cd ~/UIFProtoMF/ProtoMF_dev
for M in user_item_proto feature_user_proto feature_item_proto feature_user_item_proto feature_user_item_proto_sep; do
  python Master/scripts/run_combo.py -m $M -d hm_1_month -s 38210573 --retrain-config $B/$M/config.json \
    --profile production --n-epochs 2 --gpu-per-trial 1.0 --num-workers 1 --wandb-mode disabled \
    --skip-explanations > $B/logs/$M.log 2>&1
  echo "$M exit $?"
done
rm -rf $RAY_TMPDIR
echo BENCH_DONE
