#!/bin/bash
# dc09 SC.6 sizing — measured speed of the exact fast sampler at hyperopt concurrency (login-node
# A30, one-off). Each case: run_combo --profile smoke with <conc> trials sharing one GPU for 2
# epochs, from either the OLD code (fleet checkout, c0c6ac1) or the NEW code (sampler + control;
# a scratch copy of the laptop working tree). Per-epoch wall incl. validation is read afterwards
# from the trials' Ray progress.csv (time_this_iter_s). Usage: sc6_speed_ratio.sh <gpu> <case>...
#   case = <old|new>:<model>:<dataset>:<concurrency>
GPU=$1; shift
B=/scratch/c7031336/smoke_runs/dc09/sc6_speed
module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data WANDB_MODE=disabled OMP_NUM_THREADS=1 \
       PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU RAY_TMPDIR=/tmp/ray_s6_$GPU
mkdir -p $B/logs $RAY_TMPDIR
for c in "$@"; do
  IFS=: read V M DS CONC <<< "$c"
  if [ "$V" = old ]; then CODE=$HOME/UIFProtoMF/ProtoMF; else CODE=/scratch/c7031336/smoke_runs/sampler_check/repo; fi
  TAG=${V}_${M}_${DS}_c${CONC}
  export PROTOMF_RESULTS_PATH=$B/results/$TAG RAY_RESULTS_DIR=$B/ray_results/$TAG
  mkdir -p $PROTOMF_RESULTS_PATH $RAY_RESULTS_DIR
  GPT=$(python3 -c "print(round(1/$CONC, 4))")
  (cd $CODE && python Master/scripts/run_combo.py -m $M -d $DS -s 38210573 --profile smoke \
     --num-samples $CONC --n-epochs 2 --gpu-per-trial $GPT --num-workers 1 --wandb-mode disabled \
     --skip-explanations > $B/logs/$TAG.log 2>&1)
  echo "$TAG exit $?"
done
rm -rf $RAY_TMPDIR
echo SPEED_DONE
