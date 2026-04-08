#!/bin/bash
# Submit all experiment jobs. Edit COMBOS to control what runs.
# Usage: bash slurm/submit_all.sh

SCRIPT="slurm/run_combo.sbatch"
SEEDS=(38210573 9491758 2931009)

submit() {
    local model=$1 dataset=$2 gpu=$3 time=$4
    for seed in "${SEEDS[@]}"; do
        echo "Submitting: $model x $dataset x seed=$seed -> $gpu (${time})"
        sbatch --gres=gpu:${gpu}:1 \
               --time=$time \
               --job-name="pmf_${model}_${dataset}_s${seed}" \
               $SCRIPT $model $dataset $seed
    done
}

# --- GPU assignment ---
# A30 (24 GB): mf, acf, user_proto (lightweight, < 5 GB VRAM)
# A100 (80 GB): item_proto, user_item_proto (10-13 GB VRAM, safe headroom for 16 concurrent)

# --- Replication: incomplete combos ---
submit item_proto      amazon2014   a100  1-00:00:00
submit item_proto      ml-1m        a100  1-00:00:00
submit user_item_proto ml-1m        a100  1-00:00:00

# --- H&M 3-month ---
submit mf              hm_3_month   a30   2-00:00:00
submit acf             hm_3_month   a30   2-00:00:00
submit user_proto      hm_3_month   a30   2-00:00:00
submit item_proto      hm_3_month   a100  3-00:00:00
submit user_item_proto hm_3_month   a100  3-00:00:00

# --- H&M full ---
submit mf              hm_full      a30   5-00:00:00
submit acf             hm_full      a30   5-00:00:00
submit user_proto      hm_full      a30   5-00:00:00
submit item_proto      hm_full      a100  7-00:00:00
submit user_item_proto hm_full      a100  7-00:00:00
