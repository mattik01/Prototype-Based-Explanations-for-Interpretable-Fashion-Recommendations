#!/bin/bash
# Cold-eval runner — the committed generator for every cold row in the frozen
# reference table and both candidates (learnings.md:18: measured numbers need a
# committed generator, or they are illustrative only).
#
# Usage (LEO5 login node, one model per invocation):
#   ./run_cold_evals.sh <model>
#
# Rows generated with this exact invocation:
#   2026-07-15  S0.7 reference wave — mf, acf, user_proto, item_proto,
#               user_item_proto, lightfm_tags, lightfm_tags_ids
#               (run as a for-loop over the same body; scratch copy
#               /scratch/c7031336/run_cold_evals.sh)
#   2026-08-15  dc05 — feature_user_proto, feature_user_proto_noid
#   2026-08-15  dc01 — feature_item_proto, feature_item_proto_noid
#
# The runner itself decides per model whether the attr-kNN fallback applies
# (CF rows only) and whether the ID-column drop is required (F-S0-07(b)) — see
# utilities/cold_eval.py. Placement per CLAUDE.md: login node, not SLURM
# (post-hoc measurement on trained checkpoints, no compared row is produced here;
# the compared rows are the retrains themselves, which go through /leo5-submit).
set -u
module purge
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data
export CUDA_VISIBLE_DEVICES=1
cd /home/c703/c7031336/UIFProtoMF/ProtoMF

m="$1"
r=/scratch/c7031336/protomf_results/${m}_hm_1_month_cold_s38210573
c=/scratch/c7031336/protomf_results/${m}_hm_1_month_s38210573
echo "=== COLD EVAL $m ==="
echo "results-dir:           $r"
echo "canonical-results-dir: $c"
python -m utilities.cold_eval --results-dir "$r" \
  --variant-path /scratch/c7031336/protomf_data/hm_1_month_cold \
  --canonical-results-dir "$c" --device cuda || echo "COLDEVAL FAILED: $m"
echo "COLDEVAL_DONE $m"
