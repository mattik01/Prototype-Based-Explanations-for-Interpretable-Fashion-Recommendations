# Wave 1 — 100-trial fleet, submitted 2026-10-26

All: `--profile production` (100 trials / 100 epochs / patience 10 / grace 4), ASHA on, seed 38210573,
`--num-workers 1`, `--gpu-type any`, `--wandb-tag thesis-grade`, cluster checkout at `3ce563d`
(feat/scrutiny). Results land in `/scratch/c7031336/protomf_results/<model>_<dataset>_s38210573_n100/`.
Logs: `/scratch/c7031336/protomf_logs/pmf_<model>_<dataset>_s38210573_<jobid>.out`.

Sizing basis: measured 30-trial walls at concurrency 5 (A30) × 4.8–5 for 100 trials, concurrency
scaling, 2× margin. Heavy class (avg GPU% > 50 at conc 5) → concurrency 5, `--cpus 6`, `--mem 36G`;
light class → concurrency 8, `--cpus 9`, `--mem 44G`. Cluster state at submission: all 42 A30 + 4 A40
held (releases from ~Oct 28 morning, bulk ~Oct 29); 6 A100 idle but their nodes' SLURM-allocatable RAM
exhausted by CPU jobs (1–13 GB left, for 1.5–9 days). Earliest realistic start: Oct 28.

| # | job id | model | dataset | conc | --cpus | --mem | --time | est. wall |
|---|---|---|---|---|---|---|---|---|
| 1 | 8153730 | user_item_proto | hm_1_month | 5 | 6 | 36G | 2-00:00:00 | ~17 h |
| 2 | 8153731 | feature_item_proto | hm_1_month | 5 | 6 | 36G | 1-12:00:00 | ~13 h |
| 3 | 8153732 | feature_item_proto_noid | hm_1_month | 5 | 6 | 36G | 1-12:00:00 | ~12 h |
| 4 | 8153733 | item_proto | hm_1_month | 5 | 6 | 36G | 1-12:00:00 | ~11 h |
| 5 | 8153734 | item_proto | ml-1m | 5 | 6 | 36G | 2-00:00:00 | ~16 h |
| 6 | 8153735 | user_proto | hm_1_month | 8 | 9 | 44G | 1-12:00:00 | ~12 h |
| 7 | 8153736 | mf | hm_1_month | 8 | 9 | 44G | 1-00:00:00 | ~10 h |
| 8 | 8153737 | feature_user_proto | hm_1_month | 8 | 9 | 44G | 1-00:00:00 | ~10 h |
| 9 | 8153738 | feature_user_proto_noid | hm_1_month | 8 | 9 | 44G | 1-00:00:00 | ~10 h |
| 10 | 8153739 | lightfm_tags | hm_1_month | 8 | 9 | 44G | 1-12:00:00 | ~11 h |
| 11 | 8153740 | lightfm_tags_ids | hm_1_month | 8 | 9 | 44G | 1-00:00:00 | ~10 h |
| 12 | 8153741 | lightfm_hist | hm_1_month | 8 | 9 | 44G | 1-12:00:00 | ~12 h |
| 13 | 8153742 | lightfm_hist_ids | hm_1_month | 8 | 9 | 44G | 1-00:00:00 | ~8 h |
| 14 | 8153743 | mf | ml-1m | 8 | 9 | 44G | 1-00:00:00 | ~6 h |
| 15 | 8153744 | lightfm_hist_ids | ml-1m | 8 | 9 | 44G | 1-00:00:00 | ~7 h |

Wave 2 (after wave 1): cold single-trial retrains (`--retrain-config`, `--profile dev`) of each new
best config on `hm_1_month_cold` / `ml-1m_cold`. fUfI runs follow the dc09 scrutiny cycle (own
worktree `feat/fufi`; no pull into the main cluster checkout while any job above is pending).

Open check (cheap, login node): `start_testing` on `user_item_proto_ml-1m_s38210573_n100/best_model.pth`
under HEAD should reproduce HR@10 0.6246 / NDCG@10 0.3573 bit-for-bit → confirms the May 21 host row
and all older `_n100` rows share today's warm instrument.
