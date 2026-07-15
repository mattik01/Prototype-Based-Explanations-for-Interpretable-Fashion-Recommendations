---
date: 2026-07-15
time: "09:43"
phase: 4
---
# S0.7 fleet completed cleanly — dev-tier baseline table on hm_1_month

The full S0.7 baseline fleet (7 jobs, seed 38210573, `hm_1_month`) completed on LEO5 on 2026-07-14 between 11:51 and 15:21 — every job exited 0 with `RUN_OK: 30/30 trials`, all artifacts persisted (`best_model.pth`, config, val/test metrics, hardware logs, explanations for the proto models) under `/scratch/c7031336/protomf_results/<model>_hm_1_month_s38210573/`, W&B synced. Test-set results:

| Model | NDCG@10 | Hit Ratio@10 | Val HR@10 | Wall time |
|---|---|---|---|---|
| `mf` | 0.2233 | 0.3652 | 0.3149 | 2h46m |
| `acf` | 0.3307 | 0.5764 | 0.5884 | 3h00m |
| `user_proto` | 0.3322 | 0.5725 | 0.5824 | 3h22m |
| `item_proto` | 0.2631 | 0.4938 | 0.4993 | 2h09m |
| `user_item_proto` | 0.4164 | 0.6587 | 0.6815 | 3h21m |
| `lightfm_tags` | 0.2709 | 0.4582 | 0.4835 | 3h16m |
| `lightfm_tags_ids` | 0.3063 | 0.5087 | 0.5334 | 2h47m |

Ordering is sane: `user_item_proto` clearly leads, full ProtoMF > single-sided variants > plain MF, and the ID-augmented LightFM beats tags-only; val and test rankings agree. Per the two-tier numbers policy (2026-07-14) these are the working/dev-tier numbers — comparison-valid across the fleet, explicitly **not** thesis numbers (thesis-grade tier queued after model freeze). Next: fold into `replication_report.md` and freeze the S0 table, closing S0.

[[phase-4]] [[experiments]] [[evaluation]]
