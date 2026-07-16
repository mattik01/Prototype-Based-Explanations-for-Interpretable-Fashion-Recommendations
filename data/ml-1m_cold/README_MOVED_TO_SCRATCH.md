# ml-1m_cold — moved to LEO5 scratch (2026-07-16)

The dataset files that lived here were copied to
`leo5:/scratch/c7031336/protomf_data/ml-1m_cold/` and deleted locally
(user decision, 2026-07-16 — the variant is not consumed by any local
work; it is pinned to the future ml-1m cold-machinery enablement
work-package, see the dc05 SC.6 dossier, decision 1).

Files moved (9, combined md5-of-md5s `00ed00b52c658a0ad754cbe1fdf54b46`,
verified identical on scratch before local deletion):
cold_items.csv, cold_test.csv, cold_variant_report.json,
item_features.csv, item_ids.csv, listening_history_test.csv,
listening_history_train.csv, listening_history_val.csv, user_ids.csv

Note: scratch is not backed up. If the copy is ever lost, the variant is
deterministically regenerable from the canonical `data/ml-1m/` artifact
via the committed S0.3 cold-variant generator (seeded; see the S0.3
addendum in `Master/docs/scrutiny/s0_foundation.md`).
