---
date: 2026-06-16
time: "17:04"
phase: 2
---
# Citable justification for top-N eval design (no RMSE/AUC)

The S7 survey's §3.5.2 critique — AUC penalizes all rank positions equally, so high AUC can come from correctly ranking the irrelevant bottom-N, useless for top-N — gives a **citable, literature-grounded reason** to report position-aware metrics (`hit_ratio@10`, NDCG@K) and to **not** report RMSE/AUC. ProtoMF's eval already matches this (selects on hit_ratio@10, reports NDCG@K). Decision: add a methodology sentence citing Chen et al. (2020) §3.5.2 to pre-empt a reviewer asking why RMSE/AUC are absent. Note ProtoMF sits in the item-ranking branch (§3.5.2) while the survey's own experiments are rating-prediction/RMSE (§3.5.1) — cite the survey for taxonomy, not eval regime.

[[evaluation]] [[decisions]] [[paper-understanding]] [[phase-2]]
