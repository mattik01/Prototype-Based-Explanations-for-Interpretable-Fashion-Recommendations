---
date: 2026-07-11
time: "20:23"
phase: 4
---
# Cold-start eval methodology: anchored to LightFM (Kula 2015, B5 §5), deviations declared

Decision (ratified across the S0.3 gate + S0.7 discussion, 2026-07-11): the thesis's cold-item
evaluation is presented as *"Kula (2015) §5's cold-start protocol, adapted to a temporal split,
with validation-based stopping and a bootstrap in place of split repetitions."* This entry fixes
the exact adopt/deviate mapping for the methodology chapter.

**Adopted from B5 exactly:**
- Held-out construction: interactions of **20% of items** removed from training, becoming the
  cold test set — the model is *retrained on the reduced data* (never evaluated on a checkpoint
  that saw the cold items' interactions).
- **Cold-pool ranking** as the headline setting (B5: "items for which no collaborative
  information has been gathered") = our cold-vs-cold primary; cold-vs-all kept as secondary
  deployment-shaped diagnostic.

**Declared deviations (each argued in the S0.3 spec, `s0_foundation.md`):**
1. *Split lineage:* B5 removes items from an atemporal random split; we remove them from the
   frozen temporal leave-one-out artifact — host-protocol (ProtoMF) fidelity outranks B5, and
   cold numbers must share lineage with every other thesis number.
2. *Early stopping:* B5 stops on **test** performance (leakage by modern standards); we stop on
   warm validation — a declared improvement, not a compromise.
3. *Repetitions:* B5 runs 10 random splits; we run **one seeded split + per-item cluster
   bootstrap CIs** (scrutiny budget principle); 3 repetitions reserved for full-profile runs.
4. *Metric:* B5 reports mean ROC AUC; we keep HR/NDCG on the 1+99 protocol — sampled AUC is a
   linear function of mean rank (redundant), absolute AUC doesn't transfer to a 100-slot eval,
   and B5's anchoring lives in split design + ranking pool, not the metric. Chance floor is
   cleanly readable (HR@K = K/100 on cold-vs-cold).
5. *Config selection:* B5 fixes hyperparameters a priori; we transfer each model's **warm-only
   hyperopt winner** to a single cold retrain — selection provably blind to cold (variant val
   contains no cold positives), every model in its honestly-tuned form.

Fairness scaffold: configs warm-selected once, retrained once on `hm_1_month_cold`, all models
scored under identical seeded negative draws; baselines frozen at S0.7 (charter C6).

[[phase-4]] [[evaluation]] [[decisions]] [[data]]
