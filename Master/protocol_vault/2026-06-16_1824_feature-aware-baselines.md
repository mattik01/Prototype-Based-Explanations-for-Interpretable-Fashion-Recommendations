---
date: 2026-06-16
time: "18:24"
phase: 2
---
# Feature-aware baselines: FM / NFM / DeepFM / NCF+

Selected FM (Rendle et al. 2011), NFM (He & Chua 2017), DeepFM (Guo et al. 2017), and **NCF+** (NCF with attributes concatenated, He et al. 2017) as the natural **feature-aware baselines** to compare the feature-aware ProtoMF extension against. Rationale: they are the established way to add heterogeneous/sparse side features to CF (S7 §4.3 Generalized Factorization) — FM handles sparse one-hot categoricals (exactly H&M item attributes) via shared per-feature latent vectors; NFM/DeepFM add neural non-linear interactions; **NCF+ is a strong *and simple* feature-aware baseline — it was generally the best performer in S7's own experiments (§5.2)**. They make the right control: "does feature-grounded prototype interpretability cost recommendation quality vs. standard feature-aware models?" Note positioning contrast: this family is feature-centric (a vector per feature, interactions via dot products / MLPs) with no interpretable anchors, vs ProtoMF's prototype-centric similarity-to-anchors — the "expressive but opaque" foil.

[[experiments]] [[decisions]] [[phase-4]] [[phase-2]]
