---
date: 2026-07-11
time: "16:29"
phase: 4
tags: [thesis/eval-framework]
placed: 2026-08-20
---
# Baseline suite = published rosters (ProtoMF 5 + LightFM 2) under our protocol

S0.4 gate decision (scrutiny dossier §S0.4 rev. 1): the baseline suite copies the model rosters of the two source papers rather than a hand-rolled minimal set — deliberate credibility strategy (reviewers spend no scrutiny on copied published methodology, saving it for the novel contributions). **Roster:** the ProtoMF five (`mf`, `acf`, `user_proto`, `item_proto`, `user_item_proto` — already the reference fleet) + two B5/LightFM models chosen as *one strong feature-aware baseline* — **LightFM(tags+ids)**, B5's best variant — and *one feature-only baseline* — **LightFM(tags)** (item = attributes only, cold-capable by construction). LightFM(tags) beat LSI-LR for the feature-only slot on isolation grounds: it differs from our candidates in exactly one thing (no prototype layer), while LSI-LR differs in two (no prototypes AND content-only estimation). **Exclusions:** LightFM(tags+about) — needs user metadata text H&M lacks (user side closed per S0.2); LSI-LR/LSI-UP — dominated in every scenario in B5's own results, cited not re-run. Rosters are copied, the harness is not: all models run under the ProtoMF protocol (temporal LOO, 1+99 negatives, HR/NDCG, bias-free — declared deviation from B5's bias form, see [[2026-07-11_1556_bias-channel-popularity-prototype-interpretability]]). Implementation: in-repo `lightfm` ft_type reusing dc01's `FeatureEmbedding`, models `lightfm_tags` / `lightfm_tags_ids`, F=0-reduces-to-`mf` keystone test.

[[experiments]] [[evaluation]] [[decisions]] [[phase-4]]
