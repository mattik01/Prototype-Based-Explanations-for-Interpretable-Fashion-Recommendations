# Design Candidate Index

> Maintained by the `/design-candidate` protocol
> (`Master/docs/design_candidate_protocol.md`). One row per candidate.
> Fingerprint axes: **features-entry × grounding × read-out** — a new candidate
> must differ from every row on at least one axis.
> No overall scores or rankings here by design (see protocol ground rule 7);
> cross-candidate judgement is the user's.

| id | title | seed paper(s) | provenance | fingerprint (entry × grounding × read-out) | tags | status | date |
|----|-------|---------------|------------|--------------------------------------------|------|--------|------|
| dc01 | Feature-composed item factors under the double-tied prototype layer | B5 LightFM × B3 SVDFeature × A1 ProtoMF | corpus | features→factors (sum of feature embeddings replaces item ID table) × prototype profile via shared geometry cos(e_f, p_k), unenforced × per-prototype contribution + exact per-feature shares of activations | — | draft — awaiting review | 2026-06-11 (steps 3+5 retrofit to protocol v1.1 same day) |
| dc02 | Attribute-space item prototypes (concept-bottleneck-anchored prototype layer) | E1 CBM × A1 ProtoMF (+ C2 formulations) | corpus | prototypes live in attribute space (item branch = bottleneck over observed multi-hot; no per-item params) × grounding by construction (prototype = attribute-value weight profile a_k ∈ R^V) × per-prototype item-discriminating contributions û_k·(t*−1) + exact per-attribute shares; profiles read off A/W_t | — | draft — awaiting review | 2026-07-07 |
| dc03 | Visual exemplar prototypes (push-grounded item prototypes over frozen image content) | C1 ProtoPNet × H1b VBPR × A1 ProtoMF (+ H2 precedent) | corpus | item repr = E·(frozen image embedding), no item ID table; prototypes in projected visual space × exemplar identity via periodic push onto real catalog garments (exact post-push) × case-based scoring sheet: û_k·t*_k per prototype, rendered as exemplar photo + catalog row; no per-attribute shares | — | draft — awaiting review | 2026-07-07 |
| dc04 | Attribute-anchored prototypes (item prototypes = convex combinations of attribute-value centroids of the live item table) | none — model-independent cycle (ingredient relatives: protonets class means, TCAV, VQ codebooks; probe neighbours MMF, SAGE) | model-knowledge | features enter at the prototype parameterization (P^t = softmax(Ã)·C, entities stay free CF, item ID table intact) × grounding by construction via data anchoring (c_v = live class means, epoch-refreshed; preimage-identifiability caveat 5b/A1) × per-prototype contributions û_k·(t*_k−1) + profile a_k + exact per-anchor shares in behavioral vocabulary; cold items via anchor-average rule | — | draft — awaiting review | 2026-07-07 |
