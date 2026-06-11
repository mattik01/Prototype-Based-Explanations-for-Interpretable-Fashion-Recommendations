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
