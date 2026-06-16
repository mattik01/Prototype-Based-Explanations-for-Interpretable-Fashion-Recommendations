Run one full cycle of the Design Candidate Protocol — produce exactly one literature-grounded, in-depth architecture candidate for the feature-aware ProtoMF extension (Phase 4). No implementation, no training.

**This is a thin wrapper. The source of truth is `Master/docs/design_candidate_protocol.md` — read it; do not run from memory of it.**

## How to run

1. **Read the protocol in full** first: `Master/docs/design_candidate_protocol.md`. It is a living document — the procedure, gates, and version may have changed since last time.
2. Execute the protocol **exactly as written**, honoring its Ground rules — especially:
   - One step at a time; re-read the relevant Step section (a real Read call) before starting that step.
   - Artifact before progress: each step appends its section to the candidate file `Master/docs/design_candidates/dcNN_<slug>.md`.
   - No from-memory citations; verify against the local PDFs under `Master/literature/papers/`.
   - No cross-candidate ranking (per-requirement assessment only).
   - **Spillover capture (Ground rule 9):** when research surfaces a valuable adjacent idea or variant you can't pursue this cycle, append it to `Master/docs/design_candidates/scratchpad.md` as a `### [captured dcNN] <title>` entry instead of discarding it.
3. **Stop at Step 6.** Present a short summary and wait — the user reviews the finished candidate before the next cycle. Do not start another cycle.

## Arguments

- **no argument** → start a new full cycle (next free `dcNN`), beginning at Step 0.
- **`stepN dcMM`** (e.g. `step5 dc01`) → Re-run mode (see the protocol's "Re-run mode" section): re-run only Step N for the existing candidate `dcMM`, re-reading that step's section, redoing it against the current state of all inputs, and updating the candidate file section with a **dated edit note** plus the index row's date/status.

Inputs the protocol relies on: `Master/docs/feature_aware_solution_requirements.md` (requirements R1–R8/S1–S5), `Master/literature/feature_aware_reading_list.md` + PDFs, the candidate index and scratchpad under `Master/docs/design_candidates/`.
