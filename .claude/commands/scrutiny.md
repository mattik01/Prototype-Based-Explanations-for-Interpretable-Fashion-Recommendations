Run the Scrutiny Protocol — audit **and repair** the shared evaluation
foundation (S0) or one design candidate (SC(dcNN)): concept, implementation,
explanations, cluster-run readiness. Always scrutinize the steel-manned (best
current) version; fix along the way; every step ends at a user review gate.

**This is a thin wrapper. The source of truth is
`Master/docs/scrutiny_protocol.md` — read it in full; do not run from memory
of it.**

## How to run

1. **Read the protocol in full** first: `Master/docs/scrutiny_protocol.md`.
   It is a living document — steps, gates, and version may have changed.
2. **Verify the branch:** all scrutiny work happens on the integration branch
   `feat/scrutiny` (Ground rule 4 — single code lineage). If not on it, stop
   and resolve with the user.
3. **Load the context manifest** for the session type (the protocol's
   "Context-loading manifest" section), including the open-findings re-check
   and the learnings ledger.
4. Execute the requested step(s) **exactly as written**, honoring the Ground
   rules — especially: one step at a time with a mandatory re-read of the
   step's section; artifact before progress (append to the dossier under
   `Master/docs/scrutiny/`); steel-man rule; fix policy (trivial = fix now +
   log; else finding bundled with proposed fix → one gate); adversarial
   subagents get ONLY their named artifacts; no cross-candidate ranking.
5. **Stop at every gate.** Present the step's artifact and wait for the
   user's review before the next step.

## Arguments

- **`s0`** → run the foundation scrutiny from its next unfinished step
  (S0.1 → … → S0.7, per the dossier `Master/docs/scrutiny/s0_foundation.md`).
- **`dcNN`** (e.g. `dc01`) → run that candidate's routine from its next
  unfinished step (SC.1a → … → SC.11, per its dossier `scNN_<slug>.md`).
- **`stepX dcNN`** / **`stepX s0`** (e.g. `step4 dc02`, `step0.3 s0`) →
  Re-run mode: redo only that step against current inputs, dated edit note in
  the dossier.
- **`resume dcNN`** → post-pause re-entry (cluster runs finished): re-run the
  context load, then continue at the next unfinished post-run step (SC.8
  onward, per the dossier).

Inputs the protocol relies on: `Master/docs/feature_aware_solution_requirements.md`,
`Master/docs/design_candidates/` (index + candidate docs),
`Master/docs/scrutiny/` (dossiers, findings ledger, learnings ledger), and the
per-step manifests in the protocol doc.
