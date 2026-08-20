# Scrutiny Learnings Ledger

> Maintained by the Scrutiny Protocol (`Master/docs/scrutiny_protocol.md`,
> Ground rule 8). Watch-outs, recurring error patterns, and high-yield checks
> that later scrutiny runs must inherit. Loaded as mandatory input at every
> SC.1a.
>
> Rules: **watch-outs and patterns only** — design ideas go to
> `Master/docs/design_candidates/scratchpad.md` as `[captured scrutiny dcNN]`
> entries. Compact at every flush (SC.7 / SC.11): merge duplicates,
> generalize candidate-specific detail. Hard cap: ~one screen. Each entry is
> 1–3 lines, tagged with its origin step (e.g. `[dc01 SC.4]`).
>
> **Full compaction executed at dc01 SC.11 (2026-08-20):** 31 entries merged
> to 21; no substance dropped — origin tags preserved, candidate-specific
> detail generalized. Pre-compaction version: git history (commit before the
> SC.11 commit).

- [s0 S0.1 + dc05 review + s0 S0-build] Docs (and remembered half-facts) drift from code on load-bearing details — dedup keys, core levels, data-completeness ranges. Any sentence naming a mechanism parameter is a preprocessing claim: verify against code/artifacts/authoritative inventories, never against other docs or memory; hallucination enters through LABELS fused to correct numbers (vault 2026-07-12_1659).
- [s0 S0.1 + S0.7] Hidden aggregate invariants and degenerate scorers: `Evaluator` divides by `n_users`; all-tie rows resolve deterministically under bn.argpartition (can read 0.0 or 1.0); fp32 sigmoid saturates (logit ≥ ~16.6) and FABRICATES ties the raw ranking lacks. Any evaluator/split extension re-derives its divisor and pins a tie policy; degenerate no-signal blocks are quoted as chance-floor brackets, never as 0.00 points (F-S0-15 rules).
- [s0 S0.1] "Cold items" already leak into standard test sets via k-core + temporal LOO (amazon 7.1%) — check the train-unseen slice before attributing cold deltas to a mechanism.
- [s0 S0.3] When a spec derives from a seed paper, read that paper's experimental section BEFORE speccing; deviate only with written argument.
- [s0 S0-build] Numbers measured in a session are unreproducible unless the generator script is committed WITH the spec — commit the script or mark the numbers illustrative.
- [s0 S0.7] Gate-ratified build deviations must be edit-noted into the SPEC section they deviate from, or every future black-box audit re-flags them.
- [s0 S0.7] Absent-key config defaults are hazards (`use_bias`/`use_id_feature` default True) — parity hangs on explicit keys; cross-row conventions (shared eval RNG stream, num_workers=0 sequential access) are conventions, not guarantees — keep them pinned by tests.
- [s0 S0.7] Split audits leave seams: a spec-A commitment implemented in component B's code is invisible to both auditors — give each auditor the specs that NAME its files, cross-check spec cross-references first-party.
- [dc01 SC.1b] A formally-correct adversary claim can still overclaim its consequence (indirect/secondary effects may deliver the "absent" mechanism — reg_batch coverage separates prototypes without a repulsion term); before conceding mechanism-absence or a generic function-class reduction ("model X = model Y"), check normalization details, regularizer visibility, and the vault/empirical record.
- [dc01 SC.1b + SC.10] Fluent cross-level synthesis hides quantifier slippage (analogy ≠ equivalence; conditional ≠ absolute; canonical-given-model ≠ canonical); the sibling failure mode in ratings: grades that quietly narrow their requirement (mechanism vs delivered, derivability vs quality, staged vs primary bar) survive self-audit but not a cold read — write the narrowed scope into the grade text itself; SC.10 briefs attack both.
- [dc01 SC.1b] Thresholds are production logic, not research: name the exact underlying continuum, report it with a cheap proxy, let any discretization be a summary visualization carrying no claim.
- [dc01 SC.3] Scrutiny-phase measurements are working evidence for protocol decisions only — never pre-approved thesis material; thesis treatment gets dedicated experiments + a mathematical argument designed at writing time. Flag the restriction at acceptance time.
- [dc01 SC.1b-delta + dc05 5b/4b] Host-inherited surfaces still need explanation-validity audit — "host behavior" ≠ "no mode" when thesis weight rests on that surface; every host swap re-derives its hidden effects first-party (nothing "carried" survives unexamined); write "conserved to machine precision", never "bitwise-frozen".
- [dc05 SC.1b] Work empirically, not scholastically: concept-stage analysis produces hypotheses PLUS their designed tests; attributive readings are decided by comparative instruments run on both arms, never pre-assigned by argument. Corollary: a toy supports only claims about the population structure it actually contains (a homogeneous toy can neither exhibit nor rebut heterogeneity/crowding).
- [s0 build-ext] Empty-width tensors make slicing-derived shapes silently wrong (0-size broadcasting zeroes instead of erroring) — construct auxiliary columns from SHAPE, never by slicing; keystone reductions (F=0) are the tests that catch this class.
- [dc05 SC.3 + dc01 SC.8] Migrations (config schema, function signatures) must regression-run the SIDE harnesses, not only pinned tests — anything that hand-feeds configs to Trainer/Tester or unpacks changed returns is blast radius (twice observed: dc05 smoke_train, F-S0-16 readout harnesses).
- [dc05 SC.4] Read-out/label layers consume the BUILDER's own vocab/metadata mapping, never re-derive it; when a new dataset/layout enters scope, audit the EXPLANATION surface for hidden layout assumptions, not just the training path.
- [fU-cycle dc05 build] A shared-frame renderer's layout tolerances are part of its contract: render every new slot on toy data and eyeball BEFORE landing; fix by shortening slot-owned strings, never by reshaping the shared frame.
- [dc01 SC.6] Cluster ops correctness bundle: SLURM allocation size is a correctness factor (CPU auto-formula collapses at gpu-per-trial 1.0 — floor the CPU request; "no ray_results + flat GPU telemetry" = allocation symptom); a SIGKILLed job's .out testifies to nothing (PYTHONUNBUFFERED=1; diagnose from artifact dirs + telemetry); size walltimes from measured same-tier fleet anchors, not probe ratios; a login-node repro is valid only if the A30s were actually free — check occupancy FIRST.
- [s0 S0.7] Dataset-presence checks must verify the FILES a run consumes, not the dir; new locally-built artifacts need an explicit scp step.
- [dc01 SC.9/SC.10] A results-memo drafted from the dossier alone misses pause-period vault entries — sweep the vault directly before drafting. Present every in-scope testbed at equal weight (tables for both, explicit scope statements where only one exists); disclose committed secondary eval regimes (cold-vs-all) at the headline AND in limitations — the secondary numbers existed and disclosure strengthened the honest arm.
