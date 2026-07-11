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

- [s0 S0.1] Docs drift from pipeline code on load-bearing details (dedup keys) — verify preprocessing claims against splitter/eval code, never against other docs.
- [s0 S0.1] Hidden aggregate invariants: `Evaluator` divides by `n_users` assuming one eval row per user — any evaluator/split extension must re-derive its divisor and assert row counts.
- [s0 S0.1] "Cold items" already leak into standard test sets via k-core + temporal LOO (amazon: 7.1% of test rows on untrained embeddings) — check this slice before attributing cold-start deltas to a mechanism.
- [s0 S0.3] When a spec derives from a seed paper, read that paper's experimental section BEFORE speccing — the first S0.3 draft contradicted B5's published protocol on two axes (fraction, ranking pool) and a user gate had to catch it. Anchor to published protocol; deviate only with written argument.
- [s0 S0-build] Numbers "measured" during a spec session are unreproducible unless the measurement script is committed WITH the spec — the S0.3 collateral numbers could not be regenerated and had to be superseded by the committed generator's draw. Commit the script, or mark the numbers illustrative.
- [s0 S0-build] Doc-derived expectations about data completeness (image subdir range "000–0NN") can themselves be the error — verify against the authoritative source inventory (zip central directory) before declaring data partial or re-downloading.
- [s0 S0.7] Gate-ratified build deviations must be edit-noted into the SPEC section they deviate from, or every future black-box audit re-flags them (attr-kNN patch target).
- [s0 S0.7] All-tie score rows are NOT "chance level" under bn.argpartition — introselect resolves ties deterministically (can read 0.0 or 1.0); any degenerate/reference scorer needs an explicit tie policy, and sigmoid saturation (logit ≥ ~16.6 → exactly 1.0 in fp32) creates such ties.
- [s0 S0.7] Absent-key config defaults are hazards: `RecSys` defaults `use_bias=True`, the factory defaults `use_id_feature=True` — parity hangs on explicit keys; check both in every new config (C8 manifests record use_bias).
- [s0 S0.7] Cross-model-row eval-draw identity (shared seeded RNG stream) holds only under sequential num_workers=0 access — a convention, not a guarantee; keep it pinned by a test (t05).
- [s0 S0.7] Split audits leave seams: a spec-A commitment implemented in component B's code is invisible to both auditors (S0.4's ID-drop convention lived in cold_eval.py → F-S0-13). When partitioning audit scopes, give each auditor the specs that NAME its files, and cross-check spec cross-references first-party.
- [dc01 SC.1b] A formally-correct adversary claim ("no term X in the loss") can still overclaim its consequence — indirect/secondary effects may deliver X anyway (reg_batch coverage separates prototypes without a repulsion term). Before conceding a mechanism-absence point, check the vault + existing empirical records (the reg-sweep had already demonstrated the effect); the user gate caught this one.
- [dc01 SC.1b] Fluent cross-level synthesis is where errors hide best (user flag): the risk is quantifier slippage — analogies read as equivalences ("intrinsic counterpart" ≠ "same thing"), conditional claims read as absolute ("canonical" = canonical-given-the-model, not across retrainings). Gate-born conceptual prose destined for the thesis gets an explicit SC.10 brief to attack overstated equivalences; self-audit the "gold" passages before they enter drafts.
- [dc01 SC.1b] Thresholds are production logic, not research (user principle): when a design wants a cutoff (grouping, bucketing, tie caps), first name the exact underlying quantity (e.g. loss curvature along redistribution directions), report its continuum with a cheap proxy, and let any discretization be a summary visualization carrying no scientific claim. "Production decides; research characterizes."
