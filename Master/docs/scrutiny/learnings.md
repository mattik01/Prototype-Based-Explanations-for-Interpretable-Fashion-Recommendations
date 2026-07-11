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
