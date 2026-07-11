# Scrutiny Findings Ledger

> Maintained by the Scrutiny Protocol (`Master/docs/scrutiny_protocol.md`).
> One entry per non-trivial finding; trivial fixes get a one-line entry.
> **This ledger owns the WHY** (finding, severity, disposition, resolving
> commit); `Master/docs/modifications_log.md` owns the file-centric WHAT,
> cross-referenced by finding ID. Rationale is written once, here.
>
> **Schema per entry:**
> - ID: `F-S0-NN` (foundation) or `F-DC0N-NN` (candidate-scoped), sequential
> - applies-to: `shared` | `dc01`..`dc04` — `shared` findings are re-checked
>   at every later candidate's SC.1a
> - severity: `trivial` | `minor` | `major` | `blocker`
> - status: `open` → `fixed(<commit>)` | `wontfix` | `superseded(<by-ID>)` | `stale`
> - body: the finding, then `**Proposal:**` (the bundled fix), then
>   `**Disposition:**` (gate decision + date)
>
> Entry template:
>
> ```
> ### F-DC0N-NN [severity] [applies-to] [status]
> <one-sentence finding>
> <evidence / where>
> **Proposal:** <the bundled fix, implementation-ready>
> **Disposition:** <gate decision, date, resolving commit if fixed>
> ```

## Foundation (F-S0-…)

### F-S0-01 [trivial] [shared] [fixed]
`hm_feature_analysis.md` Part E claimed the splitter dedups `(customer, article, date)`; `data/hm/hm_splitter.py` dedups `(customer_id, article_id)` keep-first (ALL repeat purchases removed, not just same-day). Doc corrected to match code (S0.1, 2026-07-11). Commit: see S0.1 artifact commit.

### F-S0-02 [minor] [shared] [open]
Eval/train negative sampling uses the global numpy RNG inside DataLoader workers, which fork without numpy reseeding: with `num_workers>0` both workers run identical RNG streams and the parent state never advances, so val negatives are bit-identical across epochs; with `num_workers=0` (Windows) negatives are re-drawn every epoch — two different eval regimes by platform.
Evidence: `rec_sys/protomf_dataset.py` `_neg_sample_uniform`/`_neg_sample_popular` (np.random, no `worker_init_fn`); `utilities/consts.py` `NUM_WORKERS` platform switch.
**Proposal:** document-only (this entry + S0.1 dossier §3) and **wontfix**: all scrutiny runs are LEO5-only (`num_workers=2`, consistent regime); cross-worker stream duplication is masked per-user by the exclusion vector and has no plausible metric impact; the epoch-fixed val negatives are actually beneficial for early-stopping stability. Alternative (not recommended): `worker_init_fn` with fixed per-worker numpy seed — would shift every sampled metric and buy nothing scientific.
**Disposition:** _pending gate._

### F-S0-03 [minor] [shared] [open]
`Evaluator.get_results()` divides accumulated metric sums by `n_users`, silently assuming every user contributes exactly one eval row. Holds today (verified: ml-1m 6,034; amazon2014 6,950; hm_3_month 256,701 val=test=n_users; hm_1_month pending — F-S0-05), but any future split/eval change (cold-start eval!) breaks it silently.
Evidence: `utilities/eval.py:87`; `trainer.val()` / `tester.test()` construct `Evaluator(dataset.n_users)`.
**Proposal:** S0-build adds a cheap assert (total evaluated rows == n_users) in `Trainer.val()` and `Tester.test()` (no behavior change while the invariant holds; test in `dc_checks/s0/`). Constraint recorded for S0.3: the cold-start evaluator must define its own divisor, never inherit this one blindly.
**Disposition:** _pending gate._

### F-S0-04 [minor] [shared] [open]
Standard test sets already contain train-unseen items scored on random-init (never-trained) ID embeddings: amazon2014 101 items → 490/6,950 test rows (**7.1%**, +140 val rows); hm_3_month 23/256,701 rows; ml-1m 0. Paper-faithful accident of k-core + temporal LOO (k-core counts include val/test occurrences).
Evidence: S0.1 empirical check (2026-07-11, item-set diff train vs val/test).
**Proposal:** **wontfix** in the headline eval (paper comparability). Consequences routed instead: (a) S0.3 defines cold-item sets *deliberately* rather than relying on this accident; (b) thesis discloses the amazon 7.1% slice when interpreting replication numbers; (c) hm_1_month count measured once data is reachable (S0.5/S0.7).
**Disposition:** _pending gate._

### F-S0-05 [minor] [shared] [open]
`data/hm_1_month/` — the V1 primary scrutiny dataset — does not exist on the laptop (splits gitignored, live on LEO5/GPU machine); LEO5 unreachable today (no VPN). Blocks local S0.5 (signature-collision recompute needs V1 `item_features.csv`) and the F-S0-03 invariant check for V1.
**Proposal:** next VPN session, `scp` the 6 split files + `item_features.csv` from LEO5 to `data/hm_1_month/`. Prefer copy over local rebuild: `hm_splitter.py` uses non-stable `sort_values` (quicksort), so a rebuild is not guaranteed byte-identical to the canonical artifact.
**Disposition:** _pending gate._

## dc01 (F-DC01-…)

_(none yet)_

## dc02 (F-DC02-…)

_(none yet)_

## dc03 (F-DC03-…)

_(none yet)_

## dc04 (F-DC04-…)

_(none yet)_
