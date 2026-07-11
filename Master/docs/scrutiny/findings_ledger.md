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

### F-S0-02 [minor] [shared] [wontfix]
Eval/train negative sampling uses the global numpy RNG inside DataLoader workers, which fork without numpy reseeding: with `num_workers>0` both workers run identical RNG streams and the parent state never advances, so val negatives are bit-identical across epochs; with `num_workers=0` (Windows) negatives are re-drawn every epoch — two different eval regimes by platform.
Evidence: `rec_sys/protomf_dataset.py` `_neg_sample_uniform`/`_neg_sample_popular` (np.random, no `worker_init_fn`); `utilities/consts.py` `NUM_WORKERS` platform switch.
**Proposal:** document-only (this entry + S0.1 dossier §3) and **wontfix**: all scrutiny runs are LEO5-only (`num_workers=2`, consistent regime); cross-worker stream duplication is masked per-user by the exclusion vector and has no plausible metric impact; the epoch-fixed val negatives are actually beneficial for early-stopping stability. Alternative (not recommended): `worker_init_fn` with fixed per-worker numpy seed — would shift every sampled metric and buy nothing scientific.
**Disposition:** gate 2026-07-11 — accepted as proposed: document-only, wontfix.

### F-S0-03 [minor] [shared] [fixed(543e2d5)]
`Evaluator.get_results()` divides accumulated metric sums by `n_users`, silently assuming every user contributes exactly one eval row. Holds today (verified: ml-1m 6,034; amazon2014 6,950; hm_3_month 256,701 val=test=n_users; hm_1_month pending — F-S0-05), but any future split/eval change (cold-start eval!) breaks it silently.
Evidence: `utilities/eval.py:87`; `trainer.val()` / `tester.test()` construct `Evaluator(dataset.n_users)`. *(Edit 2026-07-11: hm_1_month verified after F-S0-05 resolution — val = test = 73,418 = n_users, one row per user. Invariant now confirmed on all four in-scope datasets.)*
**Proposal:** S0-build adds a cheap assert (total evaluated rows == n_users) in `Trainer.val()` and `Tester.test()` (no behavior change while the invariant holds; test in `dc_checks/s0/`). Constraint recorded for S0.3: the cold-start evaluator must define its own divisor, never inherit this one blindly.
**Disposition:** gate 2026-07-11 — accepted: assert lands in S0-build (stays open until resolving commit); S0.3 divisor constraint recorded. **Resolved S0-build (543e2d5):** implemented inside `Evaluator.get_results` (divide by counted rows + assert vs the declared expectation) rather than in `Trainer.val()`/`Tester.test()` — the assert thereby covers both call paths plus any future caller; expectation arg optional (None = caller owns its divisor, the S0.3 constraint honored). Bit-identity while rows == n_users pinned by `dc_checks/s0/t01` (11 checks green). Built under the user's away-authorization; component gate pending batch review.

### F-S0-04 [minor] [shared] [wontfix]
Standard test sets already contain train-unseen items scored on random-init (never-trained) ID embeddings: amazon2014 101 items → 490/6,950 test rows (**7.1%**, +140 val rows); hm_3_month 23/256,701 rows; ml-1m 0. Paper-faithful accident of k-core + temporal LOO (k-core counts include val/test occurrences).
Evidence: S0.1 empirical check (2026-07-11, item-set diff train vs val/test). *(Edit 2026-07-11: hm_1_month measured after F-S0-05 resolution — 3 train-unseen items, 9/73,418 test rows ≈ 0.012%. Negligible on the primary dataset; amazon remains the only materially affected set.)*
**Proposal:** **wontfix** in the headline eval (paper comparability). Consequences routed instead: (a) S0.3 defines cold-item sets *deliberately* rather than relying on this accident; (b) thesis discloses the amazon 7.1% slice when interpreting replication numbers.
**Disposition:** gate 2026-07-11 — accepted as proposed: wontfix, routed to S0.3 + thesis disclosure.

### F-S0-05 [minor] [shared] [fixed]
`data/hm_1_month/` — the V1 primary scrutiny dataset — does not exist on the laptop (splits gitignored, live on LEO5/GPU machine); LEO5 unreachable today (no VPN). Blocks local S0.5 (signature-collision recompute needs V1 `item_features.csv`) and the F-S0-03 invariant check for V1.
**Proposal:** next VPN session, `scp` the 6 split files + `item_features.csv` from LEO5 to `data/hm_1_month/`. Prefer copy over local rebuild: `hm_splitter.py` uses non-stable `sort_values` (quicksort), so a rebuild is not guaranteed byte-identical to the canonical artifact.
**Disposition:** resolved 2026-07-11 same day — VPN restored (Cisco Secure Client replaced by OpenConnect NM profile after reconnect-loop instability), all 6 files scp'd from `leo5:/scratch/c7031336/protomf_data/hm_1_month/` (byte sizes identical to source; counts match the canonical 623,230/73,418/13,651). No resolving commit — the artifacts are gitignored data files. Sanity re-checks recorded under F-S0-03/F-S0-04.

### F-S0-06 [minor] [shared] [fixed(a46d4e8)]
NaN-policy asymmetry in `feature_extraction/feature_ids.py`: `build_attr_multi_hot` raises on NaN cells, but `build_feature_ids` calls `astype(str)` first and would silently encode NaN as a legitimate `'nan'` vocab value (a phantom feature). Currently inert — V1 `item_features.csv` has 0 NaN cells across all 9 columns (measured 2026-07-11, S0.5), consistent with the 4.0 no-missingness finding.
Evidence: `feature_ids.py:67` (`vals = df[field].astype(str)`, no isna guard) vs `feature_ids.py:126-129` (explicit raise).
**Proposal:** add the symmetric NaN guard to `build_feature_ids` in S0-build — no behavior change on current data; covered by a new `dc_checks/s0/` test feeding a NaN row and expecting the raise.
**Disposition:** gate 2026-07-11 (S0.5) — accepted; lands in S0-build (stays open until resolving commit). **Resolved S0-build (a46d4e8):** symmetric guard added; `dc_checks/s0/t02` pins the raise, builder symmetry (incl. the pandas literal-`'nan'`-parses-as-NaN case), and V1 cleanliness (0 NaN, V=426 under the canonical 5). dc01 t01/t06 regressions green. Built under away-authorization; component gate pending batch review.

### F-S0-07 [major] [dc01] [open]
The S0.3 §5 convention "dc01: ID column dropped at cold inference" has no implementation anywhere: `cold_eval.build_model` rebuilds a `feature_item_proto` config natively, so a dc01 run with `use_id_feature=1` would score cold items *including their untrained per-item ID embedding row* — the same silent-depressant class as the bias gap, one level down. Dormant today (no dc01 cold run exists; no dc01 row in the S0.7 reference fleet).
Evidence: S0.7 black-box audit (`s0_7_audit_cold_machinery.md` §5, divergence #1); `grep cold feature_extraction/*.py` empty.
**Proposal:** two-part. (a) Now (S0.7): `cold_eval.load_config` refuses native cold eval of `ft_type='feature_item_proto'` configs with `use_id_feature` enabled — loud error naming the missing drop, symmetric to the `use_bias` guard; test in `dc_checks/s0/`. (b) The actual drop (zero/skip the cold items' ID rows at cold scoring, per dc01 design doc §3.4/§3.7.6) is implemented and tested at dc01's SC.3, where that design doc is in scope.
**Disposition:** S0.7 gate 2026-07-11 — approved as proposed. Guard landed (t06 pins refusal incl. the absent-key=default-True case; `_noid` passes). The generic drop primitive already covers the direct FeatureEmbedding case (see F-S0-13); the dc01 nested case stays REFUSED until dc01 SC.3 verifies it. Status stays **open** until then (part (b) outstanding). Guard commit: 30a570c.

### F-S0-08 [minor] [shared] [fixed(30a570c)]
The popularity reference row is degenerate on cold-vs-cold: all 100 candidates have popularity 0 → sigmoid maps all to 0.5 → `bn.argpartition` resolves the 100-way tie by introselect internals, which deterministically exclude index 0 — the row reads HR@10 = 0.00 instead of the advertised 0.10 chance floor (verified empirically, both first-party and by the black-box audit). Relatedly, raw-count scores saturate `torch.sigmoid` at 1.0 for pop ≥ ~17, collapsing warm order on cold-vs-all.
Evidence: `utilities/cold_eval.py:187-201` (`PopularityScorer`), `utilities/eval.py:18`; empirical all-tie probe 2026-07-11.
**Proposal:** `PopularityScorer` returns the dense popularity *rank* scaled to [0,1] plus small seeded uniform tie-break noise (deterministic under the run seed; equal-pop ties break randomly; no sigmoid saturation). Report the analytic chance line for all K on cold-vs-cold. New `dc_checks/s0/` check: all-tie input reads ≈ K/100 under the new scorer.
**Disposition:** S0.7 gate 2026-07-11 — approved; implemented as proposed (noise amplitude = half a rank gap, provably never reorders distinct popularity levels; t06 pins range, order-preservation, chance behavior on all-tie pools, seed determinism). Chance line now reported for all K. Status: **fixed** (30a570c).

### F-S0-09 [minor] [shared] [fixed(30a570c)]
Variant warm val/test negative sampling excludes only *variant* consumption, so a user's canonically-consumed cold items (their actual removed purchases) can be drawn as warm-eval negatives (~1.2% of rows) — spec-silent, uniform across models, but directionally biased: it penalizes exactly the models that score removed purchases high (cold-generalizing ones), including in early stopping.
Evidence: S0.7 audit B3; `ProtoRecDataset.load_data` CSR construction (variant files only).
**Proposal:** on marked variant dirs (`cold_items.csv` present), `ProtoRecDataset` val/test additionally folds the user's `cold_test.csv` rows into the exclusion CSR; canonical behavior untouched; t05 extended. Alternative (not recommended): document-accept.
**Disposition:** S0.7 gate 2026-07-11 — approved; implemented (exclusion-CSR fold-in, positives untouched; t06 sweeps full variant val+test — 0 violations — and pins canonical no-op). Modifications log cross-ref added. Status: **fixed** (30a570c).

### F-S0-10 [minor] [shared] [wontfix]
The S0.3 §6 "dev-profile epochs" retrain convention is operator-memory only: `run_combo.py` defaults to `--profile production`, whose `n_epochs=100` always overrides the saved config's epochs (the `RETRAIN_CONFIG_KEYS` inclusion of `n_epochs` is dead in the CLI path). Forgetting `--profile dev` silently deviates from charter C4.
Evidence: S0.7 audit divergence #5; `run_combo.py:470-474`.
**Proposal:** when `--retrain-config` is given and `--profile` is not explicitly passed, default to `dev` with a loud print. Profile already lands in the C8 manifest.
**Disposition:** S0.7 gate 2026-07-11 — **wontfix, amended per user reasoning:** profile selection is a queue-time decision for EVERY run type (all scrutiny hyperopts equally require `--profile dev` at submission; nothing is hardwired per-model anywhere), so a retrain-only default would be the fleet's sole special case. Existing safeguards suffice: resolved profile loudly printed, non-production profiles auto-tagged in W&B, C8 manifests record it, SC.8 audits it. The S0.7 run plan carries explicit `--profile dev` on every submission line. Status: **wontfix**.

### F-S0-11 [minor] [shared] [fixed(30a570c)]
S0.3 §3 mandates the variant-warm-vs-canonical-warm sanity comparison ("removal must not distort the warm regime"), but the runner only prints the variant warm numbers — the comparison is left to the operator.
Evidence: S0.7 audit divergence #4; `utilities/cold_eval.py:246-250`.
**Proposal:** optional `--canonical-results-dir` argument; when given, `report.json`/`report.md` include the canonical run's test metrics and the warm delta row.
**Disposition:** S0.7 gate 2026-07-11 — approved; implemented as proposed. Status: **fixed** (30a570c).

### F-S0-12 [minor] [shared] [wontfix]
fp32 `torch.sigmoid` saturates at 1.0 for logits ≥ ~16.6, collapsing rank order among saturated items in every eval pass (Tester and cold runner alike) — pre-existing host behavior, uniform across all rows and all historical numbers.
Evidence: S0.7 audit B2, verified empirically.
**Proposal:** document-only **wontfix**: sigmoid is rank-monotone below saturation; removing it (or ranking raw logits) would shift every historical/replication number for zero scientific gain. The one materially distorted row (popularity reference) is fixed structurally by F-S0-08's rank-based scorer.
**Disposition:** S0.7 gate 2026-07-11 — approved: **wontfix**, documented here and in the learnings ledger.

### F-S0-13 [minor] [shared] [fixed(30a570c)]
The S0.4 §2 spec commits `lightfm_tags_ids` to "cold inference via dc01's ID-column-drop convention," but no drop existed anywhere (same root cause as F-S0-07) — and unlike dc01, `lightfm_tags_ids` IS in the S0.7 reference fleet, so this could not wait for dc01's SC.3. Discovered first-party at gate execution; both black-box audits missed it because it sat in the seam between their scopes (audit A had only the S0.3 spec, audit B did not receive `cold_eval.py`).
Evidence: S0.4 §2 spec text; `grep`-verified absence pre-fix.
**Proposal (applied):** generic `drop_cold_id_rows` primitive in `cold_eval.py` — zeros cold items' per-item ID rows in a `FeatureEmbedding` item branch (q_cold = Σ_f e_f, warm rows untouched), applied automatically by the runner after checkpoint load, re-applied after the kNN weight restore, recorded in the report. Verified for the direct (lightfm) branch by t06 (repr == pure feature composition after drop; warm untouched; no-op on tags-only/CF branches). The dc01 nested case remains guarded by F-S0-07 until SC.3 verifies it.
**Disposition:** S0.7 gate execution 2026-07-11 — implemented within the ratified F-S0-07/fleet scope; flagged to the user in the gate follow-up. Status: **fixed** (30a570c). Audit-scope-seam watch-out added to the learnings ledger.

## dc01 (F-DC01-…)

_(none yet)_

## dc02 (F-DC02-…)

_(none yet)_

## dc03 (F-DC03-…)

_(none yet)_

## dc04 (F-DC04-…)

_(none yet)_
