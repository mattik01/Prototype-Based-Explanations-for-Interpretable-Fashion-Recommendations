# S0.7 black-box audit — cold-start machinery vs S0.3 spec (verbatim)

> Independent spec-vs-code subagent, 2026-07-11, HEAD `93b1edd`. Inputs per
> Ground rule 10: ONLY the S0.3 spec section (standalone extract, incl. rev. 1
> banner + §2 edit note) and the implementation files. No session context, no
> S0-build narrative (deliberate — so ratified build deviations get re-derived
> or re-flagged independently). The scrutinizing session's point-by-point
> response lives in the dossier's S0.7 section.

---

# Spec-vs-code audit — S0.3 cold-start eval (rev. 1, gate-closed spec)

Audited implementation at `/home/mattik01/Desktop/githubs/ProtoMF` against `spec_s03_cold_eval.md`. Per the spec's §2 edit note, the committed generator's draw is treated as canonical; rev. 1 row-level numbers are treated as superseded illustrations.

---

## §1 Design overview

- **Stratum TC (20% held-out, single cheap retrain)** — CONFORMS. Generator defaults `DEFAULT_FRACTION = 0.2`, `DEFAULT_SEED = 38210573` (`data/hm/make_cold_variant.py:40-41`); retrain path exists (`Master/scripts/run_combo.py:461-464`, `load_retrain_config` at :317).
- **Stratum TW + D4 readout (zero training cost)** — CONFORMS. `Master/scripts/stratified_readout.py` runs one test pass on an existing checkpoint; buckets 1–5 / 6–20 / 21–100 / >100 (`ITEM_BUCKETS`, :34) and user-history quartiles (:72-83).

## §2 Cold set + variant construction

- **"never re-split from raw, never re-k-cored"** — CONFORMS. `make_cold_variant.py` reads only the canonical split CSVs (:68-71); no k-core logic anywhere in the file.
- **Deciles over items with ≥1 train interaction; 20%/decile, `default_rng(38210573)`** — CONFORMS (`sample_cold_items`, :44-63; equal-frequency `np.array_split` deciles over (popularity, item_id)-sorted eligibles, per-decile draw without replacement).
- **Committed-draw numbers** — CONFORMS exactly. The checked-in `data/hm_1_month_cold/cold_variant_report.json` matches the edit note on every figure: 2,730 cold / 13,648 eligible, removed train 92,817 (19.48%), cold rows 121,024, 143 users / 515 rows zero-train collateral, variant val 59,066, warm test 59,048, 11,714 users below 3 train rows.
- **Zero-train-user drop + below-3 counting** — CONFORMS (:87-96; dropped users reported, <3-row users counted but not dropped).
- **ID universe byte-identical** — CONFORMS (`shutil.copy2` of `user_ids.csv`/`item_ids.csv`/`item_features.csv`, :105-106).
- **Warm test := canonical test minus cold-positive rows** — CONFORMS (:79-84; test rows with item∈C are routed to `cold_test.csv` tagged `test`).
- Unspecified extra: `cold_variant_report.json` written to the variant dir (spec only requires printing + `cold_items.csv`). Harmless.

## §3 Eval semantics

- **Primary cold-vs-cold / secondary cold-vs-all** — CONFORMS. `ColdTestDataset` pools: `self.cold_items` vs `np.arange(n_items)` (`rec_sys/protomf_dataset.py:262`); runner evaluates both, labels cold-vs-cold PRIMARY in report.md (`utilities/cold_eval.py:253-262`, `_write_markdown`).
- **Negative exclusion = canonical train+val+test consumption** — CONFORMS for the cold rankings (:264-270 builds the canonical-consumption CSR; :288-293 masks it out; the positive is prepended, never drawn).
- **Metrics HR/NDCG@{1,3,5,10,50} only** — CONFORMS (`K_VALUES` in `utilities/consts.py:27`; nothing else computed). Chance level recorded only for HR@10 (`chance_hit_ratio@10`, `cold_eval.py:258`) — cosmetic vs "HR@K = K/100".
- **Divisor = counted rows + assert vs declared expectation** — CONFORMS. `Evaluator` divides by `self.n_entries` and asserts against the declared count (`utilities/eval.py:93-98`); `evaluate_rows` declares `len(dataset)` (`cold_eval.py:97`); Trainer/Tester declare `coo_matrix.nnz` (`trainer.py:232`, `tester.py:82,119`) while the one-row-per-user invariant itself is enforced at dataset load (`protomf_dataset.py:132-139`, relaxed only when `cold_items.csv` is present). Combined semantics match; canonical numbers bit-identical since nnz == n_users there.
- **Reporting: "compare against canonical warm numbers"** — DIVERGES (minor). The runner reports the variant warm test (`cold_eval.py:246-250`) but never loads or juxtaposes the canonical run's warm numbers; the mandated sanity comparison is left to the operator. No effect on scored results.
- Beyond-spec observation (warm-test negatives): the variant warm test uses standard `ProtoRecDataset` semantics — exclusion by *variant* train+test only — so a user's canonically-consumed **cold** items (removed from variant train) can legally be drawn as warm-test negatives (and as warm-val negatives during retraining). The spec assigns the canonical-consumption exclusion only to the cold rankings, so this is spec-silent, uniform across models, and small (~1–2% of draws), but it puts the user's actual removed purchases into the negative slots of exactly the warm-regime sanity comparison §3 mandates, and marginally penalizes cold-generalizing models in early stopping. Minor; not training leakage.

## §4 Leakage checklist (item-by-item against code paths)

1. **Attributes available for cold items** — ✅ CONFORMS. `item_features.csv` copied unchanged; verified header contains only static catalog fields (`product_group_name … department_name`), no price, no dates.
2. **No transaction-derived features** — ✅ CONFORMS. `CANONICAL_FIELDS` (`cold_eval.py:52-53`) are all catalog attributes; `price_band` is wired nowhere in the cold path.
3. **No re-coring** — ✅ CONFORMS (see §2). Note the spec's own item-3 numbers ("2,911 users… 2 users dropped") are stale rev-1 text superseded by the edit note — spec-internal inconsistency, not a code issue.
4. **No temporal fields** — ✅ CONFORMS (same header check).
5. **Eval negatives exclude canonical consumption** — ✅ CONFORMS for cold rankings (code path above). The positive cannot appear among its own negatives (prepended separately; also in the consumption mask).
6. **Training negatives exclude C** — ✅ CONFORMS. `p[C]=0` in both `_neg_sample_uniform` (:153-154) and `_neg_sample_popular` (:172-174), gated on `split_set == 'train'` and `cold_items.csv` presence (:64). Adversarial probe: the gate is file-presence-based, but a missing `cold_items.csv` on (say) the cluster cannot silently disable the mask — the val split would then trip the one-row-per-user `AssertionError` (:137-139) before training starts, because variant val has 59,066 rows for 73,418 users. Fails loudly. Eval negatives keep the full catalog per spec (t05 confirms val has no mask). No leakage found.
7. **Model selection blind to cold** — ✅ CONFORMS. Variant val contains no cold positives (generator removes them); early stopping runs on the variant val loader with `optimizing_metric` defaulting to `hit_ratio@10` (`run_combo.py:497`, `trainer.py:42`).

**No leakage divergence found on any of the seven items.**

## §5 Per-model cold conventions

- **CF noise floor (i)** — CONFORMS in spirit: the retrained model is evaluated natively. Note the cold ID embeddings are not literally "random-init" after retraining — weight decay (Adam/Adagrad `wd`) shrinks them every step since they receive no data gradient — but that *is* "as the architecture actually behaves". No action.
- **Attr-kNN fallback (ii)** — DIVERGES (minor). Spec: "cold item's **representation** := mean of its top-n attribute-cosine warm neighbors' *trained* item **representations**". Code patches the **base ID-embedding table** instead (`attr_knn_patch`, `cold_eval.py:140-142` via `_find_item_embedding_weight`), then recomputes the representation through the model. Identical for `mf`; **not** identical for `acf`/`item_proto`/`user_item_proto`, where the representation is a nonlinear map (anchor softmax / prototype cosine) of the base embedding — mean-of-embeddings ≠ mean-of-representations. Changes the attr-kNN competitor's numbers only, not model rows. n=20 default ✅; double-tie handled correctly (factory ties `item_embed.embedding_layer.weight` to `item_proto.embedding_ext…weight`, `feature_extractor_factories.py:95`, so one patch covers both halves as claimed); feature-native branches (`FeatureEmbedding`, `AttributeLookup`) correctly raise → `attr_knn: skipped` (:117-120, :275-276). Weights restored via checkpoint reload before the tie diagnostic (:274). Cosine==dot shortcut valid: `build_attr_multi_hot` enforces exactly-one-value-per-field (row norm √F, `feature_ids.py:151-152`).
- **Popularity reference** — CONFORMS structurally (`PopularityScorer`, cold pop = 0), but see beyond-spec observation B1: on cold-vs-cold it is degenerate.
- **dc01 `use_bias` guard** — CONFORMS. `use_bias` recorded in the report (:243) and `load_config` hard-refuses `use_bias > 0` (:59-64); defaults to 1 when the key is absent, which is consistent-conservative with `RecSys`'s own absent-key default of `True` (`rec_sys.py:36`).
- **dc01 "ID column dropped at cold inference"** — **MISSING (major)**. No code anywhere drops the per-item ID feature column at cold inference: `grep cold feature_extraction/*.py` is empty, and `cold_eval.build_model` rebuilds the config's model natively. A `feature_item_proto` run with `use_id_feature=1` would score cold items *including their untrained per-item ID embedding row* — exactly the silent-depressant class the spec's own bias-gap note warns about, one level down. Does not affect any other model row; invalidates dc01's native cold row when it is eventually run, unless the `_noid` config is used or the drop is implemented.
- **dc02 tie-block diagnostic** — CONFORMS. Model-agnostic, 5,000-user seeded sample, full-catalog top-10 signature-class stats (`tie_block_diagnostic`, :153-184); pure dot product matches `RecSys.forward` under the enforced `use_bias=0`.
- **dc03/dc04 image-availability assert** — no implementation, but those models don't exist yet; MISSING-by-schedule, not actionable now.

## §6 Training protocol on the variant

- **Single-config retrain, no search** — CONFORMS. `--retrain-config` loads a completed run's `config.json`, strips execution keys, forces `num_samples=1` (`run_combo.py:317-331`); `start_hyper` then sets `search_alg=None` for `num_samples==1` (`experiment_helper.py:186`).
- **Same seed 38210573** — CONFORMS (`SINGLE_SEED = 38210573`, `consts.py:4`, the `-s` default).
- **Early stopping on variant warm val, not test** — CONFORMS (Trainer stops on `hit_ratio@10` over the variant val loader; no test-based stopping anywhere).
- **Per-item bootstrap CIs** — CONFORMS (`per_item_bootstrap_ci`, cluster bootstrap over cold items, 1,000 resamples, applied to HR@10/NDCG@10 of both rankings, `cold_eval.py:260-261`).
- **"dev-profile epochs"** — DIVERGES (minor/operational). `run_combo` defaults to `--profile production` (:470-474), whose `n_epochs=100` **always overrides** the saved config's `n_epochs` (profile resolution never leaves it `None`, so `RETRAIN_CONFIG_KEYS`' inclusion of `n_epochs` is dead in the CLI path). The §6 convention requires the operator to remember `--profile dev`; nothing encodes it.

## §7 D4 stratified readout

- CONFORMS functionally: per-row HR@10/NDCG@10 by user-history quartile and item-popularity bucket, one table per run, json+md output.
- Cosmetic divergence: spec names `Tester.get_test_logits(aggregated=False)` as the mechanism; the script instead uses `cold_eval.evaluate_rows` over a fresh `ProtoRecDataset` (`stratified_readout.py:52-54`) — and `Tester.get_test_logits` doesn't even accept an `aggregated` kwarg (`tester.py:112`). Functionally equivalent per-row metrics; note the negatives are re-drawn (seeded) rather than reusing the original test run's draws, so overall numbers can differ from `test_metrics.json` in late decimals.
- Unspecified extra: a separate bucket `0 (train-unseen, F-S0-04)` (:88-92) — not in the spec's bucket list, sensible and clearly labeled.

## §8 Module touch-points

All seven exist: 1) generator ✅, 2a) neg mask ✅ / 2b) `ColdTestDataset` ✅, 3) Evaluator divisor+assert ✅, 4) `utilities/cold_eval.py` with all listed components incl. `use_bias` refusal ✅, 5) `Master/scripts/stratified_readout.py` ✅, 6) both `start.py:24-25` and `run_combo.py:66-67` register `hm_1_month_cold`, `--retrain-config` in run_combo ✅, 7) tests present (`t01` divisor, `t04` generator, `t05` dataset/mask semantics, `i02` e2e incl. attr-kNN identity and `use_bias` refusal).

## §9 Declared non-adoptions

CONFORMS — code adopts none of them: no test-based stopping, splits derived from the frozen artifact, no split repetitions.

---

## Beyond-spec observations (correctness, verified empirically)

- **B1 — Popularity reference is degenerate on cold-vs-cold.** All 100 candidates score popularity 0 → after `torch.sigmoid` all logits are exactly 0.5 → a 100-way tie. `bn.argpartition` breaks ties by partition internals, not randomly or pessimistically: empirically (protomf env) the top-10 of a constant 100-slot row is indices {63…74}, **excluding the positive → HR@10 = 0**, i.e. *below* the 0.10 chance floor the spec advertises as "directly readable". The published "floor reference" row for the primary ranking is an artifact of introselect ordering. Fix: report chance K/100 analytically for the popularity row on cold-vs-cold, or add random tie-breaking. (`cold_eval.py:187-201` + `eval.py:18`.)
- **B2 — fp32 sigmoid saturation ties.** `evaluate_rows` applies `torch.sigmoid` before scoring; any logit ≥ ~16.6 becomes exactly 1.0 in fp32 (verified), collapsing order among top items. Pre-existing host behavior (Tester does the same), uniform across rows; mainly distorts the popularity row on cold-vs-all (all warm items with pop ≥ 17 tie at 1.0).
- **B3 — Warm-val/test negatives can include the user's own removed cold purchases** (detailed under §3). Spec-silent, minor, uniform.
- **B4 — `ColdTestDataset` determinism is convention-dependent.** The shared `default_rng` stream reproduces "identical draws across model rows" only under sequential access with `num_workers=0`; the runner complies (`evaluate_rows`, :96), but nothing in the dataset guards against a future consumer using workers/shuffle and silently breaking cross-row comparability.

---

## Summary of divergences (ranked)

| # | Spec item | Code reality | Severity | Leakage? |
|---|---|---|---|---|
| 1 | §5 dc01: "ID column dropped at cold inference" | No implementation anywhere; runner evaluates dc01 natively with the untrained cold ID row included | **Major** (invalidates dc01's future native cold row; other models unaffected) | No — eval-side confound, not leakage |
| 2 | §5 popularity row = readable floor reference | Cold-vs-cold popularity row is a 100-way tie resolved by argpartition internals; reads ≈0.00 instead of chance 0.10 (B1, beyond-spec) | Major *for that reference row only* | No |
| 3 | §5 attr-kNN: mean of neighbors' **representations** | Patches mean of neighbors' **base ID embeddings**, then recomputes representation (differs for acf / proto models) | Minor (changes only the fallback competitor's numbers) | No |
| 4 | §3 reporting: "compare against canonical warm numbers" | Variant warm numbers reported; no comparison against canonical performed | Minor (manual step) | No |
| 5 | §6 "dev-profile epochs" for retrains | `run_combo` defaults to production profile, which silently overrides the saved config's epochs; dev profile is opt-in | Minor / operational | No |
| 6 | §7 mechanism "via `Tester.get_test_logits(aggregated=False)`" | Own `evaluate_rows` path; named method lacks that kwarg | Cosmetic | No |
| 7 | §3 chance level "HR@K = K/100" | Only `chance_hit_ratio@10` recorded | Cosmetic | No |
| — | Unspecified extras | `cold_variant_report.json`; readout bucket 0; generator `--fraction/--seed` overrides | Cosmetic, all benign | No |

**Leakage verdict:** all seven §4 checklist items are correctly realized in the actual code paths; the file-presence gate for the training-negative mask is protected by a loud val-split assertion, so it cannot fail silently. No divergence that lets cold/future information reach training or model selection was found. The two substantive findings are the missing dc01 ID-column drop (§5, currently dormant since no dc01 cold run exists yet) and the degenerate popularity reference on the primary ranking.
