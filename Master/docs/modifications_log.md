# Modifications Log

Tracks all original repo files modified from their upstream state.

## CLAUDE.md
- Added RULES section: temp file rule, protocol entry prompting rule, modifications log rule
- Added Branch Strategy, Naming Conventions, Reserved ft_type Values, Results Storage, Modification Map, H&M Data sections

## utilities/consts.py
- Changed `DATA_PATH` from hardcoded absolute path to relative `os.path.join` (machine-independent)
- Replaced hardcoded `WANDB_API_KEY` with file-based loading from `Master/sensitive/api_keys/wandb_key.txt`
- Added `import os`

## .gitignore
- Added (already existed with `__pycache__` rules; no new entries needed at root level)
- Added ignore rule for downloaded literature PDFs (`Master/literature/papers/**/*.pdf`, ~182 MB, re-fetchable)
- Added ignore rule for mirrored run checkpoints (`Master/experiments/results/**/*.pth`, ~326 MB; archived on LEO5 + local disk — run metadata stays in git, the pre-existing tracked canonical checkpoint is unaffected)

## confs/hyper_params.py
- Added `debug_hyper_params` config with fixed tiny values for fast CPU smoke tests (embedding_dim=8, n_prototypes=3, 3 epochs, num_samples=1)

## start.py
- Added `debug` model choice that uses `debug_hyper_params`
- Fixed typo: `--multipl2erfve` → `--multiple` (accidental corruption of argparse flag name)
- Added `hm_full` and `hm_3_month` to dataset choices
- Added `hm_1_month` to dataset choices (V1 dev/judgment testbed)

## experiment_helper.py
- `start_hyper()` now respects optional `num_samples` key in config dict (falls back to global `NUM_SAMPLES`)
- Removed legacy `_metric/` prefix from `metric` and `metric_name` — newer Ray Tune no longer auto-prefixes reported metrics, causing silent hangs on metric validation
- Enhanced `WandbLoggerCallback` with console capture (`console: auto`), and `group` for grouping trials by model/dataset/seed

## experiment_helper.py (continued)
- Moved W&B metadata (model, dataset, group, tags) from environment variables to config dict (`_wandb_*` keys) — fixes incorrect tagging when `run_combo.py` chains multiple combos in the same process

## rec_sys/trainer.py
- Added `use_ray` flag (default=True) to `Trainer.__init__()` to allow standalone PyTorch training without Ray Tune
- Extracted `_report()` method that conditionally uses `ray.train.report()` or prints metrics locally
- `run()` now saves checkpoints via plain `torch.save()` when `use_ray=False`
- Ray import moved from top-level to inside `_report()` (lazy) so the module loads without Ray when not needed
- Fixed `_report()` local-print path crashing on non-numeric metric values (`eta_min=None` while improving) — only format floats with `:.4f`
- Added timing metrics to `run()`: epoch duration, elapsed time, ETA, patience counter, LR, and epoch number — all reported to W&B per epoch

## README.md
- Replaced original paper README with thesis-specific README that credits the upstream repo, describes the thesis goals (two contributions: the feature-grounding lineage and quantified interpretability), and documents the extended repository structure

## Master/scripts/gpu_graph.py
- Removed CSV logging logic (csv import, LOG_DIR, csv_file, csv_writer, all CSV writes); now purely a visual terminal tool with no file I/O

## utilities/explanations_utils.py
- Removed deprecated `square_distances=True` parameter from TSNE call (removed in scikit-learn 1.6)
- `tsne_plot` now infers save format from the file extension and saves at dpi=200 with tight bbox (was hardcoded `format='pdf'`) — explanation figures are now emitted as PNG
- `tsne_plot` gained `prototype_labels` (annotate prototype markers with derived names) and a `point_styles` seam (per-point feature-encoded glyphs; default unchanged uniform dots)
- `weight_visualization` gained `u_proto_labels`/`i_proto_labels` (prototype-name captions), a plain-language symbol legend for s/t̂/u* etc., and a minimum panel-width floor so the figure with fewer prototypes no longer collapses to an unreadable sliver
- `tsne_plot` prototype-label annotation now repels overlapping names via `adjustText` (optional dependency; graceful fallback to fixed-offset annotation when absent) with thin leader lines back to markers

## utilities/consts.py (LEO5 migration)
- `DATA_PATH` now supports `PROTOMF_DATA_PATH` env var override (falls back to relative path)
- `WANDB_API_KEY` loading wrapped in try/except with `WANDB_API_KEY` env var fallback

## experiment_helper.py (LEO5 migration)
- Added explicit `ray.init()` with `RAY_TMPDIR` env var support before `tune.run()`
- Added `storage_path` to `tune.run()` via `RAY_RESULTS_DIR` env var (directs results to scratch on HPC)

## rec_sys/tester.py (LEO5 migration)
- Added `weights_only=True` to `torch.load()` for PyTorch 2.x compatibility

## utilities/consts.py (experiment results)
- Added `EXPERIMENT_RESULTS_PATH` with `PROTOMF_RESULTS_PATH` env var override (defaults to `Master/experiments/results/`)

## experiment_helper.py (experiment results)
- `start_hyper()` now returns a rich dict (test_metrics, val_metrics, best_config, checkpoint path, trial stats) instead of just test metrics
- `start_multiple_hyper()` updated to use `result['test_metrics']` from new return format

## Master/scripts/run_combo.py (experiment results)
- Added `_save_combo_results()`: creates structured results folder per combo with checkpoint, config, metrics JSONs, hardware log, and summary.md
- Added `_generate_summary()`: produces pre-formatted replication report table rows
- `run_single_combo()` now saves results automatically after each experiment

## confs/hyper_params.py (per-execution resource config)
- Removed `num_samples: 30` override from `proto_double_tie_chose_original_hyper_params` — all models now use the same default (100)

## experiment_helper.py (per-execution resource config)
- `start_hyper()` now accepts optional `resource_cfg` dict for per-execution overrides of gpu_per_trial, cpu_per_trial, num_samples, num_workers, grace_period, patience, optimizing_metric, and n_epochs
- `load_data()` now accepts `num_workers` parameter instead of reading global constant
- Execution-level overrides injected into config dict with `_` prefix keys for passthrough to trials

## rec_sys/trainer.py (per-execution resource config)
- `Trainer` now reads `_max_patience` and `_optimizing_metric` from config namespace (falls back to consts)

## Master/scripts/run_combo.py (per-execution resource config)
- Added CLI args: `--num-samples`, `--gpu-per-trial`, `--cpu-per-trial`, `--num-workers`, `--n-epochs`, `--grace-period`, `--patience`, `--wandb-mode`, `--optimizing-metric`
- Builds `resource_cfg` dict from CLI and passes through to `start_hyper()`

## experiment_helper.py (ASHA toggle + W&B tags)
- ASHA scheduler now conditional: `ASHAScheduler(grace_period) if asha else None`
- Extra W&B tags from `resource_cfg['wandb_tags']` merged into train and test wandb.init calls

## Master/scripts/run_combo.py (ASHA toggle + W&B tags)
- Added `--no-asha` flag to disable ASHA early stopping scheduler
- Added `--wandb-tag` repeatable flag for extra W&B tags

## slurm/run_combo.slurm (ASHA toggle + W&B tags + CPU override)
- Added `--no-asha`, `--wandb-tag`, `--cpus` flags with passthrough

## Master/docs/replication_report.md (Machine column)
- Added Machine column to GPU Benchmark table for time estimation across devices

## Master/scripts/gpu_sampler.py (process-scoped metrics)
- `query_system()` now measures only the current process tree (RSS + CPU% of self + children) instead of node-wide `psutil.virtual_memory()` / `psutil.cpu_percent()` — fixes inflated RAM/CPU numbers on shared HPC nodes

## rec_sys/protomf_dataset.py (PyTorch 2.x compat)
- `T_co` import: fallback to `typing.Any` when `torch.utils.data.dataset.T_co` is removed (PyTorch ≥ 2.5)

## Master/scripts/run_combo.py (explanations auto-invocation)
- Auto-invokes `utilities.explanations.pipeline.run_explanations_pipeline(results_dir)` after `_save_combo_results()` for explainable models (`item_proto`, `user_proto`, `user_item_proto`); wrapped in try/except so failures do not invalidate the training run
- Added `--skip-explanations` CLI flag (default off) to disable the auto-invocation

## utilities/consts.py (early-stopping min_delta)
- Added `MIN_DELTA_DEFAULTS` dict (per-metric defaults, 1e-4 for all `hit_ratio@K` / `ndcg@K`) and `MIN_DELTA_FALLBACK` (1e-4) for unknown metrics

## rec_sys/trainer.py (early-stopping min_delta)
- `Trainer` now reads `_min_delta` from config (default 0.0 → preserves original "any improvement" behavior when unset)
- Early-stopping comparison changed from `curr > best` to `curr > best + min_delta`; logs `min_delta` at init

## experiment_helper.py (early-stopping min_delta)
- `start_hyper()` resolves `min_delta` from `resource_cfg`; when omitted, falls back to `MIN_DELTA_DEFAULTS[optimizing_metric]` then `MIN_DELTA_FALLBACK`
- Injects `_min_delta` into config for trial passthrough; logs the resolved value

## utilities/utils.py (W&B hardening)
- Added `safe_wandb_init`, `safe_wandb_finish`, `safe_wandb_login` — observability must never abort a run; init/finish/login failures are caught and logged, training continues without W&B

## experiment_helper.py (concurrency/W&B hardening)
- All `wandb.init`/`finish`/`login` calls (train, test, aggregate phases) routed through `safe_wandb_*` so a W&B failure can never terminate a trial
- `start_hyper()` now raises a clear `RuntimeError` if 0 trials completed (all errored), instead of failing cryptically in `get_best_checkpoint` — makes total failure explicit

## rec_sys/trainer.py (W&B hardening)
- `_report()` W&B log guard broadened from `except ImportError` to `except Exception` — a mid-training online `wandb.log` network/comm error can no longer abort training

## Master/docs/replication_report.md (restructure)
- Restructured around LEO5 as the primary machine: scoped to amazon2014/ml-1m/hm_1_month, added a paper-free Run Results table, merged cluster runs into one hardware table, dropped the Smoke Test section, and parked out-of-scope rows (hm_3_month/hm_full/lfm2b) with their numbers in a bottom Appendix

---

# dc01 — `feature_item_proto` (feature-composed item factors under the double-tied prototype layer)

Adds a new feature-aware model variant per `Master/docs/design_candidates/dc01_feature_composed_factors.md`.
Correctness proofs: `Master/temp/dc_checks/dc01/` (unit t01–t08 + integration i01–i04, i06a; F=0 keystone i02).

## feature_extraction/feature_extractors.py (dc01 P1–P3)
- Added `FeatureEmbedding` (sum-of-feature-value embeddings, `q_i = Σ_f e_f`, optional ID feature; `feature_ids` is a non-persistent buffer)
- Added `FeatureEmbeddingW` (linear projection over a SHARED `FeatureEmbedding` instance — the dc01 tie)
- `PrototypeEmbedding.__init__` gained an optional `embedding_ext=None` kwarg; `None` reproduces the original behaviour bit-for-bit (golden test t03), a passed instance is used as-is

## feature_extraction/feature_extractor_factories.py (dc01 P4)
- Added `ft_type == 'feature_item_proto'` branch, cloned from `prototypes_double_tie`: user side byte-identical; item side replaces the free ID embedding with one shared `FeatureEmbedding` feeding both the item prototype branch and the item projection (tie = shared instance). Factory is the single owner that inits the shared table once

## rec_sys/trainer.py (dc01 P6)
- `_build_model` calls `inject_feature_ids(self.ft_ext_param, dataset.data_path)` before the factory (no-op for every other `ft_type`); builds the `feature_ids` tensor at build time, never serialized into the config

## rec_sys/tester.py (dc01 P6)
- Symmetric `inject_feature_ids` call in `_build_model` (feature_ids reconstructed at test time, since the buffer is non-persistent)

## confs/hyper_params.py (dc01 P5)
- Added `feature_item_proto_hyper_params` — mirrors `proto_double_tie_chose_original_hyper_params` (directly comparable to the user_item_proto baseline) plus item-side `use_id_feature=True` and a `feature_fields` spec (5 categorical H&M fields)

## start.py (dc01 P5)
- Added `feature_item_proto` to the `--model` argparse choices and the config-selection `elif` chain

## Master/scripts/run_combo.py (dc01 P5)
- Registered `feature_item_proto` in `MODEL_CONFIGS`; deliberately NOT added to `EXPLAINABLE_MODELS` yet (run with `--skip-explanations`; pipeline integration is deferred P9)

## New files (not upstream — added for dc01)
- `feature_extraction/feature_ids.py` — `build_feature_ids` (field-offset vocab + alignment guard) and `inject_feature_ids`
- `utilities/explanations/feature_readout.py` — intrinsic per-feature read-out (`per_feature_shares`, `global_prototype_profile`)
- `data/hm/price_band.py` — price-band decile helper (P7; full H&M regeneration gated)

## confs/hyper_params.py, start.py, Master/scripts/run_combo.py (dc01 ablations §3.6)
- Added two ablation configs (deepcopies of `feature_item_proto_hyper_params`, one knob each):
  `feature_item_proto_noid` (`use_id_feature=False` — feature-composition only, capped by the no-ID
  equivalence-class ceiling: 6,517 distinct 5-field tuples among 13,651 hm_1_month items) and
  `feature_item_proto_f0` (`feature_fields=[]` — reduces to user_item_proto, the exact isolating
  control). Registered in `start.py` choices/elif and `run_combo.py` `MODEL_CONFIGS`; `import copy`
  added to hyper_params.py.

## utilities/explanations/* (dc01 P9 — intrinsic explanations; previously deferred)
- `feature_item_proto` is now wired into the explanations pipeline; the existing visualizations
  (tsne, feature small-multiples, naming, top-k, weight-viz) run for it with item prototypes
  grounded INTRINSICALLY via `cos(e_f, p_k)` instead of post-hoc nearest items.
- New `accessor/feature_item_proto.py` (composed `item_embeddings()` = q_i; `feature_value_embeddings()`;
  `has_intrinsic_item_grounding`). New `naming.name_prototypes_intrinsic` + `intrinsic_naming_config`
  (scoring='cosine'); populate the same `ProtoFeatureProfile` so namer/plots are unchanged.
  `pipeline.py` adds `feature_item_proto` to its `EXPLAINABLE_MODELS` and routes item naming through
  the intrinsic source (outputs nest under `explanations/cosine/`); user side stays post-hoc.
  `loader.py` calls `inject_feature_ids`. Five explainers' `supports()` extended.
- NOTE: `run_combo.py`'s `EXPLAINABLE_MODELS` still excludes `feature_item_proto`, so training runs
  do NOT auto-generate explanations (kept fast); invoke the pipeline standalone on the dev checkpoint
  (`python -m utilities.explanations.pipeline --results-dir <dir>`). Flip that set to auto-enable.

---

# dc02 — `attr_item_proto` (attribute-space item prototypes, concept-bottleneck-anchored)

Adds a new feature-aware model variant per `Master/docs/design_candidates/dc02_attribute_space_prototypes.md`.
Correctness proofs: `Master/temp/dc_checks/dc02/` (unit t01–t08, t10 + integration i01–i06;
spec-equivalence keystone i02 pins the repo classes to the frozen `check_claims.py` toy mechanism
the 11 design-doc claims were verified on). Zero edits to existing class bodies in
`feature_extractors.py` — the dc01 suite rerun (incl. the t03 golden, bit-identical) is the
regression gate.

## feature_extraction/feature_extractors.py (dc02 P1, P3–P4, P6 — additions only)
- Added `AttributeLookup` (parameter-free multi-hot base: the item IS its observed x_i ∈ {0,1}^V;
  non-persistent buffer, float32)
- Added `AttributePrototypeEmbedding(PrototypeEmbedding)` (prototypes A ∈ R^{K×V} in attribute
  space via `embedding_dim=V`; forward mirrors the host with `effective_prototypes()`; inherits
  the host cosine/reg machinery — bitwise host-equivalent in base mode, t02). Carries the four
  §3.5 constraint knobs, config-gated and DEFAULT OFF: K1 softplus nonnegativity (state_dict key
  stays `prototypes`, stores Ã), K2 field-crispness entropy, K3 separation hinge (plain cos),
  K4 data pull over a per-step resampled subset (dedicated torch.Generator — global RNG untouched);
  K2–K4 hook `get_and_reset_loss()`
- Added `AttributeProjection` (bias-free Linear(V, L_u) over the SHARED `AttributeLookup` — the tie)

## feature_extraction/feature_ids.py (dc02 P2)
- Added `build_attr_multi_hot` (multi-hot (n_items, V) + `field_offsets` + column (field, value)
  names; same lexicographic field-offset vocab convention as `build_feature_ids`; guards:
  item_id coverage, NaN cells, exactly one value per field)
- `inject_feature_ids` gained an `attr_item_proto` branch (same non-mutating shallow-copy
  discipline; injects `attr_multi_hot`/`n_attr_values`/`field_offsets`; config keeps only the
  lightweight `attr_fields` spec). No trainer/tester/loader edits needed — they already call it

## feature_extraction/feature_extractor_factories.py (dc02 P5)
- Added `ft_type == 'attr_item_proto'` branch, cloned from `prototypes_double_tie`: user side
  byte-identical (weight-tied CF); item side has NO per-item parameters — one shared
  `AttributeLookup` feeds `AttributePrototypeEmbedding` + `AttributeProjection`
  (`ConcatenateFeatureExtractors(..., invert=True)`). Requires `use_bias=0` (bottleneck side
  channel, §3.5(e)) — the `base_param` default
- Alignment guard `attr_multi_hot.shape[0] == n_items` (mirrors dc01's `FeatureEmbedding` assert):
  a stale/mismatched `item_features.csv` (split regenerated, item_ids.csv changed) would otherwise
  build silently and train/evaluate/explain every item with a DIFFERENT article's attributes

## confs/hyper_params.py (dc02 P7)
- Added `attr_item_proto_hyper_params` (user side mirrors `proto_double_tie_chose_original_...`;
  item side: `attr_fields` = all 9 H&M categorical columns → V=534 on hm_3_month, no item
  embedding_dim/max_norm, 7 knob keys default-off), `attr_item_proto_debug_hyper_params`
  (fixed tiny single-trial CPU smoke) and `attr_item_proto_debug_knobs_hyper_params`
  (deepcopy, all four knobs on at small weights)

## start.py (dc02 P7)
- Added `attr_item_proto`, `attr_item_proto_debug`, `attr_item_proto_debug_knobs` to the
  `--model` choices and the config-selection `elif` chain (+ imports)

## Master/scripts/run_combo.py (dc02 P7)
- Registered the three configs in `MODEL_CONFIGS`; `EXPLAINABLE_MODELS` deliberately untouched
  (auto-explanations stay opt-in, mirroring dc01 — run with `--skip-explanations` and invoke the
  pipeline standalone)

## utilities/explanations/* (dc02 P9 — intrinsic read-out)
- New `attr_readout.py`: exact per-attribute shares A_kv/(√F‖a_k‖) (sum bitwise to t*−1),
  item-discriminating contributions û·(t*−1) + disclosed user-constant Σû (§3.4 amendment),
  exact score decomposition, `prototype_attr_profile` (top values per field off A / W_t),
  `per_field_mass` (single-field-collapse diagnostic, §3.5(c))
- New `accessor/attr_item_proto.py` (`item_prototypes()` = EFFECTIVE A; `item_embeddings()` = X;
  `attr_share_matrix()`; `user_prototype_attr_profiles()` = W_t rows;
  `has_attr_space_prototypes` routing flag); registered in `accessor/__init__.py`
- `naming/namer.py`: extracted the shared back half of `name_prototypes_intrinsic` into
  `name_prototypes_from_score_matrix` (dc01's function now computes cos and delegates —
  behavior preserved, dc01 t10 reruns green); `naming/config.py`: added `attr_naming_config`
  (scoring='attr', share-unit min_score=0.02 — dc01's 0.30 cosine floor is wrong units here);
  `naming/__init__.py` exports both
- `pipeline.py`: `attr_item_proto` added to `EXPLAINABLE_MODELS`; attr route (reads `attr_fields`,
  names item prototypes from the exact-share matrix, artifacts nest under `explanations/attr/`)
  checked BEFORE the dc01 intrinsic route; user side stays post-hoc
- FOUR explainers' `supports()` extended (tsne, top_k_items, naming, feature_small_multiples).
  `weight_viz` deliberately NOT extended: its shared `weight_visualization` util renders per-item
  bars û_k·t*_k, folding the ranking-irrelevant user-constant Σû (shifted-cosine +1) into per-item
  numbers — the misleading attribution the design doc §3.4 amendment forbids; the amendment-correct
  rendering û_k·(t*_k−1) with Σû disclosed separately is produced by `readout_demo.py` /
  `readout_artifact.py` instead

## New files (not upstream — added for dc02)
- `utilities/explanations/attr_readout.py`, `utilities/explanations/accessor/attr_item_proto.py`
- `Master/temp/dc02_attr_item_proto_implementation_plan.md` (condensed plan)
- `Master/temp/dc_checks/dc02/{_harness.py, t01–t08, t10, i01–i06, readout_demo.py,
  readout_artifact.py, smoke_train.py}` (claims_spec.md / check_claims.py were pre-existing,
  frozen)
- NOTE (environment, discovered during P11): `experiment_helper.py` targets the Ray 2.x API
  (`ray.tune.search`) per the LEO5 migration, while the local laptop env has Ray 1.6.0 — so
  start.py/run_combo.py cannot run locally for ANY model (pre-existing mismatch, not dc02).
  `smoke_train.py` is the ray-free local harness (Trainer(use_ray=False) + Tester), assembling
  the run_combo-style results dir the pipeline/demo consume

# S0-build — foundation fixes (Scrutiny Protocol S0)

## utilities/eval.py (S0-build A)
- `Evaluator`: aggregated metrics now divide by the COUNTED number of evaluated rows instead of
  blindly by `n_users`; the count is asserted against the declared expectation (constructor arg
  `n_users`, now optional — None skips the check). Bit-identical while rows == n_users (holds on
  all four in-scope datasets). Why: findings ledger F-S0-03.

## feature_extraction/feature_ids.py (S0-build B)
- `build_feature_ids`: NaN guard added (raise on NaN/missing cells) symmetric with
  `build_attr_multi_hot` — previously `astype(str)` would silently encode NaN as a legitimate
  `'nan'` vocab value. No behavior change on current data (V1 has 0 NaN cells). Why: findings
  ledger F-S0-06.

## feature_extraction/feature_extractor_factories.py (S0-build C)
- New `ft_type == 'lightfm'` branch: user = plain `Embedding`, item = `FeatureEmbedding`
  (pure reuse of dc01-tested classes; no new module). Why: S0.4 CBF baseline spec.

## feature_extraction/feature_ids.py (S0-build C)
- `inject_feature_ids`: `'lightfm'` added to the feature_ids injection guard (same
  non-mutating copy discipline as `feature_item_proto`).

## confs/hyper_params.py (S0-build C)
- `lightfm_tags_ids_hyper_params` / `lightfm_tags_hyper_params`: mirror `mf_hyper_params`'
  search space exactly + canonical 5-field set + `use_id_feature` fork.

## start.py (S0-build C)
- Registered model names `lightfm_tags`, `lightfm_tags_ids`.

## Master/scripts/run_combo.py (S0-build C)
- `MODEL_CONFIGS` entries for `lightfm_tags`, `lightfm_tags_ids`.

## CLAUDE.md (S0-build C)
- `lightfm` added to the reserved `ft_type` list.

## rec_sys/protomf_dataset.py (S0-build D)
- `ProtoRecDataset`: (a) training-split negative sampling excludes the cold set when the data
  dir carries `cold_items.csv` (S0.3 leakage item 6; eval splits keep the full catalog);
  (b) val/test one-row-per-user invariant enforced at load (fatal on canonical splits, relaxed
  on the marked cold variant — F-S0-03). New `ColdTestDataset` class: cold rows with
  cold-pool/full-catalog negative sampling and canonical-consumption exclusion, seeded RNG.

## rec_sys/trainer.py + rec_sys/tester.py (S0-build D)
- `Evaluator` expectation changed from `n_users` to the dataset's declared eval-row count
  (`coo_matrix.nnz`) — identical on canonical splits (invariant checked in the dataset), correct
  automatically on the cold variant. Why: F-S0-03 / S0.3 §3.

## start.py + Master/scripts/run_combo.py (S0-build D)
- Dataset `hm_1_month_cold` registered. run_combo gains `--retrain-config <config.json>`: the
  S0.3 §6 single-config retrain convention (fixed config, num_samples=1, no search; device
  re-detected on the host).

## .gitignore (S0-build D)
- `data/hm_1_month_cold/` split artifacts ignored (derived data, reproducible via the committed
  generator).

## New files (not upstream — added for S0-build)
- `data/hm/make_cold_variant.py` — deterministic cold-variant generator (S0.3 §2)
- `utilities/cold_eval.py` — cold-eval runner: warm/cold-vs-cold/cold-vs-all, decile slices,
  per-item bootstrap CIs, attr-kNN fallback, tie-block diagnostic, popularity reference,
  use_bias refusal (S0.3 §§3–6)
- `Master/scripts/stratified_readout.py` — D4 user-history/item-popularity stratified readout
- `Master/temp/dc_checks/s0/{_harness.py, t01–t05, i01–i02}` — the S0-build test suite

## CLAUDE.md (S0-build, image-set caveat)
- "Known issue: image set may be incomplete" replaced with the verified resolution: the Kaggle
  zip's central directory holds 105,100 images in 86 subdirs 010–095; the local set matches
  exactly — complete. Why: S0.6 precondition verification (S0-build session).

## rec_sys/protomf_dataset.py (S0.7 gate fix)
- On marked cold-variant dirs, val/test negative sampling additionally excludes the user's
  removed cold purchases (`cold_test.csv` folded into the exclusion CSR; positives untouched;
  canonical datasets unaffected). Why: findings ledger F-S0-09.

## feature_extraction/feature_extractor_factories.py (dc01 SC.3 re-run — fI re-host)
- `feature_item_proto` factory branch re-pointed from the UI double-tie to the I-ProtoMF host:
  item = `PrototypeEmbedding(embedding_ext=FeatureEmbedding)` (verbatim item branch), user =
  plain `Embedding(n_users, K_t)` via the host 'prototypes' Item-Proto path; the Concatenate
  wrappers, `FeatureEmbeddingW` projection and user-prototype branch leave the live surface
  (code stays in git history; `FeatureEmbeddingW` class kept for the fUfI merge stage). Factory
  remains the single owner initializing the feature table. Guards: user branch must be
  `'embedding'`; `use_weight_matrix` must be off. Why: dc01 host redirection (dossier
  "⟳ HOST REDIRECTION" + SC.3 re-run), re-host amendment cycle af78508.

## confs/hyper_params.py (dc01 SC.3 re-run — fI re-host)
- `feature_item_proto_hyper_params` re-shaped to mirror `item_proto_chose_original_hyper_params`
  exactly (user side → plain `{'ft_type': 'embedding'}`); item side unchanged (canonical 5
  fields + `use_id_feature`). `_f0` ablation comment re-anchored: F=0+ID now reduces to
  `item_proto` (keystone i02′, bit-identical). Why: dc01 host redirection, SC.3 re-run note 3.

## feature_extraction/feature_extractors.py (dc05 build)
- Added `HistoryFeatureEmbedding` (sibling of `FeatureEmbedding`): user represented as the
  weighted sum of their train-basket attribute-word embeddings (padded non-persistent
  `hist_value_ids`/`hist_weights` buffers, mean-normalized weights, optional per-user ID row
  at offset `n_features + u`). Why: dc05 fU-ProtoMF (`feature_user_proto`), design doc §3.2.

## feature_extraction/feature_extractor_factories.py (dc05 build)
- Added `feature_user_proto` factory branch: mirrors the host 'prototypes' User-Proto shape
  with the free user embedding replaced by `HistoryFeatureEmbedding` feeding an unchanged
  `PrototypeEmbedding`; item = free `Embedding(n_items, K_u)`; factory single-owner-inits the
  history word table (F-DC01-10 pattern). Guards: item branch must be `'embedding'`,
  `use_weight_matrix` off, payload injected. Why: dc05 design doc §3.2.

## confs/hyper_params.py (dc05 build)
- Added `feature_user_proto_hyper_params` (mirrors `user_proto_chose_original_hyper_params`
  exactly + user-side `use_id_feature`/`feature_fields`), `_noid` ablation (fixed flag, never
  searched — C7/F-DC01-01), and `_debug` fixed tiny CPU-smoke config. Why: dc05 design doc §3.6.

## start.py (dc05 build)
- Registered `feature_user_proto`, `feature_user_proto_noid`, `feature_user_proto_debug` in the
  model choices + config dispatch. Why: dc05 build.

## feature_extraction/feature_extractors.py (S0-build extension, component a)
- `FeatureEmbedding` gains an optional weighted-bag layout: new `feature_weights` constructor
  arg (non-persistent buffer, same shape as `feature_ids`); forward becomes a weighted sum
  `Σ w·e_token` when weights are given (1.0 real / 0.0 padding slots), with the ID column
  concatenated before the reduction so the padding-free all-ones case is bit-identical to the
  fixed path (keystone `dc_checks/s0/t07`). `feature_weights=None` (default) preserves the
  original fixed one-value-per-field path byte-identically. Why: charter C5 ml-1m amendment —
  variable-size indicator bags (genres + Tag-Genome tags @0.8) are LightFM's native input form.

## experiment_helper.py (S0-build extension, component c)
- `start_hyper` resolves the `'canonical'` feature-fields sentinel per dataset
  (`resolve_feature_fields_in_conf`) right after setting `data_path`, BEFORE Ray serializes the
  conf — so trial configs / config.json / C8 manifests record the resolved literal list + layout.
  Why: charter C5 amendment makes the canonical field set per-dataset.

## confs/hyper_params.py (S0-build extension, component c)
- The four hardcoded H&M 5-field lists (fI, fU, fU_debug, lightfm_tags_ids) replaced by the
  `'feature_fields': 'canonical'` sentinel (single source of truth:
  `feature_ids.CANONICAL_FEATURE_FIELDS`); the F=0 ablation keeps its literal `[]` (bypasses
  resolution). Comments updated. Why: charter C5 per-dataset canonical sets.

## feature_extraction/feature_extractor_factories.py (S0-build extension, component c)
- fI and lightfm branches pass an optional injected `feature_weights` through to
  `FeatureEmbedding` (bag layout; absent = original fixed path). Why: ml-1m bags reach the
  item-side models.
