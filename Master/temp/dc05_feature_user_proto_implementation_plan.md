# Implementation + Testing Plan — `feature_user_proto` (dc05 build, SC.3-style)

> Build-session gate artifact, 2026-07-12 session (plan drafted from the reviewed
> design doc). Governing inputs: design doc
> `Master/docs/design_candidates/dc05_history_composed_user_factors.md`
> (status: reviewed — awaiting build; review closed 2026-07-12, vault
> `2026-07-12_1840`), fU brief M1–M7, scrutiny protocol SC.3 (from-scratch
> clause: condensed plan → gate → implement, dc01/dc02 build conventions).
> Branch: `feat/scrutiny`. All conventions below verified against the ACTUAL
> code this session, not the modification map.

## GO gate (read before implementing)

The mechanism, mean normalization, both configs, `ft_type` name, and renderer
defaults are already user-ratified (review 2026-07-12). This gate is about the
**build plan only**: part boundaries, test matrix, and the six resolved
decisions at the bottom. The three review-added build obligations are in the
test matrix: **A1 term-by-term objective equality (t06)**, **variant-train
weight rebuild (t02c)**, **cold-user ID-drop mirror (t07)**.

## Verified ground truth (drives the design; checked this session)

- **Injection seam:** `inject_feature_ids(ft_ext_param, data_path)` is called
  in `trainer._build_model` (trainer.py:67), `tester._build_model`
  (tester.py:50), `explanations/loader.py:46`, and `cold_eval.py:92` — the
  cold path passes `variant_path`, so a dc05 branch in this one function gets
  the variant-train weight rebuild for free (M6.1a leakage rule).
- **Split files:** `listening_history_train.csv` has columns
  `t_dat,customer_id,article_id,user_id,user,item_id,item`; the training CSR is
  built from its raw (user_id, item_id) pairs (protomf_dataset.py:90–94,
  `np.ones`). The splitter dedups (customer, article) keep-first; the builder
  defensively dedups the same key so w̄ counts exactly what the matrix sees.
- **Host path:** `user_proto` = ft_type `prototypes`, User-Proto branch of
  `create_models` (factories.py:42–51): user = `PrototypeEmbedding` (internal
  free `Embedding`), item = `create_model(item_param, n_items,
  user_n_prototypes)` → free item vector t ∈ R^{K_u}. Config =
  `user_proto_chose_original_hyper_params` (hyper_params.py:53).
- **Conventions to inherit:** non-persistent buffers (rebuilt at build, never
  checkpointed — `FeatureEmbedding`/`AttributeLookup` precedent); factory
  single-owner init of the passed-in extractor (F-DC01-10 pattern,
  factories.py:150); tensors never serialized into the Ray/JSON config
  (lightweight spec keys only); `PrototypeEmbedding(embedding_ext=...)` is the
  dc01-proven entry point and needs **zero changes**.
- **Renderer:** `breakdown.py:629` raises the deliberate deferral for
  `user_proto`/`user_item_proto` — that is the slot this build fills (fU +
  host `user_proto`, like-for-like). `Breakdown`/`BreakdownLine` dataclasses,
  `feature_code_labels`, figure/text/write helpers all exist and are reused.
- **Cold machinery:** `drop_cold_id_rows` (cold_eval.py:159) zeroes cold ITEM
  ID rows inside the item branch; `config_expects_id_drop` keys on
  `feature_item_proto`/`lightfm` — fU's item branch is a free `Embedding`, so
  the item-side drop must be a declared no-op for fU (guarded, not accidental).
- **V1 measured facts (design doc §4):** N=73,418, M=13,651, V=426,
  D_max=112 → padded buffers ≈ 99 MB — the padded layout is fine on V1; the
  `embedding_bag` fallback stays declared-only.

## Parts (in build order; each independently testable)

**P1 — `HistoryFeatureEmbedding`** (`feature_extraction/feature_extractors.py`,
sibling of `FeatureEmbedding`). Per design doc §3.2: non-persistent buffers
`hist_value_ids` (N, D_max) int64 + `hist_weights` (N, D_max) float32 (0.0 on
padding, word id 0 on padding rows — contributes exactly nothing);
`nn.Embedding(n_features + N·[use_id], d, max_norm)`; forward = weighted
gather-sum + optional `e_ID(u)` row at offset `n_features + u`;
`init_parameters` applies `general_weight_init` (dc01 `FeatureEmbedding`
parity); accepts (B,) and (B, n) index shapes like every extractor.

**P2 — builder + injection** (`feature_extraction/feature_ids.py`).
`build_user_history_weights(data_path, fields)`: reads the split dir's OWN
`listening_history_train.csv` + `item_features.csv`; reuses `build_feature_ids`
verbatim for the global field-offset vocabulary (labels stay
renderer-compatible); dedups (user_id, item_id) keep-first; per user counts
attribute values over the basket, mean-normalizes (w̄ = n/|H_u|), pads to
D_max over **distinct values per user**. Users absent from train (possible on
variants only) get an all-zero row → q_u = e_ID(u) or 0 (verified degenerate
path, 4b C3). `inject_feature_ids` gains the `feature_user_proto` branch —
same shallow-copy discipline, payload (`hist_value_ids`, `hist_weights`,
`n_features`) into the USER sub-dict copy.

**P3 — factory branch** (`feature_extractor_factories.py`), per design doc
§3.2 pseudo-code: asserts (item branch `'embedding'`, `use_weight_matrix`
off, payload injected); user = `PrototypeEmbedding(n_users, d, K_u, ...,
embedding_ext=HistoryFeatureEmbedding(...))`; item = `create_model(item_param,
n_items, user_n_prototypes)` (host sizing, t ∈ R^{K_u}); factory single-owner
`init_parameters()` on the hist embed.

**P4 — configs + registration.** `feature_user_proto_hyper_params` mirrors
`user_proto_chose_original_hyper_params` EXACTLY (search space + dev-profile
comparability, M1 stage bar) with two USER-side additions: `use_id_feature:
True` + `feature_fields` = the canonical 5 (charter C5, same list as dc01);
`feature_user_proto_noid` = deepcopy with `use_id_feature=False` (charter
C7 / F-DC01-01: never a searched flag). Register in `start.py` +
`Master/scripts/run_combo.py`. CLAUDE.md ft_type row flips "reserved at dc05
registration" → wired; `modifications_log.md` rows for upstream files.

**P5 — cold/variant conventions.** (a) fU-aware guard in
`config_expects_id_drop` / runner path: fU configs must NOT expect the
item-side drop (item branch has no `FeatureEmbedding`) — explicit, tested
no-op. (b) **Cold-user ID-drop mirror** (design doc §3.4/§3.7-6, F-S0-07
mirror): `drop_cold_user_id_rows(model, variant_path)` zeroing cold USERS'
`e_ID(u)` rows in the user branch's `HistoryFeatureEmbedding`, wired
symmetrically to the item-side function. No cold-USER variant generator exists
this phase (M6.2: scrutiny-stage) — the function + test use a toy user-list
CSV seam so the convention is pinned now and the future testbed plugs in.

**P6 — explanations wiring** (all ratified defaults from the review).
- `accessor/feature_user_proto.py` (+ registration in `accessor/__init__.py`
  and the loader/pipeline seams, mirroring `feature_item_proto.py`).
- `breakdown.py`: `compute_breakdown_feature_user_proto` — top-6 per-prototype
  personalized bars t_l·cos_l (signed), **B(t) = 1ᵀt line** with the ratified
  NOT-rank-inert wording ("non-personalized, this item's own scalar, identical
  for every user"), zoom = top community's **per-purchase shares** (top-6 by
  |share| + stated remainder; word-level zoom in the md companion), user-ID
  row as `is_id` line, footer = feature-explained fraction of the
  personalized score; self-checks: B(t) + Σ personalized == forward score,
  zoom sums to its parent bar. `compute_breakdown_user_proto` — host slot,
  same panel math on t_l·(u*_l−1) + the same B(t) line (GR7-fair), zoom =
  post-hoc top-k-nearest-users consumed-item lift, `naming_route` declared
  post-hoc. Replaces the 629 deferral for `user_proto` only
  (`user_item_proto` stays deferred to fUfI).
- Prototype cards + naming: fU user-prototype cards = intrinsic word-profile
  bars (cos(e_f, p^u_l), read-out 4); host cards = post-hoc descriptors; same
  card geometry (`proto_cards.py`/`naming.py` extension, per-side).

**P7 — dc_checks/dc05 suite + smoke** (beside the existing
`check_claims.py`/`claims_spec.md`): tests below + `smoke_train.py`-style
local CPU run via a `feature_user_proto_debug` fixed tiny config (dc02
convention) on a toy split.

## Test matrix (`Master/temp/dc_checks/dc05/`)

| id | pins |
|---|---|
| t01 | `HistoryFeatureEmbedding` unit: weighted sum vs hand-computed; padding contributes exactly 0; ID row offset; (B,) and (B,n) shapes; max_norm live |
| t02 | builder: w̄ vs hand-computed toy CSVs; vocab/offsets identical to `build_feature_ids`; dedup matches the training CSR's binary pattern; **(c) variant-train rebuild:** on a cold variant dir the weights come from the VARIANT train file — removed rows provably absent (M6.1a leakage pin) |
| t03 | factory branch: shapes, asserts fire, single-owner init reached, item table is M×K_u |
| t04 | config registration: both configs resolve in start.py/run_combo; search spaces == host's; noid differs ONLY in `use_id_feature`; no tensor in serialized config |
| t05 | **keystone bit-identity (4b C5′, i02 pattern):** fU(F=0, +ID) with host user rows copied into the ID block ≡ `user_proto` — bit-identical u*, scores, and loss on a toy split |
| t06 | **A1 term-by-term objective equality (vault 1739 obligation):** at the zero-word mapped point — rec loss, BOTH inclusion regularizers, parameter-L2 mass, max_norm feasibility each equal the host's value term by term |
| t07 | **cold-user ID-drop mirror:** drop zeroes exactly the cold users' ID rows; warm users bit-identical; noid arm clean no-op; item-side `config_expects_id_drop` no-op for fU configs |
| t08 | checkpoint roundtrip: strict `load_state_dict` (buffers non-persistent), scores identical after save→rebuild→load |
| t09 | gradient paths (i03 pattern): grads reach word table/ID rows/prototypes/item table; none on buffers; a padded slot's word row gets zero gradient from that slot |
| t10 | breakdown/readout: fU + host artifacts render; self-checks green on a trained toy checkpoint; per-purchase zoom == per-word zoom regrouped (4b C2′ on the real module) |
| i06 | regression buildcheck: every existing model (mf, user/item/ui_proto, fI×3, dc02, lightfm×2) still assembles + one train step (dc01 i06a pattern) |

## Resolved design decisions (flag at gate; one each)

1. **Padded layout, not embedding_bag** — V1 D_max=112 ≈ 99 MB buffers; the
   flat fallback stays a declared mitigation for hm_3_month+ (design doc §4).
2. **Builder dedups (user_id, item_id) keep-first** — matches splitter output
   and the binary training CSR; pinned by t02, disclosed in the docstring.
3. **User payload keys mirror dc01's item-side names** (`feature_fields`,
   `use_id_feature`) — maximal reuse of labels/renderer/ID-drop conventions.
4. **Empty-basket rows = all-zero weights** — the 4b C3-verified degenerate
   path (u* ≡ 1 → S = B(t)); no special-casing, no NaN guard needed beyond
   torch's eps cosine.
5. **`drop_cold_user_id_rows` is a separate function**, not a generalization
   of the item-side one — the two sides key on different variant metadata;
   symmetric shape, independent test.
6. **Debug config `feature_user_proto_debug`** (fixed tiny values, dc02
   convention) added for local CPU smoke — not a fleet row.

## Verification commands (always `conda activate protomf` first)

```
# unit/integration suite:
cd Master/temp/dc_checks/dc05 && for t in t0*.py i0*.py; do python $t; done
# local CPU smoke (toy split, no cluster):
python Master/temp/dc_checks/dc05/smoke_train.py
# existing-model regression:
python Master/temp/dc_checks/dc05/i06_regression_buildcheck.py
# CLUSTER (later, separate /leo5-submit per run — NOT this session):
#   run_combo.py -m feature_user_proto -d hm_1_month --profile dev
```

## Files touched

| file | change |
|---|---|
| `feature_extraction/feature_extractors.py` | + `HistoryFeatureEmbedding` (upstream file → modifications_log) |
| `feature_extraction/feature_ids.py` | + `build_user_history_weights`, inject branch |
| `feature_extraction/feature_extractor_factories.py` | + `feature_user_proto` branch (upstream → log) |
| `confs/hyper_params.py` | + 3 configs (fU, noid, debug) (upstream → log) |
| `start.py`, `Master/scripts/run_combo.py` | registration (start.py upstream → log) |
| `utilities/cold_eval.py` | + user-side drop, fU guard |
| `utilities/explanations/{breakdown.py, accessor/*, explainers/*, naming/*}` | fU + host user_proto slots |
| `Master/temp/dc_checks/dc05/*` | suite (t01–t10, i06, smoke, _harness) |
| `CLAUDE.md`, `Master/docs/modifications_log.md` | ft_type row, log rows |

## Commit discipline

Per-part commits on `feat/scrutiny` via /commit (`feat(dc05): ...`), tests
committed with the part they pin; renderer/explanations as their own commit;
docs/registration last. No push without ask.
