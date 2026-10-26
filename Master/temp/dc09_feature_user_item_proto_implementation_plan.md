# dc09 fUfI-ProtoMF — condensed implementation plan (SC.3, 2026-10-26)

> Scrutiny Protocol v1.3, SC.3 for an unbuilt candidate: preconditions → this plan → gate → build.
> Spec = design doc §3.1–§3.4, §6.3 and the SC.2 amendments (§6.4 B1–B13). Conventions = the
> dc01/dc05 builds (factory branch, config registration, accessor, read-out, dc_checks suite, smoke).
> Branch `feat/scrutiny`, HEAD 27dc914.

## 0. Preconditions

| # | precondition | status |
|---|---|---|
| P1 | dc01 + dc05 code present; their toy-tensor suites green **before** any edit (baseline for the regression sweep; pre-existing failures recorded, not fixed here) | run first thing after the gate |
| P2 | concept closed through SC.2 (F-DC09-01..13 dispositioned) | ✅ 2026-10-26 |
| P3 | real data for the data-dependent tests + smokes | datasets live only on LEO5 scratch (2026-07-16) → stage D on the login node, VPN required |
| P4 | cluster checkout untouched while wave 1 is pending | stage D uses a **detached worktree** `~/UIFProtoMF/ProtoMF_dev` (`/leo5-submit` caveat 1); needs a push of `feat/scrutiny` → your OK at that point |

## 1. Design decisions for this gate

**G1 — how the shared word table is realised (deviation from the SC.2 §3.2 wording; needs your OK).**
SC.2 wrote "compositions take an external word block (`word_table=`) and compose as word lookup +
ID lookup". Proposed instead: **one `nn.Embedding` with rows `[V words ; M item IDs ; N user IDs]`
(noid: `[V words]`), created and initialised once by the factory and handed to both compositions**
(`embedding_layer=` on `FeatureEmbedding` and `HistoryFeatureEmbedding`, plus `id_block_offset=M` on
the user side so its ID rows start after the item block). Why:
- the existing arithmetic is untouched (one lookup over `[words, ID]` indices, one reduction) → the
  dc01/dc05 single-table path is bit-identical by construction, not by re-proof;
- shared and `_sep` run the **same op sequence** (only the table differs) → the sharing keystone t07
  can assert scores **bitwise**, not just to machine precision (the SC.2 toy observation);
- every read-out and the cold step already index rows by offset (`n_features + i`,
  `id_row_offset + u`) → they keep working on the shared table without a word/ID accessor refactor;
- one table = one `max_norm` and one init owner automatically.
Cost: the item side's table also holds user-ID rows it never indexes (inert; weight decay touches
them exactly as it would in a separate table). Same identities as SC.2's text (D1–D2 untouched).
If accepted: one dated precision note on §3.2 (realisation, not mechanism).

**G2 — `_sep` = exactly today's fI item composition + fU user composition, each owning its table.**
No new code path for the arm; the 2×2 differs in one flag `share_word_table` (top-level
`ft_ext_param`) and the two `use_id_feature` flags.

**G3 — lean arm set: the four 2×2 configs only.** `feature_user_item_proto` (ids, shared — headline),
`_noid`, `_sep`, `_noid_sep`. Not built: `normalize_composed`/`_norm`, `_itemid_only`, dc08 identity
rows (factory asserts `n_history_rows == 0`). Each is ~10 lines if ever wanted.

**G4 — cold convention (F-DC09-08) applies fleet-wide, incl. the S0.4 baseline.** Mean-ID imputation
becomes the default for **every** feature-composed branch with ID rows: fI-ids, `lightfm_tags_ids`
(same `FeatureEmbedding`, same translation gauge — so the cold comparison stays like-for-like),
fU-ids user side, dc09 ids arms. `--cold-id-convention drop` stays available (the dc01-checkpoint
instrument needs both; reproduces landed dev-tier rows). Reports record the convention used.
Rides along (no behaviour change today): the user-side drop uses `fe.n_features` instead of
`fe.id_row_offset` — wrong once dc08 rows exist; unreachable (no cold-user testbed). Fixed in the
same rewrite.

**G5 — explanation stack split between SC.3 and SC.5.** Auto-explanations after training are off
fleet-wide (2026-08-17), so nothing here blocks runs. SC.3: accessor + intrinsic naming on both
sides + pipeline registration (naming, top-k, t-SNE, cards run post-hoc). SC.5 (protocol SC.5.4
"mandatory build"): breakdown slots (fUfI four-component + the deferred host UI slot), canonical-gauge
read-outs (F-DC09-07), "common to all users" line (F-DC09-09).

**G6 — timing flag (decide at the SC.4 gate, not now).** Nov-3 is 8 days out. Option: after SC.4,
run SC.6 and submit the 2×2 × {hm_1_month, ml-1m}, then do SC.5 while the jobs queue — a declared
step-order deviation (code frozen at submit; SC.5 touches only explanation code).

## 2. Build stages (one commit each, finding IDs in messages; modifications log for upstream files)

**A — core model + registration.**
- `feature_extractors.py` *(upstream)*: `FeatureEmbedding` / `HistoryFeatureEmbedding` gain optional
  `embedding_layer=` (external table: shape-checked, not owned, not initialised — `init_parameters`
  becomes a no-op for a borrowed table) and HFE `id_block_offset=0` (folded into `id_row_offset`).
  Defaults = today's code path, untouched. dc06 centring asserted off with a borrowed table.
- `PrototypeEmbedding.sim_from_embed(o_embed)`: today's `forward` tail split out; `forward` calls it
  (no new parameter, no RNG draw — dc01 t03 golden must stay green).
- `TwoRoleComposedExtractor(base, proto, out_dimension, invert)` per §3.2 (one q per side per step;
  `supports_loo` → base; inits only its linear layer with `general_weight_init`; loss → proto).
- `feature_extractor_factories.py` *(upstream)*: branch `feature_user_item_proto` — no config
  mutation (deep-copied sub-dicts), asserts (no weight matrix, both payloads injected, equal
  `n_features` both sides, `n_history_rows == 0`, `max_norm` unset), shared table per G1 or two
  owned tables per G2, single-owner init, returns `(user_fe, item_fe)` in the host's
  `[u*, û]` / `[t̂, t*]` layout.
- `feature_ids.inject_feature_ids`: new ft_type injects **both** payloads (item `feature_ids`
  [+ weights on bags]; user history + leave-one-out payload) from the seam's own data path.
- `confs/hyper_params.py` *(upstream)*: four configs = `proto_double_tie_chose_original` verbatim +
  per side `use_id_feature`, `feature_fields: 'canonical'`, user `loo_pooling: True`, top-level
  `share_word_table`. `start.py` *(upstream)* choices/dispatch + `run_combo.MODEL_CONFIGS`.

**B — cold.** `utilities/cold_eval.py`: imputation convention per G4 (cold rows := mean of that side's
warm ID rows; empty-basket cold users keep q_u = 0), CLI/param `cold_id_convention`, finders recurse
through `TwoRoleComposedExtractor.base`, `config_expects_id_drop` / `_user_id_drop` know the new
ft_type (and `lightfm` stays covered), offsets via `id_row_offset`. Pin: new `dc_checks/s0` test.

**C — explanation accessor (G5 SC.3 half).** `accessor/feature_user_item_proto.py` (both prototype
sets, both word views), registry entry, pipeline `EXPLAINABLE_MODELS`, naming route
`name_prototypes_intrinsic` per side; generic explainers' `supports()` sets.

**D — cluster (login node, detached worktree; VPN + push OK needed).** Real-data tests, a smoke
profile run of each of the four arms on hm_1_month and ml-1m (`run_combo --profile smoke`), and the
dc01-checkpoint cold instrument (gauge-shift a trained fI-ids hm_1_month_cold checkpoint, warm
scores unchanged, cold metrics under drop vs imputation — the size of F-DC09-08's effect).

**E — regression sweep** of the dc01 / dc05 / dc07 / dc08 / s0 suites vs the P1 baseline.

## 3. Tests — `Master/temp/dc_checks/dc09/`

| test | checks | where |
|---|---|---|
| t01 keystone | F = 0, ids on both sides: weight-copied fUfI ≡ `user_item_proto` — scores bitwise (f32 + f64), loss equal, grads to machine precision (4b C4) | local |
| t02 LOO | in training, both user roles see the same leave-one-out q_u (4b C8 / SC.2 D2) | local |
| t03 single-table path | `FeatureEmbedding`/HFE with the new kwargs at defaults ≡ today (outputs, param shapes, RNG); borrowed table with the same weights ≡ owned | local |
| t04 registration | four configs in both registries; canonical fields resolve on both sides; no tensor in the saved config; factory does not mutate the config | local |
| t05 cold | finders reach the composition through the wrapper; imputation covers both roles of the one shared instance; abort fires on structure drift; noid arms no-op | local |
| t06 four components | S = B_item + U-half + B_user + I-half exactly on a built model (4b C1) | local |
| t07 sharing keystone | shared ≡ `_sep` with tied word rows: scores bitwise; word-row grads = sum of the two sides' grads (sampled softmax + BCE) | local |
| t08 vocab alignment | item `feature_ids` codes ≡ the user builder's item tokens; equal `n_features` — hm_1_month + ml-1m | cluster |
| t09 checkpoint | save/load round-trip; shared table is one Parameter after load; `parameters()` dedups it | local |
| s0 pin | imputation = warm mean; warm rows untouched; gauge-shifted toy model: imputation invariant, drop not | local |

## 4. Not in SC.3

Canonical-gauge read-outs, breakdown slots, "common to all users" line (SC.5); the one-off
descriptive measurements of F-DC09-09/10/11 (post-run, SC.6 plan); any run that produces a compared
row (SC.6 → `/leo5-submit`, one per run).
