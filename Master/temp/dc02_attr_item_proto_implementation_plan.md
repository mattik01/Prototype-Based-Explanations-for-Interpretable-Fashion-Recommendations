# dc02 `attr_item_proto` — Implementation Plan (condensed)

> Authoritative design: `Master/docs/design_candidates/dc02_attribute_space_prototypes.md`
> (§3 mechanism, §3.4 read-out incl. the û_k·(t*−1) amendment, §3.5 constraints) +
> `Master/temp/dc_checks/dc02/claims_spec.md` (11 claims, black-box-confirmed by the frozen
> `check_claims.py` — never edited).
> Branch: `feat/attr-item-proto` off `feat/feature-item-proto`. Full plan (context, decisions,
> risks): approved plan of 2026-07-07 (session plan file); this doc is the repo-side condensate.
> Harness: plain assert scripts in `Master/temp/dc_checks/dc02/`, conda env `protomf`
> (torch 1.9.1 CPU), no pytest. Every upstream edit → `Master/docs/modifications_log.md`.

## Mechanism (one idea)

Item prototype space moves from CF latent space to attribute space: prototypes A ∈ R^{K×V};
item = observed multi-hot x_i ∈ {0,1}^V (9 H&M fields → V=534 on hm_3_month; exactly 9 ones/row);
t* = 1+cos(x_i, a_k); t̂ = W_t·x_i (bias-free Linear(V, L_u)); item extractor =
Concat(item_proto, item_proj, invert=True) → [t̂; t*]. User branch = host double-tie, byte-identical.
No per-item parameters; no biases (`use_bias=0` — already the `base_param` default). Host
R_batch/R_proto transplant unchanged. Four constraint knobs, config-gated, DEFAULT OFF:
K1 softplus nonnegativity (A=softplus(Ã)), K2 field-crispness entropy, K3 separation hinge
(plain cos, margin m), K4 data pull over a per-step resampled item subset (dedicated
torch.Generator — global RNG stream untouched).

## Parts (build order) and tests

| part | content | test |
|---|---|---|
| P1 | `AttributeLookup(FeatureExtractor)` — non-persistent float32 buffer (M,V); parameter-free | t01 |
| P2 | `build_attr_multi_hot(data_path, fields)` in `feature_ids.py` (+ guards: item_id coverage, NaN, row sums==F) + `inject_feature_ids` elif for `attr_item_proto` (same non-mutating copy discipline; injects `attr_multi_hot`, `n_attr_values`, `field_offsets`; config keeps only `attr_fields`) | t06 |
| P3 | `AttributePrototypeEmbedding(PrototypeEmbedding)` — base mode: super().__init__(embedding_dim=V, embedding_ext=lookup); forward mirrors host with `effective_prototypes()`; same reg accumulation | t02 (host-equivalence keystone; claim 3; claim 7) |
| P4 | `AttributeProjection(FeatureExtractor)` — SHARED lookup + Linear(V, L_u, bias=False) | t03 |
| P5 | factory `elif ft_type == 'attr_item_proto'` (user side byte-identical to double_tie; ONE shared lookup) | t04, i01, i03 |
| P6 | knobs K1–K4 in `get_and_reset_loss()` (once per step, differentiable, replica-safe) | t07 (claims 8–11 + RNG isolation) |
| P7 | `attr_item_proto_hyper_params` + `_debug` + `_debug_knobs` confs; start.py; run_combo MODEL_CONFIGS (its EXPLAINABLE_MODELS untouched) | t05 |
| P8 | keystones/roundtrips/regressions | i02 (spec equivalence vs frozen toy impl), i04 (ckpt roundtrip, base+nonneg), i05 (cold item & exact ties, claims 2+6), i06 (buildcheck) + FULL dc01 suite rerun |
| P9 | read-out: `utilities/explanations/attr_readout.py`; `AttrItemProtoAccessor` (+registry); namer refactor → `name_prototypes_from_score_matrix` (dc01 `name_prototypes_intrinsic` delegates); `attr_naming_config(min_score=0.02, scoring='attr')`; pipeline branch on `has_attr_space_prototypes` reading `attr_fields`; explainers `supports()` += | t08, t10 (incl. dc01 naming regression) |
| P10 | `readout_demo.py` / `readout_artifact.py` | manual on smoke ckpt |
| P11 | local CPU smoke: `run_combo.py -m attr_item_proto_debug[-_knobs] -d hm_3_month --skip-explanations`; then pipeline + demo on the checkpoint | pass criteria: trials complete, finite metrics, no NaN, knobs-on loss decreasing, `explanations/attr/` artifacts, exact-decomposition sum check |
| P12 | modifications_log dc02 section; CLAUDE.md reserved ft_type += `attr_item_proto`; commits (/commit) | — |

## Key invariants

- Zero edits to existing class bodies in `feature_extractors.py`; only shared-code edits are the
  guarded `inject_feature_ids` elif and the namer profile-loop extraction (dc01 t10 rerun = proof).
- X float32, non-persistent buffer (tester loads strict=True); tensors never in config.json.
- K1 state_dict key stays `prototypes` (stores Ã when nonneg); semantics fixed by config flag —
  tester/loader rebuild from the checkpoint's own config.json (hazard documented in i04).
- Read-out renders item-discriminating û_k·(t*−1) with the user-constant Σû_k disclosed separately
  (§3.4 amendment); per-attribute shares A[k,v]/(√F‖a_k‖) sum exactly to t*−1.
- `per_field_mass` per-field diagnostic ships with the read-out (collapse detector, §3.5(c)).

## Out of scope this round

Minimal single-sided variant; attribute-kNN/LightFM baselines; full-catalog tie diagnostic
(no full-catalog scoring path exists); cold-slice eval protocol; LEO5 runs; Rashomon knob tuning.
