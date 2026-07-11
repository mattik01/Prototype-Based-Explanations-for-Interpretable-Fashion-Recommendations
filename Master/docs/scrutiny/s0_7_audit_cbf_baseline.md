# S0.7 black-box audit — lightfm CBF baseline vs S0.4 spec (verbatim)

> Independent spec-vs-code subagent, 2026-07-11, HEAD `93b1edd`. Inputs per
> Ground rule 10: ONLY the S0.4 spec section (standalone extract, rev. 1) +
> the S0.5 canonical-field-set section, and the implementation files. No
> session context, no S0-build narrative. The scrutinizing session's response
> lives in the dossier's S0.7 section.

---

# Spec-vs-Code Audit: S0.4 LightFM-style CBF Baseline (`ft_type='lightfm'`)

Audited: spec_s04_cbf_baseline.md (rev. 1) + spec_s05_feature_audit.md §2 against the implementation at `/home/mattik01/Desktop/githubs/ProtoMF` (branch `feat/scrutiny`). All claims below verified by reading code and by a read-only programmatic diff of the config dicts (no training run).

---

## §1 — Isolation argument / model count

Spec: ONE trainable model, two configs via `use_id_feature`; no attribute-kNN standalone recommender; no FM/NFM/DeepFM/NCF+.

- **CONFORMS.** Exactly one code path (`ft_type == 'lightfm'`, `feature_extractor_factories.py:224-249`) serves both rows; the two configs differ only in the flag (verified programmatically, see §3.3 below). No kNN recommender model, no FM-family model exists anywhere in `confs/hyper_params.py`, `start.py`, or `Master/scripts/run_combo.py`.
- Naming: the retired working name `feature_mf` appears nowhere; all registration reads `lightfm_tags` / `lightfm_tags_ids`. **CONFORMS.**

## §2 — Model definition

**User branch = plain CF `Embedding`** — **CONFORMS.**
`feature_extractor_factories.py:235-236` builds the user side via `create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)`; both configs set `user_ft_ext_param = {"ft_type": "embedding"}` (`confs/hyper_params.py:275-277`), which instantiates `Embedding` (`feature_extractors.py:42-77`) with defaults `max_norm=None`, `only_positive=False` — a plain free CF embedding.

**Item branch = `FeatureEmbedding` summing feature-value embeddings over the canonical 5-field set, optional ID row** — **CONFORMS.**
- Factory: `feature_extractor_factories.py:244-246` builds `FeatureEmbedding(n_items, feature_ids, n_features, embedding_dim, use_id_feature, max_norm)`.
- Sum semantics: `feature_extractors.py:176-180` — `self.embedding_layer(feat).sum(dim=-2)`, i.e. `q_i = Σ_f e_f`; ID row appended at index `n_features + o_idx` when `use_id_feature=True` (lines 177-179), matching the LightFM tags+ids composition.
- Field set: `confs/hyper_params.py:281-287` lists exactly `department_name, product_type_name, section_name, colour_group_name, graphical_appearance_name` — the S0.5 §2 canonical 5, in the S0.5-listed order. No extra fields, no `price_band`, no `product_code` (S0.5 §3 dispositions honored).

**Score = plain dot product, NO biases (`use_bias=0` parity)** — **CONFORMS.**
- `rec_sys/rec_sys.py:114`: `dots = torch.sum(u_embed.unsqueeze(1) * i_embed, dim=-1)` — the same path `mf` uses; bias addition (line 116-118) is gated on `self.use_bias`, and with `use_bias=0` the bias modules are never even constructed (lines 36-41).
- Both lightfm configs inherit `base_param['rec_sys_param'] = {'use_bias': 0}` via the `**base_hyper_params` spread (`confs/hyper_params.py:6-12, 269-270`). No prototype machinery anywhere in the branch; `FeatureExtractor.get_and_reset_loss` returns the untouched `0.` for both extractors (no reg loss in the scored/objective path; t03 §5 pins this).
- **Beyond-spec observation (latent hazard, no current effect):** `rec_sys.py:36` defaults `use_bias` to **True** when the key is absent (`... if 'use_bias' in self.rec_sys_param else True`). Fleet bias-parity therefore hangs entirely on `base_param` propagation. Any future config built without the `base_param` spread would silently get biases. Severity: none today (both configs inherit it), worth a defensive note.

**Configs / roles** — **CONFORMS.** `lightfm_tags_ids` sets `use_id_feature=True` (`hyper_params.py:280`); `lightfm_tags` is the deepcopy with the flag `False` (`hyper_params.py:292-293`). Cold capability of `lightfm_tags`: with `use_id_feature=False` the item vector depends only on `feature_ids`, which covers every catalog item 0…n−1 (`feature_ids.py:48-57` coverage guard) — unseen-in-train items score finitely and non-constantly (pinned by t03 §4).

## §3 — Implementation slot, item by item

**§3.1 Factory branch** — **CONFORMS.** `'lightfm'` branch at `feature_extractor_factories.py:224-249`, `detached`-style, pure reuse of `Embedding` + `FeatureEmbedding`; no new module was added. It asserts injection happened (`:239-240`) before consuming the tensor. Init discipline is correct: the factory does no init (single consumer), and `RecSys.init_parameters` (`rec_sys.py:70-71`) initializes both extractors — unlike dc01 there is no shared instance needing single-owner init.

**§3.2 Injection guard** — **CONFORMS.** `feature_ids.py:182`: `if ft_type in ('feature_item_proto', 'lightfm'):`. Non-mutating discipline verified in code: shallow-copied outer dict + shallow-copied item/user sub-dicts, tensor added only to the copies, caller's dict returned untouched for all other types (`feature_ids.py:182-201`). Call sites bind the injected copy to a *local* variable in both `trainer._build_model` (`trainer.py:67`) and `tester._build_model` (`tester.py:50`) — `self.ft_ext_param` never gains the tensor, so `best_trial_config` → `config.json` stays clean. Additionally the tensor is a **non-persistent buffer** in `FeatureEmbedding` (`feature_extractors.py:154`), so it never round-trips through checkpoints either.

**§3.3 Search spaces mirror `mf_hyper_params` EXACTLY** — **CONFORMS** (diffed programmatically under the `protomf` env). Between `mf_hyper_params` and `lightfm_tags_ids_hyper_params`:
- Top-level key sets identical; `loss_func_aggr='mean'` both.
- Identical (shared via `base_hyper_params`): `neg_train` randint(1,50), `train_neg_strategy` {popular, uniform}, `loss_func_name` {bce, bpr, sampled_softmax}, `batch_size` {64,128,256,512}, `optim` {adam, adagrad}, `wd` loguniform(1e-4,1e-2), `lr` loguniform(1e-4,1e-1), `n_epochs`, `eval_neg_strategy`, `val_batch_size`, `rec_sys_param`.
- `embedding_dim`: `tune.randint(10, 100)` in both (`hyper_params.py:32` vs `:274`).
- The ONLY differences are exactly the spec-mandated ones: `ft_type` 'detached'→'lightfm', item side gains `use_id_feature` + `feature_fields` (canonical 5).
- `lightfm_tags` vs `lightfm_tags_ids`: programmatic deep-diff shows a **single** difference — `use_id_feature False vs True`; created via `copy.deepcopy` (`hyper_params.py:292`), independent objects confirmed. **CONFORMS.**

**§3.4 Registration** — **CONFORMS.**
- `start.py:20-21` (CLI choices) and `:63-66` (dispatch) register both names; imports at `start.py:8`.
- `Master/scripts/run_combo.py:43-44` (imports) and `:62-63` (`MODEL_CONFIGS`), choices derived from `MODEL_CONFIGS.keys()` (`:452`).
- CLAUDE.md reserved `ft_type` list contains `lightfm` (CLAUDE.md:40). **CONFORMS.**
- `hm_1_month_cold` is a valid dataset in both entry points (`start.py:24-25`, `run_combo.py:66-67`).

**§3.5 Tests** — **CONFORMS** (existence + coverage; not re-run). `Master/temp/dc_checks/s0/t03_lightfm_baseline.py` covers: injection-seam no-mutation (§1 of the test), shapes (§2), the **keystone F=0 bit-identity to `mf`** via `torch.equal` on logits (§3, test lines 86-103), cold path finite/non-constant + signature-twin exact ties (§4), no extractor reg loss (§5), checkpoint roundtrip (§6), gradient flow (§7). `i01_lightfm_toy_train.py` runs Trainer+Tester end-to-end for both configs. I independently confirmed the F=0 reduction is architecturally sound: `feature_ids` shape (n,0) → forward concatenates only the ID column → single `nn.Embedding(n_items, d)` lookup, structurally identical to `Embedding`.

## §4 — Parity contract

Code-checkable parts **CONFORM**: `NEG_VAL = 99` and uniform `eval_neg_strategy` (`utilities/consts.py:28`, `hyper_params.py:9`); metric set K ∈ {1,3,5,10,50} (`consts.py:27`); `hit_ratio@10` selection (`consts.py:29`); `SINGLE_SEED` (`consts.py:4`); both splits registered. The S0.7 reference-freeze runs themselves are future protocol execution, not code — **NOT AUDITABLE HERE** (not a divergence).

## §5 — Cost estimate

Planning-only; no code commitment. N/A.

---

## Summary table of divergences

| # | Item | Spec ref | Code evidence | Severity | Verdict |
|---|---|---|---|---|---|
| — | *No divergences found.* Every §2/§3 commitment is implemented as specified. | — | — | — | — |
| O1 | `RecSys` defaults `use_bias=True` when key absent; bias-free parity depends solely on `base_param` propagation | §2 "use_bias=0 fleet parity" | `rec_sys/rec_sys.py:36` | latent (no effect on current configs) | beyond-spec observation |
| O2 | Factory defaults `use_id_feature=True` when key absent — a misspelled flag in a future config would silently become tags+ids | §2 configs | `feature_extractor_factories.py:242` | latent (both configs set it explicitly) | beyond-spec observation |
| O3 | `base_hyper_params` tune sampler objects are the *same instances* shared across `mf` and both lightfm configs (dict spread, not deepcopy) | §3.3 "mirror exactly" | `hyper_params.py:14-25, 270` | cosmetic (Ray samples per-trial; arguably the strongest possible form of "exact mirror") | beyond-spec observation |

**Spec items with NO implementation:** none (all of §3.1–§3.5 present; §4's reference runs are scheduled work, not code).
**Unspecified extras:** none in the lightfm path — the branch adds no knobs beyond the spec'd `use_id_feature`/`feature_fields`/inherited `max_norm` passthrough (`max_norm` is unset in both configs, so inert).

**Overall verdict: FULL CONFORMANCE.** The implementation matches spec S0.4 rev. 1 and the S0.5 canonical field set on every hard commitment; the three observations above are latent robustness notes, none affecting scored results.
