# Implementation + Testing Plan — `feature_item_proto` (design candidate dc01)

> Plan only. No code written. Implementation is **GATED** — see "GO gate" below.
> Authoritative design source: `Master/docs/design_candidates/dc01_feature_composed_factors.md`.
> (Recorded copy of the approved plan; canonical plan file: `~/.claude/plans/vpn-is-active-as-spicy-backus.md`.)

## Context

**Why:** The thesis extends ProtoMF with feature-aware item prototypes. dc01 ("Feature-composed
item factors under the double-tied prototype layer") is the first design candidate: in the existing
`user_item_proto` model (`ft_type='prototypes_double_tie'`), replace the item branch's **free per-item
ID embedding** with a **feature-composed embedding** `q_i = Σ_{f∈f_i} e_f` (sum of per-feature-value
embeddings, LightFM/SVDFeature style) under the *otherwise-unchanged* prototype layer, regularizers,
losses, tie, user branch, and training loop — plus an **intrinsic per-feature read-out** (exact
additive decomposition of each item-prototype activation; global prototype attribute profile).

**Problem this addresses:** ProtoMF's prototype *meaning* is currently post-hoc (item prototypes named
by nearest items). dc01 makes item prototypes ground in feature space intrinsically and gives cold
items a representation from features alone, while keeping ProtoMF's accuracy capacity.

**Intended outcome:** a new `feature_item_proto` CLI model that (a) reduces exactly to `user_item_proto`
at F=0 (the correctness keystone, proven LOCALLY on toy tensors), (b) trains end-to-end on the canonical
`hm_1_month` split on **LEO5** (the laptop can't train it; toy I1 is the local proxy), (c) exposes the
per-feature read-out — every part individually unit-tested so integration is provably correct.

**Math status:** verified exact against the three source PDFs (LightFM, SVDFeature, ProtoMF) and the toy
script `check_claims.py` (C1–C6). `t*_k − 1 = Σ_f e_f·p_k/(‖q_i‖‖p_k‖)` is exact; F=0 reduces exactly;
per-feature shares are jointly normalized through `‖q_i‖` (not counterfactual). The implementation can
faithfully realize dc01.

### GO gate (read before implementing)
This plan is the deliverable for tonight. **Do not start coding until:**
1. The user completes the recommended reading list and signs off that dc01 is sound to build
   (validation of grounding/novelty is unfinished — see Open Questions A). Target ~2026-06-17.
2. The scope below is confirmed (defaults stated; adjust if desired).

---

## Open Questions & Pending Validation (TRANSPARENT — kept open by request)

These do **not** block writing/iterating the implementation, but several gate the *decision to build* and
the *thesis claims*. None affect whether the code can be made correct (the math checks out).

**A. Pending the user's literature validation (reading list — GO-gating, user's to close):**
- **A1 — dc01 fully validated?** The reading list (notably **S5** part-prototype challenges, **S7**
  attribute-aware survey) and a complete prior-art check are not finished. Until then the GO decision is
  *pending*, and additional open questions may surface. Plan proceeds on the math; build waits on sign-off.
- **A2 — novelty framing.** Independent verification *and* dc01's own §5b flag that the core math is "the
  distributive law" (elementary). Whether the contribution holds as publishable-incremental rests on the
  empirical behavior + the intermediate concept layer, and on S5/S7 positioning. To resolve via the reading.

**B. Empirical bets — measured by experiments later, NOT by this implementation (not blocking):**
- **B1 — R4↔R6 coupling** via `use_id_feature`: explanation fidelity vs. capacity (LightFM warm gained only
  +0.001 AUC from features → ID rows may dominate).
- **B2 — degeneration risks** the base design does NOT enforce against: diffuse/overlapping prototype
  feature-profiles (M1/M4), frequent-value dominance ("Solid" ≈ 58% of items; M2/M3).
- **B3 — R7 vocabulary bias:** taxonomy (department, lift 5.2) likely dominates perceptual features →
  fashion-explanation framing risk.
- **B4 — R4 metric undefined:** "feature-explained fraction" needs an operational definition before it is
  measurable. (Worth pinning down before the eval phase.)

**C. Cheap pre-implementation checks to schedule (not blocking the build):**
- **C1 — no-ID equivalence-class ceiling:** count distinct feature-tuples among items (quantifies
  rank-starvation under `use_id_feature=False`). One-off pandas query on `item_features.csv`.
- **C2 — H&M cluster data / `hm_1_month` pull:** the canonical `hm_1_month` (baseline split, K-core 5,
  13,651 items) is on LEO5; headless SSH from this session failed on **publickey** (needs your loaded key /
  interactive VPN auth), so the exact remote data path is unconfirmed. **Action for you:**
  `! ssh leo5 'find /scratch/c7031336 /home/c703/c7031336 -maxdepth 5 -type d -name hm_1_month'`, then scp
  per P0. Not blocking the code work (unit tests + the I2 keystone need no dataset) — it gates only the
  end-to-end smoke (I5).

**D. Data/naming caveats (informational):**
- **Resolved by this revision:** smoke/integration tests now target **`hm_1_month`** — dc01's V1 split AND
  the SAME split the recorded ProtoMF baselines ran on (**73,418 users × 13,651 items × 623,230 inter.,
  K-core 5**; `user_item_proto` baseline HR@10 0.6606 — `Master/docs/replication_report.md`). The canonical
  copy lives on **LEO5** (baselines + best configs under `/scratch/c7031336/protomf_results/…`). It is NOT on
  the laptop → Prerequisite **P0 pulls it from the cluster** (do NOT regenerate — see P0). `hm_3_month`
  (26,963) / `hm_full` (90,690) remain local for larger runs. This aligns the plan with dc01's V1 AND keeps
  feature_item_proto comparable to the baselines.
- `Master/docs/modification_map.md §B` describes an OLDER, contradictory design (`feature_aware_prototypes`,
  MLP, `dual_item_proto`). **Follow dc01** (authoritative); modification_map is explicitly "not an absolute
  source of truth" per CLAUDE.md.

**E. Explanations-pipeline integration — RESOLVED: defer (user's call).** `run_combo.py` auto-invokes
`utilities/explanations/pipeline.py::run_explanations_pipeline` for `EXPLAINABLE_MODELS`, which assumes
ProtoMF's post-hoc structure. **Decision:** initial dev runs pass `--skip-explanations`; the intrinsic
read-out (P8) is computed + unit-tested standalone now, and its INTEGRATION into the run_combo explanations
pipeline (plus adding `feature_item_proto` to `EXPLAINABLE_MODELS`) is a separate, tested follow-up **P9
(deferred)**. So P5 registers `feature_item_proto` in `MODEL_CONFIGS` only, for now.

---

## Scope (defaults — confirm or adjust)

**IN:** the dc01 "one idea" mechanism + data plumbing + the intrinsic read-out *computation* + a full
per-part unit-test + integration-test suite (the correctness deliverable).

**OUT (explicitly deferred — dc01 itself defers these; user's later decision):**
- K1–K4 constraint knobs (dc01 §3.8: profile-sharpness, feature-row-norm, ID-routing, profile-distinctness).
- §3.6 evaluation/ablation/cold-split apparatus (held-out item split, tail stratification, LightFM-MF
  baseline, `use_id_feature` True/False sweep, distinct-tuple ceiling experiment).
- Cold-inference ID-column drop; user-side features (S2); image channel (S3); full LLM/naming layer (S4).
- **P9 (deferred):** integrate the P8 intrinsic read-out into `run_combo.py`'s explanations pipeline + add
  `feature_item_proto` to `EXPLAINABLE_MODELS` (initial dev runs use `--skip-explanations`). User's call.

---

## Verified ground truth (drives the design)

> **Re-audited directly against source this round** (not via subagent): `rec_sys.py`, `protomf_dataset.py`,
> `trainer.py`, `tester.py`, `utils.py`, `experiment_helper.py`, `consts.py`, `run_combo.py`. All
> load-bearing claims below confirmed with line cites; two refinements adopted (spec-not-tensor injection;
> `persistent=False` buffer) and one test detail (`use_bias=0` in I2).

- **Score (`rec_sys/rec_sys.py`):** `score = sum(u_embed.unsqueeze(1) * i_embed, dim=-1)` → item extractor
  output last-dim MUST equal user's. Per-extractor losses summed: `rec_loss + item_fe.get_and_reset_loss()
  + user_fe.get_and_reset_loss()`.
- **Item idxs** to the item extractor are `(B, 1+n_neg)`, contiguous `0..n_items-1`; user idxs `(B,)`.
- **`PrototypeEmbedding`** hard-codes `self.embedding_ext = Embedding(...)` at line 226; `forward`
  (lines 299–324) only calls `self.embedding_ext(o_idxs)`. `init_parameters` (295–297) does **not** init
  `embedding_ext`.
- **Factory `prototypes_double_tie`** (lines 69–99) ties base embeddings by weight-tensor assignment
  (93–94); asserts `use_weight_matrix=False`; item side wrapped `invert=True`.
- **Checkpoint:** only `model.module.state_dict()` saved; `tester._build_model` rebuilds via the factory
  from the same `ft_ext_param`, then `load_state_dict`. (See the `persistent=False` refinement below —
  `feature_ids` is reconstructed at build, NOT carried in the checkpoint.) Keys are unprefixed (tester
  builds bare `RecSys`, no DataParallel) → keys match.
- **Determinism:** `reproducible(seed)` runs once before construction; weights drawn at construction.
  ⇒ injecting `FeatureEmbedding` **changes RNG draw order** vs the stock build → a "same-seed → bit-identical"
  equivalence test will NOT hold. Equivalence must be tested by **weight-copy** (see I2).
- **Env:** conda env `protomf` (python 3.9.7, torch 1.9.1+cpu, CPU only). **pytest absent.** Session default
  `python` is base 3.13 → every command must `conda activate protomf` first.
- **Local data:** `hm_3_month` (26,963 items) & `hm_full` (90,690) present with genuine features (5 of
  dc01's 6 fields); **canonical `hm_1_month` (smoke target, K-core 5, 13,651 items) is on LEO5, NOT the
  laptop** → P0 pulls it. ml-1m has NO item features (F=0/ID-only smoke only). `hm_1_month` is already a
  valid `-d` choice in `start.py`/`run_combo.py` — no dataset registration needed.
- **"dev config" = the `dev` PROFILE (corrected after user feedback):** `Master/scripts/run_combo.py`
  defines `PROFILES['dev'] = {num_samples:30, n_epochs:60, patience:7, grace_period:4}` (lines 66–70),
  selected via `--profile dev` (line 415), applied over the full search-space config. The **`hm_1_month`
  baselines ran on exactly this** (replication_report.md). So the standard run = `run_combo.py -m
  feature_item_proto -d hm_1_month --profile dev` — directly comparable to the baselines (`user_item_proto`
  HR@10 0.6606). `run_combo.py` maps models via `MODEL_CONFIGS` and auto-runs the explanations pipeline for
  `EXPLAINABLE_MODELS` (lines 42–52) — **neither includes `feature_item_proto`** → must be registered (P5).
- **Optional quick sanity (the user's parenthetical aside):** ONE trial without the full sweep —
  `run_combo.py … --num-samples 1 --n-epochs 2 --skip-explanations`, or `start.py` directly (`num_samples=1`
  ⇒ `search_alg=None`, experiment_helper.py:185–186). A fast "does it execute" check only; NOT comparable.

---

## Branch & test harness

- **Branch:** `feat/feature-item-proto` off `dev` (per CLAUDE.md branch strategy; `feature_`-prefixed naming).
- **Harness:** plain assert-scripts under `Master/temp/dc_checks/dc01/` (same style as the existing
  `check_claims.py`); each prints `PASS/FAIL` per assertion and exits non-zero on failure. No new dependency.
  (Ports to pytest trivially later if desired.)

---

## Parts (each independently testable; in build order)

**P0 (prerequisite) — obtain the canonical `hm_1_month`** · data provisioning, no repo code edit
`hm_1_month` is dc01's V1 split AND the split the recorded ProtoMF baselines ran on (**K-core 5**, 13,651
items). For feature_item_proto to stay comparable to those baselines (esp. the deferred `user_item_proto`
headline comparison), use the SAME split — **do NOT regenerate**: the splitter defaults to `-c 10`, so a
fresh run yields a different item set, and any raw-data/splitter drift breaks byte-identity.
- **Primary — pull from LEO5** (canonical source): `scp -r leo5:<path>/hm_1_month data/hm_1_month`. Remote
  path TBD — confirm interactively (`! ssh leo5 'find /scratch/c7031336 /home/c703/c7031336 -maxdepth 5
  -type d -name hm_1_month'`); headless SSH here failed on publickey (your key/VPN needed).
- **Build `item_features.csv` aligned to the canonical `item_ids.csv` if the cluster copy lacks it** (the CF
  baselines didn't need it): deterministic merge of raw `data/hm/raw/articles.csv` on `article_id`, sorted by
  `item_id` — exactly `hm_splitter.py` Step 8 (lines 147–172). ~10-line helper; preserves canonical item
  ordering so `feature_ids` (P6) align to the baseline indices.
- **Fallback (SMOKE-ONLY, if cluster unreachable):** `cd data/hm && python hm_splitter.py -lh ./raw
  -s ../hm_1_month -mo 1 -c 5`. ⚠ may NOT be byte-identical to the cluster split → valid for code-path smoke
  ONLY, not for baseline-comparable experiments. **Verify:** `wc -l data/hm_1_month/item_ids.csv` ≈ 13,651;
  all 6 CSVs present.
**Gate scope:** P0 is data only — it gates the end-to-end smoke (I5) but NONE of the unit tests (P1–P4, P8,
I1–I4 use toy/synthetic tensors) nor the F=0 keystone (I2). Code work proceeds without it.

**P1 — `FeatureEmbedding(FeatureExtractor)`** · `feature_extraction/feature_extractors.py`
Sum-of-feature-embeddings item encoder. `__init__(n_objects, feature_ids:(n_obj,F) LongTensor, n_features,
embedding_dim, use_id_feature=True, max_norm=None)`; `register_buffer('feature_ids', …, persistent=False)`
(re-audit: non-persistent → reconstructed at build, never written to the checkpoint, no shape-mismatch risk;
the learned `nn.Embedding` weights are the only saved item params);
`nn.Embedding(n_features + (n_objects if use_id_feature else 0), d, max_norm)`. `forward(o_idxs)` handles
`(B,)` and `(B,1+n_neg)`: gather `feature_ids[o_idxs]`, optionally append ID column `o_idxs+n_features`, sum
over the feature axis → `(...,d)`. `init_parameters` applies `general_weight_init`. `get_and_reset_loss`
inherits base (0.).
**Test `t01`:** shapes for `(B,)` and `(B,1+n_neg)`; output equals manual `emb(feat).sum(-2)`; ID-on vs
ID-off differs by exactly the ID row; F=0+ID returns exactly the ID-row vector; loss==0; `feature_ids` is
NOT in `state_dict()` (persistent=False) while `embedding_layer.weight` IS.

**P2 — `FeatureEmbeddingW`** · `feature_extractors.py`
Projection branch mirroring `EmbeddingW` but wrapping a **shared** `FeatureEmbedding` instance (passed in,
not constructed) + `nn.Linear(d, out_dimension, bias=False)`. `forward = linear(shared(o_idxs))`.
`init_parameters` inits **only the linear** (never the shared embedding — single-owner init).
**Test `t02`:** output shape `(B,1+n_neg,out_dimension)`; embedding identity is shared (`is`); forward ==
`linear(shared(idxs))`; calling `init_parameters` does NOT change the shared embedding's weights.

**P3 — `PrototypeEmbedding` minimal injection** · `feature_extractors.py` · REGRESSION-CRITICAL
Add optional `embedding_ext: FeatureExtractor = None` kwarg. `None` → current behavior **exactly**
(`Embedding(n_objects, embedding_dim, max_norm)`); given → use the passed instance. `forward`, reg losses,
`init_parameters` untouched.
**Test `t03` (byte-identical golden):** FIRST step on the branch, before editing, run a capture script on
current code: `reproducible(seed)`, build `PrototypeEmbedding(n,d,k)`, init, forward on fixed input, save
`state_dict + output` to `t03_golden.pt`. After adding the kwarg, assert the `embedding_ext=None` path
reproduces the golden `state_dict` + output **bit-identically**; assert the injected path uses the passed
object (`m.embedding_ext is injected`).

**P4 — Factory `feature_item_proto` branch** · `feature_extraction/feature_extractor_factories.py`
New `elif ft_type == 'feature_item_proto':` cloned from `prototypes_double_tie`. Read `feature_ids`/
`n_features` from `item_ft_ext_param` (injected by P6). **User side: byte-identical** to `prototypes_double_tie`.
**Item side:** build ONE shared `item_feat_embed = FeatureEmbedding(n_items, feature_ids, n_features, d,
use_id_feature, max_norm)`; `item_proto = PrototypeEmbedding(n_items, d, n_prototypes, …,
embedding_ext=item_feat_embed)`; `item_proj = FeatureEmbeddingW(shared=item_feat_embed,
out_dimension=user_n_prototypes)`. **The tie is the shared module instance** (no weight-tensor assignment).
Wrap `ConcatenateFeatureExtractors(item_proto, item_proj, invert=True)`. Keep `assert not use_weight_matrix`.
**Single-owner init:** the factory calls `item_feat_embed.init_parameters()` exactly once after build
(PrototypeEmbedding doesn't init its ext; FeatureEmbeddingW inits only its linear) — documented as the owner.
**Test `t04`:** returns two `ConcatenateFeatureExtractors`; item-side `item_proto.embedding_ext is
item_proj.shared` (instance identity = the tie); user side unchanged (weight-tied `Embedding`); after
`RecSys.init_parameters()` the shared table is finite and drawn once; output last-dims match
(`(B,Ku+Kt)` vs `(B,1+n_neg,Ku+Kt)`).

**P5 — Config + registration (3 places)** · `confs/hyper_params.py`, `start.py`, `Master/scripts/run_combo.py`
(1) `hyper_params.py`: add `feature_item_proto_hyper_params` = FULL search space (mirror
`proto_double_tie_chose_original_hyper_params`, `ft_type='feature_item_proto'`,
`item_ft_ext_param['use_id_feature']=True`, `feature_fields` list). This is the config `run_combo.py` applies
the `dev` profile over — same shape as the baselines' configs.
(2) `start.py`: add `'feature_item_proto'` to argparse `choices` + the `elif` chain (the direct entry point;
also serves the optional `num_samples=1` quick check).
(3) **`run_combo.py` — the dev-profile orchestrator the baselines used:** add `feature_item_proto` to
`MODEL_CONFIGS` (the new full config). **Do NOT add to `EXPLAINABLE_MODELS` yet** (Open Q E resolved: defer)
— that would trigger `run_explanations_pipeline`, which assumes the post-hoc ProtoMF structure; initial dev
runs pass `--skip-explanations`. Adding to `EXPLAINABLE_MODELS` is P9 (deferred), once the pipeline handles
the feature branch.
**Test `t05`:** import the config; assert `ft_type=='feature_item_proto'`, `use_id_feature` + `feature_fields`
present, user side mirrors `proto_double_tie`; assert `'feature_item_proto'` is in `start.py` choices AND in
`run_combo.py` `MODEL_CONFIGS` (and NOT yet in `EXPLAINABLE_MODELS`, per the defer decision).

**P6 — Data plumbing: build `feature_ids`, inject** · NEW `feature_extraction/feature_ids.py` + 2-line
guarded calls in `rec_sys/trainer.py` & `rec_sys/tester.py`
`build_feature_ids(data_path, fields, use_id_feature)`: read `item_features.csv`; **assert `item_id` covers
`0..n_items-1` exactly** (alignment guard); per field build a deterministic (lexicographically sorted) vocab
with a global field-offset; return `(feature_ids:(n_items,F) LongTensor, n_features)`. **Spec, not tensor, in
the config (re-audit refinement):** `ft_ext_param['item_ft_ext_param']` carries only the lightweight spec —
`feature_fields` (list[str]) + `use_id_feature` (bool) — NEVER the tensor (`run_combo.py:242` dumps config to
`config.json` via `json.dump(default=str)`, and Ray copies the config per trial; a 13,651×F tensor there is
unacceptable). The TENSOR is built inside `_build_model`. **Injection seam:** `trainer._build_model` AND
`tester._build_model`, guarded by `ft_type=='feature_item_proto'`, call
`build_feature_ids(self.*_loader.dataset.data_path, fields, use_id_feature)` (data_path confirmed on the
dataset, `protomf_dataset.py:42`) and pass `feature_ids`/`n_features` to the factory — symmetric train/test,
zero dataset/dataloader/rec_sys changes. Reuse the CSV/`item_id`-join *pattern* from
`data/ml-1m/build_item_features.py` (NOT its encoding — field-offset is new).
**Test `t06`:** on a tiny synthetic `item_features.csv` — shape/dtype; field-offset correctness;
`n_features == Σ per-field vocab sizes`; **alignment guard** (shuffled rows still align by item_id;
missing/extra raises); determinism (build twice → bit-identical); then on REAL `data/hm_1_month/item_features.csv` (after P0)
assert rows ≈ 13,651, every item_id 0..N-1 present, no NaN codes. (Falls back to `hm_3_month`'s file if P0
isn't done yet — the builder is dataset-agnostic.)

**P7 — H&M price-band field** · `data/hm/hm_splitter.py` (or new helper) · synthetic test LOCAL; full run GATED
Join `transactions_train.csv` → per-article mean price → decile (`pd.qcut`, 10 bins) → add `price_band`
to `item_features.csv`, aligned by `item_id`. The `hm_1_month`/`hm_3_month` `item_features.csv` lacks price
(raw join is heavy → full regeneration gated). **price_band is OPTIONAL for the base model — P6 runs on the 5
present fields.**
**Test `t07`:** in-memory synthetic `transactions`+`articles` fixture → decile values 0..9, length==n_items,
aligned by item_id, deterministic. Full H&M regeneration gated.

**P8 — Intrinsic read-out** · NEW `utilities/explanations/feature_readout.py` (+ optional hook in
`explanations_utils.py`)
`per_feature_shares(...) → c_{f,k} = e_f·p_k / (‖q_i‖‖p_k‖)` (exact additive decomposition of `t*_k − 1`;
renders the `+1` shift baseline and the joint-normalization caveat per dc01 honesty notes). Extends
`weight_visualization`. `global_prototype_profile(emb_table, prototypes) → cos(e_f, p_k)` over the vocab;
extends `get_top_k_items` (top feature-values per prototype). Reuse existing rendering hooks.
**Test `t08`:** reuse `check_claims` C1 — assert `t*_k − 1 == shares.sum(over features)` to 1e-10 (sum to
`t*_k − 1`, NOT `t*_k`); `global_prototype_profile` returns `(n_features,K)` cosines in `[-1,1]`.

---

## Integration tests

- **I1 — build + forward/backward** (toy synthetic): factory → `RecSys` → `init_parameters`; logits shape
  `(B,1+n_neg)`; `loss.backward()`; grads on `item_feat_embed`, prototypes, user table; item/user output
  last-dims equal.
- **I2 — KEYSTONE EQUIVALENCE (weight-copy)**: build `feature_item_proto`(`use_id_feature=True`, **F=0
  metadata**) and `user_item_proto` at identical `(n_users,n_items,d,Ku,Kt)`, both with
  `rec_sys_param={'use_bias':0}` (as in `base_param`, so no bias params to reconcile). Because RNG order
  differs (re-audit: even at F=0 the stock build allocates an extra pre-tie item-embedding table the shared
  design doesn't), **copy shared params** (ID-embedding rows ↔ item embedding table, item prototypes, item
  projection, ALL user-side) from one into the other → assert forward scores allclose 1e-6 AND per-parameter
  grads allclose 1e-6. Ties to `check_claims` C5. This proves the wrapper introduced no behavioral change.
- **I3 — gradient from BOTH UI-score halves** (integration C3): backprop each half (`u*·t̂`, `û·t*`)
  separately to a used `item_feat_embed` row → both grads nonzero. Confirms the shared-instance tie routes
  gradient through both the projection and the cosine.
- **I4 — checkpoint round-trip (tester path)**: save `model.module.state_dict()` and assert `feature_ids` is
  ABSENT from the keys (persistent=False) while `…embedding_layer.weight` is present; rebuild via
  `tester._build_model` (feature_ids reconstructed from the CSV spec, P6) → `load_state_dict(strict=True)`
  with NO missing/unexpected keys; forward scores reproduce pre-save exactly.
- **I5 — end-to-end run (CLUSTER, `dev` profile — matches baselines)**: the laptop can't train `hm_1_month`
  → runs on **LEO5**. Flow: `/commit` + push → `git pull` on the cluster → confirm canonical `hm_1_month`
  (P0) → SLURM job (`leo5-submit`) running `run_combo.py -m feature_item_proto -d hm_1_month --profile dev`
  (30 trials / 60 epochs / patience 7 — identical budget to the recorded baselines, so the result is
  directly comparable to `user_item_proto` HR@10 0.6606). Assert: >0 trials complete, `test_metrics.json`
  written, finite metrics, no NaN. **Quick pre-check (optional, cheap):** run once with `--num-samples 1
  --n-epochs 2 --skip-explanations` to confirm it executes before the full dev run. LOCAL proxy for the
  'assembles + trains one step' part is the toy **I1**.
- **I6 — REGRESSION (split local/cluster)**: **(a) LOCAL toy build-check** `i06a` — instantiate
  `user_item_proto` (`prototypes_double_tie`) and `mf` via the factory, run one forward/backward on a
  synthetic batch; assert finite, no errors. Proves P3's `embedding_ext=None` default path is intact (t03
  already gives unit byte-identity) — no real dataset, runs on the laptop. **(b) CLUSTER** — a quick
  `run_combo.py -m user_item_proto -d hm_1_month --num-samples 1 --n-epochs 2 --skip-explanations` confirms
  an existing model still runs unchanged through the same path.

---

## Recommended build order

P1 → P2 → P3 (leaf modules; P3 golden de-risks the whole change) → P6 builder (+t06) →
P4 (needs P1/P2/P3 + feature_ids) → P5 (incl. `run_combo.py` registration) → **I1 → I2 (keystone) → I3 → I4**
→ P8 (+t08 vs C1) → I6a (local build-check) → **[ALL LOCAL correctness proofs done here]** → `/commit` + push
→ on LEO5: `git pull` → **P0 (pull/confirm canonical hm_1_month) → quick 1-trial execute-check → I5
`dev`-profile run + I6b regression on the cluster** → P7 (price-band last; gated).
Rationale: every correctness proof (toy unit + integration, incl. the F=0 keystone) finishes LOCALLY first;
only the real-data `dev` run crosses to the cluster, once the code is proven.

## Resolved design decisions (one each)
- **Equivalence test:** weight-copy, NOT same-seed (RNG order perturbed by the injected module).
- **Alignment:** assert `item_id` covers `0..n_items-1` exactly; build by join/sort on `item_id`; reject
  missing/extra/dup.
- **Plumbing seam:** `trainer._build_model` + `tester._build_model`, guarded by `ft_type`. Inject only the
  `feature_fields`+`use_id_feature` SPEC into `ft_ext_param`; BUILD the `feature_ids` tensor in `_build_model`
  from `dataset.data_path` (never serialize the tensor into the Ray/JSON config). Zero changes to
  dataset/dataloader/rec_sys.
- **`feature_ids` buffer:** `persistent=False` — reconstructed deterministically at build, never in the
  checkpoint (no shape-mismatch risk; learned embedding weights round-trip normally).

---

## Verification commands (always `conda activate protomf` first — base python is 3.13, wrong env)

```
conda activate protomf            # cwd: repo root
# algebraic baseline (already passing):
python Master/temp/dc_checks/dc01/check_claims.py
# unit parts:
python Master/temp/dc_checks/dc01/t01_feature_embedding.py    # … t02 … t08
# integration:
python Master/temp/dc_checks/dc01/i01_build_forward_backward.py   # … i02 (keystone) … i04
# local build-check (toy; existing models still assemble after P3):
python Master/temp/dc_checks/dc01/i06a_regression_buildcheck.py
#
# === CLUSTER (LEO5) — end-to-end; laptop CANNOT train hm_1_month ===
# 1) /commit + push   2) on LEO5: git pull   3) ensure canonical hm_1_month present (P0 scp)
# 4) optional quick execute-check (one trial, no explanations):
#    python Master/scripts/run_combo.py -m feature_item_proto -d hm_1_month --num-samples 1 --n-epochs 2 --skip-explanations
# 5) standard dev run (matches baselines; submit via leo5-submit as a SLURM job):
#    python Master/scripts/run_combo.py -m feature_item_proto -d hm_1_month --profile dev
# 6) regression (existing model, quick):
#    python Master/scripts/run_combo.py -m user_item_proto -d hm_1_month --num-samples 1 --n-epochs 2 --skip-explanations
```

## Files touched
- `feature_extraction/feature_extractors.py` — P1,P2 new classes; P3 one kwarg
- `feature_extraction/feature_extractor_factories.py` — P4 new branch
- `feature_extraction/feature_ids.py` — NEW (P6 builder)
- `rec_sys/trainer.py`, `rec_sys/tester.py` — P6 guarded inject in `_build_model`
- `confs/hyper_params.py`, `start.py` — P5 config + CLI
- `Master/scripts/run_combo.py` — P5 register `feature_item_proto` in `MODEL_CONFIGS` (P9: `EXPLAINABLE_MODELS`)
- `utilities/explanations/feature_readout.py` — NEW (P8); + (P9, if integrating) a hook into
  `utilities/explanations/pipeline.py` so run_combo's auto-explanations handle feature_item_proto
- `data/hm/hm_splitter.py` — P7 price_band (gated)
- `Master/temp/dc_checks/dc01/t0*.py, i0*.py` — all tests
- `Master/docs/modifications_log.md` — log each upstream-file edit (per CLAUDE.md)

## Local-now vs gated
- **Local now (laptop CPU):** P1–P6, P8 + I1–I4 + I6a — i.e. **ALL the correctness proofs** (toy/synthetic;
  no real dataset, no GPU). This is the bulk of the "ensure correctness once integrated" deliverable.
- **Cluster (LEO5):** P0 (pull/confirm canonical `hm_1_month`) + I5 end-to-end at the **`dev` profile**
  (30/60/7/4, matches baselines) via `run_combo.py --profile dev` + I6b regression, through
  `/commit`→push→`git pull`→SLURM (`leo5-submit`). The laptop cannot train `hm_1_month`.
- **Gated:** P7 FULL price-band regeneration (heavy raw join; synthetic test t07 is local; price_band is
  optional for the base model — P6 runs on the 5 present fields).
