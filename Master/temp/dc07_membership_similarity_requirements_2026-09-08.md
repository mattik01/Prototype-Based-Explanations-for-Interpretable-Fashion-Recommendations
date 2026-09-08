# dc07 — Membership similarity for the prototype layer: requirements & handoff

**Written:** 2026-09-08, session with Claude (Fable 5.1). **Status:** requirements + experiment
design, no code written. **Path chosen by Matteo:** lightweight — build from this document, no
mandatory /design-candidate cycle; a prior-art probe is still expected inside the build session
(§10). **Tier:** everything here produces working evidence (dev tier), not thesis numbers, per
the two-tier numbers policy. **Branch:** `feat/scrutiny`.

Read first: this file; then `Master/protocol_vault/2026-08-17_1128_shifted-cosine-baseline-asymmetry-popularity-channel.md`,
`2026-07-15_1112_prototype-collapse-is-a-self-built-popularity-bias.md`,
`2026-08-17_1937_base-settings-decision-shifted-bias-off-cosbias2x2-results.md`,
`2026-08-17_1407_coverage-regularizer-uncollapses-hm-prototype-space.md`; then the five vault
entries of 2026-09-08 (fI′ rejected; presentation literature; candidate-prime shapes;
interaction-attribute taxonomy; query context at the tie; fU′ scope).

---

## 0. The question in one paragraph

ProtoMF represents a user (or item) as its shifted-cosine similarity to K prototypes,
u*_l = 1 + cos(u, p_l), and scores an item as Σ_l t_{i,l} u*_l. Every measured defect of the
lineage's composed branches that is NOT a composition defect lives in how this layer reads its
input: prototype collapse onto one direction, the shared input direction the cosine cannot
ignore, the norm handover between metadata and ID rows, the inexpressibility of evidence mass,
the +1 popularity channel, and the bottleneck relative to a plain dot product on the same
vectors. Candidate: keep the host, the prototypes, the score form and the regularisers, and
change only the similarity — softmax over dot products with a temperature (a membership
distribution over prototypes), with a sigmoid-of-dot variant as the second value of the same
switch. Question: does the bottleneck move at the HOST level (stage 1); if so, does it move for
the composed branches fI / fU (stage 2)?

## 1. Why this candidate, with evidence

**1.1 Where the loss is (measured, dev tier, seed 38210573).** The same composed user vector
under a plain dot product (`lightfm_hist`) beats fU by +0.079 HR@10 on hm_1_month and +0.044 on
ml-1m (`Master/docs/scrutiny/sc05_chapter_draft.md:144-166`). `lightfm_tags_ids` beats fI by
+0.010 on hm and by +0.046 on ml-1m (dev tier, `sc01_chapter_draft.md:129-148`); at 100 trials on
ml-1m the gap fI → lightfm_tags_ids is 0.5628 → 0.6155 (cluster results, 2026-09-08). Composition
repairs did not move this: fI′ centring cost −0.05 on the ID arm at 100 trials
(vault 2026-09-08_1509).

**1.2 Why cosine was chosen (paper, §3.1).** One stated reason: shifted cosine keeps u* in
[0,2] so per-prototype score terms are non-negative and the decomposition has no double-negative
readings. Two implicit reasons: scale-freeness (prototype norm is a pure gauge; similarity reads
as type membership), and the Li-et-al. inclusion regularisers push up a MAX similarity, which
needs a bounded similarity to be well-posed. Cosine is ProtoMF's departure from the prototype
networks it descends from (ProtoPNet / Li et al. use squared L2).

**1.3 What the shift and the cosine cost (measured).**
- The +1 on the USER-prototype side is a per-item popularity channel B(t)=1ᵀt carrying ~80 % of a
  typical score; prototypes are recruited into it as risk-free amplifiers because 1+cos ≥ 0
  (vault 2026-08-17_1128). On the item-prototype side the +1 is a dead gauge.
- cosbias2×2 (single-config retrains, SLURM 7426702–7426722): removing the shift on the I side is
  a placebo (item_proto 0.4938 → 0.4947 hm; 0.5711 → 0.5709 ml). On the U side it costs 1–3 pts
  and does NOT free the prototypes (hm 5 → 7 effective directions; ml 15 → 11). An explicit item
  bias recovers only part of it. Decision then: (shifted, bias off) stays the lineage standard;
  the 2×2 flagged that a fair comparison needs full hyperopt per cell, not retrains
  (vault 2026-08-17_1937).
- Collapse mechanism: data pressure × regulariser config; coverage term off in the hm winner
  (sim_batch 0.0018 vs 5.71 in the healthy 3-month run) (vault 2026-08-17_1407, 2026-07-15_1112).

**1.4 Precedent in the fleet.** `acf` (anchor-based CF, `feature_extractors.py:534-606`) IS a
softmax membership over anchors with entropy regularisers (temperature fixed at 1, representation
= reconstruction Σ coeff·anchor rather than the membership vector). It beats single-sided ProtoMF:
ml-1m 0.6006 vs user_proto 0.5933 / item_proto 0.5723 (prod); hm_1_month 0.5764 vs user_proto
0.5725 / item_proto 0.4938 (dev) (`Master/docs/replication_report.md:32-51`).

**1.5 Taxonomy slot.** `Master/docs/interpretability_knobs_metrics_taxonomy.md:28` row 10
("cosine-readout semantic reliability … `cosine_type` variants; no differentiable surrogate for
reliability itself", priority R1). A membership similarity is that missing surrogate — this is
the thesis home for the family (block two, knobs chapter). The frozen reference table stays
cosine; dc07 arms are a new family, never a replacement of host rows.

## 2. Candidate definition

Fixed: host architectures (`user_proto`, `item_proto`; later fI/fU), prototype count and
dimension search spaces, both regularisers and their weights' search spaces, loss (sampled
softmax), negative sampling, `use_bias=0`, dev profile, seed 38210573, fleet shape (§7.3).

Variable (one at a time): `cosine_type` ∈ {`softmax`, `sigmoid`} plus ONE new hyperparameter,
the temperature τ (softmax) / scale (sigmoid), in the search space.

**Softmax membership.** Given object vector q ∈ R^d and prototypes P ∈ R^{K×d}:
  z_l = ⟨q, p_l⟩ / τ ;  m_l = exp(z_l) / Σ_k exp(z_k) ;  representation = m ∈ Δ^{K−1}.
Score unchanged in form: Σ_l t_{i,l} m_l. Properties: m_l ∈ (0,1), Σ m_l = 1, no +1 baseline,
norm-SENSITIVE (a small ‖q‖ → flat memberships = low confidence), translation in q adds a
per-prototype offset to all users' logits (a shared prior over prototypes, not an angle collapse).

**Sigmoid membership (second value, not run in stage 1).**
  m_l = σ(⟨q, p_l⟩ / τ + b_l), b ∈ R^K learned (init 0). Independent per prototype, m_l ∈ (0,1),
no competition, no normalisation; logit exactly linear in the word rows.

**Degenerate limits that MUST be guarded (see §9):** softmax at τ → ∞ gives m_l = 1/K for
everyone, score = (1/K) Σ_l t_{i,l} — a pure per-item scalar, i.e. the popularity channel
rebuilt at the layer level. τ → 0 gives hard assignment (one prototype per user). Sigmoid with
b_l → +∞ gives m_l → 1 for all → same per-item scalar. The τ search range must be bounded and
the membership-entropy read-out (§8) is mandatory on every arm.

## 3. The math, against the problem list

Problems outlined in the 2026-09-08 discussion (numbering used below): P1 collapse; P2 shared
input direction; P3 norm handover; P4 evidence mass inexpressible; P5 ID-row absorption;
P6 layer bottleneck vs dot; P7 count bottleneck (pairs/relations); P8 +1 popularity channel;
P9 coverage regulariser switched off by hyperopt; P10 naming completeness. P5, P7, P9, P10 are
NOT similarity problems and are untouched by dc07.

| | shifted cosine (host) | softmax membership | sigmoid membership |
|---|---|---|---|
| positive & bounded | [0,2] | (0,1), sums to 1 | (0,1) |
| scale of q | ignored (P4 unsolved) | confidence (P4 addressed) | confidence |
| shared mean μ in q (P2) | collapses angles | per-prototype logit offset = shared prior, readable | same |
| +1 channel (P8) | present (U side alive) | absent; can be REBUILT via τ→∞ (guard) | absent; rebuilt via b→∞ (guard) |
| duplicate prototypes (P1) | free | free (split membership); needs coverage/entropy pressure | free |
| per-prototype score decomposition (paper §5.2) | exact | exact | exact |
| exact additive word/purchase attribution (lineage contract) | exact up to shared 1/‖q‖ | exact on LOG-ODDS: log m_l − log m_k = ⟨q, p_l − p_k⟩/τ = Σ_f c_f ⟨e_f, p_l − p_k⟩/τ; NOT on m itself | exact on the logit |
| intrinsic naming cos(e_f, p_l) | yes | yes (rows vs prototypes, unchanged) | yes |
| top-k ordering of prototypes for one object | monotone in cos | monotone in DOT, not in cos (norms enter) | same |
| regularisers 'max' | well-posed | saturates at −1, scale-locked (only sharpening raises it) | meaningful |
| regularisers 'soft'/'incl' | well-posed | DOUBLE softmax → near-dead (§4 D1) | double normalisation |

Expected mechanism on P6: with memberships the layer is no longer an angle bottleneck; ACF is the
existence proof that a membership layer reaches +0.03 over single-sided ProtoMF on ml-1m. Whether
that transfers to the prototype-score form is the stage-1 question.

## 4. Code touchpoints (verified 2026-09-08, line numbers as of commit 9619f44)

**4.1 `feature_extraction/feature_extractors.py` — `PrototypeEmbedding` (:608-767)**
- `cosine_type` dispatch `:665-673` builds `self.cosine_sim_func(x, y)` as an ELEMENTWISE
  per-prototype lambda over `(o_embed.unsqueeze(-2), self.prototypes)` → `[..., K]`. A softmax
  needs the whole K-vector; implement the new modes as a separate branch in `forward`
  (`:732-762`) or generalise `cosine_sim_func` to return the full `[..., K]` row (it already does
  shape-wise; the softmax just normalises along the last dim). Keep the three cosine modes
  bit-identical.
- New `__init__` params: `temperature: float = 1.0` (and for sigmoid a learned `nn.Parameter`
  bias of shape (K,), init 0). **Construct them AFTER the existing prototype/embedding
  construction** — `Master/temp/dc_checks/dc01/t03_prototype_injection.py` pins the RNG draw
  order of the stock module against a golden file.
- Regularisers `:675-690`, helpers `:706-721`: 'max' = −mean row/col max; 'soft' =
  entropy of `Softmax(sim_mtx)`; 'incl' = −entropy of column loads of `Softmax(sim_mtx)`.
  Under a membership similarity, 'soft'/'incl' would softmax an already-normalised vector.
  See decision D1.
- `AttributePrototypeEmbedding` (:848-996) inherits everything via `super().__init__` (:898);
  no change needed, out of scope for dc07 runs.
- `get_and_reset_loss` `:763-766` unchanged.

**4.2 Precedent code:** `AnchorBasedCollaborativeFiltering` `:573-605` — softmax memberships,
`exc = −Σ m log m` (sum, not mean), `inc = −entropy(column loads)`, loss `−δ_inc·inc + δ_exc·exc`.
Reuse the formulas if D1 chooses membership-native entropy terms; note the `.sum()` vs `.mean()`
scaling difference.

**4.3 Factory `feature_extraction/feature_extractor_factories.py`** — `cosine_type` read with a
`'shifted'` default at `:171` (fI), `:224` (fU), `:288` (attr), `:425/:429` (generic `prototypes`,
also reached by `prototypes_double_tie`, `user_proto`, `item_proto`). A new `temperature` key must
be plumbed at all four sites (same default idiom).

**4.4 Configs `confs/hyper_params.py`** — `cosine_type` is a FIXED literal `'shifted'` in all
14 occurrences (`:65, :87, :122, :132, :164, :186, :196, :224, :306, :354, :459, :488, :521, :549`),
never a `tune.*` dimension. New configs by deepcopy (pattern `:246-283`):
`user_proto_sm_hyper_params`, `item_proto_sm_hyper_params` = host dicts with
`cosine_type='softmax'` and `temperature = tune.loguniform(τ_min, τ_max)` (D2); later
`feature_item_proto_sm`, `feature_item_proto_noid_sm`, `feature_user_proto_sm`,
`feature_user_proto_noid_sm`. Sigmoid twins `_sg` built but not queued.

**4.5 Registries (both, always together — pinned by `dc01/t05_config_registration.py`,
`dc02/t05_config_registration.py`):** `start.py` import block `:4-14`, `--model` choices `:24-30`,
if/elif chain `:50-99`; `Master/scripts/run_combo.py` imports `:31-54`, `MODEL_CONFIGS` `:59-84`.
Auto-explanation headline policy `run_combo.py:99-115` — dc07 arms must NOT be headline models
in stage 1 (explanations would hit the gate in 4.6); run with `--skip-explanations`.

**4.6 Explanation stack — the affine assumption (hard gate + silent sites)**
- `utilities/explanations/breakdown.py:89-108`: `_COSINE_AFFINE = {shifted:(1,1), standard:(1,0),
  shifted_and_div:(.5,.5)}`; `cosine_affine()` RAISES on any other `cosine_type`. Correct
  failure mode; keep it refusing in stage 1. Consumers `:302-305, :395-398/:411, :476-479,
  :616-620`; zoom/share math `:313-320, :332-335` assumes act = a·cos + c and reconstructs the
  activation from per-feature shares — no softmax analogue on m; the log-odds analogue (§3)
  needs a new renderer path (stage 2 deliverable, D3).
- `utilities/explanations/feature_readout.py:42-53` `activation_from_shares(offset=1.0, scale=1.0)`
  — affine reconstruction with shifted defaults.
- `utilities/explanations/attr_readout.py:46-48, :57-61, :63-81` — HARDCODED `1 +`, no guard →
  silently wrong under a new mode. Add a guard (raise unless shifted) in stage 0.
- `utilities/explanations/history_readout.py:6-17` docstring assumes 1+cos.
- `utilities/explanations/accessor/base.py:73-83` `_shifted_cosine_sim` used for RANKING in
  post-hoc naming; ranking by cos ≠ ranking by dot under softmax — document, do not change.
- `utilities/explanations/explainers/weight_viz.py:64-70` uses `cosine_affine` → raises (fine).
- `utilities/explanations_utils.py:152-157` y-floor logic already generalised (fine).
- `Master/scripts/id_row_probe.py:248` `A = 1 + COS` HARDCODED, no `cosine_type` read → add a
  guard in stage 0 (refuse or branch).
- `Master/scripts/sc8_geometry_readout.py`, `sc8_fu_geometry_readout.py`: no shifted assumption;
  work on raw cosines/angles; usable as-is (but see §8 for what B(t) means under softmax).

**4.7 Checks that will break or must be added (`Master/temp/dc_checks/`)**
- `cosbias2x2/t01_breakdown_cosine_offset.py` case E (:269-280) pins that `cosine_affine` raises
  on unknown types — stays green as long as breakdown keeps refusing `softmax`/`sigmoid`.
- `dc01/t03_prototype_injection.py`, `dc01/t03_capture_golden.py` — RNG-order golden; keep
  construction order.
- `dc02/t02_attr_prototype_embedding.py` (:30, :50-52) — pins shifted/standard manual formulas.
- `dc01/t08_feature_readout.py`, `dc02/t08_attr_readout.py` — pin affine reconstruction; must
  stay green for cosine modes; the softmax log-odds decomposition gets its OWN check.
- New `Master/temp/dc_checks/dc07/`: t01 memberships sum to 1, positive, permutation-equivariant
  in prototypes; t02 log-odds exact decomposition over word rows for fI-shaped and fU-shaped
  inputs (compare to brute force); t03 τ-limit behaviour (τ large → uniform; score → per-item
  mean of t) as a documented property, not a keystone; t04 regulariser contract per D1;
  t05 config registration in both registries; t06 the three cosine modes bit-identical to
  golden after the change; t07 sigmoid: logit linear, bias init 0, b→∞ limit documented.

## 5. Design decisions for the receiving session (recommendation in bold)

- **D1 regulariser contract.** Host configs fix `reg_proto_type='max'`, `reg_batch_type='max'`
  (all winning configs). **Keep 'max' for stage 1 (single variable).** Note it saturates at −1
  and is scale-locked under softmax (only sharpening raises it), so its effect becomes
  "sharpen memberships" — record this. For 'soft'/'incl' under a membership mode, make the
  regulariser consume the membership directly (no second softmax); implement but do not search
  over it in stage 1. ACF-style exclusiveness/inclusiveness is the natural stage-2b axis if
  P1 collapse persists.
- **D2 temperature.** **Hyperparameter, `tune.loguniform(0.05, 2.0)`, fixed per trial**, not
  learned (learned τ makes the ID-row norm and τ jointly redundant and drifts toward the τ→∞
  degenerate). Upper bound deliberately below the uniform regime; verify with t03 what τ gives
  membership entropy > 0.9·log K for typical ‖q‖ at init and set τ_max below that.
  Sigmoid: same scale search; bias b learned, init 0, weight-decayed with the prototypes.
- **D3 explanation stack.** Stage 1: `--skip-explanations`, breakdown refuses, naming route
  (rows vs prototypes) unaffected. Stage 2 (only on a pass): implement the log-odds
  decomposition renderer ("why prototype l over prototype k" = Σ_f c_f⟨e_f, p_l−p_k⟩/τ), the
  membership entropy line ("how sure we are who you are"), and drop the baseline/personalized
  split (no +1). Do NOT extend `_COSINE_AFFINE` with a fake affine entry.
- **D4 model naming.** Suffix `_sm` (softmax) / `_sg` (sigmoid) on the host name; results dirs
  follow `run_combo.results_folder_name` automatically.
- **D5 keystone.** There is no τ that recovers shifted cosine (norms enter), so no
  bit-identity keystone exists. Pin instead: cosine modes untouched (golden), memberships on the
  simplex, log-odds exactness, and — as the sanity anchor — `acf` on ml-1m (0.6006) as the
  membership-family reference the softmax host should land near or above.
- **D6 bias arm.** Left open by Matteo. If the softmax host lands below the cosine host by more
  than the yardstick, the receiving session decides whether to run a `use_bias=1` arm.

## 6. Experiment plan

**Stage 0 — build (laptop + LEO5 login node, ~1 day).** Implement §4.1–4.5 and the guards in
4.6; write `dc_checks/dc07/t01–t07`; run the existing dc01/dc02/dc05/cosbias2x2 checks (all
must stay green); one 3-epoch smoke of `user_proto_sm` on hm_1_month on a login-node A30
(smoke-one-before-batch rule; quarantine under `/scratch/c7031336/smoke_runs/dc07/`,
`WANDB_MODE=offline`, `--skip-explanations`). Commit + push before any cluster pull (cluster
sync protocol; ask Matteo before committing).

**Stage 1 — host, 4 runs, dev tier.** `user_proto_sm` and `item_proto_sm` × {hm_1_month, ml-1m},
30 trials / 60 epochs / patience 7 / grace 4, seed 38210573, fleet shape §7.3, one `/leo5-submit`
per run (hard gate). Comparators (dev tier unless marked):

| dataset | cosine host (frozen) | acf | mf | lightfm_tags / _ids | UI host |
|---|---|---|---|---|---|
| hm_1_month | user_proto 0.5725/0.3322; item_proto 0.4938/0.2631 | 0.5764/0.3307 | 0.3652/0.2233 | 0.4582/0.2709 ; 0.5087/0.3063 | 0.6587/0.4164 |
| ml-1m (Wave D dev) | user_proto 0.5668/0.3230; item_proto 0.5711/0.3187 | 0.6006/0.3364 (prod) | 0.5022/0.2793 | 0.5565/0.3197 ; 0.5612/0.3261 | 0.6246/0.3573 (prod) |

Sources: `replication_report.md:32-51, :69-79`; `sc01_chapter_draft.md:129-148`;
`sc05_chapter_draft.md:144-166`. Noise yardstick (hm, one draw): |f0 − item_proto| = 0.0026
HR@10 / 0.0011 N@10 (`sc01_chapter_draft.md:153-156`); no ml-1m yardstick exists — quote the
hm value as scale only.

Read-outs (all pre-registered, §8): test HR@10 / NDCG@10; effective rank + prototype pairwise
cosine clone blocks; membership entropy distribution and its relation to history length /
item popularity; top-prototype score share per recommendation; winning τ and its position in
the range (if at τ_max → degenerate regime, flag).

**Stage-1 gate (decision rule, written before the runs):**
- A. Accuracy within the yardstick of the cosine host AND geometry still collapsed → the layer
  is the bottleneck independent of the similarity. Record; stop dc07 as an experiment family;
  the result goes to the knobs chapter as a controlled negative.
- B. Accuracy below the host by > yardstick → the +1 channel was load-bearing; record how
  much; D6 decides on a bias arm; stop unless the bias arm recovers it.
- C. Accuracy ≥ host with visibly healthier geometry (effective rank up, clone blocks gone,
  membership entropy varying with evidence) → stage 2.

**Stage 2 — composed branches, 8 runs, dev tier (only on C).** `feature_item_proto_sm`,
`feature_item_proto_noid_sm`, `feature_user_proto_sm`, `feature_user_proto_noid_sm` ×
{hm_1_month, ml-1m}. fU arms inherit leave-one-out pooling (commit 89a1818; default on).
Comparators: own cosine rows (fI hm 0.4984/0.2757, noid 0.4229/0.2317; fI ml 0.5156/0.2836,
noid 0.4960/0.2735; fU hm 0.5832/0.3496, noid 0.5833/0.3507; fU ml 0.5491/0.3090, noid
0.5424/0.3016) and the ceilings `lightfm_tags(_ids)` / `lightfm_hist(_ids)` (hm 0.6511/0.6464;
ml 0.6107/0.6374). NOTE: the hm fU/lightfm_hist cosine rows were trained WITHOUT leave-one-out;
if the fU 100-trial/LOO wave has landed by then, use its rows. The number that matters: the gap
to lightfm. Deliverable also includes the D3 renderer and dc07 explanation artifacts for one
user/item per arm, scrutinised before landing (standing directive on graphical outputs).

**Stage 3 — matched budget (only if stage 2 moves).** 100 trials on ml-1m for the arms that
moved (`_n100` results-dir suffix is automatic), compared against the existing 100-trial rows:
item_proto 0.5723/0.3191 (prod replication), fI 0.5628/0.3204, fI-noid 0.5156/0.2822,
lightfm_tags_ids 0.6155/0.3590, lightfm_tags 0.5905/0.3438, user_item_proto 0.6246/0.3573.

## 7. Operations

**7.1 Budget.** Stage 1: 4 submissions, hm ≈ 3–4 h each at the fleet shape (anchors:
user_proto hm 3h35m, item_proto hm 2h54m, `replication_report.md:233-239`); ml-1m dev ≈ 0.25×
prod (item_proto prod 9h46m). Stage 2: 8 submissions. Runs alongside the fU 100-trial wave;
no shared code path except `PrototypeEmbedding` (the fU arms are cosine).
**7.2 Placement.** Smokes and post-hoc instruments on the login-node A30s; every compared row
via `/leo5-submit`, std partition.
**7.3 Fleet shape** (`.claude/commands/leo5-submit.md` FLEET MODE; `slurm/run_combo.slurm`):
`--profile dev`, concurrency 5 (`--gpu-per-trial 0.2`), `--num-workers 1` (never 0 — flips the
eval-negative regime, F-S0-02), `--mem 40G` (hm peaks 26–27 GB at conc 5), A30 untyped gres,
tag `dc07-stage1`, seed 38210573, `--skip-explanations`. Results: `/scratch/c7031336/protomf_results/<model>_<dataset>_s38210573/`.
**7.4 Sync.** Laptop is source of truth; commit + push, `git pull` on LEO5; never edit on the
cluster. Modifications log: `feature_extractors.py`, `feature_extractor_factories.py`,
`hyper_params.py`, `start.py`, `breakdown.py`/`attr_readout.py` guards are upstream-or-tracked
files → entries in `Master/docs/modifications_log.md`.

## 8. Instruments and read-outs

Existing, run as-is on every dc07 checkpoint (login node):
- `Master/scripts/sc8_geometry_readout.py --results-dir … --dataset …` (item-branch prototypes:
  activation spectrum, entropy effective rank, prototype pairwise latent cosine = clone detector,
  ID-row share) → `<results_dir>/sc8_geometry/readout.{json,md}`.
- `Master/scripts/sc8_fu_geometry_readout.py` (user-branch: history-band angles, ID-energy
  share, coverage, B(t) channel, profile entropy) → `sc8_fu_geometry/`. Under softmax the B(t)
  block measures a quantity that no longer exists (no +1); read the "top-prototype share" and
  "mean membership vector" instead — small extension needed.
- Duplicate-name count via the explanation naming route is unavailable in stage 1 (renderer
  refuses); use sc8 pairwise latent cosine > 0.999 as the clone measure (the vault's definition).
New (small, login-node):
- Membership entropy H(m_u) per user and per item; distribution; Spearman vs history length
  (users) and vs train popularity (items); fraction of objects with H > 0.9·log K (uniform
  regime) and < 0.1·log K (hard regime). This is the direct test of P4 and the τ-degeneracy guard.
- Top-prototype score share Σ_l t_{i,l} m_l restricted to argmax_l m_l, averaged over top-10
  recommendations — the rediscovered-popularity detector (a dominant prototype everyone belongs
  to reproduces B(t)).
- Winning-τ report per run (from `config.json`), flagged if within 10 % of a range edge.

## 9. Risks and the devil's advocate

- **Popularity rediscovery.** Softmax has its own popularity channel at high τ (uniform
  memberships ⇒ per-item scalar). Hyperopt selects on HR@10 and a popularity ranker wins on hm;
  expect pressure toward τ_max. Guards: bounded τ (D2), entropy read-out, top-prototype share.
  If the winner sits at τ_max with near-uniform memberships, the arm is a popularity model and
  the comparison is void — report as such.
- **Collapse is not cured by the similarity.** Duplicate prototypes still cost nothing; the
  coverage regulariser stays the only push-back and hyperopt turned it off for cosine
  (vault 2026-08-17_1407). Stage 2b candidate axis: ACF-style entropy terms (D1) or a raised
  `sim_batch_weight` floor — a second variable, only after stage 1.
- **Loss of the paper's [0,2] reading and of the affine share zoom.** Positivity is kept;
  the per-feature additive zoom on m is lost and replaced by log-odds attribution. The thesis
  must present this as a change of explanation unit, not a free upgrade.
- **Ranking-by-dot vs ranking-by-cosine** in post-hoc naming (`accessor/base.py:73-83`): a
  prototype's top-k members can differ between the two orderings; disclose, do not "fix".
- **Unfair comparison tier.** Dev-tier host rows are frozen and never re-run; the dc07 arms get
  a fresh hyperopt with one extra dimension (τ). Same asymmetry every candidate had; declare it.
- **ID-row norm and τ are jointly redundant** in the ids arms (scaling q ≡ scaling 1/τ):
  keep τ fixed per trial (D2) so weight decay on the rows decides the effective sharpness.
- **Scope creep into part two.** dc07 is a knob on the existing layer, not the topic-model
  host (angle G F5). If stage 1 passes, the F5 idea becomes "dc07 plus Dirichlet-style
  shrinkage of memberships" — record as future work, do not build in this cycle.

## 10. Prior-art probe expected inside the build session (lightweight path)

Verify and cite (paper-search MCP was down on 2026-09-08; the following are from memory):
Barkan et al. ACF (CIKM 2021, local literature folder, already read for S0); Li et al. 2018
prototype networks and Chen et al. ProtoPNet 2019 (L2 similarity, log-ratio activation,
prototype projection); Hofmann pLSA / Blei LDA mixed-membership recommendation; temperature-scaled
softmax prototypes in metric learning (ProtoNets, Snell et al. 2017, softmax over negative
distances); attention-as-membership critiques (Jain & Wallace 2019; Wiegreffe & Pinter 2019)
for the reading of memberships as explanations; Steck et al. on cosine-similarity reliability
(taxonomy row 10). Record in `Master/docs/design_candidates/candidate_index.md` as dc07 with a
fingerprint waiver (read-out/grounding axis on an existing host; same treatment as dc06).

## 11. Deliverables checklist for the receiving session

- [ ] Implementation per §4 + guards (4.6) + `dc_checks/dc07/t01–t07` green; existing checks green.
- [ ] Modifications log entries; configs `user_proto_sm`, `item_proto_sm` (+ `_sg` twins, fI/fU
      `_sm` twins registered but not queued).
- [ ] Login-node smoke, exit 0, membership banner in the build print.
- [ ] Stage-1 run plan with the four `/leo5-submit` invocations and the pre-registered read-outs
      and decision rule copied verbatim from §6.
- [ ] Post-run: sc8 read-outs + the three new instruments per arm; a short memo under
      `Master/docs/scrutiny/sc07_membership_similarity.md` (or `Master/temp/` until the cycle is
      registered) applying the gate.
- [ ] Protocol entries at: build landed; stage-1 results; gate decision.
- [ ] Do not touch: the frozen reference table, cosine host configs, `_COSINE_AFFINE`.

## 12. Open questions deliberately left to the receiving session
τ range edges after the t03 measurement; whether `--skip-explanations` stays for stage 2 until
the D3 renderer exists; the D6 bias arm; whether the fU stage-2 rows wait for the LOO/100-trial
fU wave to land first (recommended: yes, for a like-for-like control).
