# fI-ProtoMF (dc01) — Results Memo

> **SC.9 chapter draft** (Scrutiny Protocol v1.2), 2026-08-20. Results-memo
> depth, not thesis prose (protocol SC.9.2).
>
> **Number status (two-tier policy, 2026-07-14):** every number here is a
> **working number** — dev-profile hyperopt (30 trials / 60 epochs / patience
> 7), **single seed 38210573**, on the `feat/scrutiny` lineage. Working
> numbers are comparison-valid across the fleet (equal treatment) but are
> **explicitly not thesis numbers**; thesis-grade tier (paper-budget × 3
> seeds) is queued after model freeze. Instrument readouts (D3/D4, geometry)
> are additionally **scrutiny working basis only** (F-DC01-06 restriction):
> thesis treatment gets dedicated experiments designed at writing time.
> All verdicts **dev-profile-provisional**; requirement-anchored absolute
> phrasing throughout (no cross-candidate ranking, GR5).

## 1. Mechanism

fI-ProtoMF is I-ProtoMF (Melchiorre et al., RecSys '22, §3.2) with exactly
one change: the item branch's free per-item embedding is replaced by a
**feature-composed embedding** in the LightFM/SVDFeature template,

> q_i = Σ_{f ∈ f_i} e_f  (+ e_ID(i) when `use_id_feature=True`),

where the e_f are learned d-dimensional embeddings of the item's
attribute values. Everything else is the host, untouched: item prototypes
P^t ∈ R^{K_t×d}; activation t\*_k = 1 + cos(q_i, p_k); the user is a free
vector u ∈ R^{K_t} in item-prototype-similarity space; score S(u,i) = u·t\*;
sampled-softmax loss with the host's two in-batch inclusion regularizers.
No bias terms anywhere (fleet convention `use_bias=0`).

Feature vocabulary: on H&M the canonical 5 fields (department, product_type,
section, colour_group, graphical_appearance; V = 426); on ml-1m
genres + tags (bags layout, 1,061 token codes). Cost vs the host: +V·d ≈
+27K parameters at d = 64 (~3%); ~1.05–1.1× step cost; the pipeline profile
(indexed gathers, in-batch losses, no precompute) is preserved.

**Arms.** `feature_item_proto` (**ids**, headline: features + ID row);
`_noid` (features only — the deliberately decoupled contrast, S1);
`_f0` (zero features + ID — architecturally the host through the fI code
path; the keystone reduction is test-pinned bit-identical at equal config).

**Cold inference.** A cold item's q_i exists from features alone; for the
ids arm the untrained ID row is **dropped** at cold scoring (LightFM
fold-in spirit; implemented as the audited `drop_cold_id_rows` machinery,
recorded per run in the cold report).

## 2. Motivation and relation to the host

The thesis's two contribution facets, and what this stage claims on each:

- **(A) Intrinsic prototype interpretability via feature grounding.** The
  composition co-embeds feature values and prototypes in one metric space.
  That buys the candidate's new interpretability axis —
  **parameter-space feature grounding with exact score attribution**:
  (i) prototype meaning is read from parameters alone
  (profile cos(e_f, p_k) — no data pass, no top-k proxy), and (ii) every
  activation splits *exactly* into named per-row summands
  c_{r,k} = e_r·p_k / (‖q_i‖‖p_k‖). On the I host this covers **100% of
  the item-discriminating score**: S − Σ_k u_k = Σ_k Σ_{r∈rows(i)} u_k·c_{r,k},
  the Σ_k u_k baseline being exactly rank-inert. The host has neither half:
  its prototype meaning is reconstructed post-hoc from top-k items, its
  activations are atomic.
- **(B) Sparsity / cold-start robustness.** A never-interacted item gets a
  meaningful, feature-computed prototype activation — the mechanism behind
  the cold contribution, with LightFM's published cold results as the
  external precedent.

**Staged bar.** The lineage grounds each variant against its own host: this
stage's empirical bet is **fI vs I-ProtoMF, like-for-like** (identical
losses, search space, eval). The paper-headline UI-ProtoMF bar is
deliberately deferred to the fUfI merge stage — a staged bar, not a lowered
one. The user side here is host behavior (free u; its coefficients are
CF-learned personalization, disclosed as such).

## 3. Experimental setup

Everything runs under the S0.6 comparison-fairness charter (see
`s0_foundation.md` §S0.6; SC.6 cites compliance clause by clause — C1–C8
audit at SC.8.1: **PASS, 11/11 runs**).

- **Datasets:** `hm_1_month` (+ `hm_1_month_cold`) primary — thin histories ×
  rich vocabulary, 67.5% signature-twin regime; `ml-1m` warm as second
  testbed — the complementary heavy × coarse regime (3.8% twins). No ml-1m
  cold (machinery deferral, recorded).
- **Base settings, evidenced:** the lineage's (shifted cosine, `use_bias=0`)
  pair is no longer merely inherited from the host — the cosbias2×2
  mechanism-tier wave (18 arms, vault 2026-08-17_1937) showed the cosine
  flip is score-inert on the I-host (placebo passed), removing the shift
  relocates popularity into the embedding geometry itself (user-cone
  displacement) while prototypes stay collapsed, and bias-on helps only
  the I-host while importing the untrained-cold-bias problem into the cold
  story. The incumbent setting keeps popularity in its most
  visible, separable channel — managed and disclosed, not amputated.
- **Eval:** temporal leave-one-out, 1 positive vs 99 uniform negatives,
  HR/NDCG@{1,3,5,10,50}, selection on val hit_ratio@10. Cold eval per the
  S0.3 spec: 20% stratified held-out items, **cold-vs-cold primary**
  (99-negative pools of cold items), single-config retrain convention,
  training-negative exclusion.
- **Runs (11 jobs, ~16–17 GPU-h, LEO5):** hm hyperopts fI/noid/f0; cold
  retrains fI/noid + cold evals; ml-1m hyperopts fI/noid; ml-1m fleet rows
  item_proto/mf/lightfm_tags/lightfm_tags_ids (computed once, charter
  lazy-addition). H&M baselines come from the **frozen S0.7 reference
  table** — never re-run.
- **The f0 yardstick:** f0-vs-frozen-`item_proto` is a same-architecture
  pair differing only in wrapper/init path — its gap (one draw) calibrates
  the single-seed noise scale that C4's budget otherwise leaves
  unquantified.
- **Tie brackets:** every cold number is quoted with its F-S0-14 bracket
  (worst / expected / best HR@10 under raw-logit tie-breaking inside the
  actual 99-negative ranking). All claim-carrying brackets below are tight
  (width ≲ 0.001), so points are quoted with bracket bounds in [·].

## 4. Results

### 4.1 Warm accuracy (R6) — both testbeds

**hm_1_month (thin × rich; HR@10 / NDCG@10):**

| row | HR@10 | N@10 | source |
|---|---:|---:|---|
| **fI (ids)** | **0.4984** | **0.2757** | this fleet |
| f0 yardstick | 0.4912 | 0.2642 | this fleet |
| fI-noid | 0.4229 | 0.2317 | this fleet |
| item_proto (host bar) | 0.4938 | 0.2631 | frozen |
| lightfm_tags | 0.4582 | 0.2709 | frozen |
| lightfm_tags_ids | 0.5087 | 0.3063 | frozen |

**ml-1m (heavy × coarse; HR@10 / NDCG@10):**

| row | HR@10 | N@10 | source |
|---|---:|---:|---|
| **fI (ids)** | **0.5156** | **0.2836** | this fleet |
| fI-noid | 0.4960 | 0.2735 | this fleet |
| item_proto (host bar) | 0.5711 | 0.3187 | this fleet (Wave D) |
| mf | 0.5022 | 0.2793 | this fleet (Wave D) |
| lightfm_tags | 0.5565 | 0.3197 | this fleet (Wave D) |
| lightfm_tags_ids | 0.5612 | 0.3261 | this fleet (Wave D) |

(No f0 arm on ml-1m by design — the noise yardstick is measured once, on
the primary testbed; the F=0 keystone itself is layout-tested
bit-identically.)

- **Noise yardstick (hm):** |f0 − item_proto| = 0.0026 HR@10 / 0.0011
  N@10 — the single-seed wrapper/init noise scale (one draw, a scale
  indicator, not a CI).
- **fI vs host, hm:** +0.0046 HR@10, within ~2× the yardstick → **warm
  parity, not a win**. N@10 +0.0126 is ~11× the yardstick's N@10 draw — a
  candidate real effect, single-seed → "no warm regression; possible NDCG
  gain; dev-provisional."
- **fI vs host, ml-1m:** −0.0555 HR@10 / −0.0351 N@10 — a real
  **regression on the heavy × coarse regime**, opposite sign to hm; fI
  also trails both lightfm rows there (but beats mf). Caveat attached:
  fI's best ml-1m trial selected popularity-sampled training negatives (a
  searched, host-inherited dimension; equal search space across rows; eval
  negatives pinned uniform) — mechanism-level attributions of ml-1m deltas
  carry that training-distribution confound.
- **The two testbeds disagree, and that is the finding:** parity on
  thin × rich, deficit on heavy × coarse — the regime boundary is part of
  the result, not noise to average away (single-seed on both).

### 4.2 The ids-vs-noid gap, decomposed (§3.6 interpretation rule)

Raw gap: hm **0.0755** HR@10; ml-1m **0.0196**. The rule forbids quoting
this as "the cost of interpretability"; the instruments now make the
decomposition visible:

- **D4 popularity strata (hm, HR@10):** noid's deficit is a **head-bucket
  phenomenon** — −0.335 vs fI on items with 101+ train interactions
  (fI 0.882, noid 0.547) — while noid *leads* on every tail bucket
  (1–5: 0.236 vs 0.077; 6–20: 0.258 vs 0.073; 21–100: 0.348 vs 0.216).
  Component (1), popularity discrimination, dominates the gap.
- **D3 popularity-hard negatives:** fI loses 2.4× more than noid when
  negatives are drawn pop^0.75 (fI −0.31, noid −0.13) — consistent with the
  ID channel carrying popularity.
- **History-quartile inversion:** fI rises with user history
  (0.481→0.514 Q1→Q4), noid falls (0.430→0.404) — noid is relatively best
  exactly for thin users (the facet-B surface).
- **Cross-dataset consistency:** the gap is 3.9× larger on twin-rich hm
  than on twin-poor ml-1m — directionally consistent with components
  (1)/(1b) (popularity blindness / angular twin-unrealizability,
  hm-concentrated) dominating over component (2) (taste-structure loss,
  which alone would appear on both). The two regimes also differ in user
  depth — ml-1m median train history 56 vs hm_1_month's 5, an ~11× gap
  (vault 2026-08-17_1407) — a second named axis behind the regime
  boundary, alongside vocabulary coarseness.
- **Registered outlook (not measured here):** the angular twin tax behind
  (1b) is predicted to be **stage-local** — the length-channel refund
  hypothesis (design doc §I.4, vault 2026-07-12_1224) expects the fUfI
  merge's linear halves to buy twin separation and popularity back through
  the norm channel, with this stage's twin-angle and D4 measurements as
  the prediction's evidence base (the refund reimports the channel's
  convicted properties — cold magnitude handicap, Steck exposure — which
  is why it is a merge-stage experiment, not a dc01 claim).

### 4.3 Cold-start (R5)

**Scope: the cold evidence is H&M-only.** The ml-1m cold block is in scope
for the thesis but deferred behind the ml-1m_cold enablement work
(bags-aware kNN/tie machinery, F-DC05-16 disposition) — every cold claim
below is a one-testbed claim and says so. Cold-vs-cold HR@10 on
`hm_1_month_cold` (2,730 held-out items; every number with its tie bracket
[worst–best]; popularity chance floor for reference):

| row | HR@10 [bracket] |
|---|---|
| lightfm_tags_ids | 0.5080 [.5074–.5089] |
| lightfm_tags | 0.4705 [.4700–.4711] |
| **fI-noid (native)** | **0.3882 [.3875–.3892]** |
| **fI (native, ID-drop)** | **0.2876 [.2869–.2882]** |
| item_proto + attr-kNN patch | 0.2358 [.2352–.2366] |
| popularity reference | 0.1202 |
| item_proto native | no signal (all-tied; bracket expected = chance 0.10) |

- **Feature composition delivers real, tie-robust cold capability:** both
  fI arms clear the chance floor by 2.4–3.2× and clear the kNN-patched host
  with disjoint brackets — where the CF host natively has *no* cold signal
  at all. This is facet B's headline evidence at this tier.
- **Both arms sit below the LightFM feature baselines:** on this testbed at
  this tier, post-hoc patching < prototype-mediated composition < direct
  feature factorization. Recorded absolutely; it feeds the narrative (what
  the prototype bottleneck costs at cold), not the R5 rating (which asks
  for meaningful cold representation — delivered).
- **noid > ids at cold is metric-valid** (brackets disjoint) but its
  *mechanism reading* stays open: the ids arm's cold deficit co-occurs with
  its head-concentration and ID-dominated compositions (ID linear share of
  ‖q‖² mean 0.31; ID/metadata norm ratio median 0.46 — mass that vanishes
  at the cold drop; F-DC01-04 operating-point mismatch), and noid's cold
  top-10 remains ~99% signature-twins in the full-catalog view — owned by
  the twin-angle/D4 instruments, not by the metric.

### 4.4 Geometry (what the accuracy numbers stand on)

Committed instrument set (F-DC01-05/08/09), run on the **hm checkpoints**
(D3/D4 for the three hm arms; geometry for fI/noid/f0/item_proto — the
primary-testbed scope the SC.6 plan committed; ml-1m mirrors are a
designable extension, not run). Working basis:

- **The activation cloud is nearly collinear on every arm** — effective
  rank 1.22 (fI) / 1.93 (noid) / 1.14 (f0, host) of K = 76–99. This is the
  geometry-level image of the S0-era finding "prototype collapse =
  self-built popularity bias" (vault 2026-07-15_1112; mechanism resolution
  vault 2026-08-20_1218), now measured on the fI lineage. The F-DC01-05
  spread-mediation weakening is real and severe. **The collapse is not
  data-fated:** a working-tier probe (vault 2026-08-17_1407) found the
  hm_3_month host checkpoint essentially un-collapsed (56/58 distinct
  prototype names) under a strong coverage regularizer
  (`sim_batch_weight` 5.71), while the collapsed hm_1_month winners here
  had that term effectively off (fI 0.0018) — collapse reads as *data
  pressure × regularizer configuration*, with a designed one-run isolating
  experiment (raise `sim_batch_weight` alone) recorded but not queued.
  Uncontrolled comparison (different K/d/population); it disproves
  "inherent", it does not yet apportion.
- **noid partially escapes:** prototype pairwise cos −0.001 (vs fI 0.274
  with literal duplicates at p99 = 1.0); its user vectors put only 6.6% of
  energy on the dominant (popularity) direction vs 63–81% for ids/f0/host —
  popularity-blind by construction, taste routed to lower directions.
  Independent structural corroboration of the D3/D4 story.
- **Twin individuation (F-DC01-09):** the ids arm does individuate
  signature twins — twin q-cos 0.58 vs the structural noid 1.0 — the
  feared max_norm suppression route was available but the hyperopt selected
  no max_norm.
- **M2 omnipresence:** fI gives `Solid` (58% of catalog) mean linear share
  0.31; noid gives it 0.058 and routes large shares to discriminative
  values. **f0 ≈ item_proto to three digits on every geometry statistic** —
  the yardstick holds at the geometry level too.

## 5. Explanation showcase

**The axis (SC.5): parameter-space feature grounding with exact score
attribution** — prototype meaning read from parameters alone; activations
split exactly into named feature summands covering 100% of the
item-discriminating score. Two standing instruments (shared with the host,
fairness contract "maximum shared principle — advantages from the algorithm
only"): **prototype cards** (question 1: what do the prototypes look like)
and the **shared breakdown renderer** (question 2: why this recommendation),
identical frame for every model, mechanism entering only through declared
slots, every artifact self-verifying (rendered parts asserted against the
model's own forward score).

**Profile validity (F-DC01-02, mechanized via the dual naming route):** on
the fI hm checkpoint, intrinsic (cosine) and post-hoc (lift) prototype names
agree — mean descriptor overlap 0.675, 91% of prototypes share ≥1/3
descriptors. The intrinsic read-out is validated in the only sense available
on this checkpoint: both routes truthfully describe the same space.

**A real side-by-side pair** (SC.8.4 spot-check on the primary testbed —
the renderer and both naming routes are ml-1m-capable and ml-1m artifacts
exist in the deep-dive tree, but the mandated spot-check pairs are hm;
same user, same item, both models; artifacts:
`Master/experiments/expl_deep_dive/4.1s_sc8_side_by_side/`, pairs for users
42/1000/5000). User 5000, item 1269 — a tail item (Costumes · White ·
Jersey):

- **fI:** S = +0.1840 = rank-inert baseline +0.0834 + item-discriminating
  +0.1006, of which **feature-explained +0.0870 vs item-ID +0.0136
  (features: 86.5%)**. The zoom decomposes the top prototype's bar exactly
  into named rows (section = H&M+ +0.0083, ID row +0.0030, product_type =
  Costumes −0.0008, …; Σ = the bar, asserted).
- **Host (item_proto):** same frame, same math on the u_k·(t\*_k−1) bars —
  but the zoom is declaredly post-hoc: nearest items after training, here
  all at saturated 1+cos = 2.000 with semantically incoherent membership
  (pyjama / shirt / dress / hat / bikini under one confident name).
- The contrast the pair demonstrates: for a *popular* item (670, users
  42/1000) fI's feature-explained fraction drops to **12.0% / 11.8%** — the
  renderer *says so* (ID row as its own line + the M3 fraction) instead of
  overclaiming grounding. The intrinsic surface degrades **honestly** into
  exactly the popularity story §4.4 measured.

**The collapse dominates both models' explanation surfaces** (R-g, measured
in the wild): 76 fI prototypes carry only 6 distinct intrinsic names; the
host's post-hoc naming is equally degenerate in its own vocabulary; 74/76
per-prototype contributions sit within ±4% of each other. Names imply
structure that is not there — on both models equally (host-inherited,
GR7-fair). The noid arm restores across-card diversity (92/99 distinct
names). Reading rule (observation log §4.1.1): saturated profile bars mark
lockstep-clique membership, not single-value meaning — clique-level reading
per the F-DC01-06 grouped-concept discipline.

## 6. Requirements scorecard (evidence-backed, SC.8.5)

Condensed from the SC.8 re-rating; all dev-provisional. **No self-rating
was contradicted by evidence** — every "partial" was confirmed as partial.

| Req | verdict | one-line basis |
|---|---|---|
| R1 tied | strong, confirmed | features are the item representation; f0 ≈ host to 3 digits; cold path 100% feature-computed |
| R2 intrinsic | strengthened, confirmed | exact decomposition self-checked on every artifact; dual-route concordance; the surface truthfully reports popularity where popularity is what the model computes |
| R3 prototype | strong (mechanism) — geometry caveat | prototype-shaped end-to-end, but 6 names / 76 prototypes in practice; collapse is config-dependent (coverage-knob probe, §4.4) and routes to the Rashomon/coverage stage |
| R4 grounded | partial, confirmed | grounding real and read out; unenforced sharpness materializes (duplicates, `Solid` omnipresence) — "partial" was the right call |
| R5 sparsity | strong, confirmed | tie-robust cold above floor and patched host (H&M-only evidence; ml-1m cold deferred); LightFM rows above both fI arms — placement note, not a rating deficit |
| R6 dense | partial — split verdict | hm: parity + possible N@10 gain; ml-1m: real regression; regime boundary named |
| R7 vocabulary | partial, confirmed | rendered names are retailer-internal taxonomy; the 4.0 tension realized |
| R8 parsimony | strong | one mechanism, nothing added |
| S1 decoupled | strong, confirmed | the noid arm produced the tying-value evidence |
| S2/S3/S4 | unchanged (re-scoped / untouched / open light experiment) | |
| S5 pipeline | strong, confirmed | fI 2h37 vs f0 1h50 hyperopt; no anomalies in 11 runs |

## 7. Limitations and honest caveats

1. **Tier.** Single seed, dev budget; every claim dev-profile-provisional.
   The f0 yardstick is one draw. Thesis-grade numbers are a separate,
   queued tier. Geometry statistics carry an extra caveat: geometry is
   underdetermined by the selection metric (loss-equivalent twin runs
   landing at 7 vs 2 effective directions — vault 2026-08-17_1407
   Rashomon note), so §4.4 characterizes *these* checkpoints, not the
   config class.
2. **Steck / read-out validity.** The trains-through-cosine steel-man
   protects the item-side semantic read-outs; the user coefficients u_k are
   dot-product-trained with no such protection — governed by the gauge
   discipline (F-DC01-08; effective-gauge residue measured as a continuum).
   The threat that actually materialized is **geometry collapse**, not
   rescaling: route agreement survives it, but names over a collapsed space
   imply structure that is not there (§5). Label confounding stands:
   nothing pins e_"Dark Blue" to darkness beyond the interactions of items
   carrying it.
3. **Share identifiability (F-DC01-06).** Splits across strongly-associated
   value cliques are gauge-like; grouped/clique-level readings are the
   identified quantities. The decoupling proxy behind this is scrutiny
   working basis — the thesis gets dedicated experiments.
4. **ml-1m regression + confound.** The −0.0555 host deficit on ml-1m is
   real at this tier and carries the popular-training-negatives caveat; the
   fI story is regime-dependent (parity on thin × rich, deficit on
   heavy × coarse), single-seed.
5. **Cold placement.** Both fI arms sit below the direct-factorization
   LightFM baselines at cold; the prototype layer's cold value at this
   stage is that it delivers *grounded, prototype-shaped* cold
   representation, not the best cold number on the board.
6. **noid mechanism reading open.** "noid > ids at cold" and the warm gap
   decomposition rest on instruments (D4, twin angles) that inform but do
   not yet close the attribution; F-DC01-04's train/cold operating-point
   mismatch is a live alternative contributor.
7. **Known quoting rules carried:** cold numbers always with brackets
   (F-S0-14, closed: disclosed-bracket layer, no S0.7 re-open); native
   no-signal CF cold blocks quoted as chance-floor brackets, their 0.00
   points an artifact of sigmoid tie-fabrication (F-S0-15, document-only).
8. **Hidden-effects routing.** Determinacy annotations, gauge disclosures,
   and per-line factorization commentary are deliberately absent from
   rendered artifacts (SC.5 re-scope) — they belong to the thesis's
   dedicated hidden-effects section with experiments designed at writing
   time.

## Provenance

Dossier: `Master/docs/scrutiny/sc01_feature_item_proto.md` (SC.1a–SC.8,
gates closed through 2026-08-20). Design doc:
`Master/docs/design_candidates/dc01_feature_composed_factors.md` (fI
re-host amendment + I.1–I.5). Frozen references: `s0_foundation.md` §S0.7.
Brackets: `Master/temp/tie_bracket_sweep_2026-08-20.md`. Instruments:
`<run>/popneg_readout|stratified_readout|sc8_geometry/` under
`Master/experiments/sc6_runs/` (cluster mirrors). Rendered artifacts:
`Master/experiments/expl_deep_dive/4.1*` incl. `4.1s_sc8_side_by_side/`;
observation log `Master/docs/scrutiny/explanation_observations.md`.
