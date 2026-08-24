# dc06 — fI′: field-mean-centred composition on fI-ProtoMF (U2b)

> Design-candidate cycle, protocol v1.3 (`Master/docs/design_candidate_protocol.md`).
> **Directed cycle** (like dc05): the user's destination directive
> (`Master/temp/composition_upgrades_plan_2026-08-24.md`, Matteo 2026-08-24) names the
> candidate — **fI′ = U2b field-mean centring ONLY**, a single-variable variant on the
> closed dc01 (`feature_item_proto`) host; knobs, not a new `ft_type`; new default if it
> wins. U1 (ID-dropout) and U3 (norm read-out) are explicitly out of scope this cycle;
> U4 parked.

## 0. Frame

- Occupied fingerprint regions (index, dc01–dc05): features→factors on the I host
  (dc01, **scrutinized, closed SC.11**) and its user-side sibling on the U host (dc05,
  hardened); attribute-space bottleneck prototypes (dc02, lukewarm); visual exemplar
  (dc03) and anchor-parameterized prototypes (dc04), both abandoned.
- This cycle is **directed**: the plan file fixes the seed (U2b field-mean centring
  inside `FeatureEmbedding`'s composition, `t_i = Σ_f (e(x_if) − μ_f) + e_ID`). As with
  dc05's M1 waiver, strict fingerprint distinctness from the parent (dc01) is waived by
  user directive; the delta axis is stated anyway: same features-entry family, but the
  composition is **gauge-fixed** (exact shared component C = Σ_f μ_f removed in-model),
  which changes the geometry that grounding and read-out operate in.
- Requirement niche served: R6 (dense-regime accuracy — fI's ml-1m −0.0555 host deficit
  is the standing sore point; ml-1m is the reference benchmark per standing rule) and
  the §4 identifiability tension (the sI gauge result: per-field offsets are exactly the
  unidentified directions of attribute profiles — centring is the canonical gauge fix).
  *(Cold-read pointer, 2026-08-24: "gauge fix" language throughout §0–§3 is governed
  by the post-5b A4 framing note in §5 — fI′ is an attribution-gauge fix AND a
  score-side intervention, never a score no-op; "canonical" = the within-field share
  gauge only.)*
- Evidence already in hand (U2a instrument, gate PASSED 2026-08-24): C ≈ 85% of mean
  metadata composition on fI; post-hoc centring lifts activation eff. rank 1.218→1.979;
  `Solid` omnipresence exposed as a C artifact (share 0.315→0.022); weighted mean is the
  effective flavour. Prototype duplication is untouched (separate `sim_batch_weight`
  route).
- Scratchpad: no open user entries left un-dispositioned from prior cycles; relevant
  open captured entries for this cycle's sweep (Step 1b): the E2/gauge material lives in
  the plan file rather than the scratchpad; captured entries on length channels
  (2026-07-12) and cold floors (2026-07-13) look adjacent but fUfI/thesis-stage — to be
  dispositioned in Step 1b.

## 1. Seed selection

### 1a. Corpus scan (directed cycle — scan confirms fit, does not re-open the choice)

The directive seed (U2b) is not a reading-list paper; its literature home was located
by the 2026-08-24 four-agent deep research (`composition_deep_research_2026-08-24.md`):
the **common-component-removal / anisotropy line** — Mu & Viswanath ICLR 2018
("All-but-the-Top"), Arora et al. ICLR 2017 (SIF common-component removal), Gao et al.
ICLR 2019 (representation degeneration / cone collapse) — all **external** to the
current reading list (added in Step 6). Corpus anchors that stay load-bearing: **B5
LightFM** (the composition being repaired is dc01's B5 sum), **B3 SVDFeature**
(features→factors template), **A1 ProtoMF** (host, shifted cosine), **F1/F2**
(identifiability framing for why geometry cleanup is an interpretability matter, not
just accuracy). Theme sweep found no in-corpus competitor mechanism for "fix the
composed geometry without changing the function class" — the corpus's own routes
(weighting = E1, redundant per the plan's reparameterization proof; attention = Tier C,
forfeits attribution) were already adjudicated in the plan.

### 1b. Scratchpad sweep (every open entry, one line each)

- *Attach feature profiles to CF prototypes (user, 2026-06-11)* — not this cycle
  (loss-enforced grounding remains its own future candidate; fI′ is single-variable).
- *Feature-injection fork: prediction vs prior (user, 2026-06-16)* — answered inline:
  centring is neither a new injection nor a prior — it is a **gauge fix inside route A**;
  route B stays open.
- *Add / interact / regularize (user, 2026-06-16)* — not this cycle (fI′ stays ADD;
  REGULARIZE route still open).
- *Attribute-source prioritization (user, 2026-06-16)* — consistent, no new consumption
  (item-side primary; fI′ is item-side).
- *Sequencing item-only → unified → separate (user, 2026-06-16)* — consistent: fI′ is a
  stage-1 refinement, no stage change.
- *Two axes + Rudin caution (user, 2026-06-16)* — **shapes this candidate**: centring is
  pure Axis A (grounding geometry); scorer untouched; Rudin axis clean — one disclosed
  linear subtraction, arguably *simpler* semantics (profiles read as within-field
  contrasts instead of raw positions).
- *Hybrid bridge desideratum (user, 2026-06-16)* — n-a (no added module).
- *[captured dc02] loss-enforced grounding* — not this cycle (future candidate).
- *[captured dc02] grounding-enforcement spectrum* — answered inline: fI′ stays at the
  unenforced/emergent point; centring changes the geometry the emergence happens in,
  not the enforcement strength.
- *[captured dc03] part/patch visual prototypes* — not this cycle (S3/future).
- *[captured dc03] multimodal frozen content space* — not this cycle.
- *[captured dc04] differentiable live centroids (REGULARIZE with read-out)* — not this
  cycle, but **adjacency noted**: U2b's differentiable per-step μ_f is likewise a live
  in-graph mean whose gradient touches all same-field rows; here it enters as a
  *subtraction* (gauge), not as prototype anchors — mechanically related, different role
  (examined in Step 3/4b).
- *[captured scrutiny dc01] user-side explanation gap* — consumed by dc05; untouched.
- *[captured scrutiny s0] dual-side candidate* — not this cycle (fUfI stage).
- *[captured scrutiny s0] 3-scenario cold testbed* — not this cycle (eval design).
- *[captured scrutiny dc01] length-channel refund* — not this cycle (fUfI-stage);
  adjacency: centring alters summand *directions*, and the ingredient-row norms remain
  the vote weights — the length-channel story is unchanged.
- *[captured scrutiny dc01] ID-residual diagnostic* — not this cycle; note: centring
  leaves e_ID untouched but changes the vocabulary gauge any residual probe reads
  against (flagged for the read-out discussion, Step 3).
- *[captured dc05] fI-vs-fU word-geometry comparison* — not this cycle (merge-stage
  evidence); caveat recorded: if fI′ becomes the new default, that comparison should be
  run on *centred* tables or the C component will dominate the Procrustes alignment.
- *[captured dc05] merge inherits live + inert baseline masses* — not this cycle
  (fUfI pre-audit).
- *[captured scrutiny dc05] cold-user degradation floors* — not this cycle
  (hidden-effects exhibit).
- *[captured dc03 abandonment] photos as rendering* — not this cycle (presentation
  layer).
- *[captured dc04 abandonment] data-anchored parameterization* — not this cycle
  (reserved future stage).

### 1c. Shortlist (from the plan's U1–U4, the natural competitor set)

1. **U2b field-mean centring** — *motivation:* R6 (attack the structural seed of the
   collapse implicated in fI's geometry; ml-1m deficit is the standing R6 sore point) +
   §4 identifiability (the sI gauge result: per-field offsets are exactly the
   unidentified profile directions — centring is the canonical fix) + R4 (profiles
   rendered in a gauge where `Solid`-type omnipresence artifacts vanish). *Distinctness:*
   vs dc02/dc03/dc04/dc05 differs on the entry axis outright; vs dc01 the strict
   distinctness is **waived by user directive** (directed cycle, dc05-M1 precedent) —
   stated delta: gauge-fixed composition ⇒ changed grounding geometry + one new
   disclosed read-out line. *R8:* one linear subtraction, one switch — comfortably one
   idea. **Chosen.**
2. **U1 ID-dropout** — motivation R5/E4 (cold operating-point mismatch, strongly
   observed); *rejected this cycle by explicit user decision* (2026-08-24: deferred —
   ids/noid endpoints bracket the axis; would drag the matched LightFM control arm).
3. **U3 norm-as-weight read-out** — motivation R2/R4 (render the learned weights);
   rejected: user-deferred to thesis-writing time; also not candidate-shaped (pure
   read-out, no mechanism).
4. **U4 FM-style pair terms** — motivation R6 (conjunction gap); rejected: user-parked
   behind trigger conditions; weakest cited+observed pairing; explanation-object growth.

### 1d. Chosen seed

**U2b field-mean centring on the dc01 host** — `t_i = Σ_f (e(x_if) − μ_f) + e_ID(i)`,
μ_f a per-field mean of the word table (flavour to fix in Step 3; U2a found the
frequency-weighted mean effective). Provenance: **corpus+external** (user-directed;
external anchors Mu & Viswanath 2018 / Arora et al. 2017 / Gao et al. 2019 to be added
to the reading list; corpus anchors B5/B3/A1). Why it won: the only shortlist member
both (a) user-adopted for this cycle and (b) already instrument-validated — the U2a
gate passed on real dc01 checkpoints (C ≈ 85% of mean metadata composition; post-hoc
centring lifts eff. rank 1.218→1.979; `Solid` share 0.315→0.022).

## 2. Mechanism extraction

PDFs read this cycle (downloaded 2026-08-24 into `papers/F_disentanglement/`):
`F6_mu_viswanath_2018_all_but_the_top.pdf` (published ICLR 2018 version, arXiv
1702.01417v2), `F7_arora_2017_sif_sentence_embeddings.pdf` (⚠️ *under-review* ICLR 2017
copy — OpenReview's camera-ready is bot-blocked; equation numbers below are from this
copy, re-check at BibTeX time), `F8_gao_2019_representation_degeneration.pdf`
(published ICLR 2019 version, arXiv 1907.12009v1). Host equations re-verified against
the local A1 ProtoMF PDF.

### 2.1 The host being modified (A1 + dc01, for self-containment)

I-ProtoMF scores `s(u,i) = Σ_k u_k · t*_k(i)` where `t*_k(i) = sim(t_i, p_k) =
1 + cos(t_i, p_k)` — the shifted cosine, values in [0,2] (A1 §3.1, Eq. 1; score form
Eq. 2, item-side symmetric §3.2). dc01 replaces the free item embedding with the
composition `t_i = Σ_f e(x_if) + e_ID(i)` — one embedding row per attribute field
value (F fields, one value each, fixed layout) plus a free per-item ID row (dc01 file
§3; `FeatureEmbedding` in `feature_extraction/feature_extractors.py`). Row norms are
unconstrained (the hm winner's hyperopt had `max_norm` available and selected none —
sc01 §4.4).

### 2.2 F6 — Mu & Viswanath, "All-but-the-Top" (ICLR 2018)

- **Observation (§1, p.1):** *every* word representation tested has (i) a non-zero
  common mean vector — ‖μ‖ up to ~1/2 of the average vector norm (Table 1, p.3 gives
  per-corpus numbers, e.g. GloVe avg ‖v‖=8.30, ‖μ‖=3.15) — and (ii) after mean
  removal is still far from isotropic: most energy sits in a handful of directions
  (~8 of 300 dims; Fig. 1 singular-value decay, p.3).
- **Algorithm 1 (§2.1, p.4)**, verbatim structure: (1) compute μ = 1/|V|·Σ_w v(w),
  set ṽ(w) = v(w) − μ; (2) PCA on {ṽ(w)}; (3) v′(w) = ṽ(w) − Σ_{i=1..D}(u_iᵀv(w))u_i.
  Rule of thumb D ≈ d/100 (§2.1 Discussion, p.4).
- **Frequency link (Fig. 2, p.4):** the top-two PCA coefficients visibly encode word
  *frequency* — the dominating directions are frequency directions.
- **Why it works (§2.2, p.5):** isotropy measure I({v}) = min_c Z(c) / max_c Z(c)
  (their Eq. 1, Z(c) = Σ_w exp(cᵀv(w))); first-order approximation shows I = 1 ⟺
  Σ_w v(w) = 0 (mean removal is exactly the first-order isotropy fix); second-order
  approximation motivates flattening the top singular values (the PC-removal step).
  Empirically isotropy rises 0.7→0.95 (word2vec), 0.065→0.6 (GloVe) (Table 2, p.6).
- **What they showed:** post-processing consistently improves word similarity (avg
  +1.7% over 7 datasets, Table 3 p.6), concept categorization (avg +2.5%, Table 4
  p.7), semantic textual similarity (avg ~4%, Table 6 p.7); word analogy only
  marginally (Table 5 p.7 — they note the analogy subtraction already cancels common
  components, an observation that transfers to *any* difference-based read-out).
- **Limitations/scope:** purely **post-hoc** post-processing of trained tables; no
  training-time variant is proposed in the paper. Domain is NLP word vectors.

### 2.3 F7 — Arora, Liang & Ma, SIF (ICLR 2017)

- **Method (Algorithm 1, p.4):** sentence embedding = (1/|s|)·Σ_{w∈s} a/(a+p(w))·v_w
  (smooth inverse-frequency weights), then subtract the projection onto the **first
  principal component** of the set of sentence vectors ("common component removal").
- **Generative justification (§3, Eq. 2, p.3):** words are emitted by
  Pr[w | c_s] = α·p(w) + (1−α)·exp(⟨c̃_s, v_w⟩)/Z with c̃_s = β·c₀ + (1−β)·c_s,
  c₀ ⊥ c_s — the **common discourse vector c₀** models the frequent/syntactic
  component shared by all sentences. The MLE for c_s is approximately the
  SIF-weighted average (Eq. 3, p.4): argmax ∝ Σ_{w∈s} a/(p(w)+a)·v_w — i.e. the
  frequency down-weighting is *derived*, not heuristic.
- **§3.1 (pp.4–5, Eqs. 4–6):** word2vec's subsampling trick is shown to be an
  implicit, biased estimator equivalent to roughly the same weighting — evidence the
  correction is latent in standard training pipelines.
- **What they showed:** +10–30% on textual-similarity tasks over unweighted averaging
  (abstract + Table 2 area, §4); robust to the corpus used for p(w) and to a wide
  range of a (§4.2).
- **Relevance pointer:** SIF's c₀/frequency correction is the precedent for the
  **frequency-weighted** flavour of μ_f. *(Amendment post-5b P8, 2026-08-24: the
  original sentence here overclaimed "estimates precisely the frequency/popularity
  direction". Corrected: the catalog-count-weighted mean estimates the
  **catalog-composition** direction; the drift direction F8's mechanism actually
  inflates is **exposure/interaction-weighted**, and the two coincide only insofar
  as attribute-value interaction shares track catalog shares. An
  interaction-weighted flavour is registered in §3.1.)*

### 2.4 F8 — Gao et al., representation degeneration (ICLR 2019)

- **Phenomenon (§3, Fig. 1, pp.2–3):** under likelihood training with weight tying,
  the word-embedding table degenerates into a **narrow cone**; cosine similarities
  between (almost) any two words are positive (§3.2, p.4) — unlike word2vec or
  classifier softmax weights, which spread around the origin.
- **Cause (§4):** vocabulary frequencies are Zipfian; rare tokens dominate the
  vocabulary. Theorem 1 (p.5): with other parameters fixed, a never-appearing token's
  embedding optimum diverges along any *uniformly negative direction* of the hidden
  states — all such tokens drift **the same way**. Theorem 2 (p.5): such a direction
  exists iff the convex hull of the hidden states does not contain the origin (likely
  under layer normalization). §4.2/Theorem 3 (pp.5–6) extends the argument to
  *rarely*-appearing tokens: low-frequency words end up close to each other in a
  local region.
- **Fix (§5, Eq. 6, p.6):** MLE-CosReg — add γ·(1/N²)·Σ_i Σ_{j≠i} ŵ_iᵀŵ_j to the
  loss (minimize pairwise cosines of normalized embeddings; equivalently flatten the
  spectrum of ŴŴᵀ, their §5 analysis). Gains: −2.0 perplexity (WikiText-2), +1.08/
  +0.93 BLEU (WMT14 En↔De) (§1, p.2–3).
- **Relevance pointer:** the literature anchor that frequency-driven common-direction
  drift is an **optimization** pathology of embedding tables (not an NLP quirk); their
  remedy is a *regularizer* (family-adjacent to the host's `sim_batch`/R_batch route),
  whereas centring removes the shared component *directly* — complementary seeds vs
  symptoms framing, matching the plan's honest-scope note.

### 2.5 The proposed mechanism (all deviations flagged — "I propose")

**fI′ composition:** `t_i = Σ_f (e(x_if) − μ_f) + e_ID(i)`, where μ_f is the mean of
field f's value embeddings over the catalog (frequency-weighted by catalog value
counts — the U2a-validated flavour; the unweighted mean is the ablation flavour).
Everything downstream (shifted cosine, prototypes, score, regularizers) unchanged.

Deviations from the source papers, each flagged:
1. **Per-field, not global, mean** (I propose): the papers centre one global mean/PC.
   Our fixed layout (exactly one value per field) makes any per-field constant a valid
   gauge; the per-field mean is the natural choice because Σ_f μ_f = C is then an
   **exactly** shared vector across all items (plan §1.E2 derivation) — stronger than
   SIF's approximate common component c₀.
2. **In-training, in-forward** (I propose): F6/F7 are post-hoc. fI′ moves the
   subtraction into the forward pass so the *optimizer* sees the centred geometry —
   the U2a post-hoc instrument showed the gauge matters; only a trained variant can
   show whether the training dynamics change (norm growth, collapse pressure).
3. **Mean only, no top-D PC removal** (I propose): Algorithm 1's step 3 is dropped —
   at V≈426 words and d≤100 with F≈5 fields, per-field means already remove up to F
   directions; PC removal would add a spectral operation with no per-field gauge
   meaning and would complicate the attribution story. Recorded as an optional
   escalation, not stacked (R8).
4. **The ID row is NOT centred** (I propose): μ over ~10⁵ per-item rows would
   reintroduce a full-catalog dependency, and e_ID is per-item free capacity — there
   is no shared-gauge argument for it. The cold rule deletes e_ID anyway.

## 2b. Prior-art probe

Queries run (web, 2026-08-24):
1. `feature embedding mean centering recommendation cold-start "sum of feature
   embeddings" anisotropy common component removal`
2. `"all-but-the-top" OR "common component removal" recommender system item embedding
   centering popularity direction`
3. `prototype-based recommendation "prototype collapse" embedding anisotropy centering
   "effective rank" ProtoMF`
4. `"field mean" OR "per-field" centering categorical feature embeddings factorization
   machines LightFM "cold start" subtract mean training`

Findings — the closest lines, each distinguished:
- **Test-Time Embedding Normalization for Popularity Bias Mitigation** (arXiv
  2308.11288): normalizes embedding *norms* at inference to damp popularity — norm-
  not mean-based, inference-only, CF embeddings, no feature composition, no
  explanation motive.
- **"Rethinking Popularity Bias in CF via Analytical Vector Decomposition"** (arXiv
  2512.10688) and **"Between-User Collapse Under Popularity-Biased Feedback"** (arXiv
  2608.02548): decompose *CF* embeddings around a popularity/common direction —
  diagnosis of free ID tables, not a training-time gauge on a feature composition;
  useful as corroborating citations that a common/popularity direction exists in
  recsys embeddings too.
- **Embedding Cultural Diversity in Prototype-based Recommender Systems** (arXiv
  2412.14329, the I2 orbit): debiases ProtoMF-style prototype spaces (popularity/
  cultural bias) by intervening in prototype space — no item-feature composition, no
  centring-of-summands mechanism; confirms the citer neighbourhood still holds no
  feature-composed variant (§I takeaway re-verified for this mechanism).
- **F5 (Loveland et al. 2026)**, already on the reading list: semantic-**centroid
  initialization** of item tokens in generative recsys — init-time, different
  paradigm; corroborates the frequency-reorganization pathology, does not perform
  in-forward centring.
- NLP centring line (F6/F7 + e.g. "Correcting Mean Bias in Text Embeddings", arXiv
  2511.11041): post-hoc, global (not per-field), no recommender, no attribution
  contract.

**Verdict: found nothing close.** No published work applies mean-centring (per-field
or global) *inside the forward composition* of feature-based entity embeddings in a
recommender — let alone under a prototype layer with an intrinsic-attribution
contract. The adaptation's novelty surface: (i) exact-C structure from the
one-value-per-field layout (vs approximate common components in NLP), (ii)
training-time differentiable gauge vs post-hoc post-processing, (iii) the double duty
as attribution-gauge fix (sI identifiability result) — none of the neighbours has any
of the three.

*(Amendment post-5b P7 (A6), 2026-08-24 — nearest unexamined field, added after the
critique: the CTR **normalization** family (batch/layer/field-wise normalization on
categorical embedding layers) also performs in-training, differentiable mean removal.
Distinguished: LayerNorm subtracts the per-vector mean **across dimensions** (a
different axis); BatchNorm-style embedding normalization subtracts a per-dimension
**batch** mean of composed vectors — stochastic, interaction-weighted, ID row
included, variance rescaled, no per-field structure, no exact-C, no attribution
contract (and the NOVA-BERT/DIF-SR line found invasive side-info fusion of this kind
harmful). None yields identified per-field profiles. The novelty claim stands as an
adaptation; the probe's query-vocabulary bias is conceded on record.)*

## 3. Adaptation design

### 3.1 Modules and parameters

**No new `ft_type`** (user decision, plan header): fI′ = a knob on
`feature_item_proto`; if it wins it becomes the new default configuration. Changes:

- `feature_extraction/feature_extractors.py :: FeatureEmbedding` (upstream-derived —
  modifications-log entry due at build time):
  - new constructor args `center_fields: bool = False`,
    `center_mode: str = 'weighted'` (`'weighted'` = catalog-frequency-weighted mean,
    the U2a-effective flavour; `'unweighted'` = plain per-field value mean),
    `center_detach: bool = False` (fallback: epoch-refreshed detached buffer).
    *(Amendment post-5b P8 (A7), 2026-08-24: third flavour
    `center_mode='interaction'` — μ_f weighted by train-interaction counts over
    field-f values instead of catalog counts; still a per-field constant, so the
    exact-C property is preserved by the C1 argument. Unstacked ablation flavour;
    one buffer swap. Post-5b P10 (A9) mitigation option, also unstacked: smoothed
    weights `p̃_f(v) ∝ count_f(v) + ε` (SIF `a/(a+p)` form) so zero-frequency rows
    enter the mean and receive live-mean gradient.)*
  - two new **non-persistent buffers** built at init from `feature_ids` (both layouts):
    `field_index` (V,) mapping vocab row → field id, and `field_probs` (F, V) — the
    row-stochastic per-field count matrix P (P[f, v] = catalog frequency of value v
    within field f under `'weighted'`; uniform over the field's values under
    `'unweighted'`). Rebuilt deterministically like `feature_ids`, never checkpointed.
  - forward change (~15 lines): compute `M = P @ E_meta` where
    `E_meta = self.embedding_layer.weight[:self.n_features]` → M has shape (F, d),
    row μ_f (one V×d-scale matmul per forward; `center_detach` swaps in a buffered M).
    Then centre each metadata token before the reduction:
    `e'_{x_if} = e(x_if) − M[field_index[x_if]]`; the ID column is **not** centred.
    Fixed layout: `emb = self.embedding_layer(feat)` (…, F+1, d) →
    `emb[..., :F, :] −= M[self.field_index[feat[..., :F]]]` → `.sum(dim=-2)` as now.
    Bag layout: same per-token subtraction on the D_max metadata slots before the
    weighted reduction (weights untouched).
- `feature_extraction/feature_extractor_factories.py`: thread the three args from the
  model config into the `FeatureEmbedding` construction (both the `feature_item_proto`
  branch and its ablation variants).
- `confs/hyper_params.py`: **no new hyperopt dimension** — `center_fields` is a
  variant switch (designed arm), not a searched knob; the fI′ arms reuse the frozen
  fI winner configs (equal-treatment comparison; two-tier numbers policy applies).
  *(Superseded by user decision 2026-08-24, recorded in §6: each fI′ arm gets its own
  dev-profile 30-trial hyperopt (fleet convention) instead of frozen configs;
  `center_fields`/`center_mode` remain fixed per arm, never searched.)*
- Losses, data pipeline, negative sampling, `Trainer`/`Tester`: untouched.

**Layout caveat (I propose, stated honestly):** the exact-C property (`Σ_f μ_f = C`
identical for every item) holds only on the **fixed one-value-per-field layout**
(hm path). On the ml-1m **bag layout** (variable genre counts), per-token centring
still removes each summand's field-mean component, but the total removed mass
`Σ_f n_if·μ_f` is item-dependent — centring there is "frequency-direction removal per
token", not "exact shared-vector removal". Both are the same code path; the claims
differ per layout and are kept separate in §4b/§5.

### 3.2 Forward pass with shapes (scoring path, batch B, 1+n_neg candidates N)

1. `o_idxs` (B, N) → `feat = feature_ids[o_idxs]` (B, N, F); +ID column → (B, N, F+1).
2. `emb = embedding_layer(feat)` → (B, N, F+1, d).
3. `M = field_probs @ embedding_layer.weight[:n_features]` → (F, d)   [new]
4. `emb[..., :F, :] = emb[..., :F, :] − M[field_index[feat[..., :F]]]` → same shape  [new]
5. `t = emb.sum(dim=-2)` → (B, N, d)  — this is `t_i = Σ_f (e(x_if) − μ_f) + e_ID(i)`.
6. Host unchanged: `t*_k = 1 + cos(t, p_k)` → (B, N, K); score `s = Σ_k u_k·t*_k` (B, N).

Parameters: unchanged set — the same (V + n_items) × d table and the host's
prototypes/user table. **Zero new learned parameters.** Tie/R1 status: identical to
dc01 (single-branch host; the composed representation is the representation that
drives the score; on this host there is no second branch to weight-share with —
R1 operative core per the 2026-07-12 clarification).

Cold path: the cold rule composes `t_cold = Σ_f (e(x_if) − μ_f)` — same gauge as
training-time metadata composition (μ from the train-fitted table; no leakage — it is
a function of parameters, not of eval data).

### 3.3 Intrinsic explanation read-out

Unchanged in form, changed in gauge — every rendered number stays a forward-pass
quantity:
- Per-prototype contribution: `u_k · t*_k(i)` (host read-out, untouched).
- Per-word shares of an activation: `t_i·p_k = Σ_f δ'_{x_if}·p_k + e_ID(i)·p_k` with
  `δ'_v = e_v − μ_f(v)` — one named term per field word + the ID residual term,
  exactly as before, now in the centred gauge.
- **One new disclosed line** on the card (fixed layout): "shared catalog component
  removed: C" — the model does not score C, and the card says so.
- The share-gauge is now **canonical**, not a rendering choice: the sI identifiability
  result (per-field offsets are exactly the unidentified directions of profiles) is
  resolved *in the mechanism* — what U2a did as a declared post-hoc gauge, fI′ makes
  the model's own arithmetic.

*(Amendment post-5b P9 (A8), 2026-08-24: the card mock-up below is ONE presentation
option, not the committed card. Contrastive (centred) shares are a different display
object with real design problems — negative shares, incomparable field-mean
concentrations across cardinalities — and the "shared component removed" line serves
the thesis reader's fidelity contract, not the end user. The presentation decision is
deferred to thesis-writing time alongside the U3 decision, taken with the U2a
instrument outputs in hand; fI′ makes the contrastive gauge the model's arithmetic if
chosen, it does not force the card.)*

Rendered example (H&M-style, dark slim jeans, warm item, prototype k = the
"dark-denim-menswear" archetype): "Activation 1.62 of *dark-denim-menswear*: Colour
'Dark Blue' (vs field average) +0.31 · Garment group 'Trousers' (vs avg) +0.24 ·
Department 'Menswear denim' (vs avg) +0.18 · Section +0.05 · Product type +0.02 ·
item-specific +0.09. Shared catalog component removed from all items: C."
Cold card: same minus the item-specific line.

### 3.4 Interpretability-constraint design (v1.1 mandatory block)

**Inheritance.** Both host regularizers survive with semantics intact — and centring
arguably *restores* part of R_proto's intent:
- `sim_proto` (R_proto, each prototype pulled toward real entity compositions):
  prototypes are now anchored among **centred** compositions. Under raw fI, C·p_k
  rewarded prototypes for aligning with the common direction (activation mass
  cheapest along C — plan §1.E2); centred, that subsidy is gone and R_proto's
  "prototype ≈ a real taste region" semantics operate on the informative part only.
- `sim_batch` (R_batch, batch entities pulled toward prototypes): unchanged
  semantics; remains the primary anti-collapse knob for *prototype duplication*,
  which U2a showed centring does NOT fix (separate phenomenon, stays with the
  `sim_batch_weight` route).

**New degeneration modes** (beyond what R_proto/R_batch cover):
- **D1 — flat-direction drift.** Centring makes per-field constant shifts exactly
  score-inert: adding any constant vector c to every row of field f leaves every t_i
  unchanged (μ_f shifts by c too — both `center_mode` flavours). The optimizer can
  therefore random-walk raw rows along F·d flat directions. The *score* and the
  *centred read-out* are immune (that is the point — the previously score-entangled
  offset is now quotiented out), but raw-row diagnostics (U3-style norm read-outs,
  cross-run table comparisons) become gauge-noisy.
- **D2 — common-direction re-emergence in the ID block.** C is removed from the
  metadata sum only; in the ids-arm nothing stops a shared component re-forming
  across the e_ID rows (the free block is the one place item-independent mass can
  still hide). Would partially re-import the collapse subsidy through the back door.
- **D3 — residual anisotropy beyond the mean.** Mean removal is F6's first-order fix
  only; the field-value clouds can still share dominant *principal* directions
  (F6 §2.2 second-order term). Eff. rank 1.98 post-centring (U2a) vs noid's 1.93
  suggests most of the fI-specific gap closes, but the remaining anisotropy is
  uncovered.
- **D4 — live-mean feedback dynamics.** With differentiable μ, every gradient into a
  value row also pushes all same-field rows through −∂μ_f (rare values get moved by
  popular items' gradients). Expected small (means are 1/V-scale aggregates) but it
  is a moving-target coupling the host never had; could slow or destabilize
  convergence.

**Constraint candidates** (designed, NOT stacked — R8; adding one is the user's call):
- For D1 — *gauge pinning*: `L_pin = λ_pin · Σ_f ‖μ_f‖²` via `get_and_reset_loss()`
  (μ_f already computed in forward; ~2 lines). Minimized exactly at μ_f = 0, i.e.
  the raw table itself becomes centred and raw ≡ centred diagnostics. Gradient
  reaches all metadata rows through P. Cost ~0. Expected effect: kills the drift,
  makes U3 read-outs gauge-clean. **Rashomon-tunable** (any λ_pin > 0 in a wide
  range should land the same function; pure-gauge knob).
- For D2 — *ID-mean pin*: `L_idpin = λ_id · ‖mean_i e_ID(i)‖²` (subsample the mean
  over the batch's items to stay S5-clean: use the batch's ID rows only,
  `get_and_reset_loss()` hook). Expected effect: blocks a shared ID component;
  leaves per-item individuation untouched. **Rashomon-tunable.** Instrument-first
  recommendation: measure `‖mean e_ID‖/mean ‖e_ID‖` on trained fI′ checkpoints
  before ever stacking this.
- For D3 — *top-D PC removal escalation* (F6 Algorithm 1 step 3, applied per
  forward to the metadata block): recorded as the escalation knob; NOT recommended —
  spectral op in the forward pass, no per-field gauge meaning, complicates
  attribution (shares would decompose over PC-projected vectors).
- For D4 — *detached mean*: the already-designed `center_detach` fallback
  (epoch-refreshed buffer; exact gauge preserved at refresh points, no gradient
  coupling). This is a designed **ablation knob of the candidate itself**, not an
  extra mechanism.

*(Amendment post-5b P12 (A11), 2026-08-24 — D1 default recommendation REVERSED: the
candidate now recommends shipping gauge-fixing as part of the default configuration,
preferably in the hyperparameter-free hard form — a **post-step gauge projection**
(after each optimizer step, subtract μ_f from field f's rows: projection onto the
μ=0 slice; thereafter raw ≡ centred, all raw-row diagnostics gauge-clean, zero new
knobs). Function-inert by C2+C7 (flat directions carry no score content — this
selects a gauge, not a function), so it does not meaningfully violate R8's
one-mechanism rule. λ_pin stays the soft alternative. Rationale: leaving D1 open
corrodes the checkpoint-instrument chain every downstream cycle relies on (5b P12).
Stacking it remains the user's mechanism decision.)*
**(User review decision, Matteo 2026-08-24: gauge-snap INCLUDED in the fI′ default —
hard post-step projection form; see the decisions block in §6.)**

Inherited Rashomon-tunable knobs: `sim_proto_weight`, `sim_batch_weight` (per the
requirements §4 note); fI′ adds λ_pin/λ_id to the Rashomon set if ever stacked.

### 3.5 Baseline gate

- **Host:** `item_proto` (f0 arm) — unchanged frozen rows.
- **The isolating ablation:** fI′ vs fI, **same arms, same frozen winner configs,
  same seeds, same tier** — ids and noid both (centring interacts with the ID row's
  freedom, so both endpoints are needed): 4 compared rows (fI′-ids, fI′-noid vs
  existing fI-ids, fI-noid), hm_1_month + ml-1m. Every compared row = SLURM +
  `/leo5-submit` + C8 manifest (hard gate).
- **Flavour ablation (cheap, optional):** weighted vs unweighted `center_mode` on one
  arm.
- **LightFM control (fairness analysis, run decision = user's):** for a pure
  dot-product scorer, subtracting a shared C from all item representations changes
  scores by a per-user constant `−q_u·C` — **exactly rank-inert at inference for
  fixed parameters** (unlike on the cosine host, where C interacts through the
  denominator). A trained `lightfm_tags(_ids)`+centring arm would therefore test only
  a *training-path* effect, not a function-class effect; this asymmetry is itself the
  attribution argument (any fI′ gain is a property of the cosine/prototype route,
  which LightFM structurally cannot share). Claim goes to 4b as C-material; the
  matched arm is recorded as optional, not demanded (contrast with U1, where the
  mismatch mechanism was shared).
  *(Amendment post-5b P6 (A5), 2026-08-24: the dismissal above is inconsistent with
  §2.5 deviation 2, whose whole rationale for in-forward centring IS the
  training-path effect — conceded. The `lightfm_tags(_ids)`+centring arms are
  ELEVATED to recommended controls (2 cheap rows; run decision stays the user's):
  equal lift ⇒ generic centring benefit; disproportionate fI′ lift ⇒
  cosine/prototype-route property. C5's inference rank-inertness becomes the
  interpretive aid — any LightFM delta is training-path by construction.)*
  **(User review decision, Matteo 2026-08-24: SKIPPED — these arms are not run; see
  the decisions block in §6.)**

*(Amendment post-5b P1/P2/P4 (A1–A3), 2026-08-24 — controls and decision rule added
after the critique:)*
- **A1 — α-downweight control arm:** `t_i = α·Σ_f e(x_if) + e_ID`, α fixed (no
  tuning) to match fI′'s trained post-centring mean metadata norm. Deconfounds
  "centred direction structure" from "less metadata mass" in the ids-arm (which
  centring rebalances toward e_ID). One extra row per dataset, ids-arm only.
- **A2 — U2a-ml pre-run gate:** before any fI′ row is queued, run the U2a instrument
  on the existing **ml-1m** fI checkpoint (local/login-node class; per-token centred
  rendering + geometry stats, both/all μ flavours). The hm evidence does not license
  the ml-1m bet (bag layout, no exact-C); a null centred-geometry result on ml-1m
  sends the candidate back to the user before SLURM spend.
  **(User review decision, Matteo 2026-08-24: OVERRIDDEN — no pre-run gate; the fI′
  arms go straight to the cluster when the wave opens. The U2a-ml analysis is
  demoted to an optional interpretation aid that may run in parallel or after; it
  gates nothing.)**
- **A3 — pre-registered decision rule:** primary endpoint = **fI′-noid vs fI-noid on
  ml-1m HR@10** (D2-free, α-confound-free). Null there = the centring hypothesis
  failed on its motivating dataset — no rescue narrative (E3 etc. may explain the
  deficit, but fI′ is then not the repair). The ids-arms are secondary and are only
  interpretable with the **D2 instrument mandatory** (`‖mean e_ID‖ / mean ‖e_ID‖`
  reported for every fI′-ids row). hm arms are parity guards: losing fI's hm warm
  parity is a defeat regardless of ml-1m. Sub-bracket (≤~0.001, F-S0-14) differences
  are reported as ties per standing rules; −0.0555 is well above that resolution.

### 3.6 Deviation summary ("the paper says" vs "I propose")

F6/F7/F8 say: post-hoc global mean/PC removal fixes anisotropy (F6 Alg. 1); the
common component is frequency-driven (F6 Fig. 2, F7 §3, F8 §4); a cosine regularizer
mitigates in training (F8 Eq. 6). **I propose:** per-field means (exact-C gauge on
the fixed layout), inside the forward pass, differentiable by default with a detached
fallback, ID row excluded, no PC removal, zero new parameters, read-out re-gauged
with one disclosure line. All four deviations argued in §2.5.

## 4. Cost & feasibility

All estimates Fermi-style, working-evidence tier.

1. **Parameter delta: exactly zero.** Same (V + n_items)·d table as fI (H&M: V ≈ 426
   metadata rows + ~10⁵ ID rows; ml-1m: genre/year vocab per the C5 amendment), same
   K prototypes, same user table. The two new buffers (`field_index` V ints,
   `field_probs` F×V floats ≈ 5×426 ≈ 2k entries; sparse in effect) are
   non-persistent, not learned.
2. **Training-time: ≤ ~1.05× per step.** Extra work per forward: one (F×V)·(V×d)
   matmul for M (≈ 5·426·100 ≈ 2×10⁵ MACs — noise next to the B·N·(F+1)·d gather-sum
   and the B·N·K·d cosine block), one gather of M rows + subtraction at
   (B, N, F, d). Backward adds the P-scatter into metadata rows. Convergence:
   possibly *faster/cleaner* (F8's degeneration argument: the shared-drift component
   no longer rides every update; SIF/F6 report better geometry without retraining at
   all) — but honestly unknown; D4 (live-mean coupling) could also slow it. No
   claimed epoch change either way; `center_detach` is the fallback if dynamics
   misbehave.
3. **S5 checklist:** sampled (batch, 1+n_neg) scoring preserved — **yes** (subtraction
   is per sampled token). Full-catalog pass per step — **none** (M touches the
   *vocabulary* table (V rows), not the item catalog; V ≈ 426 is metadata-sized, the
   ID block is never reduced over). Item branch still cheap indexed computation —
   **yes** (gather + one broadcast subtract). Losses fit `get_and_reset_loss()` —
   **unchanged** (no new loss in the candidate; the unstacked λ_pin/λ_id designs hook
   there if ever adopted). Precompute/cache — `center_detach` mode caches M per epoch;
   differentiable mode needs no cache.
4. **Budget gate: pass with wide margin.** 4 compared rows (fI′ ids/noid × hm_1_month,
   ml-1m) at the frozen fI winner configs = same wall time as the existing fI runs
   (dev-tier fleet runs were hours-class each on LEO5); optional flavour ablation +1
   row. Nothing approaches the 1-day soft bar per run. Smoke: one end-to-end
   login-node A30 instance before any wave (standing rule), VPN required.
5. **Flags:** none hard. Two soft notes — (i) bag-layout (ml-1m) claims must be
   stated in the weaker per-token form (§3.1 caveat); (ii) the U2a instrument used
   *post-hoc* centring on trained-raw checkpoints — the trained-centred geometry may
   differ in either direction; that is exactly what the runs test (the R6 bet is
   open, not banked).
6. *(Amendment post-5b P11 (A10), 2026-08-24:)* third soft flag — the frozen fI
   winner configs' `sim_batch_weight` was selected under **raw** geometry, partly
   compensating the very collapse centring targets; a fI′ result in either direction
   carries this interaction. The queued sim_batch_weight isolating rerun remains the
   primary collapse experiment; a small dev-tier `center × sim_batch_weight` 2×2 is
   the designed follow-up if fI′ shows any effect (unstacked, user's call).

## 4b. Math sanity check

**Claims spec:** `Master/temp/dc_checks/dc06/claims_spec.md` (mechanism definition +
claims C1–C10 only; no derivations, no expected outcomes). Claims cover: C1 exact
shared-C (fixed layout, both flavours); C2 per-field gauge invariance incl.
gradients; C3 exact share decomposition; C4 cold-path non-degeneracy + twin
behaviour; C5 the rank-inertness split (dot host inert / cosine host not); C6 live-
vs detached-mean gradient reach; C7 gauge-pin loss properties; C8 ID-mean-pin loss
properties; C9 bag-layout weaker form (counterexample demanded); C10 shared
component not activation-inert on the cosine host (the E2 seed claim).

**Viability self-assessment (per claim):** C1–C3, C5, C7–C10 are exact linear-algebra
identities or explicit counterexample constructions — fully toy-checkable. C4 is
toy-checkable in its structural part (nonzero, differing, twins identical); the word
"meaningful" (does the cold activation *help ranking*) is a training-dynamics
question, deliberately not claimed. C6 is a two-line autograd check. **Skipped (with
justification):** the R6 accuracy bet (does trained-centred fI′ close the ml-1m
deficit / hold hm parity) and the in-training collapse-reduction hypothesis (does
eff. rank stay high when the optimizer *trains* against centred geometry) — both
materialize only through full training; the U2a instrument is the strongest
pre-run evidence available and the SLURM arms are the real test. Also not
toy-claimed: D4 stability (live-mean feedback) — dynamics-dependent; the
`center_detach` fallback exists precisely for it.

**Black-box execution:** subagent received ONLY the claims spec; script
`Master/temp/dc_checks/dc06/t01_claims_check.py`. Report (verbatim) below.

> **dc06 black-box toy claims check — verdicts**
>
> Script: `/home/mattik01/Desktop/githubs/ProtoMF/Master/temp/dc_checks/dc06/t01_claims_check.py` (torch float64; d=8, F=3, vocabs=[4,3,5], n=10 items, K=4 prototypes, U=5 users; exact-identity tolerance 1e-9 — observed residuals were ≤ ~4e-15, i.e. float64 rounding — nonzero threshold 1e-6). The toy catalog deliberately contains two zero-frequency vocabulary values (field 0 value 3, field 2 value 4) and a twin item pair (items 0, 3) to probe edge conditions.
>
> - **C1 — CONFIRMED.** For both μ flavours, max|t_i − (t_raw − C)| = 8.9e-16 and the per-item subtracted vector's spread across items is ≤ 8.9e-16: the subtracted vector is a single shared C = Σ_f μ_f.
> - **C2 — CONFIRMED.** Adding c = 3.7·**1** to every embedding row of field 1 leaves every centred t_i unchanged to 8.9e-16 (both flavours) and leaves the gradients of a nonlinear loss on centred t_i w.r.t. all other parameters (other fields' tables, ID table) unchanged to ≤ 6.2e-15, while t_raw shifts by exactly 3.7 per coordinate.
> - **C3 — CONFIRMED.** max residual between t_iᵀp_k and the sum of per-field shares plus the ID share is 3.6e-15 over all (i,k) — exact up to float64 rounding, no structural residual.
> - **C4 — CONFIRMED.** ‖t_cold‖ = 4.29 (nonzero); two cold items with different field values differ by norm 8.41; the twin items sharing all field values have identical t_cold both centred and raw (max diff 0.0 exactly).
> - **C5 — CONFIRMED.** Dot host: replacing t_i by t_i − C left every user's ranking identical and preserved all pairwise score gaps to 4.4e-15. Cosine host: the same replacement changed the ranking for 5/5 users; concrete counterexample — user 0's ranking [8,0,**6,3,2**,5,4,9,1,7] became [8,0,**3,2,6**,5,4,9,1,7] (items 6/3/2 reorder).
> - **C6 — CONFIRMED (with stated condition).** Live μ: gradient of a loss on item 0's centred t is nonzero (up to 1.9e0) on non-carried same-field rows with positive catalog frequency; detached μ: exactly 0.0 on all of them. Condition the spec does not state: in the **weighted** flavour, a vocabulary value with zero catalog frequency gets p_f(v)=0 and hence exactly zero live-mean gradient (measured 0.0 on the two unused values) — the universal "nonzero" reading is falsified for such rows; the unweighted flavour reaches all rows (1.24e0 on the unused values).
> - **C7 — CONFIRMED (with stated condition).** L_pin = 6.50 ≥ 0 for generic embeddings; shifting each field's table by −μ_f drives L_pin to 4.9e-32 and makes centred and raw compositions coincide to 4.4e-16, and L_pin = 0 iff every weighted mean is zero (grad_v = 2·p_f(v)·μ_f). Condition: "gradient reaches every metadata row" holds only for rows with positive catalog frequency — zero-frequency rows get exactly 0.0 gradient.
> - **C8 — CONFIRMED.** L_idpin = 1.88 > 0 for a generic 4-item batch, drops to 2.7e-32 after removing the batch mean (zero iff zero mean); the gradient is bitwise identical across batch ID rows (max row-to-row difference 0.0), so every pairwise difference e_ID(i) − e_ID(j) receives exactly zero net gradient; off-batch rows get 0.0.
> - **C9 — CONFIRMED.** Counterexample to any shared-C claim in the bag layout: item A with one field-0 token (subtracted mass μ_0) vs item B with three field-0 tokens at w=1 (mass 3μ_0) differ by norm 3.83; a reweighted variant (w=0.5 each) differs by 0.96. With exactly one token per field at weight 1 the subtracted mass equals the shared C to 0.0 exactly.
> - **C10 — CONFIRMED.** Adding the shared C to all compositions changes cosine-host activations by up to 0.371, and item-dependently: witness items i=6, j=7, prototype k=0, where the activation gap t*_k(i) − t*_k(j) moves from −0.7966 to −0.5053 (change 0.2913), so the shared component is not activation-inert on the cosine host.
>
> Net: 10/10 claims confirmed as stated, with one unstated boundary condition affecting C6 and C7 alike — **zero-catalog-frequency vocabulary values are invisible to the weighted mean** (p_f(v)=0 ⇒ zero gradient through μ_f and through L_pin), so "reaches every row"/"nonzero on all non-carried rows" holds only for values that actually occur in the catalog; the unweighted flavour has no such blind spot.

**Take-away:** 10/10 confirmed; no falsifications to carry into Step 5 as defects.
One boundary condition surfaced by the check (not in my spec): **zero-frequency
vocabulary values are invisible to the weighted mean** — relevant to H&M because the
vocab is built from the catalog (zero-frequency rows are rare but possible on the
cold/derived variants); carried into §5 as a disclosure item, not a defect (such a
row also contributes to no composition, so its raw offset is score-inert anyway).

## 5. Requirements evaluation

Requirements doc re-read this step (2026-08-24 state, incl. the R1 clarification and
R6 staging notes). No Step-4b falsifications exist to fold in; the one 4b boundary
condition is disclosed under R4/S5 below.

*(Amendment post-5b P5/P9 (A4), 2026-08-24 — framing correction applying to this
whole file, incl. §0/§1 headline language: fI′ is an **attribution-gauge fix and a
score-side intervention** — two proven statements about two different maps (C2/C3 vs
C5/C10), never a score no-op. "Canonical" refers ONLY to the within-field share
gauge (the sI-identified one), not to isotropy — D3's residual anisotropy is
untouched (post-centring eff. rank ~2 is still collapse). Any earlier phrase
suggesting a pure/no-op gauge for the model itself is superseded by this note.)*

**A function-class honesty note first** (feeds several rows): centring is a pure
gauge for the *attribution* (C2/C3), but **not** a pure gauge for the *score* — C is
not activation-inert on the cosine host (C10/C5). The centred model's metadata
composition is constrained to have zero catalog-mean; in the ids-arm that lost
freedom can migrate into the ID rows (mode D2), so the *warm* function class is
nearly unchanged, while the **cold/noid function class is strictly smaller** (no
shared-component channel at all). fI′'s bet is that this removed channel was carrying
collapse subsidy, not ranking signal (U2a evidence + F6/F7/F8 mechanism); that bet is
what the runs test.

| Req | Rating | Assessment |
|---|---|---|
| R1 | **strong** | Inherited from dc01 unchanged: single-branch host, the centred composition IS the representation driving the score; nothing parallel, nothing post-hoc. Centring happens inside the forward pass, so training and explanation see the same geometry — if anything the R1 story tightens (the read-out gauge is now the model's own arithmetic, §3.3). The weight-sharing exemplar stays a merge-stage object (2026-07-12 clarification). |
| R2 | **strong** (dc01's SC.10 grade re-scope inherits) | Every rendered number remains a forward-pass quantity; the new disclosure line quotes C, itself a model quantity computed in the forward. The U2a *post-hoc* gauge rendering — a declared read-out choice on fI — is exactly what fI′ eliminates in favour of intrinsic arithmetic: the R2 delta vs dc01 is a small but genuine improvement. dc01's SC.10 re-scoped R2 wording (what "intrinsic" may claim on this host) transfers verbatim. |
| R3 | **strong** | Untouched: explanations stay "which prototype, what it means, per-word shares". Centring changes the gauge prototypes are read in, not the shape of the explanation. |
| R4 | **partial** (as dc01; legibility improved, enforcement unchanged) | Grounding remains *unenforced/emergent* shared geometry — fI′ deliberately does not move on the enforcement spectrum (Step 1b). What improves is profile *legibility*: U2a showed the raw profiles' dominant structure was the C artifact (`Solid` share 0.315→0.022, eff. rank 1.22→1.98); centred profiles read as within-field contrasts, which is also the canonical (sI-identified) gauge. Disclosure item from 4b: zero-catalog-frequency vocabulary values sit outside the weighted mean and receive no live-mean gradient — their profile rows must be marked "not catalog-grounded" if ever rendered (they contribute to no item, so nothing scores through them). |
| R5 | **partial→strong** (dc01-equal, one structural improvement) | Cold composition is non-degenerate and twin behaviour is unchanged (4b C4); the F-S0-14 bracket machinery and cold-vs-all disclosure transfer as-is. The structural improvement: cold items are now scored in the *same shared-component-free gauge* as warm metadata compositions, and the cold path is exactly where the removed channel bound (see the honesty note) — if the collapse-subsidy reading is right, cold discrimination should improve; that is an open empirical bet, not a banked gain. The E4 operating-point mismatch (ID deletion) is NOT addressed — that is U1's deferred territory, disclosed. *(Amendment post-5b P10 (A9), 2026-08-24: one stated open risk added — a cold item carrying a train-catalog-zero attribute value sums an untrained, never-centred raw row at full strength under the `'weighted'` flavour (the 4b C6/C7 blind spot); for such items the "same gauge as training" claim does NOT hold. Designed unstacked mitigations: smoothed weights (§3.1 A9 note) or the `'unweighted'` flavour. The earlier §4b "not a defect" note covered warm compositions only.)* |
| R6 | **open bet — the candidate's main motivation** | Stage bar per the staging note: variant-vs-its-own-host (and vs fI), primary UI-ProtoMF bar unassessed-deferred. fI′ exists to attack fI's ml-1m −0.0555 host deficit (candidate mechanism: E2 collapse pressure) and must hold fI's hm warm parity. Honest risks: (i) the deficit is *unattributed* — E3 (conjunctions) is a rival explanation centring cannot touch; (ii) trained-centred ≠ post-hoc-centred geometry (Step 4 flag ii); (iii) the D2 migration could neutralize the intended effect in the ids-arm (making noid the cleaner test). |
| R7 | **partial** (dc01-equal, displays more honest) | Vocabulary unchanged (catalog fields, mixed user-visibility). Centred shares stop the omnipresent-value pollution (`Solid` in 58% of the catalog dominating displays) — the rendered explanation moves closer to "what distinguishes this item", which is the user-relevant content. The taxonomy-vs-perceivable tension itself is untouched. |
| R8 | **strong** | One linear subtraction, zero new parameters, one on/off switch; ablation arms are the existing endpoints plus fI′. The constraint candidates (λ_pin, λ_id, PC removal) are designed but explicitly unstacked. Comfortably one idea at the reference granularity. |
| S1 | **n-a / served by existing baselines** | The decoupled contrast stays `lightfm_tags(_ids)`; the C5 rank-inertness result sharpens the contrast argument — a dot-product decoupled model *cannot* express the inference-time effect centring has on the cosine host, so any fI′-specific delta is prototype-route-specific by construction. ~~Matched centred-LightFM arm recorded as optional (training-path-only effect).~~ *(Cold-read fix, 2026-08-24: superseded by amendment A5 — the centred-LightFM arms are RECOMMENDED controls; see §3.5.)* |
| S2 | **n-a (item-side candidate)** | Does not preclude the user side; the fU mirror (centring the basket composition) is captured to the scratchpad this cycle (Step 6) rather than pursued (R8). Bag-layout caveat applies there with full force (variable baskets → per-token form only). |
| S3 | **n-a** | No interaction with image channels; unchanged from dc01. |
| S4 | **compatible, mildly helped** | LLM naming consumes prototype profiles; centred profiles are cleaner input (the `Solid`-artifact rows no longer dominate every prototype's top words — U2a). No mechanism contact. |
| S5 | **strong** | Full checklist pass in §4: sampled scoring preserved, no catalog pass (V-sized vocab reduction only), indexed lookups intact, no new losses; `center_detach` is the cache fallback. |

### §4 design tensions

- **Where features enter:** unchanged from dc01 (features→factors composition);
  centring is a within-slot gauge fix, not a new entry point — the §I wedge and
  fingerprint region stay dc01's.
- **Fidelity vs. accuracy:** unusually favourable configuration — attribution
  fidelity strictly improves (canonical gauge, C2/C3) while the score-side cost is
  the removal of one channel argued to be noise-for-ranking (C10 shows it *was*
  score-relevant on this host, so this is a real bet, quantified by the fI′-vs-fI
  arms; the honesty note above states the function-class asymmetry).
- **Attribution granularity:** unchanged (per-field word + ID residual).
- **R7 perceptual alignment:** unchanged vocabulary; improved display honesty (see
  R7 row).
- **Identifiability:** the headline win — the sI result (per-field offsets =
  exactly the unidentified directions of attribute profiles) is resolved *in
  mechanism*: within-field shares become canonical, and D1 turns the offset into an
  exactly score-inert direction (C2). Remaining non-identifiability (cross-field
  clique structure, F-DC01-06) is untouched and inherited.

### Part-prototype pitfalls checklist (reading-list S5)

- *Prototype quality:* improved legibility of profiles (C artifact removed); quality
  enforcement still emergent — unchanged risk class.
- *Prototype quantity:* K untouched; no new pressure either way.
- *Prototype collapse / duplication:* the candidate's target — but only the
  **activation-geometry** side (eff. rank); **duplication** among prototypes is a
  separate phenomenon U2a showed centring does not fix (stays with
  `sim_batch_weight`; the queued isolating rerun remains the primary collapse
  experiment).
- *Semantic gap:* words are catalog metadata; gap unchanged (centring renames
  nothing).
- *Perceptual mismatch (this-looks-like-that gap):* n-a (no visual prototypes).
- *Attribution leakage / unfaithful saliency:* shares remain exact numerator
  decompositions (C3); the Steck P7 comparative-reading caveat inherits unchanged.
- *Cold/degenerate activations:* non-degenerate (C4); chance-floor quoting rules
  (F-S0-15) inherit.

### Interpretable-AI placement

Skipped — the principle framework (ideas §4) remains undecided.

### 4c. Degeneration-protection gate

| Mode (Step 3) | Status |
|---|---|
| D1 flat-direction drift (raw rows non-identified up to per-field constant) | protected by proposed (unstacked) constraint — λ_pin gauge pin; until stacked: open but **score-inert by construction** (C2), affects raw-row diagnostics only |
| D2 common-direction re-emergence in ID block (ids-arm) | **unprotected — open risk**; instrument-first plan recorded (`‖mean e_ID‖/mean‖e_ID‖` on trained fI′ checkpoints); proposed (unstacked) λ_id pin designed with verified properties (C8) |
| D3 residual anisotropy beyond the mean (top-PC directions) | **unprotected — open risk**; F6-step-3 escalation knob recorded, not recommended; `sim_batch` fights the symptom in activation space (partial mitigation, config-dependent per the 2026-08-17 probe) |
| D4 live-mean feedback dynamics | protected by designed fallback knob — `center_detach` (candidate's own ablation switch); plus the 4b C6 boundary note (weighted mean never couples zero-frequency rows) |
| (inherited) prototype duplication | protected by inherited regularizer — `sim_batch` route, explicitly out of fI′'s claim scope |

## 5b. Devil's advocate

Subagent received ONLY this candidate file. Critique **verbatim**:

> The critique follows, based solely on the candidate document.
>
> 1. The empirical bet can win for the wrong reason, and the design has no control to catch it. By the document's own numbers, C is roughly 85% of the mean metadata composition on fI. Subtracting it therefore removes the bulk of the metadata sum's mass, leaving the (uncentred) e_ID row relatively dominant in the ids-arm. That is not a subtle gauge fix — it is a large step back toward the plain item_proto host, i.e. toward exactly the model that already beats fI by 0.0555 on ml-1m. If fI′-ids closes the ml-1m deficit, the most parsimonious reading is "less metadata influence is better on ml-1m," not "the collapse subsidy was removed." The obvious cheap control — a scalar down-weighting arm t_i = α·Σ_f e(x_if) + e_ID with α tuned or set to match the post-centring metadata norm — appears nowhere in §3.5. Without it, the candidate's headline success is indistinguishable from a confession that feature composition hurts on ml-1m.
>
> 2. The evidence and the target live on different layouts. Every load-bearing claim — exact shared C, the "stronger than SIF" per-field gauge, the disclosure line, the U2a instrument results (C ≈ 85%, eff. rank 1.218→1.979, Solid 0.315→0.022) — is stated on the H&M fixed one-value-per-field layout. But the candidate's declared main motivation (R6, the −0.0555 deficit) is on ml-1m, which is the bag layout, where §3.1 concedes the mechanism degrades to "frequency-direction removal per token" with item-dependent removed mass and no exact-C property. No U2a-style instrument evidence is cited for ml-1m at all. So the candidate is instrument-validated where it doesn't need to win (hm shows parity) and unvalidated where it stakes its existence. The document quietly acknowledges this split ("claims differ per layout") and then proceeds as if the hm evidence licenses the ml-1m bet.
>
> 3. The U2a evidence is geometry statistics, not ranking evidence — and its own numbers undercut the story. Post-hoc centring lifting effective rank from 1.22 to 1.98 sounds decisive until you notice (a) 1.98 in a ≤100-dimensional space is still catastrophic collapse — the "fix" moves the representation from rank-one to rank-two; and (b) the document itself reports noid sits at 1.93, i.e. centred fI merely matches the geometry of an arm that, as far as this document shows, does not resolve the ml-1m deficit either (if noid did resolve it, the honest headline would be "drop the features," not "centre them"). Nowhere is any correlation between effective rank and ranking quality established. The entire mechanism-level case rests on an aesthetic geometry metric plus an NLP analogy in which the gains were on similarity tasks, not retrieval under a learned scorer.
>
> 4. The candidate admits its target is unattributed, then bets on it anyway — and a loss is undetectable. R6 row, "honest risks": the ml-1m deficit is unattributed, with E3 (conjunctions) a rival explanation centring cannot touch; trained-centred geometry may differ from post-hoc-centred in either direction; and D2 (common-direction re-emergence in the ID rows) "could neutralize the intended effect in the ids-arm." D2 is explicitly listed as unprotected, the λ_id pin explicitly unstacked, and the D2 instrument only "recommended." So if fI′-ids shows nothing, there are at least three prepared exculpatory stories (E3 was the real cause; trained ≠ post-hoc; D2 ate the effect) and no pre-registered decision rule to distinguish failure of the hypothesis from failure of the arm. A design where every negative outcome has a ready excuse and the positive outcome has a ready confound (point 1) is not a test; it is a narrative generator.
>
> 5. Internal inconsistency: the "gauge" framing versus the candidate's own math. The title, §0, and §5 headline sell centring as "the canonical gauge fix," but C5 and C10 prove it is not a gauge on this host: it reorders rankings for 5/5 toy users and shifts activations by up to 0.371 item-dependently. The document knows this — the "function-class honesty note" says so — yet the rhetorical superstructure (canonical, gauge-fixed, pure Axis A, "arguably simpler semantics") keeps the no-op framing alive everywhere the honesty note isn't. You cannot simultaneously claim "we merely quotient out an unidentified direction" and "this changes accuracy on the cosine host, which is the point." Pick one; the requirements table currently rides both.
>
> 6. The same inconsistency guts the LightFM control dismissal. §2.5 deviation 2 stakes the candidate's entire in-forward design on training-path effects: post-hoc centring exists (F6/F7), so the only reason to move the subtraction into the forward pass is that the optimizer sees centred geometry. §3.5 then waves off the matched centred-LightFM arm because it "would test only a training-path effect." That is precisely the effect the candidate claims to introduce. Declaring the control's relevance a virtue of its absence ("this asymmetry is itself the attribution argument") is the document's most brazen move: the one arm that would separate "cosine/prototype-specific mechanism" from "generic training-dynamics benefit of centring feature tables" is demoted to optional by an argument that only covers inference-time rank-inertness, not training.
>
> 7. The novelty claim rests on four self-authored web queries and ignores the nearest field. "No published work applies mean-centring inside the forward composition of feature-based entity embeddings in a recommender" — the CTR literature routinely applies in-forward normalization (batch/layer/field-wise normalization) to categorical feature-embedding layers, which is in-training, differentiable mean removal over feature embeddings; none of the probe queries would surface it because they were phrased in the candidate's own vocabulary ("all-but-the-top," "gauge," "effective rank ProtoMF"). The residual novelty surface — per-field rather than global means, exact-C on a one-value-per-field layout, a disclosure line on an explanation card — is adaptation minutiae. Stripped of the gauge-theoretic language, the mechanism is: subtract a weighted column mean from a 426-row embedding table. That is a footnote in an ablation section, not a design candidate, and the elaborate protocol apparatus around it (10 toy claims, four degeneration modes, three unstacked constraints) reads as ceremony compensating for mechanism thinness.
>
> 8. The frequency-weighting borrow is misapplied. SIF's and F6's corrections are driven by usage frequency — how often a word occurs in the data the model trains on. The proposed μ_f is weighted by catalog composition (P[f,v] = frequency of value v among the ~10^5 items), not by transaction frequency. In a sparse-transaction regime with heavy popularity skew, the catalog distribution and the interaction distribution over attribute values can diverge substantially; the direction the optimizer actually inflates (per the F8 mechanism the document cites — gradient pressure from training examples) is the interaction-frequency direction. The document nonetheless claims the weighted mean "estimates precisely the frequency/popularity direction." It estimates the catalog direction. That the U2a post-hoc probe found it "effective" on one geometry statistic does not license the theoretical identification, and no interaction-weighted flavour is even listed as an ablation.
>
> 9. The explanation-improvement claims are wishful. The rendered card turns absolute shares into signed contrasts against a frequency-weighted mean of a merchandising taxonomy field — "Colour 'Dark Blue' (vs field average) +0.31" — where the "field average" is an unobservable synthetic vector, shares can now be negative, and fields with 100-value vocabularies and 3-value vocabularies are contrasted against incomparably concentrated means. The closing line "Shared catalog component removed from all items: C" is meaningless to any end user and exists to satisfy the author's bookkeeping, not R7. Calling this "arguably simpler semantics" inverts the truth: raw additive shares are the simple object; contrastive shares in a model-internal gauge require exactly the explanation the card cannot give. And "canonical" overclaims even internally: D3 concedes the gauge fix is first-order only, with shared principal directions untouched — so the profiles are canonical up to whatever anisotropy remains, which by the candidate's own eff. rank ~2 is most of it.
>
> 10. The cold path inherits an undisclosed failure mode from the C6/C7 blind spot. The check surfaced that zero-catalog-frequency values are invisible to the weighted mean; §4b dismisses this because "such a row contributes to no composition." But the cold/derived variants are precisely where items can carry attribute values that were zero-frequency in the training catalog — the document says so ("rare but possible on the cold/derived variants") — and for such an item the cold composition t_cold = Σ_f (e(x_if) − μ_f) sums an untrained, never-centred, never-gradient-touched raw row at full strength into a representation the candidate claims is "scored in the same shared-component-free gauge." R5 is graded "partial→strong" on the back of that gauge-consistency claim, while the one identified mechanism by which cold inputs escape the gauge is filed as a rendering disclosure ("mark the profile row 'not catalog-grounded'"), not as the cold-scoring risk it is.
>
> 11. The evaluation cannot support either verdict at the stated stakes. Four dev-tier rows on frozen fI winner configs: the hyperparameters (including sim_batch_weight, chosen under raw geometry to fight the very collapse centring claims to remove) are reused unchanged, so a null result may mean "config mismatch" and a positive result may mean "the frozen regularization plus centring double-counts" — equal treatment of configs is not equal treatment of mechanisms when the mechanism changes what the config was compensating for. No noise floor, seed variance, or minimum detectable effect is quoted anywhere against the 0.0555 target, the two-tier policy pre-declares the numbers non-thesis-grade, and the R6 row admits the actual bar (UI-ProtoMF) is "unassessed-deferred." The candidate's central requirement will therefore be adjudicated by an experiment whose resolution is unquantified against an effect size that is the entire point.
>
> 12. D1 is left open and it corrodes the project's own instrument chain. With centring live and λ_pin unstacked, raw table rows random-walk along F·d score-inert directions. The document notes this makes "raw-row diagnostics (U3-style norm read-outs, cross-run table comparisons) gauge-noisy" — but U2a-style checkpoint instrumentation and cross-run geometry comparison are this program's primary evidence-generating tools (this very candidate was justified by one), and a captured entry already warns the fI-vs-fU Procrustes comparison breaks on uncentred tables. Shipping the default configuration with the diagnostic-corrupting mode open, while the two-line fix is designed, verified (C7), and deliberately withheld for an R8 purity argument, means the candidate degrades the measurement apparatus of every downstream cycle to keep its own "one idea" ledger clean.

### Response (concede / rebut / open risk, point by point)

**P1 — metadata-downweight confound. CONCEDE.** The strongest point. Removing ~85%
of the mean metadata mass while leaving e_ID raw does shift the metadata/ID balance
in the ids-arm, and a win there would be ambiguous between "centred geometry" and
"less metadata". The proposed control is cheap and sharp: an
**α-downweight control arm** `t_i = α·Σ_f e(x_if) + e_ID`, α fixed to match fI′'s
post-centring mean metadata norm (measured from the fI′ checkpoint; no tuning) — same
metadata/ID balance, direction structure left raw. Amendment A1 below adds it to
§3.5. Note the noid-arm is confound-free on this axis (no ID row to rebalance
toward) — another reason noid is the primary endpoint (see P4).

**P2 — layout split. CONCEDE the evidence gap; the fix is cheap and gates the
wave.** U2a ran on hm checkpoints; the ml-1m claim is currently mechanism-derived
(per-token frequency-direction removal) with no instrument behind it. Amendment A2:
**U2a-ml gate** — run the same instrument on the existing ml-1m fI checkpoint
(local/login-node class) *before* any fI′ arm is queued; if centred geometry shows no
improvement there, the R6 bet loses its mechanism support on the dataset that
motivates it and the user should re-scope before spending SLURM rows. This converts
the critic's "proceeds as if licensed" into an explicit pre-run gate.

**P3 — geometry ≠ ranking; 1.98 is still collapse. CONCEDE the epistemics, with one
correction.** Correct as stated: no eff.rank↔ranking correlation is established, 1.98
is rank-two, and the NLP gains are similarity-task gains. This is exactly why fI′ is
a *designed arm to be run*, not an adopted default — the candidate's claim is "the
mechanism-level seed is real and cheap to remove", not "removal improves HR@10". One
correction to the critic's inference: the noid-geometry comparison does not imply
"drop the features" is the honest headline — noid *also* drops warm ID individuation
(twin q-cos 1.0) and is the cold winner already; the thesis question is whether ids'
warm capacity can be kept without fI's geometry pathology. Open risk stands as
written in R6.

**P4 — narrative generator / no decision rule. CONCEDE; pre-registering now.**
Amendment A3 fixes the reading grid *before* any run: (i) **primary endpoint:
fI′-noid vs fI-noid, ml-1m HR@10** (D2-free, α-confound-free); a null there =
**the centring hypothesis failed on its motivating dataset** — E3 or anything else
may explain the deficit, but fI′ is then not the repair, full stop, no rescue
narrative. (ii) ids-arm is secondary and only interpretable with the **D2 instrument
mandatory** (`‖mean e_ID‖/mean‖e_ID‖` reported for every fI′-ids row — measured, not
invoked). (iii) hm arms are parity guards: any warm-parity loss on hm is a defeat
regardless of ml-1m. This is the decision rule; deviations from it at results time
are what the scrutiny protocol exists to catch.

**P5 — gauge language rides two horses. CONCEDE the language, REBUT the alleged
contradiction.** Two different objects: centring is an *exact* gauge for the
**attribution/read-out** (C2/C3 — within-field shares, profile rendering) and a
*real intervention* on the **score function** (C5/C10). Both statements are proven,
about different maps; there is no mathematical inconsistency. But the critic is right
that headline phrases ("canonical gauge fix", "pure Axis A") let the no-op
connotation leak into score territory. Amendment A4 re-words the frame everywhere it
appears: **"attribution-gauge fix, score-side intervention"** — and "canonical"
is restricted to the within-field share gauge (P9's D3 point conceded: no isotropy
claim).

**P6 — LightFM control dismissal. CONCEDE.** The critic catches a real asymmetry in
my §3.5 argument: the candidate's own §2.5-deviation-2 rationale is training-path,
so "the LightFM arm only tests training-path effects" is an argument *for* it, not
against. Amendment A5 elevates the `lightfm_tags(_ids)`+centring arms from optional
to **recommended controls** (2 cheap rows; run decision stays the user's): equal
lift ⇒ generic centring benefit; disproportionate fI′ lift ⇒ cosine/prototype-route
property. The inference-time rank-inertness (C5) remains true and becomes the
*interpretive* aid (any LightFM delta is training-path by construction), which is
the correct use of that result.

**P7 — novelty probe blind spot; "footnote" charge. PARTIAL CONCEDE / REBUT.**
Conceded: the queries were phrased in my own vocabulary, and the CTR normalization
family is the nearest unexamined field. Distinguishing it on the merits (amendment
A6 adds this to §2b): LayerNorm on embeddings subtracts the *per-vector mean across
dimensions* (a different axis entirely); BatchNorm-style embedding normalization
subtracts a *per-dimension batch mean of composed vectors* — stochastic,
interaction-weighted, includes the ID row, rescales variance, and carries no
per-field structure, no exact-C, no attribution contract (and is precisely what the
NOVA-BERT/DIF-SR line found invasive on side-info fusion). No CTR normalization
yields identified per-field profiles — the adaptation stands, *as an adaptation*.
REBUT the "footnote, not candidate" charge on process grounds: fI′ was **sized as a
knob by explicit user decision** (single-variable, no new ft_type); the protocol
apparatus is the standard treatment every candidate gets, deliberately including
small ones — parsimony is R8's virtue, not evidence of thinness. The thesis-narrative
consequence is real though, and P7 states it well: fI′ is presented as a **repair of
dc01's composition**, not a headline mechanism.

**P8 — catalog vs interaction frequency. CONCEDE; sharp and correct.** The F8 drift
mechanism is exposure-driven (interaction-weighted); my P matrix is
catalog-composition-weighted; they coincide only insofar as attribute-value
interaction shares track catalog shares. The §2.3 "estimates precisely the
frequency/popularity direction" sentence overclaims — amendment A7 corrects it
("estimates the catalog-composition direction; the exposure direction is
interaction-weighted and may diverge") and registers a third `center_mode` flavour
`'interaction'` (μ_f weighted by train-interaction counts over values — still a
per-field constant, so exact-C is preserved by the C1 argument; unstacked ablation
flavour, one buffer swap). Which flavour wins is an empirical sub-question the U2a-ml
gate (A2) can pre-screen for free.

**P9 — explanation claims wishful. PARTIAL CONCEDE.** Conceded: contrastive shares
are a *different* object, not automatically simpler; negative shares and
heterogeneous field cardinalities are real card-design problems; the C-disclosure
line serves the thesis reader's fidelity contract, not the end user (mislabeled in
§3.3 as a card line); and "canonical" is hereby restricted per P5/A4. Retained: the
raw display's dominant structure was demonstrably an artifact (`Solid` 0.315 mean
share for a value carried by 58% of the catalog — a share display saying "this item
activates dark-denim because it is Solid, like everything else" misleads), so
*something* must give; whether the fix is mechanism-gauge (fI′) or display-gauge
(U2a rendering) is exactly the **thesis-time presentation decision** already
deferred with U3 — fI′ does not force the contrastive card; it makes the contrastive
gauge the model's own arithmetic if chosen. Amendment A8 marks the card mock-up in
§3.3 as one design option, decision deferred.

**P10 — cold zero-frequency escape from the gauge. CONCEDE; elevate to R5 open
risk.** Correctly caught: a cold item carrying a train-catalog-zero value sums an
untrained, never-centred raw row at full strength — under the weighted flavour that
row sits outside the gauge entirely, and my §4b note ("contributes to no
composition") covered only *warm* compositions. Note the risk differential vs fI is
narrow (in raw fI the same row is equally untrained — the *delta* is only the
uncentred offset), but the "same gauge as training" claim in §3.2/R5 does not hold
for such items. Amendment A9: R5 row gains this as a stated open risk; two designed
(unstacked) mitigations recorded — smoothed weights `p̃_f(v) ∝ count+ε` (SIF-form
`a/(a+p)` precedent; every row enters μ and receives (tiny) live-mean gradient), or
the `'unweighted'` flavour (no blind spot by construction, 4b-verified).

**P11 — frozen-config reuse / resolution. PARTIAL CONCEDE, PARTIAL REBUT.** The
frozen-config equal-treatment design and the two-tier policy are standing user
decisions (S0-era), and dev-tier working evidence is *by declaration* not
thesis-grade — the critic restates the policy as if it were an oversight. Conceded
substance: (i) `sim_batch_weight` was selected under raw geometry, and centring
changes what that knob was compensating — the interaction is real; the already-queued
sim_batch_weight isolating rerun is the designated primary collapse experiment, and
amendment A10 flags a small dev-tier `center × sim_batch_weight` 2×2 as the designed
follow-up **if** fI′ shows any effect (unstacked, user's call); (ii) no MDE is
quoted — honestly, none exists for single-seed dev rows; the F-S0-14
bracket/tie-language discipline is the standing mitigation, and −0.0555 is an
order of magnitude above the bracket widths (≤~0.001) seen at SC.8, so the primary
endpoint is not resolution-limited; sub-bracket differences will be reported as
ties per standing rules.

**P12 — D1 corrodes the instrument chain. CONCEDE; recommendation changed.** The
critic is right that leaving the diagnostic-corrupting mode open to keep the "one
idea" ledger clean is a false economy — especially since the fix is *function-inert*
(C2+C7: flat directions don't touch the score; pinning them selects a gauge, not a
function — so stacking it does not violate R8's "one mechanism" in any meaningful
sense). Amendment A11: the candidate now **recommends shipping gauge-fixing as part
of the default**, in the *hyperparameter-free* form in preference to λ_pin: a
**post-step gauge projection** — after each optimizer step, subtract μ_f from field
f's rows (hard projection onto the μ=0 slice; raw ≡ centred thereafter, diagnostics
gauge-clean, zero new knobs, no Rashomon surface; λ_pin remains the soft
alternative). Per protocol this stays the user's mechanism decision — but the
recommendation is now to include it, reversing §3.4's default.

**Net effect on the candidate:** no mechanism change; four wording corrections
(A4/A7/A8 + §4b note scope), one pre-run gate added (A2 — U2a-ml), three
control/ablation arms elevated or added (A1 α-control, A5 LightFM-centred, A7
interaction flavour), one decision rule pre-registered (A3), one risk elevation
(A9), one default-configuration recommendation reversed (A11). All amendments are
dated edits in the affected sections, marked "post-5b".

## 6. Fingerprint & registration

Mechanism fingerprint (3 axes):
1. **Features-entry:** dc01's features→factors sum, with the composition
   **gauge-fixed in-forward** — per-field-mean-centred summands
   (`t_i = Σ_f (e(x_if) − μ_f) + e_ID`); exact shared-C removal on the fixed layout,
   per-token frequency-direction removal on the bag layout. *(Distinctness from dc01
   on this axis waived by user directive — directed cycle, dc05-M1 precedent; the
   delta is the gauge, not a new entry point.)*
2. **Grounding:** unchanged from dc01 — unenforced shared-geometry cos(e_f, p_k) —
   but read in the **centred (canonical within-field) gauge**, which U2a showed
   removes the dominant rendering artifact.
3. **Read-out:** dc01's read-outs unchanged in form; per-word shares become
   within-field contrasts + one disclosed removed-C line (presentation decision
   deferred, A8).

Status: **reviewed — run plan fixed (2026-08-24), wave not queued.** ~~The A2 U2a-ml
gate precedes any run~~ *(overridden, see decisions below)*; every compared row =
SLURM + `/leo5-submit` + C8 manifest; two-tier numbers policy applies.

### User review decisions (Matteo, 2026-08-24 — run plan fixed)

- **A1 α-downweight control: AGREED** — stays in the plan (ids-arm companion row).
- **A2 U2a-ml pre-run gate: OVERRIDDEN** — no gate; arms go straight to the cluster
  when the wave opens. U2a-ml demoted to optional side analysis.
- **A3 reading rule: ENDORSED as written** — primary endpoint fI′-noid vs fI-noid on
  ml-1m HR@10, no rescue narratives on a null; D2 instrument mandatory on ids rows;
  hm arms are parity guards.
- **A5 centred-LightFM controls: SKIPPED** — not run; attribution rests on the C5
  rank-inertness argument alone (recorded limitation: generic-training-benefit vs
  prototype-route-property stays unseparated).
- **A7 flavours: `weighted` (default) + `interaction`** — the `interaction` flavour
  is elevated from recorded option to a run arm; `unweighted` not run.
- **A11 gauge-snap: INCLUDED in the fI′ default** — post-step projection of per-field
  means to zero (hard form, no hyperparameter); part of the shipped configuration.

**Resulting run set when the wave opens** (each row = own `/leo5-submit`, C8
manifest, frozen fI winner dev configs):
**ml-1m ONLY — zero H&M rows** (user decision, Matteo 2026-08-24, superseding the
earlier same-day 7-row plan): fI′ is **completely experimental** — it earns no H&M
rows and no default-candidacy until it proves something on the motivating dataset.
The A3 hm-parity-guard clause is consequently INACTIVE for this wave (parity would
only be tested in a follow-up wave if the ml-1m verdict is positive). Designed ml-1m
rows: fI′-ids, fI′-noid (weighted flavour, gauge-snap on), fI′-noid-interaction
(flavour arm), α-control-ids — plus the mandatory D2 instrument read-out on the
ids row. Smoke: one end-to-end login-node instance first (standing rule).

**Run treatment (Matteo, 2026-08-24, correcting the briefly-recorded "~40-run
budget" — that was a misreading of his message):** each fI′ row is a **dev-profile
hyperopt run** — `run_combo.py --profile dev`: num_samples=30 trials, n_epochs=60,
patience=7, grace_period=4 (`Master/scripts/run_combo.py` PROFILES) — on ml-1m only,
the same treatment that produced the dev-tier fleet rows it will be compared
against. This **supersedes §3.1's "reuse the frozen fI winner configs"**: fleet
equal-treatment means each arm re-tunes under its own dev sweep (fleet seed
convention). Consequence for 5b P11/A10: the config-mismatch worry softens (hyperopt
re-selects `sim_batch_weight` etc. under centred geometry), traded for
hyperopt-selection noise between arms — the standing fleet trade, accepted.
Sequencing note: the α-control row runs **after** the fI′-ids row (α is read off the
trained fI′-ids checkpoint's post-centring metadata norm — no tuning).
