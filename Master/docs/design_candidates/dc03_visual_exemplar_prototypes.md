# dc03 — Visual Exemplar Prototypes (Push-Grounded Item Prototypes over Frozen Image Content)

> Produced by the design-candidate protocol (`Master/docs/design_candidate_protocol.md`), cycle 3, 2026-07-07.
> Status: **draft — awaiting review** (registered 2026-07-07).
> *(File renamed from `dc03_draft.md` after Step 1 fixed the seed.)*
> Cycle constraints (user, 2026-07-07): (1) dc01 is implemented but not yet fully
> understood/reviewed/tested, dc02 is designed but unprocessed — this candidate treats
> both **only as occupied fingerprint space** (dedup) and must not build on their
> implementation learnings, empirical behavior, or assumed correctness; lessons from
> both are still pending. (2) dc03 must be **solidly distinct from both** dc01 and
> dc02 — ideally more distinct from each than they are from each other.

## 0. Frame

- Requirements re-read in full (living version of 2026-06-11 — unchanged since dc02's
  cycle: spectrum reframing, R7 vocabulary balance, R8 parsimony, S4 LLM naming, S5
  pipeline compatibility).
- **Occupied fingerprint space:** dc01 = features→factors (Σ feature-value embeddings
  replaces the item ID table) × shared-geometry grounding, *unenforced* × per-prototype
  contribution + per-feature shares. dc02 = prototypes live *in attribute space* (item
  branch = bottleneck over observed multi-hot) × grounding *by construction* × per-prototype
  contributions + per-attribute shares. **What dc01/dc02 share** (the distinctness
  target per the cycle constraint): both are item-side, categorical-attribute,
  score-path designs whose read-out is "per-prototype score contribution decomposed
  into per-attribute shares"; they differ mainly in *where prototypes live* and *how
  strongly grounding is enforced* (unenforced ↔ by-construction ends of one spectrum —
  see the `[captured dc02]` spectrum note).
- **Open regions for a maximally distinct dc03:** (a) features entering as a
  **prior/regularizer** on the prototype space, score path untouched (S7 §4.1.3
  REGULARIZE — entry axis; rejected by dc02 as *standalone* for lacking an intrinsic
  read-out — a dc03 there must solve that); (b) **projection/push-to-real-items**
  grounding (C1 ProtoPNet — grounding axis, twice retained as free lead); (c) the
  **image channel** (S3/H themes — a different feature modality and potentially a
  different, perceptual read-out; R7 argues for it; note CLAUDE.md's partial-image-set
  caveat); (d) user-side features (S2 — but scratchpad sequencing says item-only
  first); (e) non-prototype `S1-contrast` vehicles (capped 1–2 across the set, not
  headline).
- **Open requirement niches:** both dc01/dc02 read out in *catalog-attribute*
  vocabulary; R7's "the model should roughly see what the user sees" (images, visual
  appearance) is served by neither. R5 is served by both, but only through categorical
  attributes.
- **Open scratchpad entries** (unmarked, hence open): "Attach feature profiles to
  existing CF prototypes" (2026-06-11); "Hybrid bridge desideratum" (2026-06-16);
  `[captured dc02]` "Loss-enforced grounding" — note: this sits *between* dc01 and
  dc02 on the shared grounding-enforcement spectrum, i.e. it is in tension with this
  cycle's max-distinctness constraint; `[captured dc02]` "Grounding-enforcement
  spectrum as thesis narrative" (a framing device, not a candidate). All get formal
  dispositions in Step 1b. Marked entries (consumed by dc02) still stand as design
  principles: item-only-first sequencing, two-axes gut-check, intrinsic > post-hoc.

## 1. Seed selection

Corpus swept across all themes (B backbone, C/D prototypes, E concepts/attention,
F disentanglement, H fashion, §I forward citations). Two registered candidates →
every shortlisted seed carries explicit distinctness sentences against **both**
fingerprints: dc01 = *features→factors (Σ feature-value embeddings) ×
shared-geometry grounding, unenforced × per-prototype contribution + per-feature
shares*; dc02 = *prototypes in attribute space (bottleneck over observed multi-hot)
× grounding by construction × per-prototype contributions + per-attribute shares*.
Cycle constraint: prefer seeds that differ from both on **all three axes** (dc01/dc02
differ from each other on two — entry and grounding — while sharing read-out shape).

### 1b. Scratchpad sweep (4 open entries: 2 user, 2 captured)

1. **"Attach feature profiles to existing CF prototypes" (user, 2026-06-11)** —
   *not this cycle (its mechanism — profiles attached to CF prototypes — lives on the
   dc01↔dc02 grounding-enforcement spectrum; this cycle's max-distinctness constraint
   points away from that spectrum entirely; stays open, still a strong future seed).*
2. **"Hybrid desideratum: bridge between added module and prototype grounding"
   (user, 2026-06-16)** — *shapes this candidate (the chosen seed adds a content
   channel — a pretrained image encoder — and the bridge requirement is satisfied
   structurally: the encoder's output IS the space the prototypes live in and are
   grounded in via projection; no bolt-on module without grounding feedback).*
3. **`[captured dc02]` "Loss-enforced grounding: learned attribute profile per CF
   prototype + consistency loss" (2026-07-07)** — *not this cycle (explicitly
   mid-spectrum between dc01's unenforced and dc02's by-construction grounding —
   the least distinct possible choice under this cycle's constraint; remains a
   prime seed for a later cycle once dc01/dc02 lessons are in).*
4. **`[captured dc02]` "Grounding-enforcement spectrum as a thesis narrative"
   (2026-07-07)** — *answered inline (it is a framing device, not a candidate; noted
   that the seed chosen this cycle sits OFF that spectrum — exemplar/projection
   grounding is a different KIND of grounding, not a different strength — so the
   narrative axis stays clean: dc01/dc02/(captured) span enforcement strength in
   attribute-vocabulary grounding, dc03 contrasts the grounding vocabulary itself).*

Marked/consumed entries honored as standing principles: item-attribute-only-first
sequencing (dc03 stays item-side), two-axes gut-check (Axis A only — see chosen
seed), intrinsic > post-hoc (drives the projection/push choice below).

### Shortlist

**Seed 1 — C1 ProtoPNet (exemplar prototypes + projection/push) × H1b VBPR (visual
item representation) × A1 ProtoMF host: visual exemplar prototypes for the item
branch.** The item branch consumes a *precomputed, frozen* image embedding per
garment (S5-friendly cached lookup) through a small trainable projection; item
prototypes live in that content space; periodically each prototype is **pushed onto
the embedding of an actual catalog garment**, so at inference every prototype IS a
real, displayable item (its photo + its exact catalog attribute row). Read-out:
"your score is driven by similarity to *this garment*" — case-based, Rudin GC#4.
- *Motivation gate:* serves R7 maximally (the model literally sees a representation
  of what the user sees — the photo; explanations are shown in the most
  user-perceivable vocabulary there is), R5 (a cold item has an image → full-strength
  activation with zero interactions), R2/R3 (the activation to a pushed prototype is
  computed in the forward pass; the explanation is the prototype itself, no post-hoc
  profiling needed — push makes grounding *exact*, not a nearest-neighbor
  reconstruction), R4 (the pushed garment's catalog attribute row is an exact,
  human-readable profile inherited for free), and S3 (the untouched image niche).
- *Distinctness vs dc01:* differs on **all three axes** — entry (frozen image-content
  encoder vs Σ categorical-attribute embeddings), grounding (projection-to-real-item,
  enforced by a training-schedule step vs unenforced shared geometry), read-out
  (case-based "this garment" display vs numeric per-feature shares).
- *Distinctness vs dc02:* differs on **all three axes** — entry (continuous visual
  space from pixels vs observed multi-hot catalog attributes), grounding (exemplar
  identity vs by-construction attribute-weight profile), read-out (displayable
  exemplar + its attributes vs per-attribute numeric shares).
- *R8 parsimony:* PASS — exactly one idea: "the item prototype layer becomes a
  visual, exemplar-anchored prototype layer" (ProtoPNet's mechanism is itself one
  published idea; the frozen encoder is infrastructure, not a second mechanism —
  same conceptual size as `user_item_proto`). Patch/part-level prototypes (spatial
  feature maps) are explicitly EXCLUDED as over-machinery; whole-garment exemplars
  only, parts noted as future work.
- *Data risk (flagged now):* CLAUDE.md documents the local image set as likely
  partial (86 subdirs vs ~105k images) — re-download from Kaggle required before
  any real run. Design-time work (this cycle) is unaffected.

**Seed 2 — S7 §4.1.3 REGULARIZE route as a standalone candidate** (features as a
pure similarity-prior on items' prototype-activation profiles; score path untouched).
Re-swept from dc02's shortlist (there rejected as standalone).
- *Motivation gate:* FAILS again, for the same structural reason dc02 recorded and
  one new one: it produces no intrinsic read-out quantity (weak R2), gives truly
  unseen items no representation at all (weak R5 — a regularizer cannot represent an
  item that has no embedding), and — new this cycle — it is *constraint-shaped*:
  it presupposes some other candidate's representation mechanism to act on, so it
  cannot be "one idea" standing alone. REJECTED; remains prime Step-3 constraint
  material (for any candidate) and a citable-gap note.
- *Distinctness:* would differ from both on entry (regularizer, not score path) —
  moot given the motivation-gate rejection.
- *R8 parsimony:* PASS as one idea, but not candidate-shaped.

**Seed 3 — F1 MacridVAE micro-disentanglement: tie each prototype dimension to an
isolated attribute factor** (prototype identifiability via per-factor
regularisation).
- *Motivation gate:* serves §4 identifiability and partially R4, but F2 (the ECIR
  2025 critical review, a corpus keystone) directly warns that disentanglement does
  not deliver interpretability in recsys — betting the headline mechanism on the
  corpus's own flagged validity threat is the weakest motivation on the table; also
  no R5 story (disentangled CF factors still need interactions). REJECTED as
  headline; disentanglement penalties remain Step-3 constraint-candidate material.
- *Distinctness vs dc01/dc02:* grounding axis (disentangled-factor grounding) — moot.
- *R8 parsimony:* borderline — "make factors disentangled AND map them to attributes"
  is arguably two ideas; would need trimming even if motivated.

**Seed 4 — B8 AFM attention over feature interactions** (non-prototype explanation
vehicle; `S1-contrast` class).
- *Motivation gate:* intrinsically-weighted feature interactions serve R2/R4 in a
  non-prototype vocabulary; but it abandons R3 (prototype-shaped) by design, is
  capped at 1–2 per candidate set as contrast material, and the E4 caveat (attention
  ≠ explanation) is an open faithfulness dispute. NOT chosen as a headline cycle —
  reserved for a deliberate future `S1-contrast` cycle when the user wants the
  ablation counterpart, not under this cycle's max-distinctness brief (distinctness
  must not be bought by leaving the prototype paradigm).
- *Distinctness:* trivially large (different vehicle) — but see gate.
- *R8 parsimony:* PASS (one idea).

### Chosen seed

**Seed 1 — C1 ProtoPNet (Chen et al., NeurIPS 2019: exemplar prototypes,
projection/push, cluster+separation losses as *menu*) × H1b VBPR (He & McAuley,
AAAI 2016: visual item factors from a frozen CNN) under the A1 ProtoMF host, with
H2 (Packer, McAuley, Ramisa 2018) as the fashion-native interpretability precedent.**
Provenance: **corpus**.

Why it won: it is the only shortlisted seed that is simultaneously candidate-shaped
(a real representation mechanism, unlike Seed 2), literature-safe (unlike Seed 3's
flagged validity threat), prototype-paradigm-preserving (unlike Seed 4), and
**distinct from both registered candidates on all three fingerprint axes** — more
distinct from each than they are from each other, exactly the cycle brief. It
attacks the two requirement niches Step 0 identified as unserved (R7 perceptual
vocabulary, S3 images) while keeping the R1/R2/R5 core: content enters the very
representation the tied prototype layer scores; push-grounding makes prototype
meaning exact and intrinsic rather than profiled after training. Its salient risks
are honest and known in advance: the partial-image-set data caveat (re-download
mitigation), the whole-to-whole comparison concern from Rudin GC#4 (parts = future
work), and the question of whether visual similarity alone carries enough
recommendation signal on H&M (R6 — precisely the fidelity-vs-accuracy tension §4
asks a candidate to quantify; VBPR is the literature evidence that visual signal
helps top-N fashion recommendation).

## 2. Mechanism extraction

All four constituent papers read from the local PDFs
(`Master/literature/papers/`): C1 ProtoPNet, H1b VBPR, H2 I-VBPR, A1 ProtoMF
(host re-skim, §3). Every statement carries its in-paper location.

### 2.1 C1 — ProtoPNet (Chen, Li, Tao, Barnett, Su, Rudin; NeurIPS 2019)

**Architecture (§2.1).** Three blocks: a conv net `f` (pretrained VGG/ResNet/
DenseNet + two 1×1 conv layers), a prototype layer `g_P`, and a fully-connected
layer `h` with weight matrix `w_h` and **no bias**. The network learns `m`
prototypes P = {p_j}, each of shape H1×W1×D with H1=W1=1 in their experiments —
i.e. each prototype is a vector in the conv output's channel space, matched
against every spatial patch of the conv output z = f(x).

**Similarity unit (§2.1, displayed formula).** The j-th prototype unit computes

    g_pj(z) = max_{z̃ ∈ patches(z)} log( (‖z̃ − p_j‖₂² + 1) / (‖z̃ − p_j‖₂² + ε) )

— monotonically decreasing in the distance of the *closest* patch to p_j
(inverted-distance similarity; the max over patches is spatial pooling). Each
class k gets a pre-allocated subset P_k of prototypes (10/class in §2.1). The m
similarity scores are multiplied by w_h to produce logits (§2.1, "Finally…").

**Training (§2.2) — three cycled stages:**
1. *SGD of all layers before the last:* minimize
   `(1/n) Σ CrsEnt + λ₁·Clst + λ₂·Sep` with w_h **fixed** (class-k prototype →
   class-k logit weight = 1, non-class = −0.5), where (§2.2, displayed formulas)
   `Clst = (1/n) Σ_i min_{j: p_j ∈ P_{y_i}} min_{z̃} ‖z̃ − p_j‖₂²` (every image has
   some patch close to a prototype of its own class) and
   `Sep = −(1/n) Σ_i min_{j: p_j ∉ P_{y_i}} min_{z̃} ‖z̃ − p_j‖₂²` (patches stay far
   from other classes' prototypes). The separation cost is claimed novel (§2.2,
   "The separation cost is new to this paper").
2. *Projection / push (§2.2, "Projection of prototypes"):*
   `p_j ← argmin_{z ∈ Z_j} ‖z − p_j‖₂`, `Z_j` = all patches of training images of
   p_j's class — every prototype becomes **identical to some real training patch**,
   which is what makes it visualizable/faithful. **Theorem 2.1** bounds the logit
   change caused by the push (under conditions A1–A4 incl. Clst having done its
   job, i.e. prototypes already near patches): correct-class logit drops ≤ Δ_max =
   m′·log((1+δ)(2−δ)); if the top-2 logits are ≥ 2Δ_max apart the prediction is
   unchanged. Push cost = one feedforward pass + pooling (§2.2 end: "same time
   complexity as … global average pooling").
3. *Convex last-layer optimization (§2.2):* with conv + prototypes fixed, minimize
   CrsEnt + λ·Σ|w_h^(k,j)| over the non-class connections → sparsity ("relies less
   on a negative reasoning process").

**Visualization (§2.3):** a prototype is shown as the smallest patch of its source
image enclosing the ≥95th-percentile region of its (upsampled) activation map.

**Reasoning read-out (§2.4, Fig. 3):** per class, a scoring sheet — for each
prototype: the test image's most-activated patch, the prototype's source patch,
similarity score × class connection = points contributed; points sum to the logit.

**Claimed vs shown (§2.5, Table 1):** accuracy within ≈3.5% of the identical
non-interpretable baseline on CUB-200-2011 (e.g. VGG16 76.1±0.2 vs 74.6±0.2 —
sometimes better; Res34 79.2 vs 82.3); ensembling several ProtoPNets reaches 84.8%
(bb), "on par with some of the best-performing deep models". Limitations admitted:
accuracy gap on some backbones (Table 1); interpretability evaluated qualitatively
by reasoning-process walk-throughs, not by user study (§2.4; §1 "qualitatively
similar to … humans").

### 2.2 H1b — VBPR (He & McAuley; AAAI 2016)

**Predictor (Eqs. 1–4).** MF: `x̂_ui = α + β_u + β_i + γ_uᵀγ_i` (Eq. 1) is
extended with a **visual interaction** `θ_uᵀθ_i` (Eq. 2), where the item visual
factor is a *linear projection of frozen pretrained CNN features*:
`θ_i = E·f_i` (Eq. 3), E ∈ R^{D×F}, f_i ∈ R^F the Caffe AlexNet FC7 output
(F = 4096, "Visual Features" §), D ≈ 20 ("say 20 or so", §Preference Predictor).
Full model (Eq. 4) adds a visual bias β′ᵀf_i. **Key template facts:** the CNN is
never fine-tuned; only E (shared across all items) and the ID-based parameters
train; f_i is a precomputed lookup.

**Training (Eqs. 5–8):** standard BPR-OPT over sampled triples (u,i,j), SGA
updates; updating the visual parameters costs O(D×F) per triple ("Scalability" §)
— linear, no full-catalog pass.

**Claimed vs shown (Table 3):** on Amazon Women/Men clothing + Tradesy, VBPR
beats BPR-MF by >12% AUC on all items and >28% on **cold start** (items with <5
training feedbacks; findings 1, p. 5); visual features help clothing more than
cellphones (finding 5). Fig. 4: t-SNE of the learned visual space clusters
garments by style. Limitation admitted: visual and latent dims fixed 50/50
("Baselines" §); one image per item.

### 2.3 H2 — I-VBPR (Packer, McAuley, Ramisa; KDD AI-for-Fashion 2018)

**Mechanism (§3.1–3.2, Eq. 1).** Same predictor shape as VBPR (their Eq. 1), but
f_i is replaced by **interpretable visual features**: outputs of an attribute
classifier (ImageNet-pretrained net fine-tuned on a fashion dataset annotated
with attributes, §3.1), i.e. each dimension of f_i is an attribute probability.
**Claimed vs shown (Table 2):** interpretable features use "a small fraction"
(<5%, §4.4) of the black-box models' parameters at comparable AUC (black box
keeps a 1–2% edge on All-Items; interpretable leads on most Men's cold-start
cells). §5: the interpretable space enables user-affinity vectors over named
attributes (Eq. 5) and interactive steering (Eqs. 6–7); §6: per-feature fashion
trend curves (Eq. 8). **Relevance here:** precedent that (a) fashion
recommendation works on *interpretable* content dimensions at small accuracy
cost, and (b) attribute-grade image features are a viable f_i drop-in.

### 2.4 A1 — ProtoMF host (Melchiorre et al., RecSys 2022, §3), the transplant target

- Item-prototype branch (§3.2): `t* = [sim(t, p^t_1) … sim(t, p^t_{L^t})] ∈
  R^{L^t}` (Eq. 7) with **shifted cosine** `sim(a,b) = 1 + aᵀb/(‖a‖‖b‖)` (Eq. 1);
  I-score(u,t) = u ᵀt* (Eq. 8).
- Inclusion-criteria regularizers, item side (Eq. 9): `R_{P^t→T} = −(1/L^t) Σ_l
  max_j sim(t_j, p^t_l)` (each prototype near some in-batch item) and
  `R_{T→P^t} = −(1/M) Σ_i max_l sim(t_i, p^t_l)` (each item near some prototype);
  total loss Eq. 10: `L_rec + λ₃R_{P^t→T} + λ₄R_{T→P^t}`, with L_rec the sampled
  softmax (Eq. 3).
- Double tie (§3.3, Eqs. 11–12): `û = W^u u`, `t̂ = W^t t`;
  `UI-score = u*ᵀt̂ + ûᵀt*` (Eq. 12); loss = sum of both sides' losses.
- Interpretation practice (§1 p. 247): item prototypes interpreted via *maximally
  close entities* (top-k nearest items) — the post-hoc step this candidate's
  push-grounding replaces.

### 2.5 The composed seed mechanism (what dc03 takes from each)

From **VBPR**: the item content path `f_i (frozen, precomputed) → linear E →
item representation` — the S5-compatible way to get images into a factor model,
plus the evidence it wins exactly in cold-start fashion regimes. From
**ProtoPNet**: prototypes that are (i) trained by SGD in that content space,
(ii) periodically **pushed onto real training entities** so each prototype is
identical to an actual catalog garment's representation (whole-item analogue of
their patch push; Theorem 2.1 is the template argument that pushing barely moves
the score), and (iii) read out as a case-based scoring sheet. From **I-VBPR**:
the precedent that the content dimensions can themselves be attribute-grade — kept
as an *option* for the encoder choice, not a second mechanism. From the **host**:
everything else stays — shifted cosine, t*, dot-product score, sampled softmax,
inclusion criteria, double tie. The transplant is specified in Step 3.

## 2b. Prior-art probe

**Queries (web, 2026-07-07; reproducible):**
1. `prototype-based recommendation visual prototypes "push" projection exemplar item images interpretable`
2. `ProtoPNet recommender system "this looks like that" recommendation prototypes item embedding`
3. `visually-aware fashion recommendation prototype learning interpretable cold-start image prototypes matrix factorization`
4. `"exemplar-based" OR "case-based" explainable recommendation "actual item" prototype anchored content embedding cold start`

**Findings (nearest neighbors, all distinguished):**
- **Visually-aware MF lineage** — VBPR (our seed), Sherlock (He & McAuley 2016,
  arXiv:1604.05813; category-dependent visual embeddings), DeepStyle/GAN fashion
  (Kang et al., ICDM 2017), CausalRec (2021): visual factors in the score, **no
  prototype layer, no exemplar grounding, no intrinsic case-based read-out** —
  explanation is not their goal.
- **Prototype recsys lineage** — UIPC-MF (PAKDD 2024, §I of the reading list),
  "Preference Prototype-Aware Learning" (CIKM 2024), cultural-diversity ProtoMF
  successors (arXiv:2412.14329, 2405.17607), RVRec (2025): all **CF-only**
  prototype spaces; none grounds prototypes in item content, none pushes
  prototypes onto real items. Consistent with the reading-list §I sweep (all 31
  ProtoMF citers CF-only as of 2026-06-11); the searches surfaced no newer
  counterexample for this mechanism.
- **Exemplar-anchored prototypes outside recommendation** — ProtoECGNet
  (arXiv:2504.08713): prototypes anchored to real ECG segments for case-based
  explanation — evidence the push-anchoring template ports beyond vision
  classification, but it is classification, not recommendation.
- **H3 PAICM** (SIGIR 2019, added to the list by dc02's probe): attribute-signature
  NMF templates for outfit *matching* — no users, no ranking model, no image-space
  prototypes, no push.
- **IBR** (McAuley et al. 2015, cited in VBPR Table 3): nearest-neighbor retrieval
  in a visual space — not personalized, not prototype-based; VBPR itself beats it
  warm and matches its spirit cold.

**Verdict: found nothing close to the adaptation.** No published work puts
projection-grounded (exemplar) prototypes over a frozen-content item
representation inside a tied MF prototype layer, with a case-based intrinsic
read-out, for recommendation. The novelty wedge — grounding *recommendation*
prototypes in item content — remains open on this axis too (visual/exemplar), just
as dc01 (composed factors) and dc02 (attribute space) found on theirs. No
refinement needed; the "distinguish" list above goes into Related Work.

## 3. Adaptation design

**Name:** `ft_type: 'visual_item_proto'` — **I propose** this new reserved name
(CLAUDE.md reserves `feature_item_proto` = dc01 and `dual_item_proto`; this is a
third, to be added on registration). Slug: `dc03_visual_exemplar_prototypes`.

### 3.1 One-paragraph mechanism

The item branch of `user_item_proto` is rebuilt on frozen image content: each
article's photo is encoded **once, offline** by a frozen pretrained encoder into
`v_i ∈ R^F`; at train time a single trainable linear map `E: R^F → R^d` produces
the item representation `q_i = E v_i` that BOTH halves of the double tie consume
(prototype similarities and the projection branch). Item prototypes are vectors in
this projected visual space and are **periodically pushed onto the representation
of an actual catalog garment** (ProtoPNet §2.2 projection, whole-item analogue), so
that at inference every item prototype IS a real garment — displayable as its
photo plus its literal catalog attribute row. The user branch, scorer, loss, and
sampling are untouched host code.

### 3.2 Modules that change (real names; verified against current code)

| Piece | File | Change |
|---|---|---|
| `VisualEmbedding(FeatureExtractor)` | `feature_extraction/feature_extractors.py` | **new** — holds `visual_features` (n_items × F) as a NON-persistent buffer (dc01's `feature_ids` pattern) + `nn.Linear(F, d, bias=False)` (= E, the VBPR Eq. 3 kernel). `forward(o_idxs) = linear(visual_features[o_idxs])`. |
| `VisualEmbeddingW(FeatureExtractor)` | same | **new** — mirrors `FeatureEmbeddingW` exactly: wraps the SAME `VisualEmbedding` instance + `nn.Linear(d, L^u, bias=False)` (= W^t). *(Implementation may instead generalize `FeatureEmbeddingW` into one shared-wrapper class — cosmetic, decided at implementation time.)* |
| `ExemplarPrototypeEmbedding(PrototypeEmbedding)` | same | **new** — adds `push(all_item_reprs: Tensor[n_items, d]) -> None`: `p_k ← q_{i*(k)}, i*(k) = argmax_i sim(q_i, p_k)` (**I propose** cosine-argmax rather than ProtoPNet's L2-argmin, matching the host's shifted-cosine geometry; ProtoPNet §2.2 uses L2 because its g uses L2). Stores `exemplar_ids` (K,) buffer for the read-out. Everything else inherited — the prototype layer itself is untouched host code via `embedding_ext=` (hook already exists, `PrototypeEmbedding.__init__` line 302). |
| Factory branch `'visual_item_proto'` | `feature_extraction/feature_extractor_factories.py` | **new** — clone of `'feature_item_proto'` (lines 101–153) with `FeatureEmbedding → VisualEmbedding`: user branch byte-identical to `prototypes_double_tie`; item branch = `ExemplarPrototypeEmbedding(..., embedding_ext=visual_embed)` + `VisualEmbeddingW(shared=visual_embed, out_dimension=user_n_prototypes)`; single-owner init as in dc01's branch. |
| Push hook | `rec_sys/trainer.py` | **new, small** — every `push_period` epochs (first at `push_start_epoch`): compute `Q = visual_embed(all_items)` (ONE (n_items×F)·(F×d) matmul, off the step path), call `item_proto.push(Q)`, then run the epoch's validation so model selection only ever sees push-consistent states. Best-checkpoint saving is unchanged. |
| Precompute script | `data/hm/precompute_image_embeddings.py` | **new, offline** — frozen encoder (proposal: start CLIP ViT-B/32, F=512; ResNet-50 avgpool F=2048 as alternative — an encoder-choice experiment, not a mechanism change) over `data/hm/raw/images/`, L2-normalized rows, saved as `.npy`; injected at `_build_model` time like dc01's `feature_ids` (never serialized into Ray/JSON config). |
| Losses / data pipeline | — | **unchanged** — sampled softmax, negative sampling, `(batch, 1+n_neg)` scoring, `get_and_reset_loss()` all as in host. |
| CLI + search space *(cold-read edit, 2026-07-07)* | `start.py`, `confs/hyper_params.py` | **new, small** — model name `visual_item_proto` in the CLI choices; search-space block cloned from `user_item_proto`'s (same dims/weights; push_period & push_start_epoch fixed constants per §5b amendment A4, NOT search dims). |

**Items without images** (H&M has a small set): `v_i = 0`. PyTorch
`CosineSimilarity` eps keeps this defined; shifted cosine gives `sim = 1`
uniformly over prototypes — a neutral, non-informative but non-degenerate
activation. Documented behavior, revisit at implementation if the count matters.

### 3.3 Forward pass with shapes (B = batch, N = 1+n_neg, F = encoder dim, d = embedding_dim, L^u/L^t = user/item prototype counts)

User side (byte-identical to `prototypes_double_tie`):
```
u_idxs (B,) → u (B,d) [tied table]
u* = shifted_cos(u, P^u) (B, L^u);  û = W^u u (B, L^t)
user_repr = [u* ; û] (B, L^u + L^t)
```
Item side:
```
i_idxs (B,N) → v = visual_features[i_idxs] (B,N,F)   [buffer gather]
q = v @ E (B,N,d)                                     [ONE shared projection]
t* = shifted_cos(q, P^t) (B,N,L^t);  t̂ = W^t q (B,N,L^u)
item_repr = [t̂ ; t*] (B,N, L^u + L^t)                [Concat, invert=True]
```
Score (host `RecSys`, unchanged): `UI-score = Σ user_repr·item_repr = u*ᵀt̂ + ûᵀt*`
(A1 Eq. 12). **Parameters:** `E` (F×d, trainable, shared — this sharing IS the R1
tie on the item side, same pattern as dc01's shared `FeatureEmbedding`), `P^t`
(L^t×d), `W^t` (d×L^u), user side unchanged (tied u-table, P^u, W^u). **Frozen:**
`visual_features` (n_items×F). **Gone:** the free per-item ID embedding table.

### 3.4 Intrinsic explanation read-out

Quantities, all from the forward pass: per-prototype contributions
`c_k = û_k · t*_k` (they sum exactly to the I-score half, host Eq. 8 semantics),
and per-prototype identity `exemplar_ids[k]` = the catalog article the prototype
was pushed onto. There is **no** per-attribute share decomposition (that is
dc01/dc02's read-out shape); the explanation unit is the exemplar garment.

Rendered H&M example (user u, recommended article "slim high-waist jeans"):

> **Why this?** Your taste matched these signature garments:
> 1. **[photo]** "Skinny H.W. Denim, Dark Blue, Divided" — visual similarity
>    1.72 × your affinity 0.91 = **1.57**
> 2. **[photo]** "Mom-fit jeans, Washed Black, Denim Women" — 1.48 × 0.66 = **0.98**
> 3. **[photo]** "Rib top, Off-white, Basics" — 1.12 × 0.38 = **0.43**
> (each exemplar shown with its photo and its actual catalog attributes:
> garment group, colour, department, product type — the prototype's R4 profile
> is the exemplar's literal catalog row, exact by construction after push)

The user-prototype half (`u*ᵀt̂`) keeps the host's existing user-side explanation;
S4 LLM naming can verbalise an exemplar ("your dark-denim staple") on top.

### 3.5 Interpretability-constraint design (v1.1 mandatory block)

**Inheritance.** Both item-side host regularizers survive **and keep their
semantics** in the visual space: `R_{T→P^t}` (`sim_batch`, A1 Eq. 9 right) still
means "every item is covered by some prototype"; `R_{P^t→T}` (`sim_proto`, Eq. 9
left) still means "every prototype stays near some real item" — and it acquires a
**second job**: it is the structural analogue of ProtoPNet's cluster cost, the
very condition (Theorem 2.1, A2a) under which the push is a small perturbation of
the score. The `reg_proto_type='max'` variant is the literal Clst counterpart;
`'soft'` is a smoothed one. User-side regs untouched. All ride
`get_and_reset_loss()` unchanged.

**New degeneration modes and designed (UNSTACKED) constraint candidates:**

| # | Mode | Covered by inherited? | Constraint candidate (not stacked — user's decision) |
|---|---|---|---|
| D1 | **Exemplar duplication**: ≥2 prototypes push onto the same garment → redundant set, shrunk vocabulary | No (nothing pushes prototypes apart) | *Soft:* `L_div = (2/K(K−1)) Σ_{k<k'} max(0, cos(p_k,p_{k'}) − τ)`, hook `get_and_reset_loss()`, weight `div_weight`, cost O(K²d) — negligible. *Hard:* uniqueness-constrained push (greedy assignment of prototypes to distinct argmax garments). Expected effect: distinct exemplars, mild accuracy cost if K large. |
| D2 | **Mid-training drift**: between pushes prototypes are not real items; reading an explanation from a non-pushed checkpoint is unfaithful | Partially (`sim_proto` keeps them *near* items; final push restores exactness) | *Hard (structural):* only evaluate/save/explain post-push states (the trainer hook above already does this). *Knob:* `push_period` ↓ (cost: one catalog matmul per push). |
| D3 | **Push shock**: first push far from converged geometry → score cliff (ProtoPNet's Theorem 2.1 conditions not yet met) | Partially (`sim_proto`) | *Knob:* `push_start_epoch` warmup (ProtoPNet stages SGD before pushing, §2.2); expected effect: pushes become ε-moves; zero extra cost. |
| D4 | **Popularity capture**: prototypes all push onto head/popular garments → tail styles unrepresented | No | *Soft:* popularity-stratified candidate pool for the push argmax (restrict/reweight by item frequency deciles); cost: bookkeeping only. |
| D5 | **Semantic gap**: encoder similarity ≠ human-perceived garment similarity (S5 pitfall; R7 risk) | No — and **not lossable**: this is an encoder/eval property | Encoder choice validated on fashion retrieval (CLIP > ImageNet-ResNet for style, H2's attribute-tuned encoder as the interpretable extreme) + qualitative purity check in the eval protocol. **Open risk**, flagged for Step 5. |
| D6 | **Projection collapse**: E rank-collapses → all q_i in a low-dim cone, cosine saturates | No | *Soft:* orthogonality penalty `‖EᵀE − I_d‖²_F`, hook `get_and_reset_loss()`; cost O(F d²)/step or applied per-epoch. |

**Rashomon-tunable knobs** (requirements §4 identifiability note): inherited
`sim_proto_weight`, `sim_batch_weight`; proposed `div_weight`/τ (D1),
`push_period`/`push_start_epoch` (D2/D3), orthogonality weight (D6). Tune within
the accuracy-equivalent set for sharp, distinct, non-duplicated exemplars.

### 3.6 Baseline gate

- **CF-only `user_item_proto`** (always) — does content cost dense-regime accuracy (R6)?
- **VBPR (H1b)** — the natural external content baseline: same frozen features,
  same E-projection, no prototype layer, no intrinsic read-out. (LightFM/B5 is the
  *categorical* twin and stays dc01's baseline; VBPR is dc03's.)
- **The one-idea-isolating ablation: push vs no-push** — identical model with
  free (never-pushed) visual prototypes. Isolates exactly the exemplar-grounding
  idea; also quantifies its accuracy cost (fidelity-vs-accuracy tension, §4).
- **ID-residual ablation** — `q_i = E v_i + b_i` with a free per-item `b_i`
  (VBPR keeps γ_i alongside θ_i, Eq. 2; **I propose** the pure-content headline —
  more radical than VBPR — with this ablation quantifying what dropping per-item
  CF capacity costs; cold items simply have b_i at init).
- **Post-hoc comparison (R2 baseline):** host `top_k_items` explainer on dc03's
  own prototype space — is the pushed exemplar ≈ the top-1 nearest item anyway?
  (It should be, by construction — the point is that push makes it *the model's
  own representation*, not a reconstruction.)

### 3.7 Deviations from the source papers (all "I propose")

1. **Whole-item exemplars, not patches** — ProtoPNet pushes onto image *patches*
   (H1=W1=1 over a spatial map); we push onto whole-garment embeddings. Loses
   part-level "this collar looks like that collar"; noted as future work (the
   user's 2026-06-11 scratchpad caveat re Rudin GC#4 whole-to-whole comparison
   applies and is carried to Step 5).
2. **Cosine-argmax push** (ProtoPNet: L2-argmin) — matches host geometry.
3. **No class restriction in the push pool** (ProtoPNet pushes within the
   prototype's class; recommendation has no classes) — full-catalog pool, D4 knob.
4. **No convex last-layer stage** (ProtoPNet stage 3) — the host has no free
   last layer; û/u* are model-internal. The sparsity job (fewer active
   explanation lines) falls to the top-3 rendering, not a loss.
5. **Pure-content item branch** (VBPR keeps a parallel CF factor γ_i) — headline
   drops per-item free parameters entirely for maximal R5; the ID-residual
   ablation reintroduces them.
6. **Encoder frozen forever** (ProtoPNet fine-tunes f between pushes) — S5
   compatibility and simplicity; H2 shows frozen attribute-grade features
   suffice at small cost.

## 4. Cost & feasibility

All estimates Fermi-style; split counts as measured for dc02 (2026-07-07):
`hm_3_month` N_u=256,701, M=26,963; `hm_full` M=90,690. d=64, K=L^u=50 mid-range;
B=256, n=1+n_neg=11. Encoder options: CLIP ViT-B/32 (F=512, proposed default) /
ResNet-50 avgpool (F=2048).

### 4.1 Parameter delta (vs CF-only `user_item_proto`)

| block | host | dc03 (CLIP F=512) |
|---|---:|---:|
| user table N_u·d | 16.43M | 16.43M (unchanged) |
| user protos L^u·d + proj d·K | 6.4K | 6.4K (unchanged) |
| item table M·d | 1.73M | **0** |
| E (F·d) | — | 32.8K |
| item prototypes K·d | 3.2K | 3.2K |
| item projection W^t d·L^u | 3.2K | 3.2K |
| **total** | **≈18.2M** | **≈16.5M (−9%)** |

Item branch shrinks ~98% (1.74M → ~39K) and **no trainable parameter scales with
M** (same R5-relevant property dc02 has, reached via images instead of
attributes). ResNet F=2048 raises E to 131K — still negligible. The
`visual_features` buffer is the real memory item: fp32 55 MB (`hm_3_month`,
F=512) / 186 MB (`hm_full`); F=2048 → 221/743 MB. fp16 halves it; PCA→256 is the
fallback. GPU-fine in all variants.

### 4.2 Training-time expectation

Per-step item-branch MACs: host ≈ B·n·d·(K+L^u) ≈ 18M; dc03 ≈ B·n·F·d (the E
projection dominates) ≈ **92M** (F=512) / 369M (F=2048) — a 5×/20× item-branch
multiplier, sub-ms on an A100/A30 and small beside the unchanged rest (16M-row
user table, sampler, optimizer). **Expected wall-clock ≈1–1.5× host per epoch
(CLIP).** The **push** adds one off-step catalog pass per `push_period` epochs:
M·F·d ≈ 0.9G MACs + K·M cosines (`hm_3_month`) ≈ a fraction of a second on GPU —
irrelevant. Convergence: fewer free parameters, no cold rows to fit; but
ProtoPNet-style stage cycling (warmup → push → continue) may need a few extra
epochs to re-settle after early pushes — claimed only as "same order", arguable
and flagged.

**One-off precompute:** encode the catalog images once — ~105K images, CLIP
ViT-B/32 at ~10³ img/s on an A100 ≈ **2–3 min** (RTX 4070 TiS similar order; a
CPU-only laptop run would be hours — do it on GPU). Output `.npy` is the only
artifact the training pipeline ever touches; training never sees pixels.

### 4.3 S5 pipeline-compatibility checklist

- Sampled `(batch, 1+n_neg)` scoring preserved? **YES** — item branch = buffer
  gather + one (F×d) matmul; `RecSys.forward` untouched.
- Full-catalog pass per step? **NO** — the only catalog pass is the push, once
  per `push_period` **epochs**, off the step path.
- Item/user branch cheap indexed computation? **YES** — gather ≈ embedding
  lookup; user branch unchanged.
- Losses fit `get_and_reset_loss()`? **YES** — host regularizers reused verbatim;
  all proposed knobs (§3.5) hook the same accumulator. The push itself is not a
  loss — it is a trainer-level parameter assignment (torch.no_grad), new hook.
- Expensive per-item work → precompute/cache strategy? **YES, stated** — frozen
  encoder offline (§4.2); training-time cost is a gather.

### 4.4 Data-readiness flag (unique to this candidate)

The local image set is **likely partial** (CLAUDE.md 2026-06-07: 86 subdirs
present vs ~105K images documented). Before any real run: re-download the ~25 GB
Kaggle images and verify subdirectory range `000`–`0NN`. Also: a small fraction
of articles have no image at all → the `v_i = 0` neutral-activation path (§3.2)
plus a count check at splitter time. **Flag, with stated mitigation — not a
design defect.** Design/dev can proceed on the partial set (any subset of items
with images suffices for toy/dev runs).

### 4.5 Budget gate

Hyperopt same order as host `user_item_proto` (≈1–1.5× per step, same trial
protocol) — inside the 10-day hard and ~1-day dev budgets on LEO5. Dev iterations
on the CPU laptop are fine at 92M MACs/step; the encoder precompute is a one-off
GPU task. **No budget flags** beyond the §4.4 data-readiness flag.

## 4b. Math sanity check

**Claims spec:** written black-box-clean (mechanism definition + claims, no
derivations) at `Master/temp/dc_checks/dc03/claims_spec.md`. Claims in brief:
CL1 score's prototype half decomposes exactly into per-prototype contributions
û_k·t*_k; CL2 the shared E receives gradient through BOTH tied paths (t* and t̂);
CL3 a cold item's content row alone yields a non-constant activation pattern;
CL4 zero-image items get exactly-1 uniform activations (defined, neutral);
CL5a push is idempotent; CL5b push perturbation of the score is ≥10× smaller
when prototypes already sit near items than when random; CL5c after push every
prototype equals a real item representation bitwise; CL6 host `R_proto` descent
pulls prototypes toward their best-matching batch items; CL7 the proposed L_div
(D1) is zero/zero-gradient below τ and repels duplicates; CL8 the proposed
L_orth (D6) restores rank from collapse; CL9 cosine-argmax vs L2-argmin push
differ on unnormalized data, coincide under equal norms; CL10 greedy
uniqueness-constrained push yields K distinct exemplars.

**Viability self-assessment.** All ten are meaningful at toy scale (shapes,
gradient flow, limits, push algebra). Explicitly NOT toy-checkable and skipped
with justification: (a) R6 accuracy of a pure-visual item branch — training-
dynamics + data property, only a real run answers it (Step 5 risk); (b) real-
training push shock — reduced to its checkable core, the CL5b proximity-
monotonicity proxy; (c) D5 encoder semantic gap — an encoder/perception
property, not mathematics (Step 5 open risk).

**Subagent report (verbatim):**

> - **CL1 (decomposition): confirmed** — max |Σ_k c[b,j,k] − term2[b,j]| = 0.00e+00 over all (b,j); the prototype half of the score decomposes exactly into per-prototype contributions.
> - **CL2 (tie gradient): confirmed** — ||grad E|| from a t_star-only scalar = 3.07, from a t_hat-only scalar = 7.19; both paths deliver nonzero gradient to E.
> - **CL3 (cold item): confirmed** — activation variance across the K prototypes for a fresh L2-normalized content row: min 0.017, mean 0.100 over 20 random (v_new, E, P) draws, well above 0.
> - **CL4 (zero-image neutrality): confirmed** — v = 0 yields activations exactly [1.0, 1.0, 1.0, 1.0, 1.0] (eps-guarded cosine returns exactly 0, shifted to 1).
> - **CL5a (push idempotence): confirmed** — second push changes P by max 0.00e+00 and reselects identical argmax indices.
> - **CL5b (push perturbation monotone in proximity): confirmed** — mean |Δscore| ratio (random P) / (P at item reps + σ=0.01 noise) = 20–66x across 5 seeds (min 19.6x ≥ 10x).
> - **CL5c (exact grounding): confirmed** — every pushed prototype row bitwise-equals a row of Q (exemplar indices [19, 19, 38, 28, 6]; note duplicates can occur, which is what CL10's greedy variant fixes).
> - **CL6 (R_proto pulls prototypes to items): confirmed** — mean_k max_j cos(q_j, P_k) rose 0.460999 → 0.464185 after one GD step (lr=0.1), a strict increase.
> - **CL7 (L_div behavior): falsified (literal), 2 of 3 sub-claims pass** — L_div(orthogonal P) = 0 exactly and grad is exactly 0 when all pairwise cosines < τ (both confirmed). But the middle sub-claim fails: for two *exactly identical* prototypes, the gradient on the pair is exactly 0 (cosine is scale-invariant, so its gradient vanishes at cos = 1) and one step leaves cos = 1.0000000000 unchanged. Nuance: for a *nearly* identical pair (cos = 0.9985) the step does strictly decrease the cosine (→ 0.9984), so L_div repels near-duplicates but cannot split an exact duplicate.
> - **CL8 (L_orth counteracts collapse): falsified (literal)** — with E's columns exactly equal, GD on L_orth cannot escape: σ_min stays at round-off (4e-81 → 2e-18, i.e. σ_min/σ_1 = 2e-18) and effective rank stays 1.0000000000 (matrix_rank = 1 before and after). Analytical reason: grad = 4E(EᵀE − I) lies in the column space of E, so rank cannot grow. Nuance: from *near*-collapse (+1e-3 noise) the claim holds clearly — σ_min 1.4e-3 → 3.8e-3, eff-rank 1.06 → 1.37 — so L_orth counteracts collapse in practice (where init noise always exists) but not from the exact rank-1 manifold stated in the literal example.
> - **CL9 (cosine-argmax vs L2-argmin): confirmed** — exhibit: q0 = (10,0,...) vs q1 = (1,0.5,...), p = e1: cos-argmax picks item 0 (cos 1.000 vs 0.894) while L2-argmin picks item 1 (dist 9.00 vs 0.50); with all 40 rows rescaled to equal norm, the two selections coincide for all 50 random prototypes tested.
> - **CL10 (greedy unique push): confirmed** — 10 trials (including 5 with all K prototypes identical, i.e. all wanting the same item) plus an M = K edge case all yielded exactly K distinct exemplar indices.
>
> Summary: 8 of 10 claims confirmed at toy scale; CL7 and CL8 are falsified as literally stated (exact-duplicate/exact-collapse configurations are zero-gradient or invariant sets) but both hold under the natural nearby interpretation (near-duplicate prototypes, near-collapsed E), which is the regime any real training run occupies.

**Script:** `Master/temp/dc_checks/dc03/toy_claims_check.py` (float64, fixed
seeds, ~2 s in the `protomf` env).

**Disposition (no test-shopping; carried to Step 5 as known defects):**
- **CL7 falsified-as-stated:** the D1 *soft* constraint (L_div) cannot split
  *exactly* duplicated prototypes (zero-gradient set). Consequence: after a push
  creates exact duplicates (CL5c showed indices [19,19,…] can collide!), L_div
  alone will never separate them → if duplication protection is wanted, the
  **hard greedy-unique push (CL10, confirmed) is the right variant**, or L_div
  must act pre-push. Recorded as a defect of the D1-soft knob, not of the
  headline mechanism (which stacks neither).
- **CL8 falsified-as-stated:** the D6 knob restores rank only from *near*
  collapse, not from the exact rank-deficient manifold — acceptable in practice
  (random init), recorded verbatim.
- Bonus finding from CL5c: an UNconstrained push can genuinely produce duplicate
  exemplars even at toy scale — D1 is a real mode, not hypothetical.

## 5. Requirements evaluation

Requirements doc re-read (living version of 2026-06-11, unchanged). Interpretable-
AI placement (protocol 5.4b) skipped: the principle framework is still undecided
(ideas.md §4, "Key task — commit to one principle framework" still open).

### 5.1 Ratings + prose

| Req | Rating |
|---|---|
| R1 tied integration | **strong** |
| R2 intrinsic explanation | **strong** |
| R3 prototype paradigm | **strong** |
| R4 feature-grounded prototypes | **partial** |
| R5 sparsity robustness | **strong** |
| R6 dense-regime accuracy | **weak / open — the candidate's declared bet** |
| R7 user-perceivable vocabulary | **strong** |
| R8 parsimony | **strong** |
| S1 decoupled variant | **partial (supported via ablations)** |
| S2 user-side extension | **n-a (not precluded)** |
| S3 image features | **strong — first candidate to serve it** |
| S4 LLM naming | **strong** |
| S5 pipeline compatibility | **strong** |

> *(Cold-read edit, 2026-07-07: R2 and R7 "strong" are conditional — see §5b
> amendment A1: conditional on the pure-content headline surviving R6, and R7
> additionally on D5. Read the table together with A1.)*

**R1 — strong.** The one trainable content map E is *shared* between the item
prototype branch and the item projection branch (§3.3) — the identical sharing
pattern the host realizes with its tied embedding table and dc01 realizes with
its shared `FeatureEmbedding`. Content drives the very representation both
halves of the UI-score consume; there is no parallel reranker, no post-hoc
module. CL2 confirmed gradient reaches E through both paths.

**R2 — strong.** Every explanation quantity is computed in the forward pass:
t*, û, the products c_k = û_k·t*_k (CL1: they sum to the score half exactly),
and the prototype's identity, which after a push is a *parameter-level fact*
(CL5c: bitwise equality with a real item's representation) rather than an
after-training interpretation. The contrast with the `top_k_items` baseline is
sharp: that explainer *finds* a nearby item post-hoc; here the prototype *is*
an item, by construction, inside the scoring path. Strain: between pushes the
exemplar identity is only approximate — the D2 discipline (evaluate/save/
explain only post-push states) is load-bearing and must be honored in the
implementation, or R2 quietly degrades to "approximately intrinsic".

**R3 — strong.** The explanation stays exactly prototype-shaped: which
prototypes fire, how much they contribute, what they mean. One nuance argued
honestly: R3's text contrasts feature-tied meaning with "a cloud of
representative items"; dc03's prototype meaning IS a representative item —
but a *single, exact, model-committed* one, not a post-hoc cloud, and it
carries its catalog attribute row. The paradigm requirement (prototype-shaped
explanations) is fully met; the vocabulary question belongs to R4.

**R4 — partial, the candidate's main concession.** There is an explicit
association between each prototype and human-readable features — the pushed
exemplar's literal catalog row (garment group, colour, department, …), exact by
construction. But the *second clause* of R4 ("an item's prototype activation
can be attributed back to its features") holds only in the visual vocabulary:
activation is a holistic image similarity and does NOT decompose into
per-attribute contributions (that read-out shape is precisely dc01/dc02's
territory). If per-attribute attribution is a hard need, dc03 does not provide
it; it provides "because it looks like *this*" instead. This is a deliberate
trade — R7-maximal, R4-partial — and the single clearest axis on which the user
can rank this candidate against dc01/dc02.

**R5 — strong.** No trainable parameter scales with the number of items; any
item with an image receives a full-strength, discriminative activation pattern
with zero interactions (CL3), and score computation is identical for day-0
items. VBPR's Table 3 is the literature evidence that exactly this feature
family carries cold-start in fashion (+28% AUC cold). Two strains: items
*without images* fall back to a neutral uniform activation (CL4 — defined but
uninformative; count to be checked at split time), and the whole candidate is
hostage to the image set being restored (§4.4 flag).

**R6 — weak / open; the declared bet.** The headline drops ALL per-item CF
capacity (no ID table, no biases): item-side signal is whatever the frozen
encoder + learned E can express. VBPR keeps a parallel CF factor γ_i and its
visual-only baseline (IBR) *loses* warm-start — so literature precedent says
pure-visual is risky in the dense regime. dc03 is substantially stronger than
IBR (personalized user branch, trained E/W^t, prototype layer), but visual
space plausibly cannot express popularity, quality, or fit — attributes of
taste that are not in the photo. The ID-residual ablation (§3.6) is the
designed mitigation and would move the design toward VBPR's hybrid if needed
(at a cost to R5 purity for warm items, none for cold). This is the
fidelity-vs-accuracy tension (§4) instantiated at its sharpest point among the
candidate set so far.

**R7 — strong; the candidate's raison d'être.** "The model should roughly see
what the user sees" is satisfied more literally than any attribute design can:
the model's item knowledge is a representation of the photo, and the
explanation unit shown to the user is the photo. No retailer taxonomy needed
anywhere in the loop (the exemplar's attribute row is displayed as a bonus, not
required for the mechanism). Two strains, stated: (a) the R7 caveat demands the
features also be *among the best for recommendation quality* — on H&M this is
untested (the 4.0 feature analysis measured categorical lifts only; images were
not analyzed) and rides on the R6 bet; (b) the D5 semantic gap — a frozen
ImageNet/CLIP encoder's similarity is not guaranteed to match human garment
perception; encoder choice + a qualitative purity eval are the mitigations,
and this is an honest open risk, not a solved problem.

**R8 — strong.** One idea: *the item prototype layer becomes a visual,
exemplar-anchored prototype layer.* The push is not a second mechanism — it is
the grounding half of ProtoPNet's single published idea (prototype layer +
projection belong together; without the push the prototypes are just latent
vectors in image space and the exemplar claim evaporates). The frozen encoder
is infrastructure (VBPR treats it identically). All six constraint candidates
(§3.5) are documented, none stacked. Granularity comparison: `user_item_proto`
= "represent entities by prototype similarities"; dc03 = "let item prototypes
be real garments in image space" — same conceptual size.

**S1 — partial (well supported).** The no-push ablation IS a natural
"loosely-coupled" rung (intrinsic score, unenforced grounding), and
`top_k_items` run on dc03's own space is the fully post-hoc rung — the
candidate ships its own decoupling ladder. Not a separate decoupled variant,
hence partial, but S1's purpose (demonstrate why tying/grounding matters) is
directly servable.

**S2 — n-a, not precluded.** The user branch is untouched host code; a future
symmetric move (user attributes through a user-side content map) remains
possible. Nothing in dc03 blocks it; nothing serves it either.

**S3 — strong.** This is the image-channel candidate — the niche neither dc01
nor dc02 touches. It also gives S3 its cleanest possible form: images as the
*only* item input, so the channel's value is measured, not confounded.

**S4 — strong.** An exemplar photo + catalog row is the ideal input for the
LLM renaming layer ("your dark-denim staple"); slots into `naming/` unchanged,
names-never-scores boundary respected.

**S5 — strong.** §4.3: all checklist items green; the only catalog-scale
operation (push) is per-epoch, off the step path; per-step cost ≈1–1.5× host.
The one new pipeline element is the offline encoder precompute (one-off,
minutes on GPU).

### 5.2 §4 design tensions

- **Where do features enter?** At the representation root: the item factor is a
  projection of frozen content, and prototypes live in that projected content
  space. Third distinct answer in the set (dc01: features compose the CF
  factor; dc02: features ARE the space; dc03: *perceptual content* is the
  space, with exemplar anchoring).
- **Fidelity vs. accuracy.** Maximal fidelity coupling (the explanation object
  is the representation itself); the accuracy price is the R6 bet, and the
  push-vs-no-push + ID-residual ablations are the designed measurement of
  exactly this tension.
- **Attribution granularity.** Exemplar-level: coarsest in attribute terms (no
  per-attribute shares — R4 partial), finest in perceptual terms (a concrete
  garment, not an abstract profile). A genuinely different point on the axis
  than dc01/dc02, which is the point of this cycle.
- **R7 perceptual alignment vs. predictive strength.** dc03 takes the *opposite*
  resolution to the one sketched in the requirements (§4 suggested: strong
  internal features drive, perceivable vocabulary for read-out). Here the
  perceivable channel itself drives. If it holds accuracy (R6), R7's tension
  dissolves for this candidate; if not, the ID-residual ablation quantifies the
  price of perceptual purity. Either outcome is thesis material.
- **Identifiability.** Post-push, a prototype's meaning is maximally
  identifiable (it is a specific SKU). Distinctness across prototypes is the
  real question → D1, with the CL7-informed recommendation that the *hard
  greedy-unique push* (CL10 confirmed) is the reliable protection, not L_div.

### 5.3 Part-prototype pitfalls checklist (S5 survey, same category set as dc02's pass)

- *Prototype number trade-off (S5 §3.1.1):* as in host; K searchable; small K
  favors readability, exemplars make even large K individually readable.
- *Prototypes lack semantics (S5 §4.2):* answered by construction post-push
  (prototype = displayable garment); between pushes only approximately — D2.
- *Similarity-perception gap (S5 §3.1.2):* **the candidate's weak spot** = D5:
  latent visual similarity vs human-perceived similarity; encoder choice +
  purity eval; open risk, explicitly carried.
- *Cosine unreliability (S5 §4.4, Steck):* both sides of the item cosine are now
  constrained objects (q_i from frozen content + linear map; post-push p_k = some
  q_i), removing free-embedding norm pathologies on the prototype side entirely
  after push; E's norms remain learned. Softened vs host.
- *Prototype collapse/duplication:* real (CL5c produced [19,19,…] at toy
  scale!); D1; hard greedy push is the reliable fix (CL10), L_div only repels
  near-duplicates (CL7 falsified for exact duplicates).
- *Undepictable prototypes:* impossible post-push (every prototype is a real
  photo) — the pitfall this design eliminates most thoroughly.
- *Bottleneck information loss (S5 §3.3.2):* present in visual form — the item
  branch sees only what the photo (as encoded) shows; popularity/fit/quality are
  invisible → the R6 bet. dc03's "bottleneck" is one-sided like dc02's (user
  branch untouched), but perceptual instead of categorical.
- *Spatial/part-specific pitfalls (misalignment, patch visualization):* n-a —
  whole-garment prototypes by explicit deviation #1 (parts = future work; the
  user's Rudin-GC#4 whole-to-whole caveat from the 2026-06-11 scratchpad entry
  applies here and is acknowledged: dc03 reintroduces whole-to-whole comparison
  in the visual modality; the part-based refinement is the named future-work
  escape).

### 5.4 Degeneration-protection gate (v1.1)

| Mode (§3.5) | Status |
|---|---|
| D1 exemplar duplication | protected by proposed (unstacked) constraint — **hard greedy-unique push recommended over L_div** (CL7/CL10) |
| D2 mid-training drift | protected structurally (push-consistent eval/save discipline, trainer hook) + `push_period` knob |
| D3 push shock | partially protected by inherited `sim_proto` (CL5b/CL6: proximity keeps push a small perturbation) + `push_start_epoch` knob |
| D4 popularity capture | protected by proposed (unstacked) stratified-pool constraint; until stacked: **open risk, monitored** (exemplar popularity distribution is a one-line audit) |
| D5 encoder semantic gap | **unprotected — open risk** (not lossable; encoder choice + qualitative purity eval are the only handles) |
| D6 projection collapse | protected by proposed (unstacked) L_orth, with CL8 caveat (restores from *near*-collapse only — sufficient under random init) |

### 5.5 Step 4b falsifications folded in as known defects

1. **CL7 — L_div cannot split exact duplicates** (zero-gradient at cos=1).
   Consequence absorbed into D1's recommendation: prefer the hard greedy-unique
   push; L_div is at best a pre-push repellent.
2. **CL8 — L_orth cannot restore rank from exact collapse** (gradient stays in
   E's column space). Acceptable under random init; recorded so nobody later
   relies on L_orth as a rescue mechanism rather than a preventive.

## 5b. Devil's advocate

**Critique (subagent, received the candidate file only; verbatim):**

> 1. **R6 is not an "open bet" — the design's own citations already settle it against the headline, and the fallback destroys the headline's claims.** VBPR wins *because* it keeps γ_i and β_i alongside θ_i; the visual-only comparator in that same lineage (IBR) loses the warm regime. dc03 deletes all per-item CF capacity — no ID factor, no item bias — so it cannot express popularity, quality, fit, price tier, or seasonality, which dominate top-N fashion retail where the validation metric is hit_ratio@10 on head-heavy transactions. The escape hatch, the ID-residual ablation (`q_i = E v_i + b_i`), is the model that will actually win — but once b_i is inside q_i, the pushed exemplar's representation is no longer determined by its photo, two visually identical garments get different activations, and "your score is driven by similarity to *this garment*" becomes a false statement about the mechanism. So either the headline fails R6, or the rescue silently falsifies R2/R7. The document rates R6 "weak/open" but never confronts that its R2-strong and R7-strong ratings are conditional on the branch that fails R6.
> 2. **The entire mechanism is a rank-d linear function of a frozen encoder the design refuses to fine-tune, and D5 is misfiled as one risk among six when it is the whole design.** Every load-bearing object — q_i, both halves of the tied score (t* AND t̂ = Wᵗ E v_i), the push targets, the read-out — lives inside E·CLIP(image). H&M photos are homogeneous studio shots (flat-lay vs on-model, shared backgrounds, per-department photography conventions); frozen CLIP/ImageNet similarity on such catalogs is notoriously dominated by layout and photography style, and CLIP embeddings have well-documented anisotropy (narrow-cone geometry) that a single bias-free linear map cannot undo — D6's orthogonality knob treats the symptom in E, not the collapse already present in v_i. The doc's own §5.4 admits D5 is "unprotected — open risk," then still awards R7 "strong." A design whose raison d'être requirement rides entirely on an unprotected, untestable-before-implementation property has not earned that rating. (Also unresolved: H&M articles have multiple images; which one is v_i? VBPR's one-image limitation is quoted in §2.2 and then never addressed.)
> 3. **Push at H&M scale is structurally prone to hub capture, and the recommended fix quietly abandons the theoretical cover.** Cosine-argmax over 27K–91K items in a frozen high-dimensional embedding space is a textbook hubness setting: a few hub garments are the nearest neighbor of a large fraction of directions, so prototypes will collide on hubs — the toy check already produced duplicate exemplars at n≈40 (CL5c: [19,19,…]), and D4 (popularity capture) is "open risk, monitored." The endorsed remedy, greedy-unique push (CL10), assigns later prototypes to their best *remaining* item, not their argmax — which is precisely the configuration Theorem 2.1's proximity precondition excludes. And Theorem 2.1 is invoked as "the template argument that pushing barely moves the score" despite none of its conditions transferring: it assumes L2 log-similarity, class-allocated prototypes, and a class-structured cluster cost; the host has shifted cosine, no classes, and a max-based regularizer that is only asserted, not shown, to be "the literal Clst counterpart." CL5b is a 5-seed toy ratio, not a bound. The small-perturbation story is decoration.
> 4. **The scoring-sheet read-out explains one signed factor of one half of the score and calls it faithfulness.** The rendered explanation covers c_k = û_k·t*_k, i.e. the ûᵀt* half only; the other half, u*ᵀt̂, is waved at as "the host's existing user-side explanation" and can dominate the score. Within the explained half, û_k = (Wᵘu)_k is an unconstrained, sign-free CF quantity — ProtoPNet's entire stage-3 convex sparsification existed to prevent exactly this negative-reasoning soup, and deviation #4 drops it with "the sparsity job falls to the top-3 rendering." Rendering the top 3 of K=50 signed, mutually canceling contributions is post-hoc cherry-picking inside a nominally intrinsic explanation — the failure mode R2 was written to exclude. "Your affinity 0.91" is a number with no user-side grounding whatsoever.
> 5. **The R4 story is worse than "partial" — the displayed catalog row is a correlational decoration that can actively mislead.** The model never consumes the exemplar's attributes; they are attached to the prototype by the accident of which garment won an argmax in encoder space. If a prototype fires on photography pose or color cast (per point 2), the UI still confidently shows "Skinny H.W. Denim, Dark Blue, Divided" as its meaning. This is precisely the published critique of ProtoPNet's read-out ("this looks like that" does not mean *because* of that), and dc03 amplifies it by promoting the source image's metadata to an "exact by construction" profile. Bitwise equality of p_k with q_{i*} (CL5c) grounds the prototype in a *vector*, not in the human-legible properties the explanation displays. Exactness of the pointer is being sold as exactness of the meaning.
> 6. **The zero-image path is a degenerate ranking block, not a "neutral activation," and the whole candidate is hostage to data that currently does not exist.** With v_i = 0: t* is the all-ones vector and t̂ = 0, so *every* imageless item receives the identical score Σ_k û_k for a given user — they rank as an arbitrarily ordered tie block, and cold imageless items are strictly worse off than under any ID-based model. Calling this "non-degenerate" is spin. Meanwhile the local image set holds 86 of ~100+ subdirectories — plausibly a majority of the catalog is imageless *right now*, the original loss cause is unknown, and the mitigation is an unverified 25 GB re-download. §4.5 declares "no budget flags" and §4.3 all-green S5 for a design whose input tensor has not been shown to exist in full. dc01 and dc02 have no such existential dependency; under equal uncertainty budgets this alone should rank dc03 last on feasibility, not "flag, not a design defect."
> 7. **The novelty wedge is a training-schedule tweak, and the prior-art probe was phrased to miss the real neighbor.** The composition is VBPR's θ_i = E f_i verbatim, plus ProtoMF unchanged, plus a periodic argmax-assignment step; ProtoECGNet (found by the probe itself) already ports exemplar-push outside vision. All four probe queries use the design's own vocabulary ("push," "exemplar," "prototype"), so they structurally cannot find the actual nearest neighbor of the *user-facing artifact*: item-item "recommended because you liked / it resembles X" explanations, the oldest and most deployed explanation style in recommendation — including anchor-based CF, which the host repo literally ships as a baseline. The design's own §3.6 concedes the pushed exemplar "should be ≈ the top-1 nearest item anyway" by construction — i.e., the deliverable a committee sees is reproducible by the host's existing post-hoc `top_k_items` explainer, and the thesis-grade difference collapses to "the pointer is a parameter, not a query." That is a claim about implementation aesthetics, not about explanation quality, and nothing in the evaluation plan measures a user-visible difference between the two.
> 8. **The evaluation plan cannot detect its own failure modes.** No user study and no quantitative interpretability metric is proposed for the one axis the candidate exists for: whether exemplar explanations are perceived as faithful/useful (D5's "qualitative purity check" is eyeballing). The push-vs-no-push ablation measures the *accuracy* cost of grounding but nothing about explanation gain. Model selection interacts destructively with the mechanism: D2's discipline restricts eval/checkpointing to post-push states, so with patience=10 and push_period-coarse eval, early push shocks (D3) can trigger early stopping on a perturbation artifact, and the genuinely best (between-push) model is never eligible — an accuracy tax appearing nowhere in the "≈1–1.5× host" budget. The hyperopt space silently grows by push_period, push_start_epoch, encoder choice, div/orth weights while claiming "same trial protocol."
> 9. **The parameter story reads as elegance but describes an imbalance the convergence claim ignores, and R8's "one idea" needs a six-mode life-support system.** After the transplant, 16.43M trainable user parameters face ~39K item-side parameters (400:1); the optimizer's easiest path to fit interactions is to memorize through the user table against a nearly rigid item geometry — the item branch's 98% capacity cut is reframed as "−9% total." Convergence under cyclic disturb-and-resettle training is asserted "same order," self-flagged as "arguable." And the R8-strong rating counts the push as half of one published idea while excluding from the ledger: a new trainer hook, push scheduling (two hyperparameters), a checkpoint/eval discipline that is "load-bearing" for R2, a greedy assignment algorithm required because the soft knob is provably inert on exact duplicates (CL7), an orthogonality preventive with a known blind spot (CL8), and an offline encoder pipeline. dc03 has more moving parts whose *failure silently degrades interpretability claims* than either registered candidate; parsimony of the slogan is not parsimony of the mechanism.
> 10. **The 4b sanity check validates tautologies and skips everything that decides the design.** All ten claims are shape/gradient/argmax algebra that could not have failed (dot products decompose; gradients flow through both terms of a sum; argmax is idempotent) — and two *still* falsified as stated. The three questions on which dc03 lives or dies — R6 accuracy of a pure-visual branch, push behavior at catalog scale, encoder semantic gap — are exactly the three "explicitly NOT toy-checkable" items. The check's rhetorical function in the document is to project rigor onto a design whose risk mass sits entirely in the unchecked column; a candidate whose confirmed-claims list is disjoint from its failure-modes list has been verified in name only.

**Response (concede / rebut / open risk, point by point; amendments A1–A6 at the end are the visible edits):**

1. **Partially concede, partially rebut.** Conceded: R2/R7-strong are conditional
   on the pure-content branch, and §5.1 should have said so → **A1**. Also
   conceded: popularity/quality/season are not in the photo — the headline
   genuinely cannot express them. Rebutted on two facts: (a) IBR is not the right
   doom-precedent — it is unpersonalized retrieval with a *fixed* metric; dc03
   trains E, W^t, P^t and the whole user branch on the recommendation objective,
   so the visual space is feedback-adapted; how far that closes the gap is
   exactly what the experiment measures. (b) The ID-residual is not a silent
   falsifier of R2: post-push the prototype still equals a real garment's *full
   model representation* — the exemplar pointer stays exact; what the residual
   dilutes is R7 purity ("the photo determines the activation"), stated in A1's
   conditionality note. The fork (pure = R7-maximal/R6-risky vs residual =
   R6-safer/R7-diluted) is now explicit and is the candidate's honest shape.
2. **Concede the centrality, rebut the rating logic, answer the factual gap.**
   Conceded: D5 is the design's central external dependency, not one risk among
   six — the §5.4 table understates its weight; encoder anisotropy and
   studio-shot homogeneity are real, and E (bias-free, linear) cannot repair a
   collapsed v_i geometry → D5 note strengthened in **A1**; encoder screening
   (compare CLIP vs fashion-tuned vs H2-style attribute-grade features on a
   retrieval sanity task BEFORE training) promoted from "choice" to a required
   pre-step in **A3**. Rebutted: R7 rates the explanation *vocabulary* (photos —
   what the user sees), which no encoder defect changes; what the encoder decides
   is whether the *similarity semantics* behind that vocabulary are honest — now
   explicit as R7-strong-conditional-on-D5 (**A1**). Factual: the H&M Kaggle set
   ships **one image per article_id** (`images/0xx/<article_id>.jpg`) — the
   multiple-image objection does not arise on this dataset (to be re-verified at
   data-readiness time, **A3**).
3. **Concede the theorem framing, keep the mechanism, mark hubness.** Conceded:
   Theorem 2.1's conditions (L2 log-similarity, class-allocated prototypes,
   class-structured Clst) do NOT transfer; the file's "template argument" phrasing
   oversold it — downgraded to a heuristic analogy; the actual support is
   sim_proto's proximity force (CL6) + the CL5b toy trend + a **direct
   measurement now added to the eval plan**: report |Δscore| and Δmetrics across
   every push (**A3**). Hubness/hub-capture at catalog scale: conceded as a real
   mechanism behind D4/D1, added to the audit (exemplar concentration +
   popularity of exemplars per push, **A3**). Greedy-unique tension conceded:
   uniqueness trades proximity; alternative duplicate handling (merge/prune
   duplicated prototypes instead of reassigning) noted as the conservative
   variant in **A3**. Open risk overall.
4. **Concede the rendering, rebut the "worse than host" implication.** The
   two-halves structure and signed û are the HOST's explanation design (A1
   Eq. 12 and its §5.2 practice) — dc03 inherits them unchanged; by R2's own
   text the full decomposition is available at inference, and truncation-for-
   display is presentation. But conceded as real: an honest rendering must
   (a) show both halves' magnitudes, (b) summarize the residual mass ("47
   smaller/negative terms totalling −0.3") instead of hiding it → **A2**. Also
   conceded: dropping ProtoPNet's stage-3 sparsification leaves negative-
   reasoning soup possible; an unstacked sparsity knob (L1 on W^u/W^t rows,
   same `get_and_reset_loss()` hook) is added as constraint candidate **D7**
   (**A2**) — designed, not stacked, per v1.1 rules.
5. **Concede the epistemics, fix the claim, keep the display.** Correct: push
   grounds the prototype in a *vector*; the catalog row is the exemplar's
   metadata, not the cause of similarity. The explanation must therefore CLAIM
   only "because it is visually similar to this garment [photo]" — with the
   attribute row as *identification* of the garment, never as the *reason* →
   rendering language amended (**A2**), and R4 prose already carried this
   ("holistic," "partial"); the "exact by construction" phrase now explicitly
   scopes to the pointer, not the meaning (**A2**). The residual risk (fires on
   photography style, not garment) is D5 again — open.
6. **Concede both the phrasing and the asymmetry.** "Neutral" was optimistic:
   v_i = 0 ⇒ identical score Σ_k û_k per user ⇒ an ordered-arbitrarily tie
   block — now stated plainly (**A3**), with the count to be measured at split
   time and a decision rule (if imageless items are more than a trivial share of
   the *split*, exclude them from the item universe or defer to a hybrid — not
   this candidate). Data-hostage point conceded as a genuine feasibility
   asymmetry vs dc01/dc02: the §4.4 flag is upgraded to a **blocking
   precondition for any real run** (**A3**). It remains a data-logistics task
   with a known, cheap, verifiable fix — but until executed, dc03 is the only
   candidate whose input tensor is unverified.
7. **Rebut the collapse, concede the missing measurement and the missing
   related-work lineage.** Rebutted: (a) ACF anchors are free latent vectors —
   no exemplar identity, no image, no push; (b) "because you liked X"
   item-item explanations point into the *user's own history*; dc03's exemplars
   are system-level taste archetypes shared across users — different semantics
   and different failure modes; both lineages now owed a Related-Work
   distinction (**A6**). (c) The difference vs post-hoc top-1 is not aesthetics:
   post-hoc profiling has an unbounded, unmonitored approximation gap (the exact
   R2 complaint in requirements §5), while the pushed pointer's gap is zero by
   construction and its *accuracy price* is measurable (push ablation). BUT
   conceded: no explanation-quality metric in the plan measured a user-visible
   difference — the fidelity metric (G1, SPINRec-style path integration) is now
   named as the instrument comparing intrinsic-pushed vs post-hoc read-outs
   (**A3**). If that comparison shows no difference, the candidate's R2 value
   thins to architecture hygiene — recorded as the honest stake of that
   experiment.
8. **Concede substantially.** (a) Explanation-side eval added: G1 fidelity +
   exemplar-audit metrics (**A3**) — a user study stays out of scope per the
   requirements' non-goals. (b) The early-stopping × push-cadence interaction is
   real and subtle: mitigation specified — validate every epoch as the host
   does; make checkpoint/explain eligibility push-consistent; count patience on
   the checkpoint-eligible series only; report the between-push best as a
   diagnostic (**A4**). Conceded that this is an accuracy-relevant protocol
   detail the budget table did not price. (c) Hyperopt growth conceded and
   bounded: push_period/push_start_epoch are FIXED by convention (ProtoPNet
   practice), encoder choice is a pre-training screening (**A3**), div/orth
   knobs are unstacked — the hyperopt space proper stays the host's (**A4**).
9. **Partial concede, partial rebut.** The 400:1 parameter asymmetry and
   "memorize through the user table" pressure: conceded as a real dynamic — but
   it is a *family* property, not a dc03 defect: dc02's item side is 53K against
   the same 16.43M user table, and leaning items on content while users stay
   free is the shared design intent of the whole Phase-4 program. What IS
   dc03-specific and conceded: the interpretability claims depend on more
   operational discipline (push cadence, checkpoint rule, audits) than either
   registered candidate — R8 measures *conceptual* units, and by that measure
   the design is one idea; measured in *operational fragility*, dc03 is the
   heaviest so far. Both statements now stand side by side (**A5**).
10. **Rebut the purpose, accept the residue.** 4b's protocol role is exactly
    "falsify what is cheaply falsifiable" — tautology-adjacent claims are the
    deliverable, and two of ten still failed, one (CL5c duplicate collision)
    materially changing a design recommendation. The residue is true and was
    already stated in the viability notes: the deciding questions (R6, scale
    push behavior, D5) are empirical. No design-time process can check them;
    what it can do is name the instruments — now done (**A3**). No change to
    the 4b verdicts.

**Visible amendments (dated 2026-07-07, applied as this section's addendum —
earlier sections left as written per protocol):**

- **A1 (ratings conditionality).** §5.1 R2/R7 "strong" are hereby annotated:
  *conditional on the pure-content headline surviving R6* (the ID-residual
  fork keeps R2's exemplar pointer exact but dilutes R7 purity), and R7
  additionally *conditional on D5* (encoder similarity ≈ human garment
  similarity). D5 is re-weighted from "one of six modes" to **the candidate's
  central external dependency**.
- **A2 (read-out honesty rules + D7).** §3.4 rendering amended: explanations
  claim visual similarity only ("because it looks like this garment"); the
  exemplar's attribute row is identification, not cause; both score halves'
  magnitudes shown; truncated remainder summarized, signs included. New
  unstacked constraint candidate **D7 — connection sparsity**: L1 on W^u (and
  optionally W^t) rows, hook `get_and_reset_loss()`, the ProtoPNet-stage-3
  analogue against negative-reasoning soup; Rashomon-tunable.
- **A3 (eval-plan additions & preconditions).** (i) Pre-training encoder
  screening on a garment-retrieval sanity task (CLIP vs fashion-tuned vs
  H2-style attribute features) — required before any training run; (ii)
  per-push |Δscore| / Δmetric report (replaces the Theorem 2.1 appeal); (iii)
  exemplar audit per push: duplicate count, popularity distribution, hub
  concentration; (iv) G1 (SPINRec) fidelity metric comparing intrinsic-pushed
  vs post-hoc top-1 read-outs — the explanation-side headline experiment; (v)
  imageless-item count measured at split time, tie-block behavior recorded
  (v_i=0 ⇒ per-user constant score), threshold rule for exclusion; (vi) §4.4
  upgraded: image re-download + verification is a **blocking precondition** for
  real runs; (vii) merge/prune noted as conservative alternative to
  greedy-unique duplicate handling.
- **A4 (training protocol).** Validate every epoch (host behavior);
  checkpoint/explain eligibility restricted to push-consistent states; patience
  counted on the eligible series; between-push best reported as diagnostic.
  push_period / push_start_epoch fixed by convention, NOT hyperopted; hyperopt
  space otherwise = host's.
- **A5 (R8 clarification).** R8-strong refers to conceptual granularity (one
  idea). Operationally, dc03 carries the most interpretability-load-bearing
  discipline of the candidate set (push cadence, checkpoint rule, audits) —
  stated as its own cost line, not hidden under parsimony.
- **A6 (Related Work additions).** Distinguish from: item-item
  "because-you-liked-X" explanation lineage (user-history-anchored, not
  taste-archetype-anchored) and ACF anchors (latent, no exemplar identity);
  add ProtoECGNet as cross-domain evidence the push template ports, with the
  recommendation gap intact.

## 6. Fingerprint

- **Features-entry:** item representation = linear projection of a frozen image
  embedding (`q_i = E v_i`, VBPR template); the item ID table is gone; prototypes
  live in the projected visual-content space.
- **Grounding:** exemplar identity by projection — prototypes periodically pushed
  onto real catalog garments (ProtoPNet push, whole-item, cosine-argmax); exact
  (parameter-level) post-push, approximate between pushes.
- **Read-out:** case-based scoring sheet — per-prototype contributions
  û_k·t*_k, each prototype rendered as its exemplar garment's photo (+ catalog
  row as identification); no per-attribute shares.

Differs from dc01 on all three axes (categorical Σ-embeddings × unenforced
geometry × per-feature shares) and from dc02 on all three axes (observed
multi-hot bottleneck × by-construction attribute profiles × per-attribute
shares) — satisfying this cycle's max-distinctness brief.
