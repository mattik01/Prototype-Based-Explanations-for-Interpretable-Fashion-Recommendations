# dc01 — Feature-Composed Item Factors under the Double-Tied Prototype Layer

> Produced by the design-candidate protocol (`Master/docs/design_candidate_protocol.md`), cycle 1, 2026-06-11.
> Status: **fI-ProtoMF — re-host amendment applied (2026-07-12, gated); SC.1b-delta
> (adversarial pass on the re-hosted concept) pending.**
> Previously: concept amended (scrutiny SC.2, 2026-07-12) — findings F-DC01-01..07
> applied as dated visible amendments (dossier: `Master/docs/scrutiny/sc01_feature_item_proto.md`).
> *(File renamed from `dc01_draft.md` after Step 1 fixed the seed.)*

## ⟳ HOST REDIRECTION (2026-07-12, user decision at the SC.3 gate)

**dc01 is re-hosted from UI-ProtoMF (the double tie) to I-ProtoMF → the candidate becomes
"fI-ProtoMF": feature-composed item factors on the I-ProtoMF host.** Rationale: ProtoMF
deliberately separates U / I / UI; the thesis lineage now mirrors that — each variant is
grounded separately against its own host (fI vs I, later fU vs U, finally fUfI vs UI), so
no candidate carries an unexplainable passenger part. In fI the ENTIRE score is u·t*:
every scored point flows through the grounded item-prototype surface (no projection half,
no user-prototype opacity inside the candidate).

**What this supersedes in this document (pending the gated re-host amendment cycle):**
- §3.2/§3.3 (modules, shapes): the item branch (PrototypeEmbedding ← shared
  FeatureEmbedding) survives verbatim; the FeatureEmbeddingW projection branch, the
  Concatenate wrappers, and the user prototype branch DROP OUT. User side becomes
  I-ProtoMF's free embedding u ∈ R^{K_t}. Factory/config re-pointing lands at the SC.3
  re-run; `ft_type='feature_item_proto'` (finally literally accurate) is retained.
- §3.4 read-outs: 1–3 survive with the simplification s_k = u_k·t*_k (per-prototype
  contribution); **read-out 4 (exact linear-half attribution, F-DC01-07) is PARKED** — it
  requires a projection branch and returns at the lineage's fU/fUfI stages (the dossier's
  host-specificity note anticipated exactly this: "I-ProtoMF would be all-cosine
  attribution"). The share-identifiability discipline (F-DC01-06) applies unchanged.
- §3.6: headline baseline becomes **I-ProtoMF (`item_proto`)**, already in the S0.7
  reference fleet; the F=0+ID keystone reduction target becomes `item_proto`.
- §3.7 deviation 2 (instance-sharing tie) and the double-tie framing: parked to the merge
  stage. §4 accounting: no W matrices; user table N×K_t.
- Steck disposition (§3.4 amendment block): the "unprotected projection half" residual
  risk **disappears** (there is no dot-product half); the trains-through-cosine steel-man,
  Rashomon multiplicity, and label confounding carry unchanged.
- Everything composition-level — F-DC01-01/02/03/04/05/06 substance, cold ID-drop
  machinery, degeneration modes M1–M4, decoupling-mass results — **carries unchanged**.

The UI-hosted build (this document's original object) is not discarded: it is the
skeleton of the lineage's final **fUfI-ProtoMF merge stage** and will be re-derived there
through its own design-candidate + scrutiny cycle. The re-derivation happened at the
gated re-host amendment cycle — **the section immediately below is the current
fI-ProtoMF concept**; the UI-hosted text after it is the historical record.

## ⟳ Re-host amendment (2026-07-12 — applied after the gated SC.1a-delta cycle)

> Gate record (2026-07-12): transfer table F-DC01-01..07 ratified (six carry —
> F-DC01-02 strengthened — F-DC01-07 parked to fU/fUfI); this amendment bundle, the
> requirements re-rating, and the claims delta (C3′/C4′/C5′) approved by the user.
> Dossier: `sc01_feature_item_proto.md`, "Re-host amendment cycle — SC.1a-delta".

### A. Mechanism (fI-ProtoMF)

Host I-ProtoMF (A1 §3.2): item prototypes P^t ∈ R^{K_t×d}; item activation
t* = [1 + cos(q_i, p_k)]_k (Eq. 7); the user is a **free vector u ∈ R^{K_t}** living in
item-prototype-similarity space; score I(u,i) = u·t* (Eq. 8); loss Eq. 10 with the
item-side inclusion regularizers Eq. 9. dc01's one idea is unchanged: **q_i = Σ_{f∈f_i}
e_f (+ ID row)** replaces the free item embedding feeding the prototype layer.
Everything composition-level in this document (per-feature shares, identifiability
discipline, cold ID-drop, degeneration modes M1–M4, decoupling-mass basis) applies to
this mechanism verbatim.

### B. Modules & shapes (supersedes §3.2/§3.3)

Item branch verbatim: `PrototypeEmbedding(embedding_ext=FeatureEmbedding)`. Drops:
`FeatureEmbeddingW`, both `Concatenate` wrappers, the user prototype branch, both W
matrices. User side = I-ProtoMF's free `Embedding(n_users, K_t)` (factory
'prototypes'/Item-Proto shape). `ft_type='feature_item_proto'` retained — finally
literally accurate.

| step | tensor | shape |
|---|---|---|
| user embedding | `u = U[u_idxs]` | (B, K_t) |
| item feature lookup | `feat_ids[i_idxs]` (+ID col) | (B, 1+n_neg, F+1) |
| item composed emb | `q_i = Σ_f e_f` | (B, 1+n_neg, d) |
| item proto branch | `t* = sim(q_i, P^t)` | (B, 1+n_neg, K_t) |
| score (RecSys dot) | `u·t*` | (B, 1+n_neg) |

Parameters: feature table (V+M)×d (with ID), prototypes K_t×d, user table **N×K_t**
(dimension K_t, not d — host property). No W matrices.

### C. Read-outs (supersedes §3.4's host-dependent parts)

Read-outs 1–3 survive, simplified; read-out 4 (linear-half attribution) is **parked**
with the projection branch (returns at fU/fUfI):

1. **Per-prototype contribution:** s_k = u_k·t*_k — now covering the **entire** score.
   Item-discriminating form: u_k·(t*_k − 1), per the rank-inertness property below.
2. **Per-feature shares** c_{f,k} of each activation: unchanged (C1 verbatim; grouped
   shares + determinacy annotation per F-DC01-06 unchanged).
3. **Global prototype profile** cos(e_f, p_k): unchanged (S4 naming input).

**Rank-inert baseline (new host-specific sharpening, verified C4′):** S(u,i) =
Σ_k u_k(1 + cos_ik) = Σ_k u_k + Σ_k u_k·cos_ik, and the first term is item-independent
— identical for every item the user ranks. Hence **100% of the item-discriminating
score decomposes exactly over (prototype, feature) pairs**: S(u,i) − baseline(u) =
Σ_k Σ_{f∈f_i} u_k·c_{f,k}. Scope honesty: rank-inert means inert for ranking/eval (it
also cancels in softmax/BPR-style losses; under BCE it affects calibration, not order).
Rendering still shows the +1·u_k baseline lines (magnitude honesty), but the thesis's
attribution claim now covers the full ranking signal — strictly stronger than the UI
formulation, where only the û·t* half decomposed this way.

**S0.2 score-share obligation, fI form:** Block-U's share is structurally zero (no
user-prototype surface in the candidate). The binding disclosure becomes the
**feature-explained fraction of the item-discriminating score** (metadata rows vs the
ID row's share of Σ_k u_k·c_{f,k}) — the same M3 instrument, now covering the whole
score.

**Rendered example (re-derived; illustrative numbers, canonical 5):**

> **Why "Slim-fit dark-blue jeans" for you?** Score S = Σ_k u_k·t*_k. Top contributor:
> item prototype #12 *(profile: Trousers Men · Denim · Dark colours)* — your affinity
> u₁₂ = 0.8 × activation t*₁₂ = 1.7 → contribution +1.36 (baseline part 0.8·1 = +0.80,
> rank-inert; feature part 0.8·0.7 = +0.56). The feature match 0.7 comes from:
> department "Trousers Denim" (+0.45), colour group "Dark Blue" (+0.15), section
> "Menswear" (+0.07), graphical appearance "Solid" (+0.02), product type "Jeans"
> (+0.01), item-specific ID share (0.00).

Affinity coefficients u_k are CF-learned (they are the personalization, per-prototype
legible); there are no user-prototype lines — the grounded-vs-ungrounded visual split
of the UI rendering has no object here. Grouped-share/determinacy annotations
(F-DC01-06) apply to the feature lines unchanged.

### D. Baselines & ablations (supersedes §3.6's host anchors)

Headline baseline = **I-ProtoMF (`item_proto`)**, already in the S0.7 reference fleet.
F=0+ID keystone reduction target = `item_proto` (claim C5′). Unchanged: LightFM rows
(`lightfm_tags`/`_ids`), ablation arms (`_noid`, `_f0`), the S0.3 cold protocol as
governing spec, and the ablation-gap interpretation rule (§3.6).

### E. Deviations (supersedes §3.7's host anchors)

Deviations 1 (composition under a cosine prototype layer), 3 (per-feature
decomposition as intrinsic read-out), 4 (no bias terms; no-popularity-channel
consequence), 5 (field selection), 6 (cold ID drop — landed, 865de3b) carry with the
host renamed. **Deviation 2 (instance-sharing tie) parks**: fI has a single consumer
of the `FeatureEmbedding`; there is nothing to tie. R1's satisfaction basis is
re-derived in §G.

### F. Cost accounting (supersedes §4's numbers)

With ID: feature table (426+13,651)·d + prototypes K_t·d + user N·K_t. Delta vs
`item_proto` at equal (d, K_t): **+426·d ≈ +27K parameters at d=64** (~3% of the item
table) — negligible. Without ID: item table shrinks 13,651→426 rows (~32×). Step cost:
one gather becomes F+1=6 gathers + a sum; the K_t·d cosine still dominates →
**≈1.05–1.1× vs `item_proto`**. S5 checklist unchanged (indexed gathers, in-batch
losses, no precompute).

### G. Requirements re-rating (supersedes §5's table where host-dependent)

| Req | Was (UI) | Now (fI) | Basis |
|---|---|---|---|
| R1 tied | strong | **strong (basis re-derived)** | The "double-tie mirror" phrasing was host-stage-specific. The substance — features wired into the representation that drives the score, no parallel reranker — is satisfied maximally: features are the item representation and the score has **no other path**. The instance-sharing tie returns at the merge stage. |
| R2 intrinsic | strong (Block-I caveat) | **strengthened** | The Block-I/Block-U split dissolves: every scored quantity flows through the grounded prototype surface; the whole item-discriminating score decomposes exactly per (prototype, feature). Scope: u_k stays CF-learned; profile-meaning validity still rests on the Steck disposition's checks. |
| R3 prototype | strong | strong | Unchanged. |
| R4 grounded | partial | partial | M1 unenforced sharpness is host-independent; instruments unchanged. |
| R5 sparsity | strong | strong | Cold machinery + caveats (F-DC01-03/04) carry; the whole cold score is feature-computed. |
| R6 dense accuracy | partial | **partial (re-anchored)** | The empirical bet is now fI vs **I-ProtoMF**. |
| R7 vocabulary | partial | partial | Unchanged. |
| R8 parsimony | strong | **strong (cleaner)** | Same one idea on a strictly simpler host. |
| S1 decoupled | strong | strong | Unchanged. |
| S2 user-side | strong | **strong (re-scoped)** | User-side grounding is an explicit lineage stage (fU/afU), not a dc01 extension point. |
| S3 images | partial | partial | Unchanged. |
| S4 naming | strong | strong | Unchanged. |
| S5 pipeline | strong | strong | Unchanged (lighter, if anything). |

### H. Degeneration modes (re-check verdict)

M1–M4 transfer unchanged (item-branch geometry is verbatim; the acting regularizers
are I-ProtoMF's Eq. 9, identical in form). User side = host behavior (free embedding;
negative u_k = legitimate anti-affinity). Fresh sweep found **no I-host-specific new
mode**; the one structural novelty — u multiplies t* directly — adds no degeneration
surface beyond M1's existing reading. Mode list judged complete for fI. Steck: the
"unprotected projection half" residual risk is **gone** (no dot-product half);
trains-through-cosine steel-man now covers the entire score; Rashomon multiplicity +
label confounding carry, with the committed SC.8 profile-validity spot-check.

## 0. Frame

- The candidate index is **empty** — no fingerprint region is occupied yet. All three axes
  (features-entry × grounding × read-out) are fully open; dc01 has no distinctness
  constraint to satisfy beyond being one clean idea (R8).
- Requirements re-read in full (version of 2026-06-11, incl. the spectrum reframing, R7
  perceptual-vocabulary balance, R8 parsimony, S4 LLM naming, S5 pipeline compatibility).
- The center of gravity of the requirements: R1 (tied integration) + R4 (prototypes
  readable as attribute profiles) + R5 (cold items representable from features alone) +
  R2/R3 (intrinsic, prototype-shaped read-out). A first candidate should aim squarely at
  this core rather than at a niche.
- Key tension to keep live from §4: user-visible features are weak predictors
  (colour/pattern ~1.3× lift) while internal taxonomy is strong (department 5.2×) —
  where features enter and what vocabulary grounds the prototypes are separate choices.
- Open niches (for later cycles, not dc01's obligation): user-side features (S2), image
  channel (S3), and at most 1–2 `S1-contrast` non-prototype vehicles across the set.

## 1. Seed selection

Corpus swept across all themes (B backbone, C/D prototypes, E concepts/attention,
F disentanglement, H fashion, §I forward citations). Index is empty, so **distinctness
is trivially satisfied for every shortlisted seed** — stated once here rather than
per-seed.

### Shortlist

**Seed 1 — B5 LightFM / B3 SVDFeature template × A1 ProtoMF double-tie:**
*the item factor is constructed as a (linear) combination of feature embeddings, and that
factor feeds the unchanged item-prototype branch.*
- *Motivation gate:* serves R5 directly (a cold item's factor — hence its prototype
  activations — exists from features alone), R1 (features enter the very representation
  the tied prototype layer scores; no parallel reranker), and R4 (prototype activations
  decompose over the item's feature embeddings → per-feature attribution of prototype
  meaning).
- *Distinctness:* trivially satisfied (empty index).
- *R8 parsimony:* PASS — exactly one idea: "replace the free item ID-embedding with a
  feature-composed embedding in the item branch"; the prototype layer, losses, tie, and
  training loop are untouched. Same conceptual size as `user_item_proto` itself.

**Seed 2 — E1 Concept Bottleneck × prototypes-in-feature-space:**
*prototypes live in (or are mapped to) an explicit attribute space; an item's activation is
computed from its attribute profile, concept-bottleneck style.*
- *Motivation gate:* serves R4 maximally (a prototype literally *is* an attribute
  profile) and R2 (activations are the explanation); main risk is R6 (bottleneck may cost
  accuracy — the known CBM trade-off, E3).
- *Distinctness:* trivially satisfied (empty index); vs. Seed 1 it differs on grounding
  (explicit attribute profile vs. decomposition) and entry (prototypes in feature space).
- *R8 parsimony:* PASS — one idea (move the prototype space into attribute space).
  Retained as a strong lead for a future cycle.

**Seed 3 — B2 RLFM regression prior on item embeddings:**
*a feature-based regression predicts each item's latent factor; the factor remains free but
is shrunk toward the prediction; cold items use the prediction.*
- *Motivation gate:* serves R5 (cold-start via feature regression), but fails the spirit
  of R2/R4 — the prototype space stays purely CF; features influence *where* embeddings
  sit, not *what prototypes mean*; explanations would remain post-hoc. REJECTED on the
  motivation gate.
- *Distinctness:* trivially satisfied.
- *R8 parsimony:* PASS (one idea) — but motivation gate already rejected it.

**Seed 4 — C1 ProtoPNet push/projection step on feature profiles:**
*periodically project each learned prototype onto the feature profile of its nearest real
item, so every prototype equals an actual item's attributes.*
- *Motivation gate:* serves R4 (prototype = concrete item profile) and R3; but the
  read-out is "this looks like that item" — close to the existing post-hoc `top_k_items`
  baseline (k=1, enforced in training), so its R2 gain is the smallest of the four. Also
  does nothing for R5 by itself (cold items still need an embedding to be compared).
- *Distinctness:* trivially satisfied; differs from Seeds 1–2 on the grounding axis
  (projection-to-real-items).
- *R8 parsimony:* PASS — one idea. Retained as a free lead (possibly as a *grounding
  add-on* studied separately).

### Chosen seed

**Seed 1 — B5 LightFM (Kula 2015) / B3 SVDFeature (Chen et al., JMLR 2012) template,
transplanted under A1 ProtoMF's double-tied prototype layer.** Provenance: **corpus**.

Why it won: it is the most central attack on the requirements' center of gravity
(R1+R4+R5 simultaneously) with the smallest mechanism; it is battle-tested at the
backbone level (LightFM is also the natural external baseline, B5; SVDFeature won two
KDD Cups); and it leaves the entire explanation-side design space (Seeds 2, 4) open for
later cycles on different fingerprint axes. The interesting *novel* part — per-feature
decomposition of prototype activations as the intrinsic read-out — sits exactly on the
open wedge corroborated by the §I sweep (all 31 ProtoMF citers CF-only).

## 2. Mechanism extraction

All statements verified against the local PDFs on 2026-06-11:
`B5_kula_2015_lightfm_metadata_embeddings.pdf` (arXiv:1507.08439v1),
`B3_chen_2012_svdfeature_jmlr.pdf` (JMLR 13:3619–3622),
`A1_melchiorre_2022_protomf.pdf` (RecSys '22, pp. 246–256).

### 2.1 LightFM (Kula 2015) — the seed's core

**Representation (§2.2).** Users and items are "fully described by their features":
user u has feature set f_u ⊂ F^U, item i has f_i ⊂ F^I. The model's parameters are
d-dimensional *feature* embeddings e^U_f, e^I_f and scalar *feature* biases b^U_f,
b^I_f. The latent representation of an entity is the **sum of its features' embeddings**
(§2.2, four unnumbered display equations):

> q_u = Σ_{j∈f_u} e^U_j   p_i = Σ_{j∈f_i} e^I_j   b_u = Σ_{j∈f_u} b^U_j   b_i = Σ_{j∈f_i} b^I_j

**Score (Eq. 1).** r̂_ui = f(q_u · p_i + b_u + b_i), with f = sigmoid for binary data.
**Objective (Eq. 2).** Bernoulli likelihood over positive/negative interaction sets,
trained by asynchronous SGD with Adagrad (§2.2).

**The MF↔CB spectrum (§2.3).** If feature sets contain only per-entity indicator
("ID") features, LightFM **reduces exactly to standard MF**. Adding metadata features
shared across entities (a) reduces parameter count vs. one embedding per entity,
(b) makes cold-start representable ("Latent vectors for indicator variables cannot
be estimated for new, cold-start users or items. Representing these as combinations
of metadata features … makes it possible to make cold-start predictions", §2.3 item 2),
and (c) does *not* reduce to a pure content model, because feature embeddings are
fitted by factorising the *interaction* matrix (§2.3).

**What they showed (Table 1, §6.1).** MovieLens 10M + Tag Genome (69,878 users /
10,681 items / ~10M interactions / 1,030 tags) and CrossValidated (5,953 users /
44,200 questions / 1,032 tags); warm and item-cold-start splits; mean ROC-AUC.
Cold MovieLens: MF 0.500 (= random), LightFM(tags) 0.707, LightFM(tags+ids) 0.716.
Warm MovieLens: MF 0.762, LightFM(tags+ids) 0.763 — i.e. **matches MF when dense,
rescues cold-start when sparse**. Notably "(tags+ids) outperforms (tags) slightly …
even though no embeddings can be estimated for movies in the test set" (§6.1) —
keeping ID features for warm items helps without breaking cold-start. Fold-in of new
items needs no retraining: representation = sum of feature representations (§7.1).

**Admitted limitations.** Performance "is predicated on the availability of
high-quality metadata" (§6.1); "no easy way of incorporating visual or audio features
in the present formulation" (§8). Domain note: Lyst is a *fashion* company; the paper's
running examples (denim jacket = denim + jacket) are garment metadata (§1, §2.1).

### 2.2 SVDFeature (Chen et al. 2012) — the seed's generalisation

**Model (§2, the single display equation).** With user/item/global feature vectors
α ∈ R^n, β ∈ R^m, γ ∈ R^s:

> ŷ(α,β,γ) = (Σ_j γ_j b^(g)_j + Σ_j α_j b^(u)_j + Σ_j β_j b^(i)_j) + (Σ_j α_j p_j)^T (Σ_j β_j q_j)

"Intuitively, we use a linear model to construct user and item latent factors from
features" (§2). This strictly generalises LightFM's sums: features may carry
**real-valued weights** (their album example uses weight 0.6, §3/Fig. 1), so binary
membership, counts, or confidence scores all fit. Losses: square, negative logistic
log-likelihood, smoothed hinge; SGD training (§2). Scale evidence: Yahoo! Music,
200M ratings, <2 GB memory (§1–2); won KDD Cup 2011 & 2012 (§1, abstract).
It is a 4-page toolkit paper — no explanation/interpretability claims are made.

### 2.3 ProtoMF (Melchiorre et al. 2022) — the host architecture

**U-ProtoMF (§3.1).** L^u learned user prototypes p^u ∈ R^d. A user's new
representation is the vector of shifted-cosine similarities to all prototypes
(Eq. 1): u* = [sim(u, p^u_l)]_l ∈ R^{L^u}, sim(a,b) = 1 + a^⊤b/(‖a‖·‖b‖) ∈ [0,2].
Score (Eq. 2): U-score(u,t) = Σ_l s^user_l with **s^user = u* ⊙ t** and the item
embedding t ∈ R^{L^u} — i.e. the item embedding lives in user-prototype-similarity
space, and the score decomposes linearly into per-prototype contributions (their
explanation device, §5.2). Loss (Eq. 3): sampled softmax cross-entropy + L2. Two
"inclusion criteria" regularizers (Eqs. 4–5): R_{P^u→U} = −(1/L^u) Σ_l max_i
sim(u_i, p^u_l) and R_{U→P^u} = −(1/N) Σ_i max_l sim(u_i, p^u_l), both approximated
in-batch; total loss Eq. 6 with weights λ1, λ2. *(Repo mapping: these are
`sim_proto_weight`/`sim_batch_weight` with `reg_proto_type`/`reg_batch_type` in
`PrototypeEmbedding`.)*

**I-ProtoMF (§3.2).** Exact mirror: t* ∈ R^{L^t} (Eq. 7), I-score = Σ_l s^item_l,
s^item = t* ⊙ u with u ∈ R^{L^t} (Eq. 8), regularizers Eq. 9, loss Eq. 10.

**UI-ProtoMF (§3.3) — the double tie.** Both units share base embeddings u, t ∈ R^d
via two linear maps (Eq. 11): û = W^u u ∈ R^{L^t}, t̂ = W^t t ∈ R^{L^u}. Score
(Eq. 12): UI-score(u,t) = u*^⊤ t̂ + û^⊤ t*. Loss: sum of both losses, L_rec counted
once on the UI-score (footnote 1). *(Repo mapping: `ft_type=prototypes_double_tie`,
built by `FeatureExtractorFactory` with `ConcatenateFeatureExtractors`.)*

**What they showed (Table 2).** UI-ProtoMF significantly beats MF, RBMF, ACF on
HitRatio@10/NDCG@10 across ml-1m, AmazonVid, lfm2b-1mon (e.g. ml-1m NDCG .383 vs
MF .326). Parameter accounting (§5.1): MF has (N+M)·d; UI-ProtoMF adds ≈4K·d.

**Crucial provenance pointer for dc01.** The authors *themselves* name our direction
as open future work: "we believe that including external features of users and items,
such as contextual information or audio features, into ProtoMF might further benefit
the interpretability of the prototypes" (§6, Conclusion and Outlook). The explanation
they ship is **post-hoc** at the prototype-meaning level: item prototypes are
interpreted by nearest items, user prototypes by synthetic maximally-activating
inputs (§5.2) — exactly the gap R2/R4 target.

## 2b. Prior-art probe

Probe target: the *adaptation* — "item representation composed from feature
embeddings (LightFM/SVDFeature-style), fed through a ProtoMF prototype-similarity
layer, with per-feature decomposition of prototype activations as an intrinsic
explanation". Searches run 2026-06-11 (WebSearch):

1. `prototype-based recommendation item metadata feature embeddings explainable
   "prototypes" side information matrix factorization`
2. `"ProtoMF" feature-aware OR content OR metadata OR attribute prototypes extension`
3. `LightFM metadata embeddings prototype layer explainable recommendation cold-start
   "sum of feature embeddings" prototypes`
4. `cold-start prototype recommendation "item features" OR "side features" intrinsic
   explanation attribute profile 2024 2025`

**Findings.**
- Nothing combining feature-composed entity embeddings with a prototype-similarity
  scoring layer was found. Query 3 explicitly surfaced only LightFM itself and
  tutorials; no prototype-layer-on-LightFM work exists in the results.
- Closest hits, each distinguished: **UIPC-MF** (arXiv:2308.07048; reading-list I3) —
  adds a user↔item prototype connection matrix, **CF-only**, no item features.
  **Meta-Prod2Vec** (arXiv:1607.07326) — side information regularises item embeddings,
  but no prototype layer and no explanation mechanism. **Modular debiasing of
  prototype-based RS** (I4) — uses *user* attributes as protected variables to remove
  from prototype representations, the opposite of grounding. **Attribute Prototype
  Network** (arXiv:2008.08290, zero-shot vision) — prototypes that predict attributes
  from CNN local features; conceptually adjacent but a vision-classification method
  with no recommendation/interaction modeling (worth citing as cross-domain analogue).
- **§I forward-citation re-check:** Semantic Scholar still reports **31 citers**
  (re-queried 2026-06-11 via API) — identical to the reading-list sweep, which found
  all 31 CF-only. No new citers to classify.
- ProtoCF (D1) remains the closest recsys prior art and is already distinguished in
  the reading list: it composes prototypes from *interaction few-shot* episodes, not
  from item metadata, and offers no feature-grounded explanation.

**Verdict: found nothing close.** The specific union — features→factors backbone ×
prototype-similarity read-out × per-feature attribution of prototype activations —
appears unpublished. ProtoMF's own conclusion flags it as future work (§6), which
supports both novelty and motivation. No refinement needed at this gate.

## 3. Adaptation design

Verified against the actual code on 2026-06-11 (`feature_extractors.py`,
`feature_extractor_factories.py`); H&M numbers from `Master/docs/hm_feature_analysis.md`.

### 3.1 The one idea

Replace the **item branch's free ID embedding** with a **feature-composed embedding**
(LightFM/SVDFeature: q_i = Σ_{f∈f_i} e_f), underneath the *unchanged* double-tied
prototype layer. Everything else — prototypes, shifted cosine, inclusion regularizers,
sampled-softmax loss, user branch — stays exactly ProtoMF. The user branch remains a
pure CF ID embedding (user-side features = S2, deferred; the architecture does not
preclude it — the same class would slot into the user branch).

### 3.2 Modules and classes

**New class `FeatureEmbedding(FeatureExtractor)`** in
`feature_extraction/feature_extractors.py`:

```python
class FeatureEmbedding(FeatureExtractor):
    def __init__(self, n_objects: int, feature_ids: torch.LongTensor,  # (n_objects, F)
                 n_features: int, embedding_dim: int,
                 use_id_feature: bool = True, max_norm: float = None):
        ...
        self.register_buffer('feature_ids', feature_ids)   # per-object feature-value ids
        n_rows = n_features + (n_objects if use_id_feature else 0)
        self.embedding_layer = nn.Embedding(n_rows, embedding_dim, max_norm=max_norm)

    def forward(self, o_idxs):                  # (B,) or (B, 1+n_neg)
        feat = self.feature_ids[o_idxs]         # (..., F)
        if self.use_id_feature:                 # LightFM "tags + ids" (§2.3 / Table 1)
            feat = torch.cat([feat, o_idxs.unsqueeze(-1) + self.n_features], dim=-1)
        return self.embedding_layer(feat).sum(dim=-2)      # (..., d)
```

Because the 4.0 analysis found **0% missingness** across all categorical item columns,
every item has exactly one value per field → `feature_ids` is a dense fixed-width
`(n_items, F)` LongTensor with field-offset encoding (global vocab = concatenated
per-field vocabs). No EmbeddingBag/padding machinery needed.

**`PrototypeEmbedding` change (one line of flexibility):** its `__init__` hard-codes
`self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)`
(feature_extractors.py:226). **I propose** an optional `embedding_ext:
FeatureExtractor = None` parameter; when given, it is used instead. Forward
(feature_extractors.py:299–324) is untouched — it only calls `self.embedding_ext(o_idxs)`.

**Factory branch** in `feature_extractor_factories.py` — new `ft_type =
'feature_item_proto'` (the reserved CLAUDE.md name), cloned from
`prototypes_double_tie` (factories.py:69–99) with two edits:
1. Build ONE shared `item_feat_embed = FeatureEmbedding(n_items, feature_ids, n_feats, d)`.
2. `item_proto = PrototypeEmbedding(..., embedding_ext=item_feat_embed)`; the item's
   projection branch becomes `FeatureEmbeddingW` = the *same* `item_feat_embed`
   instance followed by `nn.Linear(d, user_n_prototypes, bias=False)` (mirroring
   `EmbeddingW`). **The tie (R1) is module-instance sharing** — the current code ties
   by weight assignment (factories.py:93–94); with a shared instance the single item
   representation q_i feeds both halves of the UI-score, which is the same tie
   philosophy one level deeper.
User side: byte-identical to `prototypes_double_tie`.

**Config/data plumbing:** `confs/hyper_params.py` gains a `feature_item_proto` search
space (same hyperparameters as `user_item_proto` + `use_id_feature` ∈ {True, False});
*(Amended 2026-07-12, SC.2 — F-DC01-01: the searched-flag phrasing is superseded and
would be bad practice — a warm-accuracy-tuned search would winnow the False arm early,
so no fairly-tuned no-ID run would exist. As implemented, there are two separate fixed
configs (`feature_item_proto`, `feature_item_proto_noid`), each with its own full
search — charter C7 ablation discipline.)*
the experiment helper loads `data/hm/item_features.csv`, builds `feature_ids`, and
passes it via `ft_ext_param['item_ft_ext_param']`. `rec_sys/protomf_dataset.py`,
`rec_sys/rec_sys.py`, `trainer.py`: **zero changes** — the extractor interface
(`forward(o_idxs)`, `get_and_reset_loss()`) absorbs everything.

### 3.3 Forward pass with shapes (batch B, n_neg negatives, d emb-dim, K_u/K_t prototypes, F fields)

| step | tensor | shape |
|---|---|---|
| user proto branch | `u* = sim(Embedding(u_idxs), P^u)` | (B, K_u) |
| user proj branch | `û = W^u·Embedding(u_idxs)` (tied emb.) | (B, K_t) |
| user output (concat) | `[u*, û]` | (B, K_u+K_t) |
| item feature lookup | `feat_ids[i_idxs]` (+ID col) | (B, 1+n_neg, F+1) |
| item composed emb | `q_i = Σ_f e_f` | (B, 1+n_neg, d) |
| item proto branch | `t* = sim(q_i, P^t)` | (B, 1+n_neg, K_t) |
| item proj branch | `t̂ = W^t·q_i` (same q_i instance) | (B, 1+n_neg, K_u) |
| item output (concat, invert) | `[t̂, t*]` | (B, 1+n_neg, K_u+K_t) |
| score (RecSys dot) | `u*·t̂ + û·t*` | (B, 1+n_neg) |

Parameters: feature table `(|F^I| + M)×d` (with ID features), prototypes `K_t×d` and
`K_u×d` unchanged, projections `d×K_u`, `d×K_t` unchanged, user table `N×d` unchanged.

### 3.4 Intrinsic explanation read-out

All quantities exist in the forward pass; nothing is recomputed post-hoc:

1. **Per-prototype contribution** (= ProtoMF §5.2, unchanged): the K_t summands of
   `û·t*` and K_u summands of `u*·t̂`.
2. **Per-feature decomposition of each item-prototype activation (the new part — I
   propose):** with shifted cosine, t*_k = 1 + q_i·p_k/(‖q_i‖‖p_k‖)
   = 1 + Σ_{f∈f_i} [e_f·p_k / (‖q_i‖‖p_k‖)]. Each feature value f of the item
   contributes the *exact* additive share c_{f,k} = e_f·p_k/(‖q_i‖‖p_k‖) to the
   activation (the +1 shift is a feature-independent constant; the norm ‖q_i‖ is a
   shared scaling — both stated honestly in the rendering).
3. **Global prototype attribute profile (R4):** for any prototype p_k, score *every*
   feature value in the vocabulary by cos(e_f, p_k) — a learned, model-intrinsic
   attribute profile computed from parameters alone (cf. LightFM's semantic tag
   embeddings, §6.3 Table 2). This is what the `naming/` layer (and S4 LLM naming)
   would consume instead of post-hoc top-k item profiles.

**Rendered example (H&M-style, illustrative numbers).** *(Amended 2026-06-11 after
Step 5b: the original example's shares wrongly summed to t* instead of t*−1; the +1
shift baseline, the joint-normalization caveat, and the grounded/ungrounded
distinction were added.)*

> **Why "Slim-fit dark-blue jeans" for you?** Score 4.2, top contributor: item
> prototype #12 *(profile: Trousers Men · Denim · Dark colours)* — your affinity
> û₁₂ = 0.8 × item activation t*₁₂ = 1.7 → contribution +1.36.
> The activation t*₁₂ = 1 (baseline shift) + 0.7 (feature match), the feature match
> coming from: department "Trousers Denim" (+0.45), colour group "Dark Blue" (+0.15),
> garment group "Trousers" (+0.07), price band 3 (+0.02), product type "Jeans"
> (+0.01), item-specific ID share (0.00).

Rendering notes (honesty requirements): the per-feature shares are **jointly
normalized** through ‖q_i‖ — they are exact additive shares of the cosine part, not
counterfactual effects (removing a feature rescales the others; 4b C6). An optional
**counterfactual read-out** — re-scoring with one feature row removed from the sum —
uses only model-intrinsic quantities and can be rendered alongside. *(Demoted
2026-07-12, SC.2 — F-DC01-07: the re-score is off-manifold; see the §3.4 amendment
block below. The exact counterfactual for the linear half is read-out 4.)* The +1·û_k
baseline is shown, never silently absorbed. User-side lines (û, u*) are
**CF-grounded only** (post-hoc named, as in the host) and must be visually
distinguished from feature-grounded item-side lines.

A cold item with zero interactions renders the *same* explanation — q_i exists from
features alone (R5; with `use_id_feature=True` the untrained ID row adds noise — at
inference the ID column can be dropped for unseen items, **I propose**, following the
spirit of LightFM's fold-in §7.1).

#### §3.4 Amendments (2026-07-12, SC.2 — dc01 scrutiny; findings F-DC01-02/03/04/06/07)

**Read-out 4 — exact linear-half attribution (F-DC01-07, adversarial-pass gift).**
Because the projection branch is linear over the composition,
t̂ = W^t(Σ_f e_f) = Σ_f W^t e_f, the u*·t̂ score half decomposes per feature
**exactly**: contribution(f) = u*ᵀ(W^t e_f) — additive, norm-free (no ‖q_i‖
entanglement, contrast 4b C6), and exactly equal to the effect of removing f on that
half: a **true counterfactual with no off-manifold pass**. Units are user-prototype
coordinates ("which feature makes this item respond to user-prototype k") — so this
read-out is feature-attributable but not prototype-meaning-shaped on the item side;
the cosine half remains the headline (R3/R4). Host-specificity, recorded: the
read-out exists only because the double-tie has a projection branch — an I-ProtoMF
build would be all-cosine attribution (grounded prototypes, entangled shares), a
U-ProtoMF build all-linear (clean shares, no grounded item prototypes); UI carries
both surfaces at once — a quiet additional argument for the double-tie host. New
claims C7a/C7b in the claims spec (verified at the SC.2 re-check). The
identifiability discipline below applies to this read-out unchanged — linearity does
not rescue identifiability.

**Share identifiability across correlated fields (F-DC01-06; plain-language source
of truth: vault `2026-07-11_2310`).** Where values of different fields co-occur
(near-)deterministically, training touches their rows only through their sum:
gradients are identical in every step, so the **sum is trained while the difference
stays frozen initialization noise** (a gauge freedom) — the printed split between
such values is not a learned quantity (exhibit: lockstep toy, claims spec C8).
Discipline adopted for ALL rendered splits (shares in both score halves, and
prototype naming): (a) **grouped shares** over strongly-associated value groups are
the identified quantities — exact trained summands of the computed score, so the
coarser reading stays intrinsic (R2-safe); per-field splits shown as
informative-but-not-identified detail; (b) prototype **naming** treats group-mates
as one concept (within-group profile ordering is noise); (c) the research-side
characterization is **threshold-free**: each split's determinacy = the loss
curvature along its redistribution direction, proxied by the pair's decoupling mass
in the catalog (committed script; computed at field level for presentation and value
level where the mechanism lives); every rendered split carries its determinacy
annotation, and grouped views are summary visualizations only — no claim rests on a
cutoff. The general resolution is stated as a **delegated math problem** (production
instantiates it): *find a fixed, data-derived, human-performable transformation of
the value set into a reduced space where all splits are identifiable up to
tolerance* — merged groups must admit one coherent name ("Jeans/Trousers-Denim"
legal, "Jeans/Red" forbidden), preserving R7 through the map. Build-time token
merging is the strongest variant (the gauge never exists) — noted, not adopted here.
Seed-to-seed split stability: deferred to full-profile stage (C4 single-seed budget).

**Steck-caveat disposition (F-DC01-02; vault `2026-06-16_2102`).** The exact
per-dimension-rescaling argument does **not** transfer intact: ProtoMF's item branch
*trains through the same shifted cosine it reads out* — a diagonal rescaling of E is
not score-preserving here, so the train-on-dot/read-on-cosine mismatch Steck targets
is absent in that form (dc01-specific steel-man). What survives: (a) **Rashomon
multiplicity** — near-equal-loss solutions with different profile geometry; the
decomposition is canonical *given the trained model*, not across retrainings;
(b) **label confounding** (5b-6) — nothing pins e_"Dark Blue" to darkness beyond the
interactions of items carrying it; (c) the **projection half** is dot-product-based
and unprotected — but it is never semantically narrated (S0.2 score-share reporting
covers it). Mitigation committed: **SC.8 profile-validity spot-check** on the real
checkpoint (legible values vs the items carrying them); metric-swap / seed-stability
probes deferred to full-profile stage.

**Cold-inference amendments (F-DC01-03/04; the bias-gap temp note is hereby folded
in and retired with a pointer).** (a) There is a **second ID-keyed channel** beside
the embedding row: the per-item bias scalar (`RecSys` `use_bias`). A cold item's
bias row is an untrained zero — a silent depressant; LightFM composes its bias from
features (b_i = Σ_f β_f), so its cold guarantee is only **half-inherited** by dc01.
Disposition: the fleet runs `use_bias=0` (charter C3) so the channel is inert in
every scored run; the cold runner refuses `use_bias≠0` without a declared policy;
feature-composed bias remains an unstacked design seed — the evidence-triggered
ablation if SC.8 shows the cold-vs-all gap dominated by its popularity component
(vault `2026-07-12_0006`). (b) The ID-column drop (deviation 6) is spec-only until
this candidate's SC.3 implements it (F-S0-07 part b); until then the runner refuses
`_ids` cold eval loudly (guard, commit 30a570c). *(Status 2026-07-12, SC.3: LANDED —
the drop covers the nested branch (one shared FeatureEmbedding feeds both halves, so one
drop covers both), verified by dc_checks/dc01/t11; refusal lifted, replaced by a
post-drop abort; `_ids` cold runs unblocked. Commit 865de3b.)* (c) **Train/cold operating-point
mismatch (F-DC01-04):** a `use_id_feature=True` model always trains with the ID row
present; cold scoring drops it — an operating point never visited in training.
Precedent that it works regardless: LightFM tags+ids ≥ tags on cold (B5 Table 1,
§6.1). Unstacked mitigation if empirics disagree: K3 ID dropout (trains the cold
path). SC.8 interprets `_ids` cold rows with this caveat. Registered hypothesis: the
warm feature-explained fraction inversely predicts the cold-vs-cold → cold-vs-all
degradation (the ID row carries popularity + twin-separation; both are lost at the
drop — vault `2026-07-12_0006`).

### 3.5 Feature fields (from the 4.0 analysis, R7-aware)

**I propose** F = 6 fields, picking one level per redundancy branch (analysis §4):
`department_name` (197; strongest generalizer, lift 5.2), `product_type_name` (102),
`section_name` (51), `colour_group_name` (48), `graphical_appearance_name` (28),
price-band decile (10). Vocab |F^I| ≈ 436. Colour/pattern/price are weak predictors
(lift 1.1–2.4) but carry R7's user-visible vocabulary; taxonomy carries accuracy —
this design lets *both* enter and lets the decomposition show which did the work.
`product_code` stays out of the model (identity-like, 7,174 values; explanation/eval
machinery only, per analysis §1).

*(Amended 2026-07-12, SC.2 — F-DC01-01: superseded by the charter. The canonical
scrutiny-phase field set is the S0.5 **five** (department, product_type, section,
colour_group, graphical_appearance), V = **426** measured on V1; `price_band` is
deferred — transaction-derived, hence leakage-unsafe under the cold variant without
an imputation policy, and irrelevant for model *comparison* though possibly relevant
for absolute performance later (S0.5 gate caveat). The implemented config already
matches the canonical 5; this section's F=6 proposal is the historical record. §4's
"≈436" cost figures shift accordingly (V=426) — immaterial to every conclusion
there.)*

### 3.6 Baselines & the isolating ablation

- **CF-only `user_item_proto`** (always): same K, d, losses — the headline comparison.
- **Exact ablation isolating the one idea:** `feature_item_proto` with
  `use_id_feature=True` and *zero metadata fields* reduces architecturally to
  `user_item_proto` (LightFM §2.3 reduction) — any delta is the metadata's doing.
  Secondary ablation: `use_id_feature` ∈ {True, False} (warm accuracy vs. pure
  feature-grounding trade — LightFM Table 1's tags vs. tags+ids).
- **LightFM-style MF** (B5): the `mf` model with the same `FeatureEmbedding` item
  branch but *no prototype layer* — separates "features help accuracy" from
  "prototype layer adds/costs what".
- Existing post-hoc `top_k_items` + `naming/` explainer on CF-only ProtoMF: the
  explanation baseline to beat (R2/§5 of the requirements).

*(Amended 2026-06-11 after Step 5b:)*
- **Cold/sparse-item evaluation commitment:** the dense leave-one-out protocol alone
  cannot measure R5. Add (a) a held-out item split (all interactions of x% of items
  moved to test, LightFM §5 style) and/or (b) tail-item stratification of the standard
  metrics (à la ProtoCF), so the R5 claim is measured, not cited.
  *(Amended 2026-07-12, SC.2 — F-DC01-01: the "and/or" hedge is superseded — the
  **S0.3 cold-start spec is the governing document** (B5-faithful 20% stratified
  held-out variant, cold-vs-cold primary + cold-vs-all secondary, training-negative
  exclusion, single-config retrain convention); BOTH branches of the old menu were
  built (held-out variant + D4/TW stratified readouts). Machinery landed and audited
  at S0-build/S0.7.)*
- **Pre-implementation data check:** count distinct 6-field feature tuples among the
  13,651 V1 items to quantify the no-ID equivalence-class ceiling (items sharing a
  full tuple are indistinguishable when `use_id_feature=False`).
  *(Amended 2026-07-12, SC.2 — F-DC01-01: EXECUTED at S0.5 under the canonical 5:
  **6,517 distinct signatures / 67.5% of items share theirs / largest class 97**.
  Metric impact bounded: in the 1+99 sampled eval, ~0.06 expected same-signature
  competitors per row (~94% of rows contain no twin of the positive) — the ceiling
  is a full-catalog phenomenon, displayed honestly by the tie-block diagnostic, not
  by the headline metric. Two disclosures: exact ties resolve by the ranking code's
  deterministic convention (disclosed, not mechanized); the warm-item ceiling is a
  property of the no-ID **ablation arm** and is never blurred into the headline
  config.)*
- **Explanation-quality comparison** must include the LightFM-MF feature-attribution
  baseline (per-feature decomposition without prototypes), not only post-hoc top-k —
  it isolates what the prototype layer adds to the explanation.
  *(Note 2026-07-12, SC.2: this baseline exists in code since S0-build —
  `lightfm_tags` / `lightfm_tags_ids`, same `FeatureEmbedding`.)*
- **Ablation-gap interpretation rule** *(added 2026-07-12, SC.2 — F-DC01-01; source:
  vault gate discussion)*: the ids-vs-noid accuracy gap is a **bundle** — (1)
  popularity blindness (noid cannot rank signature twins apart, so within-class
  popularity is unrepresentable; likely the largest chunk in fashion data), (2)
  genuine taste-structure loss (the honest "cost of grounding"), (3) minus a
  sharing/regularization offset (LightFM §2.3-1). The raw gap is **never** quoted as
  "the cost of interpretability"; it is always read against the D4 popularity strata
  (deficit concentrated on popular items → component 1; flat → component 2).
  Classification note: the noid arm remains **collaborative filtering at attribute
  granularity** — its feature embeddings are fitted purely by factorizing the
  interaction matrix (LightFM §2.3) — NOT content-based filtering; what noid loses is
  item-level collaborative memory, not collaborativeness.

### 3.7 Deviations from the source papers (all "I propose")

1. Feature-composed embedding *under a cosine prototype layer* — in neither paper.
2. Tie via shared `FeatureEmbedding` instance (adapts factories.py:93–94 weight tying).
3. Per-feature additive decomposition of shifted-cosine activations as the intrinsic
   explanation (LightFM only offers embedding-similarity justification, §6.3 item 3;
   ProtoMF only per-prototype contributions, §5.2).
4. No LightFM bias terms b_u, b_i (ProtoMF's score form has none; keeps the host loss).
   *(Amended 2026-07-12, SC.2 — F-DC01-01, consequence made explicit: LightFM's
   feature-composed biases are a class-level popularity channel; dropping them means
   the noid arm has **no popularity channel at all** — strictly more constrained than
   the published LightFM(tags) precedent. Expectations for the ablation gap are set
   accordingly (see §3.6 interpretation rule); the cold-side implications are in the
   §3.4 amendment block (F-DC01-03).)*
5. Field selection (3.5) from the 4.0 redundancy analysis rather than "all columns".
6. Cold-inference ID-column drop (3.4). *(Status note 2026-07-12, SC.2: spec-only
   until this candidate's SC.3 implements it — F-S0-07(b); the cold runner refuses
   `_ids` cold eval until then (guard, 30a570c). See §3.4 amendment block for the
   second ID-keyed channel (bias) and the train/cold mismatch caveat.)*
   *(Status 2026-07-12, SC.3: LANDED and verified — t11; refusal lifted; commit
   865de3b.)*

### 3.8 Interpretability-constraint design *(step-3 re-run under protocol v1.1, 2026-06-11)*

**Inheritance.** `sim_proto`/`sim_batch` (the inclusion criteria, host Eqs. 4–5/9;
`reg_proto_func`/`reg_batch_func` applied in `PrototypeEmbedding.forward`,
feature_extractors.py:318–322) survive **byte-identically** — they act on the
batch–prototype cosine matrix and are agnostic to how `o_embed` was produced. Two
semantic shifts to keep in view:
1. "Every prototype owned by some *item*" becomes "owned by some *feature
   composition*". Without ID rows, items sharing a 6-field tuple are one point — the
   in-batch similarity matrix contains duplicate rows, so the regularizers see fewer
   distinct objects; coverage pressure is preserved per *composition*, diluted per
   *item*.
2. The host's L2 term now shrinks shared feature rows that serve many items —
   effective per-item regularization grows with feature sharing, while rarely-updated
   ID rows feel it relatively harder. Not a defect, but the λ_L2 sweet spot may move.
Also inherited: the existing `max_norm` constructor knob now caps *feature-row* norms
— a free hard constraint, relevant to M2 below.

**New degeneration modes** (beyond what the inherited regularizers cover; M1–M3
were identified in Steps 5/5b and are consolidated here per the v1.1 audit):
- **M1 — diffuse prototype profiles:** nothing forces a prototype's cos(e_f, p_k)
  profile to be sharp/sparse; a diffuse profile reads as nothing.
- **M2 — frequent-value dominance:** high-frequency rows ("Solid", 58% of items)
  accumulate gradient mass; a large ‖e_f‖ dominates ‖q_i‖ and flattens everyone
  else's shares (4b C6 mechanics).
- **M3 — ID-row signal routing:** with `use_id_feature=True`, SGD can drain
  discriminative signal into ID rows, collapsing the feature-explained fraction
  (5b point 3).
- **M4 — profile overlap:** prototypes distinct in latent space may still carry
  near-identical *feature profiles* — distinctness in the explanation vocabulary is
  not what the inherited loss measures. *(Corrected 2026-07-12, SC.2 — F-DC01-05:
  the original parenthetical "(which `sim_proto` rewards)" misstated the host loss.
  No term rewards prototype distinctness directly — Eqs. 4–5 are inclusion criteria;
  separation arises as an **indirect, data-spread-mediated secondary effect of the
  coverage pair** (chiefly `reg_batch`: covering a spread-out entity cloud forces
  prototypes apart; collapsed duplicates waste coverage), empirically demonstrated on
  the host by the reg-sweep's prototype-centred clusters and rising purity (vault
  2026-06-15_1238, `reg_sensitivity/ANALYSIS.md`). dc01-relevant residue: the effect
  is only as strong as the composed item cloud's directional spread — see the
  spread-mediation note under the 4c table.)*

**Constraint candidates — designed, NOT stacked (all "I propose"; adopting any one
is the user's mechanism decision, and would route its checkable claims through a 4b
re-run):**
- **K1 → M1, profile-sharpness penalty.** a_k = softmax_f(cos(e_f, p_k)/τ) over the
  |F^I| ≈ 436 metadata rows (ID rows excluded); L_prof = (1/K_t) Σ_k H(a_k), weight
  λ_prof, temperature τ. Hook: parameters-only accumulator in the item
  `PrototypeEmbedding` (no data dependency), rides `get_and_reset_loss()`. Cost:
  K_t·|F^I| ≈ 28K dot products per step — negligible. Effect: prototypes anchor to
  few feature values. Risk: over-sharpening → single-feature detectors (capacity ↓).
- **K2 → M2, feature-row norm control.** Hard variant **free in existing code**:
  `max_norm` on the embedding table. Soft variant: per-field norm-variance penalty
  L_norm = Σ_field Var_{f∈field}(‖e_f‖), weight λ_norm; hook in `FeatureEmbedding`.
  Effect: no single value's row dominates ‖q_i‖. Cost trivial.
- **K3 → M3, ID-routing control.** Soft: L_id = (1/M) Σ_i ‖e_ID(i)‖², weight λ_id.
  Training-trick alternative: DropoutNet-style (B10) ID-row dropout with prob
  p_drop in `FeatureEmbedding.forward` — forces feature reliance *and* trains the
  cold path it will face at inference. Effect: raises the feature-explained
  fraction; this is the knob that walks the R4↔R6 frontier (§5 note). Cost trivial.
- **K4 → M4, profile-distinctness penalty.** L_dist = mean_{k≠k'} cos(a_k, a_{k'})
  over K1's profile vectors (shares K1's cosine matrix), weight λ_dist — the
  feature-space analogue of `sim_proto`. Cost: K_t²·|F^I| ≈ 1.8M mults per step —
  still negligible. Effect: prototypes distinct *in the explanation vocabulary*.

**Rashomon-tunable knobs** (requirements §4 note): inherited λ_sim_proto,
λ_sim_batch, max_norm; proposed λ_prof/τ, λ_norm, λ_id or p_drop, λ_dist. Tuning
protocol sketch: fix a tolerance ε on validation hit_ratio@10 around the
accuracy-optimal configuration; within the ε-set, maximize interpretability metrics
(profile entropy ↓, feature-explained fraction ↑, profile distinctness ↑) rather
than letting accuracy-driven hyperopt pick arbitrarily among near-ties.

*(Amended 2026-07-12, SC.2 — F-DC01-05 with user extensions at the gate:)* honesty
edits to this sketch. **Levers:** with K1–K4 unstacked, the actual levers are the
inherited weights (λ_sim_proto, λ_sim_batch, max_norm) plus structural knobs
(n_prototypes and others) — **indirect but host-demonstrated** (the reg-sweep moved
top-k purity 0.70→0.87→collapse); whether and how the inherited regularizers act as
indirect knobs for sharpness/distinctness specifically is a committed analysis
question, not an assumption. **Metrics:** profile sharpness and distinctness are
liked but **not exhaustive** — the interpretability-metric set is itself a research
question (literature notions include that the *features used* be human-understandable
rather than opaque/constructed — the R7 axis). **Contingency, pre-declared:** if the
empirics show diffuse/overlapping profiles anyway, adopting a purpose-built knob
(K1/K4) is a mechanism revision through its own gate + toy re-check. **Timing:**
Rashomon tuning enters the thesis late (the "Interpretability quantified" part); if
the knobs then prove insufficient, interpretability-targeted regularizers are a
future lineage candidate ("merged prime prime") — explicitly not dc01's concern.

## 4. Cost & feasibility

All estimates **Fermi-style**, V1 `hm_1_month` numbers (13,651 items, 623K train
interactions; feature vocab |F^I| ≈ 436 per §3.5), symbols: N users, M items, d
embedding dim, K_u/K_t prototypes, F = 6 feature fields.

**Parameter delta vs. CF-only `user_item_proto`** (whose count is
N·d + M·d + (K_u+K_t)·d + d·(K_u+K_t)):
- With ID features (default): **+|F^I|·d ≈ +436·d** — at d = 64 that is ~28K extra
  parameters, ~3% of the item table alone (M·d ≈ 874K). Negligible.
- Without ID features: **−M·d + |F^I|·d** — the item table shrinks ~31× (13,651 → 436
  rows). A parameter *reduction*; LightFM §2.3-1 argues this aids generalisation
  under sparsity.
- GPU memory for the `feature_ids` buffer: (13,651 × 7) int64 ≈ 0.7 MB. Trivial.

**Training-time expectation.** Per item scored, the change is 1 embedding gather →
F+1 = 7 gathers + a sum over 7 vectors of length d (~6·d adds). The downstream
shifted-cosine against K_t prototypes (~K_t·d ≈ 4,096 mults at d = K = 64) already
dominates; expected per-step multiplier **≈ 1.05–1.1×**. Epochs/convergence: no
strong argument either way — fewer effective item parameters should help statistical
efficiency (LightFM §2.3-1), but feature rows aggregate gradients from many items,
which can slow per-item fitting. Expectation: same order; flagged as unknown, not a risk.

**S5 pipeline-compatibility checklist:**
- Sampled `(batch, 1+n_neg)` scoring preserved? **Yes** — forward stays indexed by
  `o_idxs`; the feature lookup is itself an indexed gather.
- No full-catalog pass per step? **Yes** — the inclusion regularizers remain in-batch
  (PrototypeEmbedding forward is untouched).
- Item/user branch cheap indexed computation? **Yes** — 7 row gathers + sum.
- Losses fit `get_and_reset_loss()`? **Yes** — `FeatureEmbedding` adds no loss;
  existing accumulators unchanged.
- Anything expensive per item? **No** — `feature_ids` built once at startup from
  `item_features.csv` (seconds, CPU). No encoder, no cache needed.

**Budget gate.** Training cost ≈ Phase-3 CF `user_item_proto` H&M runs × ~1.1 with
the same 100-config TPE hyperopt protocol (+1 boolean hyperparameter). No >10-day
risk; dev iterations stay in the same class as existing hm_baseline runs.
**No flags raised.**

## 4b. Math sanity check

**Claims spec:** `Master/temp/dc_checks/dc01/claims_spec.md` (mechanism definition +
six claims, no derivations). In brief: **C1** exact per-feature decomposition of the
shifted-cosine activation; **C2** cold-item activations non-degenerate (differ across
items with different features, identical for identical features + zero ID, non-constant
across prototypes); **C3** gradient reaches a feature row from *both* halves of the
UI-score (the tie); **C4** score = sum of K_u+K_t independent per-prototype scalars;
**C5** F=0 (ID-only) reduces exactly to the plain-embedding model; **C6** scaling one
feature row perturbs other rows' contributions only via the shared ‖q_i‖.

**Viability self-assessment.** C1–C6 are algebra/gradient-flow properties — all
meaningfully toy-testable. Two claims were **skipped as untestable at toy scale,
deliberately**: (i) "features improve cold-start accuracy / don't regress dense
accuracy" (R5/R6) and (ii) "frequent feature values (e.g. 'Solid', 58% of items)
do not dominate the composed embeddings" — both only materialize through training
dynamics on real data; they are carried as open empirical risks into Step 5, not
as confirmed properties.

**Subagent report (verbatim):**

> All six claims checked and confirmed.
>
> **Script:** `C:\Users\glaes\...\Master\temp\dc_checks\dc01\check_claims.py`
> **Run:** `& "C:\Users\glaes\miniforge3\envs\protomf\python.exe" ...\Master\temp\dc_checks\dc01\check_claims.py` (torch, float64, <2 s, no training/datasets). Adversarial sweep: 10 seeds per claim, F ∈ {0,1,2,3,5,7,11}, param scales 0.1/1/10, (K_u,K_t) ∈ {(3,4),(1,1),(7,2)}, scale factors α ∈ {10, 0.1, −3}.
>
> **C1 — confirmed.** tstar_k − 1 = Σ_r c_{r,k} with c_{r,k} = (E[r]·P_t[k])/(‖q_i‖‖P_t[k]‖) holds exactly: max abs error 3.33e−16 over 10 seeds × 7 feature-set sizes (incl. F=0 and odd F) × 3 parameter scales.
>
> **C2 — confirmed.** Cold items (zero ID rows): (a) different feature sets give clearly different activations (min over seeds of max |Δtstar| = 0.363); (b) identical feature sets + zero ID rows give bit-identical tstar (max diff 0.0); (c) activations non-constant across k (min spread max−min = 0.504).
>
> **C3 — confirmed.** Backpropagating each score half separately to a feature row E[r] of the scored item: both gradients nonzero in all 10 seeds — min ‖∂(ustar·that)/∂E[r]‖ = 3.97, min ‖∂(uhat·tstar)/∂E[r]‖ = 0.322. The ustar·that half reaches E[r] via that = W_t q_i; the uhat·tstar half via the cosine in tstar.
>
> **C4 — confirmed.** S equals the sum of K_u + K_t independent scalar contributions {ustar_k·that_k} ∪ {uhat_k·tstar_k}: max abs error 3.55e−15 across seeds and (K_u,K_t) shapes including (1,1).
>
> **C5 — confirmed.** With F=0 (q_i = E[ID row] only), all forward outputs (ustar, tstar, uhat, that, S) are bit-identical (max diff 0.0) to a model with a plain per-item embedding table Q[i] = E[V+i], for all 6 toy items × 10 seeds.
>
> **C6 — confirmed.** Scaling one feature row E[r0] by α ∈ {10, 0.1, −3}: pairwise ratios c_{r,k}/c_{r',k'} among the other rows unchanged (max deviation 3.64e−12, pure float-division noise), and their absolute values all rescale by the single common factor ‖q_old‖/‖q_new‖ (max |c_after − factor·c_before| = 1.11e−16). The coupling acts only through ‖q_i‖, as claimed.
>
> No claim was ambiguous enough to require an alternate reading; C2(b) was tested with exactly-zero ID rows as the spec's literal construction (tiny-random ID rows would make the equality approximate rather than exact).

**Outcome:** no falsifications this cycle. The two skipped training-dynamics claims
remain open empirical questions for the eventual experiments.

## 5. Requirements evaluation

Requirements doc re-read in full before this step (2026-06-11 version).

| Req | Rating | Assessment |
|---|---|---|
| R1 tied integration | **strong** | Features *are* the item representation; both UI-score halves consume the same composed q_i (4b C3: gradient reaches feature rows from both halves). Nothing is rerankered or bolted on; the training objective is the host's, with features inside it. Strain: only the item side is feature-aware — that is a scoping choice (S2 deferred), not a tie violation. |
| R2 intrinsic explanations | **strong** | Per-prototype contributions (4b C4) and per-feature activation shares (4b C1) are literally summands of the inference-time score. Nuance: the *global* prototype profile (cos(e_f, p_k) over the vocab) is read from learned parameters rather than from any single forward pass — still model-derived with no data-side reconstruction, unlike top-k-items; a purist could call it "parameter read-out" rather than inference read-out. |
| R3 prototype paradigm | **strong** | The explanation remains "which prototypes, and what they mean"; only the *meaning* channel changes from item clouds to attribute profiles. The host's prototype layer is untouched. |
| R4 feature-grounded prototypes | **partial** | The attribution half is as strong as it gets — exact additive shares, not saliency (C1). The *profile* half is enabled but not enforced: nothing makes a prototype align sharply/sparsely with few feature values; a diffuse profile would read poorly. Also, with `use_id_feature=True` part of every activation flows through the opaque ID row — honest (rendered as "item-specific: +x") but it caps the feature-explained fraction. This is the candidate's main open risk; empirics (profile entropy across prototypes) must decide. |
| R5 sparsity robustness | **strong** | q_i exists from features alone; cold activations are non-degenerate and feature-sensitive (C2). LightFM's cold-start evidence (Table 1: 0.500→0.716 AUC) is the strongest external support any seed offers. Open: *quality* of cold ranking is a training-dynamics question (skipped 4b claim i). |
| R6 dense-regime accuracy | **partial** | Unproven by construction. For it: LightFM(tags+ids) matched/beat MF warm (Table 1) and `use_id_feature=True` preserves full per-item capacity (the model can learn around features). Against it: shared feature rows constrain item embeddings early in training and could drag dense accuracy; the host's UI-ProtoMF margin over MF could shrink. This is the empirical bet of the candidate. |
| R7 perceivable vocabulary | **partial** | Both vocabularies enter (colour/pattern/price + taxonomy, §3.5) and the decomposition *shows* which earned the score — R7's tension becomes measurable instead of decided a priori. But nothing constrains the read-out toward user-visible terms: if department dominates (lift 5.2 says it will), explanations will lean on semi-internal taxonomy names. Mitigations: department names are semi-legible ("Trousers Denim"); S4 naming can verbalise; a vocabulary-constrained grounding is a *different* candidate (future cycle), not a knob to stack here (R8). |
| R8 parsimony | **strong** | One idea: swap the item ID table for feature composition. The host is recovered exactly at F=0 (C5); the one-flag ablation isolates the idea. No auxiliary losses, no extra stages. |
| S1 decoupled variant | **strong** | The baseline set (§3.6) natively spans the spectrum: post-hoc explainer on CF-only (degenerate end), LightFM-style MF without prototype layer, and the tied candidate — the tied-vs-decoupled comparison falls out without extra design. |
| S2 user-side features | **strong** | Not precluded: the identical `FeatureEmbedding` class slots into the user branch (age band, club status); the factory change is symmetric. Nothing in dc01 assumes item-onlyness. |
| S3 image features | **partial** | The sum composition admits a precomputed image vector as one more "feature row" (projected to R^d, fixed or fine-tuned) — architecturally open, but it would break the pure categorical-lookup profile (needs a precompute cache, S5) and the image attribution would be one opaque lump, not a legible value. ~~Plus the known partial-image-set caveat.~~ *(Amended 2026-07-12, SC.2 — F-DC01-01: stale; the image set was verified COMPLETE 2026-07-11 — 105,100 jpgs, subdirs 010–095. The architectural reservations stand.)* Possible later, not natural here. |
| S4 LLM naming | **strong** | The global attribute profile (R4 read-out 3 in §3.4) is exactly the structured input a naming LLM wants — better-grounded than post-hoc top-k lift profiles, and the LLM stays outside the mechanism. |
| S5 pipeline compatibility | **strong** | Full checklist pass (§4): indexed gathers only, in-batch losses unchanged, no precompute needed, ~1.05–1.1× step cost. |

*(Note added 2026-06-11 after 5b:)* the R4 and R6 "partial" ratings are **coupled
through `use_id_feature`**: the setting that maximizes feature-explained activations
(no ID rows) constrains capacity, and vice versa. The ablation pair measures this
fidelity–capacity frontier; any mitigation knob (ID-norm penalty, ID dropout / B10)
is a mechanism revision for the user to decide on, not part of dc01.

### §4 design tensions

- **Where do features enter:** resolved as "feature encoder feeding the
  `PrototypeEmbedding` branch" (injected `embedding_ext`) + new factory `ft_type`.
  Prototypes deliberately stay in the *latent* space, not feature space — preserves
  the host's accuracy capacity; the prototypes-in-feature-space region is left to a
  future candidate (Step 1 Seed 2).
- **Fidelity vs. accuracy:** the tie is maximal (features are the representation),
  but `use_id_feature` is a continuous escape valve — the model can route signal
  through ID rows, which *degrades the feature-explained fraction visibly* (the
  ID-residual term) rather than corrupting the attribution. The True/False ablation
  quantifies the trade — this directly answers the tension's "quantify this".
- **Attribution granularity:** per-attribute-*value*, the finest level on the menu;
  since every item has exactly one value per field, per-value = per-field — no
  roll-up ambiguity.
- **R7 alignment vs. predictive strength:** both enter; the decomposition measures
  which wins; no constraint imposed (see R7 row — open, by design this cycle).
- **Identifiability:** the host's `sim_proto`/`sim_batch` regularizers are kept
  unchanged; **no feature-aware analogue is added** (R8). Risk: prototypes distinct
  in latent space may still have overlapping *feature profiles* (e.g. several "dark
  menswear" prototypes). *(Precision 2026-07-12, SC.2 — F-DC01-05: latent
  distinctness itself is not directly rewarded either — it is the indirect coverage
  effect; see the corrected §3.8 M4 and the 4c table note.)* A feature-space distinctness regularizer is the natural
  follow-up knob — deliberately not stacked into dc01; noted as the F2 caveat
  (disentanglement ≠ interpretability) demands empirical checking either way.

### Part-prototype pitfalls pass (reading-list S5 taxonomy)

- *Prototype collapse:* mitigated by the host's entropy/max regularizers (unchanged);
  slight added risk since items sharing features share embedding mass — watch
  activation spread.
- *Prototype quality/quantity:* K_u/K_t remain hyperopt dimensions, as in the host.
- *Semantic gap:* narrowed (meaning is stated in features, not inferred from item
  clouds) but not closed — diffuse profiles possible (R4 risk).
- *Prototype redundancy:* host reg + field de-redundancy (§3.5) help; feature-profile
  overlap remains possible (identifiability note above).
- *Attribution deception ("looks like that, but isn't why"):* the *arithmetic* layer
  is exact algebra (C1), not saliency estimates — no estimation gap. *(Softened
  2026-06-11 after 5b:)* this does not guarantee *semantic* faithfulness: shares are
  jointly normalized (C6), the +1 shift is feature-independent, and e_f's meaning is
  CF-trained — rendering and validation must address these (see §3.4 notes).
- *Vision-specific pitfalls (spatial misalignment, patch artefacts):* n-a for
  categorical features; their tabular analogue (attribution to vocabulary the user
  cannot perceive) is tracked under R7.

### Degeneration-protection table *(gate 4c; step-5 re-run under protocol v1.1, 2026-06-11)*

Modes and constraint candidates K1–K4 are defined in §3.8. No identified mode unlisted.

| degeneration mode | status |
|---|---|
| Prototype–object collapse / coverage (host modes) | **protected by inherited regularizer** (`sim_proto`/`sim_batch`, unchanged) — with the duplicate-row dilution caveat in the no-ID configuration (§3.8 inheritance note 1) |
| M1 — diffuse prototype profiles | **unprotected — open risk**; proposed unstacked constraint K1 (profile-sharpness penalty); primary empirical metric: profile entropy |
| M2 — frequent-value dominance | **partially protected by inherited knob** (`max_norm`, already plumbed — hard cap on row norms); soft variant K2 proposed, unstacked |
| M3 — ID-row signal routing | **unprotected — open risk**, but *measured by design* (the `use_id_feature` ablation charts the R4↔R6 frontier); proposed unstacked K3 (ID-norm penalty or B10-style ID dropout) |
| M4 — prototype profile overlap | **unprotected — open risk**; proposed unstacked K4 (profile-distinctness penalty, feature-space analogue of `sim_proto`) |

*(Table corrections 2026-07-12, SC.2 — F-DC01-05, approved itemized at the gate:)*
**M2 row precision:** `max_norm` never acts on profiles (they are angle-only); it
acts upstream by **equalizing vote weights** — the ingredient lengths in the
composed sum, whose *relative* sizes decide q_i's direction. That protects against
*loudness*-dominance (one long row hijacking the direction, plus share flattening
via ‖q_i‖) but NOT against *omnipresence*: an equal-length e_"Solid" still votes in
58% of all item sums, tilting the cloud regardless ("a choir where everyone quietly
hums the same note still has that note in every chord"). **The omnipresence residue
is recorded as a definite risk and must be tested** (user directive). **M1/M4
spread-mediation:** the force that separates prototypes is coverage of the item
cloud, and it is only as strong as that cloud's spread — dc01's composition shares
ingredients across items, which can contract the cloud into a narrower cone and
weaken the separation proportionally. M1/M4 therefore stand as **open risks with a
named, measurable weakening mechanism** (not expected failures). Instruments
committed for SC.8: t* activation spread, composed-cloud spread/tilt, prototype
pairwise angles, profile pairwise overlap, per-value share vs value frequency.

All statuses are flags, not vetoes (spectrum rule). Adopting any K-knob is a
mechanism revision for the user; if adopted, its checkable claims go through a
Step-4b re-run.

### Known defects from Step 4b

None falsified. Two claims were *not testable* at toy scale and stand as the
candidate's empirical bets: (i) cold/dense accuracy behaviour (R5/R6), (ii) whether
high-frequency feature values ("Solid", 58% of items) come to dominate composed
embeddings and flatten the decomposition.

## 5b. Devil's advocate

Black-box critique (subagent saw only this file). **Verbatim:**

> 1. **The "novel" contribution is the distributive law.** The candidate's claimed wedge — "per-feature decomposition of prototype activations as the intrinsic read-out" — is the observation that a dot product distributes over a sum: q_i·p_k = Σ_f e_f·p_k. C1 verifying this to 3e−16 is not validation of a mechanism, it is a unit test of linearity. Everything that could not fail was tested (C1, C4, C5 are algebraic identities; C3 is "gradients flow through a differentiable graph"); everything that determines whether the design works — cold/dense accuracy, frequent-value domination — was "skipped as untestable." The math-sanity section is six confirmations of trivia and two deferrals of the actual claims. The prior-art probe finding "nothing close" is equally consistent with the obvious alternative explanation: a known backbone (LightFM, 2015) under a known layer (ProtoMF, 2022) with the linearity of dot products as the connective tissue is too incremental to have been published, not too original.
>
> 2. **The prototype layer is doing no explanatory work the baseline doesn't already do.** The candidate's own §3.6 baseline — "LightFM-style MF with the same `FeatureEmbedding` item branch but no prototype layer" — has a score u·q_i = Σ_f u·e_f that decomposes per feature *exactly*, by the same algebra as C1, and gets the same R5 cold-start property for free. So per-feature attribution, the headline novelty, does not require prototypes at all. What the prototype layer uniquely adds is the prototype *vocabulary* — and the meaning of that vocabulary rests entirely on the cos(e_f, p_k) profile read-out, which the candidate itself concedes is unenforced and possibly diffuse (R4 "partial"). If the profiles come out diffuse, dc01 is the LightFM baseline plus an interpretability layer that failed to interpret, at 1.1× the cost and with ProtoMF's accuracy margin at risk (R6 "partial"). The design has no answer to "what survives if the profile-entropy empirics go badly?" — and by its own admission nothing in the loss pushes them to go well.
>
> 3. **The ID-feature escape valve makes R4 and R6 mutually exclusive, and the document refuses to say so.** With `use_id_feature=True` (the default), the model retains all M·d = 874K free per-item parameters against 436·d ≈ 28K shared feature rows, and the metadata is weak (colour/pattern ~1.3× lift). SGD will route discriminative signal into the unconstrained ID rows — the path of least resistance — and the "feature-explained fraction" of every activation collapses toward an "item-specific: +x" residual that explains nothing. The candidate's own external evidence predicts this: LightFM warm gained +0.001 AUC from tags (0.762→0.763), i.e., features contributed essentially nothing where IDs were available. With `use_id_feature=False`, you protect the explanation but cripple capacity (point 4). The evaluation table rates R4 "partial" and R6 "partial" separately, but the real situation is a fork: the configuration that earns R6 forfeits R4 and vice versa. The headline story — tied, intrinsic, feature-grounded *and* accurate — is not achievable by any single setting of the model's one new flag.
>
> 4. **The no-ID variant has a hard equivalence-class ceiling at H&M scale, and the document proved it and called it a feature.** C2(b): identical feature sets ⇒ bit-identical activations ⇒ identical scores for every user. With F=6 fields over 13,651 items, the item space collapses into (department × product_type × section × colour × pattern × price-decile) equivalence classes; vast swaths of a fast-fashion catalog — every black solid T-shirt in the same department and price band — become one item the model is structurally incapable of ranking apart. 436 feature rows cannot individuate 13,651 items; this is not a sparsity-regularization benefit (LightFM §2.3-1), it is rank starvation. The fold-in story for cold items inherits the same defect: a cold item's "explanation" is exactly the explanation of every warm item sharing its tuple, which makes the cold-start claim true but vacuous — the system recommends a *category*, dressed up as an item.
>
> 5. **The "exact additive shares" are not what a reader of the explanation will think they are — and the candidate's own C6 proves it.** The rendered shares c_{f,k} = e_f·p_k/(‖q_i‖‖p_k‖) are entangled through ‖q_i‖: C6 confirms that perturbing one feature row rescales *every other* share. So the shares are not counterfactual contributions — removing "Dark Blue" changes the printed value of "Trousers Denim" — yet the rendered example ("because of: department +0.9, colour +0.4") invites precisely a causal reading. The §5 pitfalls pass declares attribution deception "structurally avoided" because the shares are exact algebra; this conflates *arithmetic faithfulness* with *semantic faithfulness*. Worse, the shares only decompose t*_k − 1, while the +1 shift — half the activation's range, and the part multiplied by user affinity in the score — is a feature-independent constant the explanation silently carries. The document's own showcase example gets this wrong: the listed shares (+0.9+0.4+0.3+0.1+0.0 = 1.7) sum to t*₁₂ itself, not to t*₁₂ − 1 = 0.7 as C1 requires. The flagship rendered explanation contradicts the flagship verified claim.
>
> 6. **"Intrinsic" is achieved by rebranding, not by mechanism.** The R2 row's own nuance gives the game away: the global prototype profile — the thing that makes a prototype mean "Trousers Men · Denim · Dark colours" — is computed by scanning cos(e_f, p_k) over the vocabulary, i.e., a nearest-neighbor read-out in latent space over free, CF-trained embeddings. That is structurally identical to ProtoMF's post-hoc top-k-items interpretation, just with 436 feature rows substituted for the item catalog. Nothing in training makes e_f mean its label: "Dark Blue"'s embedding is fit to whatever collaborative signal correlates with dark-blue items, and a profile read from it inherits every confound. The genuinely intrinsic part (per-prototype score summands) was already ProtoMF §5.2, unchanged. The candidate's delta over the host on the intrinsic/post-hoc axis is therefore: an inherited intrinsic device, plus a new post-hoc-shaped device it relabels "parameter read-out."
>
> 7. **R7 is conceded in advance, which makes the fashion-explanation story circular.** The candidate's own data: taxonomy carries the signal (department lift 5.2), perceptual features don't (colour/pattern 1.1–2.4), and the R7 row admits "lift 5.2 says it will" dominate. So the steady-state explanation is: *this pair of jeans was recommended because it belongs to department "Trousers Denim."* That is a near-tautology — the item's catalog category explaining the item — and it is internal merchandising vocabulary besides. "The decomposition makes the tension measurable" is a researcher's consolation, not a user-facing property; a thesis whose title promises interpretable *fashion* explanations is on track to deliver org-chart attributions with colour as a +0.05 garnish. Declaring the vocabulary-constrained fix a "different candidate" (R8) means dc01, as specified, is expected to fail R7 and plans to.
>
> 8. **The accuracy upside exists only in a regime the pipeline doesn't evaluate.** The single strong piece of external evidence is LightFM's *item-cold split* (0.500→0.716 AUC, on a 1,030-tag corpus); warm, features were a wash. dc01's H&M setting is warm by construction — 623K train interactions over 13,651 items (~46/item), sampled-softmax training, and the host's hit-ratio validation over seen items; no cold-item split is anywhere committed to in §3.6's baseline plan. So against its own headline comparison (CF-only `user_item_proto`, "same K, d, losses") the design's realistic best case is parity-with-noise, and its motivated win condition (R5) is unmeasured by the stated experiments. A candidate whose only predicted accuracy benefit lives in an evaluation it doesn't schedule is betting its R6 row on hope and its R5 row on a citation.
>
> 9. **"R1 strong" overstates a one-sided, reused-module tie.** The user branch is untouched pure-CF, so the "tied integration" is half an integration; the document waves this off as scoping (S2 deferred), but it means user prototypes — half the explanation surface, half the UI-score — remain exactly as feature-blind and post-hoc-interpreted as the host's, and every rendered explanation will mix grounded item-prototype lines with ungrounded user-prototype lines ("your affinity û₁₂ = 0.8" — grounded in nothing). And the claimed tie mechanism itself is diluted: the host tied *weights across distinct embedding roles* (factories.py:93–94); dc01 substitutes *sharing one module instance with itself*, which is ordinary code reuse described as "the same tie philosophy one level deeper." The R1 row's "strong" is earned by redefining the requirement until the implementation satisfies it.

### Response (concede / rebut / open risk, point by point)

1. **Partly concede, partly rebut.** Conceded: the algebra is elementary, and 4b
   confirmed only what algebra guarantees — that is what Step 4b *is for* (falsify
   the cheaply falsifiable; the protocol forbids pretending training-dynamics claims
   were tested). Rebutted: simplicity of the read-out is not a defect in an
   interpretability design — ProtoMF's own explanation device is "a dot product is a
   sum" (§5.2). The contribution claim does not rest on the algebra but on the
   combination and its empirical behaviour in the cold/explanation regime; "too
   incremental to publish" vs. "open wedge" is a genuine framing risk for the thesis
   narrative. **Open risk: contribution narrative must be argued, not assumed.**
2. **Concede the core, rebut the conclusion.** True and important: per-feature
   attribution alone does not need prototypes — the LightFM-MF baseline decomposes
   too. What prototypes add is the *intermediate concept layer* (R3): user-side
   affinities and item-side attributions meet in K nameable style groups instead of
   436 raw feature affinities per user; that is precisely the prototype-paradigm
   requirement, not an optional extra. But the critique is right that this added
   value is real only if profiles are sharp. **Adopted: profile sharpness (e.g.
   entropy of normalized |cos(e_f, p_k)| per prototype) becomes a primary evaluation
   metric, and explanation quality must also be compared against the LightFM-MF
   feature-attribution baseline, not only against post-hoc top-k.**
3. **Concede the tension, rebut the exclusivity.** The R4↔R6 coupling through
   `use_id_feature` is real and the LightFM warm numbers do predict ID dominance.
   Not conceded as a binary fork: the feature-explained fraction is a *measurable,
   continuous* quantity, and the ablation pair is designed to chart the
   fidelity–capacity frontier rather than pick a point blindly. Single-knob
   mitigations exist (ID-row norm penalty; DropoutNet-style ID dropout, reading-list
   B10) but adding one is a mechanism revision — **per protocol rule, that decision
   is the user's; dc01 carries the coupling as its central open tension** (noted as
   an amendment in §5).
4. **Concede the math, contextualize the impact.** The no-ID equivalence-class
   ceiling is real and quantifiable *now*: count distinct 6-field tuples among the
   13,651 V1 items (cheap pre-implementation check — **adopted as amendment to
   §3.6**). Two softeners, not excuses: any feature-only method (LightFM included)
   has exactly this ceiling — it is the information-theoretic price of cold-start,
   not a dc01 artifact; and the default configuration keeps ID rows, where the
   ceiling does not exist. "Recommends a category for a never-seen item" is the
   correct inductive bias, but the critique is right that the explanation should not
   *pretend* item-level individuation for cold items. **Open risk + scheduled check.**
5. **Concede on all three sub-points — the strongest critique.** (a) The example's
   arithmetic was wrong (shares must sum to t*−1): **corrected in §3.4 with a visible
   amendment marker.** (b) Shares are jointly normalized, not counterfactual;
   rendering must say so, and a counterfactual variant (re-scoring with a feature
   removed — still purely model-intrinsic quantities) is **added to §3.4 as an
   optional second read-out**. (c) The pitfalls line "structurally avoided"
   conflated arithmetic with semantic faithfulness — **softened with a visible
   amendment.** The +1 shift baseline (û_k·1 per prototype) is now rendered
   explicitly.
6. **Partly rebut, partly concede.** Rebutted: the profile is *not* structurally
   identical to top-k items. The feature rows are the same parameters that build
   every item's scored representation, so profile and per-item attribution are
   consistent by construction; top-k-items interpretation is a separate similarity
   search over data that can disagree with the score path. Conceded: nothing in
   training pins e_f to its human label beyond "aggregates the items carrying f" —
   label confounding (correlated features) is a real validity threat, the tabular
   sibling of the F2 disentanglement caveat. **Open risk: profile validity needs an
   empirical check (e.g. does the "Dark Blue" row actually sit near dark items'
   embeddings), and thesis claims must be worded at the level the mechanism
   supports.**
7. **Partly concede, partly rebut.** Rebutted: "because it is Trousers-Denim *and
   you have high affinity to the Trousers-Denim prototype*" is not tautological —
   the personalized half (which prototypes this user loads on) is the explanation's
   substance; category-level reasons are legitimate when category preference is the
   true driver. Conceded: if department dominates everything, the *vocabulary* is
   semi-internal and the fashion-explanation story weakens; that is exactly why R7
   was rated partial and why a vocabulary-constrained grounding is left as a
   separate candidate rather than denied. **Open risk, accepted as rated.**
8. **Concede — adopted.** §3.6 named no cold-item evaluation; that was a real gap,
   not an oversight of phrasing. **Amended (visibly) in §3.6:** a cold/sparse-item
   evaluation protocol (held-out item split à la LightFM §5, and/or tail-item
   stratification à la ProtoCF) is now committed alongside the dense benchmarks. R5
   without a measurement plan would indeed be a citation, not a result.
9. **Partly concede, partly rebut.** Conceded: the feature-grounding covers the item
   half of the explanation surface; user-prototype lines remain CF-interpreted
   (their grounding is post-hoc naming, as in the host) — explanations must visually
   distinguish grounded from ungrounded lines (**adopted into §3.4 rendering
   notes**). Rebutted: "instance sharing is just code reuse" — the host's
   factories.py:93–94 tie *is also* parameter sharing; mechanically the two are the
   same thing (one tensor, two consumers), so the R1 rating does not rest on a
   redefinition. The one-sidedness is scoping (S2), with the symmetric extension
   specified and cheap.

### Amendments made after 5b (visible edits, 2026-06-11)

1. §3.4 rendered example corrected (shares sum to t*−1; +1·û baseline line added;
   joint-normalization caveat; grounded vs. ungrounded line distinction; optional
   counterfactual read-out).
2. §3.6 extended with the cold-item evaluation commitment and the distinct-tuple
   count pre-check.
3. §5 pitfalls line on attribution deception softened (arithmetic ≠ semantic
   faithfulness).
4. §5 rating-table note added cross-referencing the R4↔R6 coupling.

## 6. Fingerprint

1. **Features-entry:** features → factors (LightFM/SVDFeature template) — the item
   embedding is a sum of feature-value embeddings, replacing the free ID table
   underneath the unchanged prototype layer.
2. **Grounding:** prototype meaning via geometry sharing — feature embeddings and
   prototypes live in the same R^d; a prototype's attribute profile = cos(e_f, p_k)
   over the feature vocabulary (enabled, not loss-enforced).
3. **Read-out:** per-prototype score contribution (host §5.2) × exact per-feature
   additive shares of each item-prototype activation (jointly ‖q_i‖-normalized),
   plus the global attribute profile for prototype naming.

Status: **concept re-hosted & amended (fI-ProtoMF; re-host amendment cycle, 2026-07-12)**
— F-DC01-01..07 applied under the UI host and transferred (six carry, F-DC01-07 parked);
fI re-derivation in the "⟳ Re-host amendment" section above; read-out fingerprint:
per-prototype contribution u_k·t*_k × grouped/determinacy-annotated per-feature shares
(full-score coverage via the rank-inert baseline); linear-half attribution parked to
the merge stage.
