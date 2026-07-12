# dc05 — fU-ProtoMF: History-Composed User Factors on the U-ProtoMF Host

> Produced by the design-candidate protocol (`Master/docs/design_candidate_protocol.md`,
> v1.3), cycle 5, 2026-07-12 — **a directed cycle under the fU design-cycle brief**
> (`Master/docs/design_candidates/fu_cycle_brief.md`, v1.0; user-authorized
> modifications M1–M7 apply; where brief and protocol conflict, the brief wins).
> Status: **built — awaiting SC(dc05)** (concept review closed by user
> 2026-07-12, same day as the cycle; dispositions in vault
> `2026-07-12_1840` and as dated amendments throughout this file.
> **Build session 2026-07-12** — SC.3-style, plan gated at
> `Master/temp/dc05_feature_user_proto_implementation_plan.md`:
> `HistoryFeatureEmbedding` + `build_user_history_weights` + factory branch +
> configs (`feature_user_proto`/`_noid`/`_debug`) + cold-user ID-drop mirror +
> renderer/cards/naming slots (fU AND host `user_proto`, like-for-like);
> dc_checks/dc05 suite t01–t10 + i06 all green, incl. the keystone
> bit-identity C5′ (t05) and the A1 term-by-term objective-equality
> obligation (t06); local hm_1_month smoke via `smoke_train.py`.)
> `ft_type`: **`feature_user_proto`** (wired at the build).
> *(File renamed from `dc05_draft.md` at Step 6 registration.)*

## 0. Frame

- **Directed cycle (M1):** the design space is deliberately narrowed to the lineage's
  user-side slot — host = **U-ProtoMF** (`user_proto`), stage bar = fU vs U-ProtoMF
  like-for-like. Step 0's question is re-scoped per the brief: *which user-side
  mechanism best serves the lineage*, not *which fingerprint region is unoccupied*.
- **Occupied space (all item-side):** dc01 features→factors × shared-geometry
  grounding × per-feature shares (hardened through SC.5 — the lineage's stage-1
  exemplar this cycle mirrors structurally); dc02 attribute-space prototypes ×
  by-construction; dc03 visual exemplars × push; dc04 attribute-anchored prototypes ×
  live centroids. dc02–04 are **licensed quarry (M4)**, their scrutiny deferred.
  Distinctness from dc01 is NOT required (lineage sibling by design); distinctness
  within this cycle's own shortlist still applies.
- **The open slot this cycle fills:** S2 (user-side features) — rated
  "strong (re-scoped)" in dc01's §G precisely because it was promised to a lineage
  stage, i.e. to THIS cycle; the S0.2 §5 disposition ("promotion would be a new
  design-candidate cycle") is hereby called in. The user side is also where the
  S0.2 cold-start 2×2's unaddressed cells (cold-user × warm-item) live — with the
  recorded honesty constraint that the current split contains zero cold users.
- **The null candidate to beat:** the mirror of fI — composed q_u = Σ_f e_f feeding
  U-ProtoMF's user-prototype layer, score t̂·u*. The cycle's real job is whether the
  mirror survives the user side's different physics: thin native user features
  (~3 meaningful fields; `customers.csv` reality to verify in Step 1, never from
  memory), history-derived features (leakage + intrinsicness), the mirrored gauge
  (now on the free item vector t̂), cold-user reality (M6).
- **Directly relevant open scratchpad entries:** the two `[captured scrutiny dc01]`
  user-side entries (2026-07-11: derived read-out vs **history-composed users** —
  the priority seed material, incl. the user's own route-2 refinement) and the two
  `[captured scrutiny s0]` entries (dual-side candidate routes a/b; 3-scenario cold
  testbed — an eval-design entry, scrutiny-stage per M6.2); the length-channel
  refund entry (2026-07-12) binds this cycle's Step 3b hidden-effects pre-audit
  (fU pays the same angular tax fI does — name its fU form + instruments).
- **Requirement recently served poorly:** R2/R4 on the *user* explanation surface —
  S0.2 Facet A left user prototypes "post-hoc CF clouds"; every dc01 rendered
  explanation carries CF-only affinity coefficients. This cycle's candidate is the
  lineage's chance to ground that surface — in **item vocabulary** (what members
  buy), per the captured-entry insight, not demographic vocabulary.

## 1. Seed selection

Corpus pass per **M2 (targeted, not all-themes)**: user-side/CBF/user-modeling
material re-read in the reading list — B3 SVDFeature (real-valued feature weights),
B5 LightFM (user = sum of feature embeddings, §2.2), A1 ProtoMF (§3.1 U-ProtoMF,
the host), B6/B7 (content→factor aggregation ancestors), B10 DropoutNet (content
reliance training trick), D1 ProtoCF (few-shot prototypes — item-side, contrast),
I4 (user attributes in prototype RS — as protected variables to *remove*, the
opposite direction), S7 §4.1 (INTERACT needs both sides featurized; H&M asymmetry
argues against it), S8 (cold-start taxonomy). §I citers re-check is Step 2b work
(protocol placement), recorded as due. Per M2, seeds were sourced **lineage-first**:
the user's thesis-outline vault entry (`2026-07-11_2351`, step 5) already names the
expected shape of this cycle's candidate; the scratchpad's captured user-side
entries are the priority input; dc02–04 are licensed quarry (M4).

**User-feature reality check (M6.1 — verified against `data/hm/raw/customers.csv`,
1,371,980 rows, 2026-07-12, this session):** the native fields are
`FN` (34.8% non-null, single value 1.0 — a presence flag), `Active` (33.8%, same),
`club_member_status` (99.6% non-null, 3 values, 92.7% "ACTIVE"),
`fashion_news_frequency` (98.8%, 4 values — effectively NONE 64% / Regularly 35%),
`age` (98.8% non-null, 84 distinct integer values), `postal_code` (100%, 352,899
distinct hashed values — identity-like; largest single code 120,303 customers,
presumably a no-address placeholder). Net: **one real-valued field (age), one
near-binary field (news frequency), one near-degenerate field (club status), two
sparse flags redundant with them, one pseudo-ID.** Unlike the item side (0%
missingness), user fields have missingness (age 1.2%, flags ~65%) — any design
using them needs a missing-value bucket. This confirms the captured-entry
assessment: demographic vocabulary supports ~dozens of user equivalence classes
for V1's 73,418 users, and its explanation sentences ("because you are 34") are
R7-wrong and privacy-adjacent.

### 1b. Scratchpad sweep (10 open entries: 0 user, 10 captured)

All seven 2026-06-11/2026-06-16 user entries were consumed by dc02/dc03/dc04
markers; their standing principles are honored here (item-first sequencing — this
cycle IS its anticipated user-side successor stage; two-axes gut-check applied to
the chosen seed below; intrinsic > post-hoc drives the whole cycle).

1. **`[captured dc02]` Loss-enforced grounding (profile + consistency loss)** —
   *not this cycle (item-side grounding-enforcement spectrum member; this cycle
   fills the user-side slot; stays open, still a strong future seed).*
2. **`[captured dc02]` Grounding-enforcement spectrum as thesis narrative** —
   *answered inline (framing device, not a candidate; note: the chosen seed puts
   fU at the same "unenforced/emergent" spectrum point as dc01 — the lineage's
   user-side stage inherits dc01's grounding strength, keeping the narrative
   axis clean).*
3. **`[captured dc03]` Part/patch-level visual prototypes** — *not this cycle
   (item-side visual region; prerequisite dc03 unreviewed).*
4. **`[captured dc03]` Multimodal frozen content space** — *not this cycle
   (item-side entry-point bridge; unrelated to the user-side slot).*
5. **`[captured dc04]` Differentiable (live) centroids = REGULARIZE + read-out** —
   *not this cycle (item-side dc04 variant; REGULARIZE route stays open).*
6. **`[captured scrutiny dc01]` Grounding the USER-side explanation gap — derived
   read-out vs history-composed users (2026-07-11, + both refinements)** —
   **seed this cycle.** Its route 2 (history-composed user representation) is the
   chosen seed's core; the user's own gate evaluation ("route 2 more convincing";
   the HOW-history-enters distinction — opaque compression vs legible ingredient;
   per-purchase attribution as the new explanation dimension; priced risks:
   heavy-buyer dominance, sum-vs-mean, taste stationarity, user-capacity ceiling)
   is adopted as design input wholesale. Route 1 (projecting user prototypes
   through W_u) is *not* available on the U-ProtoMF host (no projection matrix
   exists here) — it stays fUfI/dc01-SC.8 material, and its "pilot study" role
   (community profiles coherent in item vocabulary ⇒ route 2 makes it
   load-bearing) is noted as motivating evidence to collect there.
7. **`[captured scrutiny s0]` Dual-side attribute-aware candidate (routes a/b)** —
   *shapes this candidate:* route (a) "mirror" is what this cycle executes — in
   history-vocabulary form rather than demographic form (Seed 2 below records why
   the literal demographic mirror loses); route (b) "symmetric-at-the-tie" has no
   object on a single-branch host and stays open for the fUfI merge cycle.
8. **`[captured scrutiny s0]` Realistic 3-scenario cold-start testbed** — *not
   this cycle (eval-design, not a candidate; per M6.2 the cold-USER eval spec is
   scrutiny-stage work — flagged in Step 3/5, not built).*
9. **`[captured scrutiny dc01]` Length-channel refund (2026-07-12)** — *shapes
   this candidate (binding on Step 3b: fU is predicted to pay the mirrored
   angular tax — this cycle must name its fU form and instruments; the
   composed-vector normalization no-op keystone control transfers to fU's arm).*
10. **`[captured scrutiny dc01]` ID residual vector as diagnostic read-out
    (2026-07-11, + refinements)** — *shapes this candidate (the fU user-ID row
    e_ID(u) is the mirrored residual — "everything behavior says about the user
    that their purchase-history features don't"; the same three post-hoc
    diagnostic roles and the coordinates-vs-dictionary distinction carry; recorded
    in Step 3's read-out notes, not expanded — dc01/SC.8 owns the substance).*

### Shortlist

**Seed 1 — History-composed user factors: scratchpad route 2 × B5 LightFM / B3
SVDFeature template × A1 U-ProtoMF host.** Replace U-ProtoMF's free user embedding
with a composition over the user's **train-window purchase history's item-feature
values**: q_u = Σ_f n_{u,f}·e_f (+ user-ID row), where n_{u,f} counts how often
feature value f (canonical 5-field item vocabulary, V=426) occurs among the items
the user bought in the train window (normalization sum-vs-mean = Step 3 decision).
Feeds the unchanged user-prototype layer; score = u*·t̂ with the host's free item
vector t̂ ∈ R^{K_u}. SVDFeature's real-valued feature weights (B3 §2) are the
corpus license for count-weighted composition; LightFM §2.2 explicitly defines
users as feature sums.
- *Motivation gate:* R2/R3/R4 on the user surface (user prototypes become readable
  as taste-community profiles in ITEM vocabulary — "what its members buy" — the
  narratively correct grounding per the captured entry) + R5 in its honest user-side
  form (a thin-history user's representation degrades gracefully; benefit surface =
  warm-thin users, measurable NOW via D4 strata — unlike cold users) + R1 operative
  core (features are the representation driving the score; no reranker).
- *Distinctness (M1: vs this cycle's shortlist only; dc01-distinctness waived —
  lineage sibling):* vs Seed 2 — same mechanism shape, different feature reality
  and read-out claim ("because you bought these 12 dark items" vs "because you are
  34"); vs Seeds 3/4 — entry axis (composition feeding the prototype layer vs
  prototype-side re-parameterization vs bottleneck).
- *R8 parsimony:* PASS — exactly one idea, dc01's move repeated one level up
  (interactions→free vector ⇒ aggregate of feature words), same granularity.
  The history-count matrix is data plumbing, not a second mechanism.
- *M6.1 confrontations owed (Step 3):* leakage discipline (train-window-only
  counts), intrinsicness of a history-derived composition, recompute cadence.

**Seed 2 — Literal demographic mirror of fI: B5×A1 with native customer fields.**
q_u = Σ e_f over {age band, club status, news frequency} (+ID row).
- *Motivation gate:* FAILS as headline. The M6.1 reality check above gives it
  ~dozens of equivalence classes for 73,418 users → near-total ID-routing (the
  composition would be ID row + noise), R7-wrong and privacy-adjacent vocabulary,
  and the protocolled taste argument (vault `2026-07-11_2350`: taste must be
  *revealed by consumption*, not stated by demographics; demographics are a
  too-loose proxy). Its one honest use — cold users, where nothing else exists —
  is unmeasurable in the current split (S0.2 §3) and belongs to the lineage's
  later "candidate prime" (thesis-outline step 7), explicitly not this cycle.
- *Distinctness:* vs Seed 1 — feature source/read-out claim (above).
- *R8:* PASS (one idea) — rejected on the motivation gate, recorded as the
  mirror-that-does-not-survive (the brief's central question, answered: the
  mirror's *mechanism* survives; its *feature set* must be history-derived).

**Seed 3 — Cannibalized dc04, transplanted (M4): attribute-anchored USER
prototypes.** User branch stays free CF (e_u untouched); user prototypes are
re-parameterized as convex combinations of centroids of users grouped by
history-derived taste values (p^u_k = Σ_v a_{k,v}·c_v, c_v = centroid of users
whose histories carry item-feature value v).
- *Motivation gate:* grounds the user-prototype vocabulary (R4) with zero R6 risk
  (user capacity intact — dc04's signature niche), but leaves the user
  *representation* CF-opaque: no per-purchase attribution, no R5 story for
  thin-history users (a thin user's e_u is still an underfit free vector), and
  the explanation only names communities, not why THIS user belongs to one. On
  the user side the lineage's explanation gap is precisely the representation,
  not only the prototype naming — serves the weaker half of the need.
- *Distinctness:* entry axis vs Seeds 1/2 (features parameterize prototypes, not
  the representation); grounding axis (data-anchored centroids vs shared geometry).
- *R8:* PASS (one idea, "cannibalized from dc04 §1 Seed 1 / §3"). Not chosen;
  remains quarry for a later user-side grounding-strength contrast.

**Seed 4 — Cannibalized dc02, transplanted (M4): history-count bottleneck user
branch.** No per-user parameters at all: the user representation IS the (normalized)
426-dim history-count vector x_u; user prototypes live in that count space
(a_k ∈ R^V); activation = sim(x_u, a_k), dc02's by-construction grounding mirrored
with history-derived features.
- *Motivation gate:* maximal grounding strength and maximal thin-user story, but
  it deletes ALL per-user collaborative capacity (73,418 free vectors → 0) on the
  side that carries personalization — the R6 bet is far worse than dc02's own
  (items had rich signatures; users get one shared count space), and dc02's
  scrutiny (which would calibrate the bottleneck's real cost) is deferred — the
  quarry's load-bearing lesson is not yet in.
- *Distinctness:* entry + grounding axes vs all shortlist members.
- *R8:* PASS (one idea, "cannibalized from dc02 §3"). Not chosen this cycle;
  recorded as the natural grounding-enforcement escalation IF Seed 1's emergent
  grounding disappoints at fU scrutiny.

### Chosen seed

**Seed 1 — history-composed user factors (scratchpad route 2, realized as the
B5/B3 features→factors template with history-derived user features, under A1
U-ProtoMF).** Provenance: **`scratchpad+corpus`** — seeded by the user's captured
entry (`[captured scrutiny dc01]` 2026-07-11 + refinements) and the thesis-outline
vault entry (step 5 names exactly this candidate), grounded by B5 §2.2 / B3 §2 /
A1 §3.1. Adjacent non-corpus prior art (user = aggregate of consumed items'
representations: FISM-style item-based user models, YouTube-DNN-style watch-vector
averaging) is known to exist from model knowledge and is **flagged for the Step 2b
probe** — the specific union (history-composed user over *item-attribute
vocabulary*, under a prototype-similarity layer, with per-purchase/per-feature
decomposition of user-prototype activations) is what 2b must test for novelty.

Why it won: it is the only shortlisted seed that closes the lineage's actual gap —
the user *representation* becomes a legible ingredient (history → stated
aggregation formula → representation) instead of an opaque compression, giving the
new per-purchase attribution dimension the user's refinement named; it mirrors
dc01's mechanism shape exactly (lineage coherence: same move, one level up), so
the fUfI merge inherits two structurally identical composed branches; its keystone
reduction fU(F=0, +ID) ≡ `user_proto` is well-defined (M3); its benefit surface
(warm-thin users) is measurable under the existing D4 strata while the demographic
alternative's benefit surface (cold users) is unmeasurable in this split; and the
user already judged this route "more convincing" at the captured gate. The brief's
null-candidate question is answered at the feature-set level: **the fI mirror
survives as mechanism, but only with history-derived features — the literal
demographic mirror is rejected on the recorded motivation gate.**

## 2. Mechanism extraction

All statements verified against the local PDFs on 2026-07-12, this session:
`B5_kula_2015_lightfm_metadata_embeddings.pdf` (arXiv:1507.08439v1),
`B3_chen_2012_svdfeature_jmlr.pdf` (JMLR 13:3619–3622),
`A1_melchiorre_2022_protomf.pdf` (RecSys '22, pp. 246–256). Per M2, extraction is
scoped to the **user-side load-bearing claims**; dc01 §2 remains the record for the
item-side reading of B5/B3/A1.

### 2.1 LightFM (Kula 2015) — the user-side composition template

**Representation (§2.2).** "Users and items are fully described by their features":
user u has feature set f_u ⊂ F^U; the model's parameters are d-dimensional
*feature* embeddings, and **the latent representation of user u is the sum of its
features' latent vectors** (first display equation, §2.2): q_u = Σ_{j∈f_u} e^U_j.
Feature biases sum likewise (b_u = Σ b^U_j). Score (Eq. 1):
r̂_ui = f(q_u·p_i + b_u + b_i), sigmoid for binary data. Objective (Eq. 2):
Bernoulli likelihood over positive/negative sets, asynchronous SGD + Adagrad
(§2.2). §2.1's user-side plain-language form: "the representation for a female
user from the US is a sum of the representations of US and female users."

**The MF↔CB spectrum (§2.3).** Indicator-only feature sets reduce LightFM exactly
to MF; metadata features shared across entities (1) shrink the parameter count and
improve generalisation, (2) make cold-start representable ("Latent vectors for
indicator variables cannot be estimated for new, cold-start users or items.
Representing these as combinations of metadata features … makes it possible to
make cold-start predictions", §2.3 item 2), (3) do NOT reduce to a content model —
feature embeddings are fitted by factorising the *interaction* matrix (§2.3).

**Crucial for dc05 — what LightFM's own experiments did on the user side (§5).**
"In both LightFM (tags) and LightFM (tags + ids) users are described only by
indicator features" — i.e. the published LightFM results never composed users from
anything but IDs except in ONE arm: LightFM (tags + about) on CrossValidated,
where user features are a bag-of-words of profile text. That arm was the best
model on the sparse dataset (Table 1: warm 0.695 / cold 0.696 vs 0.682/0.674 for
tags+ids), "showing the added advantage of LightFM's ability to integrate user
metadata … likely due to improved prediction performance for users with little
data in the training set" (§6.1) — the published precedent that user-side feature
composition helps exactly the **thin-user** stratum. No history-derived user
features appear anywhere in the paper.

**Direct prior art for history-composed users, found in B5 §3 (load-bearing for
Step 2b):** Soboroff & Nicholas [B5's ref 21; 1999] "represent
users as linear combinations of the feature vectors of items they have interacted
with," then run LSI on the resulting user-feature matrix — implemented in B5 §5 as
the **LSI-UP** baseline ("each row … is given by the sum of content feature
vectors representing the items that user positively interacted with"). LightFM's
§3 critique: LSI-UP's representations are estimated from **content co-occurrence,
not user actions**, and it "models user preferences as being defined over
individual features themselves instead of over items." Table 1: LSI-UP is the
weakest hybrid everywhere (CV 0.636/0.637; ML 0.687/0.681), and §6.1 explains why
("it is crucial to use collaborative information when estimating content feature
embeddings"). **The dc05-relevant reading:** the *aggregation shape* (user = sum
of consumed items' features) is 1999-old prior art; what LightFM showed is that
the aggregation only performs when the feature embeddings are **trained through
the interaction objective** — which is exactly how dc05 embeds it (trainable e_f
under the host loss), unlike LSI-UP's frozen content LSI.

**Fold-in & production (§7.1).** New entities need no retraining: "their
representation can be immediately computed as the sum of the representations of
their features" — the cold/new-user story dc05 inherits (a new user's first
purchases immediately move their representation). Admitted limitations: quality
"predicated on the availability of high-quality metadata" (§6.1); no visual/audio
channel (§8).

### 2.2 SVDFeature (Chen et al. 2012) — real-valued weights & the history license

**Model (§2, the single display equation).** With user/item/global feature vectors
α ∈ R^n, β ∈ R^m, γ ∈ R^s:
ŷ = (Σ_j γ_j b^(g)_j + Σ_j α_j b^(u)_j + Σ_j β_j b^(i)_j) + (Σ_j α_j p_j)ᵀ(Σ_j β_j q_j).
"Intuitively, we use a linear model to construct user and item latent factors from
features" (§2). The α_j are **arbitrary real values** — the album example uses
weight 0.6 (§3/Fig. 1) — so **count- or frequency-weighted composition is inside
the published model class**, not an invention of dc05. Losses: square, negative
logistic log-likelihood, smoothed hinge; SGD (§2). Scale: Yahoo! Music 200M
ratings <2GB (§1–2); two KDD Cup wins (abstract).

**The history license (§2, motivating text).** SVDFeature's own enumeration of
usable information names history explicitly: "users' browsing history over movie
reviews may correlate with users' taste over movies" and "the rating history over
similar movies directly affect whether a user will favor the current one" — i.e.
history-derived user features are a *named intended use* of the α slot. §5 lists
"efficient speedup training for user feedback information" as a toolkit feature —
pointing at the SVD++-style user-history term (user factor augmented by consumed
items' implicit factors; Koren KDD 2008 — **model-knowledge pointer, not in the
corpus**, flagged for Step 2b as the canonical history-composed-user prior art at
*item-ID granularity*). It is a 4-page toolkit paper; no
explanation/interpretability claims are made.

### 2.3 ProtoMF (Melchiorre et al. 2022) — the U-ProtoMF host

**U-ProtoMF (§3.1).** L^u learned user prototypes p^u ∈ R^d (L^u ≪ N). The user's
new representation is the vector of shifted-cosine similarities to all user
prototypes (Eq. 1): u* = [sim(u, p^u_l)]_l ∈ R^{L^u}, sim(a,b) = 1 +
aᵀb/(‖a‖·‖b‖) ∈ [0,2] (all entries of u* positive by construction — stated
below Eq. 1). Score (Eq. 2): U-score(u,t) = Σ_l s^user_l with **s^user = u* ⊙ t**,
where the item embedding **t ∈ R^{L^u} is a free vector living in
user-prototype-similarity space** ("the item embeddings dimensions can be seen as
weights of the corresponding characteristic (defined by user prototypes)" — §3.1
interpretation paragraph, p. 249). Loss (Eq. 3): sampled softmax cross-entropy +
L2. Two inclusion-criteria regularizers on the USER side (Eqs. 4–5):
R_{P^u→U} = −(1/L^u) Σ_l max_i sim(u_i, p^u_l) (every prototype near some user)
and R_{U→P^u} = −(1/N) Σ_i max_l sim(u_i, p^u_l) (every user near some
prototype), both approximated in-batch (§3.1); total loss Eq. 6 with weights
λ1, λ2. *(Repo mapping: `sim_proto_weight`/`sim_batch_weight` in
`PrototypeEmbedding`; `ft_type='prototypes'`, User-Proto branch of the factory.)*

**What they showed (Table 2).** U-ProtoMF: ml-1m NDCG .333 / HR .583 (MF: .326 /
.571), AmazonVid .152†/.276†, lfm2b-1mon .179†/.322† — single-branch hosts are
weaker than UI (.383/.657 ml-1m) but the U host improves on MF's NDCG on all three
datasets. Parameter accounting (§5.1): MF (N+M)·d; UI adds ≈4K·d (single-branch
hosts add K·d).

**The user-side explanation gap (§5.2) — dc05's target.** User prototypes are
interpreted by **synthetic maximally-activating inputs**: "we create a synthetic
similarity vector u* of an imaginary user, where a maximum value is given to the
corresponding user prototype … then compute recommendations for this imaginary
user" — a doubly indirect, post-hoc procedure (the prototype is read through the
items a fictitious extreme user would be recommended). Item prototypes get
nearest-item profiling. This is exactly the R2/R4 gap on the user surface. §5.3
adds a caution dc05 must carry: user prototypes empirically **absorb demographic
structure** (36 male- vs 9 female-associated user prototypes among 93 on ml-1m,
Fisher tests, Figure 4) — prototype "meaning" on the user side includes latent
demographics whether or not demographic features are input.

**Future-work pointer (§6).** "We believe that including external features of
users and items … into ProtoMF might further benefit the interpretability of the
prototypes" — the authors name the user-side feature direction as open.

### 2.4 The composed mechanism (dc05, stated once)

Host U-ProtoMF verbatim; ONE change: the free user embedding u ∈ R^d is replaced
by a **history-composed embedding**

> q_u = Σ_{f ∈ V} w_{u,f} · e_f (+ e_ID(u), the user's own ID row),

where V is the canonical 5-field item-attribute vocabulary (V=426), w_{u,f} =
(normalized) count of attribute value f over the items of u's **train-window**
purchase history, and e_f are trainable feature-word embeddings fitted through the
host loss (LightFM training discipline, not LSI-UP's content factorization).
u* = [1 + cos(q_u, p^u_l)]_l feeds the unchanged Eq. 2 score against the host's
free item vector t ∈ R^{L^u}. Everything else — prototypes, inclusion
regularizers Eqs. 4–5, sampled-softmax loss, negative sampling — stays exactly
A1 §3.1. Precise formulation choices (weighting w, normalization, ID row,
missing-history degeneracy) are Step 3 design decisions.

## 2b. Prior-art probe

Probe target: the *adaptation* — "user representation composed from the
train-window purchase history's item-attribute-value counts, trained through a
ProtoMF user-prototype-similarity layer, with per-feature-value and per-purchase
decomposition of user-prototype activations as the intrinsic explanation."
Searches run 2026-07-12 (WebSearch ×4, Semantic Scholar API + MCP):

1. `prototype-based recommendation user representation sum of purchased items
   attribute embeddings explainable cold-start`
2. `"ProtoMF" extension user features OR side-information OR metadata prototypes
   recommendation`
3. `user latent factor weighted sum of consumed items attribute embeddings SVD++
   FISM content variant thin history recommendation`
4. `explainable recommendation user prototype grounded purchase history
   attribution "because you bought" intrinsic explanation`
   (+ Semantic Scholar MCP query `user profile aggregation item attribute
   embeddings interpretable prototype recommendation` — zero results.)

**§I forward-citation re-check (mandated by M2/protocol):** Semantic Scholar API
re-queried 2026-07-12 — **still 31 citers**, title list identical to the
2026-06-11 classified sweep (no new citers to classify). None grounds prototypes
in features on either side; none touches user-side feature composition. The
nearest, re-confirmed: I3 UIPC-MF (CF-only connection matrix), I4 modular
debiasing (user *attributes* appear only as protected variables to REMOVE from
prototype representations — the opposite direction), I2 (bias audit/fix of the
embedding space, side information as measurement, not representation).

**Findings — closest relatives, each distinguished:**
- **SVD++ (Koren, KDD 2008) / FISM (Kabbur et al., KDD 2013)** (surfaced by
  query 3; known from model knowledge, not in the corpus): user = (free vector
  +) normalized sum of consumed items' **item-ID-level** latent factors. Same
  aggregation shape, crucially different vocabulary: the summed ingredients are
  per-item free vectors — as opaque as the free user embedding they augment — so
  no attribute grounding, no thin-user statistical sharing beyond co-consumption,
  no prototype layer, no explanation mechanism. dc05's ingredients are the 426
  shared attribute words: legible, cold-item-transferable, and two orders of
  magnitude fewer parameters.
- **LSI-UP (Soboroff & Nicholas 1999, via B5 §3/§5)**: user = sum of consumed
  items' content-feature vectors — the aggregation shape itself, 1999-old — but
  the representation is factorized from **content co-occurrence** (LSI), not
  trained through the interaction objective; B5 Table 1 shows it weakest of the
  hybrids, and B5 §6.1 attributes exactly that to the missing collaborative
  training. dc05 trains e_f through the host loss (the LightFM discipline).
- **YouTube DNN (Covington et al., RecSys 2016 — model-knowledge pointer,
  flagged)**: user = average of watched-video embeddings (+ demographics) into a
  DNN — history-composed user representation in production, but ID-granularity
  ingredients, black-box scorer, no prototype vocabulary, no attribution.
- **Attribute-to-feature mappings (Gantner et al., ICDM 2010 — model-knowledge
  pointer, flagged)**: learned map from entity attributes to pretrained latent
  factors for cold-start — a post-hoc bridge into a CF space, not a
  representation trained through the score, no prototype layer, no explanation.
- **Meta-User2Vec (UMUAI 2020, surfaced by query 1)**: user metadata labels
  embedded Doc2Vec-style for cold users — demographic-vocabulary composition,
  not history-attribute composition, no prototype layer, no intrinsic read-out.
- **KG-path "because you bought" explainers (query 4 family)**: purchase-history
  explanations via knowledge-graph paths/attention — post-hoc or
  attention-based (E4 caveat), not representation-level, not prototype-shaped.

**Verdict: found nothing close.** The aggregation shape (user = sum over
history) is classical (SVD++/FISM/LSI-UP); the specific union — **attribute-word
ingredients (not item IDs) × interaction-trained embeddings (not content LSI) ×
prototype-similarity read-out × exact per-feature/per-purchase decomposition of
user-prototype activations** — appears unpublished. ProtoMF's own §6 names the
direction as open future work, and the citers re-check confirms nobody has taken
it. No refinement needed at this gate; the SVD++/FISM lineage must be cited and
distinguished in the thesis (granularity + purpose), not ignored.

*(Amended 2026-07-12, same session, after Step 5b point 5 challenged the probe's
depth on the 2008–2016 attribute-augmented-MF family. Two further queries run:
WebSearch `attribute-enhanced SVD++ OR NSVD user latent factor item attributes
implicit feedback hybrid matrix factorization 2010..2016`; OpenAlex `user
profile item attributes latent factor model cold-start hybrid collaborative
filtering feature embeddings sum`. Findings, distinguished: **UISVD++** (user
age + movie-type attributes projected into SVD++'s factor space) — demographic
user attributes, the rejected Seed-2 territory, no history-derived features, no
prototypes, no explanation mechanism; **"Attributes Coupling based Item
Enhanced MF"** (arXiv:1405.0770) — item-attribute coupling as an embedding
regularizer (the S7 REGULARIZE route), item-side, no user composition;
**Ask the GRU** (RecSys 2016) / **Movie Genome** (UMUAI 2019) — content→factor
mappings for cold items, encoder-based, item-side, no prototype vocabulary, no
intrinsic attribution; timeSVD++-with-CNN-features hybrids (2016) — same class.
The family is real and must be cited in Related Work; none of it composes USERS
from history-derived attribute counts, none scores through a prototype layer,
none decomposes activations per purchase. Verdict unchanged, on firmer
ground.)*

## 3. Adaptation design

Verified against the actual code on 2026-07-12 (`feature_extractors.py` incl. the
dc01/dc02 additions, `feature_extractor_factories.py`, `utilities/explanations/
breakdown.py` incl. the deferred fU slot at line 629); data facts from
`customers.csv` (this session), `hm_feature_analysis.md`, and the S0.5/S0.6
canonical decisions (charter C5: 5 fields, V=426).

### 3.1 The one idea

Replace **U-ProtoMF's free user embedding** with a **history-composed embedding**:

> q_u = Σ_{f∈V} w̄_{u,f}·e_f + e_ID(u),  w̄_{u,f} = n_{u,f} / |H_u|,

where H_u = the user's **train-window** purchase set (deduped, exactly as the
training matrix sees it), n_{u,f} = how many of those purchases carry attribute
value f, and e_f are trainable word embeddings over the canonical 5-field item
vocabulary (V=426). q_u feeds the *unchanged* user-prototype layer; score =
Σ_l t_l·u*_l against the host's free item vector t ∈ R^{K_u}. Everything else —
prototypes, shifted cosine, inclusion regularizers (A1 Eqs. 4–5), sampled-softmax
loss, negative sampling, item branch — stays exactly `user_proto`. **This is
dc01's move repeated one level up** (item: interactions→free vector ⇒ sum of
feature words; user: history→free vector ⇒ mean of purchased items' feature
words), which is what makes the fUfI merge inherit two structurally identical
composed branches.

**Normalization decision (I propose — answers the captured entry's sum-vs-mean
question): mean over purchases.** Since each purchase contributes one value per
field, the per-field masses each sum to exactly 1 and the total metadata mass is
exactly F=5 rows' worth for EVERY user — structurally identical to fI's 5-row
signature + 1 ID row. Three reasons: (a) **mirror geometry** — dc01's entire
hidden-effects apparatus (vote weights, twin tax, M2 dominance analysis)
transfers with roles swapped instead of needing re-derivation under
history-length-dependent mass; (b) **heavy-buyer dominance** (the captured
entry's priced risk) is neutralized in the metadata-vs-ID balance: under raw
counts a 200-item history's word mass would drown e_ID(u) — item-level memory
suppressed exactly for the data-richest users — while under the mean the
ID row's relative influence is history-length-independent; (c) bounded per-user
gradient mass. Under the shifted cosine ‖q_u‖ is discarded, so the choice acts
ONLY through the metadata-vs-ID balance and training dynamics (verified as claim
C8′ in 4b). Alternatives recorded, not stacked: raw counts, SVD++-style
|H_u|^{−1/2}. Note: the splitter dedups (customer, article) keep-first, so
n_{u,f} counts **distinct purchased articles** — consistent with the interaction
matrix the loss factorizes; disclosed, not a choice.

### 3.2 Modules and classes

**New class `HistoryFeatureEmbedding(FeatureExtractor)`** in
`feature_extraction/feature_extractors.py` (sibling of `FeatureEmbedding`, which
cannot serve: it assumes fixed-width one-value-per-field rows; user baskets are
variable-length and weighted):

```python
class HistoryFeatureEmbedding(FeatureExtractor):
    def __init__(self, n_objects: int,
                 hist_value_ids: torch.LongTensor,   # (n_objects, D_max), padded
                 hist_weights: torch.Tensor,         # (n_objects, D_max), 0.0 on padding
                 n_features: int, embedding_dim: int,
                 use_id_feature: bool = True, max_norm: float = None):
        ...
        self.register_buffer('hist_value_ids', hist_value_ids, persistent=False)
        self.register_buffer('hist_weights', hist_weights, persistent=False)
        n_rows = n_features + (n_objects if use_id_feature else 0)
        self.embedding_layer = nn.Embedding(n_rows, embedding_dim, max_norm=max_norm)

    def forward(self, o_idxs):                       # (B,) users
        ids = self.hist_value_ids[o_idxs]            # (..., D_max)
        w = self.hist_weights[o_idxs]                # (..., D_max)
        q = (self.embedding_layer(ids) * w.unsqueeze(-1)).sum(dim=-2)   # (..., d)
        if self.use_id_feature:
            q = q + self.embedding_layer(o_idxs + self.n_features)
        return q
```

D_max = max over users of **distinct** attribute values in the train basket
(≤ 426 hard; measured at build — a padding row uses word id 0 with weight 0.0,
contributing exactly nothing). If the padded layout's memory misbehaves on the
larger splits, the declared fallback is the flat
`F.embedding_bag(..., per_sample_weights, mode='sum')` layout (identical math,
CSR-style offsets) — both are indexed gathers, no full-catalog pass. Buffers are
non-persistent (dc01/dc02 convention: rebuilt deterministically at model build,
never checkpointed).

**Data plumbing:** a builder `build_user_history_weights(train_lhs_path,
item_features_path, field_set)` beside `build_feature_ids` in
`feature_extraction/feature_ids.py`: reads the split's **train** listening
history + `item_features.csv`, produces (hist_value_ids, hist_weights) with the
SAME global field-offset vocabulary encoding as `build_feature_ids` (labels stay
renderer-compatible), mean-normalized per user. Injected into
`user_ft_ext_param` by the `inject_feature_ids`-style helper in
trainer/tester `_build_model` — never serialized into the Ray/JSON config (dc01
pattern). **Leakage rule (M6.1a, binding):** the builder consumes exactly the
split directory's own train file — on the cold/variant datasets the weights are
rebuilt from the VARIANT train file, so removed rows can never leak in through
w̄; pinned by a dc_checks test at build time.

**Factory branch** in `feature_extractor_factories.py` — new **`ft_type =
'feature_user_proto'`** (M6.4 name proposal, mirroring the reserved-names
convention; to be added to CLAUDE.md's ft_type table at registration), cloned
from the `feature_item_proto` branch with roles swapped:

```python
elif ft_type == 'feature_user_proto':
    user_param = ft_ext_param['user_ft_ext_param']
    user_n_prototypes = user_param['n_prototypes']
    assert ft_ext_param['item_ft_ext_param']['ft_type'] == 'embedding'
    assert not user_param['use_weight_matrix']
    user_hist_embed = HistoryFeatureEmbedding(n_users, user_param['hist_value_ids'],
                                              user_param['hist_weights'],
                                              user_param['n_features'], embedding_dim,
                                              use_id_feature, user_max_norm)
    user_feature_extractor = PrototypeEmbedding(n_users, embedding_dim,
                                                n_prototypes=user_n_prototypes, ...,
                                                embedding_ext=user_hist_embed)
    item_feature_extractor = FeatureExtractorFactory.create_model(
        ft_ext_param['item_ft_ext_param'], n_items, user_n_prototypes)  # free t ∈ R^{K_u}
    user_hist_embed.init_parameters()   # factory single-owner init (F-DC01-10 pattern)
```

**Config:** `confs/hyper_params.py` gains TWO fixed configs —
`feature_user_proto` (with ID row) and `feature_user_proto_noid` — each with its
own full search (charter C7 / F-DC01-01 lesson: never a searched flag).
`rec_sys/rec_sys.py`, `protomf_dataset.py`, `trainer.py`: **zero changes** (the
extractor interface absorbs everything; the user branch receives `(B,)` indices
exactly as today).

### 3.3 Forward pass with shapes (batch B, n_neg negatives, d emb-dim, K_u prototypes, D_max padded basket width)

| step | tensor | shape |
|---|---|---|
| user basket lookup | `hist_value_ids[u_idxs]`, `hist_weights[u_idxs]` | (B, D_max) |
| user composed emb | `q_u = Σ_j w_j·e_j (+ e_ID(u))` | (B, d) |
| user proto branch | `u* = 1 + cos(q_u, P^u)` | (B, K_u) |
| item embedding | `t = T[i_idxs]` | (B, 1+n_neg, K_u) |
| score (RecSys dot) | `Σ_l u*_l·t_l` | (B, 1+n_neg) |

Parameters: word table (426 + N·[use_id])×d, prototypes K_u×d, item table
**M×K_u** (dimension K_u, not d — host property). No W matrices.

### 3.4 Intrinsic explanation read-out

All quantities exist in the forward pass. First the score identity that shapes
everything on this host:

> S(u,t) = Σ_l t_l·(1 + cos(q_u, p^u_l)) = **B(t)** + Σ_l t_l·cos_l,
> with B(t) = Σ_l t_l = 1ᵀt.

**The mirror is NOT symmetric here, and the asymmetry is disclosed up front:**
in fI the +1 baseline mass Σ_k u_k was *item-independent* — rank-inert for
ranking items. In fU the baseline mass **B(t) is a per-item scalar**: ranking is
over items, so B(t) does NOT cancel — it is a learned, user-independent,
non-personalized score channel (a de-facto item bias living inside the
"bias-free" fleet). Host-inherited (published U-ProtoMF carries it identically;
GR7-symmetric), but it re-scopes two dc01 conclusions for this stage: (a) the
100%-coverage claim becomes "**100% of the *personalized* score** (S − B(t))
decomposes exactly over (prototype, history-row) pairs"; B(t) is rendered as its
own honest, atomic, non-personalized line; (b) F-DC01-09's "popularity has
nowhere to live but the ID row" does NOT transfer — on this host popularity has
a comfortable identified home in B(t) (see 3b).

Read-outs (1–2 mirror dc01's C1/C4′ machinery; 3 is the new dimension; 4 mirrors
read-out 3):

1. **Per-prototype contribution:** s_l = t_l·u*_l, rendered as the personalized
   part t_l·cos_l per prototype + the B(t) line. t_l is the item's CF-learned
   coefficient on taste community l (the personalization weights of this host);
   u*_l is the grounded activation.
2. **Per-word shares of each user activation (exact):** u*_l − 1 =
   Σ_{f∈basket} w̄_{u,f}·e_f·p^u_l/(‖q_u‖‖p^u_l‖) + ID share — each attribute
   word's exact additive share of the user's community activation, jointly
   ‖q_u‖-normalized (4b C1′/C6′).
3. **Per-purchase regrouping (I propose — the new explanation dimension):**
   because w̄_{u,f} = Σ_{i∈H_u} 1[f∈f_i]/|H_u|, the SAME sum regroups exactly by
   purchased item: c_{i,l} = Σ_{f∈f_i} (1/|H_u|)·e_f·p^u_l/(‖q_u‖‖p^u_l‖) —
   each of the user's actual purchases carries an exact additive share of each
   community activation ("you activate *dark-denim menswear* mainly because of
   these 12 purchases, +0.31 of the 0.42 activation"). Word-level and
   purchase-level are two exact readings of ONE sum — rendered as alternative
   zooms, never added together (4b C2′).
4. **Global user-prototype profile in item vocabulary (R4/S4):** cos(e_f, p^u_l)
   over the 426 words, from parameters alone — user prototypes become nameable
   taste communities ("loves dark-denim menswear, avoids sporty basics"),
   replacing the host's synthetic-maximally-activating-user procedure (A1 §5.2)
   as the R2 comparison target. This realizes the captured entry's route-2 goal:
   user prototypes grounded in **item vocabulary** — what members buy, not what
   members are.

**Renderer slot (M5 — concrete contents for `breakdown.py`'s deferred slot):**
two new compute functions behind the existing `Breakdown` dataclass, arithmetic
honesty only (SC.5 gate scope: no epistemics on figures):
- `compute_breakdown_feature_user_proto`: header per contract; **left panel** =
  top-6 per-prototype personalized bars t_l·cos_l, signed, named by the
  intrinsic user-prototype naming route (read-out 4); **baseline line** = "item
  baseline B(t) = 1ᵀt = +x — non-personalized, this item's own scalar,
  identical for every user" (the wording change from fI's rank-inert line is
  mandatory — this baseline is NOT inert for item ranking); **zoom** = the top
  community's per-purchase shares (read-out 3; top-6 purchases by |share| +
  stated remainder), the user-ID row as its own `is_id` line; **footer** =
  feature-explained fraction of the personalized score (metadata rows vs the
  user-ID row across Σ_l t_l·c_{·,l}); self-check: B(t) + Σ personalized parts
  == forward score, zoom sums to its parent bar.
- `compute_breakdown_user_proto` (host slot, built at the same time —
  like-for-like): identical panel math on t_l·(u*_l−1) + the same B(t) line
  (the host owes the same disclosure — GR7-fair), zoom = the host's post-hoc
  user-prototype profiling (top-k nearest users' consumed-item lift), declared
  post-hoc in the naming_route field.
- Prototype cards (SC.5 visualization 1) extend per-side: fU cards = intrinsic
  word-profile bars; host cards = post-hoc descriptors; same card geometry.
- *(Review note 2026-07-12 — renderer walkthrough ratified by user with spec
  defaults: top-6 bars; per-purchase zoom as the figure default (word zoom in
  the md companion); purchase labels = truncated item descriptions (ids in
  companion); host zoom = its post-hoc top-k-users profile, declared post-hoc
  (steel-man, not the harsher "no decomposition" row). B(t) line wording as
  ratified at the B(t) gate.)*

**Rendered example (illustrative numbers, canonical 5):**

> **Why "Slim-fit dark-blue jeans" for you?** Score S = B(t) + Σ_l t_l·cos_l =
> 1.92 + 0.61. Item baseline B(t) = +1.92 (non-personalized — this item's own
> scalar, same for every user). Top personalized contributor: taste community
> #7 *(profile: Trousers Denim · Menswear · Dark colours)* — this item's
> coefficient t₇ = 0.9 × your community activation cos₇ = 0.42 → +0.38. Your
> 0.42 activation of community #7 comes from: your 12 dark-denim purchases
> (+0.31), your 9 menswear-basics purchases (+0.08), your 3 swimwear purchases
> (−0.02), 13 other purchases (+0.01), your personal profile row (ID) (+0.04).
> *(Alternative zoom, same sum by attribute word: department "Trousers Denim"
> +0.29, section "Menswear" +0.07, colour "Dark Blue" +0.04, …)*

*(Honesty note added 2026-07-12 after Step 5b point 8, visible amendment: the
printed magnitudes carry the 3b epistemics — the t_l coefficients are
L2-canonicalized representatives with a frozen gauge component (3b-i), all
shares are jointly coupled through ‖q_u‖ (4b C6 — removing a purchase rescales
the others; shares are exact summands, never counterfactual effects), and
within-lockstep-clique splits are init noise (3b-ii). Per the SC.5 gate scope
these stay OFF the figures (arithmetic honesty only, both models symmetrically)
and are carried by the thesis's hidden-effects section and the thesis text
around any rendered figure.)*

A **thin user** (3 purchases) renders the same explanation with 3 purchase
lines. A **cold user** (empty history): q_u = e_ID(u) — for a never-trained
user that is noise; with `use_id_feature=False` and empty history q_u = 0 and
the eps-guarded cosine yields u* ≡ 1, so S = B(t): **the cold-user fallback is
exactly the non-personalized item baseline — a popularity-shaped ranking, which
is the honest degenerate behavior** (4b C3′ verifies the zero-vector path). One
purchase in, fold-in applies (B5 §7.1): recompute w̄_u, no retraining — fU
converts cold users to warm-thin after their first basket.

**Cold-user honesty (M6.2).** fU does NOT rescue interaction-free users — that
cell of the S0.2 2×2 needs user attributes ("candidate prime", thesis-outline
step 7) or stays open. fU's R5 claim surface is **warm-thin users** (short
histories), measurable NOW via the D4 history-length quartiles on the canonical
runs — no new eval machinery. A genuine cold-USER testbed (the 3-scenario
scratchpad capture) is **scrutiny-stage work — flagged here, not built**. Two
spec-level mirrors recorded for the build: (i) cold-user scoring drops e_ID(u)
(mirror of the dc01 ID-drop convention / F-S0-07); (ii) the S0.3 cold-ITEM
runner is not fU's claim surface (fU's item side is the host's free t — cold
items get untrained vectors, exactly as the host; the fU fleet row still runs
it for comparability, expectation: host-like chance-level cold-item behavior).

### 3.5 Feature fields & the intrinsicness note

Vocabulary = the charter C5 canonical five (V=426), unchanged — the composition
vocabulary is the ITEM side's, applied to user histories; no transaction-derived
values (price_band stays deferred), `product_code` never a model feature. User
demographics are deliberately NOT in this candidate (Step 1 Seed 2 rejection).

**Intrinsicness of a history-derived composition (M6.1b, position stated):**
R2-intrinsic means the explanation is derivable from quantities the model
computes at inference. q_u is built by a **fixed, data-derived,
human-performable aggregation formula** stated in the explanation itself ("the
average of your 27 purchases' attribute words"); every rendered share is an
exact summand of the inference-time score. The history is model INPUT (as item
attributes are input to fI), not a post-hoc reconstruction. What changes vs
demographics is the *claim type*: "because you bought these 12 dark items" is
behavioral evidence (taste revealed — vault 2026-07-11_2350), not a demographic
inference. Boundary unchanged: e_f meanings remain CF-trained (label
confounding, F-DC01-02's discipline carries), and the composition is **frozen
per split** (train-window counts) — taste stationarity is a declared limitation
(recompute cadence is a production concern: fold-in per session; research scope
is static, M6.1c).

### 3.6 Baselines & the isolating ablation

- **Headline baseline: `user_proto`** (the host), already in the frozen S0.7
  reference fleet (charter C6) — the M1 stage bar, like-for-like (same K_u, d,
  losses, dev profile).
- **Keystone reduction (M3): `feature_user_proto` with zero metadata words and
  ID row on ≡ `user_proto`**, bit-identity as the implementation target (dc01
  C5/C5′ + i02′ weight-copy pattern; 4b C5′).
- **Isolating ablation pair (C7):** `feature_user_proto` (ids) vs
  `feature_user_proto_noid` (pure basket composition). The noid arm's ceiling:
  users with identical normalized basket-word vectors are indistinguishable —
  the user-side twin count is expected FAR below fI's 67.5% (baskets are
  multisets over 426 values; collisions concentrate in minimal 3-item
  histories); **pre-implementation check committed:** count distinct normalized
  basket vectors among V1's 73,418 users (the fU mirror of dc01's 6,517
  signature count).
- **Gap-interpretation rule, fU form (I propose, mirroring §3.6 of dc01 as
  amended):** the ids-vs-noid gap decomposes into (1) genuine user-level
  individuality loss (taste not expressible as basket-word composition), (2)
  minus a sharing/regularization offset (thin users' word mass is well-trained
  where their free vectors are not) — and, **unlike fI, popularity blindness is
  NOT a component**: popularity lives in B(t)/‖t‖ on this host and is untouched
  by the user-side arm choice. The angular-tax sub-component mirrors as (1b):
  user-ID signal present but angularly unrealizable against the shared basket
  mass (3b). Raw gap never quoted as "the cost of interpretability."
- **Explanation baseline to beat:** the host's §5.2 synthetic-user prototype
  interpretation + post-hoc top-k profiling (the R2 comparison), through the
  shared renderer + cards (M5).
- **Decoupled control (S1) — promoted from optional to a named
  charter-amendment request** *(amended 2026-07-12 after Step 5b point 6,
  visible edit)*: a no-prototype "history-composed user × free item, plain dot
  product" row (lightfm-branch shaped, user side; working name `lightfm_hist`)
  is what makes an fU win attributable — separating "history-feature
  composition helps" from "the prototype layer adds/costs what". It is not in
  the frozen fleet (C6), so this candidate's SC-stage run plan will **formally
  request it as a charter amendment** (one row, dev profile); the decision
  stays the user's. Until then, two free partial discriminators exist in the
  fleet: the D4-stratified **popularity reference row** (if fU's thin-stratum
  gain merely matches popularity's, the "popularity beats an undertrained free
  vector" suspicion stands) and the `lightfm_tags*` rows (feature-composition
  benefit on the item side, same vocabulary).

### 3.7 Deviations from the source papers (all "I propose")

1. **History-count user features under a cosine prototype layer** — in neither
   paper: B5's published users are indicator-only (tags/tags+ids) or profile
   bag-of-words (tags+about); B3 names history as feature material but ships no
   user-side history featurization; A1 has no features at all.
2. **Mean-per-purchase weighting** (§3.1) — a design decision inside B3's
   real-valued-weight license; SVD++'s |H|^{−1/2} is the adjacent published
   convention (different object: item-ID factors).
3. **Per-word AND per-purchase exact decomposition of user-prototype
   activations as intrinsic read-out** (§3.4) — the novel union (2b).
4. **No LightFM biases** (host score form; fleet C3 `use_bias=0`). Consequence
   stated honestly: on the U host the score still contains the de-facto
   item-bias channel B(t) = 1ᵀt — "bias-free" is structurally weaker here than
   on the I host (3b; renderer discloses B(t) per artifact).
5. **Item-vocabulary composition for users** (no demographics) — the Seed 2
   rejection carried into the mechanism.
6. **Cold-user ID-drop convention + variant-train weight rebuild** (§3.4/§3.2)
   — spec-only until the build; both pinned by dc_checks tests then.

### 3.8 Interpretability-constraint design (protocol v1.1 mandatory block)

**Inheritance.** The user-side inclusion criteria (A1 Eqs. 4–5; `sim_proto`/
`sim_batch` in `PrototypeEmbedding.forward`, acting on the batch–prototype
cosine matrix) survive **byte-identically** — they are agnostic to how
`o_embed` was produced (the dc01 transplant argument, verbatim). Semantic
shifts to keep in view: (1) "every prototype owned by some *user*" becomes
"owned by some *basket composition*"; duplicate normalized baskets are one
point for the coverage pressure (dilution expected far weaker than fI's — §3.6
check); (2) the host L2 now shrinks shared word rows serving many users while
per-user ID rows feel it relatively harder — λ_L2 sweet spot may move; (3) the
inherited `max_norm` knob caps word-row norms = **vote-weight equalizer**
(protects against loudness-dominance, NOT against omnipresence — F-DC01-05's
distinction carries verbatim to basket words).

**New degeneration modes** (beyond inherited coverage):
- **M1′ — diffuse community profiles:** nothing forces cos(e_f, p^u_l) profiles
  sharp; a diffuse profile reads as nothing (mirror of M1).
- **M2′ — frequent-word dominance:** "Solid" sits in 58% of items and therefore
  in nearly every basket; its mean-normalized weight is its share of each
  user's purchases, so the composed user cloud tilts toward common words —
  loudness capped by `max_norm`, **omnipresence residue carries as a definite
  risk to test** (F-DC01-05 user directive, inherited).
- **M3′ — user-ID routing:** SGD drains individual signal into e_ID(u),
  collapsing the feature-explained fraction (mirror of M3; measured by design
  via the ablation pair).
- **M4′ — community-profile overlap:** prototypes distinct in latent space with
  near-identical word profiles (mirror of M4; spread-mediation caveat carries —
  the composed user cloud may contract into a cone, weakening the indirect
  coverage-separation effect).
- **M5′ — heavy-basket mean-regression (new, fU-specific):** under mean
  normalization the variance of q̄_u's metadata part shrinks with |H_u| — heavy
  buyers' compositions concentrate toward the population mean-basket direction,
  so their activations flatten toward the average taste profile exactly where
  the most data exists (the captured entry's "user-capacity ceiling" with a
  named mechanism). The ID row is the built-in counterweight (constant relative
  mass). Instrument: angle-to-mean-direction and activation spread vs
  history-length strata (3b).
  *(Precision + two insights, 2026-07-12 review sitting — user-ratified, vault
  `2026-07-12_1830`: (a) the exact reading is **noise-for-crowding**:
  averaging converges each user to their OWN taste centroid (noise reduction —
  good); the risk is that personal centroids are CROWDED by the shared
  population mass — thin users are noisy but far apart, heavy users precise but
  close together; the C9 toy is the worst case (one shared distribution).
  (b) **Division of labor in the ids arm:** the ID row's relative weight is
  constant but its training QUALITY grows with |H_u| — words carry the taste
  class, the ID row carries idiosyncrasy, and the handover strengthens exactly
  as basket individuation fades; M5′ therefore primarily threatens the NOID
  arm and any words-carry-personalization claim, and is a central input to the
  ids-vs-noid interpretation. (c) **M5′ feeds M6′:** crowded user vectors
  flatten activations, making the user-independent B(t) channel the cheapest
  differentiator — the instruments interlock (cloud spread by stratum,
  activation spread, B(t) variance share; joint signature: ids-vs-noid gap
  widening specifically in the heavy band).)*
- **M6′ — baseline-channel migration (explanation-surface mode,
  host-inherited):** score mass can migrate into the non-personalized B(t)
  line, shrinking the personalized fraction the explanation narrates.
  Instrument: variance share of B(t) across items vs the personalized
  remainder, on the real checkpoint (both fU and host — GR7-fair).
  *(Review amendments 2026-07-12, user-ratified at the B(t) deep-dive — vault
  `2026-07-12_1808`: (a) instruments promoted to PRIMARY scrutiny metrics and
  extended with the **B(t) × item-train-popularity correlation** (if high, the
  thesis text may honestly annotate the baseline line "largely popularity");
  (b) adopted terminology: B(t) is an **implicit per-item intercept (an
  item-bias channel)** created by the +1 shift's constant input coordinate —
  never bare "bias"; the supportable authorship claim is "the paper does not
  discuss the induced decomposition", not "unintended"; (c) pedagogical proof
  note with the hand-checked worked example:
  `Master/temp/bt_baseline_demo/bt_baseline_demo.pdf`, thesis-figure seed.)*

**Constraint candidates — designed, NOT stacked (adopting any is the user's
mechanism decision; claims would route through a 4b re-run):**
- **K1′ → M1′, profile-sharpness penalty:** a_l = softmax_f(cos(e_f, p^u_l)/τ)
  over the 426 metadata rows; L_prof = (1/K_u) Σ_l H(a_l), weight λ_prof.
  Hook: parameters-only accumulator in the user `PrototypeEmbedding`, rides
  `get_and_reset_loss()`. Cost K_u·V ≈ 27K dots/step — negligible.
- **K2′ → M2′, word-norm control:** hard variant free (`max_norm`); soft
  variant L_norm = Σ_field Var_{f∈field}(‖e_f‖), hook in
  `HistoryFeatureEmbedding`. Cost trivial.
- **K3′ → M3′, ID-routing control:** L_id = (1/N) Σ_u ‖e_ID(u)‖², or
  DropoutNet-style (B10) ID-row dropout p_drop in the forward — which also
  trains the cold-user fold-in path the model faces at inference (double duty,
  mirror of F-DC01-04's mitigation).
- **K4′ → M4′, profile-distinctness penalty:** L_dist = mean_{l≠l'}
  cos(a_l, a_{l'}) over K1′'s profiles. Cost K_u²·V — negligible.
- **K5′ → M6′, baseline-channel control (new):** L_base = Var_i(1ᵀt_i), weight
  λ_base — flattens the non-personalized channel, pushing item differentiation
  into the prototype-aligned (narratable) directions. Cost trivial (in-batch
  variance). Risk stated: popularity is real signal; flattening its channel may
  cost accuracy — exactly the fidelity-vs-accuracy frontier knob for this host.
- **K6′ → M5′/M2′, population-mean centering** *(added 2026-07-12 at the
  review sitting — user decision: "record the option, it is very
  interesting")*: compose, then subtract the population mean-basket direction
  before the cosine — q̃_u = q_u − m̄, with m̄ = the (train-computed, buffered)
  mean of all users' metadata compositions. Converts the representation from
  "your average basket" into "how your basket DEVIATES from the average
  shopper's", attacking heavy-basket crowding (M5′) and the omnipresence tilt
  (M2′) with one move. Cost trivial (one buffered vector, one subtraction).
  This is a real mechanism change, not a loss knob: it would need its own toy
  round (the exact-decomposition claims C1′/C2′ must be re-derived — the
  shares acquire a −m̄ term), and it changes the explanation sentences'
  meaning ("you buy more denim *than average*"). NOT stacked (R8); the
  evidence trigger is the M5′ instrument set reading badly on the heavy band.
  (The normalization exponent (mean ↔ √ ↔ sum) remains the other, weaker
  M5′ lever.)

**Rashomon-tunable knobs:** inherited λ_sim_proto, λ_sim_batch, max_norm, λ_L2;
proposed λ_prof/τ, λ_norm, λ_id or p_drop, λ_dist, λ_base; structural: K_u
(multi-role on this host — see 3b), normalization exponent. Same F-DC01-05
honesty: sharpness/distinctness are not exhaustive metrics; timing = the
late-thesis Rashomon stage.

## 3b. Hidden-effects pre-audit (M3 — new step, dc01 record as template)

*Written before Step 4 so costs/claims/evaluation inherit it. Instruments named
here are scrutiny-stage (SC.8-analog) commitments; scrutiny-phase measurements
stay working evidence only (learnings-ledger rule).*

**(i) Gauge freedom — the item-coefficient gauge (F-DC01-08 mirrored).** Every
user's activation vector u*(u) = 1 + P̂^u q̂_u lies in a fixed affine slab of
dimension ≤ d+1 inside R^{K_u} (P̂^u = norm-scaled prototype matrix). Scores
probe each item's free vector t only through inner products with slab vectors:
for any w with (P̂^u)ᵀw = 0 and 1ᵀw = 0, **t → t+w changes no score of any
(u,t) pair**; the loss gradient is provably ⊥ w, so the gauge component of
every item vector is **conserved at its initialization value to machine
precision** under SGD (exact in exact arithmetic; ≤ ulp-level rounding drift —
4b C7c corrected wording), or L2-decayed toward the slab by exactly (1−2ηλ)
per step (4b C7d).
Exact gauge dim ≥ K_u − d − 1 when K_u > d+1; effective (weakly-pinned)
directions whenever the user-activation cloud under-spans — the noid arm and
M5′'s mean-regression both aggravate under-spanning. **Touched rendered
quantities:** the per-prototype item coefficients t_l in EVERY breakdown line
t_l·cos_l — for both fU and the host (host-inherited anatomy, GR7-symmetric),
same canonicalization story (L2 = accuracy-blind minimum-norm representative).
B(t) = 1ᵀt is NOT gauge (1 is excluded by construction) — the baseline line is
an identified quantity. Discipline: per SC.5's re-scope, no gauge annotation on
rendered artifacts; disclosure lives in the thesis hidden-effects section.
Instruments: user-activation-cloud spectrum/effective rank; trained-t
in-slab vs gauge-mass decomposition, both arms. Toy exhibit: 4b C7′.

**(ii) Lockstep/identifiability on basket words (F-DC01-06 mirrored, with a new
mass).** The word-clique gauge carries verbatim (only the SUM of lockstep-pair
rows is trained; the split is frozen init noise) — but the relevant coupling is
now **basket exposure**: two words a,b are near-lockstep for the user table
when w̄_{u,a}/w̄_{u,b} is near-constant across USERS, which compounds catalog
lockstep (dept↔section cliques) with **co-purchase structure** (words bought
together in near-fixed ratios). The characterize→exhibit→remedy→delimit
discipline applies unchanged; the decoupling-mass proxy
(`Master/temp/dc_checks/dc01/decoupling_mass.py`) is available prior art and
must be re-based onto the **user-basket weight matrix** (rows = w̄_u) instead of
the item catalog — scrutiny-basis-only (no thesis quotes; F-DC01-06
restriction). Grouped shares + within-clique-ordering-is-noise apply to both
zoom types; the per-purchase regrouping inherits the discipline at clique level
(a purchase's share is a sum of word shares).

**(iii) Norm/angle channels — the one-sided bottleneck and the fU tax
(F-DC01-09 + refund entry, mirrored and sharpened).** The USER side is
direction-only (the cosine discards ‖q_u‖; host-shared). The fU form of the
angular tax: every basket contains the population-common word mass (Solid, top
departments) — call it m; a user's individuality is the small deviation Δ_u, so
same-mass users are angularly squashed ~ ‖Δ‖/‖m‖, and the user-ID row must buy
individuality back through the bottleneck against m (worst under `max_norm`,
which caps the ID row while m sums many capped rows). **The tax is one-sided on
this host:** the ITEM side is linear — ‖t‖ and B(t) = 1ᵀt are live channels —
so fU pays the tax only on the user surface (exactly the refund entry's
prediction: "fU will pay its mirror"), and the item side's length channel is
the identified popularity home. The composed-vector normalization keystone
control transfers: normalizing q_u is an exact mathematical no-op through the
cosine (harness control, 4b C8′). Instruments: user twin/near-twin angle
distributions (same-basket vs matched non-twin pairs), read jointly with the
ids-vs-`user_proto` row; M5′'s angle-to-mean vs |H_u| strata.

**(iv) Function-class honesty (dc01 §I.3 mirrored, with the bias made
explicit).** Algebraically S(u,t) = 1ᵀt + (P̂^u ᵀ t)·q̂_u: generically for
K_u ≥ d, fU's ranking function class equals **"per-item scalar bias + dot-product
MF over unit-normalized composed user embeddings"** (for K_u < d a rank-K_u
bottleneck). What keeps the (t, P^u) factorization pinned: the Eqs. 4–5
inclusion regularizers (their job), L2's implicit bias in (t, P) coordinates,
and dynamics — host precedent: published U-ProtoMF ≠ MF empirically (A1
Table 2). fU claims no ranking-expressiveness gain; its value is the grounded,
decomposable explanation structure. The collapsed form also names the
popularity channel precisely (the 1ᵀt term) — thesis-text material for the
hidden-effects section.

**(v) Operating-point / cold mismatches & popularity channels (F-DC01-03/04
analogues, for USERS).** (a) *Second ID-keyed channel:* the per-user bias
scalar (`use_bias`) — fleet-inert (C3, use_bias=0); note the structural
asymmetry: a USER bias is rank-inert for ranking items (adds a constant per
user), so even bias-on it could not silently depress cold users the way the
item bias depressed cold items in dc01 — this channel is strictly milder here.
(b) *Train/cold operating-point mismatch:* an ids model always trains with
e_ID(u) present; cold/fold-in scoring composes without it — an operating point
never visited in training (F-DC01-04 mirror; precedent B5 tags+ids ≥ tags on
cold; unstacked mitigation K3′ dropout trains exactly this path). (c)
*Popularity:* lives in B(t)/‖t‖ (identified, item-side), NOT in user-ID rows —
so the fI braid hypothesis inverts here: warm reliance on e_ID(u) predicts
degradation for **thin users** (undertrained ID rows), not for cold items.
**Registered hypothesis (the candidate's central empirical bet):** fU's
advantage over `user_proto` concentrates in the thin D4 history-length strata,
because the host's thin-user free embeddings are undertrained while fU's word
mass is trained by every user sharing it — the LightFM tags+about finding (B5
§6.1) transplanted to histories; measurable on existing D4 machinery.

**(vi) K_u multi-role (I.5 mirrored).** On the U host, K_u is simultaneously
ITEM capacity (table M×K_u), effective score rank (min(K_u, d)), explanation
vocabulary size, and gauge dimension (K_u−d−1) — vocabulary size is not a free
interpretability knob; shrinking it shrinks every item's representation.
Recorded for the late-thesis Rashomon stage.

## 4. Cost & feasibility

All estimates **Fermi-style**. Real V1 `hm_1_month` numbers measured this
session (2026-07-12) from the local canonical split (working evidence, per the
learnings-ledger rule — protocol reasoning only, no thesis quotes): N = 73,418
users, M = 13,651 items, **train file 476,394 rows** (the oft-quoted 623,230 is
train+val+test; val = test = 73,418, one row per user); train history length:
mean 6.49, median 5, p90 12, p99 24, max 89 (25.1% of users sit at the
**3-train-row floor** — the 5-core minimum after leave-one-out removes one val
and one test row; V1 is 5-core, verified against the artifact 2026-07-12 after
a review question caught the earlier "3-item k-core floor" conflation — the
thin stratum is a quarter of the population); distinct
basket words per user: mean 18.7, median 16, p99 50, **D_max = 112**; V = 426.

**Committed §3.6 pre-check, executed:** distinct normalized basket signatures =
**73,167 of 73,418 users; 450 users (0.61%) share a signature; largest class
5.** The noid arm's equivalence-class ceiling is **negligible on the user side**
(vs fI's 67.5% item-signature sharing) — history individuates users two orders
of magnitude better than the 5-field signature individuates items.

**Parameter delta.** Host `user_proto` count: N·d + M·K_u + K_u·d
(d = K_u = 64: ≈ 4.70M + 0.87M + 4K ≈ 5.58M). fU:
- With ID row (default): **+V·d = +426·d ≈ +27K at d = 64** (~0.5% of total) —
  negligible.
- Without ID row: user table 73,418·d → 426·d — a **~172× shrink** of the user
  table (−4.67M at d=64, −84% of the model), the LightFM §2.3-1 generalisation
  argument in its strongest form.
- (Protocol reference point, CF-only `user_item_proto`: (N+M)·d + (K_u+K_t)·d +
  2d² tied — the stage bar per M1 is `user_proto`, quoted above.)
- Buffers: padded (N × D_max) int64 ids + float32 weights ≈ **99 MB on V1** —
  fine on any fleet GPU; on `hm_3_month` (N = 256,701) the padded worst case is
  ~3.5× that → flat `embedding_bag` fallback (Σ_u D_u ≈ N·18.7 entries ≈ 58 MB)
  is the declared mitigation; V1 needs none.

**Training-time expectation.** User branch per batch row: D_max = 112 padded
gathers + weighted sum (~112·d ≈ 7.2K mult-adds) vs the host's single gather;
the user branch carries no (1+n_neg) factor, and the per-step total is
dominated by the item-side gathers + the (B, 1+n_neg, K_u) score dot + loss.
Expected per-step multiplier **≈ 1.05–1.2× vs `user_proto`**. Convergence: word
rows aggregate gradients from all users sharing them (statistical efficiency
for thin users — the B5 §2.3-1 mirror); no strong argument for slower epochs;
flagged as unknown, not a risk.

**S5 pipeline-compatibility checklist:**
- Sampled `(batch, 1+n_neg)` scoring preserved? **Yes** — the item branch is the
  host's plain `Embedding`; the user branch stays indexed by `u_idxs` (B,).
- No full-catalog pass per step? **Yes** — inclusion regularizers stay in-batch
  (`PrototypeEmbedding.forward` untouched).
- Branches cheap indexed computation? **Yes** — padded gathers + one weighted
  sum; no encoder.
- Losses fit `get_and_reset_loss()`? **Yes** — `HistoryFeatureEmbedding` adds
  no loss; accumulators unchanged.
- Anything expensive per item/user? **No** — basket weights built once at
  startup from the split's train CSV + `item_features.csv` (seconds, CPU;
  builder committed with the build).

**Budget gate.** Dev profile per charter C4 (30 trials / 60 epochs / ASHA);
cost ≈ the S0.7 `user_proto` hyperopt × ~1.1–1.2, plus one single-config
variant retrain when a cold-user instrument eventually exists (not this
phase). No >1-day dev risk, no >10-day risk. **No flags raised** (the
hm_3_month padded-buffer note has a named fallback).

## 4b. Math sanity check

**Claims spec:** `Master/temp/dc_checks/dc05/claims_spec.md` (mechanism
definition + nine claims, no derivations). *(Label convention, clarified
2026-07-12 SC.1a trivial fix F-DC05-05: the spec and this section use C1–C9;
other sections of this file cite the same claims as C1′–C9′ — primes mark the
fU instantiations, distinguishing them from dc01's C-series.)* In brief — per
the M3 amplified floor: **C1** exact per-word decomposition of the shifted-cosine activation;
**C2** per-purchase regrouping equals the per-word grouping exactly; **C3**
degenerate inputs (empty basket + no ID → u* ≡ 1 → S = B(t) literal torch
behavior; identical baskets → bit-identical u*; distinct baskets
non-degenerate); **C4** S = B(t) + Σ_l t_l·(u*_l−1) exactly AND B(t) is NOT
rank-inert (the mirror-asymmetry claim); **C5 keystone** fU(V=0, +ID) ≡
`user_proto` bit-identity under weight copy; **C6** joint-normalization
coupling through ‖q_u‖ only; **C7 gauge** (F-DC01-08 analogue): score
invariance under t → t+w for w ⊥ {1}∪colspace(P̂), gradient ⊥ w, component
frozen under SGD, exact (1−2ηλ) L2 decay; **C8** normalization scale-invariance
(sum-vs-mean is a no-op without ID; bites only through the metadata-vs-ID
balance); **C9** mean-regression of heavy baskets (statistical trend, M5′).

**Viability self-assessment.** C1–C8 are algebra/gradient/limit-case properties
— all meaningfully toy-testable; C9 is a statistical trend, testable as a
simulated demonstration (not an identity). **Skipped as untestable at toy
scale, deliberately:** (i) the thin-user accuracy hypothesis (3b-v) — pure
training dynamics on real data; (ii) whether community profiles come out sharp
or diffuse (M1′/M4′); (iii) M2′ omnipresence-tilt magnitude; (iv) how much
score mass real training routes into B(t) (M6′). All four are carried into
Step 5 as the candidate's empirical bets.

**Subagent report (verbatim):**

> **Script:** `/home/mattik01/Desktop/githubs/ProtoMF/Master/temp/dc_checks/dc05/check_claims.py`
> **Run:** `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate protomf && python check_claims.py` (CPU, torch float64 default dtype, seeds {0,1,2,3}, ~5 s runtime). Only input read: `claims_spec.md`. The mechanism (word/ID table E, prototypes P, item table T, mean-normalized basket weights, shifted cosine via `torch.nn.functional.cosine_similarity`, S = t·u*) was re-implemented from the spec alone.
>
> **C1 (exact per-row decomposition) — CONFIRMED**, with one measured boundary. Over 432 cases (seeds×F∈{1,2,5}×K∈{2,7}×scaleE∈{1,1e6,1e-6}×scaleP∈{1,1e3,1e-3}×ID on/off, negative values via randn):
> - Identity vs exact cosine (true norms): max abs err = **4.44e-16**.
> - Identity vs torch CosineSimilarity wherever ‖q‖·‖P_l‖ ≥ eps=1e-8: max abs err = **5.55e-16**.
> - Caveat: in 47/432 cases (the adversarial combo scaleE=1e-6 × scaleP=1e-3, giving ‖q‖·‖P_l‖ < 1e-8) torch's eps-clamp makes u* deviate from the analytic formula by up to **7.44e-01**. The claim's formula uses true norms, so the identity itself holds; the deviation is torch's documented eps guard kicking in at degenerate scales. Within the claim's stated scope (random shapes/scales, F, K, negative values at non-degenerate norm products) it is exact to float64 precision.
>
> **C2 (per-purchase regrouping) — CONFIRMED.** Grouping the same summands by word vs by purchase (multiset baskets with repeats, F∈{1,2,5}, ID on/off): max abs difference = **5.55e-16**, and both equal u*_l − 1 minus the ID term to the same precision.
>
> **C3 (degenerate inputs) — CONFIRMED.**
> (a) Empty basket, no ID: q_u = 0; torch CosineSimilarity gives u* = all-ones **exactly** (max|u*−1| = 0.0), and S(u,i) = Σ_l t_{i,l} with max abs err **1.11e-16** — literal torch behavior verified.
> (b) Two users with identical normalized weight vectors (basket vs doubled basket), no ID: u* **bit-identical** (`torch.equal` True, all seeds).
> (c) Different random baskets: min over seeds of max|Δu*| = **4.05e-01** (generically different).
>
> **C4 (score decomposition + baseline NON-inertness) — CONFIRMED.**
> (a) max|S − (B(t) + Σ_l t_{i,l}(u*_l−1))| = **1.78e-15** over all seeds.
> (b) Constructed t₂ = t₁ + v with v ⊥ (u*−1), sum(v) ≠ 0: personalized parts −0.609170217711554 vs −0.609170217711554 (diff **1.11e-16**), B(t): −1.2759 vs 3.9813, scores −1.8851 vs 3.3722 — **score order decided by B(t)**. Empirical variance of B(t) over 5000 randn-initialized items (K=7): **7.047** (≈ K, clearly nonzero).
>
> **C5 (keystone reduction) — CONFIRMED.** With V=0 word rows and ID enabled, copying host user rows U_host[u] into E[V+u] gives u* and S **bit-identical** (`torch.equal` True for all users/items, all 4 seeds).
>
> **C6 (joint-normalization coupling) — CONFIRMED.** Scaling one basket word row by α ∈ {10, 0.1, −3}: for all other rows, max|share_new − (‖q_old‖/‖q_new‖)·share_old| = **5.55e-17**; max change in any pairwise ratio of other rows' shares = **4.55e-13** (ratio arithmetic noise only). Coupling acts only through ‖q_u‖, including for negative α.
>
> **C7 (item-vector gauge, d=3, K=6 > d+1; w from SVD null space of [1ᵀ; P̂ᵀ]):**
> (a) **CONFIRMED** — t_i → t_i + w changes no score: max|ΔS| = **8.88e-16** over all users/items/seeds.
> (b) **CONFIRMED** — autograd gradient of a sampled-softmax (cross-entropy over scores) loss w.r.t. every t_i: max normalized component |grad·w|/(‖grad‖‖w‖) = **4.10e-16** (machine precision zero).
> (c) **FALSIFIED in the literal "bitwise" reading** — under plain SGD (η=0.05, 20 steps) the component of t_i along w is NOT bitwise frozen: first bit flip at step 1 (seeds 0,1,3) or step 7 (seed 2); e.g. comp 0.06280994287275797 → 0.06280994287275819. However, the max abs drift over 20 steps and all seeds is **4.44e-16** — 1–2 ulps, pure floating-point rounding of the ~1e-16 residual gradient component. The mathematical content (component conserved) holds to machine precision; the word "bitwise" is the only part that fails. Reported literally per instructions.
> (d) **CONFIRMED** — with L2 λ=0.01, η=0.05: per-step component ratio matches (1−2ηλ) with max deviation **3.89e-15** over 20 steps × 4 seeds.
>
> **C8 (normalization scale invariance) — CONFIRMED.**
> (a) No ID: w_u → α·w_u for α ∈ {2, 0.5, 3.7, 1e6, 1e-6, |H|} plus explicit sum-vs-mean weights: max|Δu*| = **4.44e-16** — sum-vs-mean is a no-op without ID.
> (b) ID enabled, |H|=20: sum vs mean weights give max|Δu*| = **0.6087** (e.g. u*₃: 0.687 mean vs 1.295 sum) — genuinely different; the choice acts through metadata-vs-ID balance.
>
> **C9 (mean-regression of heavy baskets, trend) — CONFIRMED.** V=40, d=8, 2000 users/size, multinomial baskets from random p, mean normalization, no ID. Mean angle(q_u, Σ_f p_f E[f]) in degrees for |H| = [3, 10, 30, 100, 300]:
> - seed 0: 72.35 → 59.62 → 42.94 → 26.94 → 16.05
> - seed 1: 78.33 → 68.86 → 55.92 → 38.35 → 23.83
> - seed 2: 59.91 → 42.49 → 27.19 → 15.33 → 8.95
>
> Strictly monotone decreasing in all seeds.
>
> **Summary: C1–C6, C7(a,b,d), C8, C9 confirmed; C7(c) falsified only in its literal "bitwise" wording** (drift ≤ 4.44e-16 ≈ 1–2 ulps; conserved to machine precision — the underlying gauge-conservation claim is sound, "bitwise" overstates it). Nothing was untestable at toy scale. One noted ambiguity: C1's "exact" meets torch's eps-clamp when ‖q‖·‖P_l‖ < 1e-8 (only at the extreme adversarial scale combo); against the claim's own formula with true norms it is exact everywhere tested.

**Outcome.** One literal falsification, carried per protocol into Step 5 as a
known (wording-level) defect: **C7(c)'s "frozen at initialization" must be
stated as "conserved to machine precision under SGD" — not "bitwise"** (this
also flags that dc01 §I.1's "bitwise-frozen" phrasing was environment-lucky;
noted for M7 trickle-back). The mean-regression effect M5′ (C9) is now a
demonstrated mechanic, strengthening its status from conjecture to
toy-exhibited. The four skipped training-dynamics claims remain the empirical
bets. C3(a)'s confirmed literal behavior (cold noid user → pure item-baseline
ranking) is now a verified property, not a reading of torch docs.

## 5. Requirements evaluation

Requirements doc re-read in full before this step (2026-07-12 version, incl.
the R1 clarification note). Ratings are given **in their fU form** per M6.3
(operative core on the single-branch host; the weight-sharing exemplar is
assessed at the merge). The interpretable-AI framework (§6 open question) is
still undecided → protocol 5.4b skipped, as instructed.

| Req | Rating | Assessment |
|---|---|---|
| R1 tied integration | **strong (operative core; exemplar deferred)** | The basket composition IS the user representation; the score has no path around the grounded activation vector u* (S = t·u*, and both its baseline and personalized parts flow through u*). No reranker, no bolt-on, features inside the training objective. The exemplar mechanism (cross-branch weight sharing) has no object on a single-branch host — assessed at the merge, per the R1 clarification note. Deferral stated as one. |
| R2 intrinsic explanations | **strong** | Per-word and per-purchase shares are literal summands of the inference-time score (4b C1′/C2′); the community profile cos(e_f, p^u_l) is a parameter read-out (same purist nuance as dc01's R2 row). This stage closes exactly the surface S0.2 left post-hoc: user prototypes stop being "CF clouds interpreted through synthetic users" (A1 §5.2). Scope honesty: the item coefficients t_l stay CF-learned (the mirror of fI's u_k, gauge-disciplined per 3b-i), and B(t) is an atomic non-personalized line — intrinsic ≠ fully feature-transparent. |
| R3 prototype paradigm | **strong** | The explanation remains "which taste communities, and what they mean"; only the meaning channel changes — from item clouds via synthetic probes to attribute-word profiles + the member's own purchases. Host prototype layer untouched. |
| R4 feature-grounded prototypes | **partial** | The attribution half is maximal (exact additive shares, two exact regroupings, not saliency). The profile half is enabled, not enforced — nothing forces sharp word profiles (M1′); with the ID row on, part of every activation flows through the opaque e_ID(u) (honest "your personal profile row" line, but it caps the feature-explained fraction — M3′). Same coupled open risk as dc01, measured by the same instruments. |
| R5 sparsity robustness | **partial (user-side re-scope)** | Weak-signal users: yes, and it is the candidate's central bet — a thin user's representation is built from well-trained shared words instead of an undertrained free vector; measurable NOW via D4 history-length strata (25.1% of V1 users sit at the 3-train-row floor, i.e. the 5-core minimum). Absent-signal users: honestly NO — an empty basket degrades to the non-personalized item baseline (verified literal behavior, 4b C3a) and personalizes only after the first purchase (fold-in, B5 §7.1). Cold ITEMS: untouched (host behavior — the mirror of fI not touching cold users). The genuine cold-user testbed is scrutiny-stage work (M6.2, flagged not built). |
| R6 dense-regime accuracy | **partial (stage bar: vs `user_proto`)** | For it: the B5 tags+about precedent (user-side features helped exactly the sparse regime), the negligible noid twin ceiling (0.61% exact-collision rate vs fI's 67.5% — *qualified 2026-07-12 after Step 5b point 9, visible edit: this is the **representational** ceiling (identical count vectors); how much angular individuation survives the composition, the common-mass squash (3b-iii), and mean-regression (M5′) is a **geometric** question the 3b instruments (near-twin angles, composed-cloud effective rank) exist to measure — the combinatorial number is necessary, not sufficient*), and the thin-user hypothesis (3b-v). Against it: M5′ mean-regression (heavy users' compositions flatten toward the average basket — a capacity ceiling precisely where the most data is; toy-exhibited, 4b C9), and the general fidelity-capacity trade. The paper-headline bar (UI) is deliberately deferred to the merge (staged bar, not lowered — dc01 §G P7 language applies verbatim). |
| R7 perceivable vocabulary | **partial (strengthened by the purchase reading)** | Same vocabulary tension as fI (taxonomy likely earns the shares; department names semi-legible). Genuinely better here: the per-purchase zoom states reasons in the most user-perceivable vocabulary that exists — **the user's own purchases** ("because you bought these 12 dark items"), which requires no taxonomy literacy at all. The word-level zoom keeps the fI caveats unchanged. |
| R8 parsimony | **strong** | One idea — replace the free user vector with the basket-mean composition; the host is recovered exactly at V=0+ID (4b C5′, bit-identity); two fixed configs, no auxiliary losses, no stacked knobs. Same conceptual size as `user_proto` itself. |
| S1 decoupled variant | **partial** | The spectrum's post-hoc end exists (host + its §5.2 machinery, in the fleet); but the natural mid-point control — history-composed user × plain dot product, no prototype layer — is NOT in the frozen fleet (C6) and would need a charter-level decision at scrutiny time (named in §3.6). dc01's S1-strong leaned on the already-built lightfm rows; fU has no such free lunch. |
| S2 user-side features | **strong (this candidate IS the S2 promotion)** | Delivered, in item-vocabulary form (the recorded arguments against demographic vocabulary stand — Seed 2). The demographic variant stays open as the lineage's "candidate prime" (thesis-outline step 7); the architecture does not preclude adding attribute rows to the same sum (LightFM-degradation math, outline step 7's note). |
| S3 image features | **partial** | Architecturally open exactly as in dc01: a precomputed image vector per purchased item could enter the basket sum as one more row (thesis-outline step 8 anticipates this at the lineage end); it would break the pure categorical-lookup profile (precompute cache) and attribute as one opaque lump. Not natural here; not precluded. |
| S4 LLM naming | **strong** | The community word profile (read-out 4) is exactly the structured input a naming LLM wants — strictly better-grounded than the host's synthetic-user route; the LLM names, never scores. |
| S5 pipeline compatibility | **strong** | Full checklist pass (§4): indexed gathers only, no encoder, no full-catalog pass, losses unchanged, ≈1.05–1.2× step cost, buffers ≤ ~100 MB on V1 with a named fallback for larger splits. |

*(Coupling note, mirroring dc01: the R4 and R6 "partial" ratings are coupled
through `use_id_feature` — the ablation pair charts the same fidelity–capacity
frontier one level up. New on this host: part of the R6 risk (M5′) is NOT
ID-coupled but normalization-coupled — its knob is the aggregation exponent,
a mechanism decision, not a regularizer weight.)*

### §4 design tensions

- **Where do features enter:** resolved as "history-composition feeding the
  user `PrototypeEmbedding` branch" + new factory `ft_type` — the same
  representation-level entry point as fI, one level up. User prototypes stay in
  latent space (the attribute-space/anchored alternatives were shortlist Seeds
  3–4, cannibalizable later — the grounding-strength escalation path is mapped,
  not taken, per R8).
- **Fidelity vs. accuracy:** the tie is maximal on the user surface;
  `use_id_feature` is the same continuous escape valve as dc01 (degrades the
  feature-explained fraction visibly, never corrupts the attribution). NEW:
  the non-personalized channel B(t) sits outside the grounded narrative by
  construction — fidelity claims are claims about the *personalized* score,
  and the renderer's B(t) line is what keeps that honest (M6′ instrument).
- **Attribution granularity:** per-attribute-value AND per-purchase — the
  finest granularities available, as two exact readings of one sum; cross-field
  lockstep discipline (F-DC01-06) carries at both granularities.
- **R7 alignment vs. predictive strength:** both vocabularies enter through the
  items bought; the decomposition shows which earned the score; the
  per-purchase reading partially sidesteps the taxonomy-literacy problem
  (assessed in the R7 row).
- **Identifiability:** host regularizers kept unchanged; no feature-space
  analogue stacked (K1′–K5′ designed, unstacked). Risks tracked: M4′ profile
  overlap with the spread-mediation weakening (composed user cloud may
  contract), the basket-level lockstep gauge (3b-ii), and the item-coefficient
  gauge (3b-i) — instruments named for the scrutiny stage.

### Part-prototype pitfalls pass (reading-list S5 taxonomy)

- *Prototype collapse:* mitigated by the inherited inclusion pair; duplicate
  composed points are negligible on this side (0.61% basket twins) — weaker
  dilution than fI's.
- *Prototype quality/quantity:* K_u remains a hyperopt dimension; K_u
  multi-role caveat recorded (3b-vi) — vocabulary size is not a free knob.
- *Semantic gap:* narrowed (community meaning stated in words + member
  purchases) but not closed — diffuse profiles possible (M1′, R4 risk).
- *Prototype redundancy:* M4′ open, spread-mediation caveat carried.
- *Attribution deception:* the arithmetic layer is exact algebra (C1′/C2′),
  not saliency; semantic faithfulness is NOT thereby guaranteed — e_f meanings
  are CF-trained (label confounding), shares are jointly normalized (C6′), and
  the B(t) line must never be narrated as personalized (the renderer wording is
  the guard).
- *Vision-specific pitfalls:* n-a; tabular analogue (vocabulary the user cannot
  perceive) tracked under R7.

### Degeneration-protection table (gate 4c)

Modes and constraint candidates defined in §3.8. No identified mode unlisted.

| degeneration mode | status |
|---|---|
| Prototype–object collapse / coverage (host modes) | **protected by inherited regularizer** (Eqs. 4–5 pair, byte-identical) — duplicate-basket dilution negligible (0.61%, measured §4) |
| M1′ — diffuse community profiles | **unprotected — open risk**; unstacked K1′ (profile-sharpness penalty); primary metric: profile entropy |
| M2′ — frequent-word dominance | **partially protected by inherited knob** (`max_norm` = vote-weight equalizer; loudness only); **omnipresence residue = definite risk to test** (F-DC01-05 directive, inherited); soft K2′ unstacked |
| M3′ — user-ID signal routing | **unprotected — open risk, measured by design** (ids/noid ablation pair charts the frontier); unstacked K3′ (ID-norm penalty or B10-style ID dropout — also trains the fold-in path) |
| M4′ — community-profile overlap | **unprotected — open risk**; unstacked K4′; spread-mediation weakening named and instrumented |
| M5′ — heavy-basket mean-regression | **unprotected — open risk, toy-exhibited (4b C9)**; proposed unstacked K6′ (population-mean centering, mechanism-class knob — added at review 2026-07-12) + the normalization exponent as the weaker lever; instruments: angle-to-mean & activation spread vs history-length strata, read jointly with B(t) share (M5′→M6′ link) |
| M6′ — baseline-channel migration (host-inherited) | **unprotected — open risk, disclosed by design** (renderer B(t) line); unstacked K5′ (baseline-variance penalty, with the stated accuracy risk); instrument: B(t) variance share, both arms |

All statuses are flags, not vetoes (spectrum rule); adopting any K′ knob is a
user mechanism decision routing through a 4b re-run.

### Known defects from Step 4b

One wording-level falsification: gauge components of item vectors are
**conserved to machine precision** under SGD, not "bitwise frozen" (C7c; drift
≤ 2 ulps over 20 steps). Design-doc language above uses the corrected form;
M7 trickle-back owed to dc01 §I.1's "bitwise-frozen" phrasing. Four claims
skipped as training-dynamics — the candidate's empirical bets: (i) thin-user
advantage (the central bet), (ii) profile sharpness, (iii) omnipresence-tilt
magnitude, (iv) B(t) popularity absorption.

## 5b. Devil's advocate

Black-box critique (subagent saw only this file). **Verbatim:**

> 1. **The B(t) channel guts the explanation story, and the file knows it.** The candidate's entire value proposition is "grounded, decomposable user explanations" — yet on this host the score is S = B(t) + personalized part, where B(t) = 1ᵀt is a learned, per-item, user-independent channel that is *not* rank-inert (the file's own C4 toy: score order decided by B(t), variance ≈ K over random items). How much score mass real training routes into B(t) is admitted as untested (M6′, empirical bet iv), the mitigation K5′ is unstacked and conceded to risk accuracy, and the "100% coverage" claim is quietly re-scoped to "100% of the personalized score" — a fraction that could be 10% of the ranking signal for all anyone knows. If B(t) absorbs popularity the way the file's own function-class analysis (3b-iv: "per-item scalar bias + MF") predicts, the flagship rendered example — B(t)=+1.92 vs personalized +0.61, i.e. the *illustrative* numbers already show 76% of the score being non-personalized — is the honest picture: an explanation that lovingly decomposes the minority of the score while the majority sits in an atomic "same for every user" line. That is not closing the R2/R4 gap; it is relocating it.
>
> 2. **The candidate's own physics predicts its central bet fails where it is placed.** The registered hypothesis is thin-user advantage via D4 strata. But 3b-iii states that every basket carries the population-common word mass m ("Solid" in 58% of items), individuality is a small deviation Δ_u, and users are angularly squashed ~‖Δ‖/‖m‖. This squash is *worst* for thin users: a 3-item basket (25.1% of V1 users) contributes ~15 word tokens dominated by exactly the common words, so the thin stratum — the claimed benefit surface — is the stratum whose composed vectors collapse hardest into the common cone, yielding near-uniform u* and therefore B(t)-dominated, quasi-popularity rankings. Meanwhile heavy users are eaten from the other end by M5′ mean-regression (toy-confirmed: angle to mean direction falls monotonically with |H|). Squashed thin users, regressed heavy users — the file never says where individuation actually survives. If thin users do beat `user_proto` on hit-rate, the mechanism may simply be "popularity beats an undertrained free vector," which the missing decoupled control (see point 6) cannot rule out — and the explanation would then narrate per-purchase shares of a signal that barely moved the score.
>
> 3. **The normalization choice manufactures its own headline degeneration mode and is justified by aesthetics.** Mean-per-purchase is defended by (a) "mirror geometry — dc01's apparatus transfers," (b) heavy-buyer neutralization, (c) bounded gradients. Reason (a) is convenience for the author's existing paperwork, not a property of the data; reason (c) is trivial. Reason (b) directly *creates* M5′ — the mean-regression capacity ceiling on the data-richest users, listed as unprotected with no penalty candidate ("the knob is the normalization exponent itself"). So the single most consequential free design parameter — which, per C8, acts entirely through the metadata-vs-ID balance, i.e. exactly the R4/R6 frontier — is fixed by analogy to dc01, with the alternatives (raw counts, |H|^{−1/2}) recorded and unexamined, and its known failure mode deferred to instruments that don't exist yet.
>
> 4. **The "mirror of fI" framing is genealogy doing the work of evidence — and the file breaks the mirror every time it matters.** Count the admitted asymmetries: the +1 baseline is rank-inert in fI, live in fU; F-DC01-09's popularity conclusion "does NOT transfer"; the twin ceiling is 67.5% there, 0.61% here; the popularity braid "inverts"; route 1 "is not available on this host"; the tax is "one-sided." When essentially every load-bearing hidden-effects conclusion from dc01 must be re-derived, inverted, or discarded, "dc01's move one level up" is a slogan, not a transfer of validation. The framing imports the credibility of an SC.5-hardened sibling onto a design whose actual physics are different in every disclosed instance — and the fUfI-merge payoff ("two structurally identical composed branches") is being spent now, on a merge that is itself unscoped.
>
> 5. **The novelty is a conjunction of solved problems, probed with four queries.** The file concedes: aggregation shape is 1999-old (LSI-UP), SVD++/FISM is the canonical history-composed user, LightFM defines the feature-sum, SVDFeature licenses real-valued weights, and ProtoMF §6 names the direction as future work. What remains as "the novel union" is (i) swapping item-ID ingredients for attribute-word ingredients — a substitution any reader of LightFM makes in one sentence — and (ii) an exact decomposition that is bare linearity of the cosine numerator, i.e. algebra, not a contribution. The prior-art probe is four WebSearch queries plus one Semantic Scholar query that returned *zero results* — a null result that should indict the query, not confirm the gap. Attribute-augmented SVD++/NSVD variants and content-boosted user-profile factorization literature (a large 2008–2015 body) is exactly what narrow phrase queries miss; "found nothing close" after this effort is a statement about the search, not the field.
>
> 6. **The evaluation plan cannot attribute a win, and the bar is the paper's weakest model.** S1 is admitted "partial": the decisive control — history-composed user × plain dot product, no prototype layer (essentially LightFM-with-history-features) — is not in the frozen fleet and needs a future "charter-level decision." So the headline comparison, fU vs `user_proto`, confounds "feature composition helps" with "the prototype layer adds anything": a win is uninterpretable for the thesis's actual claim. And the stage bar itself is U-ProtoMF, the weakest published host (ml-1m NDCG .333 vs UI's .383 in the file's own Table 2 extraction) — beating a model the source paper itself superseded, on a split where the central-bet stratum may not even be resolvable: with median history 5, p90 = 12, and 25.1% of users pinned at exactly the 3-item k-core floor, D4 "history-length quartiles" are quartiles of a distribution squeezed between 3 and ~12 — the thin/heavy contrast the entire bet rides on is a few purchases wide.
>
> 7. **The cold-start rhetoric is a bait-and-switch.** Step 0 motivates the slot with the S0.2 cold-user×warm-item cells; Seed 2 is executed partly *because* the demographic alternative's benefit is "unmeasurable in the current split"; the LightFM fold-in story ("converts cold users to warm-thin after their first basket") is retailed twice. Then M6.2 concedes fU does not rescue interaction-free users, the cold-user testbed is "flagged here, not built," and the split contains zero cold users anyway. Net: every cold-flavored sentence in this document decorates a candidate that will produce no cold-user measurement this cycle, against a requirement (R5) whose absent-signal half is answered "honestly NO." The honest R5 rating is not "partial" — it is "narrow claim about warm-thin users, untestable cold claim deferred indefinitely."
>
> 8. **The rendered explanation prints numbers the design's own analysis says are partly arbitrary.** Three of the file's own findings undermine the confident two-decimal example: (i) the per-prototype coefficients t_l in *every* breakdown line are gauge quantities (3b-i: t → t+w changes no score; the printed value is the L2-canonicalized representative plus frozen init noise in weakly-pinned directions) — and the design explicitly chooses "no gauge annotation on rendered artifacts"; (ii) C6 confirms every word's share is coupled to every other through ‖q_u‖, so "your 3 swimwear purchases: −0.02" is not a removable, causal contribution — delete them and all other shares rescale; (iii) 3b-ii concedes within-clique attribution splits are "frozen init noise," and a purchase's share is a sum of word shares, so per-purchase lines inherit the noise. Exact arithmetic is repeatedly conflated with true reasons; the pitfalls pass admits "semantic faithfulness is NOT thereby guaranteed" in one paragraph while §3.4's flagship example narrates causal-sounding per-purchase magnitudes in the next.
>
> 9. **The 0.61%-twin figure is a category error deployed as the R6 defense.** 73,167 distinct normalized basket vectors in 426-dim count space says nothing about distinctness *after* the pipeline the score actually sees: projection through d=64 shared word embeddings, mean normalization, common-mass squash, and shifted cosine against K_u prototypes. The relevant quantities — near-twin angle distributions, effective rank of the composed user cloud — are exactly the instruments deferred to scrutiny, and 3b-i already predicts the cloud under-spans (aggravated in the noid arm and by M5′). Quoting "0.61% vs fI's 67.5%" as evidence that "the grounding costs almost no user individuation" (R6 row) compares a combinatorial count against a geometric collapse and calls the mismatch a two-orders-of-magnitude win.
>
> 10. **The keystone reduction and the ids/noid pair validate everything except the shipped model.** C5's bit-identity holds for fU(V=0,+ID) — an arm that is never trained or evaluated; it proves the plumbing, not the design. The actual candidate (V=426 + ID) has an admitted train/serve mismatch: it always trains with e_ID(u) present and cold-scores without it — "an operating point never visited in training" — with the fix (K3′ ID-dropout, which would *also* train the fold-in path the model is advertised on) designed and deliberately unstacked. So the design ships training a path it will never evaluate (warm+ID) as its cold story, evaluates a path it never trains (fold-in without ID), and defers the one knob that would reconcile them — while the ids-vs-noid gap it *will* measure is pre-supplied with a three-part interpretation rule (§3.6) generous enough that no outcome can embarrass the candidate. A design whose every possible result has a pre-written favorable reading is not making a falsifiable bet; it is making a narrative.

### Response (concede / rebut / open risk, point by point)

1. **Concede the unknown, rebut the "relocation," adopt the instrument
   upgrade.** Conceded: the personalized fraction of the ranking signal is the
   load-bearing unknown (bet iv), and the illustrative example's 76% baseline
   is a fair picture of the risk. **Adopted: the B(t) variance share is
   promoted to a primary scrutiny-stage metric** (alongside profile entropy),
   not a footnote instrument. Rebutted: the channel is **host-inherited** —
   `user_proto`'s published explanation device (A1 Fig. 3 s^user bars) carries
   the same +1·t_l mass *unseparated inside the bars*; fU's renderer is the
   first artifact in this lineage that names and separates it. The fU-vs-host
   explanation comparison is therefore unaffected (both play under B(t)), and
   if B(t) turns out to dominate the ranking signal, that is a finding about
   the U-host *class*, exposed by our instrument — material the thesis wants,
   not a hidden defect of the candidate. "Why for YOU" is, by definition, a
   claim about the personalized part; the explanation's job is to decompose
   that part and disclose the rest, which is exactly what the design does.
2. **Rebut the direction of the squash argument — the adversary's geometry is
   backwards on its own cited numbers; concede the attribution gap.** C9's
   toy table says angle-to-mean *falls* with |H| (72° at |H|=3 → 16° at 300):
   thin users are the *farthest* from the common-mass direction — their
   deviation Δ is proportionally largest, so they are the *least* squashed
   into the common cone, not the most (the adversary conflated "few tokens"
   with "common tokens"; a 3-item basket is 3 draws with high variance, not a
   miniature of the population). The genuine thin-user problem is *noise*, not
   collapse: 3 purchases are a high-variance taste estimate — but the host's
   alternative for the same user is a nearly-untrained free vector, i.e. pure
   init noise; fU's estimate is at least data. Conceded (and already
   adopted at 5b-6): whether a thin-stratum win is "taste composition" or
   "popularity beats an undertrained vector" needs the decoupled control —
   the §3.6 amendment names the charter request and the two free fleet
   discriminators (popularity row D4 readout, lightfm rows).
3. **Rebut the "manufactures M5′" claim on the math; concede the exponent is a
   real free parameter.** 4b C8(a) proves the metadata geometry is
   normalization-invariant (without ID, sum and mean give identical u*) — so
   mean-regression of the composed direction exists under EVERY normalization;
   the exponent moves only the ID-row counterweight. Raw counts do not avoid
   M5′; they *remove the counterweight* (a 200-item history's ID row shrinks
   to nothing — item-level... user-level memory suppressed exactly for the
   data-richest users), which is strictly worse on the M5′ axis. That is a
   math argument, not aesthetics. Conceded: the exponent is consequential and
   fixed a priori; the √-variant remains recorded as the evidence-triggered
   alternative — a mechanism decision for the user if M5′ instruments read
   badly. **Open risk, as tabled (4c).**
4. **Partly concede — as a writing discipline, not a design defect.** The
   document uses the mirror as a *checklist of what must be re-derived* (every
   3b section is a first-party derivation, and every listed asymmetry was
   found BY this cycle, not imported), and nowhere does it claim dc01's
   empirical validations transfer. But the adversary is right that
   "dc01's move one level up" rhetoric could be read as inherited credibility,
   and the merge payoff is prospective. **Adopted as thesis wording rule:
   mechanism-shape identity may be claimed; validation transfer may not.**
   The learnings-ledger's "fluent cross-level synthesis" watch-out applies —
   this goes into the ledger tagged [fU-cycle] at Step 6 (M7).
5. **Concede the probe was extendable — extended in place; rebut the
   "algebra, not contribution" frame.** The 2b amendment (dated, above) ran
   the two family queries the critique named: UISVD++, attributes-coupling MF,
   Ask-the-GRU/Movie-Genome content→factor mappings — all found, all
   distinguished (demographic attributes / item-side regularizers /
   encoder-based item cold-start; none composes users from history-derived
   attribute counts, none has a prototype read-out). On "bare linearity":
   ProtoMF's own §5.2 explanation device is "a dot product is a sum" — in
   this research program the contribution is never the algebra, it is the
   *grounded structure the algebra exposes* plus its empirical behavior; that
   is the same standard dc01 was accepted under (its 5b point 1), and the
   thesis narrative must argue it, not assume it. **Open risk (narrative),
   carried.**
6. **Concede the attribution gap (adopted, §3.6 amendment); rebut "weakest
   model" as a bar critique; concede the strata-resolution worry with a named
   fallback.** The staged bar is the ratified lineage discipline (M1; R6 row
   states it as dc01's P7 did) — U-ProtoMF is the correct like-for-like host
   for a user-side mechanism, and the headline bar lives at the merge. The
   D4-resolution point is genuinely good: quartiles of a 3–12-squeezed
   distribution are a narrow contrast. **Adopted: the run plan will read the
   thin-user hypothesis on explicit history-length bands (3 / 4–5 / 6–12 /
   13+) rather than raw quartiles, and names `hm_3_month` (charter C2 written-
   reason clause) as the resolution fallback if V1's bands lack power.**
7. **Concede the rhetorical surplus; rebut the rating change.** The
   substantive content is already what the adversary demands: M6.2 says
   "honestly NO" for absent-signal users, R5's row says it, and no cold-user
   measurement is promised this cycle. The fold-in property is a real,
   verified mechanism (C3a + B5 §7.1), not decoration — but it will be
   *presented* as a deployment property, not an evaluated claim. R5 stays
   "partial (user-side re-scope)" — the requirement's own text covers the
   weak-signal half, which IS measurable and measured (D4); a "weak" rating
   would misstate a candidate that delivers the only measurable half of the
   user-side R5 there is.
8. **Concede the tension; the resolution is the ratified disclosure
   architecture, now cross-referenced at the example itself (adopted).** All
   three "arbitrariness" findings are this document's own first-party results
   (3b-i, C6, 3b-ii) — the adversary quotes the design to itself. The split
   (figures carry arithmetic honesty; epistemics live in the thesis
   hidden-effects section and the surrounding thesis text) is the user's
   SC.5 gate decision, applied symmetrically to both models. What was missing
   was a pointer at the flagship example so no reader takes the two-decimal
   numbers as causal — **added as the §3.4 honesty note (visible
   amendment).**
9. **Concede the category distinction — adopted (R6 row qualified, visible
   edit):** the 0.61% is the exact-collision (representational) ceiling;
   angular individuation after composition is a geometric question owned by
   the named 3b instruments (near-twin angles, effective rank). The number
   still carries real information — fI's 67.5% was *also* a representational
   ceiling and it correctly predicted the tie-block phenomenology — but the
   R6 row no longer lets it stand in for the geometry.
10. **Rebut the frame; concede the fold-in evaluation gap.** Keystones are
    plumbing proofs *by design* — that is their protocol role (dc01 i02
    precedent): they pin the reduction the ablation logic rests on; nobody
    claims C5′ validates the shipped model. The noid arm IS trained and
    evaluated (C7 ablation pair), so "validates everything except the shipped
    model" overstates. The interpretation rule does not immunize the
    candidate — it *classifies* failures (a gap concentrated in user-level
    individuality is a different verdict than one flat across strata), and a
    flat fU-below-host result embarrasses the candidate under every reading.
    Conceded and standing: the cold/fold-in path (score without ID) is
    advertised as a property but unevaluated this phase (M6.2's declared
    state), the train/serve mismatch carries F-DC01-04's caveat, and K3′
    adoption — which would train that path — is the user's mechanism
    decision, on the record.

### Amendments made after 5b (visible edits, 2026-07-12)

1. §2b extended with the two family-probe queries + findings (point 5).
2. §3.4 rendered example gains the epistemics pointer note (point 8).
3. §3.6 decoupled control promoted to a named charter-amendment request +
   free-discriminator note (points 2/6).
4. §5 R6 row: 0.61% qualified as representational vs geometric (point 9).
5. Response-level adoptions recorded: B(t) variance share promoted to primary
   metric (point 1); explicit history-length bands + hm_3_month fallback
   (point 6); thesis wording rule on mirror rhetoric (point 4, to ledger at
   Step 6).

## 6. Fingerprint

1. **Features-entry:** history-composed user factors — the free user embedding
   is replaced by the mean of the train-basket's item-attribute-word embeddings
   (+ user-ID row), feeding U-ProtoMF's unchanged user-prototype layer
   (LightFM/SVDFeature template, one level up from dc01; lineage sibling — dc01
   distinctness waived per brief M1).
2. **Grounding:** user-prototype meaning via shared geometry — word embeddings
   and user prototypes co-live in R^d; a taste community's profile =
   cos(e_f, p^u_l) over the item vocabulary (enabled, not loss-enforced — the
   same grounding-strength point as dc01, user side).
3. **Read-out:** per-prototype contribution t_l·u*_l with the **disclosed
   non-personalized item baseline B(t) = 1ᵀt** (NOT rank-inert on this host) ×
   exact per-word AND per-purchase shares of each community activation (two
   exact regroupings of one sum; jointly ‖q_u‖-normalized) × global community
   profile in item vocabulary for naming.

Status: **reviewed — awaiting build** (cycle closed 2026-07-12; concept review
closed by user same day — mechanism, mean normalization, configs, naming, and
renderer defaults ratified; K6′ + M5′ insights + B(t) directives added as
dated review amendments. Next: build session (SC.3-style, keystone
bit-identity vs `user_proto`, renderer/cards/naming slots + the A1
term-by-term objective-equality check) → SC(dc05) scrutiny routine).
