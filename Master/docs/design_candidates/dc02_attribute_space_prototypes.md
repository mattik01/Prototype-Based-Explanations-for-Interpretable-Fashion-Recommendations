# dc02 — Attribute-Space Item Prototypes (Concept-Bottleneck-Anchored Prototype Layer)

> Produced by the design-candidate protocol (`Master/docs/design_candidate_protocol.md`), cycle 2, 2026-07-07.
> Status: **draft — awaiting review** (registered 2026-07-07).
> *(File renamed from `dc02_draft.md` after Step 1 fixed the seed.)*
> Cycle constraint (user, 2026-07-07): dc01 is still under validation — this candidate
> treats dc01 **only as occupied fingerprint space** (dedup), and must not build on any
> of dc01's implementation learnings, empirical behavior, or assumed correctness.

## 0. Frame

- Requirements re-read in full (living version of 2026-06-11; spectrum reframing, R7
  vocabulary balance, R8 parsimony, S4 LLM naming, S5 pipeline compatibility unchanged
  since dc01's cycle).
- **Occupied fingerprint (dc01):** entry = *features→factors* (sum of feature-value
  embeddings replaces the item ID table); grounding = *shared geometry* cos(e_f, p_k),
  explicitly **unenforced**; read-out = per-prototype contribution + exact per-feature
  shares. dc02 must differ on ≥1 axis.
- **Open regions:** (a) prototypes living *in an explicit attribute/feature space*
  (dc01's rejected-but-retained Seed 2, concept-bottleneck flavor — grounding axis);
  (b) projection/push-to-real-items grounding (dc01 Seed 4 — grounding axis);
  (c) features entering as a **prior/regularizer on the prototype space** rather than
  through the score path (scratchpad, S7 §4.1.3 REGULARIZE route — entry axis; flagged
  in the scratchpad as a citable novelty gap for the linear category);
  (d) user-side features (S2) and images (S3) — untouched niches.
- **Open niche on requirements:** dc01's grounding is *unenforced* (feature meaning is
  read out of shared geometry but nothing constrains prototypes to stay feature-
  readable). A candidate whose grounding is **enforced by construction or by loss**
  serves R4 from a direction dc01 leaves open — without relying on any dc01 finding.
- **Open scratchpad entries (all currently unmarked, hence open):** "Attach feature
  profiles to existing CF prototypes" (2026-06-11); "Feature-injection fork:
  prediction vs prior" + "Three ways attributes enter: add/interact/regularize" +
  "Attribute-source prioritization; intrinsic > post-hoc" + "Candidate-selection
  sequencing: item-only first" + "Two axes (grounding vs scoring) + Rudin axis" +
  "Hybrid bridge desideratum" (all 2026-06-16). Most relevant to this cycle: the
  add/interact/regularize fork (names the unexplored REGULARIZE entry route), the
  attach-profiles-to-CF-prototypes idea (a grounding-axis seed), item-only-first
  sequencing (stage-1 scoping), and the two-axes caution (gut-check any seed for
  Axis-A substance). All get formal dispositions in Step 1b.

## 1. Seed selection

Corpus swept across all themes (B backbone, C/D prototypes, E concepts/attention,
F disentanglement, H fashion, §I forward citations). One registered candidate (dc01)
→ every shortlisted seed carries an explicit distinctness sentence against dc01's
fingerprint: *features→factors (Σ feature embeddings replaces item ID table) ×
shared-geometry grounding, unenforced × per-prototype contribution + per-feature
shares*.

### 1b. Scratchpad sweep (all 7 open user entries + 0 prior captured entries)

1. **"Attach feature profiles to existing CF prototypes" (2026-06-11)** —
   *not this cycle (competed as shortlist Seed 2, lost to by-construction grounding;
   its concrete mechanism refinement is preserved as `[captured dc02]` in the
   scratchpad and remains a strong future seed)*.
2. **"Feature-injection fork: prediction/representation vs prototype-space prior"
   (2026-06-16)** — *shapes this candidate (frames the entry-axis choice; dc02
   deliberately takes route A — features enter the representation — while route B
   remains open via shortlist Seed 3 / the captured loss-enforced variant)*.
3. **"Three ways attributes enter — add / interact / regularize" (2026-06-16)** —
   *shapes this candidate (dc02 is neither ADD nor INTERACT nor REGULARIZE at the
   score level: features enter by defining the space the prototype layer measures
   similarity in — a fourth, representation-level point the entry's taxonomy helps
   articulate; the REGULARIZE route stays open as Seed 3)*.
4. **"Attribute-source prioritization is interpretability-driven; intrinsic >
   post-hoc" (2026-06-16)** — *shapes this candidate (its core reframing — the
   interpretability win = moving prototype GROUNDING from post-hoc to intrinsic via
   feature-space prototypes — is precisely dc02's headline; item attributes primary,
   as prescribed)*.
5. **"Candidate-selection sequencing: item-only → unified → separate sets"
   (2026-06-16)** — *shapes this candidate (dc02 stays stage-1: item-attribute-only;
   user branch untouched)*.
6. **"Design caution: two axes (grounding vs scoring) + Rudin axis" (2026-06-16)** —
   *shapes this candidate (gut-check passed: dc02 is pure Axis A — it changes how
   prototypes acquire meaning, leaves the dot-product scorer untouched; Rudin axis:
   it trades expressiveness FOR simplicity, the approved direction)*.
7. **"Hybrid desideratum: bridge between added module and prototype grounding"
   (2026-06-16)** — *not this cycle (dc02 adds no separate module — nothing to
   bridge; re-enters the sweep when a hybrid candidate appears)*.

### Shortlist

**Seed 1 — E1 Concept Bottleneck × A1 ProtoMF (+ C2 formulations menu): the item
prototype space moves into attribute space.** Item prototypes are vectors in an
explicit attribute/feature space; an item's prototype activation is computed from its
attribute profile, so a prototype literally *is* a readable attribute profile.
- *Motivation gate:* maximal R4 (grounding by construction, nothing to enforce or
  read out indirectly) + R2/R3 (activations in attribute vocabulary ARE the intrinsic,
  prototype-shaped explanation) + R5 (any cold item has attributes → full-strength
  activation) + R1 (the activation vector is the item representation the tied score
  consumes); known risk is R6 — the concept-bottleneck accuracy trade-off (E1/E3).
- *Distinctness vs dc01:* differs on **grounding** (by construction in attribute
  space vs. unenforced shared-geometry read-out) and **entry** (features define the
  space the prototype layer measures similarity in, vs. features composing a latent
  CF factor under untouched CF prototypes). Read-out shape may resemble dc01's
  (per-prototype contributions) — one differing axis suffices; here there are two.
- *R8 parsimony:* PASS — exactly one idea: "move the item prototype layer into
  attribute space." Same conceptual size as `user_item_proto`.
- *Vehicle note:* this is NOT a pure concept bottleneck (the capped `S1-contrast`
  class) — the explanation vehicle remains similarity-to-prototypes; the bottleneck
  only relocates where prototypes live. Headline-eligible.

**Seed 2 — scratchpad "attach feature profiles to CF prototypes" × E2 SENN-style
consistency objective: loss-enforced grounding.** CF prototypes and item embeddings
stay untouched; each prototype gets a learned attribute profile a_k bound to its CF
activations by one consistency loss; cold items activate via the surrogate a_k·x_i.
- *Motivation gate:* R4 (learned, loss-enforced profiles) + R2/R3 + R5 (surrogate
  path) with the least R6 risk of the shortlist (CF capacity fully preserved); its
  weakness is R1/R2 fidelity — the score does not flow through a_k, so grounding
  faithfulness holds only as far as the loss binds.
- *Distinctness vs dc01:* differs on **grounding** (loss-enforced learned profile vs.
  unenforced geometry) and **entry** (features attach at the prototype level; the
  free item ID table survives, unlike dc01's replacement of it).
- *R8 parsimony:* PASS — one mechanism (profile + its binding loss).
- Provenance would be `scratchpad+corpus`. **Not chosen** (see below); mechanism
  preserved as `[captured dc02]`.

**Seed 3 — S7 §4.1.3 REGULARIZE route: features as a pure prior on the prototype
space.** Score path untouched; a feature-similarity pull regularizes items'
prototype-similarity profiles (feature-similar items → similar profiles).
- *Motivation gate:* serves identifiability (§4) and partially R4, and S7 flags this
  combination as an unobserved gap (citable novelty); but it produces **no intrinsic
  read-out quantity** by itself (weak R2) and gives truly unseen items no
  representation (weak R5) — it is constraint-shaped, not candidate-shaped.
- *Distinctness vs dc01:* differs on **entry** (regularizer, not the score path).
- *R8 parsimony:* PASS as one idea — but the motivation gate rejects it as a
  *standalone* candidate this cycle; it remains prime material for Step-3 constraint
  design and a possible future `S1-contrast`-adjacent study.

**Seed 4 — C1 ProtoPNet push/projection onto real items' attribute profiles**
(dc01's retained Seed 4, re-swept as a free lead).
- *Motivation gate:* R4 via "prototype = a real item's attributes," but the read-out
  collapses toward the existing post-hoc `top_k_items` baseline (k=1, enforced in
  training) — smallest R2 gain on the table; contributes nothing to R5 alone.
- *Distinctness vs dc01:* differs on **grounding** (projection-to-real-items).
- *R8 parsimony:* PASS — one idea. Not chosen; stays a free lead (possible grounding
  add-on study).

### Chosen seed

**Seed 1 — E1 Concept Bottleneck (Koh et al., ICML 2020) × A1 ProtoMF (Melchiorre et
al., RecSys 2022), with C2 (Li et al. 2024) as the prototype-formulation menu:
attribute-space item prototypes.** Provenance: **corpus**.

Why it won: it attacks the requirements' center of gravity (R1+R2+R4+R5
simultaneously) with the strongest possible R4 story — grounding that needs no
enforcement because it holds by construction — and it occupies exactly the open
niche identified in Step 0 (dc01's grounding is unenforced; dc02 sits at the
opposite end). Its one salient risk (R6, bottleneck expressiveness) is precisely the
fidelity-vs-accuracy tension §4 of the requirements asks a candidate to quantify,
with E1/E3 providing the literature frame. Against Seed 2 it wins on R1/R2 purity
(the explanation-bearing quantities are the score path, not an auxiliary trained-on
profile); Seed 2's mechanism is captured, not discarded. Per the cycle constraint,
nothing here builds on dc01's validation state — the seed stands on E1/C2/A1 alone.

## 2. Mechanism extraction

All statements below were verified against the local PDFs on 2026-07-07:
`E_concepts_attention/E1_koh_2020_concept_bottleneck_models.pdf`,
`A_protomf/A1_melchiorre_2022_protomf.pdf`,
`C_prototype_interpretability/C2_prototype_formulations_overview_2024.pdf` (arXiv
v4, 2026-01-07 — note: newer than the reading list's "2024" label).

### 2.1 E1 — Concept Bottleneck Models (Koh et al., ICML 2020)

**Core mechanism (§3 "Setup").** Predict a target y from input x with concepts
c ∈ R^k annotated at training time. A concept bottleneck model has the form
ŷ = f(g(x)), where g: R^d → R^k maps the input into concept space ("bone spurs",
…) and f: R^k → R maps concepts to the target. The defining property: *the
prediction relies on x only through the k-dimensional concept vector* ĉ = g(x).
"Task accuracy" = how well f(g(x)) predicts y; "concept accuracy" = how well g(x)
predicts c.

**Three training schemes (§3, enumerated list).**
1. *Independent:* f̂ trained on the **true** concepts c; ĝ trained separately on
   (x, c); at test time f̂(ĝ(x)).
2. *Sequential:* ĝ first, then f̂ trained on the predictions ĝ(x).
3. *Joint:* minimize L_Y(f(g(x)); y) + Σ_j λ L_{C_j}(g_j(x); c_j) — the weighted
   sum; standard end-to-end model = λ→0, sequential = λ→∞.

**What they showed (§4.2, Table 1).** Bottleneck models are competitive on task
error: on OAI (x-ray grading, n=36,369, k=10 clinical concepts), joint/sequential
bottlenecks *beat* the standard model (y RMSE 0.418 vs 0.441); on CUB (bird
classification, n=11,788, k=112 binary attributes denoised to class-level), the
joint model is slightly worse (0-1 error 0.199 vs 0.175) but "closes most of the
gap" (§4.2). Figure 2-Left: little or no task-vs-concept trade-off on OAI, some on
CUB.

**Intrinsic beats post-hoc concept access (§5, Table 2).** Linear probes recovering
the same concepts post-hoc from a standard model are *worse* than reading the
bottleneck directly (OAI concept RMSE 0.68 probe vs ≈0.53 bottleneck; CUB 0.09 vs
≈0.03), and post-hoc probes "do not enable interventions" (§5). Direct
evidence for the intrinsic-over-post-hoc claim (our R2), from the concept side.

**Test-time intervention (§6).** Because ŷ depends on x only through ĉ, a human
can edit a concept value and propagate the change — the interaction mode that a
bottleneck buys and a side-channel destroys.

**Admitted limitations (§8 "Discussion").** (a) Requires concept annotations at
training time. (b) **"Side channel from x → y"**: adding a direct x→y path on top
of the bottleneck is "equivalent to using the concepts as auxiliary features" and
breaks clean intervention — the bottleneck property is fragile to bypass paths.
(c) Theory sketch (§8): for linear regression, asymptotic excess-error ratio of an
independent bottleneck vs a standard model is ≤ (k/d·σ_Y² + σ_C²)/(σ_Y² + σ_C²) —
bottlenecks are effective when k ≪ d and concept noise σ_C² is low relative to
target noise.

**Setting note for us (flagged: "I propose", not in the paper).** On H&M the item
concepts are *observed at inference* (articles.csv attributes), not predicted from
raw input — g becomes the identity/oracle on given attributes. The concept-accuracy
half of CBM's problem vanishes; what remains, and what dc02 inherits, is the
*bottleneck property itself*: the item enters the score only through its attribute
vector, with the expressiveness cap that implies.

### 2.2 A1 — ProtoMF (Melchiorre et al., RecSys 2022)

**U-ProtoMF (§3.1).** L^u user prototypes p^u ∈ R^d. Eq. (1): the user
representation is the similarity vector
u* = [sim(u, p_1^u), …, sim(u, p_{L^u}^u)] ∈ R^{L^u}, with the **shifted cosine**
sim(a,b) = 1 + aᵀb/(‖a‖·‖b‖) ∈ [0,2] guaranteeing u* ≥ 0. Eq. (2): U-score(u,t) =
u*ᵀ t, where the item embedding t ∈ R^{L^u} — i.e. item dimensions are weights on
user-prototype similarities, and the score is a sum of per-prototype
contributions s_l = u*_l · t_l (the interpretability handle, §3.1 last ¶ / §5.2).

**Loss (Eq. 3).** Sampled cross-entropy/softmax over the interactions + λ_L2‖Θ‖.

**Inclusion-criteria regularizers (Eqs. 4–6).** R_{P^u→U} = −(1/L^u) Σ_l max_i
sim(u_i, p_l^u) (every prototype "gets matched" by ≥1 user) and R_{U→P^u} =
−(1/N) Σ_i max_l sim(u_i, p_l^u) (every user by ≥1 prototype), both computed
in-batch as an approximation (§3.1, last paragraph); total
L_U-Proto = L_rec + λ1 R_{P^u→U} + λ2 R_{U→P^u}. These are the repo's
`sim_proto` / `sim_batch` losses.

**I-ProtoMF (§3.2, Eqs. 7–10).** Exact mirror on items: t* ∈ R^{L^t} from item
prototypes p^t; I-score(u,t) = t*ᵀ u with u ∈ R^{L^t}; inclusion criteria Eqs.
(9)–(10) with λ3, λ4.

**UI-ProtoMF (§3.3, Eqs. 11–12).** Two linear maps tie the branches: û = W^u u ∈
R^{L^t} and t̂ = W^t t ∈ R^{L^u}; UI-score = u*ᵀ t̂ + ûᵀ t*. Loss = sum of both
sides' losses with L_rec computed once on the UI-score (footnote 1). This is the
repo's `prototypes_double_tie`.

**Evaluation (§4).** ml-1m / lfm2b-1mon / AmazonVid, leave-one-out, 99 sampled
negatives, HR@10/NDCG@10; ProtoMF beats MF, RBMF, ACF.

### 2.3 C2 — Prototype formulations overview (Li et al., arXiv:2410.08925v4)

**The menu (§3, Fig. 1–2).** Four prototype formulations: point-based Euclidean
(Eq. 1: s_L2 = log((‖z−p‖²+1)/(‖z−p‖²+ε))), point-based cosine/hyperspherical
(Eq. 2: s_cos = zᵀp/(‖z‖₂‖p‖₂)), probabilistic Gaussian (Eqs. 3–4), and their
new HyperPG (Gaussian over cosine similarities, Eqs. 5–6).

**Prototype losses (§4.2–4.3).** ProtoPNet-lineage training adds a cluster loss
(Eq. 7: pull each sample toward its class's most similar prototype) and a
separation loss (Eq. 8: push away other classes' prototypes) to the task loss:
L = L_CE + λ_Clst·L_Clst + λ_Sep·L_Sep. Structurally these are max-over-prototypes
inclusion criteria — the same shape as ProtoMF's Eqs. (4)–(5), which A1 (§3.1)
indeed attributes to the prototype-classification lineage.

**Robustness finding (Table 2, §5.2, §7).** Under a drastically simplified
training scheme (single optimizer, no warm-ups), hyperspherical/cosine prototypes
stay competitive (CUB top-1: cosine 71.7%, HyperPG 74.3% vs Euclidean 61.4%,
which collapses from 73.9%); Euclidean similarity "starts to perform worse with
higher numbers of dimensions" (§3.1.1). **Design consequence for dc02:** keep
ProtoMF's shifted-cosine similarity when relocating the prototype space — it is
both the host's native choice (A1 Eq. 1) and the most training-robust formulation
on C2's evidence; no new similarity function is introduced.

## 2b. Prior-art probe

Probe target: the **adaptation** — item prototypes that live in an (embedded)
attribute space inside a user-item MF/prototype-similarity scorer, with observed
item attributes as the only item input (concept-bottleneck property) and
per-prototype attribute-vocabulary read-out. Run 2026-07-07.

**Queries (reproducible):**
1. `concept bottleneck recommender system interpretable recommendation attributes`
2. `attribute prototype recommendation "prototype" item attributes cold-start
   matrix factorization explainable`
3. `"prototypes" feature space "side information" OR metadata recommendation
   interpretable prototype grounded attributes fashion`
4. `papers citing "ProtoMF" 2026 feature-aware OR content-based OR attribute
   prototypes`

**Findings:**
- **Nothing matching the adaptation.** No CBM-style recommender with a prototype-
  similarity representation over observed item attributes surfaced in any query.
- **Closest hit — PAICM** (Han, Song, Yin, Wang, Nie, "Prototype-guided
  Attribute-wise Interpretable Scheme for Clothing Matching", SIGIR 2019, DOI
  10.1145/3331184.3331245; verified in the actual PDF — the first WebFetch
  auto-summary of it was hallucinated and discarded). PAICM learns latent
  compatible/incompatible **attribute-interaction prototypes via NMF** as
  *templates* for interpreting top–bottom **outfit compatibility** (BPR over item
  pairs) and suggesting alternatives (Abstract; §1 contributions). Distinction:
  different task (item–item compatibility, no users, no recommendation scoring),
  different mechanism (NMF co-occurrence templates consulted for interpretation,
  not a similarity-vector representation that IS the entity's representation in
  the score), no bottleneck property, no cold-start role. **Distinguish, cite** —
  it does establish "attribute prototypes are meaningful in fashion," which
  *supports* our premise. Candidate addition to reading list Theme H.
- **GIANT** ("Explainable Recommender with Geometric Information Bottleneck",
  arXiv:2305.05331): "bottleneck" here is the information-bottleneck objective on
  a graph/review VAE — no observed-attribute concept layer, no prototypes in our
  sense. Not close.
- **Information-bottleneck recsys hits** (knowledge-refined IB, ScienceDirect
  2025) — same disambiguation: IB ≠ concept bottleneck. Not close.
- **§I forward-citation re-check:** the search surfaced the known citers (PPA++
  2026, cultural-diversity/I2 lineage) plus no new feature-grounded ones; the
  reading list's takeaway (all citers CF-only, no side-feature grounding) still
  stands for this mechanism as of 2026-07-07.

**Verdict: found nothing close — the specific adaptation appears unpublished.**
The nearest neighbours are PAICM (fashion attribute prototypes, different task &
mechanism), D1 ProtoCF (prototypes for cold-start, composed from *interaction*
few-shot, not metadata — the reading list's standing contrast), and E1 CBM itself
(classification, not recommendation). Novelty gate: **distinguish** (no refinement
needed); the candidate proceeds unchanged.

## 3. Adaptation design

**One idea (R8):** the **item prototype space moves from CF latent space into the
item-attribute space**. An item prototype is a vector of weights over the H&M
attribute *values* (a readable attribute profile by construction); an item is
represented **only** by its observed attribute multi-hot (concept-bottleneck
property: no free per-item parameters anywhere in the item branch); everything
else — user branch, shifted cosine, per-prototype score decomposition, inclusion
regularizers, loss, sampler, trainer — is the host, unchanged.

### 3.1 Data interface

`data/<hm_split>/item_features.csv` provides F=9 categorical fields per item
(verified 2026-07-07 on `hm_3_month`: product_group 17, product_type 118,
graphical_appearance 29, colour_group 49, perceived_colour_master 20, index_group
5, garment_group 21, section 54, department 221 → **V = 534** one-hot columns; the
`item` column is the original article id — an identity column, excluded). Each
item sets exactly one value per field (no missingness, 4.0 analysis), so its
multi-hot x_i ∈ {0,1}^V has **exactly F=9 ones and ‖x_i‖₂ = 3 for every item** —
a property the read-out exploits (§3.4).

Build step (in `Trainer._build_model` / `Tester`, mirroring how other artifacts
are injected at build time): assemble X ∈ {0,1}^{M×V} from the CSV, pass it into
the factory config; it is registered as a **non-persistent buffer** (never
checkpointed; rebuilt deterministically from the CSV, so checkpoints hold only
learned weights). Memory: 26,963 × 534 floats ≈ 55 MB dense — fine; a sparse/
uint8 variant is an implementation detail, not a design point.

**Honesty note (measured, not estimated):** the 9 fields induce only **12,576
distinct signatures over 26,963 items** (68.1% of items share their full
signature with ≥1 other item; largest class = 146 items; `hm_full`: 33,605 /
90,690, 77.4%, max 304). A pure attribute bottleneck therefore **scores
attribute-equivalence classes, not items** — same-signature items receive
identical scores for every user (exact ties, broken arbitrarily by sort order).
This is dc02's structural expressiveness cap; it is quantified further in Step 4
and weighed in Step 5 (R6). Note the flip side: this is precisely the mechanism
that makes cold items first-class citizens (R5) — a cold item lands exactly where
its warm signature-mates are.

### 3.2 Modules & forward pass (anti-vagueness gate)

**New ft_type (I propose): `attr_item_proto`** — the UI-ProtoMF/double-tie
analogue with an attribute-space item branch. (CLAUDE.md's reserved names
`feature_item_proto` / `dual_item_proto` denote other mechanisms — dc01's and the
dual-space idea respectively; this is a third, to be added to the reserved list.)

**New class 1 — `AttributeLookup(FeatureExtractor)`**
(`feature_extraction/feature_extractors.py`):
`__init__(self, attr_multi_hot: torch.Tensor)` registers X (M, V) as a
non-persistent buffer; `forward(o_idxs)` returns `X[o_idxs]` — shape
`(batch, 1+n_neg, V)` (or `(batch, V)` for 1-dim input). No parameters, no loss.

**New class 2 — `AttributePrototypeEmbedding(FeatureExtractor)`**: structurally
the host `PrototypeEmbedding` with the base embedding replaced by an
`AttributeLookup` and prototypes living in R^V:
- `self.prototypes = nn.Parameter(torch.randn(K, V))` — K item prototypes **in
  attribute space**.
- `forward(i_idxs)`: `x = lookup(i_idxs)` (B, 1+n_neg, V) →
  `sim_mtx = 1 + cos(x.unsqueeze(-2), prototypes)` (B, 1+n_neg, K) — the host's
  shifted cosine (A1 Eq. 1; kept per C2's robustness evidence, §2.3) → accumulate
  the host regularizers on `sim_mtx.reshape(-1, K)` (§3.5) → return `sim_mtx`.
  The returned t* ∈ R^K **is** the item representation (A1 Eq. 7 with p^t → a_k ∈
  R^V, t → x_i).
- `get_and_reset_loss()`: identical to host (`sim_proto_weight`·R_proto +
  `sim_batch_weight`·R_batch).

**New class 3 — `AttributeProjection(FeatureExtractor)`**: the double-tie
projection branch, `EmbeddingW`'s role with the shared attribute representation:
`forward(i_idxs) = self.linear(lookup(i_idxs))` with `nn.Linear(V, L_u,
bias=False)` → t̂ (B, 1+n_neg, L^u). The "tie" (R1): **both** item branches
consume the *same* x_i — features drive the one representation that the score
uses, twice, exactly as the host ties the one item embedding into both halves
(A1 §3.3). There is nothing else the item could enter through.

**Factory branch (`feature_extractor_factories.py`,
`ft_type == 'attr_item_proto'`):**
- User side: **byte-identical to `prototypes_double_tie`** — CF `PrototypeEmbedding`
  (L^u user prototypes in R^d) + `EmbeddingW` (d → K) with the embedding table
  weight-tied; u* ∈ R^{L^u}, û ∈ R^K.
- Item side: `ConcatenateFeatureExtractors(AttributePrototypeEmbedding,
  AttributeProjection, invert=True)` → [t̂ (L^u); t* (K)].
- Score (host `RecSys.forward`, **unchanged**): dots = [u*; û]·[t̂; t*] =
  u*ᵀt̂ + ûᵀt* — A1 Eq. 12 with the item halves attribute-derived.
- **`use_bias = 0` (I propose, required):** the host's per-item bias
  (`RecSys.item_bias`) is a free per-item parameter — exactly the "side channel
  from x→y" E1 §8 warns breaks the bottleneck, and it is untrained for cold
  items. The host config already exposes `use_bias`; dc02 fixes it to 0.
  (Coupling caveat: the flag also disables the user/global bias — acceptable;
  ProtoMF's headline results don't hinge on biases, and a decoupled
  user-bias-only flag is a trivial future knob.)

**Minimal variant (ablation rung, not a second idea):** the I-ProtoMF analogue —
user = `Embedding(n_users, K)`, item = `AttributePrototypeEmbedding` alone,
score = ûᵀt* (A1 Eq. 8). Same idea, half the architecture; useful to isolate the
attribute-space prototype layer from the double-tie scaffolding.

**Config (`confs/hyper_params.py`):** clone the `user_item_proto` space
(embedding_dim ∈ [10,100], n_prototypes ∈ [10,100] per side, sim weights
loguniform [1e-3,10], reg types 'max', cosine 'shifted', losses/optimizer as
host) with: item-side `n_prototypes` = K (attribute prototypes; same range to
start), `use_bias: 0`, no item `embedding_dim` (the item branch has none — its
"dimension" is V, data-determined).

**Parameters:** user table N_u×d, user prototypes L^u×d, user projection d×K
(all host); item side: **A (K×V) + W_t (L^u×V)** — replacing the host's item
table M×d + item prototypes L^t×d + item projection d×L^u. No per-item
parameters remain (M appears nowhere).

### 3.3 Where this sits vs the seeds ("the paper says" / "I propose")

- *E1 says:* prediction depends on x only through the concept layer; competitive
  accuracy; intrinsic concept access beats post-hoc probes (§2.1). **I propose:**
  concepts = observed attribute values (g = identity — E1's concept-accuracy
  concern vanishes, its expressiveness cap remains), and the c→y half is not an
  MLP but the host's prototype-similarity layer + linear score — keeping R3.
- *A1 says:* everything in §2.2 — kept verbatim except the item branch's input
  space. **I propose:** relocating p^t from R^d (CF) to R^V (attributes) — the
  entire candidate.
- *C2 says:* cosine formulations are the training-robust choice (§2.3).
  **I propose** nothing new here — host similarity kept; C2 is the justification
  for *not* introducing an L2/Gaussian formulation alongside the relocation.

### 3.4 Intrinsic explanation read-out (R2/R3/R4)

All quantities below are computed in the forward pass or are model parameters —
nothing is reconstructed post-hoc.

1. **Per-prototype score contributions** (host mechanism, A1 §5.2): the score
   splits exactly into L^u + K addends — u*_l·t̂_l (user prototype l × item's
   learned affinity to it) and û_k·t*_k (user's affinity to item prototype k ×
   the item's attribute-space activation).
2. **Prototype = attribute profile, by construction (R4):** prototype k IS
   a_k ∈ R^V — render as its top-weight values per field:
   "P7 ≈ department: Denim Trousers (0.41) · type: Trousers (0.33) ·
   colour: Dark Blue (0.19) · pattern: Solid (0.12)". No naming procedure
   needed to *ground* it (the post-hoc `naming/` layer and S4 LLM renaming remain
   available for *verbalisation* on top).
3. **Exact per-attribute decomposition of every activation:** since ‖x_i‖ = 3
   for all items, t*_k = 1 + (Σ_{v∈x_i} a_{k,v}) / (3‖a_k‖) — each of the item's
   9 attribute values contributes the additive share a_{k,v}/(3‖a_k‖).
   "This item activates P7 at 1.72: department=Denim +0.31, type=Trousers +0.24,
   colour=Dark Blue +0.11, pattern=Solid +0.04, …". Exact, not approximated.
4. **User-prototype attribute affinities:** each row W_t[l,:] ∈ R^V is the
   learned attribute profile that user prototype l responds to — the same
   render as (2) applies to the u*ᵀt̂ half.

**Rendered example (H&M-style):** *"Recommended for you: slim dark-blue jeans.
Main reason: you strongly match item prototype P7 — 'Menswear denim: Trousers,
dark tones, solid' (your affinity û_7 = 1.8, this item activates it at
t*_7 = 1.72, of which department=Denim contributes +0.31, type=Trousers +0.24,
colour=Dark Blue +0.11). Second: your 'Basics buyer' user prototype P3 responds
to this item's attributes at t̂_3 = 1.1 (top: garment_group=Jersey Basic
+0.4)."* — every number read off the forward pass.

> **Amendment (2026-07-07, post-5b critique pt. 4).** The shifted cosine's +1
> injects a *user-constant* term Σ_k û_k into every item's score (constant
> across items for a given user, hence ranking-irrelevant). Rendered
> explanations therefore report the **item-discriminating** contribution
> û_k·(t*_k − 1) per prototype — which the per-attribute shares in (3) decompose
> *exactly and completely* (they sum to t*_k − 1) — with the user-constant
> baseline disclosed once rather than folded into per-item numbers. The u*ᵀt̂
> half contains no such constant and renders as-is.

### 3.5 Interpretability-constraint design (protocol v1.1, mandatory)

**Inheritance.** Both host inclusion criteria transplant *with their semantics
intact* — arguably strengthened: R_batch (each sampled item close to ≥1
prototype) and R_proto (each prototype claimed by ≥1 item; 'max'/'soft'/'incl'
variants, `PrototypeEmbedding` lines 361–377) now operate on the attribute-space
sim matrix (B(1+n_neg), K). R_proto in particular now pulls every prototype
toward *realized attribute combinations* — it directly fights off-support
profiles (mode (b) below). `sim_proto_weight`/`sim_batch_weight` remain the
knobs; **both Rashomon-tunable**. The host `max_norm` knob becomes inert on the
item side (no item embedding); the user side inherits everything unchanged.

**New degeneration modes & designed (UNSTACKED, R8) constraint candidates:**

- **(a) Negative-weight profiles.** a_k ∈ R^V may go negative; "anti-Denim
  (−0.4)" reads worse than a parts-based profile. *Constraint:* hard
  nonnegativity by reparameterization a_k = softplus(ã_k) (zero extra loss, one
  line in `__init__`/`forward`), or projected clamp `A.data.clamp_(min=0)`
  post-step. Expected effect: NMF-like additive profiles; cost: expressiveness ↓,
  runtime ~0. Binary mechanism knob (user decision), not Rashomon-tunable.
- **(b) Off-support profiles** (weight on attribute combinations no real garment
  has, e.g. department=Menswear ∧ section=Womens Lingerie — the semantic-gap /
  R7 perceptual-alignment failure). Partially covered by inherited R_proto (see
  above), **not fully** (max-based R_proto satisfies itself with one close item
  while mass sits elsewhere). *Constraint:* prototype-to-data cluster pull
  L_push = (1/K)Σ_k min_{i∈S} (1 − cos(a_k, x_i)) over a resampled item subset S
  (C2 Eq. 7's cluster loss at prototype level), hooked via
  `get_and_reset_loss()`; or the hard variant — ProtoPNet's push (C1): every T
  epochs snap a_k ← x_{i*(k)}. Cost O(K·|S|·V) per step (sparse x: trivial).
  Weight Rashomon-tunable.
- **(c) Single-field collapse** (all prototypes differentiate only on
  `department`, the strongest signal — colour/pattern coordinates stay flat →
  explanations collapse to internal taxonomy, the exact R7 tension from the 4.0
  analysis). *Constraint:* per-field crispness L_crisp = (1/KF)Σ_k Σ_f
  H(softmax(a_k[f])) (low entropy within each field per prototype → every field
  states a clear preference), added to `get_and_reset_loss()`; cost O(K·V).
  Weight Rashomon-tunable. (Diagnostic first: per-field weight-mass report is
  free and belongs in the explanation artifacts regardless.)
- **(d) Prototype duplication** (K profiles collapsing onto the popular
  signature region; host has no inter-prototype separation and popularity-biased
  negative sampling pushes toward it). *Constraint:* separation penalty
  L_sep = mean_{k≠j} max(0, cos(a_k, a_j) − m) (C2 Eq. 8 analogue, margin m);
  cost O(K²V) once per step — negligible at K ≤ 100. Weight + margin
  Rashomon-tunable.
- **(e) Bottleneck leakage via biases.** Handled **in** the candidate (the one
  place a hard constraint is part of the design, since it preserves rather than
  adds a mechanism): `use_bias = 0` (§3.2). Not a knob.

### 3.6 Baseline gate

- **CF-only ProtoMF** (`user_item_proto`, same budget/protocol) — the host
  baseline, always (R6 bar).
- **`item_proto` (CF)** vs the **minimal variant** (§3.2) — isolates the
  relocation on the single-sided architecture.
- **The one-idea ablation:** `user_item_proto` vs `attr_item_proto` differ in
  *exactly one thing* — the space the item branch lives in (free CF factors vs
  observed attributes). No other ablation is needed to isolate the idea; the
  minimal-variant pair gives the same contrast without the double-tie scaffold.
- **LightFM (B5)** — external feature-aware baseline (natural: also
  cold-capable, also linear in feature embeddings, but *no* prototype layer and
  no intrinsic read-out).
- **Explanation baseline (R2):** the host's post-hoc `top_k_items` explainer +
  `naming/` feature-lift profiles on CF-only ProtoMF — the "post-hoc gap"
  comparison the requirements name as the baseline to move beyond.
- **Cold-item protocol:** stratified tail/cold evaluation à la D1 ProtoCF
  (items with 0 training interactions scoreable by construction — the CF
  baselines need a fallback policy; report both overall and cold-slice metrics).

> **Amendments (2026-07-07, post-5b critique pts. 1–2):**
> - **Baseline #7 — attribute-kNN cold fallback on CF-ProtoMF:** cold items
>   scored via their nearest warm attribute-neighbors' CF representations (mean
>   of top-n by attribute cosine). The strongest cheap competitor to dc02's
>   cold mechanism; its previous absence made the cold comparison too easy.
> - **Full-catalog tie diagnostic:** report signature-class statistics of each
>   model's full-catalog top-10 (how often, and how large, tie blocks appear) —
>   answers the deployment/examiner question the 99-negative protocol does not.

## 4. Cost & feasibility

All estimates Fermi-style; real counts (users/items/V) measured on the local
splits 2026-07-07; d=64, L^u=K=50 taken as mid-range of the host search space.

### 4.1 Parameter delta (vs CF-only `user_item_proto`, hm_3_month: N_u=256,701, M=26,963, V=534)

| block | host | dc02 |
|---|---:|---:|
| user table N_u·d | 16.43M | 16.43M (unchanged) |
| user prototypes L^u·d + proj d·K | 6.4K | 6.4K (unchanged) |
| item table M·d | 1.73M | **0** |
| item prototypes / A | K·d = 3.2K | K·V = 26.7K |
| item projection / W_t | d·L^u = 3.2K | L^u·V = 26.7K |
| biases (N_u+M+1) | 0.28M | **0** (`use_bias=0`) |
| **total** | **≈18.4M** | **≈16.5M (−10%)** |

The item branch shrinks ~97% (1.74M → 53.4K) and — the R5-relevant point — **no
parameter scales with M** anymore. On `hm_full` (M=90,690, V=576) the delta grows:
host item side ≈5.9M, dc02 ≈58K. Parameters grow with K·V, so even K=100
attribute prototypes cost only ~115K — parameter cost is a non-issue.

### 4.2 Training-time expectation

Per-step item-branch MACs (B=256, 1+n_neg=11): host ≈ B·n·d·(K+L^u) ≈ 18M; dc02 ≈
B·n·V·(K+L^u) ≈ 150M dense — a V/d ≈ 8× item-branch multiplier, but ~sub-ms on an
A100/A30 and still small beside the unchanged rest (user branch with its 16M-row
table, sampler, optimizer). Exploiting the exact 9-sparsity of x_i (index-sum of
9 rows) reduces it to ≈ host level; an optimization option, not a requirement.
**Expected wall-clock: ≈1–1.5× host per epoch.** Convergence: fewer free
parameters and no cold rows to fit; E1 reports bottlenecks as data-efficient
(Fig. 2-Right) — plausibly fewer epochs, claimed only as "not slower" (arguable,
flagged as such). X buffer memory: 57 MB (hm_3_month) / 209 MB (hm_full) dense
fp32 — GPU-fine; uint8/sparse fallback trivial.

### 4.3 S5 pipeline-compatibility checklist

- Sampled `(batch, 1+n_neg)` scoring preserved? **YES** — item branch is
  `X[i_idxs]` gather + two small matmuls; `RecSys.forward` untouched.
- Full-catalog pass per step? **NO** — regularizers stay batch-based (host code);
  the optional L_push knob samples a subset, never the catalog.
- Item/user branch cheap indexed computation? **YES** — buffer gather ≈ an
  embedding lookup; user branch unchanged.
- Losses fit `get_and_reset_loss()`? **YES** — host accumulator reused verbatim;
  proposed knobs (§3.5) hook the same accumulator.
- Expensive per-item precompute? **NONE** — X built once from
  `item_features.csv` at model build (seconds), non-persistent buffer.

### 4.4 Eval-protocol note on signature ties (measured)

Although 68.1% of items share their attribute signature (§3.1), duplicate classes
are small relative to the catalog: under the 99-uniform-negative protocol the
expected number of same-signature negatives per test positive is **0.035** (~3.4%
of test cases see ≥1 exact tie; mean item's class size 10.5, median 3). So exact
ties barely touch HR@10/NDCG@10 mechanics; the real R6 exposure is
*representational* — no within-class differentiation (e.g. popularity within a
signature class cannot be expressed, with `use_bias=0` not even as a bias). Taken
up in Step 5.

### 4.5 Budget gate

Same order as host `user_item_proto` hyperopt on the same split (≈1–1.5× per
step, same trial protocol) — comfortably inside the 10-day hard and ~1-day dev
budgets on LEO5. Dev iterations run on the CPU laptop (dense 150M MACs/step is
laptop-tolerable at small batch counts). **No budget flags.**

## 4b. Math sanity check

**Claims spec:** `Master/temp/dc_checks/dc02/claims_spec.md` (mechanism
definition + 11 claims, no derivations). Claims, in brief: (1) forward shapes at
(B, 1+n_neg); (2) cold-item activation finite, non-constant, exactly equal to a
same-signature warm item after a toy training run that never samples it;
(3) exact per-attribute decomposition t*_k − 1 = Σ_{v∈x_i} A_{k,v}/(√F‖A_k‖);
(4) exact per-prototype score decomposition (L^u + K addends, no hidden term);
(5) gradient reaches U, P, W_u, A, W_t and not X; (6) same-signature items tie
exactly for every user, and a per-item bias destroys the tie; (7) inherited
'max' regularizers decrease in the intended directions; (8–11) mechanics of the
four unstacked constraint candidates (softplus nonnegativity + its [1,2]
range consequence, field-crispness entropy, separation margin, data-pull
zero-iff-realizable).

**Viability self-assessment.** All 11 claims are meaningful at toy scale (shapes,
exact identities, gradient flow, degenerate/limit cases, directional derivatives).
**Skipped as untestable-at-toy-scale, with justification:** R6 accuracy parity
(materializes only through full training dynamics at H&M scale), emergence of
single-field collapse (training dynamics; only the *countermeasure's* mechanics,
claim 9, is checkable), convergence-speed expectations (§4.2), and explanation
*quality* (human judgment, not math). None of these silently dropped — they are
Step 5 material.

**Black-box execution:** subagent received ONLY the claims spec; script
`Master/temp/dc_checks/dc02/check_claims.py` (torch 1.9.1 CPU, ~3 s; toy dims
N_u=30, M=40, F=3, V={4,5,6}, d=8, L_u=4, K=5, B=8, n=4).

**Subagent report (verbatim):**

> 1. **Shapes — CONFIRMED.** score=(8,4)=(B,n); u*=(8,4), û=(8,5), x=(8,4,15), t*=(8,4,5), t̂=(8,4,4) — all exactly as annotated.
> 2. **Cold-item non-degeneracy — CONFIRMED.** 200-step Adam run (BCE + 0.1(R_batch+R_proto), loss 0.750→0.249) with item M−1 never sampled; its t* is finite, non-constant (std across k = 0.285), and bitwise equal (`torch.equal`) to the warm item sharing its X row.
> 3. **Per-attribute decomposition — CONFIRMED.** max |(t*−1) − Σ_v A[k,v]/(√F‖A_k‖)| = 5.96e-08 over all M×K pairs (‖x_i‖=√F since exactly F ones).
> 4. **Score decomposition — CONFIRMED.** Sum of the 4+5 per-prototype addends reproduces score with max error 0.00e+00 (bitwise, no hidden interaction term).
> 5. **Gradient flow — CONFIRMED.** One backward from Σscore² gives nonzero grad norms on all five params (U=1083.5, P=153.8, W_u=1429.9, A=199.3, W_t=366.8); X has `requires_grad=False`, `X.grad=None`. (Literal reading: X as-implemented-constant receives no grad; mathematically ∂score/∂X≠0 if X were a parameter.)
> 6. **Same-signature ties — CONFIRMED.** Items 0 and 39 (identical X rows) score exactly equal for all 30 users (max gap 0.0); adding a random per-item bias b_i yields max gap 1.848 — tie destroyed.
> 7. **Regularizer direction — CONFIRMED.** (a) A[k*] += 0.05·x_r for row r's argmax prototype: R_batch −1.31875→−1.32231, autograd dR/dα=−0.0723<0. (b) worst-claimed prototype rotated toward its best-claiming item's x: R_proto −1.48121→−1.48459, dR/dα=−0.0681<0. Robust across 50 random configs (0 strict-decrease failures for both). Ambiguity noted: "others held fixed" can't literally hold when editing A_k (other rows' entries in column k also move), but the decrease held generically anyway.
> 8. **C1 mechanics — CONFIRMED.** softplus(Ã) > 0 in all K×V entries; grad reaches Ã (norm 3.16); t* range [1.165, 1.750] ⊂ [1,2] — repulsion half [0,1) unreachable.
> 9. **C2 mechanics — CONFIRMED.** One-dominant-coordinate blocks: L_crisp = 2.1e-20 ≈ 0; uniform blocks: L_crisp = 1.59583 = mean ln(V_f) (the entropy maximum); random A gives 1.20 < uniform; grad(A) norm 0.090.
> 10. **C3 mechanics — CONFIRMED.** Orthogonal prototypes (cos=0 ≤ m=0.3): L_sep = 0.0 exactly; near-duplicate pair: L_sep = 0.084 > 0; grad(A) norm 0.048.
> 11. **C4 mechanics — CONFIRMED.** Prototypes set to positive multiples of x_i ∈ S_items: L_push = 4.8e-08 ≈ 0; one perturbed prototype: 0.097 > 0; negated multiple: 0.200 > 0; grad(A) norm 0.100. ("Iff" holds by Cauchy–Schwarz: cos=1 forces a positive scalar multiple.)
>
> No claim was untestable at toy scale; all 11 confirmed.

**Outcome:** no falsifications; no known defects carried into Step 5 from this
gate. Note claim 6 doubles as *verification of the structural tie limitation*
(§3.1/§4.4) — confirmed, and carried into Step 5 as a design property, not a
defect discovered here.

## 5. Requirements evaluation

Requirements doc re-read this step (version of 2026-06-11, unchanged since
Step 0). Interpretable-AI framework placement (protocol 5.4b) skipped — the
framework is still undecided per `Master/docs/notes/ideas.md` §4.

### 5.1 Ratings + prose

- **R1 Tied integration — strong.** The attribute representation is not merely
  *tied into* the score path — it **is** the item's entire representation; both
  UI-score halves (u*ᵀt̂ and ûᵀt*) consume the same x_i, mirroring how the host
  double-tie feeds one item embedding into both halves. No reranker, no parallel
  model, no post-hoc bridge. One honest nuance: the host realizes its tie by
  *weight sharing* between two learned tables; dc02's item-side tie is by
  *construction* (one input, no tables to share) — the spirit (one
  representation drives everything) is satisfied maximally, the letter (shared
  learned weights) is moot on the item side and unchanged on the user side.
- **R2 Intrinsic explanations — strong.** Every explanation quantity is either
  computed in the forward pass (u*, û, t*, t̂, the L^u+K exact score addends —
  4b claims 3–4, bitwise) or is a model parameter directly readable (A rows,
  W_t rows). Nothing is reconstructed by a separate procedure; the `top_k_items`
  baseline becomes a *comparison*, not a dependency. E1 §5/Table 2 supplies
  literature evidence that this intrinsic access is more faithful than post-hoc
  probing of an unconstrained model.
- **R3 Prototype paradigm — strong.** The explanation remains "which prototypes
  does this user/item activate, and what do those prototypes mean" — same shape
  as the host, same per-prototype contribution readout (A1 §5.2), now with
  meaning carried by attribute profiles instead of item neighborhoods.
- **R4 Feature-grounded prototypes — strong (by construction; crispness is the
  residual risk).** The prototype–feature association could not be more
  explicit: a prototype has no existence *outside* feature space. Attribution of
  an item's activation to its features is exact and additive (4b claim 3), not
  an approximation. The residual risk is not *whether* profiles are feature
  readable but whether they are *crisp* (modes (c)/(d), §3.5) — flagged, with
  designed unstacked knobs and diagnostics.
- **R5 Sparsity robustness — strong (item-side; user-side out of scope).** A
  zero-interaction item gets a full-strength, non-degenerate representation
  identical to its warm signature-mates (4b claim 2 — bitwise equality), with no
  fallback policy, no imputation, no special path: cold items are
  architecturally indistinguishable from warm ones. This is the strongest
  possible R5 answer on the item side. Cold *users* remain unsolved (user branch
  stays CF) — deliberately, per the stage-1 item-only scoping (scratchpad
  sequencing note); S2 below.
- **R6 Dense-regime accuracy — weak / the candidate's open risk.** Three
  quantified strains: (i) items collapse to 12,576 signature classes (of 26,963;
  68% share a signature) — within-class preference differences are
  *inexpressible*; (ii) with `use_bias=0`, item **popularity** is expressible
  only at class granularity — and popularity is a dominant implicit-feedback
  signal; (iii) the strongest single H&M signal (`product_code`, lift 35.9) is
  deliberately absent (identity-like, 4.0 analysis). Literature cuts both ways:
  E1 found bottlenecks *better* on OAI and slightly worse on CUB (Table 1); our
  attributes are exact (σ_C = 0 in E1's §8 analysis — the favorable regime), but
  k=V=534 vs d... E1's k≪d intuition does not directly transfer. Honest
  expectation: **competitive-to-worse on the dense slice, clearly better on the
  cold slice** — the fidelity-vs-accuracy trade-off (§4 tensions) made
  measurable, which is itself thesis material. Mitigations exist but each
  compromises the idea: re-enable item bias (small side channel, breaks exact
  ties and E1's intervention property), add price-band/product_code features
  (V grows, identity-like features re-import CF-ness), or add a free ID feature
  block — the last converts the design into feature-composed-factors territory
  (dc01's fingerprint region) and surrenders both the bottleneck and R5;
  documented as a boundary, not proposed.
- **R7 User-perceivable vocabulary — strong, with the taxonomy caveat.**
  Explanations are rendered *natively* in attribute vocabulary — colour,
  pattern, garment kind — the exact vocabulary R7 names; the image itself (S3)
  aside, nothing more user-visible is available in the data. The caveat: the
  model will lean on the strongest signals (department 5.2×, section 3.3×) —
  retailer taxonomy that is only semi-legible ("Denim Trousers" fine, "Jersey
  Fancy" less so); profiles will name those fields prominently. That is R7's
  sanctioned balance (accuracy-bearing features may drive, so long as the user
  gets insight in understandable terms) — and the per-field decomposition means
  the legible fields are always *also* stated. The §4-suggested stronger split
  (internal features drive the score, read-out constrained to perceivable
  vocabulary only) is NOT what dc02 does — recorded as a future variant.
- **R8 Parsimony — strong.** One relocation ("the item prototype space moves
  into attribute space"), everything else host-verbatim. All four constraint
  candidates remain unstacked (§3.5); `use_bias=0` is a removal, not an
  addition. Same conceptual size as `user_item_proto`.
- **S1 Decoupled variant — n-a (not this candidate's job), well-served as
  comparison.** dc02 is the maximally-coupled end of the spectrum; the S1
  contrast slot stays open for a future loosely-coupled candidate. The post-hoc
  explainer on CF-only ProtoMF (§3.6) provides the degenerate-end comparison
  already.
- **S2 User-side symmetry — partial.** Nothing precludes the mirror move
  (customer attributes → user prototypes in user-attribute space); the user
  branch is untouched host, so the extension point survives intact. H&M's thin
  customer features (age, club status) would cap its value — deferred exactly as
  the requirements defer it.
- **S3 Image features — partial.** The item interface is "any fixed vector per
  item": an image-embedding block could be appended to x_i. Two caveats: (i) a
  continuous block breaks the uniform-norm property behind the *exact*
  per-attribute decomposition (‖x_i‖ no longer constant — decomposition remains
  computable, no longer as clean); (ii) prototype coordinates over image
  dimensions are not human-readable (would need prototype-image visualisation —
  the C1 push machinery). Compatible but non-trivial; honest rating: partial.
- **S4 LLM naming — strong.** A prototype's ranked per-field weight profile is
  exactly the structured input an LLM verbaliser wants; slots into `naming/`
  unchanged. The LLM names, never scores — unaffected by dc02.
- **S5 Pipeline compatibility — strong.** §4.3: all checklist items green;
  1–1.5× step cost; no full-catalog passes; losses ride `get_and_reset_loss()`.

### 5.2 §4 design tensions

- **Where features enter:** the third listed insertion point ("prototypes living
  *in feature space* rather than CF space") taken literally — the only candidate
  region that realizes it (dc01 took the first).
- **Fidelity vs accuracy:** dc02 deliberately occupies the maximum-fidelity
  endpoint; the cost is quantified in advance (signature classes, popularity
  granularity, §4.4) and measurable against CF-only ProtoMF — the tension turned
  into an experiment rather than an assumption.
- **Attribution granularity:** per-attribute-*value*, exact, additive — the
  finest granularity available without learned concepts; no concept-learning
  leakage risk (concepts are given, E3's leakage concern does not arise).
- **R7 perceptual alignment vs predictive strength:** dc02 resolves it by using
  one vocabulary for both roles and letting the per-field read-out keep the
  legible fields visible; the alternative split-design is future work (see R7).
- **Identifiability:** inherited inclusion criteria transplant with semantics
  intact (4b claim 7); duplication/crispness knobs designed and unstacked;
  **Rashomon-tunable knobs:** `sim_proto_weight`, `sim_batch_weight` (inherited),
  L_crisp weight, L_sep weight+margin, L_push weight/period (proposed). The
  nonnegativity reparameterization is a binary mechanism decision, not a tunable.

### 5.3 Part-prototype pitfalls checklist (S5, modality-independent items)

- *Prototype number trade-off (S5 §3.1.1):* present as in host; K searchable;
  fewer = readable, more = coverage. Unchanged exposure.
- *Prototypes lack semantics (S5 §4.2):* **directly answered** — this is the
  pitfall dc02 exists to remove; prototypes are semantic by construction. S5
  §4.4 itself names "semantically annotated prototypes via CBM-style semantic
  features" as the open direction dc02 instantiates for recsys.
- *Similarity-perception gap (S5 §3.1.2):* narrowed — similarity is computed
  over human-named dimensions and decomposes onto them exactly; the residual gap
  is the cosine aggregation itself (why these 9 shares sum as they do is
  arithmetic, but weighting by ‖a_k‖ is not intuitive). Reduced, not gone.
- *Cosine unreliability (S5 §4.4, Steck):* softened but present — one side of
  every item cosine is a fixed binary vector with constant norm, removing the
  free-embedding-norm pathologies on that side; prototype norms ‖a_k‖ remain
  learned. Flag for the explanation-quality eval.
- *Prototype collapse/duplication:* mode (d); host-level exposure, L_sep
  designed (unstacked).
- *Semantic gap / undepictable prototypes:* mode (b) off-support profiles;
  R_proto partially covers, L_push/push designed (unstacked).
- *Bottleneck information loss (S5 §3.3.2):* the R6 discussion — quantified
  (12,576 classes), not hidden. **Note an internal counter-position:** the S5
  reading summary (2026-06-16) concluded "borrow the named-concept principle,
  *not* the bottleneck architecture — structurally unsuited to recommendation's
  dyadic target." dc02 knowingly deviates: the bottleneck is **one-sided** —
  the dyadic/collaborative half lives entirely in the untouched CF user branch
  (user embeddings, user prototypes), and only the item side — where attributes
  are total, exact, and observed — is bottlenecked. Whether that placement
  rescues the bottleneck for recsys is exactly what the R6 experiment decides.
  Recorded as the candidate's central open question, not silently overridden.
- *Spatial/part-specific pitfalls (misalignment, patch visualisation):* n-a —
  ProtoMF-family models are whole-object prototype-similarity (S5 summary
  positioning note).

### 5.4 Degeneration-protection gate (protocol v1.1)

| # | degeneration mode (§3.5) | status |
|---|---|---|
| (a) | negative-weight (anti-)profiles | protected by proposed, unstacked C1 (softplus nonneg); unprotected in base — open, low-severity (profiles stay readable, just signed) |
| (b) | off-support attribute profiles | partially protected by inherited R_proto (pulls prototypes toward realized batch items); fully by proposed, unstacked C4 (L_push / hard push) |
| (c) | single-field collapse (taxonomy-only prototypes) | unprotected by inherited regularizers — proposed, unstacked C2 (L_crisp); free diagnostic (per-field mass report) regardless |
| (d) | prototype duplication | unprotected by inherited regularizers (host shares this gap) — proposed, unstacked C3 (L_sep) |
| (e) | bottleneck leakage via per-item bias | protected **in-design** (`use_bias=0`, §3.2); verified 4b claim 6 |

No mode unlisted; statuses flag, none vetoes.

### 5.5 Step-4b fold-in

No falsifications — all 11 claims confirmed (first run, no re-spec). Claim 6
doubles as confirmation of the tie *limitation* (R6 discussion above): it is a
verified design property, weighed, not an undiscovered defect.

## 5b. Devil's advocate

Adversarial subagent received ONLY this candidate file. Critique **verbatim**:

> **1. The model cannot rank items — it ranks equivalence classes, and §4.4 hides this behind a weak evaluation protocol.** By the file's own measurement (§3.1), 68.1% of items share their full attribute signature (77.4% on `hm_full`, classes up to 304 items). Same-signature items score *bitwise identically for every user* — verified as claim 6. The §4.4 defense ("expected 0.035 same-signature negatives, ~3.4% of test cases") is not a property of the model; it is a property of the 99-sampled-negatives protocol. Under full-catalog ranking — what a thesis examiner or any deployment will ask about — a top-10 can be flooded by a 146-item tie block sorted arbitrarily. The file uses the leniency of its own evaluation to launder a structural defect. Worse, with `use_bias=0`, **popularity** — the file admits it is "a dominant implicit-feedback signal" — is inexpressible below class granularity. In fashion retail, where the same signature covers a bestseller and a dud, this is fatal to R6, and the file knows it: its honest expectation is "competitive-to-worse on the dense slice."
>
> **2. If the dense slice loses, the thesis contribution collapses to cold-start — which LightFM, dc02's own baseline, already delivers.** The design deliberately drops `product_code` (lift 35.9, the strongest single signal, vs. department's 5.2×). Having amputated its best feature and its bias term, the candidate pre-registers a likely accuracy loss and pivots the payoff to R5. But the cold-start mechanism is exactly "a cold item lands where its warm signature-mates are" — class-level popularity transfer. A trivial nearest-attribute-neighbor fallback on CF-ProtoMF would plausibly match it, and §3.6 conspicuously omits that baseline while giving the CF models only an unspecified "fallback policy" — a comparison designed to be lost.
>
> **3. "Grounding by construction" covers roughly 0.3% of the model.** Per §4.1, 16.43M of ≈16.5M parameters are the untouched CF user table and user prototypes. Every rendered explanation (§3.4) still leans on "your 'Basics buyer' user prototype P3" — a free CF vector whose meaning rests on precisely the unenforced post-hoc grounding dc02's headline claims to transcend. The candidate solves enforced grounding on the small, easy, observed side and declares victory over R4 while the dyadic heart of the recommender remains a black box.
>
> **4. The "exact" decomposition is exact arithmetic on a distorted quantity.** The shifted cosine puts t*_k ∈ [0,2] with 1 at orthogonality. The score addend û_k·t*_k therefore contains a constant û_k·1 that flows to the score *even when the item has zero similarity to prototype k*. The celebrated per-attribute shares (claim 3) sum only to t*_k − 1; the rendered example ("activates P7 at 1.72, of which Denim +0.31…") attributes 0.72 of the activation and silently pockets the baseline 1.0 — the majority of the number shown to the user. Add claim 8's finding that under the proposed nonnegativity constraint t* ⊂ [1,2] — prototypes can never repel — and every explanation becomes "everything weakly matches everything," with §5.3 itself conceding the ‖a_k‖ weighting "is not intuitive." The R2/R7 quality claims rest on faithfulness-to-a-misleading-decomposition.
>
> **5. The CBM pedigree is decorative, and the file admits its theory doesn't apply.** With g = identity (§2.1 "setting note"), the entire concept-prediction problem — the substance of Koh et al. — vanishes. E1's own effectiveness condition is k ≪ d; here k = V = 534 ≫ d = 64, and the file concedes the "intuition does not directly transfer." Strip the framing and the item branch is two linear maps (K×V, L^u×V) over a 9-hot vector plus cosine normalization — a linear content model with 53K parameters. The novelty claim survives a four-query prior-art probe (§2b) whose scope was drawn narrowly enough (bottleneck property + prototype read-out + this exact scorer) that nothing *could* match; "unpublished" at that specificity is cheap.
>
> **6. The R7 story is retailer org-chart dressed as user vocabulary.** The file predicts prototypes will lean on `department` (221 values, "Jersey Fancy") and `section` — semi-legible internal taxonomy — and admits degeneration mode (c), single-field collapse onto exactly these fields, is **unprotected in the base design**. Three of five degeneration modes ship unprotected (§5.4), with all four fixes "unstacked."
>
> **7. R8 parsimony is an accounting trick.** "One idea" holds only if the base model works. The file's own gate says it likely won't (modes a–d), whereupon the real candidate is host + softplus reparam + L_crisp + L_sep + L_push with weights, margins, and periods — a five-loss tuning swamp, tuned against… nothing: §3.6 defines no quantitative explanation-quality metric, and §4b defers explanation quality to "human judgment."
>
> **8. The sanity gate is validation theater, and the design overrides its own literature synthesis.** All 11 confirmed claims are shape checks and algebraic identities; every load-bearing risk (R6 parity, field collapse, convergence) was "skipped as untestable." Claim 2 "confirms" cold-start by demonstrating the tie pathology. Meanwhile §5.3 records that the author's own S5 summary concluded the bottleneck architecture is "structurally unsuited to recommendation's dyadic target" — dc02 proceeds anyway, betting the candidate on an R6 experiment it has already predicted it will lose.

### Response (point by point)

**1 — Concede the substance; rebut the "laundering" charge; OPEN RISK (the
candidate's central one).** The class-granularity ceiling and the popularity
point are real and were rated R6-*weak* before the critique. Two corrections to
the framing: (i) the 99-negative protocol is not chosen leniency — it is the
host paper's protocol and the thesis's replication-faithful benchmark, used for
*all* models compared; (ii) the tie-block-flooding scenario applies to
full-catalog top-N serving, which the thesis does not claim — but the examiner
question is fair, so the full-catalog tie behavior gets a dedicated diagnostic
(size-weighted class statistics of the top-10; cheap, added to the eval plan).
The bias knob exists as documented mitigation with its documented cost. This
risk is the R6 experiment's job to price — the spectrum rule, not a veto.

**2 — Concede the missing baseline (visible amendment below); rebut the
"collapses to cold-start" reduction.** An attribute-kNN fallback on CF-ProtoMF
(cold item scored via its nearest warm attribute-neighbors) is now baseline #7
in §3.6 — the critique is right that its absence made the cold comparison too
easy. The reduction is rebutted on scope: dc02's headline is not cold-start
accuracy, it is **intrinsic feature-grounded prototype explanations** (R2/R3/R4)
*with* cold capability; LightFM and kNN-fallbacks deliver neither prototype
structure nor an intrinsic read-out. If the dense slice loses badly AND the
cold slice only matches LightFM, what remains is the explanation pillar — that
is the honest shape of the bet, and it is the user's ranking decision.

**3 — Partially rebut; the residue is real and inherited.** Parameter count is
the wrong denominator for "how much is grounded": the *explanation vehicle* is
the prototype layer, and every quantity in the rendered explanation now has an
attribute-space reading — including the user side, because each user prototype l
carries the learned attribute-response profile W_t[·,l] ∈ R^V (§3.4 pt. 4): the
"'Basics buyer'" label is renderable from W_t, not only from post-hoc top-k. The
honest residue: that user-side grounding is *learned and score-mediated*
(geometry-grounded, dc01-style unenforced), not by-construction — the by-
construction guarantee covers the item half only. Conceded and now stated
plainly; full symmetry is exactly the S2 extension.

**4 — Concede; fixed by visible amendment (and the fix strengthens the
read-out).** The critique's sharpest point. The shifted cosine's +1 injects a
user-constant Σ_k û_k into every item's score — *constant across items for a
given user*, hence ranking-irrelevant, but misleading if rendered as part of an
item's "activation contribution." Amendment to §3.4: explanations render the
**item-discriminating** quantities û_k·(t*_k − 1) (which the per-attribute
shares decompose *exactly and completely*) with the user-constant baseline
disclosed once, not pocketed. Same treatment for the u*ᵀt̂ half (no constant
arises there). The C1 range-halving ([1,2], no repulsion) stands as the
documented cost of that unstacked knob — one more reason it is a user decision.

**5 — Partially concede, without damage.** Yes: mechanically the item branch is
a small linear-content model under the host's cosine-prototype layer — that
*simplicity is the design* (Rudin axis; scratchpad "two axes" gut-check), not a
concealed weakness; and the k≪d caveat was self-reported. On novelty: the probe
scope follows the protocol (verify the *adaptation*); the broader wedge rests on
the §I sweep (31 citers, CF-only) and S7's flagged gap, not on the four queries
alone. Conceded: the novelty is a *combination* novelty — modest by
construction, honest about it.

**6 — Concede the risk; reject the conclusion that unprotected = unaddressed.**
Mode (c) is unprotected *in the base* by protocol design (R8: constraints are
designed, costed, verified — 4b claims 9–11 — and left as the user's stacking
decision). The free per-field-mass diagnostic ships regardless, so collapse is
*detectable* from day one even if initially unprotected. R7 was rated with
exactly this caveat.

**7 — Rebut the swamp; concede the metric gap.** The five-loss scenario
contradicts the protocol's own discipline: knobs are added one at a time, by the
user, in response to *observed* degeneration — "host + all four" is nobody's
plan. The real hit is the missing quantitative explanation-quality metric: true,
and it is a requirements-level open question (§6 of the requirements doc;
ideas.md §5 already sketches candidates — dead-prototype share, name-lift
sharpness, Rashomon-set tuning). OPEN — owned by the thesis eval design, needed
by *every* candidate including the host baseline, not a dc02-specific defect.

**8 — Rebut "theater"; the S5-override stands as the candidate's declared bet.**
The 4b gate is scoped by protocol to cheap falsifiables; every load-bearing
skip is written in §4b with its reason — the opposite of theater. Claim 2
confirms cold-capability *and* the tie property because they are the same
mechanism seen from two sides; both are reported. On the S5 summary: the
one-sided placement (bottleneck only where attributes are total, exact, and
observed; the dyadic/collaborative half untouched CF) is a *reasoned* deviation
from that note, recorded in §5.3 — and "competitive-to-worse" is a risk
statement, not a prediction of loss: E1's evidence runs both directions and the
σ_C = 0 regime is the favorable one. The R6 experiment prices the bet; that is
what candidates are for.

### Visible amendments made in response (2026-07-07)

1. **§3.4 (read-out):** contributions rendered as û_k·(t*_k−1) — the
   item-discriminating part, exactly decomposed by the per-attribute shares —
   with the user-constant Σ_k û_k disclosed separately. (Critique pt. 4.)
2. **§3.6 (baselines):** added attribute-kNN cold fallback on CF-ProtoMF as an
   explicit baseline. (Critique pt. 2.)
3. **§3.6 (eval):** added full-catalog tie-block diagnostic (top-10
   signature-class statistics). (Critique pt. 1.)

## 6. Fingerprint

- **Features-entry:** prototypes live in attribute space — the item branch is a
  concept bottleneck over the observed attribute multi-hot x_i (no per-item
  parameters, no free ID channel; features do not *feed* the representation,
  they *are* it).
- **Grounding:** **by construction** — an item prototype is literally a weight
  vector over attribute values (a_k ∈ R^V); user prototypes acquire learned
  attribute-response profiles via W_t (score-mediated, unenforced on that side).
- **Read-out:** per-prototype item-discriminating contributions û_k·(t*_k−1)
  and u*_l·t̂_l, with exact additive per-attribute-value shares of every
  activation; prototype profiles read directly off the parameters A / W_t.

Distinct from dc01 on the entry axis (attribute space vs features→CF-factors)
and the grounding axis (by construction vs unenforced shared geometry).
