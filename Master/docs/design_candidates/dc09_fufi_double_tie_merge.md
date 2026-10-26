# dc09 — fUfI-ProtoMF: feature-composed user AND item factors under the UI-ProtoMF double tie

> Design Candidate Protocol v1.3 cycle, started 2026-10-26. **Directed cycle** for the
> C1 lineage's merge stage (vault `2026-07-11_2351` step 6 — "merge the two, following
> the UI-ProtoMF paper's own structure"; masterplan row 5; the 2026-10-26 wave-1
> submission record already names this cycle "dc09"). Skill argument: `fUfI`.
> **Status: reviewed — user decisions recorded 2026-10-26 (§6.2); next: `/scrutiny dc09`.**
> Cycle complete through Step 6 on 2026-10-26; 4b and 5b absorbed as dated amendments.
> Working location: written in the `feat/fufi` worktree during the cycle, then **moved into the
> main checkout on `feat/scrutiny` on 2026-10-26 at the user's decision** ("everything runs in
> the working folder"); the worktree copy was cleared. All inputs were read at `feat/scrutiny`
> @ 86f7a77 (identical content in both checkouts at the time). Cluster checkout lock (no `git pull` into `~/UIFProtoMF/ProtoMF` while
> wave-1 jobs 8153730–8153744 are pending) is untouched by this docs-only cycle.

---

## 0. Frame

### 0.1 Cycle setup — directed, and the rules it runs under

- **Host = UI-ProtoMF** (`user_item_proto`, factory path `prototypes_double_tie`, A1 §3.3).
  **Stage bar = fUfI vs UI-ProtoMF like-for-like**, and — unlike every earlier stage —
  this bar *is* the lineage's load-bearing R6 comparison (requirements R6 staging note,
  2026-08-20: "the paper-headline bar … is assessed at the merge stage"; S0 charter
  amendment 2026-08-20: ml-1m is first-class here, "not H&M-first-then-port").
- **No fUfI brief exists** (dc05 ran under `fu_cycle_brief.md`; nothing analogous was
  written for the merge). *I propose* to run under dc05's M1–M7 **by precedent**,
  restated for the merge below; each line is a proposal the user can strike at review,
  not a user-authorized deviation:
  - **M1 — directed.** Distinctness from **dc01 and dc05 is waived**: the merge is, by the
    lineage's own design, their composition under the host's double tie. Distinctness
    within this cycle's shortlist binds. Step 0's question is re-scoped to *which merge
    mechanism best serves the lineage*, not *which fingerprint region is empty*.
  - **M2 — targeted literature.** Seeds come first from the lineage (dc01's original
    UI-hosted build is the declared "skeleton of the merge stage", dc01 banner), the
    scratchpad's merge entries, and the vault's merge inventory (0.5). GR6 (no
    from-memory citations), GR8 (paper-says vs I-propose) and Step 2b (novelty) bind
    unchanged.
  - **M3 — hidden-effects pre-audit (Step 3b) mandatory.** The vault has already
    pre-registered most of the merge's inventory (0.5); 3b confirms, extends, and
    names instruments. Keystone-claim floor for 4b: **fUfI(F=0, +ID on both sides) ≡
    `user_item_proto`**, bit-identity as the eventual build target; plus the two
    single-side reductions (item side composed only ≡ dc01's original UI-hosted build;
    user side composed only).
  - **M4 — quarry license.** dc02/dc03/dc04 as before, plus dc01's superseded UI-hosted
    text (§3.2/§3.3 Concatenate wrappers, `FeatureEmbeddingW` — class kept in code for
    exactly this stage, modifications log) and dc08's coverage-split read-out.
  - **M5 — renderer continuity.** The shared breakdown renderer's UI slot (sc05
    comparability matrix rows 11/12: "deferred to the fUfI merge stage") and the
    **parked read-out 4** (F-DC01-07, exact linear-half attribution — "returns at
    fU/fUfI") are owed by this cycle's Step 3.
  - **M6 — merge-specific mandatory confrontations** (Step 3 and Step 5; silence = gap):
    1. **Which tables are tied.** One item-word table read by both sides, or two
       (fI trains "Dark Blue" as an *item property*, fU as a *taste indicator*)? Today
       no table is read by both sides (vault `2026-09-08_1705`); sharing would be the
       first cross-side tie and the strongest literal reading of R1's exemplar.
    2. **The two linear halves** — what û = W^u q_u and t̂ = W^t q_i read, what names
       their coordinates (the *other* side's prototypes), how they are attributed
       exactly (read-out 4), and their Steck exposure (dot-trained, no cosine shield).
    3. **The live intercept** 1ᵀt̂ (per-item, rank-active) now decomposes over feature
       words — LightFM's composed bias re-emerging implicitly (vault `2026-07-12_1815`).
    4. **Norm entanglement / two-sided refund** (vault `2026-08-15_1721`): the reopened
       length channel can refund the angular tax *or* weld class-level popularity into
       the directions; the lineage's built-in 2×2 (host U/I · host UI · fI/fU · fUfI)
       and the user's **length-ablation control** (scratchpad, 2026-07-12) are run-plan
       material this cycle inherits.
    5. **R1's weight-sharing exemplar finally has its object** — rate it literally, not
       by the operative-core proxy used on single-branch hosts.
    6. Per-stage R/S reading in fUfI form (R5 now two-sided: cold item × cold user ×
       cold-cold; the 3-scenario testbed entry remains the unmet prerequisite).
    7. **`ft_type` naming.** CLAUDE.md's reserved `dual_item_proto` describes a *different*
       idea ("dual CF + feature prototype spaces"); the merge needs its own name
       (working proposal `feature_user_item_proto`), decided in Step 3.
  - **M7 — trickle-back** of anything touching dc01/dc05 dossiers, S0 instruments, or
    the chapters in progress (Ch 5 fI, Ch 6 fU) — filed when it surfaces.
- **Scope facts inherited from the 2026-10-26 plan** (not re-decided here): build in a
  fixed window after this cycle, on `feat/fufi`; every reported number 100-trial,
  seed 38210573; host rows available/queued: `user_item_proto` × ml-1m n100 (May 21,
  accepted as host row) and × hm_1_month (wave-1 job 8153730, queued); the `hm_kcore`
  build/scope decision sits at "the fUfI base-settings gate" (masterplan) — a **user
  decision this cycle must surface in Step 5/6, not take**.

### 0.2 Occupied fingerprint space (`candidate_index.md`, all rows read)

| id | features-entry | grounding | read-out | host |
|----|----------------|-----------|----------|------|
| dc01 fI | item words → item factor (sum replaces item-ID table) | unenforced shared geometry cos(e_f, p^t_k) | u_k·t*_k + exact per-feature shares | I |
| dc06 fI′ | same, per-field-mean-centred | same, centred gauge | dc01 + within-field contrasts | I (ml-1m only) |
| dc02 | item branch = attribute bottleneck, no per-item params | by construction | per-prototype + per-attribute shares | UI (lukewarm) |
| dc03 | frozen image content, no ID table | exemplar push | case-based sheet | UI (abandoned) |
| dc04 | prototype parameterization P = softmax(A)·C | data-anchored | per-anchor shares | UI (abandoned) |
| dc05 fU | user factor = mean of basket words (+ user-ID row) | USER prototypes grounded in ITEM vocabulary, unenforced | t_l·u*_l over disclosed B(t) + per-word AND per-purchase shares | U |
| dc07 | unchanged | unchanged | softmax/sigmoid memberships — stopped, controlled negative | U/I |
| dc08 | identity-as-ingredient: item history rows y_j in the fU bag | dc05's, partly forfeited | dc05's + named/collaborative/personal coverage split | U |

dc02–dc04 were *drafted* on the UI host but never engaged the tie (their mechanisms
live in the item branch); dc01 was **redirected off the double tie** on 2026-07-12
precisely so the merge would get its own cycle. **No registered candidate has ever
designed, pre-audited, or scored the two-branch tie itself.**

### 0.3 The open niche — the tie, and what only exists there

Three fingerprint-level objects exist on no single-branch host:

1. **Entry:** a composed representation read in **two roles** — cosine against its own
   side's prototypes *and* linear projection into the other side's prototype
   coordinates. Composition therefore feeds the *linear* halves too: for the first
   time the lineage's ingredient-row **norms** are read by the score (the angular tax
   of fI/fU is stage-local; vault `2026-07-12_1224`).
2. **Grounding:** **both** prototype sets grounded in the one item vocabulary *(index-level under Seed A — two learned geometries; the one-geometry story lives in the `_shared` arm; A8, 5b-11)*, with the
   one-table-or-two question open (the [captured dc05] fI-vs-fU word-geometry
   comparison was designed as this cycle's evidence; the 100-trial fI and fU
   checkpoints to run it on exist on LEO5 for ml-1m, hm pending wave 1).
3. **Read-out:** the lineage's four-component inventory (t* grounded · û · t̂ · u*) can
   become **fully nameable** — the linear halves are exact linear functionals of
   composed vectors (read-out 4 returns), the live intercept becomes zoomable per
   word, and the user side's per-purchase regrouping (dc05) meets the item side's
   per-feature shares (dc01) inside one score.

**The null candidate to beat:** the straight merge — dc01's item composition
q_i = Σ_f e_f (+ e_ID(i)) and dc05's user composition q_u = mean-basket words
(+ e_ID(u)), each feeding its own side's unchanged prototype layer *and* its own
side's `EmbeddingW` projection, score u*·t̂ + û·t* exactly as the host (A1 Eq. 12),
tables separate. The cycle's real job, mirroring dc05's brief: does the straight merge
survive the two-branch physics (M6.1–M6.4), or does the tie want a different mechanism
— e.g. the scratchpad's route (b) "symmetric-by-design at the tie level", or the
[captured dc08] cross-side table tie?

### 0.4 Requirements the lineage currently serves poorly — this cycle's targets

- **R1 (exemplar).** Never assessed: on fI/fU hosts only the operative core could be
  scored (R1 clarification note, 2026-07-12). Here the exemplar is literal.
- **R6 (paper-headline bar).** Deferred at every stage by design. Measured stage bets so
  far (memory 2026-10-23): fI hm ≈ parity, fI ml-1m −0.056; fU hm +0.011 (dev),
  fU-noid ml-1m −0.009 (n100). Improvement knobs dc06/07/08 all negative. The merge is
  where "match or beat UI-ProtoMF" becomes the claim — against the healthiest H&M host
  (UI: 10 effective directions vs 5/2 for U/I; vault `2026-08-17_1823`) and the host on
  which the shifted cosine is most load-bearing (cosstd −5 pts; vault `2026-08-17_1937`).
- **R2/R3 on the UI host.** The host's own tie yields "compromised embeddings" and
  opaque projected halves (vault `2026-03-18_1234`). Composition is the one move that
  makes those halves *attributable* — the lineage's chance to convert the host's
  explainability trade-off into a gain rather than inherit it.
- **R5, two-sided.** fI covers cold items (measured), fU covers warm-thin users (D4
  strata); cold users and cold×cold remain unmeasurable in the current split.

### 0.5 Pre-registered merge-stage effects (Step 3b floor — verified from the vault this cycle, not assumed)

1. Live per-item intercept 1ᵀt̂ (rank-active) + inert per-user 1ᵀû; in fUfI the live
   one decomposes exactly over words + ID residual — "check which side is free"
   (vault `2026-07-12_1815`; scratchpad [captured dc05]).
2. Length-channel refund, **two-sided**: refund of the angular twin tax *vs* popularity
   contamination of directions through shared-row norms; noid arm strictly less
   expressive than host, ids arm equally expressive but differently priced (partial
   pooling), and **no direction-preserving dilation exists** in the composed
   parameterization (vault `2026-08-15_1721`, `2026-07-12_1224`).
3. **Correction on record:** prototypes hold *no* projection job at the tie — the
   projections are free `nn.Linear`; only entity embeddings get a second job; prototype
   norms stay pure gauge in every variant incl. UI (vault `2026-08-17_1823`). Any
   seed that ties W := P would create that job for real.
4. Why UI is the healthiest host is **open** (same near-zero coverage regulariser as the
   collapsed single hosts; candidate mechanism: the linear leg keeps embeddings
   linearly discriminative) — the merge's own geometry instruments inherit the question.
5. Base settings fixed and evidenced: shifted cosine, `use_bias=0` (vault `2026-08-17_1937`).
6. Gauge inheritances: F-DC01-08's frozen-at-init user-coefficient gauge and F-DC01-06's
   share non-identifiability (both sides now), dc05's B(t) disclosure line.

### 0.6 Open scratchpad entries that look relevant (Step 1b dispositions every open entry; relevance flag only)

- **User entry** *Candidate-selection sequencing: item-only → unified → separate sets*
  (2026-06-16) — stage 3 ("separate feature sets, mirroring `user_item_proto`") is
  literally this cycle.
- **User entry** *Design caution: two axes + Rudin axis* — the tie is the one place
  where Axis-B (scorer) and Axis-A (grounding) are structurally entangled; gut-check owed.
- `[captured scrutiny s0]` **dual-side candidate routes (a) mirror / (b) symmetric-by-
  design at the tie** — route (b) "has no object on a single-branch host, stays open
  for the fUfI merge cycle" (dc05 marker).
- `[captured scrutiny dc01]` **length-channel refund + the length-ablation control** —
  "belongs to the fUfI stage's run plan; its DC cycle inherits it".
- `[captured dc05]` **fI-vs-fU word-geometry comparison** (one-table-or-two evidence)
  and **live + inert baseline masses** (merge pre-audit) — both explicitly fUfI-stage.
- `[captured dc08]` **cross-side table tie at the merge** (history row = composed item
  vector; three-way knob separate / shared / shared+residual) — "type-correct only at
  the merge".
- `[captured dc08]` **named/unnamed coverage split as a lineage-wide backport** — the
  merge's read-out has two ID rows and two linear halves to cover.
- `[captured dc08]` **the fU HEAD may be where fU loses** (0.612 dot vs 0.544 prototype
  on ml-1m) — the merge *adds* a dot-product half to each side; adjacency to state.
- `[captured scrutiny s0]` 3-scenario cold testbed; `[captured scrutiny dc05]` cold-user
  degradation floors — the R5 two-sided reading lands on both.
- Remaining open entries (loss-enforced grounding, grounding spectrum, patch prototypes,
  multimodal content, live centroids, ID-residual diagnostic, photos-as-rendering,
  data-anchored parameterization, fU′ centring, fU′-after-identity ordering,
  identity-as-ingredient taxonomy) — swept in Step 1b.

### 0.7 Corpus logistics (flag, resolve before Step 2) — *resolved for this cycle 2026-10-26*

The 46 reading-list PDFs are gitignored and were lost with the 2026-10 laptop wipe;
this machine holds **one** PDF. GR6 (verify every cited claim against the local PDF)
cannot be met for Step 2 without the PDFs. *(Resolution, 2026-10-26: Drive never held them — `gdrive:protomf-thesis` does not exist — and LEO5 has none; the five load-bearing papers (A1 local, B5, B3, B1, I3) were re-fetched into their reading-list paths and every Step-2 pointer was checked against them; restoring the remaining ~41 PDFs is the user's separate task.)* Seed papers
this cycle will certainly need: A1 ProtoMF (§3.3 UI-ProtoMF, Eq. 12), B5 LightFM, B3
SVDFeature; others per the Step-1 choice.

---

## 1. Seed selection

### 1a. Corpus scan (all themes swept; targeted per M2 — the scan confirms where the merge's seeds live, it does not re-open the lineage)

- **A — host.** A1 §3.3 UI-ProtoMF is the host verbatim: each side's base embedding
  read in two roles (cosine against its own prototypes; free linear projection into the
  other side's prototype coordinates), score u*·t̂ + û·t* (Eq. 12; equation numbers
  re-verified against the local PDF in Step 2). The vault's code-and-paper check
  (`2026-08-17_1823`) stands: the projections are free `nn.Linear`, prototypes carry no
  projection job, nothing is shared across sides.
- **B — backbone.** B5 LightFM / B3 SVDFeature are already the lineage's composition
  template on both sides (dc01, dc05); nothing new is needed for the straight merge.
  **B1 CMF** (Singh & Gordon 2008 — *shared* latent factors across the user–item and
  item–feature matrices) is the classic ancestor of a **shared-vocabulary** merge and is
  the corpus anchor if that knob is designed (Seed B below). B12/B13 (SVD++/FISM) stay
  dc08's; B8 AFM / B9 deep-crossing are Axis-B scorer material, out of scope for a
  like-for-like merge. B10 DropoutNet remains a training trick for a later R5 arm.
- **C/D — prototypes in recsys.** D1 ProtoCF (interaction few-shot prototypes) and
  **I3 UIPC-MF** (a learned **connection matrix between user and item prototypes**,
  PAKDD 2024, CF-only) are the positioning neighbours: UIPC-MF is what a *head*
  change at the tie would resemble — relevant to rejecting Seeds C/D and to Step 2b.
- **E/F.** E2 SENN's tied fidelity objective and F1/F2 (disentanglement ≠
  interpretability) are Step 5 material; no seed. F5 (random-init SID tokens reorganise
  around popularity) is mechanism evidence for the merge's norm-entanglement worry
  (0.5 item 2), not a seed.
- **H.** No fashion-specific seed; images stay the S3 future-work channel (dc03 record).
- **§I.** The 2026-06-11 sweep found all 31 ProtoMF citers CF-only; Step 2b re-checks
  for the specific *merge* mechanism (feature-composed branches under a double tie).

### 1b. Scratchpad sweep — every open entry dispositioned (29 entries: 7 user, 22 captured)

**User entries**
1. *Attach feature profiles to existing CF prototypes (2026-06-11)* — **not this cycle**:
   the lineage grounds by shared geometry; profile-attachment stays the loss-enforced
   future candidate.
2. *Feature-injection fork: representation vs prior (2026-06-16)* — **answered inline**:
   the merge is route A on both sides; nothing enters as a prior. Route B still open.
3. *Add / interact / regularize (2026-06-16)* — **shapes this candidate (how)**: the
   straight merge is pure ADD on both sides, but the double tie's linear halves are a
   bilinear form between *composed* vectors — the first lineage stage where the two COMPOSITIONS interact bilinearly with *each other* *(corrected 2026-10-26, A8 after 5b-11: `lightfm_hist` and fI already pair one composition with a free table)* (û·t* and u*·t̂ both pair the two
   compositions through a learned matrix). Step 3 must name this: the merge is ADD inside
   each side and INTERACT across sides *via the host's own tie*, not via a new FM term.
4. *Attribute-source prioritization — intrinsic > post-hoc (2026-06-16)* — **shapes**:
   both sides stay item-vocabulary-grounded; no user attributes enter (that is candidate
   prime, vault `2026-09-08_1649`, after the merge).
5. *Candidate-selection sequencing: item-only → unified → separate sets (2026-06-16)* —
   **seed-adjacent, answered inline**: this cycle *is* the entry's stage 3 ("separate
   feature sets, mirroring `user_item_proto`"), with the correction that the lineage's
   two "sets" are the same item vocabulary in two roles, so stage 2 ("one undifferentiated
   set") and stage 3 collapse into the one-table-or-two knob (Seed A vs Seed B).
6. *Two axes + Rudin axis (2026-06-16)* — **shapes (gate applied)**: the tie is the one
   place where Axis A and Axis B are structurally entangled (the composed vector's norm
   is read by the scorer). Seed A keeps the scorer *exactly* the host's — Axis A only;
   Seeds C/D are Axis-B head changes and are rejected partly on this gut-check.
7. *Hybrid bridge desideratum (2026-06-16)* — **n-a**: no added module.

**Captured entries**
8. `[captured dc02]` *loss-enforced grounding* — **not this cycle** (future candidate;
   the merge stays at the unenforced point).
9. `[captured dc02]` *grounding-enforcement spectrum* — **answered inline**: the merge sits
   at the same unenforced/emergent point as dc01/dc05 — on *both* sides at once; the
   spectrum narrative is unchanged.
10. `[captured dc03]` *part/patch visual prototypes* — **not this cycle** (S3).
11. `[captured dc03]` *multimodal frozen content space* — **not this cycle**.
12. `[captured dc04]` *differentiable live centroids* — **not this cycle**.
13. `[captured scrutiny dc01]` *user-side explanation gap — route 1 (derived read-out
    through W^u) vs route 2 (history-composed users)* — route 2 consumed by dc05;
    **route 1 shapes this candidate**: the merge re-creates W^u, so the "project a user
    prototype through the tie into named item-prototype space" lens is available again as
    a *descriptive* read-out (enabled-not-enforced, extrapolation caveat) — owed in
    Step 3's read-out section beside the intrinsic word profiles.
14. `[captured scrutiny s0]` *dual-side candidate, routes (a) mirror / (b) symmetric-by-
    design at the tie* — **seed material**: route (a) is Seed A (both mirrors meet under
    the host tie); route (b) is Seeds C/D (a mechanism living at the tie) — shortlisted and
    rejected below on recorded gates, so (b) is now *answered*, not merely deferred.
15. `[captured scrutiny s0]` *3-scenario cold testbed* — **not this cycle** (eval design;
    still the unmet prerequisite for any cold-user or cold×cold claim — Step 5 R5 must
    say so).
16. `[captured scrutiny dc01]` *length-channel refund + length-ablation control* — **shapes
    (binding)**: Step 3b's inventory and Step 3's baseline/ablation design inherit the
    two-sided hypothesis and the user's normal-vs-equal-length control; the fI/fU arms are
    exact no-ops (keystone controls).
17. `[captured scrutiny dc01]` *ID residual as diagnostic read-out* — **shapes**: the merge
    carries two ID rows (item, user); the prototype-coordinate reading now has *two*
    prototype sets to read against, and the linear halves give each ID row a second,
    norm-sensitive image (W·e_ID). Step 3 records the discipline per row.
18. `[captured dc05]` *fI-vs-fU word-geometry comparison* — **shapes (as instrument)**:
    the designed evidence for the one-table-or-two knob. 100-trial fI and fU checkpoints
    exist on LEO5 for ml-1m; hm pending wave 1. Step 3 specifies it as a pre-build gate
    instrument (login-node class, Procrustes on centred tables per the dc06 caveat); this
    cycle runs nothing.
19. `[captured dc05]` *merge inherits one live + one inert baseline mass* — **shapes
    (binding)**: 3b floor item; the per-word zoom of the live intercept is a Step 3
    read-out.
20. `[captured scrutiny dc05]` *cold-user degradation floors* — **not this cycle**
    (hidden-effects exhibit); the merge's floor is new (both channels present) and is
    named in 3b.
21. `[captured dc03 abandonment]` *photos as rendering* — **not this cycle** (presentation).
22. `[captured dc04 abandonment]` *data-anchored parameterization* — **not this cycle**
    (reserved post-merge stage).
23. `[captured dc06]` *fU′ centring* — **not this cycle**; dc06 is ml-1m-only and
    unresolved; the merge takes *uncentred* compositions on both sides as its default
    (if dc06 ever wins, centring both sides is a later knob).
24. `[captured nl-track]` *item-identity rows in fU* — consumed by dc08; **not this cycle**:
    dc08's runs are not reported in the Nov-3 scope and the identity block would add a
    third unnamed summand to a merge that already has two ID rows.
25. `[captured dc08]` *cross-side table tie at the merge* — **seed (Seed B)**: shortlisted
    below; survives the gates; becomes the merge's single designed knob (Step 3), not the
    headline.
26. `[captured dc08]` *ordering constraint fU′ after identity rows* — **n-a** (neither
    centring nor identity rows enter).
27. `[captured dc08]` *the fU HEAD may be where fU loses* — **answered inline, partially**:
    the merge adds a dot-product (linear) half to *each* side *(A8, 2026-10-26: the 0.068 "head penalty" quoted in the entry mixes in the user-ID row's cost; see the §5.1 R6 correction)* — the same composition the
    prototype head penalised on ml-1m now also scores through a plain linear map. If the
    head penalty is real, the merge's UI-vs-fUfI gap should be *smaller* than fU's
    U-vs-fU gap on ml-1m; recorded as a pre-registered expectation for Step 5, not a
    mechanism.
28. `[captured dc08]` *identity-as-ingredient taxonomy* — **n-a** (no identity ingredient).
29. `[captured dc08]` *named/unnamed coverage split backport* — **shapes (binding)**: the
    merge's read-out must carry the two-way named-vs-unnamed split on *both* compositions
    and on both linear halves (the parameterization-invariant aggregate, dc08 4b C9);
    Step 3 specifies it.

### 1c. Shortlist

**Seed A — the straight merge: dc01's item composition × dc05's user composition × A1
UI-ProtoMF's unchanged double tie, separate word tables.** q_i = Σ_f e^I_f (+ e_ID(i)),
q_u = mean_{j∈H_u\{i}} Σ_f e^U_f(j) (+ e_ID(u)); each composed vector is read in its
side's two host roles (cosine vs own prototypes; `EmbeddingW`-style projection into the
other side's prototype coordinates); score u*·t̂ + û·t* exactly as the host.
- *Motivation gate:* R1's exemplar literally (each composed representation is the one
  base table read in two roles — dc01 §3.7 deviation 2, "parked to the merge stage");
  R6's paper-headline bar (fUfI vs UI-ProtoMF, the lineage's load-bearing comparison);
  R2/R3 on the host's opaque halves (û, t̂ become exact linear functionals of nameable
  compositions — read-out 4 returns); R5 two-sided by construction (cold item via words;
  thin user via basket words).
- *Distinctness:* dc01/dc05 — waived (M1: the merge is their composition by design);
  substantive delta anyway: a two-role read of the composed vectors (norm enters the
  score) and two grounded prototype sets in one score. dc02/dc03/dc04 — different entry
  (bottleneck / frozen image / prototype parameterization) and none engaged the tie.
  dc06 — gauge knob on one side, ml-1m-only. dc07 — read-out axis, stopped. dc08 —
  identity ingredient, absent here.
- *R8 parsimony:* **PASS** — exactly the `user_item_proto` move (the reference for "how
  big one idea is") applied to the two already-scrutinized composed branches; no new
  layer, loss, or table.

**Seed B — the shared-vocabulary merge: Seed A with ONE item-word table read by both
sides** (e^U := e^I; the user is the mean of the item side's own word compositions — the
[captured dc08] cross-side tie; B1 CMF as the corpus ancestor of shared factors across
matrices).
- *Motivation gate:* R1 beyond the exemplar (the first table read by both sides — "the
  tie applied to the vocabulary itself"); R4 (user and item prototypes become mutually
  comparable in one space — a cross-prototype similarity cos(p^u_l, p^t_k) exists for
  free, which is UIPC-MF's connection matrix as a *read-out* instead of a parameter);
  parameter parsimony (one 426-row table).
- *Distinctness:* vs Seed A on the **grounding** axis (one vocabulary geometry serving the
  item-property and taste-indicator roles vs two role-specific geometries); vs the
  registered rows as Seed A.
- *R8 parsimony:* **PASS as one knob** ("share the table"), but as a *headline* it stacks
  two moves — compose both sides *and* tie them across sides — and it pre-empts the very
  decision the [captured dc05] geometry comparison was designed to inform. It also
  couples the two roles' gradients (user-side training reshapes the item vocabulary),
  which the stage-wise lineage has so far kept separate. **Not chosen as headline; becomes
  Seed A's single designed arm in Step 3** (default separate, shared as the ablation,
  geometry comparison as the pre-build instrument). This is a user decision to flip.

**Seed C — route (b) at the tie: the cross-cosine merge.** Drop both projections; read
each composed vector against *both* prototype sets by shifted cosine: score =
Σ_l (1+cos(q_u,p^u_l))(1+cos(q_i,p^u_l)) + Σ_k (1+cos(q_u,p^t_k))(1+cos(q_i,p^t_k)).
Requires one shared space (Seed B's table).
- *Motivation gate:* R3 maximal (no linear half anywhere, every term prototype-shaped and
  nameable; no Steck exposure). **FAILS on R6-discipline and lineage logic:** it is a head
  change — fUfI-C(F=0,+ID) does *not* reduce to `user_item_proto` (no W), so the
  stage-bar comparison would confound composition with a new scorer; the length channel
  stays closed, so the registered refund hypothesis and the built-in 2×2 lose their
  object; dc07 is the local evidence that head interventions on this codebase fail *for
  a reason* (the popularity channel re-forms).
- *Distinctness:* new on the entry axis (both sides measured against both prototype sets).
- *R8:* one idea — but rejected on the motivation gate. **Captured** to the scratchpad as
  `[captured dc09]` for a post-merge head cycle.

**Seed D — prototype-defined projections (W^u := P^t, W^t := P^u).** Keep the linear
halves but make the projection matrix *be* the other side's prototype matrix, so
û_k = ⟨q_u, p^t_k⟩ is a geometric alignment rather than a free functional (the
"Route-1-style construction" the vault correction `2026-08-17_1823` names). Also
requires the shared space.
- *Motivation gate:* names the linear halves' coordinates *by construction* and removes
  2·d·K free parameters; keeps the length channel (refund testable). **FAILS the same
  like-for-like discipline** (reduction target is a *constrained* host, not the host),
  and it creates the prototype projection job for real — prototype norms stop being gauge,
  so the host's clean separation (cosine job norm-blind, no projection job) is given up
  and every prototype-norm instrument in the lineage would need re-derivation.
- *Distinctness:* grounding axis (prototypes double as projection bases).
- *R8:* one idea — rejected on the motivation gate. **Captured** as `[captured dc09]`.

### 1d. Chosen seed and provenance

**Chosen: Seed A — the straight merge under the unchanged host tie, separate tables.**
Provenance: **`lineage+corpus`** (declared label, dc05-brief style): the two
scrutinized stages dc01 (fI) and dc05 (fU) composed under A1 §3.3's double tie, with
B5/B3 as the composition template on both sides; dc01's superseded UI-hosted §3.2/§3.3
text is the item-side skeleton; dc05 §3 the user-side one. No external paper is
load-bearing for the headline; Step 2b probes the *merge* for prior art.

Why it won: it is the only shortlisted seed whose host comparison isolates exactly one
thing — "compose both sides" — with the keystone reduction fUfI(F=0,+ID) ≡
`user_item_proto` well-defined, so the lineage's load-bearing R6 claim is clean; it
keeps every stage-level finding (F-DC01-*, F-DC05-*, the refund hypothesis, the 2×2,
the length-ablation control) alive with its object intact; it mirrors the host
*exactly* (ProtoMF's own UI ties within sides, never across); and it leaves the one
genuine merge decision — one table or two — as a designed, evidence-informed arm
rather than an armchair choice. The brief-style null-candidate question is therefore
*not* answered at Step 1: whether the straight merge survives the two-branch physics
(norm entanglement, live intercept, two ID rows, LOO under a composed item side) is
Steps 3/3b/4b's job.

**Two-axes gut-check (user rule):** Axis A only — the scorer is the host's, untouched.
**Working `ft_type`:** `feature_user_item_proto` (new; `dual_item_proto` is a different
reserved idea) — confirmed or revised in Step 3.

---

## 2. Mechanism extraction

**Verification note (GR6).** The 46 corpus PDFs were lost with the 2026-10 laptop wipe and
were never pushed to Drive (`gdrive:protomf-thesis` does not exist; LEO5 holds none). For
this cycle the load-bearing papers were re-fetched on 2026-10-26 into their reading-list
paths and every pointer below was checked against the text of those files (pypdf
extraction, isolated install; page numbers = the PDF's printed page numbers):
`Master/Protomf-paper.pdf` (A1, RecSys '22 pp. 246–256);
`B_feature_aware_mf/B5_kula_2015_lightfm_metadata_embeddings.pdf` (arXiv:1507.08439v1);
`B_feature_aware_mf/B3_chen_2012_svdfeature_jmlr.pdf` (JMLR 13:3619–3622);
`B_feature_aware_mf/B1_singh_gordon_2008_cmf.pdf` (KDD '08, CMU copy);
`I_protomf_forward_citations/I3_uipc_mf_2024.pdf` (arXiv:2308.07048, for Step 2b).
Per M2 the extraction is scoped to the merge's load-bearing claims; dc01 §2 and dc05 §2
remain the records for the single-side readings of B5/B3/A1 and are **not** repeated.

### 2.1 A1 UI-ProtoMF — the host (§3.3, p. 250, Eqs. 11–12, footnote 1)

**The paper says.** After U-ProtoMF (§3.1, Eqs. 1–6: u* = [1 + cos(u, p^u_l)]_l with all
entries "positive in the range of 0 to 2"; U-score = Σ_l s^user_l, s^user = u* ⊙ t with
t ∈ R^{L^u}; sampled-softmax L_rec + λ_L2‖Θ‖; in-batch inclusion criteria R_{P^u→U},
R_{U→P^u}) and its mirror I-ProtoMF (§3.2, Eqs. 7–10), §3.3 introduces the merge:

> "A natural extension is to simply combine these two models to exploit the benefits of
> both under the hood of one single model. We provide this by introducing the UI-ProtoMF
> model, which computes the recommendation score as the sum of the scores of both models.
> While UI-ProtoMF contains both U-ProtoMF and I-ProtoMF as two separate units, the
> embeddings of users and items can be shared across these two units. To this end,
> UI-ProtoMF defines two linear transformations, one from the user embeddings to the space
> of item prototypes, and the other from the item representations to the user prototypes
> space:"
>
> û = W^u u ∈ R^{L^t}, t̂ = W^t t ∈ R^{L^u} (11)
>
> UI-score(u,t) = U-score(u,t) + I-score(u,t) = u*ᵀ t̂ + ûᵀ t* (12)
>
> L_UI-Proto = L_U-Proto + L_I-Proto, footnote 1: "L_rec is included only once and is based
> on the UI-score."
>
> "In the definition of UI-score, we particularly opt for the sum of the scores and avoid
> any non-linear combinatorial function. This design choice enables us to easily separate
> the contribution of each unit (U-ProtoMF or I-ProtoMF) to the final recommendation
> score."

So the only sharing the paper defines is **within a side**: one base embedding u ∈ R^d
feeds both the user-prototype unit (cosine) and the linear map W^u into item-prototype
coordinates; mirror for items. W^u, W^t are free matrices. Nothing is shared *across*
sides. (Code agrees — §2.5.)

**Training & tuning (§4).** Leave-one-out temporal split; 1 positive + 99 uniform
negatives at eval, negatives-in-training a hyperparameter; TPE over 100 sampled configs;
100 epochs, early stop after 10 epochs without val HitRatio@10 improvement; model with best
val HR@10 selected; 3 seeds averaged. (Our fleet: 100 trials, seed 38210573, same stop
rule — memory `project-nov3-scope`.)

**What they showed (Table 2, p. 251).** NDCG@10 / HR@10 — ml-1m: MF .326/.571, ACF
.335/.597, U-ProtoMF .333/.583, I-ProtoMF .303/.544, **UI-ProtoMF .383/.657** (significant
over MF and over ACF; AmazonVid HR the one exception). Datasets (Table 1): ml-1m 6,034 users
/ 3,125 items / 574,376 interactions after 5-core — the same split family as ours.
**Parameter accounting (§5.1):** MF has (N+M)·d; "ACF and UI-ProtoMF add K×d and 4K×d
parameters to MF, respectively" (K prototypes per side, two prototype matrices + two
K×d-sized projections).

**Explanations the host ships (§5.2).** User prototypes are interpreted by a synthetic
maximally-activating u*; item prototypes by nearest items (Table 3). A recommendation is
explained through s^user_l = u*_l · t̂_l and s^item_l = û_l · t*_l, and the paper reads
the projected coordinates as weights: "a value s^user_l can be high (or low) due to the
corresponding values of the underlying user prototype and item embedding, respectively,
u*_l and t̂_l" — t̂ is called "the item embedding (representing the movie)". **Nothing in
the paper attributes t̂_l or û_l to anything beneath them** — they are learned linear
functionals of free vectors. That is the opacity the merge composition turns into
attributable content.

**Future work the authors name (§6, p. 254).** "we believe that including external
features of users and items, such as contextual information or audio features, into
ProtoMF might further benefit the interpretability of the prototypes."

### 2.2 B5 LightFM — the composition template on both sides (§2.1–2.2, Eq. 1)

**The paper says (§2.2, verified).** "Users and items are fully described by their
features"; with d-dimensional *feature* embeddings e^U_f (user features F^U) and e^I_f
(item features F^I): q_u = Σ_{j∈f_u} e^U_j, p_i = Σ_{j∈f_i} e^I_j, biases b_u = Σ b^U_j,
b_i = Σ b^I_j; r̂_ui = f(q_u · p_i + b_u + b_i) (Eq. 1), f = sigmoid for binary data.
§2.1, plain language: "The representation for denim jacket is simply a sum of the
representation of denim and the representation of jacket; the representation for a female
user from the US is a sum of the representations of US and female users."

**Load-bearing for this cycle:** LightFM keeps **two feature sets with two embedding
tables** — F^U with e^U and F^I with e^I — by construction. A single table read by both
sides is *not* LightFM; it is Seed B's deviation (§2.4). *(Corrected 2026-10-26, A7 after 5b-3: LightFM's two tables are two **vocabularies** (user metadata vs item metadata); the paper offers no precedent for two tables over ONE vocabulary read in two roles. The two-table default of Seed A is therefore my design choice — stated as such — not a LightFM convention.)* Everything else (MF↔CB spectrum
§2.3, cold fold-in §7.1, Table 1 results, admitted limitations) carries from dc01 §2.1 /
dc05 §2.1 unchanged.

### 2.3 B3 SVDFeature — real-valued composition weights (§2, the single display equation)

**Verified:** ŷ(α,β,γ) = (Σ_j γ_j b^(g)_j + Σ_j α_j b^(u)_j + Σ_j β_j b^(i)_j) +
(Σ_j α_j p_j)ᵀ(Σ_j β_j q_j), "Intuitively, we use a linear model to construct user and
item latent factors from features"; p_j, q_j ∈ R^d per feature; SGD on square / logistic /
smoothed-hinge losses. Again **separate** user-feature factors p and item-feature factors q.
The real-valued α licenses dc05's mean-normalised basket weights (carried).

### 2.4 B1 CMF — the shared-factor ancestor (anchor for the Seed-B arm only)

**The paper says (abstract; §1; §4).** "we simultaneously factor several matrices, sharing
parameters among factors when an entity participates in multiple relations"; "whenever an
entity type is involved in more than one relationship, we tie factors of different models
together" (§1); for users×movies and movies×genres, "Since V is a shared factor we average
the losses" (§4). **I propose** the reading for Seed B: the shared "entity type" is the
*attribute word*, which participates in two compositions (item = Σ words; user = mean over
the basket of Σ words) — a word-level tie inside one score, not CMF's entity factor shared
across two matrices. The mapping is an analogy, not an instance.

### 2.5 The host as implemented in this repository (code, verified 2026-10-26)

- **`FeatureExtractorFactory.create_models`, branch `prototypes_double_tie`**
  (`feature_extraction/feature_extractor_factories.py`): asserts `use_weight_matrix` off on
  both sides; builds `user_proto = PrototypeEmbedding(n_users, d, K_u)` (base
  `Embedding(N, d)` inside) and `user_embed = EmbeddingW(N, d, out_dimension=K_t)`; mirror
  for items; **ties by weight assignment**
  `user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight`
  (same for items); wraps `ConcatenateFeatureExtractors(user_proto, user_embed,
  invert=False)` → [u*, û] ∈ R^{K_u+K_t} and `(item_proto, item_embed, invert=True)` →
  [t̂, t*] ∈ R^{K_u+K_t}. `RecSys.forward` then computes `dots = Σ(u_embed ⊙ i_embed)`,
  i.e. u*·t̂ + û·t* — Eq. 12 exactly. `Concatenate.get_and_reset_loss` sums the two
  `PrototypeEmbedding` accumulators (inclusion regularisers of both units, Eqs. 4–5 and 9).
- **Search space** (`confs/hyper_params.py`, `proto_double_tie_chose_original_hyper_params`):
  `embedding_dim` ∈ [10, 100), `n_prototypes` ∈ [10, 100) per side, `sim_proto_weight` /
  `sim_batch_weight` ∈ loguniform(1e-3, 10) per side, `reg_*_type='max'`,
  `cosine_type='shifted'`, plus the base space (loss ∈ {bce, bpr, sampled_softmax},
  neg_train ∈ [1, 50), optimizer/lr/wd, batch size). Fleet-wide `use_bias=0`.
- **The two composed branches that the merge composes:** `feature_item_proto` (dc01) builds
  `PrototypeEmbedding(..., embedding_ext=FeatureEmbedding(...))` for items against a free
  `Embedding(N, K_t)` user; `feature_user_proto` (dc05) builds
  `PrototypeEmbedding(..., embedding_ext=HistoryFeatureEmbedding(...))` for users against
  a free `Embedding(M, K_u)` item. Both branches have the factory as the single
  initialising owner of the composed table (`*.init_parameters()` called once; the
  prototype module never inits a passed-in ext). `FeatureEmbeddingW(shared, out_dimension)`
  exists (kept "for the fUfI merge stage") and wraps a shared `FeatureEmbedding` instance
  with a bias-free `nn.Linear(d, out)`; **no user-side analogue exists** — it wraps the
  `FeatureEmbedding` type specifically, while the user base is a `HistoryFeatureEmbedding`.
- **Leave-one-out plumbing:** `RecSys.forward` hands `pos_i_idxs=i_idxs[:, 0]` to the user
  extractor iff `supports_loo`; `PrototypeEmbedding.supports_loo` delegates to its ext and
  forwards `**ext_kwargs`. `ConcatenateFeatureExtractors` has **neither** a `supports_loo`
  property nor a kwargs pass-through, and `FeatureEmbeddingW.forward` takes only `o_idxs`.
  Under the merge the user's composition is consumed by *two* modules per step (cosine
  unit, projection unit); both must see the same leave-one-out vector, or the projection
  half would score the positive against a user vector that contains the positive's own
  words. Step 3 owns this (merge-specific plumbing, not a mechanism).
- **Hyper-parameter mirrors:** fI and fU search spaces mirror their single-branch hosts
  exactly, with `use_id_feature`, `feature_fields='canonical'` (hm: the S0.5 five fields,
  V=426, layout fixed; ml-1m: genres+tags@0.8 bags), and fU's `loo_pooling=True`.

### 2.6 The merged mechanism (dc09, stated once — every deviation flagged)

Host UI-ProtoMF verbatim (§2.1), with **one change applied to each side's base
embedding** — the free tables are replaced by the two already-scrutinised compositions:

> **Item (dc01, verbatim):** q_i = Σ_{f∈f_i} e^I_f (+ e^I_ID(i) in the ids arm)
> **User (dc05, verbatim):** q_u = (1/|H_u∖{i}|) Σ_{j∈H_u∖{i}} Σ_{f∈f_j} e^U_f (+ e^U_ID(u)),
> the `∖{i}` leave-one-out applying in training only
> **Prototype units (host Eqs. 1, 7):** u* = [1 + cos(q_u, p^u_l)]_{l≤K_u},
> t* = [1 + cos(q_i, p^t_k)]_{k≤K_t}
> **Linear units (host Eq. 11):** û = W^u q_u ∈ R^{K_t}, t̂ = W^t q_i ∈ R^{K_u}, W free
> **Score (host Eq. 12):** UI-score(u,i) = u*ᵀ t̂ + ûᵀ t*
> **Loss:** host Eq. 3 (L_rec once, on the UI-score) + both units' inclusion regularisers
> (Eqs. 4–5, 9) with their four weights; fleet `use_bias=0`.

Tables: **separate** e^I (item words + item IDs) and e^U (the same word vocabulary in the
taste-indicator role + user IDs) — a design choice *(A7: B5/B3's two tables are two vocabularies, not a precedent for one vocabulary in two roles)*. Sharing e^U := e^I is Seed B,
designed in Step 3 as the merge's single knob.

**"The paper says" vs "I propose":**
1. Composed base embeddings under the double tie — in no paper (dc01 §3.7 dev. 1, now on
   both sides at once). *I propose.*
2. The within-side tie realised by **sharing the composed module instance** between the
   cosine unit and the projection unit (dc01 §3.7 dev. 2, "parked to the merge stage")
   rather than by the host's weight assignment — mathematically identical, needed because
   a composition is not an `nn.Embedding` whose `.weight` can be re-pointed. *I propose.*
3. No LightFM feature biases (fleet-wide `use_bias=0`; dc01 dev. 4) — with the consequence
   recorded in the vault: the live intercept 1ᵀt̂ now plays the composed-bias role
   implicitly. *Carried deviation.*
4. History-derived user features with mean-normalised counts and train-time leave-one-out
   (dc05 §3.1; B3's real-valued α as licence) — the host's own user "embedding" is free.
   *Carried deviation.*
5. Attribution read-outs — per-feature / per-purchase shares of both cosine units
   (dc01/dc05) **plus exact linear-functional attribution of both projection halves**
   (F-DC01-07's parked read-out 4) — in no paper. *I propose; designed in Step 3.*

---

## 2b. Prior-art probe

**Probe target (the adaptation, not the seed):** "feature-composed item embedding
(sum of attribute-word embeddings) AND history-composed user embedding (mean of the
basket's attribute-word embeddings), each read in two roles under a double-tied
prototype architecture — cosine against its own side's prototypes and a free linear
projection into the other side's prototype coordinates — with exact attribution of all
four score components (two cosine units, two linear units) to feature words / purchases."

**Queries run 2026-10-26** (WebSearch ×5; Semantic Scholar API rate-limited this
session, OpenAlex used for the citer list):
1. `prototype-based matrix factorization recommendation "item features" OR "side
   information" OR metadata user prototypes item prototypes explainable feature-composed
   embeddings`
2. `LightFM metadata embeddings sum of feature embeddings combined with prototype layer
   cosine similarity prototypes explainable recommender user and item features`
3. `"ProtoMF" prototype matrix factorization extension features attributes content
   cold-start 2025 2026` (extended mode)
4. `hybrid recommender user prototypes and item prototypes both grounded in item
   attributes history-aggregated user representation attribute embeddings explainable
   prototype similarity linear projection`
5. `attribute-aware OR content-aware prototype-based recommender explainable prototypes
   composed from item metadata embeddings user and item prototypes cold-start 2024 2025
   arXiv` (extended mode)

**§I forward-citation re-check.** OpenAlex lists **38** citing works for
DOI 10.1145/3523227.3546756 (Semantic Scholar had 31 on 2026-06-11 and 2026-07-12; list
saved to the session scratchpad `prior_art/openalex_citations.json`). The seven not in
the reading list's §I, classified **by title only** this session (abstract-level
classification owed at Step 6 reading-list maintenance or at thesis RW time): *Embedding
Cultural Diversity in Prototype-Based Recommender Systems* (LNCS 2025 — the published
form of I2's arXiv), *Multi-Objective Unlearning … Preference Guided Pareto Exploration*
(2025), *Parameter-Efficient Single Collaborative Branch for Recommendation* (2025, JKU
single-branch line), *Minimal-Perturbation Counterfactuals through Guided Denoising
Diffusion* (SIGIR 2026), *RICE: … Interaction-Based Counterfactual Explanations* (LNCS
2026), *Measuring What Matters: … Counterfactual Explanations* (TORS 2026),
*Multi-scenario Knowledge-Enhanced Latent Intent Modeling* (LNCS 2026), plus several
chapters of one *Information Retrieval Series* book (Transparency; Biases, Fairness;
Privacy; Regulatory Initiatives; Introduction; Conclusions). **None of the 38 titles
names item features, metadata, attributes, or content grounding of prototypes**; the
prototype-flavoured newcomers are counterfactual-explanation and cross-domain
preference-prototype work. The feature-aware wedge, and a fortiori the merge mechanism,
remain unoccupied in ProtoMF's own citation neighbourhood.

**Findings — closest relatives, each distinguished:**

- **I3 UIPC-MF (Pan & Soo, PAKDD 2024; local PDF read this cycle).** Eq. 2: score =
  Σ_i Σ_j w_ij · sim(u, p^u_i) · sim(t, p^t_j) with a global K_u×K_t connection matrix
  replacing ProtoMF's two projections; Eq. 3 reads r_j = Σ_i w_ij sim(u, p^u_i) as the
  user's preference for item prototype j. Their own account of ProtoMF: "the lack of
  connections between user prototypes and item prototypes makes it unable to directly
  associate the relationship between them". **Distinguished:** CF-only (free u, t), a
  *head* change at the tie (exactly the Seeds C/D territory this cycle rejected), no
  attribute grounding, no composition. Their future work (Conclusion) names ours:
  "combine [the prototype] with content-based recommendation to address the cold start
  problem … incorporating more information such as user and item features". Also
  relevant as a *read-out* idea for Seed B's arm: with both prototype sets in one space
  the matrix cos(p^u_l, p^t_k) is a parameter-free analogue of w_ij.
- **TEASER — De Pauw, Ruymbeek & Goethals, "Modelling Users with Item Metadata for
  Explainable and Interactive Recommendation" (arXiv:2207.00350, 2022; surfaced by
  query 5; abstract read).** A linear CF model whose user profile lives "in the domain
  of item metadata … arguably the most interpretable domain for end users", with the
  recommendation a linear function of item metadata and that profile, and the profile
  weights user-editable. **Distinguished:** no prototype layer, no double tie, no
  cosine units; the user profile is a *free* per-user vector over metadata dimensions
  (learned, not composed from the purchase history), items enter as their raw metadata
  row; explanation = the profile itself. It is, however, the closest published
  statement of the lineage's *grounding argument* (ground user models in item
  vocabulary) and a missed neighbour for dc05 — **not in the reading list, not in
  dc05 §2b, not in the vault** (grep this session). **M7 trickle-back:** add to the
  reading list at Step 6 (Theme B, cite-in-RW beside B11 Balog et al.) and flag for the
  fU chapter's related-work paragraph.
- **B5 LightFM (Kula 2015) / B3 SVDFeature (Chen et al. 2012).** Both compose users AND
  items from features under a plain dot product with *separate* feature tables (§2.2–2.3
  above) — the merge's backbone precedent; no prototype layer, no attribution beyond the
  dot product. The straight merge is "LightFM with both compositions scored through
  UI-ProtoMF's head" — the same relation dc01/dc05 had to their hosts, now on both sides.
- **B1 CMF (Singh & Gordon 2008).** Shared factors across relations when an entity
  participates in several — the ancestor of Seed B's shared table, not of the straight
  merge (which shares nothing across sides).
- **Not close (re-confirmed):** D1 ProtoCF (interaction few-shot prototypes), the
  JKU single-branch line (RecSys 2024 / TORS 2025 / 2025 — one shared branch for all
  modalities, no prototypes, cold-start via modality dropout), the counterfactual
  explainers (LXR, RICE, diffusion counterfactuals — post-hoc, model-agnostic), PPA/PPA++
  (cross-domain preference prototypes, no content). The Innsbruck thesis-topic listing
  that query 4 returned (dbis-informatik.uibk.ac.at/node/908) is this thesis's own
  posting, not prior art.

**SC.2 amendment 2026-10-26 — F-DC09-10: positioning of the shared collapsed form.** Under the user's shared default, 3b-iv's collapsed score expands into pairwise inner products between basket words and candidate words through a learned d×d core (A = P̂^uᵀW^t and B = P̂^tᵀW^u, plus the composed per-item bias): the structure of a **factorization machine** (Rendle 2010 — one vector per feature across roles, identity core), with **SVD++ / FISM** (history-composed users; separate history and candidate vectors) and **attentional FM** (learned pair weights) as the nearest relatives; M16's twin channel is FISM-through-words. **Distinguished:** the core is prototype-mediated (P̂ᵀW, rank ≤ min(d, K)); each channel carries one per-entity cosine normalisation, which no FM has; and the explanation units are prototypes with word profiles, which no FM has. Novelty verdict unchanged; a related-work obligation is added — the thesis must position fUfI against FM/AFM/SVD++ explicitly, or a reader will read it as an FM with extra steps.

**Verdict: found nothing close — with this probe** *(qualified 2026-10-26, A12 after 5b-20: five queries, Semantic Scholar rate-limited, seven new citers classified by title only; abstract-level classification owed at thesis RW time)*. The composition template on both sides is LightFM's
own; the double tie is ProtoMF's own; nobody has put the two together, and the
*attribution of the projected halves to feature words* — the merge's new explanation
object — has no object in any of the relatives (UIPC-MF's r_j is the nearest in spirit
and is CF-only). No refinement is forced at this gate. Required citations the thesis
owes: UIPC-MF (distinguish head-change vs composition), TEASER + B11 (grounding users in
item vocabulary, prior statements of the argument), LightFM/SVDFeature/CMF (two-table vs
shared-table conventions).

---

## 3. Adaptation design

Verified against the code on 2026-10-26 (`feature_extraction/feature_extractors.py`,
`feature_extractor_factories.py`, `rec_sys/rec_sys.py`, `confs/hyper_params.py`,
`utilities/explanations/breakdown.py`). Dataset numbers: hm_1_month N = 73,418 users,
M = 13,651 items, canonical five fields, V = 426 words (`hm_dataset_notes.md`,
`hm_feature_analysis.md`); ml-1m N = 6,034, M = 3,125, V = 1,061 words (genres +
tags@0.8 bags), mean 24.3 tokens per movie (dc05 §3.1 scope note, dc08 §4).

### 3.1 The one idea, and the score it produces

Replace **both** free base tables of UI-ProtoMF by the lineage's two compositions and
change nothing else:

> **q_i = Σ_{f∈f_i} e^I_f + e^I_ID(i)** (item words + item-ID row; `FeatureEmbedding`, dc01)
> **q_u = (1/|H_u∖{i}|) Σ_{j∈H_u∖{i}} Σ_{f∈f_j} e^U_f + e^U_ID(u)** (basket words + user-ID row;
> `HistoryFeatureEmbedding`, dc05; `∖{i}` in training only)
> **u\* = 1 + cos(q_u, P^u) ∈ R^{K_u}, t\* = 1 + cos(q_i, P^t) ∈ R^{K_t}** (host Eqs. 1, 7)
> **û = W^u q_u ∈ R^{K_t}, t̂ = W^t q_i ∈ R^{K_u}** (host Eq. 11, W free)
> **S(u,i) = u\*ᵀ t̂ + ûᵀ t\*** (host Eq. 12)

*(User review decision 2026-10-26: the DEFAULT is now ONE shared word table, E^I_words = E^U_words; separate tables are the `_sep` arm. The text below describes the separate-table variant as written during the cycle; read "default"/"arm" swapped where the two are contrasted.)* Two word tables, E^I (V + M rows) and E^U (V + N rows), over the **same** vocabulary
index *(a design choice — see §2.2's A7 correction: LightFM's F^U/F^I are two vocabularies)*. Prototypes, inclusion regularisers (Eqs.
4–5, 9 — four weights), sampled-softmax loss with L_rec counted once (footnote 1),
negative sampling, `use_bias=0`: all exactly the host.

**The score identity that shapes every read-out.** Writing u\*_l = 1 + c_l(u) and
t\*_k = 1 + c_k(i) with c = cosine,

> S = **1ᵀt̂** + Σ_l c_l(u)·t̂_l + **1ᵀû** + Σ_k û_k·c_k(i)
>  = B_item(i) + U-half(u,i) + B_user(u) + I-half(u,i).

- **B_item(i) = 1ᵀW^t q_i** is a per-*item* scalar → **rank-active** (the live implicit
  intercept, vault `2026-07-12_1815`), and it decomposes exactly over the item's rows:
  B_item(i) = Σ_f (W^tᵀ1)·e^I_f + (W^tᵀ1)·e^I_ID(i) — LightFM's composed bias re-emerging
  implicitly, **nameable per word**.
- **B_user(u) = 1ᵀW^u q_u** is a per-*user* scalar → rank-inert for item ranking (shown for
  magnitude honesty, as fI's rank-inert line).
- The two personalised halves are each a sum over (prototype, named row) pairs — §3.4.

So the merge carries exactly one popularity-capable intercept, and for the first time
in the lineage it is attributable rather than opaque. *(SC.2 amendment 2026-10-26 — F-DC09-09: "exactly one" is an algebraic statement about the explicit intercepts only. The personalised halves carry **common-mode mass**: with q_u = μ + δ_u + e^U_ID(u), μ the basket-mean centroid toward which heavy users shrink (F-DC05-01), the terms (W^uμ)ᵀt\*(i) and u\*(μ)ᵀt̂(i) are per-item scalars shared by every user — on ml-1m three per-item scalars, not one, and read-out 4 would narrate them as personal taste. Described in 3b-xii; one one-off descriptive measurement; no mitigation.)*

### 3.2 Modules and classes (anti-vagueness gate)

**`ft_type = 'feature_user_item_proto'`** (new; *I propose* the name — CLAUDE.md's
reserved `dual_item_proto` names a different idea, "dual CF + feature prototype spaces",
and stays reserved for it). Config names *(as fixed at the user review 2026-10-26)*: `feature_user_item_proto` (ids on both sides, **shared** word table), `feature_user_item_proto_noid` (no ID row on either side, shared), `feature_user_item_proto_sep` (ids, separate tables), `feature_user_item_proto_noid_sep` (noid, separate) — the 2×2 of §3.7; the mixed-ID arm is optional only.

*(SC.2 amendment 2026-10-26 — F-DC09-01, the shared-table build spec at the anti-vagueness level; the module text below still describes the `_sep` layout.)* **Shared default, specified.** One word-table module `E_words` (V×d) is created and initialised **once by the factory** (single owner — the dc01/dc05 convention) and handed to both compositions; each side keeps its **own ID block** (M×d for items, N×d for users). A word *block* cannot be shared between two differently-sized `nn.Embedding`s (`[words; item IDs]` vs `[words; (dc08 rows); user IDs]`) by the host's `.weight` re-pointing, so `FeatureEmbedding` and `HistoryFeatureEmbedding` take an optional external word block (`word_table=`) and compose as word-lookup + ID-lookup on both layouts; the single-table path (no external block) must stay **bit-identical** — guards: the dc01 t03 RNG/golden pin and the dc05 keystones. `max_norm`: the two per-side slots become one key on the shared block (a double renorm is idempotent — not searched, inert). The wrapper exposes the composition (`self.base`) so both cold finders (`_find_feature_embedding`, `_find_history_feature_embedding`) reach it, and both `config_expects_id_drop` / `config_expects_user_id_drop` register the new `ft_type` — otherwise the cold runner would silently skip the cold step and the post-drop abort would not fire (the F-S0-07 silent-depressant class). `_sep` = the same factory with two word blocks. Keystone C4 unaffected (V = 0). **Sharing keystone (dc_checks/dc09 t07):** the shared model ≡ the `_sep` model with E^U_words := E^I_words tied — scores equal, ∇_{E_words} L = ∇_{E^I_words} L + ∇_{E^U_words} L (chain rule), under sampled softmax and BCE; verified at toy scale in the SC.2 round (§6.4, D1–D2). Code half lands at SC.3.

**1. `feature_extraction/feature_extractors.py` — one new wrapper, one additive method.**

The host realises the tie by re-pointing an `nn.Embedding.weight` between two modules
and calling the lookup twice per step. A composition has no `.weight` to re-point and
(on ml-1m bags) is the step's dominant cost, so *I propose* a wrapper that computes the
composed vector **once** and feeds both roles — mathematically identical to two calls:

```python
class TwoRoleComposedExtractor(FeatureExtractor):
    """One composed base vector q read in the host's two UI roles (dc09, fUfI merge):
    cosine against this side's prototypes (via `proto`, a PrototypeEmbedding whose
    embedding_ext IS `base`) and a free linear projection into the OTHER side's
    prototype coordinates (host Eq. 11). Output = cat([sims, lin]) (user side) or
    cat([lin, sims]) (item side, `invert=True`) — the ConcatenateFeatureExtractors
    layout, so RecSys's dot gives u*·t̂ + û·t* unchanged."""
    def __init__(self, base: FeatureExtractor, proto: 'PrototypeEmbedding',
                 out_dimension: int, invert: bool = False, use_bias: bool = False,
                 normalize_composed: bool = False):
        super().__init__()
        assert proto.embedding_ext is base, 'proto must wrap the same composed instance'
        self.base, self.proto, self.invert = base, proto, invert
        self.normalize_composed = normalize_composed      # §3.7 length-ablation control
        self.linear_layer = nn.Linear(base.embedding_dim, out_dimension, bias=use_bias)
        self.name = 'TwoRoleComposedExtractor'

    @property
    def supports_loo(self) -> bool:                     # RecSys.forward hands pos_i_idxs
        return bool(getattr(self.base, 'supports_loo', False))

    def init_parameters(self):                          # single-owner: NOT the base table
        self.linear_layer.apply(general_weight_init)    # (EmbeddingW's init, same draw)

    def forward(self, o_idxs, **ext_kwargs):
        q = self.base(o_idxs, **ext_kwargs) if ext_kwargs else self.base(o_idxs)   # (..., d)
        if self.normalize_composed:                      # exact no-op through the cosine
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sims = self.proto.sim_from_embed(q)              # (..., K_this) + reg accumulation
        lin = self.linear_layer(q)                       # (..., K_other)
        return torch.cat([lin, sims], -1) if self.invert else torch.cat([sims, lin], -1)

    def get_and_reset_loss(self) -> float:
        return self.proto.get_and_reset_loss()
```

`PrototypeEmbedding` gains **`sim_from_embed(o_embed)`** — the tail of today's `forward`
(cosine/membership similarity, regulariser accumulation, optional weight matrix) split
out into a method that `forward` then calls after `self.embedding_ext(...)`. Behaviour
of every existing path is unchanged (the dc01 t03 RNG/golden pin must still pass — the
split adds no parameter and no draw). `FeatureEmbeddingW` is left as is (single
consumer: none after this; kept for history).

**2. Factory branch** (`feature_extractor_factories.py`), cloned from
`prototypes_double_tie` with the two base tables swapped for the compositions:

```python
elif ft_type == 'feature_user_item_proto':
    up, ip = ft_ext_param['user_ft_ext_param'], ft_ext_param['item_ft_ext_param']
    K_u, K_t = up['n_prototypes'], ip['n_prototypes']
    assert not up['use_weight_matrix'] and not ip['use_weight_matrix']
    assert {'feature_ids', 'n_features'} <= ip.keys()                 # injected (dc01)
    assert {'hist_value_ids', 'hist_weights', 'n_features'} <= up.keys()  # injected (dc05)
    item_base = FeatureEmbedding(n_items, ip['feature_ids'], ip['n_features'], embedding_dim,
                                 use_id_feature=ip.get('use_id_feature', True),
                                 max_norm=ip.get('max_norm'), feature_weights=ip.get('feature_weights'),
                                 center_fields=ip.get('center_fields', False), ...)   # dc06 knobs default off
    user_base = HistoryFeatureEmbedding(n_users, up['hist_value_ids'], up['hist_weights'],
                                        up['n_features'], embedding_dim,
                                        use_id_feature=up.get('use_id_feature', True),
                                        max_norm=up.get('max_norm'),
                                        **_loo_kwargs(up), **_dc08_kwargs(up))      # dc08 rows default 0
    user_proto = PrototypeEmbedding(n_users, embedding_dim, n_prototypes=K_u, use_weight_matrix=False,
                                    sim_proto_weight=up['sim_proto_weight'], sim_batch_weight=up['sim_batch_weight'],
                                    reg_proto_type=up['reg_proto_type'], reg_batch_type=up['reg_batch_type'],
                                    cosine_type=up.get('cosine_type', 'shifted'), embedding_ext=user_base)
    item_proto = PrototypeEmbedding(n_items, embedding_dim, n_prototypes=K_t, ..., embedding_ext=item_base)
    user_fe = TwoRoleComposedExtractor(user_base, user_proto, out_dimension=K_t, invert=False,
                                       normalize_composed=ft_ext_param.get('normalize_composed', False))
    item_fe = TwoRoleComposedExtractor(item_base, item_proto, out_dimension=K_u, invert=True,
                                       normalize_composed=ft_ext_param.get('normalize_composed', False))
    user_base.init_parameters(); item_base.init_parameters()   # factory = single owner of both tables
    return user_fe, item_fe
```

(`RecSys.init_parameters` cascades one level: it reaches the wrappers' `init_parameters`
— the two linear layers — and the prototypes' `init_parameters`; the composed tables are
initialised exactly once here, the pinned dc01/dc05 convention.)

**3. Injection** (`feature_extraction/feature_ids.inject_feature_ids`): for this ft_type
inject **both** payloads — `feature_ids`/`n_features`/`feature_weights` (+ dc06 fields when
asked) into `item_ft_ext_param`, and `hist_value_ids`/`hist_weights`/`n_features` + the
leave-one-out payload into `user_ft_ext_param` — from the split directory's own train
file (dc05's leakage rule, binding on the variant datasets).

**4. `rec_sys/rec_sys.py`, `protomf_dataset.py`, `trainer.py`: zero changes.**
`RecSys.forward` already routes `pos_i_idxs` to the user extractor iff `supports_loo`;
the wrapper exposes it and passes it to the base, so **both** user roles see the
leave-one-out vector (the gap §2.5 flagged is closed by construction — one q).

**5. `confs/hyper_params.py`**: `feature_user_item_proto_hyper_params` = the
`proto_double_tie_chose_original_hyper_params` space verbatim (d ∈ [10,100), K_u, K_t ∈
[10,100), four reg weights loguniform(1e-3, 10), `max` reg types, shifted cosine, base
space) + per side `use_id_feature: True`, `feature_fields: 'canonical'`, user side
`loo_pooling: True`. `_noid` = both `use_id_feature: False`. Never a searched flag
(charter C7 / F-DC01-01).

**6. Explanation stack** (slots owed by M5; specified in §3.4): `breakdown.py` slots
`compute_breakdown_feature_user_item_proto` and `compute_breakdown_user_item_proto` (the
host slot deferred to this stage), accessor entries for both, intrinsic naming on both
sides via the existing `name_prototypes_intrinsic(side='user'|'item')` with each side's
own table, prototype cards per side, and the cold runner's ID-column drop on the item
table (one shared instance → one drop covers both roles, the dc01 t11 pattern). *(SC.2 amendment 2026-10-26 — F-DC09-01/08: the cold step reaches the composition through the wrapper's `base` (finder registration above); the cold **convention** is no longer the ID-column drop but mean-ID imputation — §3.4 "Cold and thin cases", S0.3 §5 amendment.)*

### 3.3 Forward pass with shapes (batch B, 1+n_neg candidates N, d, K_u, K_t, F fields, D_max basket width)

| step | tensor | shape |
|---|---|---|
| user basket lookup | `hist_value_ids[u_idxs]`, `hist_weights[u_idxs]` | (B, D_max) |
| user composed (LOO in train) | `q_u` | (B, d) |
| user cosine role | `u* = 1 + cos(q_u, P^u)` | (B, K_u) |
| user linear role | `û = W^u q_u` | (B, K_t) |
| user output | `[u*, û]` | (B, K_u + K_t) |
| item feature lookup | `feat_ids[i_idxs]` (+ID col) | (B, N, F+1) |
| item composed | `q_i = Σ_f e_f (+ e_ID)` | (B, N, d) |
| item cosine role | `t* = 1 + cos(q_i, P^t)` | (B, N, K_t) |
| item linear role | `t̂ = W^t q_i` | (B, N, K_u) |
| item output | `[t̂, t*]` | (B, N, K_u + K_t) |
| score (`RecSys` dot) | `u*·t̂ + û·t*` | (B, N) |

**Parameters:** E^I (V + M·[id])×d; E^U (V + N·[id])×d; P^u K_u×d; P^t K_t×d; W^u d×K_t;
W^t d×K_u. **Tied (R1):** within each side, one composed vector feeds both roles — the
host's exemplar, literally. *(A7 wording, 2026-10-26: "structurally" — the paper ties a per-entity *parameter* read twice; here a composition's *output* is read twice. I propose the structural reading; the identity with a re-pointed composition is with an object the paper never had.)* **Not tied:** across sides (host convention; the Seed B arm,
§3.7, sets E^U := E^I).

### 3.4 Intrinsic explanation read-out — the four-component inventory, all nameable

Every quantity below is computed in the forward pass; nothing is reconstructed.

1. **Item cosine unit t\*** (dc01 read-outs, verbatim): per-prototype activation; exact
   per-feature shares c_{f,k} = e^I_f·p^t_k/(‖q_i‖‖p^t_k‖) of t\*_k − 1 (jointly
   ‖q_i‖-normalised, F-DC01-06 grouping discipline); global archetype profile
   cos(e^I_f, p^t_k) for naming.
2. **User cosine unit u\*** (dc05 read-outs, verbatim): exact per-word **and** per-purchase
   shares of u\*_l − 1 (two regroupings of one sum, never added); community profile
   cos(e^U_f, p^u_l) in item vocabulary.
3. **Item linear unit t̂ — read-out 4 returns (F-DC01-07).** t̂ = W^t q_i = Σ_f W^t e^I_f +
   W^t e^I_ID(i): each coordinate t̂_l is an exact, **norm-free** sum over the item's rows,
   and the removal of row f changes the U-half by exactly u\*ᵀW^t e^I_f — a true
   counterfactual on this half, no off-manifold pass. *(SC.2, F-DC09-07: on the fixed-field layout the removal effect u\*ᵀW^t e^I_f is gauge-dependent (3b-xi); the identified counterfactual is **substitution within the field**, u\*ᵀW^t(e^I_{f'} − e^I_f) — catalogue-realisable ("Black instead of Dark Blue"); removal survives on ml-1m's bags layout.)* Units: user-prototype coordinates
   ("which of this item's words make it respond to community l").
4. **User linear unit û — the mirror of 3 (*I propose*).** û = W^u q_u =
   (1/|H_u|) Σ_j Σ_{f∈f_j} W^u e^U_f + W^u e^U_ID(u): exact per-word and per-purchase
   attribution of each coordinate û_k ("which of your purchases make you respond to
   archetype k"); removal of purchase j changes the I-half by exactly
   t\*ᵀW^u(Σ_{f∈f_j} e^U_f)/|H_u| (up to the 1/|H_u| renormalisation, disclosed).
   *(Corrected 2026-10-26 after 4b C2(b) — visible amendment: the per-purchase share
   s_j = W^u(Σ_{f∈f_j} e^U_f)/|H_u| is an exact SUMMAND of û; the removal counterfactual is a
   different exact quantity, Δ_j = (ā − a_j)/(|H_u|−1) with a_j = W^u Σ_{f∈f_j} e^U_f and ā the
   basket mean of the a's — it is zero when the purchase equals the basket mean and equals −s_j
   only when the other purchases cancel. Render the share; offer the counterfactual as a second
   intrinsic reading, disclosed as such. The cosine side's rule "shares are summands, not
   counterfactual effects" now has a linear-half twin; only the item-side word removal (read-out 3)
   is share = counterfactual, because q_i is a plain sum with no renormalisation.)*
5. **Intercepts.** B_item(i) = 1ᵀt̂ zoomable per word (§3.1) — the rendered baseline line
   becomes "item baseline +1.9: Trousers Denim +1.2, Dark Blue +0.3, item-specific +0.4"; *(SC.2, F-DC09-07: in ids arms these per-word numbers are gauge-dependent — rendered in the canonical per-field-centred gauge, where the item-specific line includes the field baselines.)*
   B_user(u) = 1ᵀû shown as the rank-inert line.
6. **Named/unnamed coverage, per component (dc08 A12 backport, now load-bearing).** The
   unnamed mass is the two ID rows' share across all four components; report the two-way
   split per component (not pooled — §3.6 M9), plus the item-ID and user-ID lines.
7. **Descriptive cross-side readings (enabled, not enforced — extrapolation caveat):**
   (a) route 1 (scratchpad 2026-07-11): W^u p^u_l projects community l into archetype
   coordinates; (b) **vocabulary-mediated affinity (*I propose*)**: both profile vectors
   live on the same word index, so corr_f(cos(e^U_f, p^u_l), cos(e^I_f, p^t_k)) compares a
   community with an archetype *through the shared vocabulary* even with separate tables —
   the parameter-free analogue of UIPC-MF's connection matrix; under the Seed B arm the
   direct cos(p^u_l, p^t_k) exists too. *(SC.2, F-DC09-11: under the shared default the direct matrix exists; it is rendered only if M18's one-off measurement shows the two prototype sets differ — otherwise the narrative restates one prototype twice.)*

*(SC.2 amendment 2026-10-26 — F-DC09-07, read-outs 1–6 in ids arms: every per-row quantity above — per-feature shares, per-word shares of t̂ and û, the intercept zoom, the ID lines, the per-component coverage, and the naming profiles cos(e_f, p) — is reported in the **canonical per-field-centred gauge** (3b-xi): each word row minus its field's vocabulary mean, the ID row plus the sum of the field means ("item-specific + field baselines"). Totals are unchanged; the centred numbers are gauge-invariant functions of the trained model; the uncentred ones are set by weight decay's tie-break. Noid arms are unaffected. The hm counterfactual is substitution within field; removal is offered on ml-1m's bags only. SC.5 implements the gauge in the naming/breakdown slots. F-DC09-09: whether a "common to all users" line is split out of the personalised bars — pure arithmetic — is decided at SC.5.)*

**Renderer slot (M5; against the live `Breakdown` dataclass).**
`compute_breakdown_feature_user_item_proto(model, user_id, item_id, items_info,
feature_fields, dataset_dir, naming_user, naming_item, model_label, feature_layout)`:
- `mechanism='proto'`; **bars panel** = top-6 personalised contributions pooled over both
  halves — U-half bars c_l(u)·t̂_l labelled "community l" (intrinsic user naming), I-half
  bars û_k·c_k(i) labelled "archetype k" (intrinsic item naming), signed;
- **two baseline lines**: B_item (`baseline_kind='item_intercept'`, wording "this item's
  own scalar, same for every user — of which: <top words>") and B_user (`rank_inert`).
  The dataclass holds one `baseline`; *I propose* an additive optional field
  `extra_baselines: List[BreakdownLine]` so existing slots and the self-check stay
  untouched (learnings: never reshape the shared frame; slot-owned strings stay short);
- **zoom** = the top bar's *cosine* factor (grounded side: per-purchase shares for a
  community bar, per-feature shares for an archetype bar); **alt_zoom** (md companion) =
  the same bar's *linear* factor (read-out 3 or 4: per-word shares of t̂_l or û_k); *(A6, 2026-10-26, after 5b-21/22: the linear-half zoom is labelled "exact for this half" — removing the word also moves t\*, so the full-score effect differs (4b C2(a) side note); the only recommendation-level counterfactual is a re-score with the row removed from q_i, both halves moving — offered as a disclosed second reading, item side only (the user side has none, C2(b)). The intercept zoom is labelled "baseline offset per word"; the popularity sentence is added only once the SC.8 B_item-vs-popularity correlation licenses it (dc05 B(t) precedent). Figure default: top-3 bars + "n more", the rest in the md companion.)*
- **footer** = per-component named coverage + the two ID lines;
- **self-check**: B_item + B_user + Σ(all personalised parts) == forward score; each zoom
  sums to its factor; U-half Σ_l c_l t̂_l + I-half Σ_k û_k c_k == S − B_item − B_user.
`compute_breakdown_user_item_proto` (host slot, same session): identical bar math on the
host's free u, t with both baseline lines; zooms = the host's post-hoc routes (nearest
items for archetypes, synthetic maximally-activating user per A1 §5.2 for communities),
declared post-hoc; coverage line absent (no named rows — the honest-asymmetry row).
Prototype cards: intrinsic word-profile bars on both sides; host cards post-hoc.

**Rendered example (H&M-style, canonical five; illustrative numbers):** *(Caption, A6 2026-10-26: value-level lines are illustrative; department/section/product-type are lockstep-prone and are read at clique level per F-DC01-06; individual per-prototype coefficients t̂_l, û_k are slab-projected representatives (3b-i).)* *(SC.2, F-DC09-07: per-word numbers in ids arms are read in the canonical per-field-centred gauge; "item-specific" then includes the field baselines.)*

> **Why "Slim-fit dark-blue jeans" for you?** Score 3.72 = item baseline 1.60 + your
> baseline 0.90 + personalised 1.22.
> *Item baseline +1.60* (this item's own scalar, same for every user): Trousers Denim
> +0.95 · Menswear +0.35 · Dark Blue +0.20 · Solid +0.05 · Jeans +0.02 · item-specific +0.03.
> *Your baseline +0.90* (same for every item you rank).
> *Top personalised contributors:*
> (a) taste community #7 *(Trousers Denim · Menswear · Dark colours)* — your activation
> c₇ = 0.42 × this item's response t̂₇ = 1.30 → **+0.55**. Your 0.42: 12 dark-denim
> purchases +0.31, 9 menswear basics +0.08, 3 swimwear −0.02, 13 others +0.01, your
> personal row +0.04. The item's 1.30: Trousers Denim +0.80, Dark Blue +0.30, Menswear
> +0.15, Jeans +0.03, Solid +0.00, item-specific +0.02.
> (b) item archetype #12 *(Trousers Men · Denim · Dark colours)* — this item's activation
> c₁₂ = 0.70 × your response û₁₂ = 0.45 → **+0.32**. The item's 0.70: Trousers Denim
> +0.45, Dark Blue +0.15, Menswear +0.07, Solid +0.02, Jeans +0.01, item-specific 0.00.
> Your 0.45: your 12 dark-denim purchases +0.33, menswear basics +0.10, others +0.01,
> your personal row +0.01.
> (c) … four more bars (+0.35 total).
> *Named coverage of the personalised score: 93 % (item-specific rows 4 %, your personal
> row 3 %).*

**Cold and thin cases.** *Cold item* (ID row dropped at inference, dc01 convention): q_i
exists from words → **both** t\* and t̂ exist, so a cold item is scored on both halves;
its t̂ lacks the warm ID row's length (the cold magnitude handicap — convicted in the
refund entry, now with an object). *Cold user* (empty basket, ID row dropped): q_u = 0 →
u\* ≡ 1, û = 0 → S = B_item(i) + 0 + 0 + 0: the fallback is exactly the live item
intercept, a popularity-shaped non-personalised ranking (dc05's floor, merge form).
*Thin user*: same explanation with fewer purchase lines; fold-in without retraining on
both sides (B5 §7.1).

*(SC.2 amendment 2026-10-26 — F-DC09-08, major, shared with S0.3 §5.)* **The drop convention is gauge-dependent in every ids arm.** After the drop, q_cold = Σ_f e_f carries the translation gauge's uncancelled Σ_φ v_φ (3b-xi): a common shift across cold items — rank-inert through t̂ on cold-vs-cold, but rank-active through the cosine unit (directions move) and on cold-vs-all. Two warm-identical models rank cold items differently; the landed convention scores cold items in L2's canonical gauge. **Invariant convention — mean-ID imputation.** Cold item: q_i = Σ_f e_f + ē^I_ID, with ē^I_ID the mean of the warm item-ID rows (it shifts by −Σ_φ v_φ exactly as the words shift by +Σ_φ v_φ — SC.2 round D7). Cold user *with* a basket (the dc05 fold-in case): q_u = basket mean + ē^U_ID. **Empty-basket cold user keeps q_u = 0** — there are no words, so nothing needs cancelling, and imputing ē^U_ID there would *break* invariance (D7c); the S = B_item(i) fallback and 4b C11a stand. Reading: "a new item is a typical item we know nothing specific about", instead of "an item with zero identity", which only means something in one gauge. Expected numerically small under L2 (ē_ID → small): the dc01-checkpoint instrument measures it first — gauge-shift a trained fI-ids checkpoint by the per-field correction, verify warm scores bit-identical, run the cold protocol under both conventions (login-node class). Landing: S0.3 §5 dated amendment (done at SC.2) + `cold_eval` change + dc_checks/s0 pin at SC.3, **before wave 2's thesis-tier cold retrains**; landed dev-tier cold rows disclosed, not re-run (user's call). The cold magnitude handicap above survives: ē_ID is short, so t̂ still lacks the warm row's length.

### 3.5 Feature fields & intrinsicness

Per-dataset canonical fields, unchanged (hm: the S0.5 five, V = 426, fixed layout; ml-1m:
genres + tags@0.8 bags, V = 1,061). *(SC.2 amendment 2026-10-26 — F-DC09-05: the testbeds differ in **feature provenance** — ml-1m's `tags@0.8` are Tag-Genome relevance ≥ 0.8, a score predicted from user tagging/rating behaviour (Vig, Sen & Riedl 2012; citation to verify at RW time), so ml-1m's content is partly behaviour-derived, while hm's taxonomy is static catalogue metadata. Fleet-wide (fI/fU/lightfm rows alike), disclosed in Ch. 3 Data.)* Both sides use the **same vocabulary index**; Seed A
gives each side its own rows. User demographics do not enter (candidate prime, vault
`2026-09-08_1649`, after the merge). Intrinsicness position = dc05 §3.5 verbatim: the
history is model input through a fixed, human-performable aggregation; every rendered
share is an exact summand of the inference-time score; e_f meanings stay CF-trained
(label confounding carries); taste stationarity is a declared limitation.

### 3.6 Interpretability-constraint design (v1.1 mandatory block)

**Inheritance.** All four inclusion regularisers survive byte-identically — they act on
the batch–prototype cosine matrix of each `PrototypeEmbedding`, agnostic to how q was
produced (dc01/dc05 argument, both sides at once). Semantic shifts already recorded for
fI (coverage "owned by some signature"; L2 on shared rows) and fU (coverage "owned by
some basket"; interaction-weighted user cloud, F-DC05-02) carry. **Two shifts are new
to the merge:** (a) the regularisers see only the *cosine* role — the linear role is
unconstrained except by L2 on W and on rows; (b) **row norms now carry score** (through
the other side's linear half), so the inherited `max_norm` and λ_L2 stop being pure
vote-weight equalisers and become **direct caps on the class-level popularity channel**
(vault `2026-08-15_1721` §1) — their semantics change even though the code does not. *(Softened 2026-10-26, A4 after 5b-14: because of the per-side scale gauge (3b-x), a cap on row norms is absorbed by W and binds only through L2 on W — `max_norm` constrains the channel's *parameterisation*, not its magnitude; the class-level popularity channel is bounded by λ_W (K8), not by `max_norm` alone.)*

**New degeneration modes** (beyond dc01's M1–M4, dc05's M1′–M6′, and the host-inherited
gauges of 3b, all of which carry):
- **M7 — norm-channel welding (direction contamination).** Popularity pressure on the
  linear halves inflates shared word-row norms; inflated rows drag the cosine directions
  of every carrying item/user toward popular words — popularity entering the grounded
  structure through the norm, with no bias term in sight. Instruments: popularity axis
  vs attribute axes on the same principal components (R²-style), per-field word-norm vs
  value frequency, and the **built-in 2×2** (host UI = identical tie, free embeddings:
  the control arm).
- **M8 — component migration (explanation-surface mode).** Score mass migrates among the
  four components — into the live intercept (M6′'s merge form), or into the dot-trained,
  Steck-exposed linear halves and away from the prototype-shaped cosine units — shrinking
  the narrated R3 part. Instrument: variance share of each component across (u,i) pairs
  + B_item's popularity correlation; fUfI and host UI alike. *(SC.2, F-DC09-09: the variance share is blind to common-mode mass — a per-item term shared by all users has large variance across pairs and zero personalisation. Kept, demoted; the one-off two-way decomposition of 3b-xii is the descriptive measurement.)*
- **M9 — role split of labour.** W may learn to read the ID subspace while the cosines
  read words: linear halves ID-dominated, cosine units word-dominated — a hidden division
  the pooled coverage would miss. Instrument: the per-component named/unnamed split.
- **M10 — two-row routing.** Two unnamed rows and two linear halves give idiosyncrasy four
  homes; the named fraction can collapse on one component first. Same instrument as M9,
  plus the ablation ladder (§3.7).
- **M11 — cross-side vocabulary divergence (Seed A specific).** E^U and E^I learn different
  geometries for the same word ("Dark Blue" as taste indicator vs item property);
  community and archetype profiles stop being comparable and the thesis's one-vocabulary
  story weakens. Instrument: orthogonal-Procrustes residual between E^U and E^I (the
  [captured dc05] comparison, now *intra*-model, on centred tables per the dc06 caveat).
  The Seed B arm removes the mode by construction. *(SC.2 amendment 2026-10-26 — F-DC09-03: the user's shared default removes M11 by construction; it is now the `_sep` arm's mode, and its replacement on shared rows is M15 below.)*
- **M12 — bag-size score channel (new; 3b-viii).** ‖q_u‖ shrinks as a basket spreads
  (F-DC05-01) and is now *read* by û = W^u q_u: thin, spiky baskets get larger linear
  halves than heavy, flat ones — a history-length-dependent score scale on the I-half.
  Instrument: ‖q_u‖, ‖û‖, and the I-half's score share vs |H_u| strata.
- **M13 — item-side bag-size channel (ml-1m; added 2026-10-26, A5 after 5b-13).** q_i is a *sum* over a variable-size tag bag (≤ 72 tokens, mean 24.3), so ‖q_i‖ grows with token count and B_item = (W^tᵀ1)·q_i and t̂ read it directly; tag count correlates with popularity → a rank-active popularity proxy with no semantic content, which the explanation would dutifully attribute token by token. Instruments: ‖q_i‖ vs token count vs interaction count; B_item vs token count. Unstacked knobs: the `normalize_composed` control; a mean-normalised item bag (composition change — own toy round); dc06 centring. Caveat to verify against the S0.5 record, not asserted: whether `tags@0.8` are Tag-Genome relevance thresholds (behaviour-derived content). hm is unaffected (fixed F = 5). *(Verified TRUE at SC.1a/SC.2, 2026-10-26 — F-DC09-05: Tag-Genome relevance ≥ 0.8 (S0-build ext §1; `build_item_features.py`); behaviour-derived content on the load-bearing R6 testbed — a fleet-wide disclosure, no mechanism change; see §3.5.)*
- **M14 — Steck exposure reaches the word rows (added 2026-10-26, A5 after 5b-12).** On fI/fU the word rows were shaped only through a cosine path; in the merge every e_f also receives dot-trained gradient through W. The naming profiles cos(e_f, p) are therefore read off rows trained half under a dot product — Steck's warning now applies to the naming layer, not only to the linear halves. *(Precision amendment at SC.2, 2026-10-26 — F-DC09-06: the doc was too harsh on itself. Steck's per-dimension-rescaling argument needs an objective invariant to diagonal rescaling of the embedding; the cosine units break that invariance (a diagonal D is absorbed by W → WD⁻¹ on the linear halves only), so the *arbitrariness* result does not transfer to the word rows. Honest statement: rows shaped by a mixed cosine/dot objective — semantic reading unvalidated, not provably arbitrary; F-DC01-02's trains-through-cosine argument still pins the geometry. Mitigation unchanged.)* Instrument: the profile-validity spot-check run on fUfI *and* on fI/fU over the same vocabulary (lift agreement as the comparison). Unstacked knob: K8 λ_W (bounds the dot-trained share of each row's gradient).
- **M15 — role conflict on shared rows (SC.2 amendment 2026-10-26 — F-DC09-03; shared default).** Each word row now receives four gradient sources: item cosine, item linear via W^t, user cosine via the basket mean, user linear via W^u — "Baby/Children" as evidence about the person and as a property of the thing must share one vector. The structural asymmetry is N : 1 (candidate slots vs one user slot per training row) on both datasets — mean normalisation gives each user's word part total weight F, so |H_u| does not dilute it. `_sep` is the control arm. Descriptive hypothesis, no instrument (the framing decision at the SC.1b gate: an algorithmic addition is described, not tuned): hm `_sep` gives sharper user profiles; ml-1m ties. Replaces M11, which the shared default removes by construction.
- **M7-shared — the popularity direction leaks into the user vectors (SC.2 amendment 2026-10-26 — F-DC09-03).** B_item's gradient on the positives' words lies along ŵ_t = W^tᵀ1/‖W^tᵀ1‖; under sharing that inflated component enters q_u directly through the basket mean and tilts every user vector toward popular-word directions (in `_sep` the user table was shaped by the user side alone). A second, smaller leak: B_user = w_uᵀq_u is rank-inert but **loss-active under BCE** (`bce` is in the searched loss set; under sampled softmax and BPR it cancels within the user — SC.2 round D11). Disclosure only; the shared-vs-`_sep` comparison is the measurement.
- **M16 — overlap / self-affinity channel (SC.2 amendment 2026-10-26 — F-DC09-10; shared default).** Writing the personalised part as q_uᵀM_{ui}q_i with M_{ui} = A/‖q_u‖ + Bᵀ/‖q_i‖ (A = P̂^uᵀW^t, B = P̂^tᵀW^u, 3b-iv) and expanding both compositions into word rows gives a sum over (basket word, candidate word) pairs; a word in **both** basket and candidate contributes w_{u,f}·e_fᵀM_{ui}e_f — a quadratic form in ONE row (bilinear in two independently decayed rows under `_sep`; D9). The model can raise every basket∩candidate overlap by growing ‖e_f‖, and M7's frequent-word norm inflation feeds it (that cross-link is the whole "frequency" reading — no separate instrument). **Twin channel:** basket twins of the positive (67.5 % of hm items have a word-identical twin) keep its exact word sum inside q_u^{LOO} — not leakage (the positive's own slot is removed) but a repeat-tuple channel, "you bought this exact kind before", which on hm may be the dominant learned pattern and is then an *accurate* explanation. §2b positions the collapsed shared form against FM/AFM/SVD++. One one-off measurement per dataset: the share of the personalised score carried by same-word pairs vs different-word pairs, and the fraction of training positives with a basket twin. `_sep` is the control.
- **M18 — prototype-set redundancy under one geometry (SC.2 amendment 2026-10-26 — F-DC09-11; shared default).** With composed users the user cloud is no longer a separate cloud: every q_u is a mean of item vectors (minus ID rows; D10), inside the item cloud's hull and pulled toward its middle (hm baskets of 2–10 items), and both inclusion regularisers fit their prototype set to clouds in one space. The two sets may settle on near-identical directions, in which case the cross-prototype read-out 7(b) cos(p^u_l, p^t_k) reads near-collinear and the "community vs archetype" narrative has no content. In ids arms the matrix is additionally contaminated by ID directions (87,069 ID rows vs 426 word rows in R^d — dc01's naming caveat, one-line pointer). Hypothesis with a test, not a prediction (averaged vs raw points, K_u ≠ K_t): one one-off measurement per dataset — the row-max distribution of the cross-prototype cosine matrix; `_sep` is the control. Read-out 7(b) carries a conditional caveat: shown only if the sets differ.

**Constraint candidates — designed, NOT stacked (any adoption is the user's mechanism
decision; claims would route through a 4b re-run):**
- **K7 → M7, dedicated popularity home.** Feature-composed bias b_i = Σ_f β_f + β_ID(i)
  (LightFM's own channel, "the dc01 bias-gap fix"), hook: `RecSys` bias path with the
  per-item scalar replaced by a composed one; cost V + M scalars. Caveat: the fleet's
  `use_bias=0` is an *evidenced* decision (vault `2026-08-17_1937`; bias-on neutral-to-
  negative on U/UI) — adopt only if M7's instruments read badly. Softer: within-field
  norm-variance penalty L_norm = Σ_field Var_f(‖e_f‖) on both tables (K2/K2′), or the
  inherited hard `max_norm`.
- **K8 → M8, component balance.** Separate weight decay λ_W on W^u, W^t (bounds the
  linear halves' scale — one optimizer param group, cost nil), or L_bal = Var over the
  four components' in-batch mean absolute mass. Rashomon-tunable.
- **K9 → M9/M10, ID routing.** K3/K3′ ID-row dropout on both sides (trains the cold
  operating point on both sides at once — double duty); or L_id = mean‖e_ID‖² per table.
- **K10 → M11, vocabulary alignment.** Hard: the Seed B shared table. Soft (*I propose*):
  L_align = ‖E^U_words − R·E^I_words‖²_F with R the current orthogonal Procrustes solution
  (recomputed per epoch, detached) — interpolates separate ↔ shared; cost V·d² per epoch. *(4b C14, 2026-10-26: zero iff the
  tables differ by an orthogonal map — reflections included, so a global sign flip costs
  nothing; not invariant to positive rescaling; gradient reaches both tables whenever L > 0.)*
- **K11 → M12, norm handover.** Block normalisation of the user composition before the
  linear role only (q_u/‖q_u‖·s with a learned or fixed scale s) — the length-ablation
  control's one-sided cousin; cost nil. Changes the I-half's attribution formula (shares
  acquire the 1/‖q_u‖ factor) — a mechanism change, own toy round if adopted.

**Rashomon-tunable knobs:** inherited λ_proto/λ_batch ×2 sides, λ_L2, `max_norm` (now
norm-channel-relevant), K_u, K_t (3b-vi); proposed λ_W, λ_norm, p_drop/λ_id, λ_align,
block scale s.

### 3.7 Baseline gate — baselines, the isolating ladder, the one knob, the controls

- **Headline baseline: `user_item_proto`** (UI host) — ml-1m n100 (May 21 run, accepted
  as host row), hm_1_month n100 (wave-1 job 8153730, queued). Like-for-like: same search
  space, 100 trials, seed 38210573. *(A15, 2026-10-26, after 5b-18: this comparison isolates the **composition package** — both compositions with their LOO hygiene and the ID-row repricing — not a single scalar cause; the ladder apportions by side, the ids/noid/mixed arms by ID rows, the instruments the rest. The C4 keystone proves plumbing equality at V = 0, not factor isolation.)*
- **Keystone reduction (M3):** `feature_user_item_proto` with zero words and ID rows on
  both sides ≡ `user_item_proto`, **bit-identity by weight copy** (dc01 i02 / dc05
  pattern; the wrapper's single-q path vs the host's double lookup must agree bitwise —
  a 4b claim).
- **Isolating ladder (*I propose*; each rung one config, the host search space):**
  host UI → **fI-on-UI** (item composed, user side F=0+ID — dc01's original UI build,
  re-derived here) → **fUfI** (both composed). The third rung's complement, **fU-on-UI**
  (user composed, item free), is a *user decision* — it completes a 2×2 of sides but is
  the kind of ablation the Nov-3 scope dislikes; recorded as optional. *(A2, 2026-10-26, after 5b-17: promoted — with fI-on-UI it forms the clean {free, composed}² factorial on the UI host that the contamination question (M7) actually needs; the lineage's U/I/UI/fI/fU set varies head and composition together and is not that factorial. Still the user's arm-count call.)*
- **ID ablation:** `_noid` (both rows off) vs ids — the C7 pair. *(A2, 2026-10-26, after 5b-4/6: the ml-1m n100 evidence runs the two sides' ID rows in **opposite** directions — fI ids > noid by +0.047, fU noid > ids by +0.047 — so the **mixed arm `_itemid_usernoid`** (item ID row on, user ID row off) is the evidence-favoured arm and is promoted from "optional" to the arm this candidate recommends for the user's decision; the symmetric mixed arm stays optional. Arm count remains the user's call.)* **User review decision 2026-10-26: ids-on-both and noid-on-both are the two ID arms (mirroring fI/fU); the mixed arm is optional only — the ml-1m user-ID-row risk is acknowledged and carried into the ids arm's reading.**
- **The merge knob — one word table or two.** *(User review decision 2026-10-26: **shared is the default** — E^U_words := E^I_words, both ID blocks stay separate; `_sep` (two tables) is the designed single-variable arm. Reasons given at review: the strongest R1 story (first cross-side tie), one geometry so user and item prototypes are directly comparable, M11 removed by construction; the risk — two roles forced into one geometry — is what the `_sep` arm measures.)* Original cycle text: E^U := E^I (one word table; both ID blocks
  stay separate). Default separate; shared as the designed single-variable ablation. The
  [captured dc05] geometry comparison on the n100 fI/fU checkpoints (ml-1m available now;
  hm after wave 1) is the **pre-build instrument** that may flip the default — this cycle
  runs nothing. *(Demoted 2026-10-26, A10 after 5b-10: a Procrustes residual between an fI table (d=78) and an fU table (d=92) trained in different models is ill-posed across dimensions and weak across hosts — at most a prior. The proper instrument is the intra-model residual between fUfI's own two tables (M11), and the one-table decision is settled by running the `_shared` arm.)*
- **S1 decoupled control (charter-amendment request, as dc05 did for `lightfm_hist`):**
  both compositions under a **plain dot product** — user `HistoryFeatureEmbedding` ×
  item `FeatureEmbedding`, no prototypes, no W (working name `lightfm_tags_hist`, ids
  variant `lightfm_tags_hist_ids`). This is LightFM proper (two-sided), and it is what
  makes an fUfI win attributable to the head rather than to two-sided composition.
- **Length-ablation control (user's design, scratchpad 2026-07-12; run-plan material):**
  `normalize_composed=True` on both wrappers — composed vectors forced to unit length
  where they feed both roles; exact no-op for the cosine units (keystone control), closes
  the length channel for the linear halves; trained normal-vs-constrained, never post-hoc.
  Isolates the refund/contamination channel inside one architecture. *(Sharpened 2026-10-26, A11 after 5b-17: it isolates that the length channel *matters* — it closes ID-row refund, class-level norm popularity, the ml-1m tag-count channel (M13) and the bag-size channel (M12) at once — not the refund-vs-contamination sign; the sign is read from the twin-separation and popularity-correlation instruments of 3b-iii.)* The fI/fU arms of
  the same flag are exact no-ops (harness controls, 4b claims).
- **Explanation baselines:** the host slot (post-hoc §5.2 routes, declared), the
  `lightfm_*` exact per-feature bars, dc01/dc05 renderings for the single-side stages.
- **Gap-interpretation rule, fUfI form (*I propose*, extending dc01 §3.6):** the
  ids-vs-noid gap bundles (1) item/user-level collaborative memory loss (twins now
  separable only via ID rows); (1b) the angular tax **is refunded in the linear halves**
  (twins separate via W(e_i − e_j)) — so (1b) should *shrink* vs fI/fU, a measurable
  prediction; (1c) **noid is no longer popularity-blind**: class-level popularity lives
  in word-row norms through the linear halves and in B_item — read the D4 strata against
  this; (2) taste-structure loss; (3) sharing offset. The fUfI-vs-UI gap is read against
  the fI-vs-I and fU-vs-U gaps (the lineage's three stage bets side by side) and against
  the 2×2 for the contamination sign.

### 3.8 Deviations from the source papers ("I propose", GR8)

1. Composed base embeddings under A1's double tie — in no paper (both sides at once).
2. Within-side tie by a shared composed instance computed once (`TwoRoleComposedExtractor`)
   instead of A1's weight re-pointing — mathematically identical; implementation choice.
3. No LightFM biases (fleet `use_bias=0`); B_item plays the composed-bias role implicitly
   and is disclosed per word.
4. History-derived user features, mean-normalised, train-time leave-one-out (dc05).
5. Exact attribution of the two linear halves (read-outs 3–4), the per-word intercept
   zoom, the per-component coverage split, and the vocabulary-mediated cross-side affinity
   — in no paper.
6. Separate word tables as default with the shared table as an arm — LightFM's
   convention made into a designed decision (B1 CMF as the sharing ancestor). *(Inverted at the user review 2026-10-26, recorded at SC.2 — F-DC09-04: **shared** word table as default, `_sep` as the arm — CMF's sharing convention is the designed decision; LightFM's two tables are the arm.)*
7. Cold ID drop on both tables; `normalize_composed` as a control flag — lineage conventions. *(SC.2, F-DC09-08: the cold convention is mean-ID imputation on both sides, not the drop — §3.4.)*

### 3.9 Testbeds and the two user decisions this cycle surfaces (not takes)

Both testbeds first-class: **ml-1m carries the load-bearing R6 comparison** (charter
amendment 2026-08-20), hm_1_month the feature-information/cold story. Two decisions sit
with the user at the "fUfI base-settings gate" (masterplan): (a) whether `hm_kcore` is
built for the merge (its collapsed warm space is the standing argument); (b) the table
default — separate (this candidate) or shared (Seed B) — to be informed by the geometry
instrument before the build. *(A10/A16, 2026-10-26: the geometry comparison is a weak prior only (5b-10); the decision is the `_shared` arm. Evaluation notes owed by 5b-16: the merge rows must report the charter's committed secondary read-out with popularity-sampled negatives (S0 charter C3, D3) beside the uniform 1+99 protocol, since the merge opens popularity channels; the evaluator's tie policy is pinned and disclosed (F-S0-15) and the noid twin tie is the known equivalence-class ceiling (dc01 §3.6).)*

---

## 3b. Hidden-effects pre-audit (M3 — dc05 record as template; run before Step 4)

*Instruments named here are scrutiny-stage commitments; scrutiny measurements are
working evidence only (learnings-ledger rule). Pre-registered merge effects (Frame 0.5)
are confirmed or sharpened here, not re-derived from scratch.*

**(i) Gauge freedoms — they move from the coefficient vectors into W.** On the single
hosts the free coefficient vector (u ∈ R^{K_t} in fI; t ∈ R^{K_u} in fU) carried a frozen
gauge component (F-DC01-08 and its dc05 mirror). On the UI host there is **no free
coefficient vector**: t̂ = W^t q_i and û = W^u q_u. The activation vectors still lie in
affine slabs — u\*(u) ∈ 1 + span(P̂^u) ⊂ R^{K_u}, t\*(i) ∈ 1 + span(P̂^t) ⊂ R^{K_t} — so
for any Δ ∈ R^{K_u×d} with 1ᵀΔ = 0 and P̂^uΔ = 0 (columns in the left null space of the
slab), W^t → W^t + Δ changes **no score of any pair** (u\*ᵀΔq_i ≡ 0); likewise W^u with
the item slab. Exact gauge dimension d·(K_u − d − 1) in W^t and d·(K_t − d − 1) in W^u
whenever K > d + 1 (reachable: K, d ∈ [10, 100)); the loss gradient is provably
orthogonal to these directions, so the gauge components of **W** are conserved at
initialisation to machine precision (or L2-decayed toward the slab). *(4b C6, 2026-10-26:
confirmed with the slab held fixed — invariance, gradient orthogonality 2e-16, conservation
6e-16, exact (1−lr·wd) decay, null-space dimension K−d−1. When the prototypes train the slab
rotates, and the conserved quantity is the component relative to the CURRENT slab (dc01 C9(b2)
reading); the verifier's 4.8e-2 drift of step-0 coordinates is that rotation, not a violation.)* **Touched rendered
quantities:** every per-prototype coefficient line t̂_l, û_k and their per-word zooms
W e_f (read-outs 3/4) — the slab projections are identified, the gauge parts are init
noise; **B_item = 1ᵀt̂ is NOT gauge** (1 is excluded), the intercept and its per-word zoom
are identified quantities. Host-inherited (UI's own W carry the same anatomy;
GR7-symmetric). Discipline as before: no gauge annotation on figures; thesis
hidden-effects section; instruments = activation-cloud spectra on both sides, trained-W
in-slab vs gauge-mass decomposition. Toy claims for 4b (C9-style, now on W). *(A4, 2026-10-26, after 5b-2/14 — three additions. (1) **Effective gauge:** C6's K−d−1 is the generic count; the realistic slab is spanned by the *effective* prototype directions — UI's measured ≈ 10 of 76 (vault `2026-08-17_1823`) would leave ≈ 65 of 76 coordinates of t̂ softly score-invisible. Instrument: slab spectrum of the trained model; the per-prototype lines of read-outs 3/4 are read as slab-projected representatives. (2) **Optimizer:** conservation (C6c/d) is a plain-SGD statement; under Adagrad/Adam the gauge component is NOT conserved — it stays score-inert, which is the identifiability-relevant property. (3) **Global scale gauge, per side (new, 3b-x):** (E^I, W^t) → (c·E^I, W^t/c) leaves every component unchanged (t\* scale-free in q_i; t̂ and B_item absorb c; the I-half never reads ‖q_i‖), likewise (E^U, W^u); only L2/`max_norm` break it. Consequences: absolute word-row norms are unidentified; every norm instrument is table-relative (normalise by the table's mean row norm or by ‖W‖); the "no per-entity dilation" statement in (iii) is true and orthogonal to this.)* *(SC.2 wording, F-DC09-02: the gauge parts of W are not "init noise" — they are optimiser-path-dependent and score-inert, decayed only by weight decay; Adam's preconditioner has a null-space component even when the raw gradient has none.)*

**(ii) Lockstep / identifiability — on both tables, and in the linear halves too.**
Catalog cliques (department↔section; F-DC01-06) on E^I, basket co-purchase lockstep on
E^U (dc05 3b-ii) carry verbatim; the linear halves inherit them unchanged — W e_a and
W e_b for a lockstep pair split the trained sum W(e_a + e_b) by frozen init noise.
Grouped shares at clique level on all four components; decoupling-mass proxy
(`dc_checks/dc01/decoupling_mass.py`) re-based per table — scrutiny basis only.

**(iii) Norm/angle — the length channel opens on both sides, two-signed (the merge's
central hidden effect).** Each side's ‖q‖ is discarded by its cosine unit and *read* by
the other side's linear half. Consequences, pre-registered and now with their object:
(a) **refund** — twins q = m + e_i, m + e_j separate in t̂ via W^t(e_i − e_j) with no ‖m‖
suppression (the angular tax is stage-local); (b) **contamination** — a shared row's norm
serves two masters (mixture weight in the cosine; class-level magnitude in the linear
half), and no parameter move changes an item's norm without changing its direction
(there is no per-entity dilation in the composed parameterisation; vault `2026-08-15`
§4). Which sign dominates is empirical. Instruments: twin-pair angle distributions (both
sides) **plus** twin-pair *linear-half* separation ‖t̂_i − t̂_j‖ (new — the refund's direct
measure), word-row norm vs value frequency and vs interaction popularity, direction-
contamination R² (M7), the 2×2 (host UI as free control), and the length-ablation
control. **Expressivity by arm:** noid is strictly less expressive than host UI (additive
span of V rows); ids is a reparameterisation of the host (equal function class,
differently priced — class-level signals cheap, idiosyncratic ones full price through
ID rows: partial pooling under weight decay).
*(Sharpened 2026-10-26 after 4b C10: the cosine-unit squashing of twins is O(1/‖m‖) and
monotone only once the shared word mass exceeds the ID-row scale — below it the twin
difference can first grow (6/8 toy seeds). The realistic regime is the monotone one (five
capped word rows vs one ID row), but the tax statement is a large-‖m‖ statement, not a
global one; the linear-half separation W^t(e_i − e_j) is exactly m-independent at every scale.)*

**(iv) Function-class honesty.** Algebraically,
S = (W^tᵀ1)·q_i + q̂_uᵀ(P̂^uᵀW^t) q_i + (W^uᵀ1)·q_u + q̂_iᵀ(P̂^tᵀW^u) q_u:
a **composed per-item linear bias** w_tᵀq_i (the live intercept, w_t = W^tᵀ1), a
rank-inert per-user term, and **two bilinear forms** of rank ≤ min(d, K) between one
side's *normalised* composition and the other side's *unnormalised* one (A = P̂^uᵀW^t,
B = P̂^tᵀW^u, both d×d). Generically for K_u, K_t ≥ d this is "feature-composed bias +
two-channel MF with one normalised leg per channel" — strictly richer than fI/fU (norm
channel present) and, in the ids arm, equal in function class to host UI. *(Corrected 2026-10-26, A9 after 5b-9: "strictly richer than fI/fU" is wrong for the noid arm — fUfI-noid has NO free table on either side, so it contains neither fI's class (free user) nor fU's (free item); it is an additive-span restriction with the norm channel open. Only the ids arm equals the host's class.)* What pins the
(P, W) factorisation: the four inclusion regularisers (their job), L2's implicit bias in
(P, W) coordinates, and dynamics — host precedent: UI ≠ MF empirically (A1 Table 2). fUfI
claims no expressiveness gain from the prototype layer; its value is that every term of
this collapsed form is now attributable to named rows.

**(v) Operating-point / cold mismatches and popularity homes.** *Cold item:* the ID drop
applies to one shared instance → both roles; t̂ loses the warm ID row's length (cold
magnitude handicap — first time it bites in the lineage); B_item for a cold item is
computed from words alone (a feature-composed popularity prior — LightFM's cold
mechanism, intact). *Cold user:* S = B_item(i) (3.4). *Train/cold mismatch* on both
sides (ID present in training, absent at cold scoring; K9 dropout is the unstacked
mitigation; B5 precedent tags+ids ≥ tags). *Popularity homes, enumerated:* (a) B_item
(per item, word-decomposable, rank-active); (b) ‖q_i‖ via t̂ magnitude — the ID row's
length, and class-level via word-row norms; (c) prototype geometry (user cone — the
cosbias2x2 lesson that the additive instruments are a **lower bound** on popularity
mass; report geometry indicators alongside). Registered expectations: (1) the fUfI-vs-UI
gap is smaller than the fU-vs-U gap on ml-1m if the fU head penalty was partly the
missing linear half (scratchpad 1b-27); (2) twin/popular-item recovery vs fI (refund);
(3) class-aligned popularity re-forms in word-row norms (contamination) — adjudicated by
(iii)'s instruments. *(A3, 2026-10-26, after 5b-7/8/19 — per-arm statement + disconfirmation clause, dc08 A11 precedent. Power: SE(HR@10) ≈ 0.006 on 6,034 ml-1m users (binomial), single seed by the Nov-3 scoping decision — every difference is read against that. Expectations by arm: fUfI-mixed and fUfI-noid > fU-noid (0.6082) and > fI-ids (0.5628); fUfI-ids > fU-ids (0.5613). *(SC.2 consistency amendment 2026-10-26 — F-DC09-04: arms restated for the user's 2×2 {ids, noid} × {shared, `_sep`} — fUfI-noid-shared / -noid-sep > fU-noid (0.6082) and > fI-ids (0.5628); fUfI-ids-shared / -ids-sep > fU-ids (0.5613); the mixed arm is optional only, and the ml-1m user-ID-row cost is read into the ids arms. Disconfirmation (a)(b)(c) unchanged. **SC.1b gate record — F-DC09-12, REJECTED by the user:** the adversary noted three inflations in the candidate's favour — "best arm" over the 2×2 adds ≈ 1σ of selection, the quoted SE is one row's binomial although the comparison is paired on the same users, and expectation (1) is not wired into the clause — plus the missing shared-vs-`_sep` expectation. The user declined a repair ("let's just run the tests and look at them and reason about them"): the clause stands as accepted; the arms are run, looked at and reasoned about at SC.8, where "best of four" is said when it is best of four.)* **Disconfirmation:** (a) if the best fUfI arm ≤ max(fI, fU) on ml-1m at n100, the merge adds nothing and the lineage's R6 story is negative; (b) if the best arm < UI-host row − 2·SE ≈ 0.612 HR@10, the merge fails the bar; (c) a win whose linear-half named coverage is below the cosine units' is M9 realised, not the intended win — reported as such. The bar is OUR UI-host row (0.6246, like-for-like by charter), not the paper's 0.657 (3 seeds, their tuning). Open risks named by 5b-8: ASHA may prune slow-starting pooled-row models (instrument: rung-survival / learning-curve comparison; guard: the profile's grace period); the fU optimum at d=92, K=98 sits near the search box's ceiling — report optima against bounds; widening the box breaks like-for-like.)*

**(vi) K multi-role, merge form.** K_u is: user-prototype count, **output dimension of
W^t** (coordinates of the item's linear half), and gauge dimension factor in W^t; K_t
mirrors. Unlike the single hosts, K is **no longer entity capacity** (both entities are
compositions in R^d) — the Rashomon coupling weakens on the capacity axis and reappears
on the projection axis. Recorded for the late-thesis Rashomon stage.

**(vii) Leave-one-out under the tie.** The training positive's words are removed from
q_u before **both** user roles (one q) — necessary: with LOO on the cosine role only, the
linear role would score the positive against a user vector containing the positive's own
words (train/eval mismatch on the I-half). Under the Seed B arm an extra coupling
appears: the positive's rows receive gradient both as item summands (q_i) and as basket
summands of every other buyer (q_u) — not leakage (the positive's own slot is removed),
but a shared-row coupling to instrument (word-row gradient norms by role).

**(viii) Bag-size channel becomes score-visible (new, M12).** dc05 recorded that
‖q_u‖'s metadata part shrinks as baskets spread (F-DC05-01; vault `2026-08-24_1229`,
pointer only) and that under the cosine this acted only through the metadata-vs-ID
balance. In the merge, û = W^u q_u reads ‖q_u‖ directly: thin/spiky users carry a
systematically larger I-half than heavy/flat users at equal direction. On ml-1m the
per-user token mass also varies 11–48.8 tokens per movie (dc05 §3.1 scope note), so
mass heterogeneity enters the score twice (direction and length). Instruments: ‖q_u‖,
‖û‖, I-half score share vs |H_u| and vs token-mass strata, fUfI vs host UI (whose free u
has no such coupling). *(A13, 2026-10-26, after 5b-15 — two more instruments and one expectation. **Heavy users (ml-1m):** q_u is the mean of thousands of token rows and shrinks toward the global mean (F-DC05-01), so the user-ID row can dominate both roles — expectation: fUfI on ml-1m behaves close to fI-on-UI + a user-ID row; instrument: ID share of ‖q_u‖ by |H_u| band (dc05 M5′ set), both roles. **Thin users (hm, 25 % at three rows):** *(SC.2, A13 precision — 3 TRAIN rows, the 5-core floor after LOO; q_u^LOO is never empty on V1)* LOO removes a third of the basket per training step, so q_u's direction swings between consecutive positives and û reads the swing at full norm; |H_u| = 1 noid users have q_u = 0 and train only through the intercept (C11a). *(does not occur on V1 — recorded for other splits)* Instrument: per-step angular change of q_u^LOO by |H_u| band; share of users scored by the intercept alone.)*

**(ix) Intercept pair (confirmed, with the zoom).** One live per-item intercept
B_item(i) = (W^tᵀ1)·q_i, one inert per-user B_user(u) — exactly the vault's prediction;
new here: the live one is attributable per word and per ID row, and its popularity
correlation plus its share of score variance are primary instruments (M6′ → M8).

**(x) Global scale gauge, per side (added 2026-10-26, A4 after 5b-14).** (E^I, W^t) → (c·E^I, W^t/c) and (E^U, W^u) → (c·E^U, W^u/c) leave every score component unchanged (the cosine units are scale-free in q, the linear units and intercepts absorb c, the other side never reads this side's ‖q‖); only L2 and `max_norm` break the symmetry. Hence absolute word-row norms are not identified — every norm-channel instrument (3b-iii, M7, M12, M13) is reported table-relative (normalised by the table's mean row norm or by ‖W‖), and `max_norm` constrains the channel's parameterisation, not its magnitude (§3.6, softened). Toy claim for a future round, not run this cycle: exact invariance of S and of all four components under the per-side rescaling.

*(SC.2 amendment 2026-10-26 — F-DC09-02: **false under the shared default.** Rescaling the shared word block rescales q_u's word part too, so û = W^u q_u changes unless W^u is rescaled as well; the per-side gauge survives only in `_sep`. What survives under sharing is the **joint** gauge — c on the word block and on both ID blocks, 1/c on both W — exact for S and all four components (SC.2 round D4). Consequence: the relative scale of a word row as read through W^t vs through W^u is an *identified* quantity under sharing (pure gauge in `_sep`); ‖W^t‖_F/‖W^u‖_F is recorded as the one descriptive number for it. **[Corrected below — F-DC09-13.]** **Centroid identity (companion, D3):** under sharing q_u^{words} = (1/|H_u|) Σ_{j∈H_u} (q_j − e^I_ID(j)) exactly, and in the noid arm q_u = mean_{j∈H_u} q_j — the user *is* the centroid of the purchased item vectors; LOO form: the mean over H_u∖{i}, on hm a mean of ≥ 2 item vectors (5-core, 3-train-row floor). The I-half ûᵀt\* is then FISM-through-words: the user's position among the items she bought, read against the item prototypes.)*

*(SC.2 correction 2026-10-26 — F-DC09-13, toy round D4(d), re-verified first-party: the "identified" sentence above holds in the **noid-shared** arm only. In the ids arms the deployed score function admits a cross-side rescaling T_c — word block and item-ID block ÷c, W^t ×c, W^u ÷c, user-ID row e^U_ID(u) → c·e^U_ID(u) + (c − 1/c)·m_u (m_u the full-basket word mean) — under which q_i → q_i/c and q_u → c·q_u: every full-basket score, all four components and every inclusion regulariser are unchanged while ‖W^t‖_F/‖W^u‖_F is multiplied by c². Only the training-time leave-one-out breaks it (residual (c − 1/c)(m_u − m_{u,−i}) = O(1/|H_u|)), so in ids arms the ratio is pinned by the LOO objective alone and is quoted for noid-shared only. **Disclosure:** the same map moves the user-side split between word part and user-ID row, so in every ids arm with a user-ID row (shared and `_sep`) that split is set by leave-one-out training alone — strongly on hm (3–10 train rows), weakly on ml-1m (|H_u| ≈ 95); on the U host (dc05 fU-ids) the freedom is per user, because q_u reaches the score only through the cosine unit. User-side per-word shares and the ID-share-by-history-band reading on ml-1m therefore partly reflect weight decay and the optimiser path (→ hidden-effects section, shared with dc05). Description only: no instrument, no read-out change (a scale freedom, untouched by 3b-xi's canonical gauge), no mechanism change; noid arms unaffected.)*

**(xi) Translation gauge of the ids arms — a read-out gauge, not a knob (SC.2 amendment 2026-10-26 — F-DC09-07, major).** On the fixed-field layout (hm: exactly one word per field, F = 5) the map e_f ← e_f + v_{φ(f)} for every word f of field φ, with e^I_ID(i) ← e^I_ID(i) − Σ_φ v_φ and e^U_ID(u) ← e^U_ID(u) − Σ_φ v_φ, leaves every q_i and every q_u — full and LOO, as long as |H_u∖{i}| ≥ 1, true on V1 — unchanged, hence every score, every loss and every inclusion regulariser; only L2/`max_norm` break it (dimension F·d = 5d *[SC.2 precision 2026-10-26, toy D5: under the shared default; 2F·d under `_sep`, where each table carries its own shift]*). Verified D5. On ml-1m (bags of varying size) the map is exact at inference with bag-count-corrected ID shifts and broken in training by LOO's varying bag mass — a near-symmetry, pinned by the data as well as by weight decay. **Read-out consequence:** in any ids arm every per-row share of all four components, the per-word intercept zoom, the ID line, every between-field comparison, the named-coverage ratio and the naming profiles cos(e_f, p) are set by the regulariser's tie-break, not by the data. **Identified:** within-field contrasts e_{f'} − e_f, and totals. The lineage knew the symmetry (dc06's canonical within-field gauge — a *composition* knob, rejected as a training mechanism at the fI′ cycle) but never drew the read-out consequence; dc01-ids and dc05-ids rendered shares on hm carry it (M7 trickle-back → the hidden-effects section). **Fix, read-out time only** (no training change; the user's "uncentred compositions" scoping stands): all per-row read-outs, coverage and naming in ids arms are reported in the **per-field-centred canonical gauge** — e_f − mean_φ over the field's vocabulary rows (the vocabulary average is the stated convention; the frequency-weighted average is an equally invariant alternative with different numbers), ID line = e_ID + Σ_φ mean_φ, read "item-specific + field baselines". Centred quantities are gauge-invariant functions of the trained model; totals unchanged (D6). Noid arms are already identified. The rendered counterfactual on fixed-field data becomes **substitution within field** — u\*ᵀW^t(e_{f'} − e_f), exact on the linear half, catalogue-realisable, identified ("Black instead of Dark Blue"); removal survives on the bags layout. Thesis footprint: one paragraph in the explanations chapter; one hidden-effects entry shared with dc01/dc05.

**(xii) Common-mode mass inside the personalised halves (SC.2 amendment 2026-10-26 — F-DC09-09, descriptive).** With q_u = μ + δ_u + e^U_ID(u) — μ the basket-mean centroid toward which heavy users shrink (F-DC05-01) — the terms (W^uμ)ᵀt\*(i) and u\*(μ)ᵀt̂(i) are per-item scalars shared by every user: on ml-1m three per-item scalars, not one (§3.1 amended). Rank-active and popularity-capable — one-line disclosure, no mitigation; read-out 4 would narrate them as personal taste. M8's variance share is blind to it (large variance across items, none across users). One one-off descriptive measurement per dataset: the two-way decomposition (item main / user main / interaction) of each score component on a sampled user×item grid, reporting the user-independent share of each personalised half (D8 shows the shape at toy scale). Descriptive hypothesis: on ml-1m the common-mode share exceeds the ID-row share. *(SC.2 precision 2026-10-26, toy D8: the common-mode share approaches the whole personalised half only when the entire per-user part δ_u + e^U_ID(u) is small — in ids arms the user-ID row stays personal even for heavy users, which is exactly the contrast this hypothesis reads.)* Renderer consequence — a "common to all users" line, pure arithmetic — decided at SC.5.
---

## 4. Cost & feasibility

All estimates **Fermi-style**. Anchors read on 2026-10-26 from the 100-trial result dirs
on LEO5 (`/scratch/c7031336/protomf_results/*_ml-1m_s38210573_n100/summary.md`,
`hardware_summary.json`; working evidence per the learnings-ledger rule):

| run (ml-1m, n100, ASHA, seed 38210573) | best d / K | wall | concurrency | peak VRAM | avg GPU % |
|---|---|---|---|---|---|
| `user_item_proto` (host UI) | d=73, K_u=76, K_t=11 | 6h10m | 5 | 12.5 GB | 96 |
| `feature_item_proto` (fI) | d=78, K_t=77 | 11h53m | 8 | 21.1 GB | 99.6 |
| `feature_user_proto` (fU) | d=92, K_u=98 | 7h59m | 5 | 4.8 GB | 56 |
| `user_proto` (U host) | d=72, K_u=88 | 9h07m | 5 | 2.9 GB | 24 |

hm_1_month 100-trial walls are not yet measured (wave 1 queued); the wave-1 sizing
estimates are UI host ≈ 17 h, fI ≈ 13 h, fU ≈ 10 h (`Master/temp/wave1_submissions_2026-10-26.md`).
Dataset facts: hm_1_month N = 73,418, M = 13,651, V = 426, F = 5 fixed, user D_max = 112;
ml-1m N = 6,034, M = 3,125, V = 1,061, item bags ≤ 72 tokens, user D_max = 1,022 (dc05 §4).

### 4.1 Parameter delta (analytic; vs CF-only `user_item_proto` at equal d, K_u, K_t)

Host UI: (N + M)·d + (K_u + K_t)·d + d·(K_u + K_t) [W^u: d×K_t, W^t: d×K_u].
fUfI replaces the two entity tables by (V + M·[id])·d and (V + N·[id])·d; prototypes and
W unchanged.

- **ids arm: Δ = +2V·d** — hm: +852·d ≈ **+62 K at d=73** (+1.0 % of the host's 6.36 M);
  ml-1m: +2,122·d ≈ **+155 K at d=73** (+23 % of the host's 0.67 M — ml-1m's vocabulary is
  large relative to its N + M = 9,159; the words are the model there).
- **noid arm:** entity tables (N + M)·d → 2V·d — hm 6.36 M → 62 K (≈ **100× shrink**);
  ml-1m 0.67 M → 0.155 M (≈ 4.3×). The B5 §2.3-1 generalisation argument, two-sided.
- **`_shared` arm (Seed B):** one word table → Δ = +V·d vs host (hm +31 K; ml-1m +77 K).
- Non-persistent buffers (rebuilt at build, never checkpointed): item `feature_ids` M×(F or
  D_item) — tiny; user `hist_value_ids`/`hist_weights` N×D_max — hm 99 MB, ml-1m 74 MB
  (dc05 §4); leave-one-out payload M×D_item — tiny. No new buffer kind.

### 4.2 Training-time expectation

Per step the host does: two lookups, two cosines (B·K_u·d and B·N·K_t·d), two
projections (B·d·K_t and B·N·d·K_u), one dot. fUfI adds the two compositions:
- item: (F+1) = 6 gathers + sum on hm (negligible, dc01 §F: ≈1.05–1.1×); on ml-1m the
  (B, N, ≤72, d) bag gather — **fI's dominant cost and the reason for fI's 21 GB peak**;
- user: (B, D_max, d) gather + weighted sum — no N factor; on ml-1m D_max = 1,022 makes it
  ≈10× the user cosine's work (1,022·92 vs 98·92 mult-adds per row) but still far below
  the item side's N-axis tensors; dc08 measured the fU step at 3.1 ms on an A30.
So **fUfI step ≈ fI step + fU user composition ≈ 1.1–1.3× fI's**, i.e. ≈ 2× the host UI's
step (the host's 6h10m vs fI's 11h53m already shows the composition's cost on ml-1m).
Convergence: W^u, W^t are host parameters (no new cold-start of a layer); word rows pool
gradients across entities (statistical efficiency, B5 §2.3-1); no argument for slower
epochs — unknown, not a risk.

**Memory flag (real, with mitigation).** fI peaked at 21.1 GB at concurrency 8 on ml-1m
(bags). fUfI's per-trial footprint ≈ fI's + fU's → at concurrency 8 it would exceed an
A30's 24 GB. **Mitigation: concurrency 5** (≈ 5 × ~3.5 GB ≈ 18 GB, fits A30; A40/A100 give
headroom), and/or the `pooling='auto'` flat `embedding_bag` on the user side (dc08 A10,
already built: 1,198 → 80 MB per trial on ml-1m). The item-side bag gather has no flat
path today — if memory binds, that is the first optimisation (a `FeatureEmbedding`
embedding_bag mode mirroring `HistoryFeatureEmbedding`'s; build-time option, not design).

### 4.3 Wall-clock, Fermi

- **ml-1m, 100 trials, concurrency 5:** fI at concurrency 8 took 11.9 h at 99.6 % GPU (so
  concurrency buys little); fUfI ≈ 1.2× fI ≈ **14–18 h**; host UI needed 6 h.
- **hm_1_month, 100 trials:** wave-1 estimates UI 17 h, fI 13 h, fU 10 h → fUfI ≈ **20–26 h**
  (the item side is cheap on hm; the user side's D_max = 112 is small; the double tie's two
  projections and two cosines dominate as in the host).
- Each 100-trial run is **≈ one day of wall-clock** on one GPU, inside the fleet's
  2-day walltime class. Cold single-config retrains: minutes–an hour each.

### 4.4 S5 pipeline-compatibility checklist

- Sampled `(batch, 1+n_neg)` scoring preserved? **Yes** — item branch indexed by `i_idxs`
  (B, N), user branch by `u_idxs` (B,); the wrapper only concatenates.
- No full-catalog pass per step? **Yes** — inclusion regularisers stay in-batch inside the
  two `PrototypeEmbedding`s (`sim_from_embed` keeps the same accumulators).
- Branches cheap indexed computation? **Yes** — gathers + sums + two small matmuls; no
  encoder; the composed vector is computed once per side per step.
- Losses fit `get_and_reset_loss()`? **Yes** — the wrapper forwards its prototype module's
  accumulator; `RecSys` sums user + item as today.
- Anything expensive per entity → precompute? **No** — both builders run once at startup
  (seconds, CPU) from the split's own train file; buffers non-persistent.

### 4.5 Budget gate (compute rules: cluster-only; > 10 days hard-infeasible; > ~1 day soft for dev)

- **Default plan** (ids + noid × ml-1m + hm_1_month) = **4 runs × ≈ 1 day** of GPU time,
  one SLURM job each (one run = one job), parallelisable → wall bounded by queue, not
  compute. Within every limit. Dev-profile (30-trial) smokes ≈ 5–8 h each.
- **Full designed set** (+ `_sep` ×2 *(SC.2: shared is the default, `_sep` the arm)*, fI-on-UI ×2, `lightfm_tags_hist` ×2,
  length-ablation ×2) = 12 runs ≈ 12 GPU-days — feasible but **scope-flagged**: the Nov-3
  plan dislikes many ablations and the hm fleet is still queued; arm selection is the
  user's (§3.7). Nothing here breaches a hard limit.
- **Schedule flag (not compute):** wave 1 cannot start before ~2026-10-28; fUfI runs queue
  behind it on the `feat/fufi` worktree with the cluster-checkout lock in force. A 100-trial
  fUfI row per dataset is realistic within the window only if the build lands early in it.
- **Keystone + toy checks:** CPU, seconds; login-node A30 smokes per the standing policy.

**Minimum attributable set and priority under scope cuts (A2, 2026-10-26, after 5b-5):** per dataset, (1) UI host + fUfI best arm + `lightfm_tags_hist` — without the third row an fUfI−UI delta attributes to nothing; (2) the ids/noid pair (mixed optional — user review); (3) `_sep`; (4) the fI-on-UI / fU-on-UI factorial; (5) length-ablation. A cut that drops (1)'s control should be refused rather than reported as attributable.

**Verdict: feasible; two flags** — VRAM at high concurrency (mitigated by concurrency 5 /
flat pooling) and arm-count vs the Nov-3 window (user's scoping call). No veto. *(SC.2, F-DC09-04 — S5 build-cost sentence: the shared default adds a one-off build cost — the external word block in both compositions with a bit-identity guard on the single-table path, the sharing keystone t07, the cold finders/registration and the imputation convention (F-DC09-08) — all CPU-test class; no run-time cost.)*

---

## 4b. Math sanity check

### 4b.1 Claims spec (written before execution; the subagent receives ONLY this file)

`Master/temp/dc_checks/dc09/claims_spec.md` — mechanism definition + fourteen claims, no
derivations, no expected outcomes. In brief: **C1** exact four-component decomposition,
1ᵀû rank-inert / 1ᵀt̂ rank-active; **C2** exact linear-half attribution on both sides
(per-word removal = share on the U-half; per-purchase share ≠ removal effect in general
on the I-half); **C3** the fI/fU cosine-share identities survive inside the merged score;
**C4 keystone** fUfI(V=0, +ID both) ≡ `user_item_proto`, scores bit-identical and
gradients equal between the single-q wrapper and the host's double lookup; **C5**
single-side reductions; **C6** gauge freedom in W (invariance, gradient orthogonality,
conservation under SGD, exact (1−lr·wd) decay, null-space dimension K−d−1); **C7** the
live intercept decomposes per row and is not gauge; **C8** LOO reaches both roles (and
what a one-role LOO would get wrong); **C9** normalisation controls (cosine units
invariant; exact no-op incl. gradients on linear-free hosts; one-sided block
normalisation); **C10** twin separation squashed in the cosine unit, exactly m-independent
in the linear unit; **C11** cold-user and cold-item degenerate paths; **C12** bag-size
channel trend; **C13** collapsed function class identity; **C14** K10 alignment penalty
properties.

### 4b.2 Viability self-assessment (mandatory, written before execution)

C1–C11, C13, C14 are algebraic / gradient / limit-case properties — all meaningfully
toy-testable at (N, M, V, d, K) ≈ (6, 8, 5, 3, 8). C12 is a statistical trend, testable as a
simulated demonstration only (not an identity). **Skipped as untestable at toy scale,
deliberately** (carried into Step 5 as empirical bets): (i) which sign of the norm channel
dominates — refund vs contamination (3b-iii / M7) — pure training dynamics on real data,
adjudicated by the 2×2 and the length-ablation control; (ii) component migration (M8) and
role split (M9) magnitudes; (iii) cross-side vocabulary divergence (M11) — needs trained
tables; (iv) the R6 bets (fUfI vs UI; fUfI−UI gap vs fU−U gap); (v) whether profiles come
out sharp or diffuse on both sides (M1/M1′). C6(e)'s dimension count is exact only for
generic P; degenerate prototypes (duplicates) enlarge the null space — the toy uses random
P and the claim is stated for the generic case.

### 4b.3 Subagent report (verbatim)

Subagent hand-back headline (verbatim): *"Black-box verification of the dc09 claims spec is
done. … Runtime 3.1 s. Headline: 11 claims confirmed outright; three carry a falsified
sub-clause — C2(b)'s stated equality condition is wrong (when a purchase equals the basket
mean, removing it changes û by exactly 0, not −s_j; equality holds iff the OTHER purchases
cancel), C10's 'decreasing over a sweep' is non-monotone while ‖m‖ is below the ID-row scale
(6/8 seeds; the limit → 0 and O(1/‖m‖) tail hold), and C14(c)'s literal 'c ≠ 1' fails at
c = −1 (−I is orthogonal, L = 0)."* The subagent read only `claims_spec.md`. Script:
`Master/temp/dc_checks/dc09/check_claims.py`; report: `Master/temp/dc_checks/dc09/report.md`,
reproduced verbatim below (blockquoted).

> # dc09 claims check -- independent black-box toy verification (C1-C14)
>
> - Script: `/Users/glaes/Desktop/ProtoMF-fufi/Master/temp/dc_checks/dc09/check_claims.py` (mechanism implemented from `claims_spec.md` alone; no other repository file read)
> - Run command (from this directory): `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate protomf && python check_claims.py`
> - Environment: torch 2.5.1, numpy 2.4.2, CPU, float64 default (float32 where C4 names it)
> - Toy sizes: 4 shape configs (V 5-8, M 8-10, N 6-8, d 3-5, K_u 4-12, K_t 5-9; K_u<=d+1 and K_t=d+1 included), 6 seeds, 5 parameter scale sets (1e-3 ... 1e3, mixed), basket sizes 1-6 with forced |H|=1 users, fixed and variable word-set sizes
> - Runtime: 3.1 s
>
> ## Verdict summary
>
> | Claim | Verdict |
> |---|---|
> | C1 | CONFIRMED |
> | C2 | FALSIFIED (final clause of (b) only; all identities and the 'not in general equal' statement CONFIRMED) |
> | C3 | CONFIRMED |
> | C4 | CONFIRMED |
> | C5 | CONFIRMED |
> | C6 | CONFIRMED |
> | C7 | CONFIRMED |
> | C8 | CONFIRMED |
> | C9 | CONFIRMED |
> | C10 | CONFIRMED (tends to 0; linear-unit difference exactly independent of m); the parenthetical 'decreasing in ||m|| over a sweep' FALSIFIED as a global statement -- non-monotone at small ||m|| in 6/8 seeds, monotone once ||m|| >= ID-row scale |
> | C11 | CONFIRMED (S = 1^T t_hat to machine precision, 1e-16 relative; not bitwise through the matmul path) |
> | C12 | CONFIRMED |
> | C13 | CONFIRMED |
> | C14 | CONFIRMED for c > 0 (a, b, c); literal (c) wording FALSIFIED at c = -1 (sign flip is orthogonal) |
>
> ## Interpretations made (spec ambiguities resolved by the verifier)
>
> 1. Sampled-softmax training step: one positive per user (drawn from H_u), n_neg = 4 uniform negatives, the LOO user
>    vector q_u^{LOO}(positive) scores the positive AND its negatives; loss = mean cross-entropy, target index 0.
> 2. C2(a) "changes the score by exactly -u*^T W_t E_I[f]" is read, per its own dash clause, as the change of the
>    u*^T t_hat half (q_u fixed, u* fixed). The change of the full S is also reported (it differs, since t* moves).
> 3. C5 "V_items = 0": the item table has no word rows (E_I in R^{M x d}); the users still compose from the items'
>    word sets through E_U's V word rows (and symmetrically for V_users = 0). The comparison model looks the free table
>    up twice (host style) and composes the other side once.
> 4. C6 A_u = [1^T ; P_hat_u] in R^{(d+1) x K_u} is read as [1^T ; P_hat_u^T] (P_hat_u is K_u x d). For (c)/(d) the
>    null space is the one of A_u at step 0, so P_u is frozen while W_t, W_u, P_t, E_I, E_U train (if P_u also trains
>    the null space rotates; that drift is reported separately as informative).
> 5. C9(b) "free t" / "free u": a free per-item table in R^{M x K_u} / per-user table in R^{N x K_t}; both the
>    user-cosine and the item-cosine variants are tested, with the sampled-softmax loss and LOO.
> 6. C10: twin items share word set {0,1,2} (F = 3) with m = c * m0 swept over c in [1e-3, 1e3]; e_i, e_j fixed.
> 7. C11(a): the cold user is user 0 with an emptied basket in the noid-user arm (q_u = 0 exactly by construction).
> 8. C14: R in O(d) (reflections allowed, as the SVD Procrustes minimiser U V^T yields) acting on the embedding
>    dimension row-wise: L = sum_f ||E_U[f] - R E_I[f]||^2 = ||E_U - E_I R^T||_F^2.
> 9. "Machine precision" verdict threshold: max relative error < 1e-11 (float64); "float tolerance" for cosine
>    invariances: < 1e-9 abs; bit identity: torch.equal. Scale sets mix 1e-3 and 1e3 across tables, randn gives negatives.
> 10. Invariance claims (C6a, C7) are measured against the rounding floor of the invariant quantity (the magnitude of the
>     terms being summed, e.g. sum_k |u_hat_k||t*_k|), because with ||Delta|| = 1e-6 ||W|| the only observable change is
>     S's own float64 rounding; the deviation relative to the would-be change ||Delta|| ||q|| is reported alongside.
> 11. Ranking comparisons (C1) ignore pairs tied to within 1e-9 max|S| (twin items with identical word sets in the noid
>     arm have scores that differ only at the ULP level and can tie after a constant shift).
> 12. C12 uses a paired design (nested baskets of one random item order per trial, 20000 trials) so that the trend is
>     not masked by Monte-Carlo noise; the claim is statistical, so sample means and their standard errors are reported.
>
> ## Per-claim measurements
>
> ## C1 -- four-component decomposition; intercept kinds
> - instances: 360 (4 configs x 6 seeds x 5 scale sets x 3 id-arms)
> - decomposition S = 1^T t_hat + sum_l (u*_l-1) t_hat_l + 1^T u_hat + sum_k u_hat_k (t*_k-1): max rel err 5.342e-16
> - 1^T u_hat rank-inert: pairwise item-order mismatches between S and S - 1^T u_hat = 0 (over all users/instances; 745 exactly/near-tied pairs -- twin items with identical word sets in the noid arm, |dS| <= 1e-9 max|S| -- excluded from all order comparisons)
> - 1^T t_hat varies across items: min over instances of (max_i - min_i) 1^T t_hat_i / max_i |1^T t_hat_i| = 6.838e-01
> - removing 1^T t_hat changes item rankings: instances with >=1 user whose ranking changes = 216/360; users with changed ranking per instance min/max = 0/8; total flipped pairs = 17945
> - **Verdict: CONFIRMED**
>
> ## C2 -- exact linear-half attribution, both sides
> - (a) t_hat = sum_f W_t E_I[f] + W_t E_I[V+i]: max rel err 3.684e-16
> - (a) removal of word f from q_i: change of the u*^T t_hat half vs -u*^T W_t E_I[f]: max rel err 1.256e-14 over 1960 removals x all users
> - (a) side note: change of the FULL score S vs -u*^T W_t E_I[f]: relative deviation min/max = 1.351e-10/3.674e+08 (the cosine half also moves, so the per-word share is exact for the linear half only)
> - (b) u_hat = (1/|H|) sum_j sum_f W_u E_U[f] + W_u E_U[V+u]: max rel err 5.031e-16
> - (b) per-purchase shares s_j sum to the word part of u_hat: max rel err 5.691e-16
> - (b) counterfactual removal of purchase j (re-mean over |H|-1) vs -s_j: ||delta + s_j||/||s_j|| min/max = 1.200e-01/4.843e+00 over 1960 (u,j) pairs -> NOT equal in general (confirmed)
> - (b) derived closed form delta = (a_bar - a_j)/(|H|-1) with a_j = W_u sum_f E_U[f], a_bar = basket mean: max rel err 7.825e-16
> - (b) equality condition test, basket {A,B,C} with single disjoint words, word sums (m-dl, m, m+dl) so a_B == basket mean word sum:
>     ||delta_B|| = 6.206e-17   ||s_B|| = 4.023e-02   ||delta_B + s_B|| = 4.023e-02   | alt. construction (others cancel, a_B != mean by 3.136e-02): ||delta_B + s_B|| = 1.289e-16
>     ||delta_B|| = 0.000e+00   ||s_B|| = 1.870e+00   ||delta_B + s_B|| = 1.870e+00   | alt. construction (others cancel, a_B != mean by 1.393e+00): ||delta_B + s_B|| = 1.579e-15
>     ||delta_B|| = 0.000e+00   ||s_B|| = 1.725e+01   ||delta_B + s_B|| = 1.725e+01   | alt. construction (others cancel, a_B != mean by 1.407e+01): ||delta_B + s_B|| = 2.275e-15
>     ||delta_B|| = 5.684e-14   ||s_B|| = 1.699e+02   ||delta_B + s_B|| = 1.699e+02   | alt. construction (others cancel, a_B != mean by 1.323e+02): ||delta_B + s_B|| = 2.562e-14
>     -> when a_j equals the basket mean, removing j leaves the mean unchanged (delta = 0) and delta != -s_j; equality delta = -s_j holds instead iff W_u a_j = W_u * (basket TOTAL word sum), i.e. the other purchases cancel.
> - **Verdict: FALSIFIED (final clause of (b) only; all identities and the 'not in general equal' statement CONFIRMED)**
>
> ## C3 -- cosine-unit exact shares inside the merged score
> - t*_k - 1 = sum_r E_I[r].P_t[k]/(||q_i|| ||P_t[k]||) (rows = words + ID): max abs err 6.661e-16
> - u*_l - 1 = sum_r w_r E_U[r].P_u[l]/(||q_u|| ||P_u[l]||), w = count/|H| (words), 1 (ID): max abs err 1.110e-15 (max word count within a basket = 4)
> - per-purchase regrouping == per-word grouping: max rel err 3.357e-16
> - **Verdict: CONFIRMED**
>
> ## C4 -- keystone reduction to the host (V = 0, ID rows both sides)
> - (a) scores bit-identical (float64): torch.equal on all 120 instances = True; max abs diff = 0.000e+00
> - (a) scores bit-identical (float32): torch.equal on all 120 instances = True; max abs diff = 0.000e+00
> - (b) sampled-softmax loss values torch.equal (float64): True
> - (b) grad max rel err E_I~T: 0.000e+00
> - (b) grad max rel err E_U~U: 0.000e+00
> - (b) grad max rel err P_u~P_u: 0.000e+00
> - (b) grad max rel err P_t~P_t: 0.000e+00
> - (b) grad max rel err W_u~W_u: 0.000e+00
> - (b) grad max rel err W_t~W_t: 0.000e+00
> - **Verdict: CONFIRMED**
>
> ## C5 -- single-side reductions
> - item side reduced (V_items=0, ID on) vs free item table looked up twice + composed users: torch.equal on all 120 instances = True; max abs diff 0.000e+00
> - user side reduced (V_users=0, ID on) vs free user table looked up twice + composed items: torch.equal on all 120 instances = True; max abs diff 0.000e+00
> - **Verdict: CONFIRMED**
>
> ## C6 -- gauge freedom lives in W
> - (a) S invariant under W_t -> W_t + Delta (A_u Delta = 0), Delta scales 1e-3/1/1e3 (incl. ||Delta|| = 1e-6 ||W_t|| and 1e6 ||W_t||): max |dS| / rounding floor of S (max_(u,i) sum_l |u*_l||t_hat_l| + sum_k |u_hat_k||t*_k|) = 4.716e-16; max |dS| / (||Delta||_F max||q_i|| sqrt(K_u) 2) = 1.420e-13 (this second ratio is only informative when ||Delta|| >> ||W||; for ||Delta|| << ||W|| the floor is S's own rounding)  [540 gauge shifts incl. mirror]
> - (b) <grad_W_t L, Delta> / (||grad|| ||Delta||): max 2.076e-16
> - (a,b) mirror W_u with A_t: max |dS| / floor 3.986e-16 (vs would-be change 1.586e-10); |<grad_W_u, Delta>| normalised max 2.865e-16
> - (e) nullity of A = [1^T; Phat^T] measured vs K-d-1 (0 if K<=d+1), all seeds/scales agree = True; (K, d, measured, expected): [(4, 4, 0, 0), (5, 4, 0, 0), (6, 3, 2, 2), (7, 3, 3, 3), (8, 3, 4, 4), (9, 5, 3, 3), (10, 3, 6, 6), (12, 5, 6, 6)]
> - (c) plain SGD, 40 steps, lr=0.05, P_u frozen (A_u fixed), all other params trained, 2 configs x 4 seeds: null-space coords of W_t max rel drift 6.329e-16; row-space coords min rel movement 2.769e-01
> - (c) informative: with P_u ALSO trained the step-0 null space rotates; drift of step-0 null coords = 4.827e-02 (claim presumes a fixed A_u)
> - (d) weight decay wd=0.1 (grad += wd*W): per-step null coords vs (1-lr*wd) x previous: max rel err 6.530e-16; after 40 steps vs (1-lr*wd)^T x initial: max rel err 7.059e-16
> - **Verdict: CONFIRMED** (interpretation for (c)/(d): A_u fixed, i.e. P_u frozen while W_t and all other tables train)
>
> ## C7 -- live intercept attributable and not gauge
> - 1^T t_hat = sum_f (W_t^T 1).E_I[f] + (W_t^T 1).E_I[V+i]: max rel err 1.388e-15
> - 1^T t_hat invariant under gauge shifts Delta (A_u Delta=0), 270 shifts (Delta scales 1e-3/1/1e3): max |change| / rounding floor (max_i sum_l |t_hat_l|) = 2.764e-16; max |change| / (||Delta|| max||q_i|| sqrt(K_u)) = 1.977e-13
> - **Verdict: CONFIRMED**
>
> ## C8 -- leave-one-out reaches both roles
> - u_hat' - u_hat = W_u (q_full - q_loo): max rel err 5.507e-16 over 2155 (u, positive) pairs
> - ||u_hat' - u_hat|| / ||u_hat||, |H|>=2 and positive's word sum != basket mean: min 2.622e-02, max 3.357e+00 (generically nonzero)
> - pairs where the positive's word sum == basket mean (baskets of twin items): 20, max ||u_hat' - u_hat||/||u_hat|| there = 0.000e+00 (zero as predicted)
> - same for |H|=1 (LOO word part = 0): min 5.189e-01
> - control: u* also differs between q_loo and q_full (min over pairs of max abs diff 0.000e+00) -- both roles see the same q_loo in the merged model
> - constructed: positive B with word sum == basket mean -> ||u_hat' - u_hat|| = 0.000e+00; positive A (!= mean) -> 1.226e+00
> - **Verdict: CONFIRMED**
>
> ## C9 -- normalisation controls
> - (a) normalised model: u*, t* max abs change 4.441e-16; u_hat/t_hat min rel change 4.571e-01; S min rel change 4.571e-01
> - (b) no-linear-unit models (u*^T t free t; u^T t* free u), q normalised before the cosine unit: scores max rel err 4.160e-16; grads (E tables, P, free table) max rel err 1.029e-15
> - (c) one-sided block normalisation: u*, t* max abs change 0.000e+00 (bit-identical path); u_hat(c q) vs u_hat(q), c in {1e-3,0.37,1,2,1e3}: max rel err 4.608e-16
> - **Verdict: CONFIRMED**
>
> ## C10 -- twin separation: cosine squashed, linear not
> - sweep of word-mass scale c (m = c*m0, ||m|| = c||m0||): ['1e-03', '3e-03', '1e-02', '3e-02', '1e-01', '3e-01', '1e+00', '3e+00', '1e+01', '3e+01', '1e+02', '3e+02', '1e+03']
>   seed 0: ||t*(i)-t*(j)|| = 1.76e+00 1.77e+00 1.79e+00 1.84e+00 1.97e+00 2.13e+00 9.89e-01 2.12e-01 6.06e-02 1.86e-02 5.84e-03 1.84e-03 5.82e-04  | increases: whole sweep 5, c>=1 only 0 | last/first = 3.30e-04 | log-log slope over c in [10,1000] = -1.009 | ||e_i||/||m0|| = 0.75
>   seed 1: ||t*(i)-t*(j)|| = 2.13e+00 2.13e+00 2.13e+00 2.11e+00 2.04e+00 1.73e+00 1.01e+00 3.89e-01 1.28e-01 4.10e-02 1.30e-02 4.11e-03 1.30e-03  | increases: whole sweep 0, c>=1 only 0 | last/first = 6.10e-04 | log-log slope over c in [10,1000] = -0.997 | ||e_i||/||m0|| = 0.81
>   seed 2: ||t*(i)-t*(j)|| = 2.65e+00 2.65e+00 2.64e+00 2.58e+00 2.27e+00 1.22e+00 4.42e-01 1.52e-01 4.99e-02 1.60e-02 5.10e-03 1.62e-03 5.11e-04  | increases: whole sweep 0, c>=1 only 0 | last/first = 1.93e-04 | log-log slope over c in [10,1000] = -0.995 | ||e_i||/||m0|| = 0.69
>   seed 3: ||t*(i)-t*(j)|| = 1.36e+00 1.36e+00 1.36e+00 1.37e+00 1.33e+00 1.02e+00 4.59e-01 1.53e-01 4.86e-02 1.53e-02 4.85e-03 1.53e-03 4.85e-04  | increases: whole sweep 3, c>=1 only 0 | last/first = 3.56e-04 | log-log slope over c in [10,1000] = -1.000 | ||e_i||/||m0|| = 0.39
>   seed 4: ||t*(i)-t*(j)|| = 2.64e+00 2.64e+00 2.65e+00 2.66e+00 2.68e+00 2.57e+00 1.60e+00 5.65e-01 1.78e-01 5.58e-02 1.76e-02 5.57e-03 1.76e-03  | increases: whole sweep 4, c>=1 only 0 | last/first = 6.66e-04 | log-log slope over c in [10,1000] = -1.002 | ||e_i||/||m0|| = 0.68
>   seed 5: ||t*(i)-t*(j)|| = 1.61e+00 1.61e+00 1.62e+00 1.63e+00 1.67e+00 1.70e+00 1.19e+00 4.19e-01 1.29e-01 4.01e-02 1.26e-02 3.99e-03 1.26e-03  | increases: whole sweep 5, c>=1 only 0 | last/first = 7.81e-04 | log-log slope over c in [10,1000] = -1.005 | ||e_i||/||m0|| = 0.80
>   seed 6: ||t*(i)-t*(j)|| = 1.80e+00 1.80e+00 1.80e+00 1.81e+00 1.80e+00 1.59e+00 8.28e-01 2.81e-01 8.89e-02 2.81e-02 8.87e-03 2.80e-03 8.86e-04  | increases: whole sweep 3, c>=1 only 0 | last/first = 4.92e-04 | log-log slope over c in [10,1000] = -1.001 | ||e_i||/||m0|| = 0.34
>   seed 7: ||t*(i)-t*(j)|| = 9.68e-01 9.69e-01 9.72e-01 9.84e-01 1.02e+00 1.12e+00 9.09e-01 3.09e-01 9.24e-02 2.85e-02 8.94e-03 2.82e-03 8.91e-04  | increases: whole sweep 5, c>=1 only 0 | last/first = 9.21e-04 | log-log slope over c in [10,1000] = -1.008 | ||e_i||/||m0|| = 0.57
> - cosine-unit difference tends to 0 in all 8 seeds: True (tail slope ~ -1, i.e. O(1/||m||))
> - strictly decreasing over the WHOLE sweep in all seeds: False (non-monotone in 6/8 seeds while ||m|| < ||e||: the difference first grows slightly before the word mass dominates); decreasing for all c >= 1 (||m|| >= ||m0||, i.e. word mass at or above the ID-row scale) in all seeds: True
> - linear-unit difference t_hat_i - t_hat_j vs W_t(e_i - e_j) across the sweep: max abs err 2.076e-12, max rel err 2.106e-13 (float64 rounding at c=1e3)
> - **Verdict: CONFIRMED (tends to 0; linear-unit difference exactly independent of m); the parenthetical 'decreasing in ||m|| over a sweep' FALSIFIED as a global statement -- non-monotone at small ||m|| in 6/8 seeds, monotone once ||m|| >= ID-row scale**
>
> ## C11 -- degenerate paths
> - (a) cold user (empty basket, no ID row), 120 instances: q_u == 0 exactly: True; u* == 1 exactly (torch.equal): True; u_hat == 0 exactly: True
> - (a) S(u,i) == 1^T t_hat_i: via the matmul score path torch.equal = False, max abs diff 7.276e-12, max diff / sum_l |t_hat_l| = 2.258e-16 (machine precision; bitwise identity fails only because the matmul reduces in a different order than .sum); via element-wise product + .sum torch.equal = True
> - (b) cold items (no ID row, words present): min ||q_i||/mean||E_I word row|| = 3.057e-01; min over items of (max_k - min_k) t*_k = 2.794e-01; min ||t_hat||/(||W_t|| ||q_i||) = 2.112e-01
> - (b) twin cold items with identical word sets: identical t* and t_hat (torch.equal) = True; max abs diff 0.000e+00
> - **Verdict: CONFIRMED (S = 1^T t_hat to machine precision, 1e-16 relative; not bitwise through the matmul path)**
>
> ## C12 -- bag-size channel (statistical trend)
>   V=8, d=3, F=3, 20000 paired baskets: E||basket mean|| for n=1..10 = 1.632 1.340 1.234 1.178 1.143 1.117 1.099 1.086 1.075 1.066 (s.e. <= 0.0042) | strictly decreasing: True
>   V=50, d=5, F=3, 20000 paired baskets: E||basket mean|| for n=1..10 = 1.669 1.201 1.003 0.894 0.819 0.766 0.726 0.694 0.668 0.647 (s.e. <= 0.0033) | strictly decreasing: True
> - u*(c q_u) vs u*(q_u), c in {1e-3,0.5,3,1e3}: max abs diff 4.441e-16; ||u_hat(c q)|| vs c ||u_hat(q)||: max rel err 3.787e-16
> - **Verdict: CONFIRMED**
>
> ## C13 -- collapsed function class
> - S = (W_t^T 1).q_i + qhat_u^T (Phat_u^T W_t) q_i + (W_u^T 1).q_u + qhat_i^T (Phat_t^T W_u) q_u: max rel err 8.207e-16 over 240 instances (ids and noid arms)
> - **Verdict: CONFIRMED**
>
> ## C14 -- K10 Procrustes alignment penalty (interpretation: R in O(d) acts on the embedding dimension, row-wise)
> - (a) E_U = R0 E_I (R0 random orthogonal, det +1 and -1), 6 seeds x 3 shapes x 3 scales: max L_align/||E_I||^2 = 1.195e-30 (zero to machine precision)
> - (a) random unrelated tables: min L_align/||E_I||^2 = 3.316e-01 (> 0); the converse (L=0 => E_U = R E_I) is definitional with R the Procrustes minimiser
> - (b) R held constant: min ||grad_E_U||/(2 sqrt L) = 1.000e+00, min ||grad_E_I||/(2 sqrt L) = 1.000e+00; both equal 2 sqrt(L) to rel err 6.661e-16 -> nonzero whenever L > 0
> - (c) E_U = c * E_I (otherwise identical): L_align/||E_I||^2 per (seed, d, c), predicted (|c|-1)^2:
>     seed 0 d=3 c=+0.001: L/||E_I||^2 = 9.980010e-01  predicted (|c|-1)^2 = 9.980010e-01
>     seed 0 d=3 c=+0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
>     seed 0 d=3 c=+0.999: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
>     seed 0 d=3 c=+1.001: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
>     seed 0 d=3 c=+2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
>     seed 0 d=3 c=+1000: L/||E_I||^2 = 9.980010e+05  predicted (|c|-1)^2 = 9.980010e+05
>     seed 0 d=3 c=-0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
>     seed 0 d=3 c=-1: L/||E_I||^2 = 5.020856e-32  predicted (|c|-1)^2 = 0.000000e+00
>     seed 0 d=3 c=-2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
>     seed 1 d=4 c=+0.001: L/||E_I||^2 = 9.980010e-01  predicted (|c|-1)^2 = 9.980010e-01
>     seed 1 d=4 c=+0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
>     seed 1 d=4 c=+0.999: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
>     seed 1 d=4 c=+1.001: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
>     seed 1 d=4 c=+2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
>     seed 1 d=4 c=+1000: L/||E_I||^2 = 9.980010e+05  predicted (|c|-1)^2 = 9.980010e+05
>     seed 1 d=4 c=-0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
>     seed 1 d=4 c=-1: L/||E_I||^2 = 3.467819e-31  predicted (|c|-1)^2 = 0.000000e+00
>     seed 1 d=4 c=-2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
>     seed 2 d=5 c=+0.001: L/||E_I||^2 = 9.980010e-01  predicted (|c|-1)^2 = 9.980010e-01
>     seed 2 d=5 c=+0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
>     seed 2 d=5 c=+0.999: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
>     seed 2 d=5 c=+1.001: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
>     seed 2 d=5 c=+2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
>     seed 2 d=5 c=+1000: L/||E_I||^2 = 9.980010e+05  predicted (|c|-1)^2 = 9.980010e+05
>     seed 2 d=5 c=-0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
>     seed 2 d=5 c=-1: L/||E_I||^2 = 5.353706e-31  predicted (|c|-1)^2 = 0.000000e+00
>     seed 2 d=5 c=-2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
> - (c) L > 0 for every positive c != 1 tested: True; c = -1 gives L = 0 (R = -I is orthogonal): True -> the literal 'c != 1' wording fails at c = -1; not scale-invariant for c > 0 confirmed
> - **Verdict: CONFIRMED for c > 0 (a, b, c); literal (c) wording FALSIFIED at c = -1 (sign flip is orthogonal)**

### 4b.4 What this changes in the candidate (no re-spec, no second round)

- **C2(b) — FALSIFIED sub-clause, my error, amended visibly in §3.4 read-out 4.** The spec's
  equality condition for "share = removal effect" on the user linear half was wrong. The
  verifier's closed form is exact: Δ_j = (ā − a_j)/(|H|−1). Consequence for the design: on
  the linear halves only the **item-side word removal** is a true counterfactual (q_i is a
  plain sum); the **user-side per-purchase share** is an exact summand but not a removal
  effect, because the basket mean renormalises. The renderer renders shares; the
  counterfactual is a second, disclosed reading. Carried into Step 5 as a known defect of
  the Step-3 text (fixed), not of the mechanism.
- **C10 — scope sharpened (3b-iii).** The twin-tax squashing is monotone only for shared
  mass above the ID-row scale; the limit and the O(1/‖m‖) tail hold, and the linear-half
  independence of m is exact. The refund hypothesis is unchanged; its statement is now
  regime-qualified.
- **C14(c) — wording (§3.6 K10).** "c ≠ 1" → positive rescaling; reflections are orthogonal.
- **C6 — confirmed under the fixed-slab reading; the rotating-slab note is added to 3b-i.**
  Read-outs 3/4 inherit the W-gauge disclosure exactly as fI's u_k and fU's t_l did.
- **C4 keystone — strongest possible result:** scores bit-identical in float64 *and*
  float32, loss bit-identical, all parameter gradients identical (0.0 relative error)
  between the single-q wrapper and the host's double lookup. The build's i02-style
  keystone has a verified target.
- **C11 — "exactly" means exact arithmetic:** S = 1ᵀt̂ for a cold user holds to 2e-16
  relative through the matmul path, bitwise only through an element-wise sum; wording in
  §3.4 ("exactly") is read as exact arithmetic, as elsewhere in the lineage.
- Everything else (C1, C3, C5, C7, C8, C9, C12, C13) confirmed at machine precision or as
  the stated trend; the LOO-reaches-both-roles argument (C8) and the normalisation
  controls (C9) are now verified math for the build.

---

## 5. Requirements evaluation

Requirements doc re-read in full this session (version with the R1 clarification note of
2026-07-12 and the R6 staging note of 2026-08-20). Evidence base for the accuracy rows:
the 100-trial ml-1m table read from LEO5 on 2026-10-26 (test HR@10 / NDCG@10, seed
38210573; working evidence per the learnings-ledger rule):

| run | HR@10 | NDCG@10 | | run | HR@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| `user_item_proto` (UI host) | **0.6246** | 0.3573 | | `lightfm_hist` (hist × free item, dot) | 0.6238 | **0.3665** |
| `user_proto` (U host) | 0.6142 | 0.3587 | | `lightfm_tags_ids` (free user × words+ID, dot) | 0.6155 | 0.3590 |
| fU ids / fU noid | 0.5613 / 0.6082 | 0.3189 / 0.3507 | | `lightfm_tags` | 0.5905 | 0.3438 |
| fI ids / fI noid | 0.5628 / 0.5156 | 0.3204 / 0.2822 | | `item_proto` (I host) | wave-1 job 8153734, pending | |

hm_1_month has no 100-trial rows yet (all queued in wave 1); dev-tier readings: fI ≈ host
parity, fU +0.011 vs host (memory 2026-10-23).

### 5.1 Ratings with prose

| Req | Rating | Basis (one line; prose below) |
|---|---|---|
| R1 tied | **strong** (within-side exemplar + cross-side word tie by default) *(SC.2, F-DC09-04)* | one composed representation read in two roles per side — the exemplar's *structure*; across sides the shared word table is the DEFAULT (the first cross-side tie in the lineage), `_sep` prices it *(A1/A7, 5b; re-scoped at SC.2 after the user's review decision)* |
| R2 intrinsic | **strong on derivability — conditional on R4** *(SC.2, F-DC09-04/07)* | all four components are exact sums over named rows (named coverage measured per component); per-line identifiability is an R4/§4 matter — partly repaired (the canonical per-field gauge, 3b-xi) and partly disclosed (the W-slab, 3b-i) *(A1, 5b-2/9/19)* |
| R3 prototype | **strong for the cosine units / partial for the linear halves** (host-inherited, graded as inheritance; conditional on R4) | cosine units: which prototypes, what they mean; linear halves: prototype-indexed by *position* through a free W *(A1, 5b-1)* |
| R4 grounded | **partial** | unenforced shared-geometry grounding, now on two sides; M1/M1′/M11 open |
| R5 sparsity | **strong** (cold items, under the invariant cold convention — SC.2, F-DC09-08) / **partial** (cold users) | both units exist from words; cold q = words + mean warm ID row (gauge-invariant); empty-basket cold-user fallback = live intercept; cold×cold unmeasurable |
| R6 dense accuracy | **partial — unassessed** *(SC.2, F-DC09-04)* | UI n100 is the bar; the mechanism-derived expectation is not positive except through the unmeasured refund (3b-iii); the arms are run and reasoned about (3b-v) |
| R7 vocabulary | **partial** | same taxonomy-heavy canonical five; new: class-level popularity stated per word |
| R8 parsimony | **strong — two moves, the second isolated by `_sep`** *(SC.2, F-DC09-04)* | the `user_item_proto` move plus the cross-side word tie; knob and controls are arms |
| S1 decoupled | **strong-with-a-gap** *(SC.2, F-DC09-04)* | three single-side decoupled rows exist in the fleet (`lightfm_tags*`, `lightfm_hist*`); the two-sided twin `lightfm_tags_hist` is the SC.6 decision *(A1, 5b-5)* |
| S2 user-side | **strong** (re-scoped) | history-grounded user side in; demographics = candidate prime after the merge |
| S3 images | **partial** | one more row type in either composition; rendering salvage available |
| S4 naming | **strong** (conditional on R4; profiles read in the canonical gauge in ids arms — SC.2, F-DC09-07) | intrinsic profiles on both sides feed the naming layer; the word rows are partly dot-trained (M14, precision-amended) — profile validity to be spot-checked *(A1, 5b-12)* |
| S5 pipeline | **strong** | all checklist items yes; VRAM flag mitigated |

**R1 — strong (within-side exemplar; table re-scoped by A1).** Each side's single composed vector is read in the host's two roles —
cosine against its own prototypes and linear projection into the other side's prototype
coordinates — which is precisely the `prototypes_double_tie` exemplar the requirement
names, applied to feature-composed representations instead of free tables. The operative
core holds maximally too: there is no path from features to score except through those
two tied roles; no reranker, no bolt-on. Where it strains: *across* sides nothing is tied
in the default arm (a two-table design choice — A7 withdrew the LightFM-convention justification), so the "strongest realisation"
(one vocabulary table read by both sides) is an ablation, not the headline; and the tie's
linear roles are dot-trained, so half of what the tie shares is not cosine-protected.
Open risk: none on satisfaction; the Seed B arm may yet prove the better default.

**R2 — strong on derivability, conditional on R4; partial on per-line identifiability** *(re-rated 2026-10-26, A1 after 5b-2/9/19; the prose below is the pre-5b argument, still valid for what it claims)*. The host's §5.2 explanation stops at
"t̂_l is the item embedding's weight on prototype l" — an unattributed learned functional.
Here every one of the four components is an exact sum over named rows (4b C1–C3, C7
confirmed at machine precision): per-feature shares of t\*, per-word/per-purchase shares
of u\*, per-word shares of t̂ and û, and the live intercept's per-word zoom. Prototype
meaning is read from parameters on both sides (no synthetic maximally-activating user,
no nearest-item profiling needed). Strains: (a) the linear halves' coordinates are named
by the *other* side's prototypes through a free W — attributable, but the per-prototype
lines carry a frozen gauge component (3b-i; slab projection identified, disclosure owed to
the hidden-effects section); (b) the user-side per-purchase share on the linear half is a
summand, not a removal effect (4b C2(b), amended); (c) e_f meanings remain CF-trained
(label confounding carries). None of these is a post-hoc reconstruction; they are honesty
conditions on an intrinsic read-out.

**R3 — strong for the cosine units, partial for the linear halves** *(re-rated 2026-10-26, A1 after 5b-1: host-inherited, graded as inheritance)*. The explanation stays "which prototypes does this pair
activate, and what do those prototypes mean" on both sides, with meaning tied to words.
The caveat is inherited from UI-ProtoMF itself: the two linear halves are indexed by
prototypes but computed linearly, not by similarity — the host's own §5.2 reads them as
weights, and so do we. The merge does not add a non-prototype vehicle; it adds attribution
to the host's existing non-similarity half.

**R4 — partial.** Grounding is by shared geometry, unenforced (the lineage's standing
point), now on two prototype sets. Nothing forces profiles sharp (M1/M1′), nothing forces
the two tables to agree on what a word means (M11, new), and the profile-validity
spot-checks (F-DC01-02 / F-DC05-03a) are owed on both sides at scrutiny. What improves vs
fI/fU: the item-side intercept zoom gives a second, independent reading of each word's
role (its class-level popularity weight), and the Seed B arm would make community and
archetype profiles live in one geometry. *(SC.2, F-DC09-04: one geometry is now the default; whether the two prototype sets then stay distinct is M18.)*

**R5 — strong for items, partial for users, two-sided by construction.** A cold item (ID
row dropped) has both t\* and t̂ from words (4b C11b), and its intercept B_item is a
word-composed popularity prior — LightFM's cold mechanism, intact, plus the prototype
read-out. New cost: the cold magnitude handicap now has an object (t̂ lacks the warm ID
row's length; measured by the S0.3 cold-vs-all protocol). A cold user falls back to S =
B_item(i) exactly (C11a) — a popularity-shaped, non-personalised ranking — the honest
degenerate behaviour, with fold-in after the first purchase; warm-thin users are the
measurable surface (D4 strata). Cold users and cold×cold stay **unmeasurable** in the
current split (3-scenario testbed entry, unmet).

**R6 — partial; the bet is finally assessable and it is not comfortable.** The staging
note is discharged: fUfI vs UI-ProtoMF *is* the paper-headline comparison. *(A3 wording, 2026-10-26: "paper-headline" means the lineage's headline — the comparison is against our own UI-host row, like-for-like; the paper's 0.657 is a different protocol and seed count.)* Evidence as of
today (ml-1m, n100): the single-side stages lost to their hosts — fI 0.5628 vs `item_proto`
(pending at n100; ≈ −0.056 at dev tier), fU 0.5613 (ids) / 0.6082 (noid) vs 0.6142 — while
the **plain-dot** compositions nearly match the UI host: `lightfm_hist` 0.6238 (HR) and
0.3665 (NDCG, above UI's 0.3573), `lightfm_tags_ids` 0.6155. So on ml-1m the compositions
are not what loses; the cosine head is (scratchpad 1b-27). *(Corrected 2026-10-26, A8 after 5b-6: that inference over-reached — `lightfm_hist` pairs the composition with a FREE item table. The supportable decomposition is: fU-noid vs `lightfm_hist` = 0.016 (prototype head + item-table dimension K_u vs d), fU-ids vs fU-noid = 0.047 (the user-ID row under this search). The larger measured cost is the user-ID row, which the default two-ID-row arm carries — hence A2. And fUfI's linear halves are bilinear forms between a normalised and an unnormalised composition with no free table — "a norm-reading channel", not "exactly the channel the dot rows have".)* The merge adds to each side
exactly the linear, norm-reading half those dot rows have, which is the mechanical
reason to expect fUfI above fI and fU; whether it reaches 0.6246 is open. Three risks
stated plainly: (i) the fU **ids** arm is 0.047 *below* its noid arm on ml-1m — the user-ID
row hurts there — so the default two-ID-row arm may underperform a mixed arm (user noid,
item ids), which is why §3.7 records the mixed arms as a user decision rather than
dismissing them; (ii) the ml-1m vocabulary is 23 % of the host's parameters (4.1) — the
words are the model there, and lockstep/omnipresence pressures are strongest; (iii) hm has
no 100-trial evidence at all yet. Pre-registered expectation (3b-v): fUfI − UI gap smaller
than fU − U gap on ml-1m. *(SC.2 amendment 2026-10-26 — F-DC09-04/05: grade re-read as "partial — unassessed"; the mechanism-derived expectation is not positive except through the unmeasured refund. Feature provenance differs between testbeds (F-DC09-05): ml-1m's `tags@0.8` are Tag-Genome relevance ≥ 0.8 — a score predicted from user tagging/rating behaviour (Vig, Sen & Riedl 2012; citation to verify at RW time) — so the load-bearing R6 testbed's content is partly behaviour-derived, while hm's taxonomy is static catalogue metadata; fleet-wide, disclosed in Ch. 3 Data.)*

**R7 — partial.** The vocabulary is unchanged (canonical five on hm: department, colour,
section, graphical appearance, product type — taxonomy-heavy; ml-1m genres + tags), so
the balance point is the lineage's. One genuine R7 gain: the live intercept's per-word
zoom turns the host's opaque popularity channel into a user-perceivable sentence
("Trousers Denim sells well this month"), and the per-purchase zoom speaks in purchases.
Colour/pattern vs department weighting remains an empirical outcome, not a design choice.

**R8 — strong.** The candidate is the `user_item_proto` move — the requirement's own unit
of "one idea" — applied to the two scrutinised compositions; no new layer, loss, or table.
The shared-table knob, the fI-on-UI rung, the length-ablation control and the two-sided
LightFM row are arms around it, each a single variable. The temptation this cycle resisted
(Seeds C/D: a new head at the tie) is recorded in the scratchpad.

**S1 — partial** *(re-rated 2026-10-26, A1 after 5b-5: designed, not in the fleet)*. The two-sided LightFM row (`lightfm_tags_hist`, history × words under a
plain dot) is the exact decoupled twin: same compositions, no prototype head, no W. It
does not exist in the fleet and is requested as a charter amendment (dc05 precedent). The
existing `lightfm_*` rows and the host's post-hoc slots complete the ladder from
decoupled to tied.

**S2 — strong (re-scoped).** The user side is feature-aware through history (dc05), the
merge inherits it, and the architecture does not preclude demographics: candidate prime
(vault `2026-09-08_1649`, shape 2 preferred) adds a second user-side layer after the merge.

**S3 — partial.** Images can enter either composition as one more row type (a frozen
image embedding projected to d, "one more summand"), and the rendering salvage
(thumbnails on purchase/exemplar lines) applies now; no mechanism this cycle.

**S4 — strong.** Both prototype sets have intrinsic word profiles; the `naming/` layer's
intrinsic route exists for both sides; the LLM renaming slot is unchanged.

**S5 — strong.** 4.4: all five checklist items yes; the one flag (fI-sized VRAM at high
concurrency on ml-1m) has two mitigations in hand.

### 5.2 §4 design tensions

- **Where do features enter?** At both entity representations (route A, both sides) — and,
  for the first time in the lineage, features reach the score through a **non-cosine**
  path: the host's own linear halves read the compositions directly. Prototypes stay free
  (no features at the prototype parameterisation; dc04's axis stays abandoned).
- **Fidelity vs accuracy.** The merge *loosens* the single-side constraint: the norm channel
  opens (expected accuracy), and the price is that two score components are dot-trained,
  Steck-exposed and W-gauged (fidelity). The lineage turns this into a measurement rather
  than an assumption: component variance shares (M8), the 2×2 with host UI as control
  (M7), and the user's length-ablation control isolate the channel within one architecture.
- **Attribution granularity.** Per word, per purchase, per prototype, per component; clique
  level is what is identified (F-DC01-06 on both tables and inside W e_f); per-field
  splits informative. New granularity: the intercept per word.
- **Perceptual alignment (R7).** Unchanged vocabulary; new user-perceivable content in the
  intercept zoom; whether colour or department carries the shares is empirical.
- **Identifiability.** The gauge moves from coefficient vectors into W (3b-i, verified C6);
  lockstep persists on both tables; the Seed B arm and K10 are identifiability choices
  about the vocabulary itself (one meaning per word, or two). Rashomon-tunable knobs are
  listed in §3.6; `max_norm` changes semantics at this stage (it now caps the popularity
  channel), so its Rashomon reading differs from fI/fU's.

### 5.3 Part-prototype pitfalls checklist (reading-list S5; modality-independent items)

- *Prototype number:* K_u, K_t ∈ [10, 100) searched; each is also the other side's
  projection width (3b-vi) — vocabulary size is not a free knob.
- *Prototype quality / semantics (§4.2 "unsupervised prototypes lack semantics"):* both
  sets nameable from parameters; grounding unenforced (R4 partial).
- *Similarity-perception gap (§3.1.2):* similarity is to word-grounded prototypes and shares
  are stated in vocabulary; latent cosine ≠ human similarity persists; label confounding.
- *Cosine unreliability (§4.4, Steck):* the cosine units train through the cosine they
  read (steel-man); the linear halves are dot-trained and exposed — the first lineage stage
  with that exposure, disclosed. *(Corrected 2026-10-26, A14 after 5b-12: the steel-man covers the units, not the rows — every word row now also receives dot-trained gradient through W (M14), so the naming profiles are read off partly dot-trained vectors; spot-check owed.)* *(Precision at SC.2, F-DC09-06: "exposed" ≠ "provably arbitrary" — the cosine units break the diagonal-rescaling invariance Steck's argument needs; the rows are shaped by a mixed objective, semantic reading unvalidated.)*
- *Collapse / duplication:* inherited coverage regularisers (`max` types); UI is empirically
  the healthiest host (10 effective directions, mechanism open) — effective-direction and
  cone instruments carry.
- *Bottleneck assumption (§3.3.2):* n/a in the ids arm (reparameterisation); the noid arm is
  an additive-span restriction, declared.
- *Spatial / part-level issues:* n/a (whole-entity prototypes).
- *Safety / bias:* user prototypes absorb demographics (A1 §5.3); word profiles make that
  auditable directly. *(Corrected at SC.2, 2026-10-26 — F-DC09-04: "gender is not in any vocabulary" is withdrawn — hm's section and department words (Ladieswear, Menswear, Baby/Children, Divided, …) encode the customer segment, so user prototypes are segment-aligned by construction; the profiles make this auditable, and the thesis says so.)*
- *Generalisation:* cold items via words; cold users via the live-intercept fallback; the
  3-scenario testbed is the unmet eval prerequisite.

### 5.4 Interpretable-AI placement

Skipped — the principle framework (`ideas.md` §4) remains undecided (as at dc08).

### 5.5 Degeneration-protection gate (completeness, not veto)

| Mode (origin) | Status |
|---|---|
| M1 / M1′ diffuse profiles (dc01/dc05) | **unprotected — open risk**; K1/K1′ sharpness penalty designed, unstacked; profile-validity spot-checks owed both sides |
| M2 / M2′ frequent-word dominance (dc01/dc05) | partially protected by inherited `max_norm` (loudness); omnipresence unprotected; K2/K2′ unstacked. *New semantics:* `max_norm` now also caps the popularity channel (3.6) |
| M3 / M3′ ID routing (dc01/dc05) | **unprotected — measured by design** (ids vs noid; per-component coverage split); K9 dropout unstacked |
| M4 / M4′ profile overlap (dc01/dc05) | partially protected by inherited coverage regularisers (indirect separation); K4 unstacked; UI-host health advantage unexplained |
| M5′ heavy-basket crowding (dc05) | **unprotected trend**; instruments by |H| band; K6′ centring unstacked (dc06 unresolved, ml-1m only) |
| M6′ → M8 component migration (dc05 → merge form) | **unprotected — open risk**; primary instruments (component variance shares, intercept popularity correlation); K8 λ_W unstacked |
| M7 norm-channel welding (merge) | **unprotected — open risk**; partially bounded by inherited `max_norm`/λ_L2; K7 composed bias / K2 norm variance unstacked; the 2×2 measures it |
| M9 role split of labour (merge) | **unprotected**; per-component coverage split is the detector; K9 unstacked |
| M10 two-row routing (merge) | **unprotected**; same detector; ids/noid ladder isolates it |
| M11 cross-side vocabulary divergence (merge; `_sep` arm only — SC.2) | **removed by construction in the shared default**; in `_sep`: unprotected — open risk; K10 soft alignment unstacked; intra-model Procrustes residual is the instrument |
| M12 bag-size score channel (merge) | **unprotected**; instruments by |H| and token-mass strata; K11 block normalisation unstacked |
| M13 item-side bag-size channel, ml-1m (merge; A5) | **unprotected — open risk**; instruments (‖q_i‖/B_item vs token count vs popularity); `normalize_composed` control, mean-normalised bags, centring unstacked |
| M14 Steck exposure of the word rows / naming layer (merge; A5) | **unprotected — disclosed**; profile-validity spot-check fUfI vs fI/fU; K8 λ_W unstacked |
| M15 role conflict on shared rows (merge; SC.2, F-DC09-03) | **unprotected — described**; `_sep` is the control; no instrument (framing decision) |
| M7-shared popularity-direction leak into q_u, incl. the BCE-active B_user (merge; SC.2, F-DC09-03) | **unprotected — disclosed**; the shared-vs-`_sep` comparison is the measurement |
| M16 overlap / self-affinity + twin channel (merge; SC.2, F-DC09-10) | **unprotected — disclosed**; one one-off overlap/twin measurement; `_sep` control; FM positioning in §2b |
| M18 prototype-set redundancy (merge; SC.2, F-DC09-11) | **hypothesis**; one one-off cross-prototype cosine measurement; read-out 7(b) conditional |
| Translation gauge of the ids arms (3b-xi; SC.2, F-DC09-07) | not a loss-protectable mode — **read-out discipline**: canonical per-field-centred gauge for all per-row read-outs, naming and coverage; substitution counterfactual on fixed-field data; dc01/dc05-ids hm shares carry it (hidden-effects section) |
| Common-mode mass in the personalised halves (3b-xii; SC.2, F-DC09-09) | **described**; one one-off two-way decomposition; renderer line at SC.5 |
| W-gauge (3b-i; host-inherited) | not a loss-protectable mode — **disclosure discipline** (slab projection identified; L2 canonicalises); verified C6 |
| Lockstep splits, both tables + W e_f (F-DC01-06) | **read-out quotient** (grouped shares at clique level); not regulariser-protected |
| Steck exposure of the linear halves (host-inherited) | **unprotected — disclosed**; exact attribution does not rescue the train/read mismatch; the cosine *units* train through the cosine they read, but the word *rows* they read are also dot-trained (M14) — "protected" no longer describes the rows *(A14)* |
| Cold magnitude handicap in t̂ (3b-v) | **unprotected — measured** (S0.3 cold-vs-all); K9 ID dropout trains the cold path |

No identified mode is unlisted. Every "unprotected" row carries a designed, unstacked
knob or a committed instrument; none is a veto.

### 5.6 Step-4b findings folded in as known defects

- **C2(b) — defect in the Step-3 text (amended visibly in §3.4):** the user-side linear
  per-purchase share is a summand, not a removal counterfactual; the correct counterfactual
  Δ_j = (ā − a_j)/(|H|−1) is intrinsic too and is offered as a disclosed second reading.
  Mechanism unaffected; renderer contract sharpened (render shares, label counterfactuals).
- **C10 — scope qualification (3b-iii):** the twin-tax statement holds in the large-shared-
  mass regime; sub-ID-scale baskets of words can behave non-monotonically. Realistic regime
  unchanged; the thesis sentence must carry the qualifier.
- **C14(c) — wording (§3.6 K10):** reflections are orthogonal; "positive rescaling".
- Boundary notes carried: C11 "exact" = exact arithmetic (2e-16 through matmul); C6
  conservation is relative to the current slab when prototypes train.
- Confirmed and now load-bearing for the build: C4 keystone (bitwise scores, exact
  gradients), C8 (LOO must reach both roles — the single-q wrapper does), C9 (the
  normalisation control is an exact no-op on linear-free hosts, incl. gradients).

---

## 5b. Devil's advocate

### 5b.1 Critique (verbatim; the subagent received only this candidate file, read all 1,862 lines, ran no code)

Saved as `Master/temp/dc_checks/dc09/devils_advocate_critique.md`; reproduced blockquoted:

> # Adversarial review — dc09 fUfI-ProtoMF (double-tie merge)
>
> Input: `/Users/glaes/Desktop/ProtoMF-fufi/Master/docs/design_candidates/dc09_fufi_double_tie_merge.md` only (all 1,862 lines). No other files read, no code run.
>
> ## A. Where the ratings are too generous or quietly narrow the requirement
>
> **1. R3 "strong (host-shaped)" is a grade for inheriting the host's weakest property, and the design filter guarantees nothing better was possible (§1c Seeds C/D, §3.4 items 3–4, §5.1 R3).** Half of the score — û·t* and u*·t̂ — is a free linear map whose coordinates are *labelled* by the other side's prototypes but have no geometric relation to them; row l of W^t is "community l" purely by position. Seed D (W := P), the one shortlisted mechanism that would make the label true by construction, was rejected because fUfI-D(F=0) does not reduce to `user_item_proto`. So the reduction keystone — a fine engineering check — was promoted to the design gate, which excludes by fiat any change to the host's non-prototype half, and then "no worse than the host" is scored "strong". Under Seed A this gets worse: û_k is W^u applied to a vector in E^U-geometry, landelled on index k of P^t which lives in E^I-geometry; the two tables are free to diverge (the author's own M11), so "your dark-denim purchases make you respond to archetype 12" is a sentence stitched across two unrelated geometries through a free matrix.
>
> **2. R2 "strong, strengthened vs host" ignores that the rendered per-prototype bars are gauge-dependent, and the toy's gauge count understates the real one by roughly an order of magnitude (§3b-i, §3.4 renderer, §5.5 "no gauge annotation on figures").** Under Δ with 1ᵀΔ=0, P̂^uΔ=0, the *sum* Σ_l c_l(u)·t̂_l is invariant but every individual bar c_l·t̂_l moves — the bars panel is exactly the quantity the gauge scrambles, and the stated discipline is to not say so on the figure. C6 counts the null space as K−d−1 for generic full-rank P (146 of 5,548 W^t entries at the host optimum d=73, K_u=76 — negligible). But the vault's own measurement (§0.4: UI has ~10 effective prototype directions) means rank(P̂^u) ≈ 10, the slab is ≈ 11-dimensional, and ≈ 65 of 76 coordinates of t̂ are (softly) invisible to the score. §4b.2 even concedes "degenerate prototypes enlarge the null space — the toy uses random P"; the realistic case is the degenerate one, and the per-prototype lines the renderer prints are mostly init noise plus Adam drift (C6's conservation is a plain-SGD result; elementwise preconditioning does not preserve the null-space split). The rendered example also prints word-level shares for "Trousers Denim / Jeans / Menswear" — exactly the department↔section lockstep words F-DC01-06 says are identified only at clique level.
>
> **3. R1 "strong" while the author's own "strongest literal reading" is demoted to an ablation, justified by a LightFM convention that does not apply (§0.1 M6.1, §2.2, §3.3, §5.1 R1).** M6.1 says sharing the table "would be the first cross-side tie and the strongest literal reading of R1's exemplar"; §5.1 then rates R1 strong with that tie as an arm. The two-table default is defended as "LightFM's F^U/F^I convention" — but LightFM's user features and item features are *disjoint vocabularies* (demographics vs metadata), so of course they have two tables; LightFM offers no precedent for two tables over *one* vocabulary read in two roles. The "paper says" (one free parameter read twice) and "I propose" (one pooled function output read twice) are also blurred in "the host's exemplar, literally": what is tied in the paper is a per-entity parameter; what is tied here is already shared across all items via the word rows, and "mathematically identical" is identity with a hypothetical re-pointed composition the paper never had.
>
> **4. R5 and R6 are each graded on the arm that favours them; no single arm serves both (§5.1 R5/R6, §3.7, §3b-v).** The evidence table shows fI ids beats fI noid by +0.047 while fU *noid* beats fU ids by +0.047 — opposite signs — so the default two-ID-row arm is the one the data disfavour, and the obvious mixed arm (item ids, user noid) is "not in the default plan". R5 "strong (cold items)" rests on dropping an ID row from a model trained with it on *both* roles — a train/test mismatch the document acknowledges, leaves unmitigated (K9 unstacked) and unmeasured — or on the noid arm, which forfeits the ID-row refund that R6's "mechanical reason to expect" leans on and leaves exact-twin items tied on both halves.
>
> **5. S1 "strong" and R8 "strong" grade a plan, not a candidate (§5.1 S1/R8, §4.5).** The decoupled two-sided control `lightfm_tags_hist` does not exist, requires a charter amendment, and sits in the "full designed set" the Nov-3 scope is said to dislike. R8 counts the mechanism as "one idea" but the attributable experiment is 12 runs (shared ×2, fI-on-UI ×2, two-sided LightFM ×2, length-ablation ×2 on top of 4 default) — and the controls that make the headline interpretable are precisely the ones a scope cut removes first, leaving a bare fUfI-vs-UI number that attributes to nothing.
>
> ## B. What the author is too close to see
>
> **6. The inference "the compositions are not what loses; the cosine head is" is not supported by the rows quoted (§5.1 R6, §1b-27).** `lightfm_hist` is hist × *free* item; `lightfm_tags_ids` is *free* user × words — both are single-side compositions paired with a free table; no two-sided dot row exists. The ids/noid split contradicts the head story: if the head lost, fU noid should lose as much as fU ids, yet the gaps are 0.006 vs 0.053 — the ID row on a composed side under this search is the better-supported culprit, and the default fUfI arm has two of them. And by the author's own 3b-iv algebra, fUfI's "linear halves" are bilinear forms between one *normalised* composition and one *unnormalised* one with no free table anywhere — not "exactly the channel the dot rows have".
>
> **7. The pre-registered expectation is underdetermined and there is no falsification condition (§3b-v, §5.1 R6).** "fUfI−UI gap smaller than fU−U gap" never says ids or noid; the fU−U gap is 0.053 (ids) or 0.006 (noid), so the bar is either trivial or a parity claim within noise. Nowhere does the document say what result would count *against* the merge or the lineage's R6 bet; "whether it reaches 0.6246 is open" is not a kill criterion. A "win" via ID-dominated linear halves (M9) would satisfy the expectation while gutting R2, and the pre-registration cannot distinguish that from the intended outcome.
>
> **8. Statistical power is never mentioned, and the optimisation protocol is biased against the candidate (§5 table, §4.2, §2.1).** One seed, 100 TPE/ASHA trials, no CI: on 6,034 ml-1m users the standard error of HR@10 is ≈ 0.006, so UI 0.6246 vs `lightfm_hist` 0.6238 vs `user_proto` 0.6142 vs fU-noid 0.6082 are all within 1–2 SE *before* adding search variance. The host replication itself is 0.032 below the paper's UI-ProtoMF (0.657, 3 seeds) — the "paper-headline bar" is set three points under the paper. §4.2 says convergence speed is "unknown, not a risk"; under ASHA it is exactly a risk: pooled word rows must learn before any item is distinguishable, free tables memorise immediately, and early rungs prune slow starters — a systematic edge for the host (and a plausible reason fU ids < fU noid). The fU optimum at d=92, K_u=98 sits at the ceiling of a box the host fits comfortably inside (d=73, K_u=76, K_t=11) — "like-for-like search space" measures the box, not the architecture.
>
> **9. Toy-verified identities are used as evidence for empirical claims they cannot touch (§4b, §5.1 R2, §3b-iv).** All fourteen claims are algebraic/gradient identities or constructions; "confirmed at machine precision" says nothing about whether any row means anything — which is R4 (partial, spot-checks "owed on both sides"). An exact decomposition over unverified rows is exactly as "intrinsic" as an exact decomposition over random features; grading R2 strong on C1–C3/C7 while R4 is partial separates "computed from parameters" from "parameters mean something" and gives the first the credit due the second. The "strictly richer than fI/fU" function-class claim is also false for the noid arm: fUfI-noid *removes* fI's free user table and fU's free item table — it is not a superset of either.
>
> **10. The cycle's declared question is deferred, and the one genuine design decision is punted to a weak instrument (§0.3, §1c Seed B, §1d, §3.7, §3.9).** §0.3 says the "real job" is whether the straight merge survives the two-branch physics or the tie wants a different mechanism; §1d says that is Steps 3/3b/4b's job; 3b then lists every physics question as "empirical, adjudicated by instruments". Parsimony is applied asymmetrically — composing two sides is "one move" (waived), sharing one table is "two moves" (demoted) — though the shared table is the only decision LightFM did not already make. The deciding instrument is a Procrustes residual between an fI table (d=78) and an fU table (d=92) trained in *different models* with different hosts and other sides, on *centred* tables although the deployed model is uncentred; alignment between two foreign tables says little about whether one table inside one model is better.
>
> **11. Novelty claims that are not novel (§1b-3, §0.3 item 1).** "The first lineage stage with an INTERACT term between item words and user-basket words" — `lightfm_hist`-style plain-dot compositions already have exactly that bilinear form, and fI's u·t* interacts a free user with item words. "Both prototype sets grounded in the one item vocabulary" is an *index-level* statement under Seed A; the geometries are two, and M11 admits it. The one-vocabulary story the thesis wants lives in the ablation arm.
>
> **12. The Steck shield for prototype naming is weakened for the first time, and §5.5 says the opposite (§5.3, §5.5 "cosine units protected").** Prototype meaning is read as cos(e_f, p_k) from the word rows. On fI/fU those rows were shaped only through a cosine path; in the merge every e_f also receives dot-trained gradient through W. The naming profile is now read off vectors half-trained under a dot product — Steck's warning applies to the naming layer itself, and "the cosine units train through the cosine they read" no longer describes the rows they read.
>
> ## C. What breaks at scale
>
> **13. On ml-1m the item side is a SUM over variable-size bags, so tag count becomes a rank-active popularity proxy — unlisted among M7–M12 (§3.1, §3.6 M12, §4.1).** ‖q_i‖ grows with bag size (≤72 tokens, mean 24.3); B_item = (W^tᵀ1)·q_i and t̂ read that norm directly. Tag count in MovieLens correlates with rating volume, so the model gets a free popularity feature with no semantic content, and the explanation will dutifully name it as "+0.02 from each of 40 tags". M12 covers only the user side (which is mean-normalised). If the `@0.8` bags are Tag-Genome relevance thresholds, the "content" is itself computed from rating behaviour — the 23 %-of-parameters vocabulary on the load-bearing R6 dataset is partly compressed CF signal. The one control that removes the channel (`normalize_composed`) is scope-flagged, not default.
>
> **14. The gauge audit missed the global scale gauge that matters most for a design whose central hidden effect is "norms now score" (§3b-i/iii, §3.6).** (E^I, W^t) → (c·E^I, W^t/c) leaves S unchanged in every component (t* is scale-free in q_i; t̂ and B_item absorb c; the I-half never sees q_i's scale); likewise (E^U, W^u). Only L2 balances it. Hence absolute word-row norms are meaningful only relative within a table, "word-row norm vs frequency" instruments need a normalisation they are not given, and §3.6's claim that `max_norm` becomes "a direct cap on the class-level popularity channel" is partly illusory — W absorbs the cap. 3b-iii's "no per-entity dilation exists" is true and beside this point.
>
> **15. The user side is structurally near-inert on ml-1m and noisy on H&M, so the merge's two halves are each strongest on a different dataset (§3.1, §3b-vii/viii, §3.9, §4.5).** ml-1m users average ~165 ratings × ~24 tokens: q_u is the mean of thousands of token rows, shrinking toward the global mean (F-DC05-01) with the ID row dominating both roles — fUfI on the R6-load-bearing dataset is approximately fI-on-UI plus a user ID row. On hm_1_month, 25 % of users have 3 rows: LOO removes a third of the basket per step, so q_u's direction swings between consecutive positives and û reads that swing at full norm; |H|=1 users in the noid arm have q_u = 0 and train only the intercept. The `hm_kcore` decision is pushed to the user while wave 1 and the window close; and since the thesis is about fashion, the realistic Nov-3 outcome is a merge accuracy claim on movies and an explanation claim on H&M without a 100-trial row.
>
> **16. Sampled evaluation rewards exactly the channel the merge opens, and the document never mentions it (§2.1 protocol, §3.1).** 1 positive + 99 uniform negatives at HR@10 is known to inflate popularity-exploiting models relative to full ranking. The merge's new mechanisms — a rank-active composed intercept and norm-carried class popularity — are popularity channels. A fUfI gain under this protocol cannot be separated from a sampled-metric artifact unless a full-ranking or popularity-stratified read-out is in the plan; none is. In the noid arm on H&M, items sharing a 5-word tuple score identically on both halves, so a twin of the positive among the 99 negatives is a coin-flip — tie-handling in the evaluator is load-bearing and unstated.
>
> ## D. The experimental plan
>
> **17. The "built-in 2×2" is not a factorial and its control arm cannot carry the instrument it is assigned (§0.1 M6.4, §3.6 M7, §3.7).** fI and fU live on the I and U hosts, not the UI host, so {host U/I, host UI, fI/fU, fUfI} varies head and composition together; the clean factorial {free, composed}² on the UI host needs fU-on-UI, which is "optional". M7's control is "host UI = identical tie, free embeddings" — but direction-contamination R² and word-norm-vs-frequency need word rows the host does not have, so the control measures accuracy difference, not contamination. The length-ablation control closes ID-row refund, class-level norm popularity, ml-1m tag-count, and the user bag-size channel simultaneously, so it can show the length channel matters but cannot adjudicate refund-vs-contamination — the one pre-registered sign question it is said to "isolate".
>
> **18. fUfI vs UI changes more than "compose both sides" (§1d, §2.6, §3.7 gap rule).** Training-time LOO on the user side is a procedure the host does not run; pooled word rows change gradient statistics and the ASHA learning curve; the two ID rows reprice the function class under L2; the ml-1m vocabulary is 23 % of the parameters. The C4 keystone proves the plumbing at V=0 (where composition and LOO are never exercised), not that the comparison isolates one factor. The §3.7 gap-interpretation rule then lists five bundled causes for the ids-vs-noid gap and offers no way to apportion them — so the headline number, if obtained, will be read through a narrative, not a decomposition.
>
> **19. ID routing through the linear halves is the predicted equilibrium, not a "risk" — and R2 is graded before coverage is known (§3.6 M9/M10, §5.5, §5.1 R2).** The linear halves are the cheapest path for idiosyncratic signal: no normalisation, no prototype bottleneck, a direct bilinear with the other side, bounded only by L2. A 100-trial search on HR@10 will find it. dc08 already recorded dc05's grounding as "partly forfeited" when identity rows were added; the merge adds two unconstrained rooms for the same tenant. The honest headline is "nameable in principle, coverage to be measured", not R2 strong.
>
> **20. The prior-art verdict is a statement about the probe (§2b).** Five web queries, Semantic Scholar rate-limited, seven new citers classified "by title only", and a dc05 neighbour (TEASER) discovered only now — the probe's own disclosed method would not have found a close relative if one existed, so "nobody has put the two together" is overstated as a categorical.
>
> ## E. The explanation claims
>
> **21. "Exact counterfactual" is exact for one of four components; the full-score effect differs by up to eight orders of magnitude in the author's own verification (§3.4 item 3, §4b.3 C2(a) side note).** Removing word f changes the U-half by exactly u*ᵀW^t e_f, but the verifier reports the full-score change deviating from that by relative 1.4e-10 to 3.7e+8 because t* moves too. The fine print says "on this half"; the explanation surface says "Dark Blue +0.30" with no qualifier. A reader asked to accept this as a counterfactual on the *recommendation* would be right to refuse, and the user-side version is already conceded (C2(b)) to not be a counterfactual at all.
>
> **22. The intercept zoom's semantics are unlicensed, and the card is dominated by non-personalised mass (§3.1, §3.4 items 5–6 and the rendered example, §5.1 R7).** B_item = 1ᵀW^t q_i exists because the shifted cosine adds +1 — switch `cosine_type` to standard and it vanishes; reading (W^tᵀ1)·e_f as "Trousers Denim sells well this month" is a popularity story about an activation offset whose popularity correlation is listed as a *future* instrument. The rank-inert B_user is printed inside "Why for you?" — a number that changes no ranking shown as a reason. In the author's own illustrative numbers baselines are 67 % of the score and the card carries ~40 numbers across four components, two regroupings and two baselines; "nameable" is not "perceivable", and R7 partial is generous for a surface that explains a third of the score to a shopper in forty figures.
>
> **23. "Intrinsic" is asserted in the Rudin sense while every semantic check is deferred (§3.5, §3.4 item 7, §5.1 R4).** The grounding is unenforced on both sides, the profile-validity spot-checks are owed, the two tables may assign one word two meanings (M11), and the cross-side readings (route 1, vocabulary-mediated affinity) are flagged "enabled, not enforced — extrapolation caveat". What the candidate actually guarantees is *additivity over named indices*. Whether those indices name anything is the whole question, and the document's rating structure lets it be answered "partial" under R4 while R2/R3/S4 are scored "strong" on the same unknown.

### 5b.2 Response — point by point (concede / rebut / open risk); amendments A1–A16 below are applied as visible, dated edits

1. **R3 — concede, re-rated; rebut the "fiat".** The linear halves are prototype-indexed by position, not by similarity; the grade is now split (cosine units strong, linear halves partial, "graded as inheritance"). The keystone/like-for-like discipline is not this cycle's invention — it is the user-ratified staged-bar decision (dc01 SC.1b-delta P7); Seeds C/D are captured for a post-merge head cycle, which is the lineage's natural next stage. The "stitched across two geometries" sentence is real under Seed A (M11); the `_shared` arm removes it by construction; which is default is the user's decision (§3.9). **A1.**
2. **R2 — concede, re-rated.** (a) Individual bars are gauge-dependent, only their sums are invariant — now in the grade text and the renderer notes. (b) The realistic gauge is the *effective* one under low-rank P (UI ≈ 10 effective directions): F-DC01-08's under-spanning clause applies with force; instrument: slab spectrum of the trained model. (c) C6(c)/(d) are plain-SGD statements; under Adagrad/Adam the gauge component is not conserved — it remains **score-inert**, which is what identifiability needs; conservation was never load-bearing. (d) The example's per-value lines are illustrative; the thesis reads them at clique level (F-DC01-06) — caption added. **A1, A4, A6.**
3. **R1 — concede two errors.** LightFM's two tables are two *vocabularies*; it is no precedent for two tables over one vocabulary — §2.2 corrected; the two-table default is my choice, stated as one. "Literally" → "structurally": the paper ties a per-entity parameter, we tie a composition's output (I propose). R1 stays strong for the within-side exemplar; the grade text now says the cross-side tie is untied by default and is the user's decision. **A1, A7.**
4. **R5/R6 arms — concede.** The ml-1m evidence favours item-ids/user-noid; that mixed arm is promoted from "optional" to the evidence-favoured arm for the user's decision. The train/cold mismatch is unmitigated (K9 unstacked) but **not** unmeasured — the S0.3 cold-vs-all protocol is committed. **A2.**
5. **S1 — concede** (→ partial: designed, not in the fleet). **R8 — rebut**: R8 grades the mechanism, and the mechanism is one move; **concede the scope risk**: the controls are what a cut removes first. Minimum attributable set and priority order added. **A1, A2.**
6. **Head inference — concede; corrected.** The rows do not support "the head is what loses": `lightfm_hist` pairs the composition with a *free* item table; the honest decomposition is fU-noid vs `lightfm_hist` = 0.016 (head + item-table dimension) and fU-ids vs fU-noid = 0.047 (the user-ID row under this search). The merge's default arm carries two ID rows — the measured culprit — hence A2. "Exactly the channel the dot rows have" → "a norm-reading channel (bilinear, no free table)". **A3, A8.**
7. **Falsification — concede.** Per-arm expectations and a disconfirmation clause added to 3b-v (dc08 A11 precedent): the merge "adds nothing" if its best arm ≤ max(fI, fU) on ml-1m at n100; "fails the bar" if < UI − 2·SE (≈ 0.012 HR@10); a win with linear-half named coverage below the cosine units' is M9 realised, not the intended win. **A3.**
8. **Power / protocol — concede power, partially rebut the bar.** SE(HR@10) ≈ 0.006 on 6,034 users; single seed is the Nov-3 scoping decision and differences must be read against that SE — stated. The bar is *our* UI-host row, like-for-like by charter, not the paper's 0.657 (3 seeds, their tuning) — wording fixed. ASHA pruning of slow starters (pooled rows learn before any item is distinguishable): **open risk**, unverified; instrument: rung-survival/learning-curve comparison; the only guard is the profile's grace period. Search-box ceiling (fU at d=92, K=98 of 100): **open risk**; report optima against bounds; widening the box breaks like-for-like — user decision. **A3.**
9. **Toy ≠ semantics — concede both.** R2/R3/S4 are now explicitly *conditional on R4* in the grade text: the candidate guarantees additivity over named indices; whether the indices mean anything is R4's open question with owed spot-checks. The function-class sentence was wrong for noid: fUfI-noid is an additive-span restriction and contains neither fI's nor fU's class (each has a free side); ids equals the host's class. **A1, A9.**
10. **Deferred question / instrument — concede the instrument, partially rebut the rest.** Cross-model Procrustes between tables of different d is ill-posed; demoted from "pre-build gate" to weak prior; the proper instrument is the intra-model residual between fUfI's own two tables (same d), and the one-table decision becomes an arm comparison. A DC cycle designs tests and runs none (GR4): naming the physics questions with their instruments *is* the deliverable. Parsimony asymmetry is a choice (the staged plan), not a theorem — stated. **A10.**
11. **Novelty wording — concede**; §1b-3 and §0.3 corrected (first stage where both compositions interact bilinearly with *each other*; "one vocabulary" is index-level under Seed A). **A8.**
12. **Steck on the word rows — concede fully (new mode).** Every e_f now receives dot-trained gradient through W; the naming profiles cos(e_f, p) are read off rows trained half under a dot product. Added as **M14** with the profile-validity spot-check (fUfI vs fI/fU on the same vocabulary) as instrument; §5.3/§5.5 corrected. **A5, A14.**
13. **Item-side bag-size channel — concede fully (new mode M13).** On ml-1m ‖q_i‖ grows with token count and B_item, t̂ read it; tag count correlates with popularity. Instruments: ‖q_i‖ vs token count vs popularity, B_item vs token count; mitigations: the `normalize_composed` control, mean-normalised item bags (a composition change, own toy round), dc06 centring. The Tag-Genome caveat (whether tags@0.8 are behaviour-derived relevance thresholds) is **flagged for verification against the S0.5 record**, not asserted. **A5.**
14. **Global scale gauge — concede fully (new).** (E^I, W^t) → (c·E^I, W^t/c) and its user-side twin leave every component unchanged; only L2/`max_norm` break the symmetry. Consequences: absolute row norms are unidentified; norm instruments must be table-relative (normalise by the table's mean row norm or by ‖W‖); §3.6's "`max_norm` caps the popularity channel" is softened — W absorbs the cap, which then binds only through L2 on W. **A4.**
15. **Dataset asymmetry — concede as open risks.** ml-1m: q_u ≈ global mean + ID row for heavy users (fUfI ≈ fI-on-UI + user ID); hm: 25 % three-row users — LOO removes a third of the basket per step and û reads the direction swing at full norm; |H| = 1 noid users have q_u = 0 and train only through the intercept. Instruments added (per-step q_u direction change by |H| band; ID share of ‖q_u‖ by band). The realistic Nov-3 outcome (accuracy claim on movies, explanation claim on fashion without a 100-trial row) is the scoping risk already flagged in §3.9/§4.5 — the user's call. **A13.**
16. **Sampled evaluation — partially rebut, concede the omission.** The protocol is the charter's and shared by every row; a committed secondary read-out with popularity-sampled negatives exists (S0 charter C3, D3) and is now *required* for the merge rows; the tie policy is pinned and disclosed (F-S0-15), and the noid twin tie is the known equivalence-class ceiling (dc01 §3.6). Added to the run-plan notes. **A16.**
17. **2×2 — concede.** The clean factorial {free, composed}² on the UI host is {UI, fI-on-UI, fU-on-UI, fUfI}; fU-on-UI is promoted alongside fI-on-UI for the contamination question (arm count = user decision). Host UI controls the tie's *geometry* pathologies (cone, effective directions), not word-norm contamination — wording fixed. The length-ablation control shows that length matters, not its sign; the sign is read from the twin-separation and popularity-correlation instruments. **A11, A2.**
18. **Isolation — concede wording.** The host comparison isolates the *composition package* (both compositions with their LOO hygiene and repricing); the ladder apportions by side, the ids/noid pair by ID rows; the C4 keystone proves plumbing equality at V = 0, not factor isolation. **A15.**
19. **ID routing — concede.** Per-component named coverage is a primary read-out; a win with linear-half coverage below the cosine units' is folded into the disconfirmation clause as "M9 realised". R2 re-rated accordingly ("nameable in principle; coverage measured"). **A1, A3.**
20. **Probe — concede wording**: "found nothing close with this probe", limits stated. **A12.**
21. **Counterfactual label — concede.** The linear-half word zoom is labelled "exact for this half"; the only recommendation-level counterfactual is re-scoring with the row removed from q_i (both halves move) — offered as a disclosed reading; the user side has none (C2(b)). **A6.**
22. **Intercept semantics / card — concede partially.** The popularity sentence becomes conditional on the SC.8 correlation instrument (dc05 B(t) precedent); the +1 shift is the evidenced base setting (vault 2026-08-17: popularity's measurable home), so the channel is a design property, its *reading* empirical. B_user stays as arithmetic honesty, labelled "does not affect your ranking". Card density: figure default top-3 bars + "n more" summarised — a renderer default for the build; R7 stays partial. **A6.**
23. **Intrinsic vs semantic — concede the structure.** Written into the grades (A1): additivity over named indices is guaranteed; meaning is R4's question.

### 5b.3 Amendments made after 5b (visible, dated 2026-10-26)

- **A1** §5.1 table: R1, R2, R3, S1 re-rated with the narrowed scope written into the grade text; R2/R3/S4 marked conditional on R4.
- **A2** §3.7: mixed arm (item ids, user noid) promoted to evidence-favoured; fU-on-UI promoted for the contamination factorial; minimum attributable set + priority order under scope cuts.
- **A3** §3b-v: per-arm expectations, SE note, disconfirmation clause (incl. M9-realised condition); "paper-headline bar" → "the UI-host row".
- **A4** §3b-i: effective gauge under low-rank P; adaptive-optimizer note; new (x) global scale gauge per side; §3.6 `max_norm` statement softened.
- **A5** §3.6 + §5.5: new modes **M13** (item-side bag-size channel, ml-1m) and **M14** (Steck exposure of the word rows / naming layer), each with instruments and unstacked knobs.
- **A6** §3.4: linear-half zoom labelled "exact for this half"; full-score word counterfactual as a disclosed reading; intercept zoom's popularity reading conditional on the SC.8 correlation; example caption on clique-level reading; renderer default top-3.
- **A7** §2.2 / §3.3: LightFM two-table precedent corrected; "literally" → "structurally".
- **A8** §1b-3 / §0.3 / §5.1 R6: novelty and head-inference wording corrected.
- **A9** §3b-iv: noid function class corrected.
- **A10** §3.7 / §3.9: geometry instrument demoted; one-table decision = arm comparison.
- **A11** §3b-iii / §3.7: length-ablation control isolates "length matters", not the sign.
- **A12** §2b: verdict qualified.
- **A13** §3b-vii/viii: thin-user LOO swing and heavy-user inertness instruments.
- **A14** §5.3 / §5.5: Steck row corrected.
- **A15** §3.7: "isolates the composition package".
- **A16** §3.9: popularity-sampled-negatives secondary read-out required; tie policy pointer.

---

## 6. Fingerprint & registration

### 6.1 Mechanism fingerprint

- **Features-entry:** *both* compositions at once under the host's double tie — dc01's item
  composition (words + item-ID row) and dc05's user composition (basket-mean words + user-ID
  row, train-time leave-one-out) replace both free base tables of UI-ProtoMF; each composed
  vector is read in **two roles** (cosine against its own prototypes, free linear projection
  into the other side's prototype coordinates), so composition feeds the linear halves and
  the norm channel opens — the lineage's first non-cosine path from features to score.
- **Grounding:** unenforced shared geometry on both sides (cos(e^I_f, p^t_k), cos(e^U_f, p^u_l));
  **one shared word table** by user decision (2026-10-26) — one vocabulary AND one geometry, the lineage's first cross-side tie; two tables (`_sep`) as the arm; prototypes free; W free.
- **Read-out:** the four-component inventory, all nameable — fI's per-feature shares of t\*,
  fU's per-word/per-purchase shares of u\*, exact linear attribution of t̂ (read-out 4, a
  true counterfactual for its half) and of û (exact shares; the removal counterfactual
  Δ_j disclosed separately), the live intercept 1ᵀt̂ zoomed per word, per-component
  named/unnamed coverage; cross-side vocabulary-mediated affinity as a descriptive reading.

**Distinctness:** differs from every registered row on the **entry** axis — no single-branch
candidate reads a composed vector in two roles or routes features through a linear half;
dc01/dc05 distinctness waived (lineage merge, M1). Not `S1-contrast`.

### 6.2 Status and what the user is asked to decide

**Status: in scrutiny — SC.1a/SC.1b/SC.2 closed (2026-10-26; amendment bundle + toy round D1–D11, §6.4); next SC.3.** *(Original status line: draft — awaiting review, 2026-10-26.)* 4b: 11/14 claims confirmed outright *(re-scoped at SC.2 — F-DC09-01: the 4b round ran on the separate-table mechanism; the shared default's claims are the SC.2 round, §6.4)*,
three sub-clauses falsified and amended (C2(b) share≠counterfactual on the user linear
half; C10 regime; C14 wording); keystone C4 bit-identical with exact gradients. 5b: 23-point
critique; 16 amendments A1–A16 absorbed, among them four re-ratings (R2, R3, S1 down; R1
scope narrowed), two new degeneration modes (M13 item-side bag-size channel on ml-1m; M14
Steck exposure of the word rows), the per-side scale gauge (3b-x), the mixed ID arm
promoted, a disconfirmation clause, and the LightFM two-table "precedent" withdrawn.

**User review decisions (Matteo, 2026-10-26):** (1) word tables — **shared is the default**, `_sep` is the arm ("definitely test both"); (2) ID arms — **ids-on-both and noid-on-both only**, mixed arm optional; run plan = the 2×2 {ids, noid} × {shared, sep} per dataset, trim at the run-plan gate if budget bites; (3) `hm_kcore` **deferred**; (4) TEASER → the user's reading priority list (added to the reading list's ⚡ block); (5) pre-registration + disconfirmation clause (3b-v) **accepted** as the reading rule; (6) `ft_type` `feature_user_item_proto` accepted (CLAUDE.md edit at build, versioned); (7) two-sided LightFM control `lightfm_tags_hist` — charter-amendment request stands, to be decided at SC.6. Next: `/scrutiny dc09`.

Decisions as posed at the end of the cycle (for the record):
1. **Table default** — separate (this candidate) or shared (`_shared`, Seed B); the geometry
   comparison is only a weak prior (A10); the arm decides.
2. **Arm set and priority** — the evidence-favoured mixed arm (`_itemid_only`: item ID on,
   user ID off) vs ids vs noid; fU-on-UI / fI-on-UI factorial; the two-sided LightFM control
   `lightfm_tags_hist` (charter amendment); length-ablation. Minimum attributable set per
   dataset: UI host + best fUfI arm + `lightfm_tags_hist` (§4.5).
3. **`hm_kcore`** — build for the merge or not (masterplan gate).
4. **`ft_type` name** `feature_user_item_proto` (CLAUDE.md's `dual_item_proto` stays reserved
   for its own idea) — CLAUDE.md edit at build time, with versioning.
5. **Pre-registration** — accept 3b-v's per-arm expectations and disconfirmation clause as
   the run plan's reading rule.
6. **TEASER** (De Pauw et al. 2022) — add to the fU chapter's related work and dc05's
   positioning (M7 trickle-back).

### 6.3 Build notes for whoever implements this (not design)

- `PrototypeEmbedding.sim_from_embed(o_embed)`: split today's `forward` tail into a method;
  `forward` unchanged in behaviour — re-run the dc01 t03 RNG/golden pin.
- `TwoRoleComposedExtractor(base, proto, out_dimension, invert, normalize_composed)`: one q per
  side per step; `supports_loo` delegates to `base`; `init_parameters` touches only its
  linear layer; `get_and_reset_loss` forwards `proto`'s accumulator.
- Factory branch must **not** mutate the config dicts (the host's `prototypes_double_tie`
  overwrites `ft_type` in place — saved configs show `'embedding_w'`); inject both payloads.
- Arm names *(fixed at review 2026-10-26)*: `feature_user_item_proto` (ids, shared table), `_noid` (noid, shared), `_sep` (ids, separate tables), `_noid_sep`; optional `_itemid_only`; `_norm` (length control). Follow
  dc08's naming lesson — no `_noid` beside a present ID row.
- Keystone tests (dc_checks/dc09): t01 weight-copy bit-identity vs `user_item_proto`
  (scores f32/f64, loss, grads — 4b C4); t02 LOO reaches both roles (C8); t03 `normalize_composed`
  no-op on the cosine units (C9); t04 renderer self-check (both baselines + four components);
  t05 cold ID drop covers both roles; t06 four-component decomposition (C1) on a trained
  smoke checkpoint.
- Renderer: `extra_baselines` optional field; slot `compute_breakdown_feature_user_item_proto`
  + host slot `compute_breakdown_user_item_proto`; labels per A6.
- Smokes on the LEO5 login-node A30s (policy); runs at concurrency 5; cluster-checkout lock
  until wave 1 has started; worktree `feat/fufi`.
- *(SC.2, 2026-10-26)* Shared block: `E_words` created once by the factory and passed to both compositions (`word_table=`); the single-table path must stay bit-identical (dc01 t03 + dc05 keystones); one `max_norm` key; `_sep` = two blocks. Worktree note: the cycle moved into the main checkout (`feat/scrutiny`); `feat/fufi` is unused.
- *(SC.2)* Cold: register `feature_user_item_proto` in `config_expects_id_drop` / `config_expects_user_id_drop`; the finders reach the composition through the wrapper's `base`; convention = **mean-ID imputation** (F-DC09-08) on both sides, empty-basket users stay q_u = 0; S0.3 §5 amendment (done) + `cold_eval` change + dc_checks/s0 pin; the dc01-checkpoint instrument runs before wave 2's cold jobs are submitted.
- *(SC.2)* New keystones: t07 sharing keystone (shared ≡ `_sep` tied; scores equal, summed gradients; sampled softmax + BCE); t08 canonical-gauge invariance of rendered shares on a gauge-shifted checkpoint (SC.5 owns it); t05 re-targeted to the imputation convention.
- *(SC.2)* One-off descriptive measurements (post-hoc on trained checkpoints, login-node class, once per dataset, no loops): the two-way decomposition (F-DC09-09); same-word vs different-word pair share + twin fraction (F-DC09-10); cross-prototype cosine row-max (F-DC09-11). No per-user hit-vector wiring (F-DC09-12 rejected).

### 6.4 Scrutiny amendments applied at SC.2 (visible, dated 2026-10-26; findings F-DC09-01..12 in `Master/docs/scrutiny/findings_ledger.md`; dossier `sc09_fufi_double_tie_merge.md`)

- **B1** §3.2 / §6.3 — shared-table build spec (F-DC09-01): one word block, single-owner init, per-side ID blocks, cold finders + `config_expects_*` registration, sharing keystone t07.
- **B2** 3b-x / 3b-i — per-side scale gauge false under sharing, the joint gauge survives; centroid identity; "init noise" → optimiser-path-dependent (F-DC09-02).
- **B3** §3.6 / §5.5 — M15, M7-shared (+ the BCE-active B_user), M11 re-scoped to `_sep` (F-DC09-03; instruments dropped by the user's framing decision).
- **B4** 3b-v / §3.8 / §4.5 / §5.1 / §5.3 / 3b-viii / §6.2 — consistency with the review decisions; grade texts R1, R2, R5, R6, R8, S1, S4; safety row corrected (F-DC09-04).
- **B5** §3.6 M13 / §3.5 / §5.1 R6 — Tag-Genome provenance verified TRUE, disclosure (F-DC09-05).
- **B6** §3.6 M14 / §5.3 — Steck precision (F-DC09-06).
- **B7** 3b-xi / §3.4 / §5.1 S4 / §5.5 — translation gauge of the ids arms; canonical per-field-centred read-out gauge; substitution counterfactual (F-DC09-07, major).
- **B8** §3.4 / §3.8 / §5.1 R5 + S0.3 §5 (foundation doc) — invariant cold convention, mean-ID imputation; empty-basket users unchanged (F-DC09-08, major, shared).
- **B9** §3.1 / 3b-xii / §3.6 M8 / §5.5 — common-mode mass; one one-off two-way decomposition; renderer line at SC.5 (F-DC09-09, trimmed).
- **B10** §2b / §3.6 M16 / §5.5 — FM/AFM/SVD++ positioning; overlap + twin channel; one one-off measurement (F-DC09-10, trimmed).
- **B11** §3.6 M18 / §3.4 7(b) / §5.5 — prototype-set redundancy hypothesis; one one-off measurement; conditional caveat (F-DC09-11, trimmed).
- **B12** 3b-v — F-DC09-12 REJECTED by the user; record only.
- **B13** 3b-x / 3b-xi / 3b-xii — toy-round consequences: the cross-side ratio is identified in noid-shared only and the user-side word/ID split is pinned by leave-one-out alone in ids arms (F-DC09-13, disclosure); precision clauses P1 (3b-xii, D8) and P2 (3b-xi, D5).
- **SC.2 toy round** (`Master/temp/dc_checks/dc09/claims_spec_sc2.md` → `check_claims_sc2.py`, `report_sc2.md`): claims D1–D11 — results and any consequence recorded in the dossier's SC.2 section and summarised here: D1–D3, D5–D7, D9 CONFIRMED (D1 bitwise in f64 and f32; D7 shows the "drop" convention flipping cold rankings under the gauge, mean-ID imputation invariant); five sub-clauses FALSIFIED — D4(d) a real error in the F-DC09-02 amendment (→ B13, F-DC09-13), D8(b) a wording gap (→ P1), D4(a)/D10/D11(a) spec-wording or recipe errors with no design-doc consequence. SC.2 gate closed 2026-10-26 (user: "yes i accept").
