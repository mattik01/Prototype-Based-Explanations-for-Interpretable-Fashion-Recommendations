# dc08 — Item-identity history rows in the fU composition (working title)

> Design Candidate Protocol v1.3 (`Master/docs/design_candidate_protocol.md`).
> Directed cycle, user-invoked 2026-09-09: *"for the scratchpad entry about the
> history ids for fu"* → seed is the open scratchpad entry
> `[captured nl-track abandonment] Item-identity rows inside the fU history
> composition (2026-09-09)`.
> Lineage stage: **fU** (U-ProtoMF host, dc05). Sibling-cycle precedent for the
> waived distinctness rule: dc05-M1 → dc06 (fI′) → dc07.

> ---
>
> ### ⚠️ READ THIS FIRST — amendment index (added at the Step-6 cold-read gate)
>
> The adversarial pass (§5b) changed this design. **§§0–5 are left as written**
> (the protocol forbids silently retrofitting them), so the sections below are
> superseded in the listed respects by amendments **A1–A12 in §5b.2**, which are
> binding on the build and the run plan:
>
> | amendment | supersedes | in one line |
> |---|---|---|
> | **A1** | §3.4 | coverage read-out restricted to a canonical gauge: three-way split **arm C only**; arm D gets the two-way named-vs-unnamed split (the invariant part); identity-block mean removed at read time |
> | **A2** | §3.6 | new **capacity-control arm**: identity block present but frozen at random init — C − A_frozen is the mechanism contrast |
> | **A3** | §3.6 | **both** regime hypotheses pre-registered (mine: gains where histories are long; NAIS §4.3: gains where data is sparse) with the discriminating outcome named |
> | **A4** | §0, §2.3 | motivation's sign corrected — **FISM 66.47 loses to MF-BPR 66.64** on ml-1m; first negative literature datum |
> | **A5** | §4.5 | the inherited **+0.02 HR@10 gate is dropped**, replaced by a measured ≥3-seed noise band |
> | **A6** | §2b, §4.5 | the cosine-discards-magnitude property is a **confound under measurement**, not a contribution; dot twin becomes a **paired 2×2** (head × identity) |
> | **A7** | §3.6, §4.5 | **H&M is a capacity-pathology arm, non-diagnostic** (874 K new params vs 476 K observations) — it is no longer "the falsification arm" |
> | **A8** | §3.5, §4.5 | dilution/popularity instruments become **required**; **K_ycenter named in advance** as the first repair |
> | **A9** | §3.4 | identity line labelled **per item by measured buyer count**; a second, collapsed-coverage rendering is required |
> | **A10** | §4.3, §4.4 | flat `embedding_bag`/offsets layout **required on ml-1m** (the (B, D_max′, d) autograd intermediate ≈ 708 MB was not counted) |
> | **A11** | §5.2, §4.5 | explicit **disconfirmation clause** — no gain beyond the noise band on ml-1m = the accuracy bet failed; the "measurement instrument" framing may not rescue it |
> | **A12** | §2b, §3.4 | the coverage split is a **lineage-wide read-out repair to backport to dc01/dc05**, not dc08's contribution |
>
> **Two strategic questions for the user** (§5b.2 points 13 and 17): fU's *head*
> penalty (0.612 under a dot vs 0.544 under the prototype head, same composition)
> is ~3× the gain this candidate hopes to buy — so the head, not the composition,
> may be where fU actually loses; and this is the third consecutive sibling-cycle
> knob on an existing host after dc07 stopped as a controlled negative.
>
> ---

---

## 0. Frame

**Occupied fingerprint space (from `candidate_index.md`, all rows read).**

| id | features-entry | grounding | read-out |
|----|----------------|-----------|----------|
| dc01 (fI) | item content words → item factors (sum replaces item-ID table) | unenforced shared geometry cos(e_f, p_k) | per-prototype contribution + exact per-feature shares |
| dc06 (fI′) | same entry, gauge-fixed (per-field-mean-centred summands) | same, read in the centred gauge | dc01 read-outs + within-field contrast shares |
| dc02 | item branch = attribute bottleneck (no per-item params) | by construction (prototype = attribute-value profile) | per-prototype + per-attribute shares |
| dc03 | frozen image content, no item-ID table | exemplar identity via push | case-based exemplar sheet |
| dc04 | prototype *parameterization* (P = softmax(A)·C over live class means) | data anchoring | per-anchor shares |
| dc05 (fU) | user factors composed from the attribute words of the purchase history (+ free user-ID row) | unenforced shared geometry of USER prototypes in ITEM vocabulary | per-prototype t_l·u*_l + per-word AND per-purchase shares |
| dc07 | entry unchanged | grounding unchanged | read-out changed (softmax/sigmoid memberships) — **stopped, controlled negative** |

**Covered vs. open.** The *features-entry* axis is densely occupied on the
content side: every registered candidate that touches entry replaces or
re-parameterizes representations with **content/attribute** material (words,
images, class means). What no registered row carries is a **collaborative
identity channel inside a composed representation** — i.e. a summand that is
neither a content word nor an entity's own free row, but *another entity's*
free row entering through the interaction set. dc01's item-ID row e_ID(i) and
dc05's user-ID row e_ID(u) are both **self**-rows: each is learned from that one
entity's own interactions, so neither can carry "these *other* users bought the
same exact item". That slot — SVD++/FISM's y_j — is the open niche this cycle
targets, on the fU host.

**Requirement served poorly by the current lineage state.** Two rows:

- **R6 (no accuracy regression / accuracy bar).** dc05's fU is the lineage's
  weakest accuracy story: on ml-1m the content-only history mean scores
  HR@10 0.612 (`lightfm_hist`) / fU 0.544 against a U-host that the frozen
  reference table puts far above; the nl-track research memo
  (`Master/llm_usage/abandoned_tracks/nl_track_2026-09-08/research/user_side_algorithms.md`
  §6, §8) records FISM at 66.5–70.2 HR@10 on ml-1m under a 1+99 protocol *from
  item identity alone, with no user row and no nonlinearity*. If that ordering
  survives our protocol, the missing channel — not composition cleverness — is
  where fU's headroom lives. That conclusion is exactly what the abandoned
  nonlinearity track returned as its **one keeper**.
- **R5 (sparsity robustness), user side.** fU's cold/thin-user claim currently
  rests entirely on content words; a thin-history user gets a thin bag. Item
  identity rows are a *different* kind of evidence for the same user and shift
  what "thin" means (an item bought by few people is thin in identity but may be
  rich in words, and vice versa) — the honest R5 reading changes and must be
  re-derived, not inherited.

**Open scratchpad entries that look relevant to this cycle** (full one-line
dispositions belong to Step 1b; this is the Step-0 relevance flag only):

- `[captured nl-track abandonment] Item-identity rows inside the fU history
  composition (2026-09-09)` — **the seed** (user-directed).
- `[captured dc06] fU′ — field-mean centring of the dc05 history composition
  (2026-08-24)` — same host, same composition, *different* single variable;
  the R8 boundary between the two must be stated, not assumed.
- `[captured dc05] Comparing fI's and fU's learned word geometries` and
  `[captured dc05] The merge inherits one live and one inert baseline mass` —
  both are fUfI-merge entries that this cycle *adds a table to*: a new
  item-keyed table in the user branch is a new candidate for cross-side tying
  (the seed entry already flags this as a merge-stage knob).
- `[captured scrutiny dc01] The ID residual vector as a diagnostic read-out` —
  the new rows y_j are unnamed by construction, so its "opaque residual" problem
  reappears at a new place and its remedies (prototype-coordinate reading first,
  sparse vocabulary fit second) are the available prior art.
- `[captured scrutiny s0] Realistic 3-scenario cold-start testbed` and
  `[captured scrutiny dc05] Cold-user degradation floors` — the R5 re-derivation
  above lands on both.

**Host / lineage note.** This is a **directed sibling cycle** on the dc05 host,
in the shape dc06 took on the dc01 host: one added mechanism inside an existing
`ft_type`'s composition, not a new fingerprint region on grounding or read-out.
Distinctness from dc05 is therefore waived by the same precedent; distinctness
*within this cycle's shortlist* still binds, and the R8 argument (one idea, not
"fU + centring + rows") is a live gate, not a formality.

---

## 1. Seed selection

### 1a. Corpus sweep (targeted, per the fU brief's M2)

This is a directed sibling cycle on an existing host, so the sweep is targeted
at the one axis in question — *what may be a summand of a composed user
representation* — rather than an all-themes pass.

- **Theme B (backbone).** B3 SVDFeature and B5 LightFM are the lineage's
  standing entry mechanism (features → factors) and are already spent on dc01/
  dc05: both compose an entity from **content** rows. B4 Factorization Machines
  models feature×feature interactions, not identity-in-history. **B1 CMF**
  (Singh & Gordon) is the closest corpus relative in spirit — one entity's
  factors informed by a second relation — but it shares a *matrix* across
  relations rather than injecting an item row into a user vector. **Gap
  confirmed: the corpus has no interaction-set-as-representation model.** The
  literature that does is SVD++ (Koren, KDD 2008) and FISM / NAIS (Kabbur et al.
  KDD 2013; He et al. TKDE 2018) — **not in the reading list**, hence a
  `corpus+external` cycle with a Step-6 reading-list addition.
- **Theme A.** A2 ACF weights *anchors* by a user-conditioned softmax; it is a
  read-out mechanism (and dc07 just ran that axis to a stop), not an entry one.
- **Themes C/D/E/F/H.** No new material bears on the summand question: C/D are
  prototype-vehicle themes (D1 ProtoCF is the nearest — few-shot item prototypes
  from *neighbourhoods* — a Step-2b prior-art probe target rather than a seed);
  E is grounding/attention; F is geometry (already consumed by dc06); H is
  visual (dc03, abandoned).
- **§I forward citations.** Re-checked at row level: I1–I5 remain CF-only
  extensions of ProtoMF (embedding enhancement, cultural bias, prototype
  connection matrix, hierarchy, debiasing). None composes a user from item
  identity. The §I claim stands as of this sweep; a fresh forward-citation
  re-count is Step 2b's job, not Step 1's.

### 1b. Scratchpad sweep — every open entry dispositioned

| entry | disposition |
|---|---|
| `[captured nl-track abandonment]` Item-identity rows inside the fU history composition (2026-09-09) | **seed this cycle** — the user's directed invocation; the whole candidate. |
| `[captured dc06]` fU′ — field-mean centring of the dc05 composition (2026-08-24) | **not this cycle (R8)** — same host and same composition, but a *different* single variable (gauge fix vs. new channel). Stacking both would be two ideas; and its own caveat ("dc06's D2 lesson transfers — ID rows can re-house the removed component") gets *sharper*, not resolved, once a second unnamed row type exists. Sequencing note recorded for the user: if both land, centring must be re-derived over a composition that now contains unnameable summands. |
| `[captured scrutiny dc01]` The ID residual vector as a diagnostic read-out (2026-07-11, + two refinements) | **shapes this candidate** — the y_j rows are unnamed by construction, so this entry's problem (opaque learned residual) and its resolved method (prototype-coordinate reading = exact, canonical, K numbers; sparse vocabulary fit = second, ill-posed-when-overcomplete probe) transfer verbatim to the new table and become §3's name-coverage machinery. |
| `[captured dc05]` The merge inherits one live and one inert baseline mass (2026-07-12, + extension) | **shapes this candidate (bounded)** — the U-host's live baseline B(t)=1ᵀt is untouched by a change to the *user* composition, but the entry's general rule ("wherever a shifted-cosine side is scored against free per-item coefficients, a live item intercept exists — check which side is free") must be re-run because this cycle adds a *second* item-keyed free table. Step 3b's job. |
| `[captured scrutiny dc05]` Cold-user degradation floors (2026-07-13) | **shapes this candidate** — the degradation-floor probe is exactly the instrument that tells whether identity rows fill the U-host's open popularity channel (a y_j table is a per-item free parameter set: prime popularity real estate). Its gate decision (dataset strategy unchanged) stays binding; nothing here re-opens it. |
| `[captured scrutiny s0]` Realistic 3-scenario cold-start testbed (2026-07-11) | **not this cycle (why)** — unchanged prerequisite for cold-*user* claims, and this candidate does not create it. Recorded consequence: identity rows are a **cold-item liability** (a new item has an untrained y_j entering every buyer's vector), which is an R5 statement this cycle must make honestly in §5 without the testbed. |
| `[captured dc02]` Loss-enforced grounding: profile per CF prototype + consistency loss (2026-07-07) | **not this cycle (why)** — grounding-strength axis; orthogonal to the summand question and a full second mechanism (R8). |
| `[captured dc03]` Part/patch-level visual prototypes (2026-07-07) | **not this cycle** — visual granularity; dc03 lineage abandoned, entry survives as future work. |
| `[captured dc03]` Multimodal frozen content space (image ⊕ attributes) (2026-07-07) | **not this cycle** — same reason; also an explicit ensemble-risk entry under R8. |
| `[captured dc04]` Differentiable (live) centroids = REGULARIZE with a read-out (2026-07-07) | **not this cycle** — prototype-parameterization axis, reserved (per the dc04-abandonment entry) for the interpretability-targeted future stage. |
| `[captured dc03 abandonment]` Garment photos as explanation RENDERING (2026-07-13) | **answered inline** — orthogonal and compatible: a purely presentational layer. Small positive interaction worth noting: per-purchase zoom lines are exactly where an identity summand would be rendered, so photos would show *the item whose identity row fired*. No mechanism claim either way. |
| `[captured dc04 abandonment]` Data-anchored prototype parameterization (2026-07-13) | **not this cycle** — explicitly parked to the post-merge interpretability stage; revival condition (solve identifiability first) unmet. |
| user entries 2026-06-11/06-16 (attach profiles; injection fork; add/interact/regularize; attribute-source prioritization; sequencing; two-axes caution; hybrid desideratum) | all carry prior dispositions (dc02/dc03/dc04/dc06). Two are re-applied here as standing rules rather than re-dispositioned: **"three ways attributes can enter"** — this candidate is none of the three (identity rows are not attributes at all, a fourth entry kind worth naming); **"design caution: two axes + Rudin"** — gut-check applied below. |
| `[captured dc05]` Comparing fI's and fU's word geometries (2026-07-12) | already answered at dc06; noted only because this cycle adds a table that would join that comparison at the merge. |

### 1c. Shortlist

**Seed A — SVD++/FISM item-identity rows as summands of the dc05 composition**
(external: Koren 2008 SVD++; Kabbur et al. 2013 FISM · corpus host: dc05 = B5×B3×A1 on U-ProtoMF).
Mechanism in one line: give every item a second, free vector y_j and add it to the
composition of every user who bought it, alongside that item's attribute words —
`q_u = mean_{j ∈ H_u \ {i}} ( Σ_f e_f(j) + y_j ) + e_ID(u)`.

- *Motivation gate:* serves **R6** (the lineage's largest measured accuracy gap
  is on fU/ml-1m, and the external precedent attributes several HR points
  specifically to item identity in the history) and **R5** in a re-derived form
  (a second, independent evidence kind per user — thin-in-words users may be
  rich-in-identity), while **R2/R3/R4 are put explicitly at risk** and must be
  paid for with a name-coverage read-out rather than assumed.
- *Distinctness:* **dc01/dc06** — item-side hosts, content rows only, no
  cross-user channel. **dc02** — item branch is an attribute bottleneck with no
  per-item parameters at all (the exact opposite move). **dc03** — frozen visual
  content, no free identity table. **dc04** — features enter at the prototype
  parameterization, entities untouched. **dc05** — same host, and distinctness
  from it is *waived* by the sibling-cycle precedent (dc05-M1 → dc06 → dc07);
  the substantive difference is a new summand kind (another entity's free row)
  where dc05 has only content words + a self-row. **dc07** — read-out axis
  (membership similarity), entry untouched.
- *R8 parsimony:* **one idea** — "another entity's identity is a summand of my
  representation" — SVD++'s single move, at `user_item_proto` granularity. The
  two attached obligations are not second ideas: leave-one-out masking is a
  correctness condition of the same summand (the positive's own row must not sit
  inside its own predictor), and name-coverage reporting is a disclosure rule on
  the existing read-out, not a mechanism.

**Seed B — NAIS target-conditioned attention over the history**
(He et al. TKDE 2018). Weight each history slot by its similarity to the
*candidate* item, with the β-smoothed softmax denominator.

- *Motivation gate:* serves R6 (measured +1.6–3.2 HR points over FISM on ml-1m
  in the authors' own code) and arguably R2 (per-slot weights look like an
  explanation).
- *Distinctness:* differs from every registered row on the read-out axis
  (per-candidate weighting) — genuinely open space.
- *R8 parsimony:* **fails here.** It is a weighting *mechanism* on top of
  whatever the summands are, so relative to this cycle's question it is a second
  idea; it silently changes the explanation contract from a per-user profile to
  a per-candidate one; its measured gain depends on a magnitude trick (β<1) that
  the host's shifted cosine cannot see; and E4a/E4b (attention-is-not-
  explanation) plus dc07's just-recorded negative make "attention weights = the
  explanation" the weakest available claim in this codebase. **Rejected before
  deep read**, exactly as the parsimony filter intends. The nl-track memo ranked
  it last for the same reason.

**Seed C — reuse the host's own free item vector as the history row**
(no new table: `q_u = mean_j ( Σ_f e_f(j) + t̂_j ) + e_ID(u)`, with t̂_j the
U-ProtoMF item vector that already exists in R^{K_u}).

- *Motivation gate:* same R6 target at **zero parameter cost**, and it is the
  strongest realization of R1's tie exemplar available on a single-branch host
  (one table read by both the scoring side and the composition side).
- *Distinctness:* differs from dc05 by the same new-summand-kind axis as Seed A.
- *R8 parsimony:* one idea. **Rejected on mechanism, not parsimony:** t̂_j lives
  in *prototype-coordinate* space (dimension K_u, the slab the user vector is
  compared against), while the composition lives in the *word/embedding* space
  that feeds the prototype layer — the two are different spaces with different
  dimensions, so the reuse is not even type-correct without an added projection,
  which is the second idea again. SVD++ itself keeps y_j separate from the item
  factor table q_i for the analogous reason. Recorded as a **merge-stage knob**
  (at fUfI both sides have word-space item tables and the tie becomes
  type-correct) — captured to the scratchpad rather than pursued.

### 1d. Chosen seed and provenance

**Chosen: Seed A.** It is the only shortlist entry that is simultaneously (i) the
user's directed seed, (ii) one clean summand-level idea at the lineage's
granularity, (iii) aimed at the lineage's largest open accuracy gap with an
external precedent measured on *our* dataset under a candidate-count-matched
protocol, and (iv) genuinely uncovered by every registered fingerprint.

**Provenance: `scratchpad+corpus+external`** — seeded by the user's captured
scratchpad entry, hosted on a corpus-grounded candidate (dc05 = B5 LightFM ×
B3 SVDFeature × A1 ProtoMF), with the summand mechanism taken from external
literature (SVD++, FISM; NAIS as the near-relative to distinguish from) that
Step 6 adds to the reading list. Every claim attributed to those papers is
verified in Step 2 against the sources themselves — the nl-track memo's own
flag that the SVD++ formula there was reproduced *from memory* is treated as
unverified until Step 2 closes it.

**Two-axes gut-check** (standing scratchpad rule, 2026-06-16): this is **Axis A
work with an honest interpretability cost**, not Axis-B expressiveness — the
scorer is untouched and the representation changes, but the added summand is
*unnameable*, so unlike dc01/dc05 it moves the grounding fraction **down** while
(hypothetically) moving accuracy **up**. Under the Rudin axis that is a cost
that must pay for itself, and the candidate is therefore framed from the start
as a **priced trade**, with the explained-fraction read-out as the price tag —
not as a free win.

**Working `ft_type` proposal:** knobs on the existing `feature_user_proto`
(a `history_id_rows` flag + its LOO masking), mirroring the dc06/dc07 precedent
of adding knobs rather than a new `ft_type`; the four-arm design (words only /
+user row / +item history rows / +both) is then a config matrix, not four models.
Confirmed or revised in Step 3.

---

## 2. Mechanism extraction

Sources actually read this cycle (no from-memory citations; the nl-track memo's
own ⚠️ "SVD++ formula from memory" flag is hereby **closed by first-hand
verification**):

| ref | how it was read |
|---|---|
| **Koren, KDD 2008**, *Factorization meets the neighborhood* | full text, open mirror `people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf` (pp. 426–434 as published). Eq. (13), (15), §4 benefit list, Table 1 read directly. |
| **He, He, Song, Liu, Jin, Chua, TKDE 2018**, *NAIS* | full text, arXiv:1809.07053 PDF. Eqs. (3), (4), (5), (9); §4.1 protocol; Tables 1, 3, 5, 6. |
| **Kabbur, Ning & Karypis, KDD 2013**, *FISM* | ⚠️ **secondary source only** — the KDD PDF is paywalled; the prediction rule was verified from W.K. Pan's SZU lecture slides (`csse.szu.edu.cn/staff/panwk/recommendation/IRT-Ch04-OCCF-Slides/FISMauc.pdf`, slides 5 and 7), cross-checked against NAIS Eq. (3), which reproduces the same model. **Marked as a fidelity gap to close** before any thesis sentence leans on FISM specifics beyond what NAIS independently states. |
| **dc05** (`dc05_history_composed_user_factors.md` §3.1–3.4, 3b) | the host; read in full for the composition, the read-outs, and the B(t) asymmetry. |

### 2.1 SVD++ — the seed mechanism (Koren 2008)

Koren builds up in three steps, and the *third* is the one this candidate takes.

**Step 1 — Asymmetric-SVD (Eq. 13).** The user's free factor vector is *deleted*
and replaced entirely by the items they touched:

> r̂_ui = b_ui + q_iᵀ ( |R(u)|^{−1/2} Σ_{j∈R(u)} (r_uj − b_uj)·x_j + |N(u)|^{−1/2} Σ_{j∈N(u)} y_j )   — Eq. (13), §4

with three per-item factor tables q_i, x_i, y_i ∈ R^f; R(u) = items u rated,
N(u) = items for which implicit feedback is available. Koren lists four benefits
in §4, of which **#3 is directly load-bearing for this thesis**, quoted:
*"Latent factor models such as SVD face real difficulties to explain
predictions. After all, a key to these models is abstracting users via an
intermediate layer of user factors. This intermediate layer separates the
computed predictions from past user actions and complicates explanations.
However, the new Asymmetric-SVD model does not employ any level of abstraction
on the users side. Hence, predictions are a direct function of past users'
feedback. Such a framework allows identifying which of the past user actions are
most influential on the computed prediction, thereby explaining predictions by
most relevant actions."* (§4, benefit 3.)

**Step 2 — SVD++ (Eq. 15), the seed proper.** Koren then puts the free user row
*back in*, alongside the implicit-feedback sum:

> r̂_ui = b_ui + q_iᵀ ( p_u + |N(u)|^{−1/2} Σ_{j∈N(u)} y_j )   — Eq. (15), §4

*"Now, a user u is modeled as p_u + |N(u)|^{−1/2} Σ_{j∈N(u)} y_j. We use a free
user-factors vector, p_u, much like in (12), which is learnt from the given
explicit ratings. This vector is complemented by the sum … which represents the
perspective of implicit feedback. We dub this model 'SVD++'."* (§4.)

**Step 3 — what Koren says it costs.** Immediately after: *"SVD++ does not offer
the previously mentioned benefits of having less parameters, conveniently
handling new users and readily explainable results. This is because we do
abstract each user with a factors vector. However, as Table 1 indicates, SVD++
is clearly advantageous in terms of prediction accuracy."* (§4.)
**Table 1** (Netflix RMSE, lower better): SVD 0.9046 / 0.9037 / 0.9000 →
Asymmetric-SVD 0.9037 / 0.9013 / 0.9000 → SVD++ 0.8952 / 0.8924 / 0.8911 at
50 / 100 / 200 factors. Training: plain gradient descent on the regularized
squared error (Eq. 14 for Asymmetric-SVD), 30 iterations, step 0.002, λ₅ = 0.04
on Netflix (§4).

**This is the accuracy-for-explainability trade named by the author of the
mechanism, in the same paragraph, with the numbers attached.** It is the exact
trade this candidate re-runs in prototype space — and the reason §1 framed the
candidate as a priced trade rather than a free win.

*Verification note:* Koren's y_j is defined over the **implicit-feedback** set
N(u), which on Netflix he admits is weak — *"we do not really have much
independent implicit feedback for the Netflix dataset"* (§4) — and in the
Netflix runs N(u) is effectively the set of rated items. In our setting all
feedback is implicit, so N(u) = R(u) = the train basket; the distinction
collapses and the SVD++ sum becomes literally "the items you bought".

### 2.2 FISM — the same summand without any user row (Kabbur et al. 2013)

Prediction rule (SZU slides 7; ⚠️ secondary source), for a **training** item
i ∈ I_u:

> r̂_ui = b_i + Ū_u^{−i}·V_iᵀ,  Ū_u^{−i} = |I_u \ {i}|^{−α} · Σ_{i'∈I_u\{i}} W_{i'·},  0 ≤ α ≤ 1

and for a **candidate** item j ∉ I_u:

> r̂_uj = b_j + Ū_u·V_jᵀ,  Ū_u = |I_u|^{−α} · Σ_{i'∈I_u} W_{i'·}

Two things matter and both are verifiable in NAIS's independent reproduction:

1. **The target-item exclusion is part of the model, not an implementation
   detail.** NAIS Eq. (3) — *"ŷ_ui = p_iᵀ( |R⁺_u|^{−α} Σ_{j∈R⁺_u \ {i}} q_j )"*
   with *"The symbol \{i} corresponds to the constraint of diag(S) = 0 in
   Equation (2), to avoid the modeling of self-similarity of the target item"*
   (§2.2). NAIS footnote 1: *"The bias terms in the original paper are omitted
   for clarity"* — i.e. FISM proper carries b_i, NAIS drops it.
2. **Two tables per item, by design.** NAIS §2.2: *"Note that in FISM, each item
   has two embedding vectors p and q to differentiate its role of a prediction
   target or a historical interaction, which can also increase model
   expressiveness."* The history row and the scored row are **deliberately not
   the same vector** — which is precisely the type-correctness point that killed
   shortlist Seed C on the fU host.
3. **α is a length-normalisation knob**, not a weighting of items: α = 1 gives
   the mean, α = 0 the raw sum. NAIS tested *"α from 0 to 1 with a step size of
   0.1"* (§4.2).
4. **A train/test asymmetry is baked in** (visible by comparing the two slide
   formulas): at training the pool is |I_u| − 1 items normalized by
   |I_u\{i}|^{−α}; at test it is |I_u| items normalized by |I_u|^{−α}. The
   authors do not treat this as a problem; for us it is a stated design point
   (§3), because our host's cosine reacts differently to it than a dot product.

### 2.3 NAIS — what the extra machinery buys, measured (He et al. 2018)

Included here not as a seed (rejected in §1) but because it supplies the
**only measurement of the identity channel on our own dataset**, and the
protocol is close enough to name.

- **Protocol** (§4.1): *"We adopt the leave-one-out evaluation protocol …, which
  holds out the latest interaction of each user as the testing data … each
  testing instance is paired with 99 randomly sampled negative instances; then
  each method outputs prediction scores for the 100 instances (1 positive plus
  99 negatives), and the performance is judged by Hit Ratio (HR) and Normalized
  Discounted Cumulative Gain (NDCG) at the position 10."* Dataset (Table 1):
  MovieLens 1,000,209 interactions / 3,706 items / 6,040 users — i.e. **ml-1m**,
  in the NCF preprocessing (downloaded from the NCF repo, §4.1 fn. 3).
- **Numbers, embedding size 16 (Table 5), MovieLens HR@10 / NDCG@10 (%):**
  Pop 45.36 / 25.43 · ItemKNN 62.27 / 35.87 · MF-BPR 66.64 / 39.73 ·
  MF-eALS 67.88 / 39.83 · MLP 68.41 / 41.03 · **FISM 66.47 / 39.49** ·
  NAIS-concat 69.72 / 41.96 · NAIS-prod 69.69 / 41.94.
- **Numbers at other sizes (Table 6), MovieLens HR@10:** FISM 61.71 (k=8),
  69.29 (k=32), **70.17 (k=64)**; NAIS-prod 64.50 / 70.91 / 71.82.
- **The attention gain is small and needs a magnitude trick.** NAIS Eq. (9)
  divides by `[Σ exp f(p_i,q_j)]^β` and §3.1 states plainly: *"we find such a
  standard solution of attention does not work well in practice — it
  underperforms FISM significantly"* at β = 1; *"when β is smaller than 1, the
  value of denominator will be suppressed, as a result the attention weights
  will not be overly punished for active users."* Default β = 0.5 (§4.2).
- **Authors' own framing of item-based CF's regime** (§4.3, observation 3):
  *"on MovieLens user-based models perform better than FISM, while on Pinterest
  FISM outperforms user-based models. Since user interactions of the Pinterest
  data are more sparse, it reveals that item-based CF might be more advantageous
  for sparse datasets."*

**Honest reading of these numbers for our purposes (I state, the papers do
not).** FISM 66.47–70.17 HR@10 on ml-1m sits *at or above* MF-BPR (66.64) and
below MLP/NAIS. Our own `lightfm_hist` (content-composed history under a plain
dot) measures 0.612 and fU 0.544 on the S0 split. The protocols differ in three
named ways — held-out item (their *latest* interaction vs. our canonical split's
choice), preprocessing (NCF's vs. S0's), and negative-sampling seeds — so **no
subtraction between their table and ours is licensed**, and this candidate must
not quote one. What the comparison does license is a *directional* expectation:
on ml-1m a history made of identity rows is a competitive representation on its
own, whereas our content-only history is measurably behind our own CF hosts.
Whether that gap is the identity channel is the empirical question this
candidate exists to answer.

### 2.4 The host in one paragraph (dc05, for self-containment)

fU replaces U-ProtoMF's free user row with
**q_u = Σ_{f∈V} w̄_{u,f}·e_f + e_ID(u)**, w̄_{u,f} = n_{u,f}/|H_u| — the
*mean over train-window purchases* of the item attribute words (V = 426 on H&M's
canonical 5 fields; ml-1m uses genres + tags@0.8 bags, so per-user token mass
varies), plus one free user-ID row. q_u feeds the unchanged user-prototype
layer: **u\* = 1 + cos(q_u, P^u)** ∈ R^{K_u}, and the score against the host's
free item vector t ∈ R^{K_u} is **S(u,t) = Σ_l t_l·u\*_l = B(t) + Σ_l t_l·cos_l**
with **B(t) = 1ᵀt** — a per-item, user-independent, *live* score channel (the
U host's de-facto item bias; dc05 §3.4, verified 4b C4′). Read-outs: per-prototype
contribution t_l·u\*_l; exact per-word shares of each activation; the exact
**per-purchase regrouping** of the same sum; and word-vocabulary profiles
cos(e_f, p^u_l) naming the taste communities. Two arms exist: `feature_user_proto`
(with e_ID(u)) and `_noid` (without).

### 2.5 The transplant in one line (stated here, designed in §3)

SVD++'s Eq. (15) and fU's q_u are **the same shape**: *free entity row + pooled
sum over the interaction set*. fU pools **content words**; SVD++ pools **item
identity rows**. The candidate is to let fU's pool carry both summand kinds —
i.e. to put Koren's y_j inside dc05's mean — with FISM's `\{i}` exclusion as a
non-negotiable correctness condition, and with Koren's own §4 explainability
caveat carried openly as the price.
---

## 2b. Prior-art probe

**What is being probed.** Not the seed (SVD++/FISM identity pooling is 2008/2013
textbook material and no novelty is claimed for it) but the **adaptation**: a
pooled user representation whose summands are *both* item **content words** and
item **identity rows**, feeding a **prototype-similarity layer**, read out as
exact per-purchase and per-word shares with a declared named/unnamed coverage
split.

**Queries run (2026-09-09, reproducible):**

1. `prototype-based recommendation SVD++ item identity rows user representation interpretable prototypes 2025`
2. `"FISM" OR "SVD++" history item embeddings combined with side information attribute embeddings user representation "cold" hybrid pooling`
3. `ProtoMF prototype-based matrix factorization citations 2026 user prototypes purchase history composed user representation explainable`
4. `"item ID embeddings" plus "content features" pooled user history representation "leave-one-out" masking FISM style hybrid explainable share attribution`

**Findings.**

- **Nothing found that puts a pooled history inside a prototype layer.** The
  prototype-recsys results are the ones already registered in the reading list —
  ProtoMF itself (A1), UIPC-MF (I3, arXiv 2308.07048), the cultural-diversity
  ProtoMF work (I2, arXiv 2412.14329) — plus one 2026 item not previously seen,
  **PPA++** (*Preference Prototype-Aware Learning with LLM for Universal
  Cross-Domain Recommendation*, Data Science and Engineering 2026), whose
  "proto-decoder" matches user preferences to prototypes in a *cross-domain
  transfer* setting with an LLM in the loop. Different problem (domain transfer),
  different apparatus (LLM), and no composed-user-representation claim.
  **Correction, made at Step 6 (visible, not silently fixed):** PPA++ was *not*
  new — it is already recorded in the reading list's §I.2 ("PPA (CIKM 2024) +
  PPA++ (2026) preference prototypes for cross-domain"). Calling it a fresh find
  was my error; the substance is unchanged (different problem, LLM apparatus, no
  composed-user claim) and no reading-list addition was needed. The §I sweep's
  standing conclusion (ProtoMF citers are CF-only) survives this re-check.
- **The nearest published relative is DNCF** (*Dual-embedding based Neural
  Collaborative Filtering*, arXiv:2102.02549). Verified from the paper's own
  abstract: it learns, in addition to a primitive embedding, *"an additional
  embedding from the perspective of the interacted items (users) to augment the
  user (item) representation"*, explicitly *"inspired by the idea of SVD++"*.
  Crucially it is **ID-history only — no content/side-information channel** —
  and the combination feeds an MLP scorer. So it occupies the "free row + ID
  history" cell (SVD++ generalized to neural CF), not this candidate's
  "content words + ID history, read through prototypes" cell.
- **Content-side pooling** is B5 LightFM's registered move (items as sums of
  content-feature latents) and remains the lineage's own basis. No result mixes
  it with an SVD++ identity pool in one sum.
- **A search-result claim I decline to carry.** One snippet asserted that "both
  SVD++ and FISM can be viewed as a special case of DNCF, which uses element-wise
  sum as embedding combination function." That sentence was not verified against
  the paper (the abstract does not state it), so it is **not cited** anywhere in
  this candidate. Flagged rather than dropped, since it would be the right thing
  to check if a related-work paragraph ever needs a unifying frame.

**Verdict — distinguish (not refine).** The mechanism's two halves are each
well-published, and their *combination in one pooled user vector* is a natural
enough move that absence of a found precedent is weak evidence; a hybrid
"content + ID history" pool very plausibly exists somewhere in the CTR/sequential
literature under different vocabulary. **This candidate therefore claims no
novelty on the pooling itself.** What is distinguished, and stated as the
contribution surface:

1. **The host.** The pooled vector feeds a *prototype-similarity* layer, not a
   dot product or an MLP. That changes the mechanism's behaviour non-trivially —
   the cosine discards the pooled magnitude, which is exactly the quantity
   FISM's α and NAIS's β exist to control (§2.2/2.3). No prior work reports what
   happens to an SVD++-style identity pool under a shifted cosine.
2. **The read-out.** SVD++'s explainability is, in Koren's own §4, *given up*
   relative to Asymmetric-SVD. This candidate keeps the pool but adds an exact
   per-summand attribution with an explicit **named/unnamed coverage split** —
   i.e. it re-prices the trade Koren named instead of accepting it silently.
3. **The measurement.** The four-arm design (words / +user row / +identity rows /
   both) isolates, inside one architecture and one code path, how much of a
   composed user's power is content and how much is identity — a decomposition
   the cited papers do not run because none of them has both channels.

Honest statement of standing: this is a **known-mechanism, new-host,
new-read-out** candidate. If the thesis needs a novelty claim from it, the claim
is (1)+(2), not "we invented history pooling".
---

## 3. Adaptation design

### 3.1 The one idea, stated exactly

> **q_u = (1/|H_u|) · Σ_{j ∈ H_u} ( Σ_{f ∈ f_j} e_f  +  λ_y · y_j )  +  e_ID(u)**

where H_u is the user's train-window purchase set (deduped, exactly as today),
f_j the item's attribute-word bag, e_f the existing shared word table, e_ID(u)
the existing optional user row, and **y_j ∈ R^d a new free per-item "history
row"** — Koren's y_j (Eq. 15) and FISM's W_{i'·}, transplanted into dc05's mean.
λ_y = 1 by default (§3.6 K_y).

Everything else is untouched: **u\* = 1 + cos(q_u, P^u)**, score
**S(u,t) = Σ_l t_l·u\*_l = B(t) + Σ_l t_l·cos_l**, the shifted cosine, the
inclusion regularizers, sampled-softmax loss, negative sampling, and the free
item vector t ∈ R^{K_u}. dc05's own words on its mechanism apply verbatim; the
only change is *what may be a summand*.

**Why this is one idea and not two.** The composition is already a weighted bag
over a row table. This adds one row *kind* to that table. It does not add a
layer, a loss, a nonlinearity, or a second representation — it widens an
existing sum. The four-arm matrix below is a config sweep over two existing
booleans, not four mechanisms.

### 3.2 Modules and parameters — what actually changes

**The implementation insight that keeps this cheap.** `HistoryFeatureEmbedding`
(`feature_extraction/feature_extractors.py:384`) does not know that its rows are
"words". It holds `hist_value_ids (N, D_max)` and `hist_weights (N, D_max)` as
non-persistent buffers, embeds them and sums:
`q = (embedding_layer(ids) * w.unsqueeze(-1)).sum(dim=-2)`. **An item-identity
row is therefore just another entry in that bag**, with id pointing into a new
block of the same `nn.Embedding` and weight 1/|H_u|. No new forward path, no new
class.

Concretely:

1. **Row layout (builder change, `build_user_history_weights`).** The embedding
   table grows from `[metadata words: V] ++ [user ID rows: N]` to
   **`[metadata words: V] ++ [item history rows: M] ++ [user ID rows: N]`**.
   The class's ID-row offset is already `o_idxs + self.n_features`, so the
   change is expressed by passing `n_features = V + M` when history rows are on
   — the ID-row indexing needs **no code change at all**. History row for item j
   sits at index `V + j`.
2. **Per-user bag (builder change).** For each user u, append |H_u| slots:
   `(id = V + j, weight = λ_y/|H_u|)` for every j ∈ H_u, alongside the existing
   word slots `(id = f, weight = n_{u,f}/|H_u|)`. Padded width grows
   `D_max → D_max' ≤ D_max + max_u |H_u|`.
3. **Leave-one-out payload (builder change).** `item_token_ids (M, D_item)` /
   `item_token_w` currently hold each item's *word* bag; they gain one slot
   holding `(V + j, λ_y)`. The existing exact algebraic subtraction
   `q = (|H_u|·q̄_u − e_pos) / (|H_u| − 1)` then removes the positive's words
   **and its own identity row** in the same operation, exactly — because the
   pooling weight is uniform 1/|H_u| across all summands. *Correction to the
   seed entry (visible, per GR8):* the scratchpad entry required "mask-then-pool
   — the algebraic mean subtraction only works for the uniform mean of words".
   Verified against the code: the subtraction is exact for **any** summand
   carried at uniform weight 1/|H_u|, identity rows included; mask-then-pool
   becomes necessary only if per-slot weights stop being uniform (i.e. under an
   attention variant, which this candidate rejects). This is a Step-4b claim
   (C3), not an assertion.
4. **Config surface (`confs/`, both registries).** Two new booleans on
   `feature_user_proto`: `use_history_id_rows` (default False → today's fU
   bit-for-bit) and its weight `history_row_weight` (λ_y, default 1.0). No new
   `ft_type` — the dc06/dc07 precedent.
5. **Nothing else moves.** `FeatureExtractorFactory`, `RecSys`, the trainer, the
   loss accumulator, the sampler: unchanged. `supports_loo` already routes
   `pos_i_idxs` into the forward.

**Parameter delta** (analytic; numbers in §4): **+M·d**, one d-vector per item.

**What is tied (R1).** The identity rows sit *inside* the same representation
that drives the score, pooled by the same mean, normalized by the same ‖q_u‖,
and read by the same prototype layer — there is no parallel channel and no
reranker. The R1 exemplar (weight sharing across branches) has no object on this
single-branch host, per the requirements doc's 2026-07-12 clarification; the
merge-stage tie question this creates is captured in the scratchpad, not built.

### 3.3 Forward pass with shapes
(batch B, n_neg negatives, d = emb dim, K_u prototypes, D_max' padded bag width)

| step | tensor | shape |
|---|---|---|
| user bag lookup | `hist_value_ids[u_idxs]`, `hist_weights[u_idxs]` | (B, D_max') |
| embed + pool | `q_u = Σ_s w_s · E[ids_s]` (words **and** identity rows) | (B, d) |
| LOO (train only) | `q_u ← (|H_u|·q_u − e_pos) / (|H_u|−1)`, `e_pos` = positive's words **+ its y** | (B, d) |
| + user row | `q_u ← q_u + E[u_idxs + (V+M)]` | (B, d) |
| user proto branch | `u* = 1 + cos(q_u, P^u)` | (B, K_u) |
| item embedding | `t = T[i_idxs]` | (B, 1+n_neg, K_u) |
| score | `Σ_l u*_l · t_l` | **(B, 1+n_neg)** |

The `(batch, 1+n_neg)` sampled-scoring profile is preserved exactly — the user
branch is computed once per row and is independent of the candidate set.

### 3.4 Intrinsic explanation read-out

The score identity is dc05's, unchanged:
**S(u,t) = B(t) + Σ_l t_l·cos_l**, B(t) = 1ᵀt rendered as its own honest,
non-personalized line.

dc05's read-outs 1–4 survive **structurally intact**, because an identity row is
just another summand of the same sum. What changes is that the sum is now
**partly unnameable**, and the read-out must say so on the artifact rather than
in a footnote.

1. **Per-prototype contribution** (unchanged): s_l = t_l·u\*_l.
2. **Per-word shares** (unchanged in form, reduced in coverage): the exact
   decomposition of u\*_l − 1 now reads
   **Σ_{f} w̄_{u,f}·⟨e_f, p^u_l⟩/(‖q_u‖‖p^u_l‖) + Σ_{j∈H_u} (λ_y/|H_u|)·⟨y_j, p^u_l⟩/(‖q_u‖‖p^u_l‖) + ID share**.
   Still exact, still summing to the activation; the middle block is new and
   **unnamed**.
3. **Per-purchase regrouping** (unchanged in form, *strengthened* in meaning):
   each purchase j already carried a share Σ_{f∈f_j}(1/|H_u|)⟨e_f,p^u_l⟩/…; it
   now carries **that plus its own identity share**. The per-purchase line
   remains exact and complete — and this is the read-out that best survives the
   change, because a purchase is nameable (it is a garment the user bought) even
   when its identity row is not. This is precisely Koren's §4 Asymmetric-SVD
   benefit — *"explaining predictions by most relevant actions"* — recovered on
   a host that Koren said gives it up.
4. **Prototype profiles** cos(e_f, p^u_l) (unchanged): profiles are read over
   the **word** rows only; identity rows are excluded from naming by
   construction, since they have no label.

**New, mandatory read-out — the coverage split (I propose).** Every rendered
explanation carries an explicit three-way mass split of the personalized
activation:

> **named (words) — collaborative (identity rows) — personal (user ID row)**,
> as signed shares summing to u\*_l − 1, plus a scalar **name-coverage** =
> named / (|named| + |collaborative| + |personal|), reported per prototype and
> per user.

This is not decoration: it is the price tag on Koren's trade, and without it the
candidate would be claiming a grounding it does not have. dc05's `naming/`
route needs a token label for the identity block — proposal: render it as the
**purchase itself** in the per-purchase zoom ("*because you bought this exact
jacket* +0.08 — an item-identity effect, not an attribute effect") and as an
aggregate line in the per-word zoom.

**Concrete H&M-style rendering** (schematic, one user × one candidate item):

```
Why this item?                                       score 3.41
  baseline (non-personalized item mass B(t))               +1.90
  community "dark denim menswear"   t_l=0.42 × u*_l=1.71   +0.72
      named attributes           +0.44   (61%)
        · Department: Menswear            +0.19
        · Colour group: Dark Blue         +0.14
        · Garment group: Trousers         +0.11
      item identity              +0.21   (29%)
        · "Slim Tapered Jeans"            +0.09
        · "Denim Jacket Loose"            +0.07
        · 9 further purchases             +0.05
      you (personal residual)    +0.07   (10%)
  community "sporty basics"      t_l=0.11 × u*_l=0.55       −0.21
  … (K_u rows)                                      name-coverage 0.61
```

The same numbers regrouped by purchase (read-out 3) give the alternative zoom:
"your 12 dark-denim purchases contribute +0.31 of this 0.72 — 0.19 through their
attributes, 0.12 through the items themselves." **Word-level and purchase-level
are two exact readings of one sum and are never added together** (dc05 4b C2′
discipline, inherited).

### 3.5 Interpretability-constraint design (v1.1 mandatory block)

**Inheritance.** The host's two regularizers act on the prototype-similarity
matrix, not on the composition, so they *run* unchanged. Do their semantics
hold?

- **`sim_proto` (row/"each prototype is close to some entity")** — semantics
  hold, with a caveat worth stating: it now pulls prototypes toward composed
  user vectors that contain unnamed mass, so "a prototype resembles some real
  user" stays true while "a prototype resembles some describable taste" weakens
  in proportion to 1 − name-coverage. The regularizer cannot see the difference.
- **`sim_batch` (column/"each entity is close to some prototype")** — semantics
  hold unchanged; it is about coverage of the user cloud, which is agnostic to
  what q_u is made of.

**New degeneration modes** (not covered by the inherited pair):

- **D1 — Name-coverage collapse.** Training has no reason to prefer named mass.
  If identity rows are the stronger signal (the whole premise), gradient descent
  routes representation through them and the word shares shrink toward noise:
  the model gets better and the explanation gets emptier. *Directly measurable*
  by the §3.4 coverage scalar. This is the candidate's central risk and it is
  **structural, not incidental**.
- **D2 — Channel non-identifiability between y and e_ID(u).** Both are free and
  unnamed. For a user whose purchases are unique to them, a shift can be moved
  between their ID row and their items' identity rows with (nearly) no change in
  q_u — so the "personal vs collaborative" split in the read-out is only as
  identifiable as the overlap structure makes it. Exactly dc06's D2 lesson
  ("ID rows re-house removed components") at a new site, and a mirror of
  F-DC01-06's frozen-difference anatomy.
- **D3 — Popularity re-homing into the user branch.** dc07's stage-1 result is
  the standing warning: *the popularity channel re-forms whatever similarity the
  layer is given*. On this host popularity already has a home in B(t) = 1ᵀt.
  Identity rows give it a **second** home — a popular item's y_j appears in many
  users' q_u, so a prototype aligned with the mean identity row is a popularity
  prototype dressed as a taste community. Inherited regularizers do not see it.
- **D4 — Frequency/degree gauge in the identity block.** y_j rows of rare items
  receive gradient from few users; their scale and direction are weakly
  determined, while a shared component across all y_j (the "everyone buys
  things" direction) is strongly determined and carries no information — the
  degeneration F6/F8 describe for word tables, now on identity rows. This is
  what makes the dc06/fU′ interaction non-trivial (scratchpad, captured today).

**Constraint candidates — designed, deliberately NOT stacked (R8).**

| id | targets | mechanism | hook | expected effect | cost | Rashomon-tunable |
|----|---------|-----------|------|-----------------|------|------------------|
| **K_y** | D1 | the weight λ_y itself, as a *declared interpretability knob* rather than an accuracy knob: sweep λ_y ∈ {0, 0.25, 0.5, 1} and pick inside the accuracy-tolerance band by name-coverage | config only | trades identity mass for named mass continuously; λ_y = 0 recovers today's fU exactly | zero code | **yes — the primary one** |
| **K_cov** | D1 | soft floor on named mass: `L_cov = mean_u max(0, τ − named_u/(named_u+|coll_u|+|pers_u|))²`, τ ∈ [0.3, 0.6], weight λ_cov | `get_and_reset_loss()` on the user extractor | pushes representation back into nameable rows; direct optimization of the price tag | one extra pooled forward over the split masses per batch (cheap; the shares are already computed) | yes |
| **K_yreg** | D4, D3 | L2 on the identity block only: `L_y = (1/|H_u|)Σ_j ‖y_j‖²`, or `max_norm` on that row block | existing `max_norm` arg / loss accumulator | shrinks weakly-determined rare-item rows and caps the popularity direction's magnitude | negligible | yes |
| **K_ycenter** | D4 | subtract the (buffered, train-computed) mean identity row before pooling: `ỹ_j = y_j − ȳ` | forward, one buffer | removes the information-free shared component; the identity-block analogue of dc06 | negligible; **but interacts with dc06 — see the captured ordering entry** | yes |
| **K_dropout** | D2 | dropout on the user ID row during training (dc05's parked K3′), which also trains the fold-in path | forward | forces the split to be identified by data rather than by initialization | negligible | yes |

None is added. λ_y (K_y) is the one the run plan will *use*, because it is a
config sweep rather than a mechanism — and because it is the honest instrument
for the fidelity-vs-accuracy tension this candidate embodies.

### 3.6 Baseline gate & the isolating ablation

**The four arms** (two booleans, one code path, one dataset each):

| arm | words | user ID row | identity rows | status |
|-----|-------|-------------|---------------|--------|
| A — `feature_user_proto_noid` | ✓ | — | — | **exists** (dc05) |
| B — `feature_user_proto` | ✓ | ✓ | — | **exists** (dc05) |
| C — `feature_user_proto_hist` | ✓ | — | ✓ | **new** |
| D — `feature_user_proto_ids_hist` | ✓ | ✓ | ✓ | **new** |

**The ablation that isolates the one idea: C − A** (identity rows added, nothing
else changed, no user row to confound the attribution). **D − B** is the same
contrast in the presence of the user row and answers the *redundancy* question
(D2): if D − B ≪ C − A, identity rows and the user ID row are substitutes, and
the read-out's personal/collaborative split is weakly identified.

**Standing baselines** (per the staged-bar discipline):

- **Host bar: `user_proto`** — the like-for-like stage bar (fU vs its own host);
  the paper-headline UI bar remains deferred to the merge (R6 staging note).
- **`lightfm_hist` / `lightfm_hist_ids`** — the existing decoupled dot-product
  controls. **A dot-product twin of C is the cheapest possible first test** and
  is strongly recommended as the *first* run: `lightfm_hist` + identity rows
  isolates the channel with no cosine, no prototypes, no magnitude discarding.
  If it does not move there, the prototype arm is unlikely to (and the nl-track
  memo pre-registered exactly this gate at +0.02 HR@10).
- **Literature reference point, quoted as context only, never subtracted:**
  FISM 66.47 HR@10 (k=16) / 70.17 (k=64) on ml-1m under a 1+99 protocol
  (NAIS Tables 5/6, §2.3) — protocol-incompatible with ours by three named
  differences; it motivates the bet, it does not score it.

**Dataset expectation, stated before the runs (falsifiable):** ml-1m is where
this should pay (mean |H| ≈ 93 train rows/user, 3,125 items — long histories,
dense identity signal, and the arm where fU is measurably weakest). H&M
`hm_1_month` is where it should be **null or negative** (mean |H| = 6.49, 13,651
items — most identity rows see ~6 users, so they are noise with parameters). If
the result is the reverse, the mechanism is not doing what this document claims.

### 3.7 Deviations from the source papers ("I propose", GR8)

| # | source says | this candidate does | why |
|---|---|---|---|
| 1 | SVD++ (Eq. 15): user = free row + **|N(u)|^{−1/2}**·Σ y_j; FISM: |I_u|^{−α}, α tuned | **uniform mean (α = 1)** over purchases, for *all* summand kinds | inherits dc05's mean decision unchanged; under the shifted cosine the pooled magnitude is discarded anyway, so α acts only through the metadata/identity/ID balance — and keeping it at 1 keeps the exact per-purchase attribution and the exact LOO subtraction. **Recorded as an unstacked knob**, and flagged as one place where our host genuinely differs from both papers. |
| 2 | SVD++/FISM pool **only** identity rows | pools identity rows **together with content words** in one mean | the whole point: the two summand kinds are then commensurable, jointly normalized, and jointly attributable — which is what makes the coverage split meaningful |
| 3 | FISM/SVD++ carry per-item biases b_i | **no bias terms** (the fleet is bias-free) | fleet-wide charter property, not a choice here; consequence noted in §5 (the U host's B(t) already plays that role) |
| 4 | NAIS: per-slot attention weights | **rejected** (§1 Seed B) | R8 + read-out contract + E4a/E4b + dc07's negative |
| 5 | Koren §4: SVD++ gives up "readily explainable results" | keeps the pool **and** adds an exact per-summand attribution with a declared coverage split | the contribution surface (§2b); it does not refute Koren — it prices what he conceded |
| 6 | neither paper has a prototype layer | the pool feeds `1 + cos(q_u, P^u)` | the host; and the reason no prior result predicts our outcome |
---

## 4. Cost & feasibility

All figures **Fermi-style**, computed from the split statistics recorded in
dc05 §4 and S0 (`hm_1_month`: N = 73,418 users, M = 13,651 items, train
476,394 rows, mean |H_u| = 6.49, median 5, max 89, D_max = 112 word slots,
V = 426. `ml-1m`: N = 6,034, M = 3,125, train 562,308 rows ⇒ mean |H_u| ≈ 93.2).
Reference config d = K_u = 64.

### 4.1 Parameter delta

**+M·d — one d-vector per item, nothing else.**

| | host `user_proto` | dc05 fU (+ID) | **dc08 arm D** | dc08 as % of host |
|---|---|---|---|---|
| `hm_1_month` | N·d + M·K_u + K_u·d ≈ **5.58 M** | +V·d = +27 K (+0.5%) | **+M·d = +874 K** | **+15.7%** |
| `ml-1m` | ≈ **590 K** | +V·d | **+M·d = +200 K** | **+33.9%** |

Protocol reference point (CF-only `user_item_proto`): (N+M)·d + (K_u+K_t)·d +
2d² tied; the operative stage bar remains `user_proto` per the staged-bar
discipline.

**The uncomfortable reading, stated rather than buried:** the *noid* arm (C) was
dc05's parameter-economy headline — a **~172× shrink** of the user table
(73,418·d → 426·d). Arm C gives most of that back in item form (426·d +
13,651·d). dc08 therefore **spends dc05's parameter-efficiency argument** to buy
a collaborative channel; that is a real cost on the LightFM-generalisation
narrative (B5 §2.3) and must not be presented as free. What survives untouched:
the model is item-parameterized rather than user-parameterized, so Koren's
Asymmetric-SVD benefit #2 (new users handled without re-training) still holds —
arm C composes a never-seen user from purchases alone, with **zero**
user-specific parameters.

### 4.2 Training-time expectation

Per training row the user branch pools D_max' slots instead of D_max.

- `hm_1_month`: mean real slots 18.7 words → 18.7 + 6.49 ≈ **25.2** (+35%);
  padded width 112 → ≤ 201. This is a gather-and-sum over ≤ 201 rows of width
  64 — ~10⁴ FLOPs/row, dominated by the (1+n_neg) scoring and the K_u×d
  prototype math. **Expected wall-clock multiplier ≈ 1.0–1.05×.**
- `ml-1m`: mean real slots ≈ (distinct words per user) + 93. FLOPs still
  negligible; the padded **buffer** is the quantity to watch (§4.3).

**Convergence:** arguable in one direction only. On `hm_1_month` each of the M
new rows sees gradient from only ~6.5 distinct users on average (mean |H| =
6.49). Expect no change in convergence *speed*, but a wide spread in row
*quality* — rare-item identity rows stay near-init. That is degeneration mode
D4, and the reason K_yreg was designed (and left unstacked).

### 4.3 Buffer / memory

Non-persistent padded buffers, int64 ids + float32 weights ≈ 12 B/slot:

- `hm_1_month`: 99 MB → **≈ 177 MB** (N × (112+89) × 12 B). Fine on any fleet GPU.
- `ml-1m`: N is small (6,034), so even a padded width of ~2,700 costs
  ≈ **196 MB** — affordable, but the padding waste is extreme (mean |H| 93 vs
  max in the low thousands). **Declared mitigation (dc05's, unchanged):** the
  flat `embedding_bag`/offsets layout, Σ_u D_u ≈ 3 M entries ≈ 36 MB. Not needed
  at V1 scale; specified so the build has a fallback rather than a surprise.
- `hm_3_month` (N = 256,701) would require the flat layout — the same conclusion
  dc05 already reached; this candidate does not change it.

### 4.4 S5 pipeline-compatibility checklist

| check | verdict |
|---|---|
| Sampled `(batch, 1+n_neg)` scoring preserved? | **Yes** — the user branch is computed once per batch row, independent of the candidate set; the score tensor's shape is unchanged. |
| No full-catalog pass per step? | **Yes** — nothing iterates over M; the identity rows touched per row are exactly that user's own \|H_u\| (a gather). |
| Item/user branch still cheap indexed computation? | **Yes** — one `nn.Embedding` gather + weighted sum, the existing code path; the item branch is untouched. |
| Losses fit `get_and_reset_loss()`? | **N/A for the candidate** (it adds no loss). The designed-but-unstacked K_cov / K_yreg both fit the accumulator (§3.5). |
| Anything expensive per item → precompute/cache? | **No per-item encoder.** The only precompute is the builder emitting the enlarged bag once at model-build time, from the split's own train file — the same discipline and leakage boundary as dc05's existing buffers (train window only; verified at build, not assumed). |

**One genuine new pipeline obligation.** `item_token_ids` must gain the item's
own identity slot so the LOO subtraction removes it. If that slot is omitted,
the model trains with **the positive's own identity row inside its own
predictor** — a leak that would present as a large accuracy win. This is the
highest-consequence implementation detail in the candidate: pre-registered as a
build-time assertion (the `last_forward_loo` introspection already exists) and
as Step-4b claim C3.

### 4.5 Budget gate

Dev-profile runs on this host are fleet-routine (dc05's arms A/B ran at the dev
tier). Arms C/D add ~1.0–1.05× wall clock and +0.2–0.9 M parameters.

- **Cheapest possible first test — the dot-product twin:** `lightfm_hist` +
  identity rows on ml-1m. No prototypes, no cosine, hours not days; login-node
  class under the current smoke policy. Pre-registered gate from the nl-track
  memo: +0.02 HR@10 over `lightfm_hist`.
- **Then, only if that moves:** arms C and D on ml-1m at the dev tier (30-trial
  hyperopt each, fleet convention), each its own `/leo5-submit` — the standing
  one-submit-per-run gate.
- **hm rows are the falsification arm, not the hope** (§3.6 predicts null to
  negative there); cheap, and worth running for exactly that reason.

**Verdict: no budget-gate breach.** Nothing approaches the >1-day
soft-infeasible line for a dev iteration, let alone the 10-day hard line. The
scarce resource is cluster queue time, not compute per run — which is why the
dot-twin-first ordering exists: spend as little queue as possible before the
idea has earned a prototype-arm slot.
---

## 4b. Math sanity check

### 4b.1 Claims spec (written before execution; the subagent received *only* this)

The full spec handed to the black-box subagent is reproduced here verbatim in
substance: mechanism definition (row layout V words ++ M identity rows ++ N user
rows in one table; padded-bag composition; algebraic leave-one-out; the
`u* = 1 + cos(q_u, P)` / `S = Σ_l t_l u*_l` head), then eleven claims:

- **C1 — bag form = mathematical form.** The padded-bag pooling equals
  q_u = (1/|H_u|)Σ_{j∈H_u}(Σ_{f∈f_j}e_f + λ_y y_j) + [use_id]·e_ID(u).
- **C2 — exact reduction.** λ_y = 0 recovers today's fU (identity slots inert).
- **C3 — the algebraic leave-one-out is exact for identity rows too.**
  (|H|·q − e_pos)/(|H|−1) with e_pos = Σ_{f∈f_i}e_f + λ_y y_i equals the
  from-scratch mean over H_u\{i}; |H_u| = 1 gives exactly zero metadata mass.
  **This is the claim that decides §3.2's correction to the seed entry.**
- **C4 — exact additive decomposition** of cos_l into per-slot shares (words and
  identity rows alike) under the shared 1/(‖q_u‖‖P_l‖) normalization.
- **C5 — two exact regroupings of one sum** (by row kind vs. by purchased item).
- **C6 — score decomposition** S = B(t) + Σ_l t_l cos_l with B(t) user-independent.
- **C7 — scale invariance**, and the *consequence*: the pooling exponent α is
  inert only when `use_id = False`; with a user row added after pooling it is
  not, because the user row is not rescaled.
- **C8 — gradient reaches y_j** for every purchased item, and is exactly zero for
  a leave-one-out-removed positive.
- **C9 — identity/user-row non-identifiability** (degeneration mode D2): does an
  exact compensating family exist for a user whose purchases are unique to them?
  The subagent was asked to derive the exact form itself, or to characterise what
  actually holds.
- **C10 — shared-component sensitivity** (D4): adding c to every y_j shifts every
  q_u by exactly λ_y·c, |H_u|-independently — and is it rank-inert?
- **C11 — degenerate empty history**: literal numerical behaviour of the cosine
  at q_u = 0.

Spec file: `Master/temp/dc_checks/dc08/claims_spec.md` (persisted copy);
scripts: `Master/temp/dc_checks/dc08/check_claims.py`, `check_edges.py`.

### 4b.2 Viability self-assessment (mandatory, written before execution)

| claim | meaningful at toy scale? | note |
|---|---|---|
| C1, C2, C4, C5, C6, C11 | **Yes** — pure algebra/identity claims | these are exactly what toy checks are for |
| C3 | **Yes, and it is the highest-value check** — it decides a correctness condition that would otherwise show up as a spurious accuracy win | |
| C7 | **Yes** for the invariance; the α-with-user-row consequence is also exact algebra | |
| C8 | **Yes** — one backward pass, gradient presence/absence is structural, not dynamical | |
| C9, C10 | **Yes as existence/structure questions** (construct the family, measure the residual). What is **not** testable at toy scale: whether training *actually* exploits the freedom — that is a dynamics question for the fleet instruments | |
| **Deliberately NOT claimed** | — | The candidate's central bets are **not** toy-checkable and no toy test is proposed for them: (i) that identity rows raise accuracy on ml-1m (training dynamics + real data), (ii) that name-coverage collapses (D1 — a dynamics outcome), (iii) that popularity re-homes into the identity block (D3 — dc07 showed this is decided by hyperopt, not by algebra). These are **empirical bets, flagged as such**, and §5 rates them as open risks rather than as design properties. Skipping is legitimate; silent skipping is not — this row is the disclosure. |

### 4b.3 Subagent report (verbatim)

Scripts: `Master/temp/dc_checks/dc08/check_claims.py` (main battery) and
`Master/temp/dc_checks/dc08/check_edges.py` (edge probes). The subagent received
only the spec above; it re-implemented the mechanism from the spec text and
imported no project code.

> ## dc08 claims check — independent black-box verification
>
> Scripts: `check_claims.py` (main battery) and `check_edges.py` (edge probes). Toy scale V=11, M=9, N=7, d=5, K=4; 20–40 random worlds per claim; numpy/torch float64 unless noted. No project code imported — the mechanism was re-implemented from the spec text alone.
>
> **C1 — Bag form = math form: CONFIRMED.** max abs error 1.78e-15 over 30 worlds × λ_y ∈ {0, 0.5, 1, 2} × use_id ∈ {F,T} × all users, including a forced |H_u| = 1 user and a forced repeated-word user (items sharing words). The count-aggregated padded bag reproduces the definition to float64 round-off.
>
> **C2 — Reduction at λ_y = 0: CONFIRMED, and bit-identical (0.000e+00), not merely close.** Zero-weight identity slots contribute exact `+0.0` and IEEE addition of zero is a no-op, so removing them from the bag changes nothing. Caveat worth stating: this is bit-identity only because the identity rows are finite; a NaN/Inf row would leak through a 0.0 weight.
>
> **C3 — Algebraic leave-one-out is exact: CONFIRMED.** max abs error 2.67e-15 vs recomputation from scratch over `H_u \ {i}`, at both λ_y = 0 and λ_y = 1. Identity-row removal verified independently by a perturbation probe (shift `y_i` by +100 and confirm the LOO output does not move: max drift < 1e-9), so the positive's identity row is removed as well as its words. The |H_u| = 1 case returns exactly zero metadata mass.
>
> **C4 — Additive decomposition of the activation: CONFIRMED.** max abs error 5.55e-16 between `cos_l` and the sum of per-slot shares plus the user-row share. Expected — the decomposition is just linearity of the inner product with a shared scalar denominator, so it is exact by construction, not an empirical finding.
>
> **C5 — Two exact regroupings: CONFIRMED.** grouping (a) by row type err 6.66e-16; grouping (b) by purchased item plus user-row share err 6.66e-16; per-item totals vs `cos_l − user-row share` err 6.66e-16. Note the regrouping by item requires un-aggregating the word counts back to their source items — the bag itself has already lost that attribution, so (b) is recoverable only with `f_j` in hand, not from `(ids, weights)`.
>
> **C6 — Score decomposition: CONFIRMED.** max abs error 1.78e-15; `B(t_i) = Σ_l t_{i,l}` is bit-stable when `q_u` is replaced. Trivially true given `u*_l = 1 + cos_l`. Consequence the spec does not draw: B is a pure per-item popularity/bias offset that survives any user-side change, so it is a free additive popularity channel in the head.
>
> **C7 — Scale invariance: CONFIRMED, with one caveat the spec omits.** Positive rescaling of `q_u`: max |cos(q) − cos(sq)| = 2.22e-16 over s ∈ {1e-3, 0.5, 2, 1e3}. Pooling exponent `1/|H_u|^α`: with use_id=False, max Δcos = 3.33e-16 (inert, as claimed); with use_id=True, max Δcos = 9.64e-01 (large — the identity row is not rescaled, so direction genuinely moves). Caveat: α is also inert with use_id=True whenever |H_u| = 1 (measured Δcos exactly 0), since 1^α = 1. So "α matters when use_id=True" holds only for |H_u| ≥ 2 — the single-purchase users are unaffected regardless.
>
> **C8 — Gradient flow to identity rows: CONFIRMED for the positive statement, FALSIFIED in the strict "exactly zero" wording.** Without LOO, exactly the rows `y_j`, j ∈ H_u receive gradient (|grad| ≈ 2.06e-01 for both members of H_u = [3,5]) and every non-purchased row is exactly 0.0. With LOO applied to positive j=3, its gradient drops to 0.000e+00 while the remaining member keeps 4.71e-01. But the zero is a floating-point cancellation of two equal-magnitude paths, not a structural zero: sweeping |H_u| = 2..8, the residual is 0 for even n and up to 1.4e-17 (float64) / **1.5e-08 (float32)** for odd n, where `n·(λ_y/n)` is not exactly λ_y. In fp32 training the removed positive receives a ~1e-8 gradient, not exactly zero. Harmless in practice, but the spec should say "algebraically zero, numerically at round-off" rather than "exactly zero".
>
> **C9 — Identity/user-row non-identifiability: CONFIRMED as a mechanism, but the spec states it imprecisely on two counts.** The exact compensating family is: add δ to `e_ID(u)` and subtract **δ/λ_y** from `y_j` for every j ∈ H_u. (The spec's `(|H_u|/λ_y)·δ/|H_u|` algebraically equals δ/λ_y, so it is correct but needlessly written; the |H_u| factors cancel.) Residual over ALL users' compositions: 4.44e-16 without LOO and 8.88e-16 with LOO applied — it survives the leave-one-out path too, which the spec does not claim. Two corrections: (i) it is a **d-dimensional** family (δ is a free vector in R^d, here d=5), not "one-parameter"; (ii) it requires λ_y > 0 and use_id = True — at λ_y = 0 no finite compensation exists and moving the user row alone drifts `q_u` by 2.37e+00. Disjointness is load-bearing: a control where one of user 0's items is shared with user 1 moves user 1's composition by 3.20e-01.
>
> **C10 — Shared-component sensitivity: CONFIRMED, and it is NOT rank-inert.** Adding a constant c to every identity row shifts `q_u` by exactly λ_y·c for every user, max error 8.33e-16, independent of |H_u| as claimed. But the shift changes item ranking for **5 of 7** users in the toy battery — it moves `q_u`'s direction, hence every `cos_l`, hence the `Σ_l t_{i,l} cos_l` term, which is item-dependent. So the identity block has an unpenalised shared direction that every user absorbs identically yet that still reorders recommendations; it is a genuine free parameter, not a gauge freedom.
>
> **C11 — Empty history: CONFIRMED literally, but the behaviour is a knife-edge.** With use_id=True, `q_u = e_ID(u)` exactly (err 0.0); with use_id=False, `q_u = 0` exactly, `torch.nn.functional.cosine_similarity` returns exactly `[0,0,0,0]` in both float64 and float32, so `u*_l = 1` exactly (max |u*−1| = 0.0) and `S(u,i) = Σ_l t_{i,l}` to 4.44e-16. Blunt caveat: this holds only at *exactly* zero. Probing tiny non-zero q in fp32, cosine returns the full-magnitude true direction down to ‖q‖ ≈ 1e-8 (e.g. q = 1e-8·1 gives cos ≈ [−0.45, 0.41, 0.60, −0.23]), and only below torch's eps=1e-8 clamp does it decay toward 0. So a user with a numerically-tiny-but-nonzero composition gets an arbitrary direction at full strength, not the graceful `u*=1` fallback. Also worth noting the fallback itself is degenerate: every empty-history user with use_id=False gets the identical ranking, ordered by `B(t_i)` alone — i.e. pure item popularity/bias.
>
> ### Summary
> Confirmed: C1, C2, C3, C4, C5, C6, C7, C10, C11. Confirmed-with-correction: C9 (family is d-dimensional not one-parameter; requires λ_y > 0 and use_id; spec's coefficient is a non-reduced form of δ/λ_y). Falsified in the strict wording: C8 ("exactly zero" gradient on the LOO-removed positive is only round-off zero — 1.5e-8 in fp32 at odd |H_u|). Nothing was untestable at toy scale. The two substantive findings beyond the spec's own framing are C10's non-rank-inertness (a real free direction in the identity block) and C11's knife-edge fp behaviour around the zero composition.

### 4b.4 What this changes in the candidate (no re-spec, no second round)

Carried forward as recorded facts, not as edits that quietly repair earlier
sections:

1. **C3 confirmed ⇒ §3.2's correction to the seed entry stands.** The algebraic
   mean-subtraction *does* remove the positive's identity row exactly; the seed
   entry's "mask-then-pool, algebraic only works for words" requirement is
   superseded for uniform pooling. Recorded with its perturbation probe.
2. **C8's strict falsification is a build-time note, not a design defect.** The
   LOO-removed positive receives ~1.5e-8 gradient in fp32 at odd |H_u|. This is
   round-off, not leakage of the positive's signal, but the build's assertion
   must be written as "algebraically zero / ≤ round-off", not "== 0", or the
   dc_checks suite will fail spuriously on the real fp32 path. **F-DC08-01.**
3. **C9's two corrections are absorbed into degeneration mode D2, which is
   thereby made worse than §3.5 stated.** The compensating family is
   **d-dimensional**, not one-parameter — i.e. the personal/collaborative split
   in the read-out has a d-dimensional exact ambiguity for users whose purchases
   are theirs alone, and it survives leave-one-out. It vanishes only at λ_y = 0
   or without the user row. **Direct consequence for the run plan: arm C (noid +
   identity rows) is the arm whose read-out split is identified; arm D's is
   not.** That elevates C from "the clean ablation" to "the arm whose
   explanation claim is defensible", and it is a stronger reason to prefer C
   than anything §3.6 argued. **F-DC08-02.**
4. **C10 is a genuine new finding and it upgrades D4 from "sloppy geometry" to a
   real free direction.** A constant added to every identity row shifts every
   q_u by exactly λ_y·c and **reorders recommendations** (5 of 7 toy users) — so
   the identity block's shared component is *not* a gauge freedom that washes
   out, it is an unpenalised parameter direction with ranking consequences. This
   makes K_ycenter (§3.5) substantially more attractive than it looked when
   designed, and it sharpens the captured fU′-ordering entry: centring the
   identity block is not cosmetic. **F-DC08-03.**
5. **C11's knife-edge is inherited, not introduced** (it is the host's cosine
   behaviour at ‖q‖→0, already on dc05's record as 4b C3), but the subagent's
   sharper reading is worth carrying: the graceful `u* = 1` fallback holds only
   at *exactly* zero, and a tiny-but-nonzero composition yields an arbitrary
   direction at full strength. Relevant to arm C's thin/empty-history users.
   **F-DC08-04** (inherited-severity note).
6. **C6's aside is a restatement of the host's known B(t) channel** (dc05 §3.4);
   noted as independent confirmation, not as news.
7. **C5's aside is an implementation constraint worth keeping:** the per-item
   regrouping needs `f_j` at render time, because the pooled bag has already
   aggregated word counts across items. The renderer must load the item→words
   map — which it already does for dc05's read-out 3.
---

## 5. Requirements evaluation

Requirements doc re-read this cycle (Step 0 and again here), including the R1
clarification note (2026-07-12) and the R6 staging note (2026-08-20).

### 5.1 Rating table with prose

| # | rating | assessment |
|---|--------|------------|
| **R1 — tied integration** | **strong** | The identity rows are summands of the *same* pooled vector that the prototype layer reads and the score consumes — same mean, same ‖q_u‖ normalization, same gradient path (4b C8: gradient reaches y_j for exactly the purchased items). There is no parallel model, no reranker, no post-hoc fusion. The R1 exemplar (cross-branch weight sharing) has no object on this single-branch host per the 2026-07-12 clarification, so the operative core is what is assessed, and it is met cleanly. **Where it strains:** the candidate *creates* the merge-stage tie question rather than answering it (should y_j be fI's item word-vector?), which is captured to the scratchpad, not built. |
| **R2 — intrinsic explanation** | **partial, and knowingly so** | Every quantity in the read-out is computed in the forward pass and the decomposition is exact (4b C4/C5/C6, errors ≤ 6.7e-16). But a summand with no label is not an *explanation* — it is an honest accounting line. So the intrinsic property (fidelity) is preserved at 100% while the *explanatory* property degrades in proportion to 1 − name-coverage. This is precisely the trade Koren stated in his own §4 ("SVD++ does not offer … readily explainable results"), reproduced with a price tag attached instead of accepted silently. **Open risk D1** (coverage collapse) is not hypothetical: nothing in the objective prefers named mass. |
| **R3 — stays in the prototype paradigm** | **strong** | Prototypes, shifted cosine, per-prototype contributions t_l·u\*_l: all untouched. A recommendation is still explained by which taste communities the user activates. What changes is what *feeds* the activation, not what the explanation is shaped like. dc07's cautionary case does not apply — this candidate does not touch the similarity function. |
| **R4 — feature-grounded prototypes** | **partial (unchanged in kind, diluted in degree)** | Prototype *profiles* are still read as cos(e_f, p^u_l) over the word vocabulary — the grounding mechanism is identical to dc05's and is not weakened by the presence of identity rows (profiles are read from parameters, not from activations). What is diluted is the *item-side* grounding of a given activation: part of it now comes from rows with no attribute meaning. The coverage scalar is the honest measure of the dilution, and K_y/K_cov are the designed responses. |
| **R5 — sparsity robustness** | **weak-to-partial, and the honest answer is uncomfortable** | Three separate cases, and they do not point the same way. *(a) Cold **items** (the H&M cold variant):* **no change either way.** A cold item is scored through its free host vector t ∈ R^{K_u}; it is in nobody's train history, so its y_j never enters any q_u. **This corrects my own Step-1b guess that identity rows are a cold-item liability — they are neither liability nor help there.** *(b) Cold / new **users**:* arm C keeps the property that matters — zero user-specific parameters, so a never-seen user composes from purchases alone (Koren's Asymmetric-SVD benefit #2 preserved, §4.1). But their identity rows are only as good as the items are popular. *(c) **Thin** users, the measurable case:* a user with 1–3 purchases now leans on 1–3 identity rows that were themselves trained from ~6 users each on H&M — i.e. the sparsity regime is exactly where the added channel is *weakest*, since y_j's quality scales with item popularity. That is the reverse of what R5 asks for, and it is why §3.6 predicts a null-to-negative H&M result. **The candidate's R5 story is essentially "no regression, no gain"; anyone quoting it as a cold-start contribution would be overclaiming.** |
| **R6 — no dense-regime regression** | **partial — this is the candidate's whole bet, and it is unproven** | The bet: ml-1m gains (mean \|H\| ≈ 93, dense identity signal, the arm where fU measurably trails), H&M is flat-to-negative (mean \|H\| = 6.49). External evidence is directional only — FISM 66.47–70.17 HR@10 on ml-1m under a *different* protocol (§2.3), never subtractable against our 0.612/0.544. Per the R6 staging note the paper-headline UI bar is **deferred to the merge and unassessed here**; the operative bar is fU-vs-`user_proto` like-for-like. Costs on the ledger: +M·d parameters (+15.7% hm, +33.9% ml-1m) and the forfeit of dc05's 172× user-table-shrink argument (§4.1). |
| **R7 — user-perceivable vocabulary** | **partial, with one genuine positive** | The identity block is the *least* user-perceivable thing possible — an opaque per-item vector. But the **per-purchase read-out is arguably more perceivable than the word read-out**: "because you bought this jacket" names an object the user actually chose, versus "because Department = Menswear". So the candidate moves *down* on vocabulary and *up* on referent concreteness. The H&M image salvage (captured scratchpad entry) fits here naturally: the per-purchase line can carry the garment photo, which is literally what the user saw. R7's balancing clause ("the features used should be among the best they can be for recommendation quality, *and also* attributable to an understandable vocabulary") is satisfied on the first half and only half-satisfied on the second. |
| **R8 — parsimony** | **strong** | One idea: a second row kind in an existing bag. No new layer, loss, nonlinearity, or representation. Implementation is a builder change plus two config booleans (§3.2); the four arms are a config matrix over two existing switches. The five constraint knobs are **designed and explicitly unstacked**. The one place R8 is under pressure is λ_y: it is a knob, and a knob that is swept is close to a second degree of freedom — mitigated by declaring it an *interpretability* knob (Rashomon-tunable) rather than an accuracy one. |
| **S1 — decoupled variant** | **strong (already built)** | `lightfm_hist` / `lightfm_hist_ids` are the existing decoupled dot-product controls, and the dot-twin of arm C is the recommended *first* run (§4.5) — so this candidate's cheapest test is literally its S1 variant. That is an unusually clean instance of S1 doing real work rather than being a courtesy ablation. |
| **S2 — user-side features** | **n-a / already resolved upstream** | dc05 settled the user-side question by choosing item vocabulary over demographics; this candidate adds a third summand kind to the same user representation but introduces no user attributes. Demographics remain "candidate prime" material. |
| **S3 — images** | **n-a for the mechanism; positive for presentation** | No image channel. But see R7: the per-purchase zoom is the natural home for garment thumbnails (the dc03-abandonment salvage), which would make the *unnamed* channel visually legible without any mechanism claim. Worth flagging to the renderer work, declared as presentation. |
| **S4 — LLM naming** | **partial / newly relevant** | Word profiles are nameable exactly as in dc05. The identity block is not — but an LLM naming layer could plausibly label a *cluster* of identity rows by the catalog rows of the items in it ("this identity direction = fast-fashion basics"), which would be a post-hoc reading of an unnamed channel, honestly labeled. Interesting, out of scope, not pursued. |
| **S5 — pipeline compatibility** | **strong** | Every checklist row passes (§4.4): sampled (B, 1+n_neg) scoring preserved, no full-catalog pass, indexed gather only, no per-item encoder, no new loss. Costs: +35% pooled slots on H&M (negligible FLOPs), buffer 99 → 177 MB, with the flat `embedding_bag` layout as the declared fallback at larger N. |

### 5.2 §4 design tensions

- **Where do features enter?** This candidate enters at the same place dc05
  does — the composed entity representation — but with a summand that is *not a
  feature at all*. That is worth naming as a **fourth entry kind** beside the
  scratchpad's add / interact / regularize trio: **identity-as-ingredient**. It
  is the one entry kind that adds representational power without adding
  vocabulary, which is exactly why it is both promising and interpretability-
  negative.
- **Fidelity vs. accuracy.** This is the candidate's defining axis, and unlike
  most candidates it can be *measured continuously*: λ_y sweeps the trade, and
  name-coverage measures the price at every setting. That makes dc08 the
  lineage's most direct empirical instrument for the tension the requirements
  doc names — arguably its strongest thesis contribution independent of whether
  the accuracy bet pays.
- **Attribution granularity.** Unchanged in machinery (exact per-slot shares),
  changed in composition: attribution is now over *three* kinds (word, item
  identity, user residual) instead of two. 4b C5 confirms the two regroupings
  stay exact; 4b C9 shows the identity/user split is **d-dimensionally
  non-identified** for users with unique baskets — so the three-way split is
  reportable in arm C and only weakly meaningful in arm D.
- **R7 perceptual alignment.** See the R7 row: down on vocabulary, up on
  referent concreteness. The tension is not resolved, it is relocated.
- **Identifiability of prototypes.** Two new pressures, both from 4b: the
  d-dimensional identity/user family (C9) and the **non-rank-inert shared
  direction in the identity block** (C10 — a constant added to every y_j
  reorders recommendations for most users). The second is the more surprising
  one and was not anticipated in §3.5's D4 phrasing, which called it a
  "gauge"; the check shows it is not a gauge but a free, unpenalised, ranking-
  relevant direction.

### 5.3 Part-prototype pitfalls checklist (reading-list S5)

- *Prototype quality:* unchanged mechanism, **diluted evidence** — a prototype's
  activation is now partly driven by unnameable mass, so "is this a good
  prototype?" needs the coverage scalar alongside the usual quality read.
- *Prototype quantity:* K_u untouched; no new pressure either way.
- *Prototype collapse / duplication:* **new pressure, from D3.** A popular
  item's y_j sits in many users' compositions, so a prototype aligned with the
  mean identity direction is a popularity prototype wearing a taste-community
  label. dc07's stage-1 result (the channel re-forms whatever similarity the
  layer is given) is the standing warning, and C10 shows the shared direction is
  ranking-relevant rather than inert. `sim_batch` does not see this.
- *Semantic gap:* **widened by construction** — the identity block has no
  semantics at all. This is the candidate's honest defect and the reason for the
  mandatory coverage split.
- *Perceptual mismatch:* n-a (no visual prototypes); see S3 for the optional
  presentation layer.
- *Attribution leakage / unfaithful saliency:* shares remain exact numerator
  decompositions (4b C4/C5). One build-level leakage risk of a different kind —
  the positive's own identity row inside its own predictor — is closed by the
  LOO path (4b C3 confirmed) with the fp32 caveat F-DC08-01.
- *Cold/degenerate activations:* the empty-history fallback is inherited and
  degenerate (all such users ranked by B(t) alone), with 4b C11's knife-edge
  sharpening: the graceful u\* = 1 holds only at exactly zero.

### 5.4 Interpretable-AI placement

Skipped — the principle framework (ideas §4) remains undecided.

### 5.5 Degeneration-protection gate (completeness, not veto)

| mode (§3.5) | status |
|---|---|
| **D1 — name-coverage collapse** | **unprotected — open risk, and it is the central one.** Not visible to either inherited regularizer. Measured (not prevented) by the mandatory coverage read-out; K_y (λ_y sweep, Rashomon-tunable) and K_cov (coverage floor loss) designed and unstacked. |
| **D2 — identity / user-row non-identifiability** | **unprotected — open risk, worse than §3.5 assumed.** 4b C9: the compensating family is **d-dimensional** and survives leave-one-out; it vanishes only at λ_y = 0 or without the user row. Consequence recorded as **F-DC08-02**: arm C's three-way read-out split is identified, arm D's is not. K_dropout designed, unstacked. |
| **D3 — popularity re-homing into the identity block** | **unprotected — open risk.** Inherited regularizers do not see it; B(t) already gives popularity one home and this adds a second. Instrument-first plan: identity-share vs. item-popularity correlation, and membership/activation-vs-popularity Spearman as in the dc07 stage-1 instruments. K_yreg designed, unstacked. |
| **D4 — shared component / frequency gauge in the identity block** | **unprotected — open risk, and upgraded in severity by 4b C10** (the shared direction is *not* rank-inert — it reordered 5 of 7 toy users). K_ycenter designed, unstacked, and now materially more attractive than when written; interacts with the fU′/dc06 ordering (captured scratchpad entry, today). |
| (inherited) prototype duplication | protected by inherited regularizer — `sim_batch` route, unchanged and out of this candidate's claim scope. |
| (inherited) dead prototypes | protected by inherited regularizer — `sim_proto` route; semantics hold with the R4-dilution caveat noted above. |

**Four of six modes unprotected.** That is not a veto (spectrum rule), but it is
the honest shape of this candidate: it buys a collaborative channel and pays in
interpretability surface, with every payment identified, measurable, and
answerable by a designed knob that is deliberately not stacked in.

### 5.6 Step-4b findings folded in as known defects

- **F-DC08-01** (from C8, strict falsification): the LOO-removed positive's
  gradient is round-off zero, not structural zero — ~1.5e-8 in fp32 at odd
  |H_u|. Build assertions must be written as tolerance checks.
- **F-DC08-02** (from C9): D2's ambiguity is d-dimensional and LOO-surviving;
  arm D's personal/collaborative read-out split is not identified.
- **F-DC08-03** (from C10): the identity block's shared direction is a free,
  ranking-relevant parameter, not a gauge.
- **F-DC08-04** (from C11, inherited severity): the u\* = 1 fallback is exact
  only at ‖q_u‖ = 0; tiny-but-nonzero compositions get an arbitrary direction at
  full strength.
- **Not falsified, and worth stating plainly:** the mechanism's algebra is
  clean — bag-form equivalence, λ_y = 0 bit-identical reduction, exact LOO
  including identity rows, exact activation decomposition, exact double
  regrouping, exact score split (C1–C7, errors ≤ 2.7e-15). The candidate's risk
  is **not** mathematical; it is empirical (does it help?) and epistemic (does
  the explanation survive?).
## 5b. Devil's advocate

### 5b.1 Critique (verbatim; the subagent received only the candidate file)

> # Adversarial critique — dc08 (external pass, 2026-09-09)
>
> *Written against the file alone. Deliberately one-sided: this is the case for rejecting the candidate, not a balanced review.*
>
> **1. The mandatory new read-out — the candidate's own contribution claim #2 — is not invariant to the model's own parameterization, and §4b proves it.**
>
> §3.4 makes name-coverage `named / (|named| + |collaborative| + |personal|)` a *mandatory* read-out and §2b(2) makes it the contribution surface: "it re-prices the trade Koren named." §4b.3 C9 then confirms an **exact, d-dimensional, LOO-surviving** compensating family: add δ to `e_ID(u)`, subtract `δ/λ_y` from every `y_j`, j ∈ H_u, and `q_u` is unchanged to 4.4e-16. Two models that are *functionally identical* — same scores, same ranking, same activations — report different "collaborative" and "personal" masses, hence different name-coverage. The price tag moves while the goods do not. §4b.4(3) notices this only as a fact about arm D's read-out; it does not notice that it invalidates the metric that §3.4 declared non-optional. Worse, C10 shows the *named* fraction is also gauge-dependent in arm C: adding a constant c to every `y_j` shifts every `q_u` by λ_y·c, which changes every cos_l and therefore every share, named ones included. A "price tag" that a reparameterization can re-print is not a measurement; it is a report on which `nn.Embedding` block a number happened to be stored in.
>
> **2. Even setting invariance aside, "named vs collaborative" labels parameter blocks, not semantics.**
>
> Nothing in the objective prevents `y_j` from learning `Σ_{f∈f_j} e_f` — i.e. an identity row that is *exactly* an item's content, counted as unnamed — nor prevents `e_f` from drifting into item-identifying directions. On ml-1m in particular (§2.4: genres + tags@0.8 bags), the word bag is close to an item fingerprint for 3,125 movies; "named" mass there is already covert identity mass. So the C−A contrast in §3.6 does not isolate "content vs identity"; it isolates **capacity**. The mechanism this candidate claims to measure is not separable from the confound it did not control.
>
> **3. The pre-registered dataset expectation (§3.6) is contradicted by the literature the document itself quotes, in the same document.**
>
> §2.3 quotes NAIS §4.3 verbatim: *"on MovieLens user-based models perform better than FISM, while on Pinterest FISM outperforms user-based models. Since user interactions of the Pinterest data are more sparse, it reveals that item-based CF might be more advantageous for sparse datasets."* §3.6 then pre-registers the exact opposite: gains on **dense** ml-1m (|H|≈93), null-to-negative on **sparse** H&M (|H|=6.49) — and says "If the result is the reverse, the mechanism is not doing what this document claims." The one published statement about *regime* that the document cites predicts the reverse. Either the quote was read and its implication suppressed, or the prediction was written from the parameter-count intuition alone. Either way, the "falsifiable" prediction is pointed the wrong way relative to its own evidence base.
>
> **4. The R6 motivation performs the subtraction it forbids, and the number it leans on is a wash on the very table it cites.**
>
> §0 juxtaposes FISM 66.5–70.2 HR@10 against "our `lightfm_hist` 0.612 / fU 0.544" and concludes "the missing channel — not composition cleverness — is where fU's headroom lives." §2.3 and §5.1 then declare that no subtraction is licensed. You cannot found the candidate on a comparison and then disclaim it; the disclaimer protects the conclusion from scrutiny while the conclusion keeps doing the work. And on the cited table (§2.3, k=16) FISM is **66.47 vs MF-BPR 66.64** — the identity-only history model *loses to plain MF with free user factors* on ml-1m. The document prints both numbers and never states this. The external precedent for "identity is where the headroom is" says, on ml-1m specifically, that identity pooling buys nothing over a free user row.
>
> **5. FISM's specifics are load-bearing but were never read.**
>
> §2's source table marks FISM ⚠️ **secondary source only** — W.K. Pan's lecture slides plus NAIS's reproduction — flagged as "a fidelity gap to close." Yet FISM supplies (a) the α semantics that §3.7 deviation #1 turns on, and (b) the HR numbers in §0 and §4.5 that motivate the whole cycle — and those numbers are *He et al.'s re-implementation*, not Kabbur et al.'s. Meanwhile the `+0.02 HR@10` gate in §4.5 is inherited verbatim from the abandoned nl-track memo — the same memo whose SVD++ formula was flagged as reproduced from memory. §1d closes the memo's ⚠️ for the formula and silently keeps the gate. The quantitative spine of the motivation is: a third-party reimplementation of an unread paper, under a protocol the document declares incompatible, with a threshold carried over unexamined from an abandoned track.
>
> **6. The host is presented as a contribution when the document's own analysis shows it removes the mechanism's operating knob.**
>
> §2b(1): "the cosine discards the pooled magnitude, which is exactly the quantity FISM's α and NAIS's β exist to control. No prior work reports what happens to an SVD++-style identity pool under a shifted cosine." Read that again as an engineer rather than as a novelty claim: SVD++ uses `|N(u)|^{−1/2}`, FISM tunes α ∈ [0,1], NAIS needs β<1 and says plainly that β=1 *"underperforms FISM significantly."* All three are magnitude/confidence controls on the pooled history. The proposed host throws that quantity away by construction. "No prior work reports what happens" is not an open question worth a cycle; it is a warning that the transplant discards the part the source authors found necessary. A host that disables the mechanism is a confound, not a contribution surface.
>
> **7. α is frozen at 1 for read-out convenience, and §4b measured that this choice is not innocuous.**
>
> §3.7 deviation #1 justifies α=1 on the grounds that "under the shifted cosine the pooled magnitude is discarded anyway." §4b.3 C7 then measures, for `use_id=True`, **Δcos = 9.64e-01** under a change of pooling exponent — i.e. α materially changes the representation's *direction* in arm D, and the document's own justification is falsified for half the design matrix. In arm C the document concedes α "acts only through the metadata/identity/ID balance" — which is precisely the balance the entire candidate is about. So the single most consequential hyperparameter of the transplanted mechanism is pinned to a value chosen because it keeps the algebraic LOO subtraction exact (§3.2(3)) and the per-purchase attribution clean. Interpretability tooling is dictating the mechanism's strength, and then the accuracy bet will be scored as though the mechanism had been fairly represented. A null on ml-1m will be uninterpretable: mechanism, or α=1?
>
> **8. There is no regime in which the identity block can work, and the two failure modes are the document's own D3/D4.**
>
> Take the two datasets seriously.
>
> *ml-1m, the arm where the gain is predicted:* `q_u` pools ~93 identity rows. The mean of 93 rows concentrates on the shared component `ȳ` at rate ~1/√93; the user-specific deviation is what carries taste, and it is the part that shrinks. The cosine then discards magnitude, so what survives is a direction dominated by the block's common component — which §3.5 D4 identifies as information-free and §4b C10 proves is **not rank-inert** (it reordered 5 of 7 toy users). That is D3 verbatim: "a prototype aligned with the mean identity row is a popularity prototype dressed as a taste community." The regime where the bet is placed is the regime where the pathology is strongest, and the document never connects D4's 1/√|H| dilution to §3.6's dense-data prediction.
>
> *H&M, 73,418 users / 13,651 items / 476,394 train rows:* +M·d = **874 K new free parameters against 476 K observations** — roughly two new parameters per interaction in the new block alone, each row seeing ~6.5 gradient touches per epoch on average and, given Zipfian degree, a *median* item seeing 1–3. §4.2 calls this "a wide spread in row quality — rare-item identity rows stay near-init." That is not a spread; that is an unidentified block with more parameters than data.
>
> Dense → washes into the popularity direction. Sparse → untrained. The candidate has no operating window, and both closures are derivable from statements already in the file.
>
> **9. At H&M sparsity, "collaborative" is a mislabel, and the rendered artifact in §3.4 will state something false.**
>
> With mean |H| = 6.49 across 13,651 items and 73,418 users, most items are bought by a handful of users and many by one. §4b C9's compensating family is exact precisely when a user's purchases are theirs alone — and at H&M's density that is close to the typical case. So on H&M the identity block is, to a good approximation, a *re-parameterized user-ID table*: private memorization wearing a collaborative label. Arm D is then predicted, from the document's own C9, to be ≈ arm B; and arm C smuggles per-user free parameters back into the "noid" arm that existed to demonstrate their absence. The §3.4 mock-up renders `item identity +0.21 (29%)` **nested under a community row** ("dark denim menswear"). On H&M that 29% is not community evidence — it is the model remembering this user. The read-out will assert co-purchase structure where the parameters encode none, which is a fidelity failure of exactly the kind R2 exists to prevent.
>
> **10. The §3.4 rendering pre-supposes the central risk does not occur, and the read-out claimed to "best survive" fails where the bet is placed.**
>
> The concrete artifact shows name-coverage **0.61** — comfortably above K_cov's proposed τ ∈ [0.3, 0.6] floor. Those numbers are invented, and they are the only picture the document offers of its deliverable. D1 ("name-coverage collapse... structural, not incidental," §3.5) is the declared central risk, and the illustration quietly assumes it away. Separately: §3.4 read-out 3 (per-purchase regrouping) is called "the read-out that best survives the change." It survives at |H| = 6.49 (H&M — where §3.6 predicts null) and degenerates into "88 further purchases +0.04" at |H| ≈ 93 (ml-1m — where the gain is predicted). The strongest surviving explanation and the predicted accuracy gain are on different datasets, and the document presents each in isolation.
>
> **11. §4.2/§4.3's cost analysis omits the one tensor that actually scales, on the dataset that matters.**
>
> §4.3 counts the padded *buffer* (ids + weights, 12 B/slot): ml-1m ≈ 196 MB at padded width ~2,700. §4.2 says "FLOPs still negligible." Neither counts the forward intermediate. The code path in §3.2 is `(embedding_layer(ids) * w.unsqueeze(-1)).sum(dim=-2)`, whose pre-reduction tensor is **(B, D_max′, d)** and is retained by autograd. At B = 1024, D_max′ = 2,700, d = 64, fp32: **≈ 708 MB for one intermediate**, before the K_u × d prototype math and the (1+n_neg) scoring. That is the real reason the flat `embedding_bag`/offsets layout exists, and §4.3 relegates it to "not needed at V1 scale; specified so the build has a fallback rather than a surprise." It is needed at V1 scale on ml-1m — the arm the candidate exists to run. The "≈1.0–1.05× wall clock" and "no budget-gate breach" verdicts in §4.2/§4.5 rest on a FLOP count for a memory-bound gather with ~97% padding waste (mean 93 real slots in a 2,700-wide bag).
>
> **12. The empirical bet is hedged into unfalsifiability: every cell of the outcome matrix has a pre-written favourable reading.**
>
> - ml-1m gains → the bet pays.
> - H&M null/negative → "**hm rows are the falsification arm, not the hope**" (§4.5); the prediction is confirmed.
> - Both null → §5.2 has already banked the win: "λ_y sweeps the trade, and name-coverage measures the price at every setting. That makes dc08 the lineage's **most direct empirical instrument** for the tension the requirements doc names — **arguably its strongest thesis contribution independent of whether the accuracy bet pays**."
> - Coverage collapses → D1 was pre-declared "measured (not prevented)", so collapse is a *result*, not a failure.
>
> Only one outcome is admitted as falsifying: the *reversal* (H&M gains, ml-1m does not) — which, per point 3, is the outcome the cited literature actually predicts. The document has arranged matters so that the likeliest disappointing result (flat everywhere) is scored as a successful measurement. Add to this that there is no gate on the prototype arm at all (only the dot-twin's inherited +0.02), no seed count, no variance estimate, and dev-tier single-seed 30-trial hyperopt as the instrument — and a 0.02 HR@10 "gain" is not distinguishable from hyperopt noise by anything specified here.
>
> **13. The dot-twin-first plan is an asymmetric gate that cannot confirm the candidate — and the host penalty it ignores is larger than the gain it seeks.**
>
> §4.5 makes `lightfm_hist` + identity rows the first run, and §3.6 states the logic in one direction only ("if it does not move there, the prototype arm is unlikely to"). Fine as a stop, useless as a go. But the numbers in §0 say something the document never confronts: the *same* composition scores **0.612 under a plain dot (`lightfm_hist`) and 0.544 under the prototype head (fU)**. A 0.068 host penalty — three times the +0.02 gate, and comparable to the entire gain the candidate hopes to buy. fU's diagnosed problem is the layer above the composition, and this candidate changes the composition. §0 asserts "the missing channel — not composition cleverness — is where fU's headroom lives"; the file's own two numbers say the largest identified loss is the head. Passing the dot-twin gate would demonstrate that the channel works *under a dot product* — i.e. would re-confirm SVD++ — while saying nothing about whether it survives the cosine, which is the only thing §2b claims is new.
>
> **14. The contribution surface, taken at the document's own valuation, is an ablation.**
>
> §2b concedes the standing honestly: "**known-mechanism, new-host, new-read-out**," with DNCF (arXiv:2102.02549) already occupying "free row + ID history, SVD++-inspired," and no novelty claimed for the pooling. That leaves three claims. Claim (1), the host, is undermined by points 6–8: the prototype head removes the mechanism's control knob and supplies the popularity pathway that D3 describes. Claim (2), the read-out, is undermined by points 1–2, and is in any case **not specific to this candidate** — dc05 arm B and dc01 already carry unnamed `e_ID` mass, and F-DC01-06's "frozen-difference anatomy" is cited in §1b as prior art for exactly this problem. If a named/unnamed coverage split is genuinely mandatory for honesty, it was mandatory two candidates ago; presenting it as dc08's contribution is billing a lineage-wide bookkeeping repair to this cycle's account. That leaves claim (3), the four-arm decomposition — which is an experiment, not a design. A whole cycle, +874 K parameters on H&M, and four cluster runs to obtain a fingerprint-table cell (§0) plus a table of ablations.
>
> **15. R8 is applied asymmetrically: it waves through what the author wants and blocks what the design needs.**
>
> §1c calls this "one idea," discounting LOO masking as "a correctness condition of the same summand" and the coverage split as "a disclosure rule on the existing read-out, not a mechanism." But §3.4 says of that same read-out: "**This is not decoration: it is the price tag on Koren's trade, and without it the candidate would be claiming a grounding it does not have.**" A component whose absence would void the candidate's R2/R4 compliance is load-bearing, and load-bearing components count. Meanwhile §3.7 deviation #2 — pooling content words and identity rows in one mean, which neither source does and which is "the whole point" — is not counted as an idea at all. And the five knobs that would address the four unprotected degeneration modes are all excluded on R8 grounds. R8 is thus doing the work of licensing the mechanism while forbidding its repairs. Note also that §4b.4's F-DC08-03 concludes K_ycenter is "substantially more attractive than it looked when designed" — K_ycenter *is* dc06's centring, applied to the new block, and dc06 was excluded from this cycle by R8 in §1b. The firewall is already leaking, and the recorded consequence is self-inflicted rework: "if both land, centring must be re-derived over a composition that now contains unnameable summands."
>
> **16. Nothing in the design changed as a result of any concession — including the §4b findings.**
>
> Inventory the concessions: §0 pre-frames a "priced trade"; §1d's two-axes gut-check pre-concedes the interpretability cost; §2.1 borrows Koren's own admission (from an author who *abandoned* the explainable variant to get the accuracy); §5.1 R5 pre-concedes "anyone quoting it as a cold-start contribution would be overclaiming"; §5.5 concedes **four of six degeneration modes unprotected** and immediately neutralises it ("not a veto (spectrum rule)"); §4.1 concedes dc05's 172× parameter-economy headline is spent, then recovers the narrative one sentence later with Koren's benefit #2 — which point 9 undercuts, since rare-item `y_j` are per-user parameters in all but name. Now name one design decision that any of these changed. There is none: every knob is "designed, deliberately NOT stacked," and §4b.4 opens by declaring the math-check findings are "carried forward as recorded facts, **not as edits that quietly repair earlier sections**." That is the protocol's anti-retrofit rule being used to leave a read-out in the design that the same section just proved is gauge-dependent. Disclosure has been substituted for engineering throughout. A document that has pre-conceded every objection is not thereby immune to them — it has simply arranged for none of them to have consequences.
>
> **17. The "open niche" is an artifact of the author's own bookkeeping.**
>
> §0 establishes the gap from `candidate_index.md`'s fingerprint table: no registered row carries "a collaborative identity channel inside a composed representation." But the host is a *collaborative filtering* model end to end — prototypes, the free item table `t`, and `B(t)` are all collaborative. The novelty is an empty cell in a taxonomy the author designed, which is not a research reason. And the sibling-cycle precedent invoked to waive distinctness (§0, §1c: "dc05-M1 → dc06 → dc07") is worth reading as evidence rather than as license: dc07 was a knob on the same host, and it **stopped as a controlled negative** three weeks ago, defeated by exactly the mechanism (popularity re-forming under a new similarity) that §3.5 D3 predicts here. The document cites dc07's outcome as a warning and does not draw the inference about the pattern that produced it.
>
> **Bottom line.** The mathematics is clean and the disclosure is exemplary; neither is the problem. The problem is that (i) the mechanism's own control knob is deleted by the chosen host and then frozen at a value picked for read-out convenience; (ii) the mechanism has no operating density — it washes into a popularity direction at ml-1m's |H|≈93 and is untrained at H&M's |H|=6.49, with 874 K parameters against 476 K observations; (iii) the read-out sold as the contribution is provably not invariant to the model's own parameterization, by the file's own §4b C9/C10; (iv) the external evidence predicts the opposite regime from the pre-registered one, and shows the mechanism at a wash against plain MF on the target dataset; and (v) the outcome space has been arranged so that no result can disconfirm the candidate. This should not be built as specified. If anything here is worth a run, it is the `lightfm_hist` + identity-row dot twin — as a cheap negative control for the *host*, not as stage one of a four-arm prototype campaign.

### 5b.2 Response (point by point; amendments are visible, earlier sections are not silently rewritten)

This is a strong pass and it lands. I concede more than I rebut. Per the
protocol's anti-retrofit rule I do **not** edit §§0–5 to hide the hits; instead
the response records **amendments A1–A12**, each naming the section it modifies,
and those amendments are binding on the build and run plan.

**1. Coverage metric not parameterization-invariant — CONCEDE the C9 half, REBUT the C10 half.**
The C9 half is right and it is the single most damaging finding in this
document: with `use_id = True` and λ_y > 0, δ can be moved between e_ID(u) and
{y_j} with the function unchanged, so the personal/collaborative split — and
hence any coverage number computed from it — is not a property of the model.
Calling it "the price tag" while it is re-printable was wrong. **But the C10
half is a misreading:** adding c to every y_j *changes the ranking* (4b C10
measured 5 of 7 users reordered), so it produces a **different model**, not a
reparameterization of the same one; §5.5 already recorded this as "not a gauge
but a free direction". A metric that differs between two different models is not
thereby non-invariant. → **A1 (amends §3.4):** the coverage read-out is
**restricted to a canonical gauge**: (i) the three-way split is reported **only
in arm C** (no user row ⇒ the C9 family provably does not exist — it requires
`use_id = True`); (ii) in arm D only the *named vs unnamed* two-way split is
reported, since that aggregate **is** C9-invariant (δ moves mass between the two
unnamed blocks, never into or out of the named block); (iii) the identity
block's mean is removed **at read time** before shares are computed, with the
removed mass disclosed as its own line (dc06's presentation discipline). The
mandatory-read-out claim survives only in that restricted form.

**2. "Named vs collaborative" labels blocks, not semantics; C−A isolates capacity — CONCEDE both.**
Nothing stops y_j from learning its own item's content, and on ml-1m's
genres+tags bags the word bag is already close to an item fingerprint, so the
label is weaker there than §3.4's rendering implies. And C − A does confound the
mechanism with added capacity. → **A2 (amends §3.6, adds an arm):** the run
matrix gains a **capacity control** — arm A with the identity block present but
**frozen at random init** (`requires_grad = False`), i.e. identical parameter
count and identical composition arithmetic, zero learned identity signal. C −
A_frozen is then the mechanism contrast; C − A is the capacity+mechanism
contrast; the difference between the two differences is the capacity share. Plus
a diagnostic (parameters only, no run cost): **cos(y_j, Σ_{f∈f_j} e_f)** per
item — if identity rows converge on their own content, the "named/unnamed" label
is misleading and must be disclosed as such rather than reported straight.

**3. The pre-registered regime prediction contradicts the NAIS quote — CONCEDE the contradiction, with one precision.**
The precision: NAIS's sentence compares *item-based vs user-based architectures
across datasets* (Pinterest ≈ 27 interactions/user vs MovieLens ≈ 165 in the NCF
preprocessing), not "adding y to an existing content pool"; and H&M at |H| = 6.49
is far below both, where the identity rows have ~6 gradient touches each, so the
Pinterest→sparser-is-better extrapolation does not obviously reach us. **But the
substance stands: the one published regime statement I quoted points the other
way from my prediction, and I did not confront it.** → **A3 (amends §3.6):** two
hypotheses are registered with their sources — **H-dense** (mine: gains where
histories are long) and **H-sparse** (NAIS §4.3: identity pooling is *more*
advantageous as data sparsifies) — together with the discriminating outcome
(H&M gains + ml-1m flat ⇒ H-sparse; the reverse ⇒ H-dense; flat everywhere ⇒
neither, see A10). Registering the competitor is not hedging: it names in
advance which result kills which hypothesis.

**4. §0's motivation performs a subtraction it later forbids, and FISM 66.47 < MF-BPR 66.64 was printed and never stated — CONCEDE fully.**
This is the cleanest hit in the critique. The comparison did rhetorical work
under a disclaimer. And the omitted line is material: on the very table cited,
identity-only pooling **loses to plain MF with free user factors** on ml-1m.
→ **A4 (amends §0 and §2.3):** the motivation is restated as *"fU trails its own
host and we do not know why; item identity is one untested channel"*, and the
FISM-vs-MF-BPR wash is recorded as **evidence against** the channel being large
on ml-1m — the first genuinely negative literature datum in this file.

**5. FISM never read primarily; the +0.02 gate inherited unexamined — CONCEDE.**
→ **A5 (amends §4.5):** the literature-inherited +0.02 threshold is **dropped**.
It is replaced by a **measured** gate: run the existing `lightfm_hist` config at
≥3 seeds to establish the dev-tier noise band, and require the identity arm to
clear that band. The FISM primary source stays a flagged fidelity gap; no
thesis sentence may lean on FISM specifics until it is read.

**6. The host deletes the mechanism's control knob; "no prior work reports it" is a warning, not a niche — CONCEDE the framing, REBUT the implication.**
Conceded: §2b(1) sold a confound as a contribution surface. The cosine does
discard exactly the quantity SVD++/FISM/NAIS all found necessary to control.
Rebutted: the host is not a design choice this candidate is free to make — it is
the thesis lineage's stage-2 host, fixed since 2026-07-12. The right response is
therefore not "pick another host" but "measure the host's effect". → **A6
(amends §2b and §4.5):** the claim is re-labelled from *contribution* to
**confound under measurement**, and the dot twin is promoted from a one-way gate
to a **paired 2×2** (head ∈ {dot, prototype} × identity ∈ {off, on}), which
estimates the host penalty, the channel, and their interaction in one design.

**7. α frozen at 1 for read-out convenience — REBUT for the arm that matters, CONCEDE for the other.**
4b C7 measured α **exactly inert** when `use_id = False` (Δcos = 3.3e-16) and
materially active when `use_id = True` (Δcos = 0.96). Arm C is the `use_id =
False` arm — and by F-DC08-02 it is also the only arm whose read-out split is
identified. So in the arm this candidate should report, α = 1 is provably
lossless with respect to direction, and λ_y (not α) is the balance knob; the two
are not substitutes and λ_y is swept. The concession: in **arm D**, α is a real
and unswept degree of freedom, and a null there is indeed ambiguous between
mechanism and α. → recorded, no amendment needed beyond A1's arm-C restriction,
which already makes C the reporting arm.

**8. No operating window (dilution at ml-1m, no data at H&M) — CONCEDE the H&M half fully; CONCEDE the ml-1m half in the cosine-specific form only.**
*H&M:* +874 K free parameters against 476 K observations, median item seen by
1–3 users, is a decisive statistic and §4.1/§4.2 did not frame it as one.
**Consequence I accept and that costs me a rhetorical position:** a null on H&M
is then *uninformative about the mechanism* — it is a capacity pathology, not
evidence — so H&M can no longer be called "the falsification arm". → **A7
(amends §3.6 and §4.5):** H&M is reclassified from *falsification arm* to
**capacity-pathology arm**, explicitly non-diagnostic for the mechanism; this
deletes one of the pre-written favourable readings the critique lists in P12.
*ml-1m:* the 1/√|H| dilution argument is real but proves too much on its own —
FISM pools ~165 identity rows on ml-1m and reaches 66.5–70.2 HR@10, so
mean-of-many-identity-rows is not intrinsically fatal. What is specific to us is
that FISM keeps the magnitude while our cosine discards it, leaving the shared
direction non-removable at score time. So the point survives **in its
cosine-specific form**, which is exactly D3/D4. → **A8 (amends §3.5, §4.5):**
the dilution instruments (‖ȳ‖ / mean‖y_j‖; identity-share vs |H_u| strata;
identity-share vs item-popularity Spearman, dc07's instrument) are moved from
"nice to have" into the **required** read-out set, and **K_ycenter is designated
the candidate's first repair** if they fire — named in advance rather than
discovered afterwards.

**9. At H&M sparsity "collaborative" is a mislabel and the artifact will state something false — CONCEDE, and this one changes the deliverable.**
If most items have 1–3 buyers, y_j is private memorization, and rendering it
nested under a community row as collaborative evidence is a fidelity failure of
the kind R2 exists to prevent. → **A9 (amends §3.4):** the identity line is
labelled **by measured buyer count**: items with fewer than n_min distinct train
buyers render as *"item-specific (this purchase, remembered)"*, at or above
n_min as *"co-purchase evidence"*, with n_min declared and the population split
reported alongside. The rendering is thereby made honest per item rather than
per architecture.

**10. The illustration assumes the central risk away — CONCEDE.**
The 0.61 coverage in §3.4's mock-up is invented and flatters the design.
→ folded into **A9**: the section must show **two** renderings, one at healthy
coverage and one at collapsed coverage (what the artifact looks like when D1
happens), so the reader sees the failure mode's deliverable, not only the
success mode's.

**11. The autograd intermediate (B, D_max′, d) was never counted — CONCEDE fully; this is a straight error.**
At B = 1024, D_max′ ≈ 2,700, d = 64, fp32 the retained pre-reduction tensor is
≈ 708 MB, and the LOO path adds a second (B, D_item, d). §4.2's "FLOPs
negligible" and §4.3's "fallback, not needed at V1" are wrong on ml-1m, which is
the arm the candidate exists to run. → **A10 (amends §4.3 and §4.4):** the flat
`embedding_bag`/offsets layout is **required on ml-1m**, not a fallback; the S5
checklist row "cheap indexed computation" is downgraded from unqualified *Yes*
to *Yes only under the flat layout*; the padded layout stays acceptable on H&M
(D_max′ ≤ 201).

**12. The outcome space is arranged so nothing disconfirms — CONCEDE the structure, and remove the rescues.**
→ **A11 (amends §5.2 and §4.5), the disconfirmation clause:** *if the ml-1m
prototype arms show no gain over their host beyond the measured noise band
(A5), the candidate's accuracy bet has **failed**, and the "most direct
empirical instrument for the fidelity-vs-accuracy tension" framing in §5.2 may
**not** be used to rescue it.* A failed mechanism whose knob can be swept is a
failed mechanism. Coverage collapse (D1) likewise counts as a **negative result
about the design**, not merely as a measurement. With A7 removing the H&M
consolation and A3 naming the competing hypothesis, the matrix now has a
majority of disconfirming cells.

**13. The host penalty (0.612 vs 0.544) is three times the hoped gain, and the dot twin cannot confirm — CONCEDE, and this is the most useful thing in the critique.**
The file's own two numbers say the largest identified loss in fU sits in the
**head**, not in the composition — and this candidate changes the composition.
That is a scope-level finding, not a detail. → covered by **A6**'s paired 2×2,
and **surfaced to the user in the Step-6 summary as an open strategic question**:
if the head is where fU loses 0.068, the highest-value next cycle may be a
head-side investigation, not another composition knob. That is the user's call,
not mine.

**14. The contribution surface is an ablation; the coverage split is a lineage-wide repair billed here — CONCEDE claim (2), REBUT the framing of claim (3).**
Conceded: if a named/unnamed coverage split is mandatory for honesty, it was
mandatory at dc01 and dc05, whose unnamed e_ID mass has the same problem.
→ **A12 (trickle-back, M7 spirit):** the coverage split is filed as a
**lineage-wide read-out repair to be backported to dc01 and dc05**, not as
dc08's contribution; dc08's claim reduces to (1) the host measurement and (3)
the decomposition. Rebutted only on tone: an experiment that decomposes a
representation's power into content / identity / personal channels **is** a
legitimate research output, and "a table of ablations" is what a stage of this
lineage is supposed to produce. But the critique is right that this is a smaller
claim than §2b implied.

**15. R8 applied asymmetrically — PARTIALLY CONCEDE.**
Conceded: the coverage split is load-bearing (my own §3.4 says the candidate
would be overclaiming without it), so counting it as "disclosure, not mechanism"
was convenient. Conceded: the K_ycenter/dc06 firewall is leaking, and A8 now
designates dc06's centring as this candidate's first repair — which is exactly
the entanglement §1b excluded. Rebutted: deviation #2 (pooling both summand
kinds in one mean) is not a second idea — it *is* the idea; "add identity rows
to the existing pool" and "pool them together" are the same sentence.

**16. No concession changed the design — this response is the answer.**
A1–A12 change the read-out's scope (A1), the arm set (A2), the pre-registration
(A3, A11), the motivation's sign (A4), the gate (A5), the run design (A6), the
role of the H&M rows (A7), the required instruments and the named first repair
(A8), the rendered artifact (A9), the required data layout (A10), and the
ownership of the coverage claim (A12). Whether that is enough is the user's
judgement; the criticism that disclosure had been substituted for engineering
was fair as the document stood.

**17. The niche is bookkeeping; the sibling-cycle pattern is evidence — PARTIALLY CONCEDE.**
Rebutted on the niche: the host being a CF model is not the point — dc05's
*composition* provably cannot carry cross-user identity (a user row is trained
from one user's rows alone; a word row carries no item identity), so the absent
channel is a structural fact about the representation, not a taxonomy cell.
Conceded on the pattern, and it is worth the user's attention: this is the third
consecutive sibling-cycle knob on an existing host (dc06, dc07, dc08); dc07
stopped as a controlled negative, defeated by popularity re-forming under a new
similarity — the same mechanism D3 predicts here. **Surfaced in the Step-6
summary**, since deciding what that pattern means for the lineage's next cycle
is a strategic call for the user.
---

## 6. Fingerprint & registration

### 6.1 Mechanism fingerprint

- **Features-entry:** *identity-as-ingredient* — the pooled user representation
  gains a summand that is not a feature: one free per-item history row y_j
  (SVD++/FISM) added to dc05's mean of attribute words, inside the same bag,
  under the same 1/|H_u| weight and the same ‖q_u‖ normalization. No new layer,
  loss, or representation; the composition's *summand kinds* change from two
  (words, self-row) to three (words, other-entities' identity rows, self-row).
- **Grounding:** *unchanged and partly forfeited* — user prototypes are still
  grounded by shared geometry in the item word vocabulary (cos(e_f, p^u_l),
  unenforced, dc05's mechanism); the identity block is **ungrounded by
  construction** and enters the read-out as declared unnamed mass. The
  candidate's grounding position is therefore "dc05's grounding, over a
  representation that is only partly nameable, with the nameable fraction
  measured".
- **Read-out:** *dc05's read-outs with a coverage split* — per-prototype
  t_l·u\*_l over the disclosed B(t) baseline; exact per-word shares; the exact
  per-purchase regrouping (which **gains** the identity share and is the read-out
  that survives best); prototype word profiles; **plus** the mandatory
  named / collaborative / personal mass split with a name-coverage scalar,
  restricted per amendment A1 to the canonical gauge (three-way split in arm C
  only; named-vs-unnamed two-way split in arm D), and labelled per item by
  measured buyer count (A9).

**Distinctness:** differs from every registered row on the **entry** axis
(no other candidate's representation contains another entity's free identity
row); dc05-distinctness is waived by the sibling-cycle precedent (dc05-M1 →
dc06 → dc07), with the substantive difference stated above.

### 6.2 Status and what the user is being asked to decide

**Status: `draft — awaiting review`, with a heavy 5b bundle (A1–A12) attached.**
The cycle did not end where it started: the adversarial pass materially changed
the design, and two of its points are strategic rather than technical (see the
summary). The candidate as it now stands is **arm C-centric** (the only arm
whose read-out split is identified, and the only arm where α is provably inert),
**paired-2×2 first** (head × identity, to separate the host penalty from the
channel), and **explicitly disconfirmable** (A11).

### 6.3 Build notes carried for whoever implements this

- Row layout `[words V] ++ [item history rows M] ++ [user rows N]`; pass
  `n_features = V + M` so the existing ID-row offset arithmetic is unchanged
  (§3.2).
- `item_token_ids` **must** carry the item's own identity slot, or the positive
  leaks into its own predictor (§4.4). Assertion written as a tolerance check,
  not `== 0` (**F-DC08-01**, 4b C8: ~1.5e-8 in fp32 at odd |H_u|).
- Keystone reductions to verify in `dc_checks/dc08`: λ_y = 0 ⇒ bit-identical to
  today's fU (4b C2 confirmed at 0.0); identity rows frozen ⇒ the A2 capacity
  control.
- Flat `embedding_bag`/offsets layout **required on ml-1m** (A10).
- Instruments required in the read-out set (A8): ‖ȳ‖/mean‖y_j‖, identity-share
  vs |H_u| strata, identity-share vs item-popularity Spearman, cos(y_j, content)
  per item (A2), buyer-count distribution (A9).

### 6.4 Build record (2026-09-09, same session as the cycle)

Implemented on `feat/scrutiny`, single code lineage, following the dc06/dc07
knobs-not-a-new-`ft_type` pattern.

**Code.** `HistoryFeatureEmbedding` gained `n_history_rows` /
`history_row_weight` / `freeze_history_rows`; the table layout is now
`[words: V] ++ [item identity rows: M] ++ [user rows: N]` and the user-ID row is
addressed through a new `id_row_offset` property (`is_history_row(code)` marks
the unnameable block). **The forward pass is unchanged** — the design's central
implementation claim held: an identity row is just another slot of the existing
bag, so `build_user_history_weights(history_id_rows=True, history_row_weight=λ_y)`
emits `(V + j, λ_y/|H_u|)` per purchase plus the matching `(V + j, λ_y)` slot in
the leave-one-out payload, and the *existing* algebraic mean-subtraction does the
rest. Wiring: `inject_feature_ids` reads the two knobs and returns
`n_history_rows`; `_dc08_kwargs` mirrors `_loo_kwargs` at both factory sites.
Read-outs: `per_purchase_rows` now adds each purchase's own `λ_y·y_i` (without
it the per-purchase regrouping would silently stop summing to the word-level
reading); the breakdown labels identity slots as
`"<item> [item identity]"` and renders the **A1 coverage split** — three-way
named/identity/personal in the noid arm, named-vs-unnamed only with a user row,
with the non-identifiability stated on the artifact.

**Arms registered:** `feature_user_proto_noid_itemid` (arm C, reporting),
`feature_user_proto_itemid` (arm D), `feature_user_proto_noid_itemid_frozen` (A2 capacity
control), `lightfm_hist_itemid` / `lightfm_hist_ids_itemid` (dot half of the A6 2×2).

**Checks.** `Master/temp/dc_checks/dc08/t01_identity_rows.py` and
`t02_readout_exactness.py` — **all pass**, both layouts × both id arms:
builder slot placement, `history_id_rows=False` byte-identical to dc05, 4b C1
bag≡math form, **4b C3 leave-one-out exact including the identity row**,
`id_row_offset`, 4b C8 gradient reaches purchased rows / ≤ round-off on the
LOO-removed positive, A2 freeze, the full inject→factory→RecSys chain for both
`ft_type`s, and read-out exactness (word rows and purchase rows both sum to q_u;
shares sum to `cos(q_u, p_l)`; the three-way split partitions the mass).

**Regression sweep:** dc01, dc05, dc07 suites re-run. `dc05/t14` needed one
update (the builder's `return_item_tokens` tuple grew from 6 to 7 with
`n_history_rows`) and passes. Three failures elsewhere (`dc01/t05`, `dc01/t10`,
`dc05/t04`) were verified to **pre-date this work** by re-running with the
registry edits stashed — untouched, not caused here.

**New finding — F-DC08-05 (bit-identity at λ_y = 0 is layout-dependent).** The
4b C2 keystone reproduces exactly in the `fixed` layout but differs by ~6e-8 in
`bags`. Cause: the identity slots contribute exact `+0.0`, but they widen the
padded bag, and torch's blocked `sum(dim=-2)` associates the *same* nonzero
terms differently at a different width. The composition is unchanged; only the
reduction order is. The keystone that actually protects existing runs is
`history_id_rows=False`, which is byte-identical (tensors untouched) and is
pinned as such.

**A10 IMPLEMENTED (2026-09-09, same session — user directive: ml-1m is the
primary dataset, so the padded path was not acceptable).**
`HistoryFeatureEmbedding` gained `pooling ∈ {padded, bag, auto}`. In `bag` mode
the padded tensors are replaced at build time by a flat CSR view
(`flat_ids` / `flat_w` / `row_ptr`) and the composition is pooled with
`F.embedding_bag(mode='sum', per_sample_weights=…)` over a ragged gather, which
**never materialises the `(B, D_max, d)` tensor**. `auto` (what every dc08 arm
sets) switches at `D_max > 256`, so H&M stays on the padded path bit-for-bit and
ml-1m switches. The read-outs go through a new layout-agnostic
`user_slots(idx)` accessor. Measured at ml-1m shape
(`t04_scale_benchmark.py`: N = 6,034, mean |H| ≈ 109, max 2,000, W ≈ 300 word
slots, d = 100, B = 512):

| | padded | bag |
|---|---|---|
| padded columns D_max | 2,300 (82% padding) | — |
| buffers (ids + weights) | 167 MB | flat, 2.47 M real slots |
| retained `(B, D_max, d)` fp32 intermediate | **471 MB** (≈2× during backward) | **none — fused** |
| fwd+bwd, one batch | 1.10 s | 0.043 s |

**25× faster and the intermediate disappears**; compositions agree to 5.2e-7
relative, gradients to ~1e-7 relative (`t03_bag_pooling.py`, which pins
equivalence of composition, leave-one-out, gradients and read-outs across both
poolings, both layouts, both id arms, including empty-history and
single-purchase users).

*Numerical note found while benchmarking and dismissed:* the algebraic
leave-one-out `(|H|·q − e_pos)/(|H|−1)` amplifies any error in `q` by `|H|`. On
real builder output this is harmless — `e_pos` is exactly one of `|H|` summands,
so the cancellation is ~1/|H| and the relative error is preserved. It only
looked alarming on the synthetic benchmark bag, which is deliberately
inconsistent with its item-token table.
- **A9's buyer-count labelling** — the renderer marks identity lines as
  *"[item identity]"* and deliberately avoids the word "collaborative"; the
  per-item buyer-count threshold and its population disclosure are not yet
  computed.
- **A5's measured noise band, A8's dilution instruments, K_ycenter** — run-plan
  and instrument work, not model code.
- **No training run has been made, and ml-1m has not been exercised end-to-end**
  (the split is not staged on the laptop — `dc05/t11`/`t12` skip for the same
  reason). Everything above is toy-scale and synthetic-scale evidence; the real
  ml-1m chain (builder → factory → trainer, and the actual D_max the split
  produces) is unverified. That smoke belongs on the LEO5 login node.
  **Nothing here is evidence about the mechanism.**

### 6.5 Cluster verification + first queued run (2026-09-09)

**Real ml-1m geometry measured on LEO5** (not estimated): canonical fields
`['genres','tags']`, layout `bags`, **V = 1,061** nameable word rows,
**M = 3,125** identity rows, **N = 6,034** users, **|H_u| mean 93.2 / median 56 /
p90 222 / max 1,415**, **D_max = 2,437** padded columns at **22.2% fill**
(176 MB of padded buffers). `auto` therefore selects **`bag`** — A10 was load-
bearing, not a precaution.

**GPU numbers, A30, d = 100, B = 512** — and a correction to the CPU benchmark
in §6.4:

| pooling | peak VRAM | fwd+bwd |
|---|---|---|
| `bag` | **80 MB** | 0.059 s |
| `padded` | **1,198 MB** | 0.021 s |

On **GPU the padded gather is ~3× faster** (parallel gather beats
`embedding_bag`'s atomic scatter-add); the 25× speedup in §6.4 was a **CPU**
measurement and must not be quoted as a general claim. What `bag` buys on the
fleet is **memory**: 15× less per trial, and at the fleet's concurrency the
padded intermediate alone would approach an A30's 24 GB. The justification is
trial concurrency, not speed.

**Login-node smoke: PASSED** — `feature_user_proto_noid_itemid × ml-1m`, 2 trials ×
2 epochs, `RUN_OK: 2/2`, 2m08s, metrics produced end-to-end. Caveat recorded:
that smoke ran **CPU-only** (Peak VRAM 0 MiB, GPU util 0%) — a login-node
Ray/resource artefact, not a code issue; `torch.cuda.is_available()` is True in
the job env and the CUDA path was exercised separately (table above). Its output
directory was **quarantined** to
`_SMOKE_loginnode_feature_user_proto_noid_itemid_ml-1m_s38210573` because dev/smoke
runs share the results folder name with the real dev run and would otherwise
have been mistaken for it.

**Measured training cost** (A30, d = 64, K_u = 64, B = 512, real split):
**3.1 ms/step**, 1,098 batches/epoch → **~3.4 s/epoch**, **105 MB peak VRAM per
trial**. This is what sized the job (the replication report has no
`feature_user_proto × ml-1m` benchmark row).

**Queued: job 7786946** — `feature_user_proto_noid_itemid × ml-1m`, `--profile dev`
(30 trials / 60 epochs / patience 7 / grace 4), seed 38210573, ASHA on,
concurrency 5, `num_workers 1` (fleet convention, F-S0-02 eval-draw regime),
1× any GPU, 10 CPUs, 40G, 3h cap, tag `dc08`. Results folder
`feature_user_proto_noid_itemid_ml-1m_s38210573`.

**The contrast this run buys.** Arm A already exists as a **dev-profile fleet
row at the same seed**: ml-1m HR@10 `fU-noid = 0.5317` (with `fU (ids) = 0.5436`,
`lightfm_hist = 0.6120`, `lightfm_hist_ids = 0.6400`; nl-track SP0 memo, which
reproduces the fleet rows bit-for-bit on identical negatives). So this run is
directly comparable and the headline quantity is **C − A = result − 0.5317**.

**Reading rules that bind before the number arrives** (A3, A5, A11):
- **H-dense** (this candidate's registered prediction) expects a gain here;
  **H-sparse** (NAIS §4.3, which points the other way) expects the gain on hm
  instead. This single run can support H-dense but cannot discriminate the two
  on its own — that needs the hm row, which A7 has already reclassified as a
  capacity-pathology arm.
- The gate is **not** the retired +0.02 (A5): it is a **measured** dev-tier noise
  band, which does not yet exist for this combo. Until it does, a small positive
  C − A is not a result.
- Still missing for a defensible reading: the **A2 capacity control**
  (`feature_user_proto_noid_itemid_frozen`, which separates mechanism from the +M·d
  capacity) and the **A6 dot twin** (`lightfm_hist_itemid`, which separates the
  channel from the host). Both are registered and unqueued.

### 6.6 The dc08 ml-1m wave as queued (2026-09-09)

Three dev-profile runs, each its own `/leo5-submit` invocation (standing hard
gate), identical resource shape and fleet conventions (seed 38210573,
concurrency 5, `num_workers 1`, ASHA on, `hit_ratio@10`, 1× any GPU / 10 CPUs /
40G / 3h, tag `dc08`):

| job | model | role | contrast it enables |
|---|---|---|---|
| 7786946 | `feature_user_proto_noid_itemid` | **arm C — the reporting arm** | C − A vs the existing fleet row `fU-noid = 0.5317` |
| 7787048 | `feature_user_proto_itemid` | **arm D — the full model** (queued 2026-09-09 on user request) | D − B vs the existing fleet row `fU (ids) = 0.5436`; accuracy read only — its personal-vs-collaborative read-out split is NOT identified (4b C9 / F-DC08-02) |
| ~~7787013~~ | `feature_user_proto_noid_itemid_frozen` | A2 capacity control | **CANCELLED before starting** (user decision 2026-09-09: the fleet already carries many runs; scope held to the new history-id variant on ml-1m only) |
| ~~7787014~~ | `lightfm_hist_itemid` | A6 dot twin | **CANCELLED before starting** (same decision) |

**Scope as actually run (user decisions 2026-09-09): two runs — 7786946 (arm C) and 7787048 (arm D).** The
A2 capacity control and the A6 dot twin were queued and then cancelled before
starting, on the grounds that the fleet already carries many runs and this cycle
should test the new history-id variant on ml-1m and nothing else. Consequence to
carry into the read, stated here rather than discovered later: **C − A conflates
the mechanism with the +M·d capacity it adds** (no frozen control), and **the
result cannot be attributed to the identity channel as opposed to the prototype
head** (no dot twin). The 2×2 is therefore incomplete — the two existing fleet
rows (`fU-noid = 0.5317`, `fU ids = 0.5436`, `lightfm_hist = 0.6120`) give only
the identity-off column. What the C/D pair *does* buy on its own: the ids-vs-noid
contrast **within** the identity-row design, i.e. whether a user-ID row still
earns its place once item-identity rows are present — the D2 redundancy question
(§3.6), which is answerable from these two runs plus their two existing partners
without any further compute. Both controls stay registered and unqueued; if the dev number motivates
a production promotion, they are the first things to add back, since a bigger
search budget does not fix an attribution gap.

**Comparability constraint for any 100-trial follow-up (recorded before the
results arrive, so it cannot be rationalised afterwards).** The user's intent is
to promote this to `production` (100 trials) if the dev numbers look promising.
The trap: **arm A (`feature_user_proto_noid` × ml-1m) exists only at dev tier.**
Running arm C at 100 trials against an arm A at 30 trials would credit the
mechanism with a search-budget difference — the exact comparability error the
two-tier numbers policy exists to prevent. So a production promotion is **not one
run, it is a matched pair at minimum** (`feature_user_proto_noid_itemid` *and*
`feature_user_proto_noid`, both at 100 trials, same seed), and ideally the
frozen control too. Production runs land in `<model>_<dataset>_s<seed>_n100/`,
so they never overwrite their dev twins.
