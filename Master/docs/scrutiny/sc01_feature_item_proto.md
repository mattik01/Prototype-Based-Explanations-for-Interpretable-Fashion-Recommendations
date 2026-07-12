# SC(dc01) — Scrutiny Dossier: `feature_item_proto`

> Scrutinized under **Scrutiny Protocol v1.0** (`Master/docs/scrutiny_protocol.md`,
> 2026-07-11). Branch: `feat/scrutiny`. Candidate design doc:
> `Master/docs/design_candidates/dc01_feature_composed_factors.md` (2026-06-11,
> post-5b amendments). Findings: `findings_ledger.md` (F-DC01-NN).
>
> **Precondition declaration (explicit, per SC preamble):** S0 is complete
> through S0.7 **part 1** (charter C1–C8 ratified, S0-build landed and audited,
> instrument fixes F-S0-07..13 dispositioned). The S0.7 **reference fleet is
> still in flight** on LEO5 (jobs 6988922/60–64/66) — the frozen reference
> table does not exist yet. This routine starts early under the protocol's
> stated allowance; **SC(dc01) re-checks against S0.7 part 2 once the fleet
> lands** (at the latest at SC.6, which needs the frozen numbers). No step in
> the SC.1a–SC.5 block consumes reference numbers.

---

## SC.1a Concept self-scrutiny (2026-07-11)

**Scope:** ruthless first-party audit of the corrected-to-date dc01 concept —
the design doc as amended after Step 5b (2026-06-11), read against the S0
foundation (charter, S0.3 cold spec, S0.5 feature decisions) and current code
where doc claims are load-bearing.

### 0. Preamble (manifest, learnings, open-findings re-check)

- **Manifest loaded:** protocol (SC.1a section re-read), findings ledger,
  learnings ledger, requirements doc (2026-06-11 version incl. spectrum
  reframing), candidate index, dc01 design doc, dc01 claims spec + check
  results, `dc01_feature_composed_bias_gap.md`, supervisor presentation
  draft, the four interpretability vault entries, S0 dossier in full.
- **Learnings watch-outs applying here:** (1) *docs drift from code on
  load-bearing details* — hit immediately: design doc §3.5 says F=6 fields,
  the implemented config uses the canonical 5 (verified
  `confs/hyper_params.py:225-231`) → F-DC01-01. (2) *Absent-key config
  defaults are hazards* — factory defaults `use_id_feature=True` when the key
  is absent (`feature_extractor_factories.py:130`); dc01's configs carry the
  key explicitly, and the F-S0-07 guard covers the absent-key case at cold
  eval. (3) *Anchor to published protocol* — dc01's cold story is now anchored
  via S0.3's B5-faithful spec, which supersedes the design doc's looser §3.6
  commitment. (4) *Audit-scope seams* — respected in this step's subagent
  handoff plan (SC.1b gets the design doc + requirements doc only, per GR10).
- **Open/shared findings re-check against current code:**
  - **F-S0-07 [major] [dc01] [open]** — correct as recorded: part (a) guard
    verified present (`utilities/cold_eval.py:69-76` refuses
    `feature_item_proto` with `use_id_feature` enabled, including absent-key);
    part (b), the actual ID-row drop for dc01's nested branch, remains
    unimplemented — **scheduled work for SC.3 of this routine**, not a new
    finding. No dc01 cold run can silently mis-score in the meantime (loud
    refusal).
  - All other shared findings are `fixed`/`wontfix` with dispositions that
    still match the code; none stale.

### 1. Known-issues disposition (steel-man before adversary)

The design doc already carries a strong adversarial record (Step 5b, nine
points + responses; Step 4b, two deliberately-skipped claims; §3.8/gate-4c,
modes M1–M4) plus the post-hoc bias-gap note. Walked item by item:

| Known issue | Status today | Reasoning |
|---|---|---|
| 5b-1: contribution = "distributive law", narrative risk | **open (thesis-level)** | Unchanged; not a mechanism question. The narrative burden (combination + empirical behaviour, not algebra) lands at SC.9/SC.10. |
| 5b-2: prototype layer adds nothing over LightFM-MF attribution unless profiles are sharp | **open risk — now instrumented** | The committed comparison baseline exists in code since S0-build (`lightfm_tags`/`lightfm_tags_ids`, same `FeatureEmbedding` → same per-feature decomposition without prototypes). Profile-sharpness (entropy) stays the committed primary metric; measured at SC.8. |
| 5b-3: R4↔R6 coupling via `use_id_feature` | **open by design — ablation committed** | Charter C7 fixes the ablation discipline; `feature_item_proto_noid` + `_f0` configs exist. The frontier is charted, not decided a priori. |
| 5b-4: no-ID equivalence-class ceiling — scheduled tuple count | **resolved (measured, S0.5)** | Canonical-5 numbers: **6,517 signature classes / 67.5% of items share a signature / max class 97**. Cold-pool companion (committed draw): 40.8% within-pool twins, but ~**0.06 expected same-signature negatives per 99-negative row** — the ceiling is largely invisible in the 1+99 protocol; the honest display is the tie-block diagnostic (built, model-agnostic). Structural property stands, disclosure obligation stands. |
| 5b-5: shares-sum error, joint normalization, arithmetic≠semantic faithfulness | **resolved (doc amended 2026-06-11)** | Rendering obligations (sum to t*−1, +1 baseline shown, normalization caveat, optional counterfactual read-out) carry forward as SC.5 acceptance criteria. |
| 5b-6: profile validity — e_f's meaning is CF-trained, label confounding | **open risk** | Folded into the Steck disposition below (F-DC01-02): profile-validity check committed for SC.8. |
| 5b-7: R7 — taxonomy will dominate the vocabulary | **open, accepted as rated** | Unchanged; measurable via the decomposition itself. |
| 5b-8: no cold evaluation committed | **superseded (S0.3)** | The S0.3 spec (20% held-out, cold-vs-cold primary, B5-anchored) is strictly stronger than the doc's §3.6 amendment; machinery built and audited (S0-build D, S0.7 audits). |
| 5b-9: one-sided tie; grounded vs ungrounded explanation lines | **resolved-as-instrumented (S0.2)** | S0.2 made Block-I/Block-U **score-share reporting binding** for SC.5/SC.8 — stronger than the doc's rendering note. The one-sidedness itself is scoping (S2), upheld by S0.2's side disposition. |
| 4b-skip (i): cold/dense accuracy through training dynamics | **open — now measurable** | Exactly what S0.3 + the S0.7 reference fleet exist for; lands at SC.8. |
| 4b-skip (ii): frequent-value dominance ("Solid", 58%) | **open — partially instrumented** | `max_norm` knob plumbed (hard variant of K2); detection metric (share flattening / norm spread) to be fixed in the SC.6/SC.8 metric list. |
| M1 diffuse profiles / M2 value dominance / M3 ID routing / M4 profile overlap | **open by design (K1–K4 unstacked)** | Re-checked for completeness in §2c below — list stands. |
| Bias-gap note (`dc01_feature_composed_bias_gap.md`) | **dispositioned by S0 — doc fold-in pending** | Fleet runs `use_bias=0` (charter C3), so the channel is inert in every scored run; the cold runner refuses `use_bias≠0` without a declared policy; feature-composed bias remains an unstacked design seed (scratchpad). The design doc itself never mentions the channel → F-DC01-03. |

No new finding below re-discovers anything in this table (protocol SC.1a.2).

### 2. The scrutiny proper

#### 2a. Mathematical soundness

- **Derivations:** the decomposition identities (C1, C4, C6), the F=0
  reduction (C5), gradient paths through both score halves (C3), and cold
  non-degeneracy (C2) were verified to float precision by the 4b black-box
  check (10 seeds, F ∈ {0..11}, adversarial scales). Re-read this session; no
  gap found in the claim set itself.
- **Limit cases, checked fresh:**
  - *F=0 + ID:* exact host reduction (C5; pinned bit-identical by the
    dc_checks keystone). Sound.
  - *All-zero composed vector:* `cos(0, p)` is 0/0 — undefined. Reachable only
    if every looked-up row is exactly zero (never in float practice after
    init/training; cold path drops the ID row but feature rows are trained).
    Not a concept defect; flagged as an SC.4 robustness look (eps or assert in
    the cosine path).
  - *Norm scale:* q_i sums 6 rows (5 features + ID) → init-time E‖q_i‖ is
    ~√6× a single-row embedding. Cosine branch is scale-invariant; the
    projection branch (W_t·q_i) and L2/`max_norm` semantics see the larger
    scale. Same freedom the host has (norms unconstrained), but the hyperopt
    optimum (λ_L2, max_norm) may sit elsewhere — already anticipated in §3.8
    note 2. No defect; watch at SC.8 anomaly triage.
- **Invariants:** shifted cosine keeps t* ∈ [0,2]; shares c_{f,k} are signed
  (negative shares are legitimate anti-affinities and must be rendered as
  such — rendering note carried to SC.5).

#### 2b. Expressiveness

- **More information than the host?** Yes — 5 categorical fields (V=426
  values) the host cannot see.
- **Able to leverage it?** Yes — gradient reaches feature rows from both
  score halves (C3); capacity with `use_id_feature=True` is a **strict
  superset of the host's function class** at equal d (zero the feature rows →
  exact host; C5 is the F=0 special case of this argument). Optimization
  dynamics, not capacity, is the open half (4b-skip i).
- **Capacity ceilings, quantitative:** no-ID configuration is capped at
  **6,517 distinguishable item classes** (S0.5, canonical 5) — 47.7% of the
  13,651-item catalog is signature-unique; the rest share. With ID: no
  ceiling. Both numbers now stand in the record; the doc's "~31× table
  shrink" framing (§4) stays honest only next to the class-count disclosure.
- **Gradient concentration:** a shared feature row aggregates gradients from
  every item carrying the value ("Solid": 58% of items) — statistical
  efficiency upside (LightFM §2.3-1) and dominance risk (M2) are the same
  mechanism; empirics decide (4b-skip ii).

#### 2c. Degeneration-mode coverage (completeness re-check)

Design doc lists host modes (inherited regularizers) + M1–M4 with unstacked
knobs K1–K4. Fresh sweep for unlisted modes:

- *Feature-rows-to-zero collapse* (model silently becomes the host): this is
  the **extreme of M3**, and it is *detectable by design* — the
  feature-explained fraction metric reads ~0. Covered.
- *Duplicate-row dilution of in-batch regularizers* (no-ID config): already
  noted, §3.8 inheritance note 1. Covered.
- *ID-row norm inflation* (composed vector dominated by the ID row warm,
  making the cold drop a large distribution shift): interaction of M2×M3;
  surfaced separately as F-DC01-04 because it has a *cold-eval* consequence
  the mode list doesn't state.
- No other reachable mode found. **Mode list judged complete** (with the
  F-DC01-04 interaction made explicit).

#### 2d. Requirements delta (re-rating vs the doc's own table)

| Req | Doc rating | Re-rating | Delta justification |
|---|---|---|---|
| R1 tied | strong | **strong (confirmed)** | 5b-9's "redefinition" charge was correctly rebutted: instance sharing *is* parameter sharing, same mechanism as factories.py weight tying. One-sidedness is ratified scoping (S0.2). |
| R2 intrinsic | strong | **strong, with a now-binding caveat** | S0.2 sharpened the claim boundary: intrinsic grounding covers **Block I only**; score-share reporting is mandatory at SC.5/SC.8. Not a downgrade — the mechanism is unchanged — but the rating is honest only with the share number attached. |
| R3 prototype paradigm | strong | strong (no delta) | — |
| R4 grounded prototypes | partial | **partial (confirmed, now instrumented)** | Profile sharpness unenforced (M1) stands; entropy metric + lightfm attribution baseline now exist to measure it. Rating was neither generous nor harsh. |
| R5 sparsity robustness | strong | **strong mechanism, two caveats now on record** | (a) bias channel — inert under fleet `use_bias=0`, guarded otherwise (F-S0-07/F-DC01-03); (b) trained-with-ID → cold-without-ID operating-point mismatch (F-DC01-04). Neither breaks the mechanism claim; both must ride the empirical interpretation. |
| R6 dense accuracy | partial | partial (no delta) | The empirical bet; reference fleet in flight. |
| R7 vocabulary | partial | partial (no delta) | Accepted as rated (5b-7). |
| R8 parsimony | strong | strong (no delta) | Still one idea; K-knobs still unstacked. |
| S1 decoupled variant | strong | **strong (strengthened)** | The spectrum's middle rung is no longer hypothetical: `lightfm_tags`/`lightfm_tags_ids` are built, tested, and in the reference fleet. |
| S2 user-side | strong | strong (no delta) | S0.2 confirms non-preclusion. |
| S3 images | partial | **partial — stale caveat** | The row cites "the known partial-image-set caveat"; the image set was verified **complete** on 2026-07-11 (105,100 jpgs, 010–095). The architectural reservations (opaque lump attribution, S5 cache) stand; the data caveat is dead → trivial edit in the amendment bundle. |
| S4 LLM naming | strong | strong (no delta) | — |
| S5 pipeline | strong | strong (no delta) | Confirmed in implementation (indexed gathers only). |

No rating was found too generous or too harsh enough to flip; the deltas are
caveats-now-on-record, not grade changes.

#### 2e. Cold-start mechanism check (incl. bias channel and ID-keyed side channels)

Inventory of **ID-keyed channels** that survive at cold inference:

1. **Item ID embedding row** — the known one; drop convention spec'd (§3.4),
   guarded until implemented (F-S0-07(b) → SC.3).
2. **Item bias scalar** — the bias-gap note's channel; inert fleet-wide
   (`use_bias=0`, charter C3) and the runner refuses otherwise. Not in the
   design doc → F-DC01-03.
3. **No third channel found:** user branch is ID-keyed but cold *items* don't
   touch it; regularizer state is in-batch and stateless; `feature_ids`
   buffer is static catalog data (cold-safe by S0.3 leakage item 1).

**Does the S0.3 eval detect dc01's failure modes?** Walked mode by mode:
cold-vs-cold isolates pure feature skill (calibration confounds excluded by
construction); cold-vs-all is the designated surface where a mis-handled ID
channel *would* show (and is exactly where the guard currently refuses to run
dc01's `_ids` config until SC.3 lands the drop — correct sequencing, since
SC.3 precedes SC.6); the tie-block diagnostic displays the no-ID ceiling
honestly; D4 strata measure the warm-thin claim. **Verdict: yes — the
instrument detects what this design could break, and refuses what it cannot
yet score.** One interpretive caveat added as F-DC01-04 (the drop changes the
operating point for `_ids`-trained models; LightFM's own tags+ids cold result
is the precedent that it works anyway — Table 1, tags+ids ≥ tags on cold).

#### 2f. Steck cosine-caveat applicability

The vault caveat (2026-06-16_2102): cosine over MF embeddings can be
near-arbitrary because dot-product training objectives are invariant to
per-dimension rescalings (compensated between factors), which change cosines.
Applied to dc01's two read-out surfaces (shares c_{f,k}; profiles
cos(e_f, p_k)):

- **The exact-invariance argument does NOT transfer intact — and that is a
  (partial) steel-man.** ProtoMF's item branch *trains through the same
  shifted cosine it reads out*: a per-dimension rescaling of E (and hence
  q_i) is **not** score-preserving here, because sim(q_i, P^t) is
  cosine-valued in the loss itself (compensating a diag D on E inside
  cos(Dq, p′) cannot be absorbed by any fixed re-parameterization of P^t and
  W_t across all items). The read-out metric *is* the scoring metric — the
  specific train-on-dot/read-on-cosine mismatch Steck targets does not apply
  in that form. This argument is dc01-specific, load-bearing, and currently
  written nowhere → part of F-DC01-02's amendment.
- **What survives of the threat:** (a) *Rashomon multiplicity* — many
  near-equal-loss solutions with different profile geometry (Steck's point
  generalized beyond exact invariances); (b) *label confounding* (5b-6) —
  nothing pins e_"Dark Blue" to darkness beyond the interactions of items
  carrying it; correlated attributes inherit each other's meaning; (c) the
  **projection half** (t̂ = W_t·q_i) is dot-product-based and enjoys no such
  protection — but it is also not semantically read (Block-U-adjacent,
  reported via score share, never narrated as feature meaning).
- **Mitigation (proposed, cheap, non-stacked):** a **profile-validity
  spot-check at SC.8** on the real checkpoint — for a handful of legible
  values (dark colours, a department, a pattern), check that items carrying
  the value actually score high on cos(q_i, e_f)-adjacent probes /
  that the value's top-profile prototypes activate on items carrying it.
  Plus the standing rendering rule: profiles are *model-semantic*, not
  human-semantic, until validated — thesis wording at the level the mechanism
  supports (5b-6 response). Metric-swap / seed-stability probes stay
  full-profile-stage options (single-seed budget, C4).

### 3. Findings (proposals bundled; decision at this gate)

| ID | Severity | One-liner | Proposal (bundled) |
|---|---|---|---|
| F-DC01-01 | minor | Design doc §3.5/§4 field set (F=6 incl. price-band, "V≈436") superseded by charter C5 canonical 5 (V=426); implementation already canonical | Dated amendment to §3.5 (+§4 numbers): canonical 5, price_band deferred per S0.5 with the recorded performance-upgrade caveat; note config compliance. Includes the trivial S3 stale-image-caveat edit. |
| F-DC01-02 | minor | No Steck-caveat disposition in the design doc; the read-out's semantic reading is undefended in writing | Dated amendment adding §2f's analysis: trains-through-cosine argument (why the exact Steck form doesn't transfer), surviving risks (Rashomon multiplicity, label confounding, unprotected projection half), committed SC.8 profile-validity spot-check. |
| F-DC01-03 | minor | Bias channel (ID-keyed item-bias scalar at cold inference) absent from the design doc; lives only in a temp note | Dated amendment to §3.4/§3.7: the channel, its S0 disposition (fleet `use_bias=0`, runner refusal, F-S0-07 guard), feature-composed bias as unstacked scratchpad seed. Temp note then retired (pointer left). |
| F-DC01-04 | minor | Trained-with-ID → cold-without-ID operating-point mismatch: the `_ids` config never trains the composition it is cold-scored on | Dated amendment to §3.4 cold notes: name the mismatch, cite the LightFM tags+ids cold precedent as why it is expected to work, mark K3's ID-dropout as the unstacked mitigation if empirics disagree; SC.8 interprets `_ids` cold rows with this caveat. No mechanism change. |

All four are doc-level amendments (no code changes; no behavior change to any
scored path). Per the fix policy they are bundled here rather than silently
applied because they amend the *concept record* — SC.2 applies them as dated,
visible amendments after this gate.

### 4. Gate (SC.1a)

**Status: CLOSED — ratified by user 2026-07-11, all four items.**
(1) Known-issues disposition table accepted (5b-4 resolved by S0.5, 5b-8
superseded by S0.3); (2) F-DC01-01..04 approved as proposed — amendments
applied at SC.2; (3) requirements re-rating accepted (no grade flips,
caveats recorded); (4) proceed to SC.1b. → Next step: SC.1b.

---

## SC.1b Adversarial pass (2026-07-11)

**Setup (Ground rule 10):** independent subagent, inputs restricted to the
dc01 design doc (incl. its Step 5b critique + author responses) and the
requirements doc, plus the V1 scale facts (73k/13,651/623k, 5 fields, V≈426,
top value 58%). Blind to S0, to this session, and to the codebase. Brief per
protocol: what did the previous adversary miss; which rebuttals do not hold;
what breaks at H&M scale.

### Critique (verbatim)

> 1. **The inherited inclusion regularizers do not merely "survive byte-identically" (§3.8) — under H&M's feature skew they actively reward the degeneration modes M2 and M4.** With "Solid" on 58% of items and every q_i a sum of only ~7 rows, the composed item cloud is strongly anisotropic: most items share a large common component, so pairwise cosines concentrate and the item cloud collapses toward a dominant cone. Now look at what the host regularizers (Eqs. 4–5/9) optimize: R_{P→U} rewards each prototype for being near *some* in-batch item, and R_{U→P} rewards each item for being near *some* prototype — both are jointly maximized by parking all prototypes inside the dominant cone, and the host has **no prototype–prototype repulsion term** (§4's parenthetical "which `sim_proto` rewards [distinctness]" misreads Eq. 4: multiple prototypes may share the same best-matching object at zero cost). In the CF host this is benign because free ID embeddings spread out; with composed embeddings the geometry that made the regularizers harmless is gone. Consequences: t* spreads compress (the per-feature shares of t*−1 shrink toward noise against the +1·û baseline the explanation must carry), and M4 (profile overlap) becomes the *expected* outcome, not an "open risk." The §3.8 claim that the regularizers are "agnostic to how o_embed was produced" is true of the code and false of the loss landscape; the 4c table's "M2 partially protected by max_norm" is also hollow — max_norm caps row *norms*, but 58%-frequency dominance is directional (a unit-norm e_Solid still sits in most q_i) and gradient-frequential (e_Solid receives 2–3 orders of magnitude more updates than a tail colour row; LightFM's per-feature Adagrad adaptivity, which handled exactly this, is silently not discussed as a dropped mitigation).
>
> 2. **The no-ID arm cannot represent item popularity — the single strongest signal in fast-fashion implicit feedback — so the "fidelity–capacity frontier" the author promises in rebuttal 3 is confounded before it is measured, and the ID arm's explanations will degrade worst exactly on the items users see recommended.** Deviation 4 (§3.7) drops LightFM's per-entity bias terms; with `use_id_feature=False` the model must score all items in a feature-tuple class identically, so within-class popularity variance (hero items vs. dead stock, enormous at ~46 mean interactions/item under fast-fashion skew) is *unrepresentable*. The no-ID arm will therefore lose the ablation for popularity-expressiveness reasons that have nothing to do with the R4↔R6 explanation trade-off — the frontier reading is confounded. In the ID arm, sampled-softmax gradient routes that same popularity signal into ID rows, so ‖e_ID‖ grows precisely for popular items; those items dominate q_i's direction (feature shares collapse toward "item-specific: +x") *and* dominate recommendation lists — i.e., the shipped system's user-facing explanations are least feature-grounded on the items most often explained. Rebuttal 3's answer to "mutually exclusive fork" — "measurable, continuous, the ablation charts the frontier" — fails on its own terms: two configurations do not chart a frontier, every knob that would (λ_id, p_drop, K3) is explicitly excluded from dc01, and §3.2 additionally makes `use_id_feature` a **hyperopt dimension inside a warm-hit-ratio TPE search**, which contradicts §3.6's controlled-ablation framing: TPE tuned on warm validation will winnow the False arm early, so the no-ID configuration never receives a fairly tuned run.
>
> 3. **The per-feature shares are non-identifiable across the three nested taxonomy fields, so the flagship attribution splits one fact into arbitrary, seed-dependent fractions.** §3.5 admits department_name (197), product_type_name (102), section_name (51) simultaneously; in a retail hierarchy these are strongly mutually predictive (a "Trousers Denim" department nearly determines product type "Jeans" and a menswear section). When several feature rows are near-collinear, infinitely many allocations of the shared category direction across e_department, e_product_type, e_section yield the *same* q_i, the same score, and the same t* — but different rendered shares. The rendered example itself exhibits the symptom: "+0.45 Trousers Denim, +0.07 Trousers, +0.01 Jeans" distributes a single semantic fact across three vocabulary entries in proportions the training objective does not determine. §4's "attribution granularity" tension declares the problem solved ("per-value = per-field — no roll-up ambiguity"), which answers the wrong question: the ambiguity is *across correlated fields*, not within one. No cross-seed stability check of the shares is scheduled anywhere, and C1's exactness is irrelevant to it — the decomposition is exact for whichever arbitrary allocation SGD happened to land on. This is a distinct, sharper failure than the conceded point-6 label-confounding risk: even with perfectly meaningful e_f directions, the *split* remains meaningless.
>
> 4. **The counterfactual read-out adopted in response to critique 5 is invalid by the document's own standards — while the genuinely counterfactual half of the score sits unused.** The amendment to §3.4 proposes re-scoring with one feature row removed from the sum. But the model is trained under 0% missingness (§3.2: every item has exactly one value per field), so q_i minus a field is an input composition the model has never seen; the re-score is an off-manifold evaluation with no validated meaning. It also violates R2's letter: a second scoring pass performed to construct the explanation is "reconstructed afterwards by a separate procedure," the exact definition of post-hoc the requirements use against `top_k_items`. Meanwhile the design overlooks that half the UI-score already decomposes exactly *and* counterfactually: t̂ = W^t Σ_f e_f = Σ_f W^t e_f, so feature f contributes precisely u*ᵀW^t e_f to the u*·t̂ half — additive, norm-free, and equal to the true effect of deleting f on that half. dc01 built its entire read-out on the norm-entangled cosine half (§3.4 read-out 2), inherited the C6 coupling problem from that choice, and then patched it with an off-manifold procedure — when the linear half offered the clean attribution natively. Rebuttal 5's fix is thus both unnecessary (for half the score) and unsound (as adopted).
>
> 5. **The cold-evaluation commitment adopted from critique 8 is internally incoherent: the ID-drop rule biases cold scores in the magnitude-sensitive half, and model selection remains warm.** Deviation 6 (§3.7) drops the ID column for unseen items at inference, so a cold item's q sums F rows while every warm competitor's q sums F+1 including a *trained* ID row. The û·t* half is scale-invariant in q_i, but u*·t̂ is not: warm items carry a systematic, learned magnitude/direction bonus that cold items structurally lack, so in a 1+99 candidate set of one cold positive against warm negatives the protocol has a built-in handicap the document never analyzes (LightFM's Table 1 evidence does not transfer: different score form, biases present, AUC not sampled top-k). Separately, the amendment changes only the *test* protocol: the validation metric for model selection stays warm hit_ratio@10, so every hyperparameter — including `use_id_feature` and any regularizer weights — is chosen by warm performance, and the R5 numbers will describe a warm-tuned model evaluated cold. And the commitment itself is hedged as "(a) … and/or (b)" — an and/or is a menu, not a protocol. The concession in rebuttal 8 ("R5 without a measurement plan would be a citation, not a result") is thus only half-earned: R5 now has a measurement, but no cold-aware selection and a biased comparison rule.
>
> 6. **In the no-ID configuration, tuple-class ties make the 1+99 metric partially an artifact of sort order — and the one cheap check that would size this was adopted but never run.** C2(b) proves identical tuples ⇒ bit-identical scores. Fast-fashion catalogs concentrate items in a few head tuples (black/solid/jersey basics), so among 99 sampled negatives the probability that at least one exactly ties the positive is non-trivial precisely for head-class positives; the positive's rank inside a tie block is then decided by argsort stability, not by the model. No tie-handling rule is specified anywhere in §3.6. The distinct-tuple count over the 13,651 V1 items — the pre-check the author "adopted as amendment" in rebuttal 4 — costs one groupby and was still not executed before this candidate went to review, although its outcome could kill the no-ID arm outright (if, say, only ~6–8k distinct tuples exist, half the catalog is structurally unrankable within-class). Rebuttal 4 therefore rests on an unexecuted check, and its softener "any feature-only method has exactly this ceiling — the information-theoretic price of cold-start" is a false equivalence: for *cold* items the ceiling is inevitable; `use_id_feature=False` imposes it on all 13,651 *warm* items too, which is a dc01 configuration choice, not information theory.
>
> 7. **"R8 strong" is achieved by externalizing the mechanisms the design's own analysis says it needs — leaving the evaluated candidate a different object from the one that could work.** The 4c table lists four degeneration modes, of which three are "unprotected — open risk" and one only nominally protected (see point 1); for each, a fix (K1–K4) is fully designed in §3.8 and then excluded to preserve the one-idea rating, with adoption deferred to "a mechanism revision for the user" and a 4b re-run. Simultaneously, the §3.8 Rashomon tuning protocol promises to maximize "profile entropy ↓, feature-explained fraction ↑, profile distinctness ↑" within the accuracy-tolerance set — but with all K knobs unstacked, the tunable knobs are the inherited λ_sim_proto/λ_sim_batch/max_norm, none of which acts on profile sharpness (M1) or profile distinctness (M4) *by the document's own degeneration table*. The Rashomon procedure therefore has no lever connected to the quantities it claims to optimize. The realistic trajectory — dc01 trains, profiles come out diffuse/overlapping as points 1–2 predict, a K knob is adopted, the candidate is re-derived — means dc01 as frozen is the untested precursor of the actual thesis model, and its R8 "one clean idea" rating is an accounting artifact of where the candidate's boundary was drawn, not a property of what will be run.

### Response (concede / rebut / open risk, point by point)

**P1 — regularizer geometry under skew. Partial concession, revised at gate
review (user challenge, 2026-07-11): the separation effect is real but
indirect; the "expected outcome" upgrade was an overclaim.**
*(Original response conceded "no prototype–prototype repulsion anywhere" and
upgraded M1/M4 to "expected under skew"; the user challenged this against
the vault record. Re-examined against
`2026-06-15_1238_protomf-two-regularizers-geometric-effect.md` and the
reg-sweep evidence — the revision below supersedes the original wording.)*
- **Conceded (narrow):** there is no *direct* pairwise prototype term —
  Eq. 4 (`reg_proto`) in isolation lets two prototypes share one
  best-matching object at zero cost. The doc's parenthetical "(which
  `sim_proto` rewards)" is imprecise on both counts: distinctness is not a
  direct reward, and the effect runs chiefly through the *other* term.
- **Rebutted (systemic — the adversary missed the secondary effect):**
  separation IS produced indirectly by the coverage pair, chiefly
  `reg_batch`: every entity rewarded for having some prototype near it is a
  facility-location-style objective — a collapsed duplicate prototype wastes
  coverage, and relocating it to an uncovered region strictly improves the
  loss; jointly with `reg_proto` the dynamics are soft-k-means-like, stable
  at *distinct* cluster centers. Empirically demonstrated on the host: the
  reg-sweep t-SNE grids show prototype-centred clusters, and rising purity
  (ml-1m 0.70→0.87; H&M purity responds at x0.01–x0.1) is impossible under
  prototype collapse. Vault note 2026-06-15_1238 records exactly this
  geometry.
- **What survives for dc01:** the separating force is
  **data-spread-mediated** — only as strong as the entity cloud's spread. If
  dc01's composition concentrates item directions (the 58% shared-component
  concern = M2 interaction), the separation mechanism weakens
  *proportionally*. M1/M4 therefore remain **open risks with a named,
  measurable weakening mechanism** — not "expected outcomes."
- **Still conceded:** `max_norm` does not address *directional* dominance
  (4c "partially protected" overstates). **Still rebutted:** "Adagrad
  silently dropped" — optimizer choice `{adam, adagrad}` is in the inherited
  search space (`confs/hyper_params.py:21`).
**Adopted → F-DC01-05 (as revised):** doc corrections (parenthetical fixed to
the indirect-coverage mechanism with the vault/sweep citations, not deleted)
+ committed SC.8 geometry metrics (t* activation spread; prototype pairwise
latent cosine; profile pairwise overlap; per-value share distribution vs
value frequency).

**P2 — popularity expressiveness confound + stale §3.2. Concede both
sub-points with one correction; rebut the TPE-winnowing charge against the
implementation.**
*(Gate-review addendum, 2026-07-11, user directive — record the bias
distinction: LightFM's published tags-only variant still carried
feature-composed BIAS terms (b_i = Σ_f β_f — a class-level popularity
channel); our fleet is bias-free (S0.4 parity decision), so dc01's noid arm
has NO popularity channel at all — strictly more constrained than the
published LightFM(tags) precedent. Expectation-setting and §3.7 deviation-4
consequence noted in the amendment bundle.)*

*(Second gate-review addendum, 2026-07-11, user directive — record what the
ids-vs-noid gap ENTAILS. The measured gap is a bundle of three components:
(1) **popularity blindness** — noid cannot rank signature twins apart, so
within-class popularity signal is unrepresentable; real accuracy loss,
mundane cause, likely the largest chunk in fashion data; (2) **genuine
taste-structure loss** — the honest "cost of grounding": preference
distinctions the 5 fields cannot express; (3) **a partial offset** — shared
feature rows act as regularization and can help sparse items (LightFM
§2.3-1). The raw gap must never be quoted as "the cost of interpretability";
the D4 popularity strata separate (1) from (2) (deficit concentrated on
popular items → (1); flat across strata → (2)). This decomposition is
amended into §3.6 as the ablation's interpretation rule. Related
clarification, same directive: the noid arm remains **collaborative
filtering at attribute granularity**, NOT content-based filtering — its
feature embeddings are fitted purely by factorizing the interaction matrix
(LightFM §2.3: "does not reduce to a pure content model"); what noid loses
is item-LEVEL collaborative memory, not collaborativeness. The S0.4 rev. 1
record already draws this line — LightFM(tags) was preferred over LSI-LR
precisely because LSI-LR is content-only estimation while LightFM(tags) is
collaboratively trained.)*
Conceded: (a) the no-ID arm cannot represent within-signature-class
popularity variance (C2(b) is exactly this), so part of any noid-vs-ids
accuracy delta is popularity expressiveness, not the R4↔R6 explanation
trade — **the ablation delta must be interpreted with this confound named**,
and the popularity-stratified D4 readout is the instrument that partially
separates it (if the noid deficit concentrates on high-popularity positives,
that is the popularity channel, not the grounding cost). (b) The ID-arm
dynamics prediction (popularity signal routes into ID rows → feature-explained
fraction lowest on the most-recommended items) is plausible and cheap to
measure: **feature-explained fraction stratified by item popularity** joins
the SC.8 metric list. Rebutted-as-implemented: §3.2's "use_id_feature ∈
{True, False}" search dimension would indeed let warm-tuned TPE winnow the
False arm — but the implementation and charter C7 already run **two separate
fixed configs** (`feature_item_proto` / `_noid`, verified
`confs/hyper_params.py:252-253`), each with its own full search. The doc
sentence is stale, not the design — **folded into F-DC01-01's
doc-alignment bundle.**

**P3 — share non-identifiability across correlated fields. Concede — the
strongest new point. Adopted → F-DC01-06.**
Correct, and genuinely distinct from 5b-6 (label confounding): even with
perfectly meaningful directions, where values of different fields co-occur
(near-)deterministically, only the **sum** of their rows is pinned by the
objective; the split across e_department/e_product_type/e_section is
gauge-like and seed-dependent. The rendered example's "+0.45/+0.07/+0.01"
performs precision the training objective does not define. §4's
"per-value = per-field" line answers within-field roll-up, not cross-field
correlation — the critique stands. Mitigation adopted (doc amendment):
(a) rendering/reporting must offer **grouped shares over
strongly-associated field cliques** (the product-kind clique
department+product_type+section vs colour vs pattern — clique membership
from the 4.0 FD/NMI analysis), with the per-field split shown as
informative-but-not-identified detail; (b) thesis wording states the
identified quantity (clique sum) vs the conventional one (per-field split);
(c) cross-seed share-stability check deferred to full-profile stage (C4
single-seed budget) — recorded, not silently dropped.

**P4 — off-manifold counterfactual + the missed linear-half attribution.
Concede both; adopt the better read-out → F-DC01-07.**
The identity t̂ = W^t(Σ_f e_f) = Σ_f W^t e_f makes the u*·t̂ score half
**exactly, additively, norm-freely attributable per feature**:
contribution(f) = u*ᵀ(W^t e_f), and it equals the true removal effect on
that half — no ‖q_i‖ entanglement (C6's problem), no off-manifold pass.
This was genuinely missed: the design doc's read-out list (§3.4) covers only
the cosine half plus per-prototype contributions. Adopted: (a) add the
linear-half per-feature decomposition as read-out 4 — it also softens the
S0.2 Block-U opacity note for dc01 specifically (the projection branch
becomes feature-attributable, though still not prototype-shaped);
(b) demote the §3.4 counterfactual re-score to an optional *diagnostic*
with the off-manifold caveat stated (trained under 0% missingness → a
removed-field composition is out-of-distribution; heuristic meaning only).
Partial rebut of the R2-letter charge: the re-score uses only the model's
own forward function on intrinsic quantities — it is not `top_k_items`-class
data-side reconstruction; but the spirit-level point (a second constructed
pass, unvalidated input regime) is conceded and the demotion resolves it.

**P5 — cold-eval incoherence. Rebut — resolved by S0.3, which the blind
adversary could not see; one framing note adopted.**
(a) The primary cold ranking is **cold-vs-cold** (S0.3 §3, B5-faithful):
every candidate slot is feature-only, so the warm-ID magnitude bonus the
critique derives cannot bias the headline number. The cold-vs-warm
calibration confound exists exactly where S0.3 puts it — the *secondary*
cold-vs-all diagnostic, kept *because* it exposes calibration effects
(dc01's magnitude analysis here is a good sharpening of what that diagnostic
can show; noted for SC.8 interpretation). (b) Warm-only model selection is
a **deliberate, ratified anti-leakage choice** (S0.3 §4.7: "configs cannot
be selected *for* cold performance"); the R5 claim is scoped accordingly —
"how does the warm-selected model generalize cold," the deployment-shaped
question. (c) The "and/or menu" is superseded: S0.3 is a full committed
protocol (20% stratified held-out + TW/D4 strata — both branches of the old
menu, specified). **Doc consequence:** the §3.6 cold paragraph should point
to S0.3 as the governing spec — folded into F-DC01-01's alignment bundle.

**P6 — tie artifact + "unexecuted check." Rebut with S0.5's numbers (blind
spot of the isolation); adopt one disclosure line.**
The count ran at S0.5 (2026-07-11): **6,517 distinct 5-field signatures /
67.5% of items share / max class 97** — almost exactly the critique's
"~6–8k" kill-threshold guess, so the concern was right to raise and is now
quantified. Decisive for the *metric*: in the 1+99 protocol the expected
number of same-signature competitors per row is **~0.06** (cold pool;
full-catalog sampling the same order) — tie blocks are structurally rare in
the sampled eval, so argsort-order effects are bounded well below a
percentage point; the honest display of the full-catalog ceiling is the
tie-block diagnostic (built, model-agnostic, runs for every model).
Adopted: one disclosure line in the doc (ties resolved deterministically by
`bn.argpartition` order — a known convention, learnings ledger) rather than
a tie-handling mechanism. The "false equivalence" jab is accepted as
phrasing: the warm ceiling is a configuration property of the noid *arm*,
which is an ablation, not the headline config — wording amended accordingly.

**P7 — R8 as boundary accounting; Rashomon levers. Partial rebut with
host evidence; concede the contingency, which the protocol already owns.**
Rebutted: "no lever connected to the interpretability quantities" is
empirically false on the host — the reg-sensitivity sweep
(`Master/experiments/reg_sensitivity/ANALYSIS.md`) shows the inherited
regularizer weights move top-k categorical purity through a clear
inverted-U (ml-1m genre purity 0.70 → 0.87 → collapse); the inherited knobs
act on prototype–item geometry, which is upstream of both profiles and
sharpness. Indirect ≠ absent. Conceded: no inherited knob *directly*
targets profile sharpness (M1) or profile distinctness (M4); if SC.8
empirics show diffuse/overlapping profiles, adopting a K-knob is a **gated
mechanism revision with a 4b re-run** — that is the protocol's designed
escalation path, not an accounting trick; and the SC.10 verdict on dc01 is
dev-profile-provisional and evidence-scoped either way. The Rashomon sketch
wording is amended to name its actual levers (inherited weights, indirect)
and the K-contingency explicitly — folded into F-DC01-05's bundle.

### Fold-in summary (per protocol SC.1b.2)

- **New findings:** F-DC01-05 (regularizer-geometry corrections + SC.8
  geometry/stratified metrics), F-DC01-06 (cross-field share
  non-identifiability → grouped-share rendering + wording rule),
  F-DC01-07 (linear-half exact attribution added; counterfactual re-score
  demoted to diagnostic).
- **Extended:** F-DC01-01's alignment bundle gains the stale §3.2
  search-dimension sentence, the §3.6 pointer to S0.3 as governing cold
  spec, and the two disclosure lines from P6.
- **SC.8 metric commitments accumulated this step:** t* activation spread;
  prototype pairwise latent cosine; profile pairwise overlap; per-value
  share vs value frequency; feature-explained fraction stratified by item
  popularity; noid-vs-ids delta read jointly with the D4 popularity strata
  (P2 confound).
- **Registered hypothesis (gate walkthrough, the ID/twins/popularity braid):**
  the warm feature-explained fraction should inversely predict the `_ids`
  config's cold-vs-cold → cold-vs-all degradation — the ID row carries
  popularity + twin-separation, both lost at the drop, so how much a model
  leaned on IDs (the honesty meter) and how much it loses against warm
  incumbents should be two views of one number. Related interpretation rule:
  the cold-vs-all gap decomposes into a REAL effect (new items genuinely
  lack popularity standing) plus artifact — read via the popularity-decile
  cold slices; if the popularity component dominates, the bias-gap note's
  feature-composed bias becomes the evidence-triggered ablation
  (S0.4 gate anticipation).

### Gate (SC.1b)

**Status: CLOSED — ratified by user across an extended point-by-point
walkthrough (2026-07-11 → 2026-07-12), one point per sitting, deep-dive
mode.** Itemized outcomes:

- **P1:** response REVISED at gate on user challenge — the "no repulsion"
  concession overclaimed; separation is an indirect, data-spread-mediated
  secondary effect of the coverage pair (vault + reg-sweep evidence).
  F-DC01-05 approved in the revised form, itemized 1.1–1.4 with user
  extensions (omnipresence residue = definite risk, MUST be tested;
  metric set non-exhaustive/research question; broader knob inventory;
  further regularizer-lever analysis; "merged prime prime"
  interpretability-regularizer candidate parked at the late-thesis
  Rashomon stage). Learnings entry added (check empirical record before
  conceding mechanism-absence claims).
- **P2:** conceded with recorded additions — LightFM bias distinction
  (noid arm has NO popularity channel), the three-component gap
  decomposition + interpretation rule, noid = attribute-level CF not CBF
  — all folded into F-DC01-01's bundle.
- **P3:** F-DC01-06 approved in final form: characterize → exhibit →
  remedy → delimit, threshold-free research side ("production decides;
  research characterizes" — learnings entry), determinacy = loss curvature
  along redistribution directions with decoupling mass as proxy, grouped
  shares + naming discipline, delegated production math problem with the
  human-performability condition. Plain-language source of truth: vault
  `2026-07-11_2310`.
- **P4:** F-DC01-07 approved, (a)+(b) both — linear-half read-out added,
  off-manifold re-score demoted; host-specificity insight recorded
  (I-ProtoMF all-cosine / U-ProtoMF all-linear / UI both — a quiet
  double-tie argument).
- **P5:** rebutted via S0.3 (cold-vs-cold primary neutralizes the
  handicap; warm-only selection is ratified anti-leakage; the "menu" is
  superseded); the magnitude-deficit mechanism adopted as a named SC.8
  interpretation tool. The ID/twins/popularity **braid** analysis and the
  registered hypothesis (feature-explained fraction inversely predicts
  cold-vs-all degradation) captured in vault `2026-07-12_0006`.
- **P6:** rebutted with S0.5's numbers (6,517 signatures; ~0.06 tie
  competitors/row — metric impact bounded); disclosure adoptions in
  F-DC01-01; **user directive: testing warranted** (measured tie incidence
  at SC.8, tie-block diagnostic every row, twin spot-checks on real
  checkpoints) — recorded in the vault braid entry.
- **P7:** closed by the user's lineage framing — the C1 lineage makes the
  foundational ideas explicit and staged; nothing is mangled into one
  candidate (vault `2026-07-11_2351`).
- **F-DC01-01 extensions:** all six approved after plain-language
  restatement.

**Session-level yields recorded alongside:** the verbatim-capture rule
(protocol skill amended — conciseness subordinate to fidelity; three vault
entries rewritten under it); the thesis-outline C1 lineage (committed
direction, vault `2026-07-11_2351` + memory); the taste-is-revealed
argument (vault `2026-07-11_2350`); user-side grounding routes 1/2
captured to the design scratchpad; two learnings-ledger entries
(fluent-synthesis quantifier slippage; thresholds are production logic).

→ Next step: **SC.2** — apply the full amendment set (F-DC01-01..07) to
the design doc as dated visible amendments, decoding from the vault
entries per the verbatim-capture rule; then the one-round black-box toy
re-check of every mathematical claim the amendments touch (linear-half
exactness + counterfactual property; the lockstep frozen-difference
exhibit).

---

## SC.2 Concept fixes + re-verification (2026-07-12)

### 1. Amendments applied (dated, visible; decoded from the vault entries)

All seven findings applied to `dc01_feature_composed_factors.md`:

| Finding | Sections edited |
|---|---|
| F-DC01-01 | §3.2 (searched-flag → two fixed configs); §3.5 (canonical 5, V=426, price_band deferral); §3.6 (S0.3 governing-spec pointer; tuple count EXECUTED with S0.5 numbers + tie disclosures; lightfm-baseline exists note; NEW ablation-gap interpretation rule + CF-not-CBF classification); §3.7 dev. 4 (no-popularity-channel consequence); §5 S3 row (stale image caveat struck) |
| F-DC01-02 | §3.4 amendment block: Steck disposition (trains-through-cosine steel-man; surviving risks: Rashomon multiplicity, label confounding, unprotected projection half; SC.8 profile-validity spot-check committed) |
| F-DC01-03 | §3.4 amendment block: second ID-keyed channel (bias scalar), S0 dispositions, feature-composed bias as evidence-triggered ablation; temp note `dc01_feature_composed_bias_gap.md` retired with pointer |
| F-DC01-04 | §3.4 amendment block: train/cold operating-point mismatch, LightFM tags+ids precedent, K3 as unstacked mitigation, SC.8 caveat + registered braid hypothesis |
| F-DC01-05 | §3.8 M4 parenthetical corrected (indirect coverage mechanism); §5 identifiability-tension precision note; 4c table note (vote-weights / loudness-vs-omnipresence; omnipresence = definite risk MUST be tested; M1/M4 spread-mediation + committed instruments); §3.8 Rashomon amendment (real levers, non-exhaustive metrics, further regularizer analysis, merged-prime-prime parking) |
| F-DC01-06 | §3.4 amendment block: full identifiability treatment (frozen-difference mechanism; grouped shares + naming discipline; threshold-free determinacy = curvature via decoupling mass; delegated math problem with human-performability; toy exhibit; seed-stability deferred) |
| F-DC01-07 | §3.4: read-out 4 (exact linear-half attribution + host-specificity I/U/UI note); counterfactual re-score demoted inline + in block |

Status updates: candidate index row → **concept amended (scrutiny SC.2)**,
2026-07-12; design doc header + closing status line updated; claims spec
extended with C7a/C7b/C8 (dated SC.2 section).

### 2. Black-box toy re-check (Ground rule 10 — subagent saw ONLY the claims spec)

Script: `Master/temp/dc_checks/dc01/check_claims_sc2.py` (~30 s, torch
float64, protomf env). **Verbatim report:**

> **C7a — CONFIRMED.** `ustar·(W_t q_i) = Σ_r ustar·(W_t E[r])` holds exactly across an adversarial sweep of 450 trials: 10 seeds × F ∈ {1, 2, 3, 5, 20} × parameter scales {1e−3, 1, 1e3} × dims (d, K_u, K_t) ∈ {(4,1,1), (8,3,4), (16,5,7)}. Max abs error 1.46e−11 (occurs at scale 1e3), max relative error 2.65e−14 — float-tolerance exact.
>
> **C7b — CONFIRMED (both halves).** Linear half: removing row r0 (r0 varied across trials) changes `ustar·that` by exactly `−ustar·(W_t E[r0])`; max abs err 1.46e−11, max rel err 2.43e−13 over the same 450 trials. Cosine half: the remaining rows' shares c_{r,k} do change under removal, and precisely as the spec says — they rescale by the common factor ‖q‖/‖q′‖ (max deviation from exact rescale 2.22e−16), with that factor ≠ 1 in every trial (min |factor − 1| = 1.57e−4, typically far larger). So linear per-row terms are removal-exact; cosine shares are not.
>
> **C8 — setup:** fixed toy dataset (N=10 users, M=30 items, V=12 feature values, F=3 + ID, d=8, K_u=3, K_t=4); values a, b co-occur on exactly the 15 even-indexed items, never appear otherwise; full-batch BCE-with-logits on fixed labels; plain `torch.optim.SGD`.
>
> **C8(a) — CONFIRMED.** Over 300 steps × 5 seeds (no weight decay): `grad E[a] − grad E[b]` is bitwise zero at every step (max inf-norm gap 0.000e+00, all seeds); `E[a] − E[b]` stays at its initial value up to update-arithmetic rounding (max drift ≤ 1.33e−15); meanwhile `E[a] + E[b]` moves substantially (‖Δ‖ 0.33–0.64 per seed).
>
> **C8(b) — CONFIRMED.** With SGD weight_decay=0.1 (lr=0.05, 300 steps): ‖E[a]−E[b]‖ decays 1.8437 → 0.4098; the decay ratio matches the theoretical (1−lr·wd)^300 = 2.222922e−01 to relative deviation 1.6e−15 (the loss-gradient parts cancel exactly; only decay acts on the difference).
>
> **C8(c) — CONFIRMED under the metric the claim specifies (relative dispersion), with one caveat.** 8 seeds, same data, same training [...] The split varies wildly across seeds: the intra-pair a-fraction c_a/(c_a+c_b) (k-summed) spans 0.06 to 1.50 across seeds; relative dispersion (std/|mean|) of the split c_a−c_b: **9.0** (literal), **54.5 / 96.5** (learnable, items 0/4). The sum's relative dispersion is far smaller: **1.36** (literal), **1.12 / 1.16** (learnable) — a factor 6.6×–83× below the split's [...] Consistency check: the k-summed a-fraction is identical for both eval items (as algebra predicts) — confirming the split is a property of the frozen E[a]−E[b] gauge, not of the item. **Caveat (adversarial finding):** in ABSOLUTE std the sum is not more stable than the split [...] at toy scale the trained solution itself is not seed-stable (many optima, plus prototype-index permutation across seeds), so the sum is not run-invariant. The split's huge relative dispersion is partly driven by its near-zero mean — which is itself the expected signature of a pure gauge quantity.

*(Full report incl. ambiguity resolutions preserved in the transcript;
the three resolutions — prototype-permutation-invariant k-summed reading
for (c), both literal and best-case training configs, float-exactness
reading for (a) — are all judged faithful.)*

### 3. Response to the C8(c) caveat

**Accepted; it sharpens the exhibit's thesis wording rather than
weakening the claim.** What C8 establishes — and (a) establishes it
*exactly, bitwise* — is that the lockstep pair's difference is **pure
gauge**: identical gradients every step, difference frozen at
initialization (or decayed by L2), and the rendered split's
item-independence confirms it reads the frozen gauge, not the data. The
"sum is trained" clause holds in the relative-dispersion sense the spec
named, but at toy scale the trained optimum is itself seed-unstable, so
absolute stability of the sum is NOT demonstrated (and should not be
claimed). Thesis exhibit therefore leads with (a)/(b) — "the split
literally never receives a training signal" — and uses (c) only for the
split-wanders-while-fraction-is-item-independent visual. Wording note
added to the F-DC01-06 amendment's exhibit sentence via this dossier
(design doc already phrases the exhibit neutrally).

### 4. Ledger updates

F-DC01-01..07 → **fixed** (SC.2 amendments applied 2026-07-12; resolving
commit **8055422**; vault entries b169397; skill rule 08dd0fa). F-S0-07
part (b) remains open → SC.3.

### Gate (SC.2)

**Status: CLOSED — ratified by user 2026-07-12 ("yes to all").**
(1) Amendment set accepted as faithful; (2) re-check verdicts + C8(c)
disposition accepted (exhibit leads with the exact gradient-identity
demonstration); (3) proceed to **SC.3 in a fresh session** — headline
item: F-S0-07(b), the actual ID-drop for dc01's nested cold branch
(unblocks `_ids` cold runs; runner refusal then lifts), plus dc_checks
extensions where behavior changes. Commits: 8055422 / b169397 / 08dd0fa
(+ this gate-closure edit).

---

## SC.3 Implementation alignment (2026-07-12)

**Scope (per the SC.2 gate handoff):** patch the implementation to the amended
concept. Headline: **F-S0-07(b)** — the cold-inference ID-column drop for
dc01's nested item branch. Plus the one other code-touching commitment from
the amendment set: **F-DC01-06's counting script**. Protocol section re-read;
manifest loaded (findings ledger, learnings, requirements doc, candidate
index, amended design doc, implementation plan, modification map advisory).

### 1. Audit: implementation vs the amended concept

Walked the amended design doc against the code on `feat/scrutiny`:

- **Configs (F-DC01-01 alignment):** `confs/hyper_params.py` carries the two
  fixed configs (`feature_item_proto`, `feature_item_proto_noid`) plus the
  `_f0` keystone ablation; item side uses the canonical 5 fields (V=426);
  fleet `use_bias=0` via `base_hyper_params`. **Code already canonical — no
  change needed** (as SC.1a recorded; re-verified).
- **Plumbing:** `inject_feature_ids` handles `feature_item_proto` at both
  build seams (trainer/tester `_build_model`) AND at `cold_eval.build_model`;
  tensor-never-in-config discipline intact.
- **Factory/tie:** item branch = `Concatenate(PrototypeEmbedding(embedding_ext
  = FeatureEmbedding), FeatureEmbeddingW(shared = SAME instance), invert)` —
  matches §3.2; single-owner init in place.
- **No other amended-concept item requires code at SC.3:** read-out 4 +
  grouped-share rendering are SC.5 (explanations pipeline); SC.8 geometry
  metrics are SC.6/SC.8; the all-zero-composition eps look is SC.4 (as
  flagged at SC.1a). **No new findings.**

### 2. F-S0-07(b) — nested ID-column drop: VERIFIED, guard lifted (865de3b)

**Key audit result: no new machinery was needed.** `cold_eval.py`'s generic
`_find_feature_embedding` already recurses `Concatenate → PrototypeEmbedding
.embedding_ext` to dc01's nested `FeatureEmbedding`, and the R1 tie itself
closes the case: because ONE shared instance feeds both item halves, zeroing
the cold ID rows once covers BOTH halves of the UI-score (q_cold = Σ_f e_f
everywhere). What was missing was *verification* — exactly what F-S0-07(b)'s
disposition assigned here.

- **New test `dc_checks/dc01/t11_cold_id_drop.py` (29 checks, ALL PASS):**
  (A) structural — drop reaches the nested shared instance; cold composed
  embedding == pure feature composition; BOTH halves change for cold items,
  projection half == linear(pure composition); warm outputs bit-identical;
  drop == manual zeroing (full state-dict equality); `_noid` no-op.
  (B) guard semantics — dc01+ids ACCEPTED by `load_config` (incl. absent-key
  = factory-default True), `use_bias≠0` still refused; `config_expects_id_drop`
  truth table (dc01/lightfm/CF × ids/noid).
  (C) **full-runner e2e on a REAL trained dc01 toy checkpoint** — report
  records the drop; attr-kNN skips cleanly (CF-only seam raises); metrics
  finite; cold-vs-cold HR@10 0.149 vs ~0.098 chance floor on the toy.
  (D) restore/re-apply — checkpoint reload revives ID rows (drop is
  eval-time-only); re-drop restores the zeroed state.
- **Guard change (the fix):** the load-time refusal of
  `feature_item_proto`+`use_id_feature` is **lifted**, replaced by
  `config_expects_id_drop(conf)` + a **post-drop abort** in `run_cold_eval`:
  any tags+ids config whose built model the drop cannot reach aborts loudly.
  Rationale: refusal-by-config-name protected only dc01; the abort protects
  the *invariant* (no cold item is ever scored on an untrained ID row) against
  future structure drift, for every current and future tags+ids branch.
- **s0/t06 updated** to pin the new semantics (refusal→acceptance flips +
  expectation predicate); **s0/i02 e2e re-run green** (runner change is a
  no-op for CF rows). Full dc01 suite (t01–t10, i01–i06a) re-run green at
  baseline before the change and untouched by it (no model-code edits).
- **Consequence:** dc01 `_ids` cold runs are unblocked for SC.6. F-S0-07 →
  **fixed(865de3b)** in the ledger.

### 3. F-DC01-06 — decoupling-mass counting script (9f11577)

`dc_checks/dc01/decoupling_mass.py` — the committed determinacy proxy
(§3.4 discipline; vault 2026-07-11_2310): per cross-field value pair, the
catalog mass where the pair decouples (= the only items generating curvature
along the redistribution direction). Value level (mechanism) + item-weighted
field level (presentation); threshold-free continuum; synthetic selftest
(lockstep→0, independent→high) passes. **V1 canonical-5 numbers:**

- 8,745 observed cross-field value pairs; only **2 perfect-lockstep pairs**,
  covering 3 items — pure gauge is essentially absent on V1.
- Weakest field pair: **department×section, 0.705** item-weighted mean
  normalized decoupling (next: department×product_type 0.815) — so even the
  closest association is far from lockstep at field level, and section-vs-
  department is empirically the closest pair (the "clique membership is an
  output, not an assumption" stance confirmed).
- Near-lockstep pairs with real mass exist exactly where the grouped-shares
  discipline aims: Swimwear↔"Womens Swimwear, beachwear" (0.029),
  Jersey Basic↔"Womens Everyday Basics" (0.032), Denim trousers↔Denim Men
  (0.068) — these splits are the ones the renderer must annotate/group.
- Field-level CSV committed (4K); value-level CSV (788K) untracked,
  regenerated deterministically by the script (script+data on both machines).

### 4. Ledger / doc updates

- F-S0-07 → **fixed(865de3b)** (part (b) closed by verification; resolution
  note in the ledger). F-DC01-06 entry annotated with the script commit +
  headline numbers.
- Design doc §3.4(b) + §3.7 dev. 6: "spec-only until SC.3" status notes
  updated to LANDED (dated).
- `modifications_log.md`: no entry — `cold_eval.py` and the dc_checks files
  are not upstream-repo files (GR11 boundary).

### 5. Test-run record

| suite | result |
|---|---|
| dc01 t01–t06, t08–t10, i01–i04, i06a (baseline, pre-change) | ALL PASS |
| dc01 **t11** (new, 29 checks) | ALL PASS |
| s0 t06 (updated semantics) | ALL PASS |
| s0 i02 e2e (regression after runner change) | ALL PASS |
| decoupling_mass --selftest | PASS |

Commits: **865de3b** (F-S0-07(b) fix + t11 + t06), **9f11577** (decoupling
script + field-level CSV), plus this dossier/ledger edit commit.

### Gate (SC.3)

**Status: OPEN — awaiting user review.** Both work items executed under
their previously ratified dispositions (F-S0-07(b): S0.7 gate; F-DC01-06
script: SC.1b gate). Decisions requested: (1) accept the verification-based
closure of F-S0-07(b) incl. the guard-lift semantics (refusal → post-drop
abort); (2) accept the decoupling-mass script + its V1 headline numbers as
the F-DC01-06 measurement basis; (3) confirm no further SC.3 scope (audit
found code already aligned with the amended concept) → proceed to SC.4
(implementation scrutiny, black-box spec-vs-code) next.
