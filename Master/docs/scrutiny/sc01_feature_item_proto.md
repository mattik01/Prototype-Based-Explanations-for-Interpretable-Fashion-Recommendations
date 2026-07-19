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

---

## ⟳ HOST REDIRECTION (2026-07-12 — user decision at the SC.3 gate)

**Before the SC.3 gate was reviewed, the user redirected the candidate: dc01 is
re-hosted from UI-ProtoMF to I-ProtoMF → "fI-ProtoMF".** The thesis lineage now
mirrors ProtoMF's own U/I/UI separation: fI vs I-ProtoMF, then fU (history-grounded
user side) vs U-ProtoMF, then afU (+user attributes; working name), finally the fUfI merge vs
UI-ProtoMF; route-1 (user prototypes pushed through the tie into item-prototype
space) is presented, not implemented. Full statement: design doc redirection banner
+ masterplan Current Position banner (both updated this session). Later stages are
NEW candidates with their own DC + scrutiny cycles when their time comes.

### Effect on this dossier's closed steps (transfer audit, first-pass)

**Carries unchanged (composition-level, host-agnostic):** F-DC01-01 (canonical 5;
configs), F-DC01-02 substance (trains-through-cosine steel-man, Rashomon, label
confounding — the "unprotected projection half" risk DISAPPEARS in fI), F-DC01-03
(bias channel), F-DC01-04 (train/cold operating-point mismatch), F-DC01-05
(regularizer coverage geometry + committed SC.8 instruments), F-DC01-06 in full
(identifiability discipline + decoupling script + V1 numbers), claims C1/C2/C6/C8,
the SC.2 toy exhibits, and ALL of S0 (the instrument and the queued reference fleet
— the ProtoMF five already contain `item_proto`, `user_proto`, `user_item_proto`:
every host baseline the restructured lineage needs; fleet still fully PD as of this
session, nothing wasted, nothing new to submit).

**Parked, not lost (returns at fU/fUfI stages):** F-DC01-07's linear-half read-out
(requires a projection branch; the SC.1b host-specificity note predicted this),
C7a/C7b (verified math, later-stage property), the instance-sharing tie (§3.7 dev. 2),
the double-tie "both surfaces at once" argument.

**SC.3 work of this session:** survives. The drop primitive + post-drop abort are
host-agnostic (`_find_feature_embedding` nests through PrototypeEmbedding with or
without the Concatenate wrapper; `config_expects_id_drop` keys on the retained
ft_type). t11 parts A/B/D are structure-level and carry; part C (e2e) gets an
fI-variant at the SC.3 re-run. Decoupling script untouched.

### Re-run plan (delta-scoped, gated)

1. **Re-host amendment cycle** (≈ SC.1a-delta + SC.2-delta, one–two sittings):
   dated re-derivation of design doc §3.2/§3.3/§3.4(readouts)/§3.6/§3.7/§4 for the
   I host; F-DC01-01..07 transfer table ratified; requirements re-rating (R2
   expected to STRENGTHEN — the whole score is grounded; R1 tie-language revised;
   S1 row updated); degeneration-mode re-check (M1–M4 transfer; user side is host
   behavior); restated claims C3′ (single-path gradient), C4′ (score = Σ_k u_k·t*_k),
   C5′ (F=0+ID ≡ `item_proto`) through ONE black-box toy re-check.
2. **SC.1b-delta:** short adversarial pass on the re-hosted doc (GR10) — cheap
   insurance; the prior adversary's UI-specific points (P4 linear half, P5 warm
   magnitude bonus via the t̂ half) dissolve structurally in fI (t* is
   scale-invariant in q_i), and that dissolution itself deserves hostile review.
3. **SC.3 re-run:** factory branch re-pointed to the I host (item branch verbatim;
   user branch = I-ProtoMF free embedding), configs (`feature_item_proto`,
   `_noid`, `_f0` retained), keystone i02′ vs `item_proto`, t11 part-C fI variant,
   dc_checks port (t04/t05/i01–i04 small edits). The UI-hosted factory branch is
   removed from the live config surface (single lineage); its code remains in git
   history + FeatureEmbeddingW stays for the merge stage.
4. **SC.4–SC.6:** never run — no rework; they run against fI directly.

### Gate status

The SC.3 gate as posed above is **superseded in part**: decisions (1) drop-closure
semantics and (2) decoupling-mass basis still stand for review (both artifacts are
host-agnostic and carry into fI); decision (3) "proceed to SC.4" is replaced by
"proceed to the re-host amendment cycle". Combined review happens when scrutiny
resumes.

### Combined SC.3 gate — CLOSED (2026-07-12, resume sitting)

Deep-dive walkthrough (plain-language build-up per gate convention), decisions:

1. **F-S0-07(b) closure — ACCEPTED as proposed.** Verification-based closure
   ratified (t11 as the proof; one shared FeatureEmbedding covers both halves),
   **including the permanent guard semantics**: load-time refusal →
   `config_expects_id_drop` + post-drop abort (invariant-protecting, covers all
   current/future tags+ids branches; absent-key=True hazard handled).
2. **F-DC01-06 measurement basis — ACCEPTED with a scope restriction (user
   directive):** script + V1 numbers are the **scrutiny-phase working basis
   only** — SC.5 renderer grouping/annotation and protocol reasoning may use
   them; **the thesis may NOT quote them yet.** When the thesis reaches the
   identifiability discussion, dedicated experiments + mathematical
   argumentation will be designed on top of this work; until then this is a
   tracked open issue (ledger annotation on F-DC01-06; learnings entry added —
   scrutiny evidence ≠ thesis evidence).
3. **Next step:** re-host amendment cycle (SC.1a-delta + SC.2-delta) confirmed
   as the path, **not started this sitting** (user: not yet). SC.4–SC.6 run
   against fI directly, per the redirection plan.

### SC.5 pre-directive (user, 2026-07-12 — recorded at the gate-resume sitting)

Before SC.5 runs, the user clarified *why the explanation system exists*: it is
the thesis's **standing comparison instrument**. Every host-vs-variant
comparison (here fI-ProtoMF vs I-ProtoMF; later fU|U and fUfI|UI) runs the same
two-question check for both models: (1) what do the prototypes look like —
post-hoc on the host vs intrinsic on the variant; (2) what does a concrete
recommendation look like, broken down through those prototypes (shared
renderer, same user/item, side by side). Answered in part graphically, across
the in-scope datasets (user: "both, even all three" — the exact per-variant
dataset list is pinned at SC.5/SC.6, not assumed here). The graphics must be
refined for **repeated use**: maximally insightful, compact enough to recur at
every stage × dataset. Protocol doc updated to v1.2 (SC.5 preamble +
changelog); dc01's SC.5 builds to this standard.

---

## ⟳ Re-host amendment cycle — SC.1a-delta (2026-07-12)

**Scope (redirection plan step 1):** dated re-derivation of the design doc's
host-dependent sections (§3.2/§3.3/§3.4 read-outs/§3.6/§3.7/§4) for the
I-ProtoMF host; F-DC01-01..07 transfer table for ratification; requirements
re-rating; degeneration-mode re-check; restated claims C3′/C4′/C5′ for the one
black-box toy re-check (run at SC.2-delta, after this gate). Protocol v1.2
re-read; scrutinized under v1.2 from here on (v1.0 governed SC.1a–SC.3, per
the dossier header note — asymmetry declared).

### 0. Preamble

- **Manifest reloaded:** protocol (full), findings ledger, learnings ledger,
  requirements doc, candidate index, amended design doc (incl. redirection
  banner), dc01 claims spec + SC.2 check results, factory + config code on
  `feat/scrutiny` (branch verified).
- **Ledger re-check:** no `open` findings anywhere (F-S0-* all
  fixed/wontfix; F-DC01-01..07 all fixed). Nothing stale found: every
  disposition re-read against current code still matches (F-S0-07's new guard
  semantics verified present in `utilities/cold_eval.py`; configs carry
  canonical 5; `use_bias=0` fleet-wide).
- **Learnings watch-outs applying here:** *docs drift from code* — the
  re-derivation below re-verified every code-referencing claim against the
  live factory/configs; *absent-key defaults are hazards* — the fI factory
  branch must carry explicit `use_id_feature` handling (SC.3 re-run note 3
  below); *fluent cross-level synthesis hides quantifier slippage* — the R2
  strengthening claim below is stated with its exact scope (grounding covers
  prototype meaning + item-side attribution; user coefficients stay
  CF-learned).

### 1. F-DC01-01..07 transfer table (proposed for ratification)

| Finding | Transfer verdict | Reasoning |
|---|---|---|
| F-DC01-01 (canonical 5; two fixed configs; tie/twin disclosures; gap-interpretation rule) | **carries unchanged** | All composition-level or eval-protocol-level. Config re-pointing (user branch → plain embedding) is SC.3-re-run work, not a concept change. |
| F-DC01-02 (Steck disposition) | **carries, strengthened** | The trains-through-cosine steel-man now covers the ENTIRE score (I-score = u·t*, cosine-valued in the loss); the "unprotected projection half" residual risk disappears structurally — there is no dot-product half. Rashomon multiplicity + label confounding carry unchanged; SC.8 profile-validity spot-check unchanged. |
| F-DC01-03 (ID-keyed bias channel) | **carries unchanged** | `RecSys.use_bias` is host-agnostic; fleet `use_bias=0` + runner refusal unchanged. |
| F-DC01-04 (train-with-ID / cold-without-ID mismatch) | **carries unchanged** | Property of the composition + drop convention, not of the host. LightFM precedent, K3 mitigation pointer, braid hypothesis all unchanged. |
| F-DC01-05 (indirect coverage separation; omnipresence residue MUST be tested) | **carries unchanged** | The item-side inclusion regularizers (paper Eq. 9) are identical in I-ProtoMF; the composed-cloud spread mediation is item-branch geometry. All committed SC.8 instruments unchanged. |
| F-DC01-06 (share identifiability; grouped shares; decoupling proxy) | **carries unchanged, in full** | Pure composition-level (q_i = Σ_f e_f linearity). Script + V1 numbers + the scrutiny-only quotability restriction unchanged. |
| F-DC01-07 (linear-half read-out) | **PARKED** (per ledger scope note) | Requires a projection branch; no object in fI. Returns at fU/fUfI. C7a/C7b remain verified math. Status stays `fixed` — the amendment was faithfully applied; the redirection re-scopes. |

### 2. Re-derivation for the I host (proposed amendment content)

**The mechanism, restated (fI-ProtoMF).** Host I-ProtoMF (paper §3.2): item
prototypes P^t ∈ R^{K_t×d}, item activation t* = [1 + cos(q_i, p_k)]_k,
user = free vector u ∈ R^{K_t} living in item-prototype-similarity space,
score I(u,i) = u·t* (Eq. 8), loss Eq. 10 with item-side inclusion
regularizers Eq. 9. dc01's one idea is unchanged: q_i = Σ_{f∈f_i} e_f (+ID
row) replaces the free item embedding feeding the prototype layer.

**§3.2 modules (re-derived):** item branch verbatim —
`PrototypeEmbedding(embedding_ext=FeatureEmbedding)`. Drops: FeatureEmbeddingW,
both Concatenate wrappers, user prototype branch, both W matrices. User side =
I-ProtoMF's free `Embedding(n_users, K_t)` (factory 'prototypes'/Item-Proto
shape, `feature_extractor_factories.py:53-63`). `ft_type='feature_item_proto'`
retained.

**§3.3 shapes (re-derived):**

| step | tensor | shape |
|---|---|---|
| user embedding | `u = U[u_idxs]` | (B, K_t) |
| item feature lookup | `feat_ids[i_idxs]` (+ID col) | (B, 1+n_neg, F+1) |
| item composed emb | `q_i = Σ_f e_f` | (B, 1+n_neg, d) |
| item proto branch | `t* = sim(q_i, P^t)` | (B, 1+n_neg, K_t) |
| score (RecSys dot) | `u·t*` | (B, 1+n_neg) |

Parameters: feature table (V+M)×d (with ID), prototypes K_t×d, user table
**N×K_t** (note: dimension K_t, not d — host property). No W matrices.

**§3.4 read-outs (re-derived).** Read-outs 1–3 survive with simplification:

1. *Per-prototype contribution:* s_k = u_k·t*_k — now covering the **entire**
   score (in UI it was one half). Item-discriminating form: u_k·(t*_k − 1),
   because of the rank-inertness note below.
2. *Per-feature shares of each activation:* c_{f,k} unchanged (C1 carries
   verbatim).
3. *Global prototype profile* cos(e_f, p_k): unchanged. (4. parked.)

**New host-specific sharpening (proposed, goes through the SC.2-delta
re-check as part of C4′):** the +1 shift aggregate is **rank-inert**. S(u,i)
= Σ_k u_k(1 + cos_ik) = Σ_k u_k + Σ_k u_k·cos_ik, and the first term is
item-independent — identical for every item the user ranks. Hence **100% of
the item-discriminating score decomposes exactly over (prototype, feature)
pairs**: S(u,i) − baseline(u) = Σ_k Σ_{f∈f_i} u_k·c_{f,k}. Scope honesty:
rank-inert means inert for ranking/eval (and it cancels in softmax/BPR-style
losses; under BCE it affects calibration, not order). The rendered breakdown
still shows the +1·u_k baseline lines (magnitude honesty), but the thesis's
attribution claim can now be stated over the full ranking signal —
strictly stronger than the UI formulation, where only the û·t* half
decomposed this way. The rendered example re-derives accordingly (single
score line, affinity u_k in place of û_k; no grounded-vs-ungrounded line
split — there are no user-prototype lines; the user side appears only as
the per-prototype affinity coefficients, disclosed as CF-learned).

**S0.2 score-share obligation, fI form:** Block-U's share is structurally
zero (no user-prototype surface in the candidate). The binding disclosure
becomes: **feature-explained fraction of the item-discriminating score**
(metadata rows vs the ID row's share of Σ_k u_k·c_{f,k}) — same M3
instrument, now covering the whole score.

**§3.6 (re-derived):** headline baseline = **`item_proto`** (I-ProtoMF),
already in the S0.7 reference fleet — nothing new to submit. F=0+ID keystone
reduction target = `item_proto` (C5′). LightFM rows, ablation arms
(`_noid`, `_f0`), cold protocol (S0.3), gap-interpretation rule: all
unchanged.

**§3.7 deviations (re-derived):** dev. 1 (composition under cosine prototype
layer), 3 (per-feature decomposition as read-out), 4 (no bias terms + the
no-popularity-channel consequence), 5 (field selection), 6 (cold ID drop,
landed) carry with host renaming. Dev. 2 (instance-sharing tie) **parks** —
fI has a single consumer of the FeatureEmbedding; nothing to tie. R1
language re-derived in §3 below.

**§4 accounting (re-derived):** with ID: feature table (426+13,651)·d +
prototypes K_t·d + user N·K_t. Delta vs `item_proto` at equal (d, K_t):
**+426·d ≈ +27K at d=64** (~3% of the item table) — negligible. Without ID:
item table shrinks 13,651→426 rows (~32×). Step cost: the composed lookup
replaces one gather with F+1=6 gathers + sum; the K_t·d cosine still
dominates → ≈1.05–1.1× vs `item_proto`. S5 checklist unchanged (indexed
gathers, in-batch losses, no precompute).

### 3. Requirements re-rating (delta vs the amended doc's table)

| Req | Was (UI) | Now (fI) | Justification |
|---|---|---|---|
| R1 tied | strong | **strong (basis re-derived)** | The "double-tie mirror" phrasing was host-stage-specific. The requirement's substance — features wired into the representation that drives the score, no parallel reranker — is satisfied maximally: features are the item representation, and the score has **no other path** (single score term through the grounded branch). Instance-sharing tie returns at the merge stage. |
| R2 intrinsic | strong (Block-I caveat) | **strengthened** | The Block-I/Block-U split dissolves: every scored quantity flows through the grounded prototype surface; the whole item-discriminating score decomposes exactly per (prototype, feature). Scope kept honest: user coefficients u_k stay CF-learned (they are the personalization, per-prototype legible), and profile *meaning* validity still rests on F-DC01-02's checks. |
| R3 prototype | strong | strong | Unchanged; explanation stays "which prototypes and what they mean". |
| R4 grounded | partial | **partial (unchanged)** | M1 (profile sharpness unenforced) is host-independent; instruments unchanged. |
| R5 sparsity | strong | strong | Cold machinery + caveats (F-DC01-03/04) carry; q_i from features alone; the whole cold score is feature-computed. |
| R6 dense accuracy | partial | **partial (comparison re-anchored)** | The empirical bet is now fI vs **I-ProtoMF** — the host baseline changes, the bet's nature doesn't. |
| R7 vocabulary | partial | partial | Unchanged. |
| R8 parsimony | strong | **strong (cleaner)** | Same one idea on a strictly simpler host; fewer moving parts (no W, no user prototypes). |
| S1 decoupled | strong | strong | `lightfm_tags`/`_ids` rows unchanged. |
| S2 user-side | strong | **strong (re-scoped)** | User-side grounding is now an explicit lineage stage (fU/afU), not a dc01 extension point. Non-preclusion holds by construction of the lineage. |
| S3 images | partial | partial | Unchanged. |
| S4 naming | strong | strong | Unchanged (profile read-out identical). |
| S5 pipeline | strong | strong | Unchanged (lighter, if anything). |

No rating flips down; R2 strengthens with stated scope.

### 4. Degeneration-mode re-check

- **M1–M4 transfer unchanged** — all four live in the item
  branch/composition geometry, which is verbatim. The coverage regularizers
  acting on them are I-ProtoMF's Eq. 9, identical in form to the UI item
  side. Committed SC.8 instruments unchanged.
- **User side = host behavior:** free embedding in R^{K_t}, no candidate
  mechanism touches it — no new modes (negative u_k = legitimate
  anti-affinity, host semantics, rendered as such).
- **Fresh sweep for I-host-specific modes:** none found. The one structural
  novelty — u multiplies t* directly — introduces no new degeneration
  surface beyond M1 (a diffuse prototype makes u_k an opaque CF weight,
  which is exactly M1's existing failure reading).
- Mode list judged **complete** for fI.

### 5. Restated claims for the SC.2-delta black-box re-check

To be appended to `claims_spec.md` (subagent gets ONLY the spec, per GR10);
C1/C2/C6/C8 carry as verified (composition-level, mechanism identical);
C7a/C7b parked with read-out 4:

- **C3′ (gradient through the single path):** ∂S/∂E[r] ≠ 0 in general for a
  scored item's feature row, with S = u·t*(q_i) — the sole score path.
- **C4′ (exact full-score decomposition + rank-inert baseline):**
  S = Σ_k u_k·t*_k exactly (K_t independent summands); S = Σ_k u_k +
  Σ_k Σ_r u_k·c_{r,k} exactly; and for any two items i, j:
  S(u,i) − S(u,j) is independent of the Σ_k u_k term (baseline rank-inert).
- **C5′ (host reduction):** F=0 (ID row only) is functionally identical to
  I-ProtoMF (`item_proto`): plain per-item embedding feeding the same
  prototype layer, free user vector in R^{K_t} — forward outputs and score
  equal for matched weights.

### 6. Notes for the SC.3 re-run (recorded now, executed there)

1. Factory: fI branch = 'prototypes' Item-Proto shape with
   `embedding_ext=FeatureEmbedding`; the UI-hosted branch leaves the live
   config surface (code stays in git history; FeatureEmbeddingW stays for
   the merge stage).
2. **Init ownership:** the current branch has the factory single-owner-init
   the shared table (two consumers). In fI there is one consumer — decide
   and pin who initializes (PrototypeEmbedding-ext vs factory), keystone
   test must stay bit-identical vs `item_proto`.
3. Configs: `feature_item_proto*` re-shaped to mirror
   `item_proto_chose_original_hyper_params` exactly (user side → plain
   `embedding`) + the item-side feature additions; explicit
   `use_id_feature` key retained (absent-key hazard).
4. Keystone i02′: F=0+ID vs `item_proto` bit-identity; t11 part-C fI
   variant (e2e cold run on a trained fI toy checkpoint).

### Gate (re-host SC.1a-delta)

**Status: CLOSED — ratified by user 2026-07-12 ("I approve them all"), all
four items:** (1) transfer table ratified; (2) amendment bundle approved for
application at SC.2-delta incl. the rank-inert-baseline sharpening (subject
to C4′); (3) re-rating + mode-list completeness accepted; (4) C3′/C4′/C5′
claims delta approved for the one-round black-box re-check. → SC.2-delta.
Decisions as posed:
1. Ratify the F-DC01-01..07 transfer table (§1).
2. Approve the re-derivation amendment bundle (§2) for application to the
   design doc as dated amendments at SC.2-delta — including the new
   rank-inert-baseline sharpening (subject to its C4′ verification).
3. Accept the requirements re-rating (§3: R2 strengthened with stated
   scope; R1 basis re-derived; no downgrades) and the mode-list
   completeness verdict (§4).
4. Approve the C3′/C4′/C5′ claims-spec delta (§5) for the one-round
   black-box toy re-check at SC.2-delta.

---

## Re-host SC.2-delta — amendments applied + re-verification (2026-07-12)

### 1. Amendments applied (dated, visible)

- **Design doc:** new section "⟳ Re-host amendment (2026-07-12)" inserted
  directly after the redirection banner — subsections A (mechanism)
  through H (degeneration modes), carrying the full gated bundle incl. the
  rank-inert-baseline sharpening, the fI rendered example (canonical 5),
  the S0.2 score-share obligation in fI form, and the re-rating table.
  Banner + header + closing status lines updated (UI text below the
  amendment explicitly the historical record).
- **Claims spec:** "fI re-check" section appended (mechanism delta +
  C3′/C4′/C5′ + scope fences: C1/C2/C6/C8 carry, C7a/C7b must not be run
  against fI).
- **Candidate index:** dc01 row → *concept re-hosted & amended (fI;
  re-host amendment cycle closed) — SC.1b-delta pending*.
- **Findings ledger:** dc01 preamble note updated — transfer table
  RATIFIED 2026-07-12 (six carry, F-DC01-02 strengthened, F-DC01-07
  parked). No finding status changes (none were open).

### 2. Black-box toy re-check (GR10 — subagent saw ONLY the claims spec)

Script: `Master/temp/dc_checks/dc01/check_claims_fi.py` (float64, protomf
env; 3,240 configs = 12 seeds × F ∈ {0,1,2,3,5,11} × 5 (K_t,d) shapes incl.
K_t=1 × parameter scales {1e-3, 1, 1e3} + per-tensor scale combinations).
**Verbatim report:**

> **C3′ (gradient through the single path) — CONFIRMED**, with one mathematically forced caveat at d=1.
> - d ≥ 2: 9,504/9,504 feature-row gradient trials non-zero; minimum gradient norm relative to the natural scale (Σ|u_k|/‖q‖) = 2.66e-2 — comfortably non-zero, scale-invariantly. Autograd gradient additionally cross-checked against the closed-form ∂S/∂q = Σ_k u_k(P_t[k]/(‖q‖‖p_k‖) − cos_k·q/‖q‖²): max relative error 2.33e-11.
> - d = 1: all 2,376 trials have gradient exactly ~0 (max relative norm 3.96e-16). This is not a falsification — in 1D, cos(q,p) = sign(q)·sign(p) is locally constant, so the true gradient is identically zero. "Non-zero in general" holds; d=1 is a degenerate embedding dimension no real configuration uses.
>
> **C4′(a) (S = Σ_k u_k·tstar_k, independent summands) — CONFIRMED.** 81,000 trials (every item in every config; each of the K_t summands computed independently scalar-by-scalar and fsum'd). Max relative error 2.46e-13, max abs error 1.82e-12 (worst cases at scale 1e3). Float64-exact.
>
> **C4′(b) (S = Σ_k u_k + Σ_k Σ_r u_k·c_{r,k}) — CONFIRMED.** 81,000 trials. Max error scaled by the total magnitude of the summed terms: 3.28e-16 (machine epsilon level); max abs 2.43e-11 (at scale 1e3). Metric note: a naive relative error misleadingly reports 1.0 in the d=1/K_t=1 corner where tstar = 0.0 exactly (S = 0 exactly while the decomposition reconstructs 0 with eps-level cancellation residue); the condition-aware metric shows the identity holds at machine precision everywhere. At d ≥ 2 even the naive relative error is at float tolerance.
>
> **C4′(c) (rank-inert baseline) — CONFIRMED.** Candidate sets of 25 items (≥ 20) per config. Argsort of S(u,·) identical to argsort of the feature part S(u,·) − Σ_k u_k in 3,240/3,240 configs (stable argsort; exact ties occur only in the degenerate d=1 shape and are preserved identically, since subtracting the same float constant maps equal floats to equal floats). Pairwise-difference identity: max scaled error 4.19e-16, max abs 9.10e-13. Cross-check with the algebraically equivalent feature part u·(tstar−1): argsort identical 5/5, pairwise max rel 1.37e-12.
>
> **C5′ (host reduction, F = 0) — CONFIRMED, bit-identical.** 13,500 (item × config) trials at F=0 with matched weights Q[i] = E[V+i] (V=0), same P_t, same U: max abs diff of tstar = 0.0, max abs diff of S = 0.0 — exact bit-identity, not just tolerance-level.

Ambiguity resolutions (condition-aware error metric for exact-zero targets;
C3′ tested F ≥ 1 only; stable-argsort tie-robust reading for C4′(c)) — all
judged faithful. C7a/C7b correctly not run (scope fence held).

### 3. Response to the d=1 caveat

Accepted as stated — a 1-dimensional embedding makes every cosine locally
constant (±1), so zero gradient is the *correct* value there, for the host
exactly as for fI; the search space's d ∈ [10, 100) never visits it. No
doc consequence beyond this note.

### 4. Outcome

All three restated claims **CONFIRMED**; the rank-inert-baseline sharpening
is now verified math and stands in the design doc unconditionally. The
re-host amendment cycle is complete.

### Gate (SC.2-delta)

**Status: CLOSED — ratified by user 2026-07-12 ("yes please proceed. I
approve").** (1) Applied amendment set accepted as faithful; (2) re-check
verdicts accepted incl. the d=1 disposition; (3) proceed to **SC.1b-delta**,
then the SC.3 re-run. Cycle commit: **af78508**.

---

## Re-host SC.1b-delta — adversarial pass on the fI concept (2026-07-12)

**Setup (GR10):** independent subagent; inputs restricted to the re-hosted
design doc (incl. all prior critiques/responses/amendments) and the
requirements doc, plus the V1 scale facts (73,418/13,651/623,230; 5 fields,
V=426; "Solid" 58%; 6,517 signatures / 67.5% share). Blind to S0, this
session, the dossier, and the codebase. Brief: delta-focused — attack the
structural-dissolution claims, the rank-inert-baseline sharpening, the
re-rating, and I-host-specific new weaknesses; do not re-litigate
dispositioned items unless the re-host changes them.

### Critique (verbatim)

> 1. **The re-host creates an exact user-side gauge freedom that voids the identifiability of every rendered explanation line — the F-DC01-06 pathology relocated to the user vector, and the "fresh sweep" missed it.** §H claims the sweep "found no I-host-specific new mode; the one structural novelty — u multiplies t* directly — adds no degeneration surface." False. Since t* = 1 + P̂q̂_i with P̂ the K_t×d matrix of unit prototypes, every item's activation vector lies in one fixed affine subspace of dimension ≤ d. For any per-user shift w with P̂ᵀw = 0 and 1ᵀw = 0, u → u+w preserves S(u,i) = u·t* **exactly for every item** — all losses, including BCE calibration, are unchanged; only L2 sees it. Whenever K_t > d+1 this exact gauge has dimension ≥ K_t−d−1 per user; and whenever the composed item cloud under-spans R^d — guaranteed in the noid arm (6,517 signatures built from 426 rows) and aggravated by the contraction the document itself records ("contract the cloud into a narrower cone") — an effective gauge exists even for K_t ≤ d+1. Along these directions the rendered affinity ("your affinity u₁₂ = 0.8"), every contribution s_k = u_k·t*_k, and even the sign read in §H as "legitimate anti-affinity" are pinned by nothing but L2 shrinkage — an accuracy-blind selector — i.e., "not a learned quantity" in exactly the sense F-DC01-06 applies to lockstep feature rows. The document imposes grouped-share/determinacy discipline on item-feature splits but none on the user coefficients that multiply **every** line of the fI explanation. This gauge did not exist under the UI host, where u ∈ R^d fed a cosine surface nonlinearly and a globally shared W^u; it is a genuinely new I-host defect sitting precisely where §H declares none.
>
> 2. **Steck did not disappear; the entire score IS the unprotected dot-product half, and ranking-wise the prototype layer is a null reparameterization of the candidate's own baseline.** The redirection note says the "unprotected projection half" risk "**disappears** (there is no dot-product half)" and §H that "trains-through-cosine steel-man now covers the entire score." Both attend only to the item factor. S(u,i) = u·t* is literally a dot product whose left factor is a free per-user vector trained through nothing but that dot product — the object Steck's caveat targets, now carrying 100% of the score instead of half. Worse: S(u,i) − Σ_k u_k = uᵀP̂q̂_i = (P̂ᵀu)·q̂_i. Writing v_u := P̂ᵀu ∈ R^d, fI's ranking function class is, generically for K_t ≥ d, exactly dot-product MF over unit-normalized composed item embeddings plus a per-user constant — i.e., the `lightfm_tags` baseline with normalized items (and a rank-K_t bottleneck below d). The prototype layer contributes **zero ranking expressiveness**; the (u, P̂) factorization along which the whole explanation decomposes is invisible to the ranking loss and is pinned only by the Eq. 9 inclusion regularizers, whose weights the document's own Rashomon note concedes are chosen by accuracy-driven hyperopt among near-ties. This retroactively guts the author's rebuttal to prior point 2: the "intermediate concept layer" defense held on the UI host, where u*·t̂ is nonlinear in u and norm-sensitive in q_i; on the I host the concept layer is provably a coordinate change. The re-host did not simplify the architecture — it degraded it into a linear reparameterization of its own ablation baseline plus regularizers.
>
> 3. **The rank-inert baseline subtracts the wrong constant, and the "100%" claim inflates coverage while attribution quality strictly regressed.** (a) Only the exact all-ones shift is removed. The catalog common mode of t* — large at H&M by the document's own mechanisms ("Solid" on 58% of items, five fields, cone contraction, 67.5% shared signatures) — is quasi-constant across items yet sits fully inside the "item-discriminating" decomposition. Honest centering would be u·(t* − t̄) against the catalog mean; it is not proposed, and could not be, because t̄ is not per-item feature-decomposable — the exact (prototype, feature) additivity story survives only with the uninformative baseline. Rendered share magnitudes therefore systematically overstate discriminative relevance in precisely the regime H&M creates. (b) "100% of the item-discriminating score decomposes exactly over (prototype, feature) pairs" counts the (prototype, ID-row) atoms as pairs in the headline config: decomposed ≠ explained; 100%-decomposition is compatible with an arbitrarily small feature-explained fraction (M3 unchanged). (c) "Strictly stronger than the UI formulation" measures breadth only: the parked read-out 4 was the document's sole norm-free, true-counterfactual attribution ("no off-manifold pass"); every surviving fI share is ‖q_i‖-entangled (C6) and F-DC01-06-gauge-limited. Full coverage of a strictly weaker attribution type is a trade, not a strengthening.
>
> 4. **R1 "strong (basis re-derived)" is prior point 9 realized in full: the requirement was rewritten until a tie-free model satisfies it — and the rewritten criterion is vacuous.** The requirements document, unamended, defines R1 as "mirroring the `prototypes_double_tie` philosophy where the user-embedding and item-prototype branches share weights," with the doc's scope line reading "analogous to the `prototypes_double_tie` variant." §E concedes: "fI has a single consumer of the `FeatureEmbedding`; there is nothing to tie." The re-derived basis — "features wired into the representation that drives the score, no parallel reranker … the score has no other path" — is satisfied identically by `lightfm_tags`; a criterion that cannot distinguish the candidate from its own no-prototype baseline is a truism, not a requirement. Round one, the author defended a diluted tie as "the same tie philosophy one level deeper"; round two, the tie is gone entirely and the rating's confidence went **up**. The honest entry is: R1-as-written is unsatisfiable on the I host and its satisfaction is deferred to the merge stage — a deferral, not "strong."
>
> 5. **Block-U opacity was dissolved by amputation, and the opacity re-enters as a factor of every line — where the S0.2 instrument can no longer see it.** "Block-U's share is structurally zero (no user-prototype surface in the candidate)": zero because the surface was deleted, not because anything was explained. Every fI line s_k = u_k·t*_k is half ungrounded — u_k is a raw CF scalar with no grounding pathway at all, strictly less interpretable than the UI host's user prototypes, which at least admitted the host's post-hoc naming. §C's "per-prototype legible" borrows legibility from prototype-profile validity, which the same document rates R4-partial with M1 open. Meanwhile the parameter mass is telling: at d = K_t = 64, the ungrounded user table is N×K_t ≈ 4.7M parameters against ~27K interpretable metadata rows — the model is overwhelmingly its unexplained half, yet the score-share disclosure that policed exactly this mass under the UI host now reads "structurally zero." S2 "strong (re-scoped)" is the deleted user-side surface rebranded as a future lineage stage; R2 "strengthened" is earned by counting only the right-hand factor of each product.
>
> 6. **New at H&M scale in the cosine-only form: the shared-signature common mode angularly squashes the ID row — an unlisted fourth component of the ids/noid gap bundle and a concrete downward mechanism for the re-anchored R6 bet.** Signature twins have q_i = m + e_i, q_j = m + e_j with m the shared 5-row sum; their angular separation scales as ‖e_i − e_j‖/‖m‖ — first-order suppressed by the same omnipresence mass M2 records ("an equal-length e_Solid still votes in 58% of all item sums"). Under the UI host, the linear half individuated twins via W^t(e_i − e_j), independent of ‖m‖; fI's only individuation channel is angular and m-suppressed. With 67.5% of items sharing signatures, `use_bias=0` fleet-wide, no LightFM biases (deviation 4), and the cosine discarding ‖q_i‖, direction is the *sole* channel for both popularity and twin separation, competing for one summed vector's angular budget. Unaccounted consequences: (i) the ablation-gap interpretation rule's bundle needs a component it does not list — ID signal present in parameters but unrealizable through the cosine, which mimics "popularity blindness" in the D4 strata and will misclassify component 1 vs 2; (ii) the registered hypothesis (warm feature-explained fraction predicts cold degradation) is confounded by it; (iii) versus `item_proto`, whose free q_i suffers no shared-mass suppression, fI_ids pays this angular-compression tax on the majority of the catalog — a mechanism for losing the R6 bet that §H's "no new mode" verdict never surfaces. The re-hosting materially changes this dispositioned item.
>
> 7. **K_t now does triple duty — user capacity, score rank, explanation vocabulary — and the R6 re-anchoring quietly lowers the thesis bar.** "User table **N×K_t** (dimension K_t, not d — host property)": with 73,418 users, accuracy-driven hyperopt will size K_t as user capacity, but K_t is simultaneously the number of nameable concepts (S4), the number of explanation lines, the effective score rank min(K_t, d) (point 2), and the exact-gauge dimension K_t − d − 1 (point 1) — one knob, four roles, so the Rashomon protocol ("within the ε-set, maximize interpretability") no longer has a fixed-capacity axis to move along: shrinking the vocabulary now shrinks every user's representation. None of this existed under the UI host, where user dimensionality (d) and prototype counts were independent knobs. Simultaneously, "the empirical bet is now fI vs **I-ProtoMF**" re-anchors R6 — whose requirement text says "beating the paper baseline is the primary thesis bar" — to the weakest member of the host family (the paper's Table-2 headline is UI-ProtoMF); passing the re-anchored bet is compatible with sitting well below the original bar, with reconciliation deferred to a merge stage that exists only as a promise. R8 "strong (cleaner)" is financed the same way: read-out 4 parked, deviation 2 parked — the complexity was not removed, it was exported to undesigned future stages and the cleanliness claimed now.

### Response (concede / rebut / open risk, point by point)

**P1 — user-coefficient gauge freedom. Concede the mathematics and the
sweep miss; rebut "created by the re-host"; adopt the discipline →
F-DC01-08.**
- **Conceded (math, verified):** for w with P̂ᵀw = 0 and 1ᵀw = 0,
  S(u+w, i) = S(u, i) exactly for every item (t*(i)·w = 1ᵀw + q̂ᵢᵀP̂ᵀw = 0).
  Exact dimension ≥ K_t − d − 1 when K_t > d+1 (reachable in the search
  space: K_t, d ∈ [10,100) independently); effective gauge under
  activation-cloud under-spanning (noid arm aggravated). Moreover the
  dynamics mirror C8 precisely: the loss gradient wrt u is a combination of
  t* vectors, all ⊥ w — so **the gauge component of u is frozen at
  initialization** (or decayed by L2), the same frozen-difference mechanism
  F-DC01-06 exhibits for lockstep rows. §H's "no new mode" verdict was
  wrong on this surface — the sweep treated "user side = host behavior" as
  "no mode," which conflates *scored behavior* (unchanged) with
  *explanation validity* (unaudited). Learnings entry added.
- **Rebutted (attribution):** the gauge is a property of **I-ProtoMF's own
  score form and explanation device** (paper §5.2 renders u_k·t*_k lines on
  the host too) — every fI-vs-host comparison carries it identically on
  both sides. It is host-inherited surface, not a defect the composition
  introduces; but since fI's thesis weight rests on these lines, inheriting
  it without discipline would be indefensible — hence adoption rather than
  deflection.
- **Adopted → F-DC01-08:** extend the F-DC01-06 discipline to the user
  coefficient vector: (a) gauge structure named in the design doc (exact
  dim, effective under-spanning case, frozen-at-init dynamics, L2 as the
  accuracy-blind minimum-norm canonicalizer — u's rendered values are
  *canonical representatives*, disclosed as such); (b) SC.8 instruments:
  activation-cloud spectrum/effective rank, decomposition of trained u
  into in-span vs gauge mass (both arms; noid aggravation check);
  (c) rendering rule for SC.5: affinity lines carry the same
  determinacy-annotation discipline as feature splits, for **both** fI and
  the host baseline (shared instrument, GR7-fair); (d) toy check C9
  (gauge invariance; frozen gauge component under SGD; L2 decay) at the
  application round.

**P2 — "null reparameterization" + Steck re-scope. Partial concede
(wording + one honest paragraph); rebut the consequence chain.**
- **Conceded:** §H's sentence "trains-through-cosine now covers the entire
  score" overclaims — it is true of the **item-side semantic read-outs**
  (shares, profiles: the quantities read *as meaning*), while the user
  coefficients are plain dot-product-trained parameters with no Steck-style
  protection; their validity is governed by F-DC01-08's discipline, not by
  the cosine argument. §H is re-scoped accordingly. Also conceded: the
  function-class observation is correct as algebra — generically for
  K_t ≥ rank d, ranking(fI) = ranking of (P̂ᵀu)·q̂_i, a dot-product model
  over **unit-normalized** composed embeddings; recorded honestly in the
  design doc.
- **Rebutted (the "degraded into its own baseline" conclusion):**
  (i) *normalized ≠ unnormalized*: the reduction lands on MF over q̂_i,
  which is **not** `lightfm_tags` (free norms) — the critique itself
  builds P6 entirely on that very difference (norm channel discarded);
  it cannot simultaneously hold that fI ≡ lightfm and that fI lacks
  lightfm's norm channel. (ii) The identical "coordinate change" argument
  applies verbatim to **published I-ProtoMF** vs normalized-MF — yet the
  host's Table 2 separates them empirically: parameterization,
  regularizers (Eq. 9 acts on the (u, P̂) factorization the collapsed form
  erases — pinning it is their *job*, not an embarrassment), L2's implicit
  bias in (u, P) vs v coordinates, and optimization dynamics all differ.
  Function-class equality ≠ model equality — the learnings-ledger pattern
  (formally-correct absence/equality claim, overclaimed consequence).
  (iii) dc01 never claimed the prototype layer adds ranking
  expressiveness (5b-2 settled that in round one); its claimed value is
  the factorized, groundable explanation structure — whose meaningfulness
  is exactly what the committed instruments (M1 sharpness, profile
  validity, F-DC01-08 spectrum) measure. (iv) K_t < d gives a genuine
  rank bottleneck — noted as a capacity fact in the amendment.
- **Adopted:** one design-doc paragraph (amendment set A): the reduction
  identity, the normalization difference, the bottleneck case, why
  parameterization+regularization still differentiates (host precedent),
  and the honest statement that the (u, P̂) factorization is pinned by
  regularizers/init/L2 — not by the ranking loss (connecting to
  F-DC01-05's Rashomon honesty and F-DC01-08).

**P3 — baseline centering, ID atoms, "strictly stronger." Concede all
three as wording/display obligations; no mechanism change.**
(a) Correct: rank-inert removal covers only the exact all-ones shift; the
catalog **common mode** of t* is quasi-constant and sits inside the
decomposition. Exact additivity exists only against the all-ones baseline
(t̄ is not per-item decomposable) — that is *why* it is kept, but rendered
shares must not be read as discriminativeness. Adopted: common-mode
disclosure in the rendering notes + the committed discriminative
companions named as such (per-value share vs value frequency; cloud
tilt/spread — both already in the SC.8 set; plus catalog-mean share
alongside rendered shares where cheap). (b) Correct: "(prototype,
feature) pairs" silently includes ID atoms in the ids arm — reworded to
"(prototype, row) pairs; the feature-explained fraction (binding
disclosure) separates metadata from ID mass." (c) Correct: "strictly
stronger" compared breadth while attribution *type* regressed (read-out 4
was norm-free and counterfactual; cosine shares are ‖q‖-entangled) —
reworded to a declared trade: full coverage by the entangled-share type
now; the complementary linear channel returns at fU/fUfI.

**P4 — R1 vacuity charge. Rebut vacuity; concede the presentation; the
requirements-doc reading is put to the user.**
- **Rebutted:** R1 was never the *distinctness* requirement — the
  requirement set is a conjunction, and what separates fI from
  `lightfm_tags` is R3/R4 (prototype paradigm, grounded prototypes), not
  R1. A requirement satisfiable by a baseline is not thereby vacuous
  (R8 is satisfied by plain MF; nobody calls parsimony a truism). R1's
  operative core — features inside the scored representation, no parallel
  reranker/bolt-on — does real work: it is what rejected Seed 3 (RLFM
  regression prior) at Step 1 and what forbids post-hoc rerankers.
- **Conceded:** the requirement's *exemplar mechanism* (weight sharing
  across branches) is unsatisfiable on a single-branch host, and §G's
  "strong (basis re-derived)" glossed that too smoothly — the honest form
  is: **strong on the operative core; the exemplar tie-mechanism is
  host-inapplicable at this stage and returns at the merge** (a deferral,
  stated as one). §G row reworded.
- **Put to the user (living-document decision):** a dated clarification
  note in the requirements doc §R1 recording the per-host reading
  ("exemplar mechanism applies where the host has two branches to tie;
  operative core is the per-stage criterion"). Declined = the §G rewording
  alone carries it.

**P5 — "amputation" + parameter mass. Rebut the framing; concede and fix
the "structurally zero" phrasing.**
- **Rebutted:** the user side was not deleted from an explanation surface
  fI claims — it was **rescoped by the ratified lineage decision** (each
  side grounded against its own host; fU/afU own the user surface). The
  like-for-like comparison (fI vs I-ProtoMF) has the identical N×K_t CF
  user table and identical u_k factors on both sides; the candidate's
  delta is item-side by construction, and the parameter-mass arithmetic
  indicts the host family, not the delta. "Strictly less interpretable
  than UI's user prototypes" compares across *stages*, which the lineage
  does deliberately and openly (banner, index, masterplan).
- **Conceded:** declaring the S0.2 obligation "structurally zero" was an
  own-goal formulation — it reads as closing a disclosure that should
  have *morphed*. Restated (amendment set A): every line s_k = u_k·t*_k
  is a product of a CF-learned personalization coefficient (disclosed;
  F-DC01-08 discipline) and a grounded activation (feature-explained
  fraction disclosed); the per-line factorization IS the fI form of the
  score-share disclosure. R2 "strengthened" stands with its stated scope
  (prototype *meaning* + item-side attribution intrinsic — vs the host's
  post-hoc-only meaning); S2 row stands (the lineage is the user-side
  plan, not a rebranding of loss).

**P6 — angular individuation tax. Concede substantially — the strongest
new point; adopt → F-DC01-09.**
Correct and genuinely new: in fI the cosine discards ‖q_i‖ and the fleet
runs bias-free, so **direction is the sole expressive channel** — twin
separation and popularity compete for one summed vector's angular budget,
and the shared signature mass m suppresses twin separation first-order
(‖e_i − e_j‖/‖m‖; worst under max_norm, which caps ID rows while m sums
five capped rows). Under UI, the linear half individuated twins
independent of ‖m‖ — this is a real re-host delta to a dispositioned
item, exactly what this pass exists to catch. **Adopted → F-DC01-09:**
(a) design-doc note: fI(ids) expressiveness is direction-only; named
mechanism + max_norm interaction; (b) the §3.6 ablation-gap interpretation
rule gains sub-component **(1b) "ID signal present but
angularly unrealizable"** with the explicit warning that it mimics
popularity blindness in D4 strata (component-1 vs component-2
misclassification risk); (c) the braid hypothesis (F-DC01-04) gains this
as a confounder note; (d) SC.8 instruments: twin-pair angle distribution
(angle(q̂_i, q̂_j) for same-signature pairs vs matched non-twin pairs), read
jointly with the ids-vs-`item_proto` comparison (host q_i carries no
shared-mass suppression — a named candidate mechanism for an R6 shortfall,
falsifiable). Partial reservation (not a rebut): ID rows can rotate q̂
substantially when unconstrained (no max_norm) and prototype geometry can
amplify small angular differences — the tax's *size* is empirical; the
mechanism is real and now instrumented.

**P7 — K_t multi-role + R6 bar + R8. Concede one recorded note; rebut the
"quietly."**
(a) K_t coupling (user capacity = score rank = vocabulary = gauge dim) is
**host-inherited** (I-ProtoMF itself sizes user capacity by n_prototypes)
— but the Rashomon consequence is real and previously unrecorded:
vocabulary size is not a free interpretability knob in fI; shrinking it
shrinks user capacity. Adopted as a note in the §3.8 Rashomon timing
amendment (late-thesis stage concern, per the F-DC01-05 timing decision).
(b) R6 re-anchoring: rebutted as "quietly" — the staging is the ratified
lineage decision, recorded in the banner, index, and masterplan; the
UI-level bar returns at the merge stage *by design*. Adopted for
explicitness: §G R6 row states the two-level bar (stage bet: fI vs
I-ProtoMF now; paper-headline bar: at the fUfI merge). (c) R8: rebutted —
parking is staging, and each later stage carries its own DC + scrutiny
cycle; the complexity is not hidden, it is scheduled. No change.

### Fold-in summary

- **New findings:** **F-DC01-08** (user-coefficient gauge/canonicalization
  discipline + Steck re-scope + S0.2-obligation restatement; minor,
  host-inherited but thesis-surface-critical), **F-DC01-09** (fI
  direction-only expressiveness: angular twin-individuation tax +
  gap-rule sub-component 1b + braid confounder + twin-angle instrument;
  minor).
- **Amendment set A (wording/honesty, no new IDs):** P2 function-class
  paragraph; P3(a) common-mode disclosure, (b) "(prototype, row)"
  rewording, (c) "strictly stronger" → declared trade; P4 §G R1 row
  rewording (+ optional requirements-doc R1 clarification note — user
  decision); P5 S0.2-obligation restatement; P7 R6 two-level-bar sentence
  + Rashomon K_t-coupling note.
- **Toy re-check commitment:** C9 (gauge invariance exact; gauge component
  frozen under SGD; L2 decay) — one round, at the application step,
  black-box per GR10.
- **SC.8 metric additions:** activation-cloud spectrum / effective rank;
  trained-u in-span vs gauge-mass decomposition (both arms); twin-pair
  angle distribution vs matched non-twins.
- **Learnings entries:** (1) host-inherited surfaces still need
  explanation-validity audit — "host behavior" ≠ "no mode" when the
  candidate's thesis weight rests on that surface; (2) generic
  function-class reductions (rank/reparameterization arguments) are
  formally cheap and consequence-expensive — check normalization details,
  regularizer visibility, and host empirical precedent before conceding
  "model X = model Y."

### Gate walkthrough addendum (2026-07-12, deep-dive sitting)

Decision 2 was walked through in plain-language build-up mode (per gate
convention), one finding at a time. Recorded outcomes:

- **F-DC01-08 walked and accepted** (geometry → blind directions → frozen
  gauge → remedy quotient/canonicalization/annotation). User probe on the
  comparison anchor clarified: fI's headline baseline is I-ProtoMF (not
  UI); F-DC01-08 is fully **symmetric** (host carries the identical
  anatomy → renderer annotates both sides).
- **F-DC01-09 walked and accepted** ("a fair gap that one we should
  measure and mention"). Two clarifications fixed at the gate, now part
  of the finding's interpretation record: (a) **bottleneck vs tax role
  split** — the direction-only bottleneck is host-shared (context for
  absolute numbers, never explains an fI-vs-host delta); the anchored-
  direction tax is fI-specific (the candidate explanation for an ids-arm
  shortfall vs `item_proto`); (b) **the tax is an ids-arm phenomenon** —
  in noid, twins are the same vector (the known ceiling; nothing new);
  in ids, the ID row must buy twin separation back *through* the angle
  bottleneck against the shared mass, which is why the measured ids-vs-
  noid gap **understates** what item-level memory buys (sub-component 1b
  mimics popularity blindness in D4).
- **NEW registered hypothesis (length-channel refund), user-adopted:**
  prompted by the user's question *"does a fUfI model, once working,
  have the ability to leverage vector length?"* — yes: the merge's linear
  halves see full vectors (twins separate via W(e_i−e_j), no ‖m‖
  suppression; popularity can live in ID-row length), so the angular tax
  is **stage-local** and the merge should refund it — recovery expected
  specifically on twins and popular items, with the single-sided stages'
  tax measurements as the prediction's evidence base. The refund
  reimports the cold magnitude handicap and the Steck linear-half
  exposure (antidote = parked read-out 4). Verbatim capture: vault
  `2026-07-12_1224`; scratchpad `[captured scrutiny dc01]` length-channel
  entry.
- **Thesis-structure directive (user, recorded):** the accumulated hidden
  effects and registered hypotheses (identifiability gauges, tax, braid,
  refund, …) become a dedicated thesis section (working shape: "model
  mathematics, hypothesis risks, and experiments" — name/structure TBD):
  reader first gets the algorithm, then is challenged with the hidden
  effects one by one (what, when, sign, measurement), inviting the
  careful reader to find further unmeasured effects. Verbatim source:
  vault `2026-07-12_1224`. SC.9's chapter draft and the SC.5 instrument
  design take this as standing input.

### Gate (SC.1b-delta)

**Status: decisions 1–2 CLOSED (2026-07-12, deep-dive walkthrough);
decisions 3–4 OPEN.**
1. ✅ Point dispositions P1–P7 — approved ("seems reasonable, it checks
   out").
2. ✅ **F-DC01-08** and **F-DC01-09** approved with bundled proposals,
   plus the walkthrough clarifications and the refund-hypothesis
   registration above.
3. ✅ **Amendment set A approved**; R1 clarification note: **APPLY**
   (user 2026-07-12 — "apply the note please", after the
   note-vs-decline trade-off was laid out; rationale accepted: the
   requirements doc briefs every future black-box reviewer across ≥3
   more stages whose hosts misfire on the literal double-tie language).
4. ✅ **C9 approved** (explicit user message; also contained in
   F-DC01-08's approved bundle).
5. Then: apply the bundle (SC.2-style application round: design-doc
   amendments incl. the refund-hypothesis note, ledger entries finalized,
   learnings flush, C9 re-check) → **SC.3 re-run** (factory/config
   re-pointing, keystone i02′, t11 part-C fI variant).

---

## Re-host SC.1b-delta — application round (2026-07-12)

### 1. Amendments applied (dated, visible)

- **Requirements doc:** R1 clarification note applied (dated, marked
  "clarification, not a weakening"; per-host reading: exemplar
  weight-sharing mechanism applies where the host has two branches;
  operative core is the per-stage criterion; exemplar assessed at the
  merge stage).
- **Design doc, new §I of the re-host amendment block:** I.1 F-DC01-08
  user-coefficient gauge discipline (characterize→exhibit→remedy→delimit,
  SC.8 instruments, shared-renderer fairness); I.2 F-DC01-09
  direction-only expressiveness + angular twin tax (+max_norm
  interaction, ids-arm scoping, twin-angle instrument); I.3
  function-class honesty paragraph; I.4 registered length-channel-refund
  hypothesis; I.5 Rashomon K_t-multi-role note.
- **Design doc, dated inline corrections:** §C "(prototype, row)"
  rewording + catalog-common-mode disclosure + "strictly stronger" →
  declared trade (P3); §C S0.2 obligation restated as per-line
  factorization disclosure (P5); §H mode-sweep verdict corrected (two
  modes found where "none" was declared — lesson ledgered) + Steck
  sentence re-scoped to item-side read-outs (P2); §G R1 row (operative
  core / exemplar deferral, pointer to the requirements-doc note) and
  R6 row (two-level bar) reworded (P4/P7); old-§3.6 gap-interpretation
  rule extended with sub-component (1b); old-§3.4 braid hypothesis
  gains the F-DC01-09 confounder note.
- **Claims spec:** C9 section appended (gauge space G(P̂), claims a/b1/b2/c).
- **Learnings ledger:** two entries flushed (host-inherited surfaces need
  explanation-validity audit; function-class reductions are formally
  cheap / consequence-expensive).

### 2. Black-box C9 re-check (GR10 — subagent saw ONLY the claims spec)

Script: `Master/temp/dc_checks/dc01/check_claims_c9.py` (float64, protomf
env; shapes (K_t=8,d=3) and (K_t=12,d=5), gauge dims 4/6). **Verbatim
report:**

> **C9(a) — exact score invariance: CONFIRMED.** 1755 scored pairs across 2 shapes × 5 seeds × 3 parameter scales (0.1, 1, 10) × 3 gauge-shift magnitudes (‖w‖ = 0.5, 5, 100). Max relative error |S(u+w,i) − S(u,i)| / (‖w‖·‖tstar‖) = 1.417e-14. Pure float rounding; no violation.
>
> **C9(b1) — gauge component frozen when prototypes/item table frozen: CONFIRMED.** [Both readings run: U-only trainable, and U+E trainable.] Max drift of gauge coefficients from initial values, over all steps/users: 2.887e-15 (U-only) and 3.553e-15 (U+E). Meanwhile the in-span component moved substantially (min largest per-user in-span displacement 1.562 / 1.529). Frozen-gauge + moving-in-span both hold.
>
> **C9(b2) — grad wrt user vectors orthogonal to the CURRENT gauge space, everything trainable: CONFIRMED.** Gauge basis recomputed from current P_t every step. Max over all steps/users/basis vectors of |gᵀw|/‖g‖ = 1.573e-16 — machine epsilon. The gauge space visibly rotates with P_t, yet the gradient never acquires a gauge component.
>
> **C9(c) — L2 canonicalization, exact (1 − lr·wd)^T decay: CONFIRMED.** Max relative deviation from the per-step prediction over all 300 steps, all coefficients, all 6 runs = 1.460e-14; empirical decay ratio matched (1−0.0005)^300 = 0.860675689719 to all printed digits.

All four sub-claims confirmed; F-DC01-08's mathematical basis (invariance,
frozen-at-init, L2 canonicalization) is now pinned by committed toy
evidence, symmetric to C8's treatment of the item-side gauge.

### 3. Gate-adjacent capture (user, same sitting): the length-ablation control

While the round ran, the user proposed the **length-ablation experiment**
for the refund hypothesis: train fUfI normally and with entity vectors
forced to equal length; compare (fI/fU arms expected null). Adopted with
three sharpenings (constraint hits composed/entity vectors, not ingredient
rows — those are direction vote-weights; the fI/fU arms are exact no-ops →
harness keystone controls; compensation via the user vector is only
partial — a per-user vector cannot restore a per-item scalar, so the
constrained fUfI re-routes into the fI regime and the gap isolates the
length channel *within one architecture*, deconfounding the refund
hypothesis from everything else the merge adds). Also prices the
length-regularizer continuum (K2/max_norm/L2 endpoints) and the
Steck-exposed part (unit norms make the linear halves cosine-like).
Captured: scratchpad length-channel entry, ↪ refinement. Belongs to the
fUfI stage's run plan; its DC cycle inherits it.

### 4. Status

Ledger: F-DC01-08 / F-DC01-09 → **fixed** (amendments applied this round;
resolving commit recorded in the ledger entries). Candidate index →
*concept re-hosted, amended & adversarially re-checked (fI) — SC.3 re-run
pending*. **SC.1b-delta is CLOSED in full** (all gate decisions 1–4
ratified; application round complete). → Next step: **SC.3 re-run**
(factory/config re-pointing to the I host, keystone i02′ vs `item_proto`,
t11 part-C fI variant, dc_checks port).

---

## ⟳ SC.3 re-run — implementation re-pointed to the I host (2026-07-12)

**Scope (redirection plan step 3 + SC.1a-delta §6 handoff notes, all four
executed):** factory/config re-pointing, init-ownership decision, keystone
i02′ vs `item_proto`, t11 part-C fI variant, dc_checks port. Protocol SC.3
section re-read; manifest loaded (ledger re-check at session start: no `open`
findings anywhere; none stale).

### 1. Factory + configs (commit 998d67b)

- **Factory `feature_item_proto` branch** now builds the fI shape: item =
  `PrototypeEmbedding(embedding_ext=FeatureEmbedding)` — the item branch
  VERBATIM from the UI build — user = I-ProtoMF's free `Embedding(n_users,
  K_t)` via the host 'prototypes' Item-Proto path. The Concatenate wrappers,
  `FeatureEmbeddingW` projection and user-prototype branch leave the live
  surface (git history keeps them; the `FeatureEmbeddingW` class stays for
  the fUfI merge stage). Two loud guards added: the user branch must be
  `'embedding'` (a stale UI-shaped config now fails at build, not silently);
  `use_weight_matrix` must be off (the user vector lives in R^{K_t}).
- **Init ownership (handoff note 2, decided and pinned):** the **factory**
  stays the single initializing owner of the feature table. Rationale:
  `PrototypeEmbedding` never inits a passed-in ext (contract pinned by the
  t03 golden regression), and `RecSys.init_parameters` cascades only one
  level — so no other path ever reaches the nested table; an explicit
  factory call is the only unambiguous owner. Same convention as the UI
  build, now with a single consumer. Distribution note: constructor default
  and `general_weight_init` are both N(0,1) — the decision is ownership
  clarity, not numerics. Pinned by t04 (`recsys_init_does_not_redraw_feat_table`
  + `recsys_init_redraws_user_table`).
- **Config** re-shaped to mirror `item_proto_chose_original_hyper_params`
  EXACTLY (user side → `{'ft_type': 'embedding'}`); item side unchanged
  (canonical 5, explicit `use_id_feature` key retained — absent-key hazard).
  `_noid`/`_f0` deepcopies inherit mechanically; `_f0` comment re-anchored
  to `item_proto`. t05 now pins the mirror structurally: the item side's
  additions vs the host config are exactly `{use_id_feature, feature_fields}`.

### 2. Keystone i02′ + dc_checks port (commit 3adadde)

- **Keystone i02′ — fI(F=0, +ID) ≡ `item_proto`: BIT-IDENTICAL.** Weight-copy
  methodology as before (RNG draw order differs); forward scores, loss, and
  all three per-parameter gradients (user table, P^t, item table) equal under
  `torch.equal`, not tolerance — matching the C5′ toy-check finding. The
  handoff's "keystone must stay bit-identical" bar is met exactly.
- **Suite port:** harness fI param builder (+`item_proto_param`); i01 shapes;
  i03 renamed → `i03_gradient_single_path.py`, re-scoped to the single score
  path (C3′ integration analogue) and extended with a no-grad-on-unused-row
  check (indexed-lookup discipline); i04 unchanged semantics; t11 parts A/C
  ported (part A: drop reaches `PrototypeEmbedding.embedding_ext`, the t*
  surface — the WHOLE fI score path — changes for cold items, warm rows
  bit-identical, manual-zeroing state equality, `_noid` no-op; part C: full
  runner e2e on a REAL trained fI toy checkpoint — drop recorded, attr-kNN
  seam skips, metrics finite; 26 checks green). dc02's i06 regression
  build-check updated to the fI param (it exercised the dc01 branch with the
  UI shape — the only cross-suite breakage, caught by the full sweep).
- `cold_eval._find_feature_embedding` docstring re-anchored (fI = the live
  nested case; Concatenate recursion documented as the merge-stage path).
  No behavior change anywhere in cold machinery — F-S0-07(b)'s guard
  semantics (`config_expects_id_drop` + post-drop abort) carry unchanged and
  t11 part B re-pins them.

### 3. Scope addition, flagged: minimal explanations re-point (commit 59bad13)

Not in the four handoff notes, added under the steel-man rule (SC.5 must not
open against crashing code): `FeatureItemProtoAccessor` re-shaped to the I
host (item prototypes + intrinsic grounding; `has_user_prototypes` and
`has_projections` now False), and `weight_viz` drops `feature_item_proto`
from `supports()` (no projection branch exists; the fI per-prototype
breakdown u_k·t*_k is the SC.5 shared-renderer build, per protocol v1.2
§SC.5.4). **Mechanical alignment only** — comparability matrix, gating-
consistency check, attribution gates, and the renderer are SC.5 work and
remain untouched. t10 extended to pin the new flags + gating.

**Known-deferred (recorded, not fixed here):** the three SC.5-era demo
scripts (`dc_checks/dc01/readout_demo.py`, `readout_artifact.py`,
`Master/temp/run_explanations_hm.py`) still assume the UI shape
(`model_1`/`model_2` handles). They are one-off checkpoint demos outside the
live pipeline; SC.5's renderer build supersedes or ports them. `run_combo`'s
`EXPLAINABLE_MODELS` still excludes `feature_item_proto` (unchanged, wired
at SC.5) — no training-path surface can hit the explanations code meanwhile.

### 4. Test-run record

| suite | result |
|---|---|
| dc01 t01–t11 (incl. ported t04/t05/t10/t11) | ALL PASS |
| dc01 i01, **i02′ keystone (bit-identical)**, i03′, i04, i06a | ALL PASS |
| s0 t01–t06, i01–i02 (cold machinery regression) | ALL PASS |
| dc02 t01–t08, i01–i06 (shared-factory regression; i06 param updated) | ALL PASS |
| decoupling_mass --selftest | PASS |

Commits: **998d67b** (factory+configs; modifications-log entries for both
upstream files), **3adadde** (dc_checks port + keystone + t11), **59bad13**
(accessor/weight_viz minimal re-point), plus this docs commit. Ledger: no
new findings — all work executed under previously ratified dispositions;
F-S0-07 entry annotated (t11 now fI-shaped).

### Gate (SC.3 re-run)

**Status: OPEN — awaiting user review.** Decisions requested:
1. Accept the factory/config re-point incl. the two new build guards
   (user-branch `'embedding'` assert; `use_weight_matrix` off).
2. Ratify the **init-ownership decision** (factory single-owner, as argued
   in §1).
3. Accept keystone i02′ (bit-identity vs `item_proto`) + the suite port +
   the t11 fI verification as closing the handoff notes.
4. Accept the flagged **scope addition** (minimal accessor/weight_viz
   re-point + known-deferred demo scripts) — or direct otherwise.
5. Proceed to **SC.4** (implementation scrutiny: black-box spec-vs-code
   against the re-hosted design doc + this implementation).

**Gate closure (2026-07-12, same sitting): CLOSED — user accepted all five
decisions verbatim ("accept all we can proceed"): re-point + guards accepted;
init-ownership (factory single-owner) RATIFIED; keystone/suite/t11 closure
accepted; accessor scope addition accepted (demo-script deferral stands);
proceed to SC.4.**

---

## SC.4 Implementation scrutiny (2026-07-12)

**Scope:** the fI implementation as landed at the SC.3 re-run (commits
998d67b/3adadde/59bad13). Protocol SC.4 section re-read. Black-box
spec-vs-code subagent (GR10) received ONLY the corrected design doc (incl.
re-host amendment block, which its banner declares governing) + the
implementation files (factory, extractors, feature_ids, hyper_params,
rec_sys, trainer/tester seams, cold_eval); blind to the dossier, ledger,
protocol, tests, and this session.

### 1. Black-box spec-vs-code report (verbatim)

> AUDIT REPORT — dc01 fI-ProtoMF (`feature_item_proto`) governing spec vs implementation
>
> Scope note: audited against the governing spec = HOST REDIRECTION banner + "⟳ Re-host amendment" §§A–I + the carried-unchanged parts of §§3.4–3.8 (as amended). UI-hosted text treated as historical record per the banner.
>
> DIVERGENCES
>
> 1. **Minor — init-ownership asymmetry between fI's item table and the host baseline's item table.**
> Spec: §B "Item branch verbatim", §D "F=0+ID keystone reduction target = `item_proto` (claim C5′)" and the like-for-like comparison discipline (also the config's own "mirrors item_proto_chose_original_hyper_params EXACTLY … directly comparable" framing, `confs/hyper_params.py:203-210`). The spec is silent on parameter initialization.
> Code: `feature_extraction/feature_extractor_factories.py:150` — the factory calls `item_feat_embed.init_parameters()` (→ `general_weight_init`). In the host `item_proto` build, the nested item `Embedding` is **never** re-initialized: `RecSys.init_parameters` cascades one level to `PrototypeEmbedding.init_parameters` (`feature_extraction/feature_extractors.py:409-411`), which touches only the weight matrix, so the host item table keeps `nn.Embedding`'s construction init (N(0,1)).
> Consequence: unless `general_weight_init`'s Embedding branch is exactly N(0,1) — not verifiable here, `utilities/utils.py` is outside my allowed file set — the fI feature table (and therefore the `_f0` keystone arm) starts training from a different init distribution than `item_proto`'s item table, and in any case consumes the RNG stream differently. The C5′ **forward** equivalence itself is unaffected and verified by reading: with `feature_fields=[]`, `n_features=0` and `q_i = E[0 + i]` — a plain per-item table lookup (`feature_extractors.py:176-180`), architecturally identical to `Embedding`. But a seed-matched `_f0` run is not init-identical to a seed-matched `item_proto` run, which slightly weakens the "any delta is the metadata's doing" reading of the keystone control and the fI-vs-host like-for-like bet (R6′). The code marks this as a deliberate pinned decision ("Init ownership (pinned at the SC.3 re-run)", factories.py:146-149); the governing spec document does not record it.
>
> 2. **Minor — cold-eval applies the ID-column drop before the warm sanity block and the tie-block diagnostic, not only at cold scoring.**
> Spec: §3.4 cold notes / deviation 6 commit the drop for **cold inference** ("at inference the ID column can be dropped for unseen items"; "(b) The ID-column drop … LANDED — the drop covers the nested branch … refusal lifted, replaced by a post-drop abort").
> Code: `utilities/cold_eval.py:311-320` applies `drop_cold_id_rows` once at session start; the subsequent **warm test on the variant** (line 324-326) and the full-catalog `tie_block_diagnostic` (line 368-371) therefore score cold items (as sampled negatives / catalog members) with zeroed rather than untrained-random ID rows.
> Consequence: the "warm test (variant)" sanity number for `_ids` runs is measured on a partially-modified model, so the F-S0-11 warm-delta against the canonical reference is not a pure like-for-like of trained weights. Defensible as "unseen items are always scored feature-only," and the governing detail lives in the S0.3 spec (outside my allowed set) — flagged so it can be checked there; the dc01 document itself neither commands nor forbids this sequencing.
>
> 3. **Cosmetic — §A anchors the fI loss as host Eq. 10 (sampled softmax + Eq. 9 + L2); the config searches `loss_func_name` ∈ {bce, bpr, sampled_softmax}.**
> Code: `confs/hyper_params.py:211-239` inherits `base_hyper_params` (line 18). This is the in-repo host `item_proto` search space verbatim, which the spec's own comparability discipline mandates (F-DC01-01 / §D "same search space"), and the Eq. 9 regularizers ride `get_and_reset_loss()` under every loss. A doc-vs-config framing mismatch only; no behavioral divergence from the operative commitment.
>
> SPEC COMMITMENTS NOT FOUND IN PROVIDED FILES — all explicitly staged elsewhere by the document, hence OUT OF SCOPE rather than divergences:
> - §C read-outs 1–3 (per-prototype contribution s_k = u_k·t*_k, per-feature shares c_{f,k}, global profile cos(e_f, p_k)), the rank-inert baseline rendering, the per-line factorization / feature-explained-fraction (M3) disclosure, grouped-share + determinacy annotations (F-DC01-06), and the §I.1 canonical-representative disclosure in the shared GR7-fair renderer — explanations-stage deliverables; not present in the audited files.
> - §I.1/§I.2/§H SC.8 instruments (activation-cloud spectrum, gauge-mass decomposition, twin-pair angle distribution, per-value share vs frequency, cloud spread/tilt, profile-validity spot-check) — committed to SC.8.
> - F-DC01-06 decoupling-mass script and the C9/i02′/t11 toy verifications — referenced to `Master/temp/dc_checks/dc01` (outside the audited set); their claims are gate-recorded in the doc, not re-verifiable here.
> - K1–K4 constraint knobs (§3.8) — correctly NOT implemented for dc01 ("designed, NOT stacked"); absence is compliance.
>
> VERIFIED FAITHFUL (spot-checked, no divergence): I-host topology exact — user = free `Embedding(n_users, K_t)` via the host Item-Proto factory path (factories.py:125-126), item = `PrototypeEmbedding(embedding_ext=FeatureEmbedding)` with `use_weight_matrix=False` (factories.py:132-144), no FeatureEmbeddingW/Concatenate/user-proto/W-matrix remnants in the branch, entire score = u·t* via the RecSys dot (rec_sys.py:114) with no other item-side path (bias disabled fleet-wide, `base_param use_bias=0`, hyper_params.py:11); §B shapes table matches the code exactly, including the (…, F+1) ID-column concat at offset `n_features` (feature_extractors.py:176-180); parameter accounting checks out ((426+13,651)·d with ID, 426·64 ≈ 27K ≈ 3% of the 874K item table, 13,651→426 ≈ 32×, F+1 = 6 gathers with the canonical 5 fields, which the config lists in the S0.5 order); F-DC01-01 discipline honored (`use_id_feature` is a fixed flag in two separate full-search configs `feature_item_proto`/`_noid`, plus `_f0`, each a one-knob deep copy — hyper_params.py:247-255); guarded, non-serializing injection seam in both `trainer._build_model` (trainer.py:67) and `tester._build_model` (tester.py:50) with the non-persistent `feature_ids` buffer and alignment/NaN guards (feature_ids.py:48-77, feature_extractors.py:143-144, 154); deviation 6 landed as specced — `drop_cold_id_rows` zeroes cold ID rows through the nested `PrototypeEmbedding.embedding_ext` (cold_eval.py:142-172), the former `_ids` refusal is lifted and replaced by the post-drop abort (cold_eval.py:315-320), `use_bias≠0` is refused (cold_eval.py:62-66), and the D4 popularity strata / tie-block / cold-vs-cold-primary machinery the §3.6 interpretation rule presupposes is present.
>
> VERDICT: The implementation is a faithful build of the governing fI-ProtoMF spec. Every structural, shape-level, and convention-level commitment I could check against code holds exactly — host topology, single grounded score path, composition semantics, ablation-arm discipline, injection/checkpoint hygiene, and the landed cold ID-drop with its loud failure mode. The only substantive finding is the init-ownership asymmetry (item 1), which does not touch the mechanism but slightly perturbs the spec's like-for-like and keystone comparison instruments and is undocumented in the spec itself; item 2 is a sequencing nuance to reconcile against the S0.3 spec; everything else the spec commits to at implementation level is either present or explicitly staged to a later scrutiny step. No major or blocker divergences found.

### 2. Response (point by point)

**P1 — init-ownership asymmetry: conceded as a SPEC-DOCUMENTATION gap;
the numeric concern verified void; trivial fix applied.** First-party
verification (this session): `nn.Embedding.reset_parameters` and
`general_weight_init`'s Embedding branch are BOTH `init.normal_` N(0,1) —
the host and fI item tables start from the *same distribution*; only RNG
stream consumption differs, which is inherent to any architecture change,
moot under the charter (no seed-matched pairwise runs anywhere in the
comparison design), and exactly why the keystone proves equivalence by
weight-copy rather than by seed. The decision WAS pinned and gate-ratified
(SC.3 re-run §1) — but only in the dossier and a code comment; the
auditor is right that the spec itself was silent. **Fix (trivial, applied
now per fix policy):** dated init-ownership note added to design doc §B
recording ownership + the distribution-parity argument. Ledger: F-DC01-10
(trivial, one-liner).

**P2 — drop-before-warm-test sequencing: rebutted as designed-and-required;
disposition recorded, no change.** In the variant-trained model the cold
items' ID rows are untrained-random noise; scoring them on those rows in
ANY context — including as sampled negatives of the variant warm test or
inside the tie-block diagnostic — is precisely the F-S0-07/F-S0-13
silent-depressant class the runner exists to prevent. The drop-at-load
sequencing enforces the ratified global invariant ("no cold item is ever
scored on an untrained ID row"), matches the model's defined deployment
convention (unseen items are scored feature-only), and is the SAME
sequencing already audited and accepted for the S0.7 `lightfm_tags_ids`
reference rows — fleet-uniform, hence GR7-fair. The auditor's residual
point (the F-S0-11 warm-delta for `_ids` runs compares a drop-applied
variant model against a canonical model) is noted for SC.8's
charter-compliance reading: the warm rows themselves are unaffected
(warm ID rows untouched — t11 pins bit-identity of warm outputs); only
cold-as-negative rows differ, ~their catalog share, and feature-only is
the honest convention for them.

**P3 — loss framing: conceded as cosmetic, no change.** §A restates the
host paper's Eq. 10; the operative commitment (host search space verbatim,
F-DC01-01) is what the config implements. The Eq. 9 regularizers ride
`get_and_reset_loss()` under all three searched losses.

### 3. First-party audit record

- **Contract conformance:** fI uses `PrototypeEmbedding` exactly as the
  host `item_proto` does (reg losses accumulate in forward, drained via
  `get_and_reset_loss()` in `RecSys.loss_func`); user side is host
  `Embedding` (base-class zero loss). Both build seams inject correctly;
  DataParallel/device handling untouched host code; `feature_ids` buffer
  non-persistent (moves with device, absent from checkpoints — i04).
- **Optimization parity:** composed lookup = one batched gather over
  (B, 1+n_neg, F+1) + sum — no Python loops, no full-catalog pass (i03
  additionally pins zero gradient on unused vocab rows). K_t·d cosine
  dominates step cost, ≈1.05–1.1× vs host (§4 accounting re-checked).
- **All-zero composed vector (the SC.1a-flagged robustness look):
  dispositioned document-only.** torch's cosine eps makes cos(0,p) = 0 →
  t* = 1 (neutral, finite); gradient at exactly-zero is finite (eps-inflated
  ~1e8 — but the point is unreachable: every summed row exactly zero
  simultaneously is measure-zero under float training; the cold path is
  eval-only; the host's free embedding carries the identical theoretical
  corner). No eps/assert added — a guard would protect nothing real and
  touch the scored path.
- **Test adequacy vs claims spec:** C1/C2/C6/C8 (check_claims.py,
  composition-level, carry), C3′/C4′/C5′ (check_claims_fi.py), C9
  (check_claims_c9.py) all have committed green checks; keystone i02′
  bit-identical; integration i01/i03′/i04/t11 cover build-forward-backward,
  single-path gradient, checkpoint round-trip, cold drop e2e. Adequate.
- **Feature entry end-to-end:** canonical 5 in config (S0.5 order) →
  `build_feature_ids` (NaN-guarded, V=426 on V1 — s0/t02) → injection seam
  (non-serializing, both seams) → `FeatureEmbedding` (alignment assert).
  Verified as a chain; no gaps.

### 4. Outcome

No code fixes required. One trivial spec fix applied (F-DC01-10: init-
ownership note into design doc §B). P2 disposition recorded for SC.8's
compliance reading. Candidate implementation stands **faithful to the
governing spec** by independent audit + first-party verification.

### Gate (SC.4)

**Status: OPEN — awaiting user review.** Decisions requested:
1. Accept the P1 disposition (spec-silence conceded; distribution-parity
   verified; trivial doc note applied as F-DC01-10).
2. Accept the P2 rebuttal (drop-at-load is the invariant working as
   ratified; fleet-uniform; SC.8 note recorded) — no change.
3. Accept the P3 cosmetic disposition — no change.
4. Accept the first-party audit incl. the document-only disposition of the
   all-zero-composition corner.
5. Proceed to **SC.5** (explanation scrutiny: comparability matrix,
   attribution gates, the shared user-perspective breakdown renderer build
   per protocol v1.2).

**Gate closure (2026-07-12, same sitting): CLOSED — user accepted all five
decisions ("we can accept all of those"). P1–P3 dispositions stand as
recorded (F-DC01-10 applied); first-party audit accepted; → next step:
SC.5 (explanation scrutiny under protocol v1.2 — comparability matrix,
attribution gates, shared breakdown renderer).**

---

## SC.5 Explanation scrutiny (2026-07-12) — pre-gate analysis & build proposal

**Scope:** protocol v1.2 SC.5 section re-read (incl. the "why this step carries
thesis weight" preamble — the artifacts designed here are the standing
host-vs-variant comparison instrument for every lineage stage). Manifest loaded:
the full `utilities/explanations/` tree (pipeline, loader, items_info, all five
accessors, all five explainers, naming layer, `feature_readout.py`,
`attr_readout.py`), `utilities/explanations_utils.py` (original-paper baseline),
`Master/scripts/run_combo.py` explanations wiring, the three deferred demo
scripts + the dev-checkpoint artifact
(`dc_checks/dc01/item_prototype_explanations.md`), design doc §C/§3.4 as
amended. Ledger re-check at session start: no `open` findings anywhere; none
stale. Per GR1 gate mechanics this is a fix/build-bearing step: findings + the
renderer build proposal are decided at THIS gate; approved work is applied
after it and the artifact finalized before SC.6.

Verified enabling fact: `cosine_type` is fixed `'shifted'` in every config
(12/12 occurrences in `confs/hyper_params.py`) — the pipeline's
`_shifted_cosine_sim` post-hoc assumption matches the scored model for every
in-scope run.

### 1. Comparability matrix

Columns: the original ProtoMF models vs fI. ✓ = produced by the live pipeline;
— = absent (justification in the notes; every absence is either
host-structural, algorithmically impossible for that model, or an approved
exclusion). Baseline rows (`lightfm*`, `mf`, popularity) have no
prototype surfaces; they enter at row 10, where the shared renderer defines
their comparison cells.

| # | artifact | `item_proto` (host) | `user_proto` | `user_item_proto` | `feature_item_proto` (fI) |
|---|---|---|---|---|---|
| 1 | top-k items per **item** prototype (CSV) | ✓ | — (a) | ✓ | ✓ (post-hoc route on the composed q_i cloud — kept deliberately as the R2 comparison point) |
| 2 | top-k items per **user** prototype (CSV) | — (a) | ✓ | ✓ | — (a) |
| 3 | post-hoc naming, item side (lift over top-k) | ✓ | — (a) | ✓ | **— currently skipped** → F-DC01-13 (b) |
| 4 | intrinsic naming, item side (cos(e_f, p_k)) | — (c) | — (c) | — (c) | ✓ |
| 5 | naming transparency dumps (names + full stats CSVs) | ✓ | ✓ | ✓ | ✓ |
| 6 | TSNE, item-prototype space (named) | ✓ | — (a) | ✓ | ✓ |
| 7 | TSNE, user-prototype space (named) | — (a) | ✓ | ✓ | — (a) |
| 8 | weight_viz (paper's per-recommendation bars) | — (d) | — (d) | ✓ | — (d) |
| 9 | feature_small_multiples (item side) | ✓ | — (a) | ✓ | ✓ |
| 10 | per-feature exact shares + global profile (`feature_readout`) | — (c) | — (c) | — (c) | ✓ |
| 11 | **user-perspective recommendation breakdown** (SC.5.4) | ○ build | ○ deferred (e) | ○ deferred (e) | ○ build |

Justifications: **(a)** host-structural — that prototype side does not exist on
the model (fI's user-side absences exactly mirror the `item_proto` column:
host-parity, GR7-symmetric). **(b)** the pipeline's `elif` routes fI to
intrinsic-only naming; the post-hoc route would run fine on fI
(`item_to_item_proto_sim` exists) and running BOTH is the mechanized
profile-validity comparison F-DC01-02 committed for SC.8 — proposed as
F-DC01-13, the one unjustified gap found. **(c)** algorithmically impossible —
requires learned feature-value rows sharing the prototype metric space (the
candidate's novelty; declared, not a gap). **(d)** projection branches don't
exist (fI: SC.3 re-point, code-comment-documented; single-branch hosts never
had them; the paper's own artifact is UI-only); the breakdown renderer (row 11)
is the shared replacement that covers every model. **(e)** see §4 scope
proposal — slots for models outside dc01's stage comparison are deferred to
the lineage stage that consumes them (fU / fUfI), the frame being
model-agnostic.

### 2. Algorithmic-novelty attribution gates (GR7)

Per new/altered artifact: *could original ProtoMF have produced this exact
artifact?*

- **Intrinsic naming route** (`name_prototypes_intrinsic` +
  `intrinsic_naming_config`): **NO.** Requires learned per-feature-value
  embeddings living in the same metric space as the prototypes; only the
  composition q_i = Σ e_f puts them there. *Enabling algorithmic property
  (thesis sentence): because items are sums of feature-value rows, feature
  values and prototypes are co-embedded, so a prototype's meaning is read from
  parameters alone — no data pass, no top-k proxy.* → genuine novelty.
- **Per-feature exact shares** (`per_feature_shares`): **NO.** Exactness is the
  linearity of q_i in its rows under the cosine's joint normalization — no
  free-embedding model has composing rows to decompose over. → novelty.
- **Global prototype profile** (`global_prototype_profile`): **NO** (same
  co-embedding requirement). → novelty.
- **Naming layer, transparency dumps, TSNE labels, small multiples**: shared
  pipeline work and ALREADY serving every model (the post-hoc route consumes
  only weight matrices every model exposes; registered for all five model
  types). Nothing here is candidate-gated. → **no backport due**.
- **Breakdown renderer** (row 11): shared by construction — it IS the
  fairness-contract instrument; each model's mechanism plugs into declared
  slots.

**Verdict: no backports required → no retro-invalidation sweep this pass.**
The baseline's post-hoc naming route was steel-manned in earlier shared work
(lift/raw scoring, transparency stats, full reconstructability) and serves all
models identically.

### 3. New interpretability axis + visualization

**The axis (one name): parameter-space feature grounding with exact score
attribution.** Two inseparable halves: (i) prototype meaning is read from
parameters alone (cos(e_f, p_k) — no interaction data, no top-k proxy at
read time); (ii) every activation splits exactly into named-feature summands,
and on the I host these cover **100% of the item-discriminating score**
(§C rank-inert baseline, C4′). The host has neither: its prototype meaning is
reconstructed post-hoc from top-k items, and its activations are atomic.

**Proposed visualizations (both feed SC.9 → both implemented):**
1. **Prototype cards** — the standing "what do the prototypes look like"
   figure (v1.2 question 1): one compact card per prototype; fI card =
   intrinsic profile bars (top cos values, signed, lockstep-grouped); host
   card = post-hoc lift descriptors + top-k exemplar items. Same card
   geometry for both models; the mechanism difference (parameters-only vs
   data-pass) is stated in the card header — the asymmetry IS the finding.
   Grid of K_t cards per model, two grids side-by-side.
2. **The renderer's grounding zoom** (§4) — the standing "why this
   recommendation" figure (v1.2 question 2).

### 4. Breakdown renderer — build proposal (SC.5.4, mandatory build)

**Contract (fairness):** one renderer, one layout, one visual language; every
model gets the identical frame (header: user, item + metadata line, model,
total score; left panel: per-prototype contribution bars, signed, sorted,
named; right panel: mechanism zoom; footer: fixed-order disclosure lines).
Model-specific content enters ONLY through declared slots. Advantages may come
from the algorithm, never from rendering effort. Non-interactive, thesis-figure
quality, compact for repeated use (target ~7×4.5 in, top-6 prototypes,
top-6 zoom lines).

**Exactness self-check (code contract):** bars are computed from the model's
own forward quantities via the accessor; at render time the renderer asserts
Σ(bars) + disclosed baseline == S(u, i) to float tolerance — every rendered
figure is self-verifying, loudly.

**Per-model slot map (proposed to build now):**

| model | contribution panel | naming route | mechanism zoom | disclosure footnotes |
|---|---|---|---|---|
| `feature_item_proto` | s_k = u_k·(t*_k−1) bars + baseline Σ_k u_k·1 disclosed (rank-inert, shown never absorbed) | intrinsic | per-row shares c_{r,k} of the top prototype: signed bars, ID row visually set off, **feature-explained fraction** line (M3, binding), lockstep groups + determinacy annotations (F-DC01-06) | joint normalization (5b-5); catalog common-mode caveat (§C); u_k = canonical (min-norm) representative (F-DC01-08); per-line factorization u_k(CF) × t*_k(grounded) (S0.2 obligation, fI form) |
| `item_proto` (host) | **identical math** — the host has the same shifted-cosine item branch and free u, so the u_k·1 / u_k·(t*_k−1) split applies verbatim | post-hoc (lift) | top-k exemplar items of the top prototype, declared post-hoc | u_k canonical-representative (same gauge anatomy — GR7-symmetric, per F-DC01-08 disposition); naming-is-post-hoc declaration |
| `lightfm_tags` / `_ids` | exact per-feature bars u·e_r (no prototype layer — that absence is the comparison) | n/a | the bars ARE feature-level already; `_ids`: ID row set off | no-prototype declaration; joint scale honesty |
| `mf` | total score only | n/a | "no decomposable explanation surface" stated in-frame | the honest-asymmetry row |
| popularity | total + popularity percentile | n/a | "global popularity rank" statement | non-personalized declaration |
| `user_proto`, `user_item_proto` | **deferred to fU / fUfI stages** — outside dc01's stage comparison (staged-bar decision, §G R6′); the frame is model-agnostic, adding them later is slot work only | | | |

(Note: the u_k·1 baseline split is applied only where mathematically justified
— on I-hosted models, where the +1 shift is exactly rank-inert; a footnote
states this so no reader mistakes layout parity for math parity.)

**Files:** `utilities/explanations/breakdown.py` (renderer + slots + CLI
`--results-dir --user [--item]`, item defaulting to the model's top-1
recommendation); `utilities/explanations/determinacy.py` (value-pair
normalized-decoupling from `items_info` at runtime — same math as
`dc_checks/dc01/decoupling_mass.py`, no CSV dependency; exact-lockstep groups
+ continuous per-pair annotations); `utilities/explanations/explainers/breakdown.py`
(registered explainer: default sample users → top-1 item, so auto-pipeline
runs render N default breakdowns); `utilities/explanations/explainers/proto_cards.py`
(prototype cards, §3). Tests: `dc_checks/dc01/t12_breakdown_renderer.py`
(exactness assert incl. adversarial check, slot coverage, disclosure presence,
determinacy grouping on a synthetic lockstep pair, e2e on the fI and
`item_proto` toy checkpoints). Demo scripts `readout_demo.py` /
`readout_artifact.py` / `run_explanations_hm.py` get a superseded-by header
(kept — preservation rule).

**Threshold discipline (learnings ledger):** grouping happens ONLY at exact
lockstep (decoupling = 0, a provable gauge — threshold-free); all other
associations render as continuous determinacy annotations. Display floors
(min_score, top-N) are presentation knobs, carrying no claim. The decoupling
proxy is used as **scrutiny-phase working basis only** (F-DC01-06 restriction:
no thesis quotes; the tracked "re-check at SC.5" is hereby executed —
annotations cite the proxy as working input).

**Datasets in scope (v1.2 pin):** hm_1_month (fI exists only where item
features do). The renderer itself is dataset-agnostic for CF models.

### 5. Gating-consistency check

- **`run_combo.EXPLAINABLE_MODELS` = {item_proto, user_proto, user_item_proto}
  still excludes `feature_item_proto`** while
  `pipeline.EXPLAINABLE_MODELS` includes it — the deferred P9, due exactly
  here → **F-DC01-11** (the SC.3 known-deferred note anticipated this).
  `attr_item_proto` stays excluded until dc02's SC.5 (no cross-candidate
  work).
- Explainer `supports()` sets vs accessor flags, walked for fI: weight_viz
  excludes fI ✓ (has_projections=False); naming/tsne/top-k/small-multiples
  include fI ✓ (has_item_prototypes=True); user-side artifacts skip ✓
  (has_user_prototypes=False); intrinsic route keyed on
  `has_intrinsic_item_grounding` with the dc02 attr flag checked first ✓
  (fI: attr flag absent→False). All pinned by t10. **Consistent; no silent
  drops** (pipeline failures print loudly, naming failure degrades visibly).
- Observation, no change proposed: `supports()` name-lists duplicate accessor
  capability flags — a maintenance seam, but host idiom and t10-pinned.

### 6. Findings (bundled proposals — decided at this gate)

- **F-DC01-11 [minor]:** wire `feature_item_proto` into
  `run_combo.EXPLAINABLE_MODELS` (auto-explanations after training); pin in
  t10.
- **F-DC01-12 [minor]:** the gated rendering disciplines (F-DC01-06 grouped
  shares + determinacy annotations; F-DC01-08 canonical-representative
  disclosure, both models; M3 feature-explained fraction; 5b-5 baseline/
  normalization honesty; S0.2 per-line factorization) exist only as doc
  commitments — nothing in the live pipeline implements them. Proposal: the
  §4 build (determinacy module + naming lockstep-merge + renderer disclosure
  slots) is their implementation.
- **F-DC01-13 [minor]:** enable the post-hoc naming route on fI alongside the
  intrinsic one (dual-route, artifacts nested under `lift/` vs `cosine/`) —
  the mechanized input for the SC.8 profile-validity spot-check (F-DC01-02).
- Housekeeping: commit the untracked
  `dc_checks/dc01/decoupling_out/decoupling_value_level.csv`; superseded-by
  headers on the three demo scripts.

### 7. Observations recorded for SC.8 (working evidence, no action here)

The dev-checkpoint artifact (`item_prototype_explanations.md`, illustrative
only) already exhibits exactly what the committed SC.8 geometry instruments
will quantify: prototypes #6 and #9 with IDENTICAL intrinsic profiles (M4
overlap/collapse signal), t* = 2.000 saturation on most-activating items, and
"Solid" appearing in 7/11 profiles (M2 omnipresence). The renderer and the
SC.8 spot-check will surface these on the real scrutiny checkpoints.

### Gate (SC.5)

**Status: OPEN — awaiting user review.** Decisions requested:
1. Accept the comparability matrix incl. all absence justifications (§1).
2. Accept the attribution-gate verdicts: three genuine novelties, no backports
   → no retro-invalidation sweep (§2).
3. Accept the axis naming + the two standing figures (prototype cards +
   grounding zoom), both to be implemented (§3).
4. Approve the renderer build as designed (§4) — incl. the **slot-scope
   decision**: build fI, item_proto, lightfm×2, mf, popularity now; defer
   user_proto / user_item_proto slots to their lineage stages.
5. Approve F-DC01-11 (run_combo wiring), F-DC01-12 (disciplines build),
   F-DC01-13 (dual naming route on fI), + housekeeping.
6. On approval: build lands supervised, tests green, candidate index →
   **hardened**, artifact finalized before SC.6.

### Gate closure + application round (2026-07-12, same sitting)

**Gate CLOSED — decisions as negotiated in the walkthrough:**
1. Comparability matrix + absence justifications: **accepted**.
2. Attribution verdicts (three novelties, no backports, no sweep): **accepted**.
3. Axis + both standing figures (prototype cards, grounding zoom): **accepted, built**.
4. Renderer design + slot scope (defer user_proto/user_item_proto to fU/fUfI): **accepted**.
5. Findings: **F-DC01-11 approved as proposed. F-DC01-12 approved RE-SCOPED (user
   decision, deep-dive): arithmetic honesty stays in the rendered artifacts (+1
   baseline shown, signed shares, ID-row line + feature-explained fraction); the
   effect-epistemics layer (determinacy annotations, lockstep grouping,
   canonical-representative/gauge and per-line-factorization footnotes) does NOT
   enter rendered explanations — re-routed to the thesis's dedicated hidden-effects
   section with experiments designed at writing time; consequently no determinacy
   module and no naming lockstep-merge were built. Dated re-route edit notes applied
   to design doc §3.4 F-DC01-06 block, §I.1, §C (so spec-vs-code audits do not
   re-flag). F-DC01-13 approved as a scrutiny diagnostic — thesis use of the route
   comparison deliberately left open.** Housekeeping approved.

**Build record (post-gate application):**
- `utilities/explanations/breakdown.py` — shared renderer: five slots (fI incl.
  ablation arms via CLI dispatch, item_proto, lightfm×2, mf, popularity), identical
  frame/visual language, per-artifact exactness SELF-CHECK (parts asserted against
  the model's own forward score before writing; zoom asserted against its parent
  bar), figure (PNG) + markdown companion, CLI
  (`python -m utilities.explanations.breakdown`), loader/items_info data-dir
  overrides.
- `utilities/explanations/explainers/breakdown.py` — auto-invoked for the stage
  pair (fI + item_proto): sample users → top-1 item → breakdown into the
  explanations dir.
- `utilities/explanations/explainers/proto_cards.py` — the question-1 standing
  figure: per-prototype profile cards, same geometry for intrinsic and post-hoc
  routes (post-hoc cards additionally list their exemplar evidence — the declared
  route asymmetry); outlier-robust shared scale.
- `pipeline.py` — dual naming route (F-DC01-13): post-hoc lift naming CSVs land in
  `explanations/lift/` beside the intrinsic `cosine/` pass, fI only.
- `run_combo.py` — `feature_item_proto` in EXPLAINABLE_MODELS (F-DC01-11; headline
  model only, arms stay CLI-reachable per host convention).
- Commits: **16070a0** (build + wiring), **aafe733** (tests + housekeeping).

**Figure-scrutiny record (user directive, mid-build: "inspect graphical outputs
very carefully"):** every figure type was rendered on a toy and inspected at
full resolution; defects found and fixed before landing — value-label/tick-label
collisions on large negative bars (margins), ID-row label truncation, zoom-title
truncation mid-formula (both slots), figure-level text wrap overflowing panels
(explicit per-panel wrapping), near-blank 4.6-inch canvas on statement-only slots
(compact 2.0-inch frame), popularity "top X%" off-by-one-rank (rank 1/202 read
"top 0.0%", now 0.5%), cards' shared x-scale vulnerable to off-card lift outliers
(scale over shown entries only), plural nit. Final renders of all five breakdown
slots + both card routes + the md companion verified clean; exactness identities
re-checked by hand on the rendered numbers.

**Test record:** dc01 t01–t12 (t12 new: 37 checks) + i01/i02′/i03′/i04/i06a,
s0 t01–t06 + i01–i02, dc02 t01–t08 + i01–i06, decoupling selftest — **ALL PASS**.
t05's stale "not explainable yet" pin flipped to the F-DC01-11 semantics (the one
expected regression, analogous to s0/t06 at F-S0-07(b)).

Ledger: F-DC01-11/12/13 → **fixed (16070a0)**. Candidate index → **hardened**.
→ Next step: **SC.6** (cluster-run plan — requires the S0.7 reference fleet's
frozen table; fleet still in flight on LEO5 at closure time).

---

## ⏸ STRATEGIC PAUSE (2026-07-12, user decision at the SC.5 close-out sitting)

dc01's scrutiny pauses HERE, after SC.5 (pre-run block complete; candidate
hardened). **Resume condition: cluster progress** — when the S0.7 reference
fleet lands: results into `replication_report.md` + the S0 dossier's frozen
table (S0.7 part 2), then dc01 resumes at **SC.6** (run plan under the charter).
The SC.6-needs-the-frozen-table dependency was assessed at this sitting as a
sequencing safeguard (pre-spend sanity + precondition re-check), medium weight —
honored as written.

**Interposed work (user direction):** a directed design-candidate cycle for the
lineage's user-side stage — **fU on the U-ProtoMF host** — runs next, in a fresh
context, under `Master/docs/design_candidates/fu_cycle_brief.md` (modification
layer M1–M7 on the DC protocol: literature de-emphasized, mathematical
soundness amplified incl. a mandatory hidden-effects pre-audit seeded by this
dossier's F-DC01-06/08/09 anatomies, explicit license to cannibalize
dc02/03/04, comparison-instrument continuity per protocol v1.2). dc02–dc04
scrutiny is deferred; their candidate files become licensed quarry. Learnings
from the fU cycle that touch this paused work are filed back as `[fU-cycle]`
learnings-ledger entries + dated resume notes here (brief M7).

---

## SC.6 Cluster-run plan (2026-07-15)

> Resume from the strategic pause: the resume condition (S0.7 part 2 — the
> frozen reference table) was met 2026-07-15 (S0 gate closed, S0 COMPLETE).
> Executed under protocol v1.2 in re-entry form (`/scrutiny step6 dc01`);
> manifest re-loaded, SC.6 section re-read.

### 0. Preamble — precondition re-check (per the dossier header's declaration)

- **S0.7 part 2 landed:** frozen reference table exists (s0_foundation.md,
  2026-07-15) — 7 canonical hyperopts + 7 cold retrains + 7 cold evals + D3/D4,
  single seed 38210573, dev tier, C8 manifests. The fI stage bar is
  **`item_proto` warm 0.4938 / 0.2631 (HR@10/N@10), cold-cold 0.0000 (tie
  floor, R-a), +kNN 0.2358**. Feature baselines: `lightfm_tags` 0.4582 warm /
  0.4705 cold-cold; `lightfm_tags_ids` 0.5087 / 0.5080. Interpretation rules
  R-a..R-g accompany all readings; two-tier numbers policy applies (fleet/dev
  numbers = working numbers, not thesis numbers).
- **Open-findings re-check:** no `open` findings remain in scope — F-DC01-01..13
  all fixed/parked-as-fixed; F-S0-01..13 fixed/wontfix. The wontfix conventions
  that constrain THIS step: F-S0-10 (every submission line carries explicit
  `--profile dev` — operator discipline, no code default), F-S0-02/12 (eval
  regime + sigmoid conventions, LEO5-only runs).
- **Learnings applied here:** file-level dataset-presence checks before
  feature-aware submissions (ml-1m `item_features.csv` scp'd during the
  walltime probes, commit 14efb96 — re-verify at submission); smoke-one-before-
  batch; one `/leo5-submit` per run (hard gate).
- **Walltime inputs:** `Master/temp/dc01_dc05_walltime_probes_2026-07-15.md` —
  fI is compute-free over its host (×1.04 hm, ×0.95 ml-1m); anchors item_proto
  hm dev fleet 2h08m.

### 1. Dataset pin (decision requested at this gate)

Per charter C2 (as amended 2026-07-12) the per-candidate dataset list pins at
SC.5/SC.6; dc01's SC.5 matrix was hm-scoped and left the run pin to here.
**Proposed pin: `hm_1_month` (+ `hm_1_month_cold`) primary; `ml-1m` (warm) as
the second quantitative testbed.**

- `hm_1_month`: the charter's primary testbed — thin histories × rich
  vocabulary, the 67.5% signature-twin regime where the ids-vs-noid bundle
  (§3.6 interpretation rule) and the twin-angle instrument (F-DC01-09) have
  their object.
- `ml-1m` (warm): the complementary regime (heavy × coarse, 3.8% twins) — the
  ids-vs-noid gap there is nearly twin-free, so the cross-dataset read is the
  cheapest separator of gap component (1)/(1b) (popularity/twin effects,
  hm-concentrated) from component (2) (taste-structure loss, should appear on
  both). Written reason per C2: charter's own second-testbed amendment; the
  comparison instrument (renderer/cards, F-DC05-19 tags naming) is already
  ml-1m-capable.
- **`ml-1m_cold`: NOT pinned for dc01** (proposed). The bags-aware kNN/tie
  machinery is explicitly deferred to the "ml-1m_cold enablement work"
  (F-DC05-16 disposition: "not part of any candidate's SC routine"), so CF
  reference rows could not carry their kNN column — the comparison would be
  half-defined. dc01's quantitative cold claim rides H&M (full machinery,
  2,730-item pool); ml-1m cold also carries the weaker "interaction-hidden,
  not new-release" epistemic status (S0.3 addendum A4). If later wanted, it is
  one enablement work-package + one retrain per arm — nothing here precludes
  it.

### 2. Run manifest

All runs: branch `feat/scrutiny` (commit hash recorded at submission), seed
38210573, uniform-99 negatives, `use_bias=0` (explicit in `base_hyper_params`),
`feature_fields: 'canonical'` (resolves per dataset: hm → S0.5 five, V=426,
fixed layout; ml-1m → genres+tags@0.8 bags — resolved literals land in the C8
manifest), model selection on val `hit_ratio@10`. Hyperopts: dev profile
(30 trials / 60 epochs / patience 7 / ASHA grace 4), **explicit `--profile
dev`**. One run = one job = one `/leo5-submit`.

**Wave A — dc01 hyperopts, hm_1_month (primary):**

| # | Run | Config | Est. | `--time` |
|---|---|---|---|---|
| R1 | fI headline | `feature_item_proto` (ids, canonical 5) | ~2h13m | 8:00:00 |
| R2 | ablation A | `feature_item_proto_noid` (C7 fork) | ≤R1 | 8:00:00 |
| R3 | ablation B | `feature_item_proto_f0` (F=0+ID keystone arm) | ~2h08m | 8:00:00 |

R3's role (recorded reading): architecturally `item_proto` through the fI code
path (keystone i02/C5′ pins bit-identity at equal config), but a seed-matched
hyperopt is NOT init-stream-identical to `item_proto`'s (SC.4 §B init-ownership
record, F-DC01-10) — so R3-vs-frozen-`item_proto` is the **single-seed noise
yardstick at identical architecture**: it calibrates how much of any fI-vs-host
delta the wrapper/init path alone can produce, exactly what C4's single-seed
budget otherwise leaves unquantified. Read at SC.8 alongside R1-vs-bar.

**Wave B — cold retrains + evals, hm_1_month_cold (S0.3 §6 convention —
depends on Wave A best configs):**

| # | Run | Source config | Est. | `--time` |
|---|---|---|---|---|
| R4 | fI cold retrain | R1 best, `--retrain-config`, `--profile dev` | ≤2h | 4:00:00 |
| R5 | noid cold retrain | R2 best, same convention | ≤2h | 4:00:00 |

Then cold evals (`python -m utilities.cold_eval`, login-node A30 per the
small-workload policy — post-hoc measurement on trained checkpoints, the same
placement the S0.7 wave used) with `--canonical-results-dir` pointing at the
R1/R2 dirs (F-S0-11 warm-delta row). R4 exercises the ID-row drop end-to-end
on a real run for the first time (F-S0-07(b) unblocked, t11-pinned;
`config_expects_id_drop` aborts loudly if it cannot apply). R5 is native
feature-only (drop no-op, pinned). **No f0 cold retrain:** F=0 + ID-drop
leaves an empty representation — the arm is degenerate at cold by
construction, and its cold information is already the frozen `item_proto`
row.

**Wave C — dc01 hyperopts, ml-1m (parallel to Wave A):**

| # | Run | Config | Est. | `--time` |
|---|---|---|---|---|
| R6 | fI headline | `feature_item_proto` | ~1h10m | 4:00:00 |
| R7 | ablation A | `feature_item_proto_noid` | ~1h10m | 4:00:00 |

No R3-analogue on ml-1m (the noise yardstick needs measuring once, on the
primary testbed; the f0 keystone itself is layout-tested bit-identically —
S0-build-ext keystone (a)).

**Wave D — ml-1m reference-fleet rows (charter C6 lazy addition — this run
plan is the first to cite them; they are fleet rows, computed once, not dc01
property):**

| # | Run | Config | Role for dc01 | Est. | `--time` |
|---|---|---|---|---|---|
| R8 | `item_proto` ml-1m | host search space | the fI stage bar on testbed 2 | ~1h14m | 4:00:00 |
| R9 | `mf` ml-1m | fleet minimum | context row | ~1h | 4:00:00 |
| R10 | `lightfm_tags` ml-1m | bags layout | feature-only baseline (§3.6/§D) | ~1h | 4:00:00 |
| R11 | `lightfm_tags_ids` ml-1m | bags layout | strong feature-aware baseline | ~1h | 4:00:00 |

Not cited here, therefore not submitted under this plan: `user_proto` ml-1m
(dc05's stage bar — its SC.6 cites it), popularity reference (cold-context row;
follows the ml-1m_cold enablement), all ml-1m cold retrains (§1). The existing
prod-profile ml-1m replication rows (e.g. item_proto 0.5723) are replication
context only — budget-unequal (100/100 vs dev 30/60), pre-scrutiny lineage —
and are never compared against dev rows (two-tier discipline).

**Post-run measurements (SC.8 scope, named here for completeness, all
login-node small workloads on the produced checkpoints):** D3/D4 readouts for
R1/R2/R3 (the §3.6 gap rule needs D4 popularity strata); the committed SC.8
instrument set — twin-angle distribution (F-DC01-09), gauge-mass/activation-
spectrum decomposition (F-DC01-08), the M1/M4 geometry set (F-DC01-05:
t*-spread, prototype pairwise cosine, profile overlap, per-value share vs
frequency); profile-validity spot-check via the dual naming route
(F-DC01-02/13); rendered-explanation spot-check incl. ≥3 shared-renderer
breakdowns fI-vs-item_proto (protocol SC.8.4). Auto-explanations for R1/R6
ride run_combo (F-DC01-11).

### 3. Charter compliance (clause by clause)

- **C1 single lineage:** all runs from `feat/scrutiny`; commit hash recorded
  per C8 manifest; SC.6 artifact committed before submission so the hash
  contains it.
- **C2 datasets:** hm_1_month + hm_1_month_cold primary; ml-1m per the
  second-testbed amendment with the §1 written reason; no other datasets
  (hm_3_month not invoked; amazon2014 out of feature-aware scope).
- **C3 eval protocol:** temporal LOO, 1+99 uniform, HR/NDCG@{1,3,5,10,50},
  val `hit_ratio@10` selection, `use_bias=0` fleet-wide (explicit key,
  absent-key hazard checked), cold eval per S0.3 (cold-vs-cold primary,
  single-config retrain, training-negative exclusion, warm-val stopping);
  D3/D4 as secondary readouts at SC.8.
- **C4 budget & seeds:** dev profile 30/60/7/4 on every hyperopt; single seed
  38210573; no multi-seed, no split repetitions; one run = one job; envelope
  check in §4.
- **C5 features:** `'canonical'` sentinel → per-dataset canonical sets
  (S0.5 five on hm; B5's genres+tags@0.8 bags on ml-1m); no price_band, no
  product_code; no per-candidate deviation.
- **C6 baselines:** hm comparisons run against the FROZEN table — no hm
  baseline is re-run. ml-1m fleet rows R8–R11 enter under the lazy-addition
  amendment (first-citing plan), each its own submission, computed once for
  all future consumers.
- **C7 ablation discipline:** exactly the design doc's committed arms —
  `use_id_feature` fork (R2/R7) + F=0 keystone arm (R3); no exploratory
  sweeps; ablation interpretation pre-committed (§3.6 rule + R3's yardstick
  reading).
- **C8 manifests:** every run records model/config, resolved feature fields +
  layout, `use_id_feature`, negatives, seed, profile, dataset(+variant),
  branch+commit, `use_bias`. SC.8 audits against this list.

### 4. Compute estimate vs the envelope

11 SLURM jobs: Wave A ~6.6h + Wave B ~3h + Wave C ~2.3h + Wave D ~4.3h ≈
**16–17 GPU-hours total; largest single job ~2h13m** (limits 4–8h, sized from
measured anchors ×2+ margin). Envelope (vault 2026-06-07_2310): soft flag
>1 day — nothing within an order of magnitude; hard limit >10 days —
irrelevant. No flags, no mitigations needed. Login-node cold evals + readouts:
minutes each on the A30s.

### 5. Preconditions & submission discipline

1. **Commit + push this artifact; `git pull` on LEO5** (cluster-sync protocol;
   laptop = source of truth). Verify HEAD on LEO5 == submitted hash.
2. **File-level data checks on scratch (learnings rule — files, not dirs):**
   hm_1_month + hm_1_month_cold splits (existed for the S0.7 wave, re-verify);
   ml-1m: all 6 split files + `item_features.csv` (scp'd at the walltime
   probes; re-verify). ml-1m_cold scp stays deferred (not pinned).
3. **`/leo5-load` before the first submission** (partition choice; congestion
   policy: submit as early as this gate permits).
4. **Smoke-one-before-batch:** R1 is submitted first, alone, and verified
   end-to-end (trials starting, W&B rows tagged dev, C8 manifest written,
   explanations artifacts produced) before R2–R11 are submitted. The walltime
   probes covered the retrain-config path on both datasets; R1's verification
   covers the hyperopt path.
5. **One `/leo5-submit` invocation per run** (standing hard gate), each line
   carrying explicit `--profile dev` (F-S0-10 discipline).
6. Wave B waits for Wave A completion (needs best configs); Waves C/D are
   independent and may go up immediately after R1 verifies.

### 6. What SC.8 will consume

Comparison rows assembled at SC.8: R1/R2/R3 vs frozen `item_proto` /
`lightfm_tags(_ids)` / `user_item_proto`-as-context (warm, hm); R4/R5 vs the
frozen cold columns under rules R-a..R-e; R6/R7 vs R8/R10/R11 (warm, ml-1m);
D4 strata + twin-angle instrument joined per the §3.6 interpretation rule;
R3-vs-bar as the noise yardstick on every warm delta. Anomaly handling: at
most one documented rerun round (Ground rule 6).

### Gate (SC.6)

**Status: OPEN — decisions taken one at a time (user directive at this gate).**
1. Dataset pin (§1): hm_1_month(+cold) + ml-1m warm; **ml-1m_cold explicitly
   not pinned** for dc01 (deferred to the enablement work-package).
   **→ RATIFIED 2026-07-15** after a deep-dive walkthrough (three grounds:
   no valid CF referee column on ml-1m cold until the bags-aware kNN exists —
   F-DC05-16 deferral honored; the "interaction-hidden, not new-release"
   epistemic caveat makes ml-1m cold permanently secondary; deferral closes
   no door — variant data already built).
2. Run manifest (§2): R1–R11 incl. R3's noise-yardstick reading, no f0 cold
   retrain, and the Wave-D row selection (no `user_proto`, no popularity row).
   **→ R3 half RATIFIED 2026-07-15** (plain-language walkthrough: f0 = the
   empty control, fI with zero metadata fields = item_proto through the fI
   code path; run on H&M only as the single-seed noise yardstick; no cold
   retrain — degenerate at cold by construction, ID-drop leaves nothing).
   Wave-D row selection decided separately below (decision 3).
   **→ Wave-D selection RATIFIED 2026-07-15 as proposed:** R8–R11 only
   (`item_proto` = stage bar, `mf` = fleet floor, lightfm×2 = feature
   baselines); `user_proto` stays with dc05's SC.6 (no cross-candidate
   work — batching option offered, declined by approval as proposed);
   popularity row follows the ml-1m_cold enablement.
3. Compliance + discipline (§3, §5): esp. the R1-first smoke sequencing and
   the 11-submission count.
4. On approval: commit artifact → push → LEO5 sync → `/leo5-load` →
   submissions begin (R1 first), each via its own `/leo5-submit`.

**Gate CLOSED 2026-07-15 — all four decisions ratified one at a time (user
directive), decisions 1–3 as annotated above; decision 4 RATIFIED AMENDED
(user):** the §5.4 wait-for-R1-completion sequencing is replaced — LEO5 is too
congested to serialize a full SLURM run before the fleet queues. Instead:
**judgment-scoped login-node smokes** (A30, small-workload policy) cover the
genuinely untested paths — the fI *hyperopt* path on hm_1_month (fixed layout;
probes covered only the retrain-config path) and the bags-layout hyperopt path
on ml-1m (fI + one lightfm row — the Ray/builder seam) — via short
explicit-override runs (~2 trials × 3 epochs); **then the whole fleet queues
immediately** (R1–R3, R6–R11 = 9 jobs), each via its own `/leo5-submit` with
explicit `--profile dev`. R4/R5 (cold retrains) remain structurally dependent
on Wave A best configs and follow at Wave A completion. §5.1–5.3 unchanged
(commit+push first, file-level data checks, `/leo5-load`).

### Run-submission record (2026-07-15, same sitting)

**Preconditions executed:** artifact committed (4d33617) + pushed; LEO5 pulled
to 4d33617 (`feat/scrutiny`, verified); file-level data checks passed for
hm_1_month / hm_1_month_cold / ml-1m (ml-1m `item_features.csv` md5-identical
to local canonical, header carries genres+tags); `/leo5-load`: 36/42 A30 free.
**Cluster checkout intentionally held at 4d33617 until the fleet completes**
(later doc commits are not pulled mid-fleet, so every C8 manifest records the
plan's own hash).

**Login-node smokes (decision-4 amendment), quarantined under
`/scratch/c7031336/smoke_runs/dc01_sc6/`:**
- fI × hm_1_month (fixed layout, hyperopt path): trials trained cleanly
  (sentinel resolution, Ray sampling, train/eval loop all good); exited
  nonzero at best-checkpoint extraction — both toy trials (2×3 epochs)
  declined from random init, so no checkpoint existed. Verdict: toy-budget
  edge, not a path defect; post-training save path separately proven by the
  2026-07-15 timing probes (full artifact set incl. metadata.json on both
  datasets). GO.
- fI × ml-1m (bags layout, hyperopt path): same pattern, same verdict — bags
  builders inside Ray trials proven on cluster. GO.
- lightfm_tags × ml-1m: **full end-to-end PASS** (`RUN_OK 2/2, results
  persisted`) — covers the entire chain incl. checkpoint extraction on the
  bags layout. GO.
- **Observation (no fix now):** `experiment_helper.start_hyper` crashes with a
  bare `AttributeError` when NO trial ever improves over init (no checkpoint
  to extract) — unreachable at dev/production budgets; candidate trivial
  guard, to be raised at SC.7.

**Submissions (one `/leo5-submit` each, all dev profile, seed 38210573,
conc 5, num-workers 1, A30, tag `dc01-sc6`; ml-1m fleet rows additionally
`s0-fleet-ml1m`):**

| # | Job | Model × dataset | `--time` |
|---|---|---|---|
| R1 | 7041210 | feature_item_proto × hm_1_month | 8:00:00 |
| R2 | 7041242 | feature_item_proto_noid × hm_1_month | 8:00:00 |
| R3 | 7041244 | feature_item_proto_f0 × hm_1_month | 8:00:00 |
| R6 | 7041245 | feature_item_proto × ml-1m | 4:00:00 |
| R7 | 7041247 | feature_item_proto_noid × ml-1m | 4:00:00 |
| R8 | 7041255 | item_proto × ml-1m (fleet row) | 4:00:00 |
| R9 | 7041256 | mf × ml-1m (fleet row) | 4:00:00 |
| R10 | 7041261 | lightfm_tags × ml-1m (fleet row) | 4:00:00 |
| R11 | 7041263 | lightfm_tags_ids × ml-1m (fleet row) | 4:00:00 |

R4/R5 (hm_1_month_cold retrains + cold evals) follow at Wave A completion
per the S0.3 §6 convention (`--retrain-config` from R1/R2 best configs,
`--time 4:00:00`; cold evals on the login node with
`--canonical-results-dir`). → Next protocol step: **SC.7** (flush &
retrospective), separate sitting.

**Amendment (2026-07-15, same day — recorded-hash deviation, ratified at
the dc05 SC.6 gate as its D-2):** the "checkout held at 4d33617 until the
fleet completes" intention above is superseded — dc05's SC.6 run plan was
gated and submitted while this fleet was still PENDING, which required
pushing + pulling the dc05 artifacts on LEO5. Jobs of this fleet that
start after that pull record the newer commit in their C8 manifests
instead of 4d33617. The delta 4d33617→(dc05 SC.6 commit) was verified
**docs-only plus the `run_combo.slurm` `--gpu-type any` extension** — no
change to any code path these A30-typed jobs execute (they submit with
their original sbatch scripts, already queued; the wrapper change affects
only future submissions). Run behavior is therefore untouched; SC.8's
compliance audit reads this note instead of flagging the hash mismatch.

### Wave A/C/D completion + Wave B submission (2026-07-19)

**All 9 hyperopts COMPLETED** (queue wait ~1–2 days, far under the July 21–25
scheduler estimates): R1 2h37, R2 2h21, R3 1h50, R6 3h50, R7 2h58, R8 3h13,
R9 1h46, R10 1h57, R11 2h00. All checkpoints + metrics verified present.
**Anomaly note for SC.8 triage: R6 (fI × ml-1m) finished 10 min inside its
4:00:00 limit** — the walltime memo's ml-1m ratios (probe-derived, solo-GPU)
underestimated hyperopt wall time by ~3×; several ml-1m rows ran 2–4× their
estimates. No run was lost; future ml-1m sizing should anchor on these
measured fleet times, not the probe ratios (learnings candidate at SC.7).

**First-glance test HR@10 / NDCG@10 (working evidence, two-tier policy —
readings belong to SC.8):** hm: fI 0.4984/0.2757 vs frozen `item_proto` bar
0.4938/0.2631; f0 yardstick 0.4912/0.2642 (fI−host delta ≈ within yardstick
noise); noid 0.4229/0.2317. ml-1m: fI 0.5156/0.2836, noid 0.4960/0.2735,
`item_proto` 0.5711/0.3187 (dev ≈ its prod 0.5723 — dev tier lands close
here), `mf` 0.5022/0.2793, `lightfm_tags` 0.5565/0.3197, `lightfm_tags_ids`
0.5612/0.3261. Notable for SC.8: fI trails its host by ~0.055 HR@10 on
ml-1m (opposite of the hm picture); lightfm rows sit between.

**Wave B submitted (2026-07-19, one `/leo5-submit` each):** R4 = job
**7121151** (fI cold retrain, R1 best config), R5 = job **7121152** (noid
cold retrain, R2 best config); hm_1_month_cold, `--profile dev`,
gpu-per-trial 1.0, num-workers 1, A30, 4:00:00 — the S0.7 retrain
conventions verbatim. Cluster at 68bfe4f (code-identical to the plan hash
per the D-2 note; local-ahead commits verified docs-only). Login-node cold
evals (`-m utilities.cold_eval`, `--canonical-results-dir` → R1/R2 dirs)
follow completion.
