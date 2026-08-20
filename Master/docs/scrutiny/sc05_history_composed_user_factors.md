# SC(dc05) — Scrutiny dossier: fU-ProtoMF (history-composed user factors, `feature_user_proto`)

> Candidate: dc05 (`Master/docs/design_candidates/dc05_history_composed_user_factors.md`,
> produced 2026-07-12 by the directed fU cycle under `fu_cycle_brief.md` M1–M7;
> concept review closed and BUILT same day — dc_checks/dc05 t01–t10+i06 green).
> Protocol version in force: **v1.2** (2026-07-12). Branch: `feat/scrutiny`.
>
> **Precondition declaration (protocol SC preamble):** SC(dc05) starts with S0
> complete through S0.6 + S0-build + S0.7 part 1 (machinery verified, fleet
> SUBMITTED — `user_proto` is fleet row 3, `s0_foundation.md` §S0.7), but the
> **S0.7 part-2 frozen reference table does not exist yet** (runs on LEO5).
> Per protocol: this routine says so explicitly and will **re-check against S0
> once the table lands** — SC.6 (run plan) cannot close before it does; the
> pre-run concept/implementation/explanation steps (SC.1a–SC.5) do not depend
> on the numbers.

---

## SC.1a Concept self-scrutiny (2026-07-12)

### 1. Preamble

**Manifest loaded:** protocol (full re-read), findings ledger, learnings
ledger, requirements doc (incl. R1 clarification note), candidate index, dc05
design doc (full, incl. all dated amendments), fU cycle brief,
`dc_checks/dc05/claims_spec.md` + `check_claims.py` results (verbatim in
design doc 4b), supervisor presentation, vault interpretability set
(intrinsic-over-posthoc, posthoc-naming-vs-model, Steck cosine caveat,
feature-injection fork). Code spot-reads for first-party verification:
`explanations/pipeline.py`, `hyper_params.py`, `cold_eval.py`,
`s0_foundation.md` fleet section.

**Learnings watch-outs that apply here** (full ledger reviewed):

- *Docs-drift on preprocessing claims* — dc05's "n_{u,f} counts distinct
  purchased articles" leans on the splitter dedup; consistent with the
  F-S0-01-corrected record. The 5-core/3-floor hallucination was already
  caught and vaulted (2026-07-12_1659) — the doc carries the corrected form.
- *Hidden aggregate invariants / divisor* — dc05 changes no eval rows now;
  recorded below as a constraint on the future cold-USER eval spec.
- *Committed measurement scripts* — the §4 measured numbers checked for a
  committed generator: **none found** → finding F-DC05-04.
- *Absent-key config defaults* — SC.4 checklist item (configs claim explicit
  `use_id_feature`; verify keys incl. `use_bias` there).
- *Fluent cross-level synthesis / mirror rhetoric* — already self-caught at
  5b-4; the "mechanism-shape identity may be claimed, validation transfer may
  not" rule is in the ledger `[fU-cycle dc05 5b]`.
- *Thresholds are production logic* — the 5b-6 history-length bands
  (3 / 4–5 / 6–12 / 13+) are eval presentation binning over a reported
  continuum, not mechanism thresholds; compliant.
- *Host-inherited surfaces need explanation-validity audit* — the ITEM-side
  coefficients t_l and B(t) got exactly this (3b-i, M6′); the remaining
  validity gap is the missing committed profile-validity check → F-DC05-03.
- *Scrutiny measurements ≠ thesis material* — §4 numbers carry the
  restriction banner already; F-DC05-04's script inherits it.

**Ledger re-check (open + shared):** no `open` findings remain anywhere in
the ledger (all `fixed`/`wontfix`). Shared findings walked against dc05:
F-S0-02 (worker RNG regimes) applies unchanged, wontfix stands; F-S0-03
divisor assert covers all eval paths incl. the fU cold runner (t07 exercised
it at build); F-S0-04 train-unseen test items — noted for SC.8 row
interpretation (fU's item side is host-free-t; such rows score on untrained
vectors exactly as the host); F-S0-06 NaN guard — dc05's new builder
`build_user_history_weights` must inherit the symmetric NaN policy → SC.4
checklist item, not a concept issue (V1 item features have 0 NaN); F-S0-07/13
ID-drop — dc05 built the cold-USER mirror at build time (t07), concept
convention recorded in §3.4; F-S0-08/09/11/12 — S0 machinery, no dc05
interaction beyond the cold-USER spec constraints noted in §3e below. Nothing
marked stale.

### 2. Known-issues dispositions FIRST (steel-man before adversary)

The file's own recorded weaknesses, walked one by one:

| # | Recorded issue | Disposition |
|---|---|---|
| 1 | 4b C7(c) "bitwise-frozen" falsified (wording level) | **Resolved.** Corrected wording ("conserved to machine precision") used throughout; M7 trickle-back to dc01 §I.1 applied (commit b059a5f); learnings entry `[fU-cycle dc05 4b]` exists. |
| 2 | Four skipped training-dynamics claims = the empirical bets: (i) thin-user advantage, (ii) profile sharpness, (iii) omnipresence-tilt magnitude, (iv) B(t) absorption | **Open by design** — run-stage; instruments committed (D4 bands, profile entropy, per-value share vs frequency, B(t) variance share + popularity correlation). SC.6/SC.8 own them. |
| 3 | 5b-1 B(t) guts the story | **Resolved as scoped + open bet.** Rebuttal (host-inherited, first artifact to name/separate it) stands; B(t) variance share promoted to primary metric (adopted); bet (iv) remains the load-bearing unknown — correctly carried. |
| 4 | 5b-2 squash-vs-thin-stratum geometry | **Resolved (rebutted on C9's own numbers)** — thin users are farthest from the mean direction; the genuine thin-user problem (noise, not collapse) is honestly stated. Attribution gap conceded → decoupled control request (§3.6). |
| 5 | 5b-3 normalization "manufactures M5′" | **Resolved (rebutted via C8a) + open risk as tabled** — mean-regression is normalization-invariant in the metadata geometry; exponent stays the recorded evidence-triggered alternative. My §3a finding F-DC05-01 sharpens the §3.1(b) wording this rebuttal leans on. |
| 6 | 5b-4 mirror rhetoric | **Resolved as writing discipline** — ledger rule filed; doc claims shape identity only. |
| 7 | 5b-5 novelty = conjunction; probe depth | **Resolved (probe extended in place: UISVD++, attributes-coupling MF, Ask-the-GRU, Movie Genome — all distinguished) + standing narrative burden** — the thesis must argue "grounded structure the algebra exposes", not assume it. Carried to SC.9/SC.10. |
| 8 | 5b-6 attribution + strata resolution | **Resolved by adoption** — `lightfm_hist` charter-amendment request named in §3.6 (decision deferred to SC.6 gate, the user's); explicit history-length bands + `hm_3_month` fallback adopted. |
| 9 | 5b-7 cold rhetoric | **Resolved** — R5 row and M6.2 state the honest scope (warm-thin measurable; absent-signal NO; fold-in presented as deployment property, not evaluated claim). |
| 10 | 5b-8 printed numbers partly arbitrary | **Resolved by the ratified disclosure architecture** — §3.4 honesty note added at the flagship example; epistemics live in the thesis hidden-effects section (SC.5 gate scope, both models symmetrically). |
| 11 | 5b-9 0.61% category error | **Resolved by the R6 qualification** (representational vs geometric ceiling; instruments own the geometry). |
| 12 | 5b-10 keystone validates plumbing only; fold-in path unevaluated | **Resolved as protocol role (keystones ARE plumbing proofs) + standing open state** — train/serve mismatch carries F-DC01-04's caveat mirrored (3b-v b); K3′ adoption = user mechanism decision, on record. |
| 13 | Review-sitting additions: K6′ centering (unstacked, evidence trigger = M5′ instruments on the heavy band), M5′ noise-for-crowding precision + ids-arm division-of-labor + M5′→M6′ interlock (vault 1830), B(t) terminology + magnitude directives (vault 1808) | **Standing, consistent** — nothing in this scrutiny contradicts them; F-DC05-01/02 below extend the same anatomies. |

No new finding below re-discovers anything in this table.

### 3. The scrutiny proper

**(a) Mathematical soundness.** Re-derived first-party: the score identity
S = B(t) + Σ_l t_l·cos_l; exactness of the per-word/per-purchase regroupings
(machine-precision-confirmed, 4b C1/C2); the gauge geometry (u* lies in
1 + {P̂x} of dim ≤ d; gauge dim ≥ K_u−d−1 — the doc's "≤ d+1" slab bound is
conservative and correct); C8's scale-invariance argument; the keystone
reduction's well-definedness; dedup consistency of n_{u,f}. All sound. Two
refinements found, both extending the M5′ anatomy:

- **Norm-channel handover (→ F-DC05-01).** §3.1 reason (b) claims that under
  the mean "the ID row's relative influence is history-length-independent."
  Only the ID row's *weight* (1 vs metadata's total 5) is length-independent.
  Its *influence* on q̂_u depends on ‖Σ_f w̄_{u,f}e_f‖, and that norm
  generically shrinks as the basket spreads: under near-orthogonal word
  embeddings ‖Σ w̄_f e_f‖ ≈ scale·‖w̄‖₂, and w̄ is spikier for thin baskets
  (per-field mass 1 over ≤3 values) than for heavy ones (~50 distinct words).
  So relative ID influence mechanically GROWS with |H_u| — a second, geometric
  handover channel that compounds M5′(b)'s training-quality handover, and a
  (mild) overstatement in the very sentence the 5b-3 rebuttal leans on. Status:
  hypothesis-with-instrument (trend claim, C9-class, not an identity).
- **Interaction-weighted coverage cloud (→ F-DC05-02).** The in-batch
  inclusion regularizers (A1 Eqs. 4–5) see each user with multiplicity
  ∝ |H_u| (one training sample per interaction). Under fU, heavy users are
  exactly the mean-regressed, crowded ones (M5′/C9), so the batch cloud the
  coverage forces act on is denser at the crowded center than the
  user-uniform cloud — prototype placement is pulled toward the mean-basket
  cone, and the thin stratum (the central bet's benefit surface) is
  relatively under-covered by R_{U→P}. The sampling is host-inherited; the
  consequence is fU-specific (host heavy-user embeddings have no structural
  crowding). §3.8's inheritance note covers duplicate baskets but not this
  frequency weighting.

**(b) Expressiveness.** The information delta is honestly identified: the
history itself is already in the interaction matrix the host factorizes; what
fU adds is the **item-feature table entering the user side** plus a
parameter-sharing inductive bias (426 shared words vs 73,418 free vectors) —
the doc's LightFM framing (3b-v hypothesis) carries exactly this and claims
no ranking-expressiveness gain (3b-iv). Gradient paths: e_f aggregates
gradients across all users sharing the word, scaled by w̄ — the statistical
efficiency the bet rides on; verified plumbing at build (t09). Capacity
ceilings stated at both levels (representational 0.61%; geometric =
instrumented, R6 row as qualified). No delta.

**(c) Degeneration-mode coverage.** Inherited coverage pair + M1′–M6′ + the
mode table is complete as far as I can push it; the two §3a refinements
extend M5′'s anatomy rather than open new modes. The knob inventory
(K1′–K6′, normalization exponent, inherited regularizers/max_norm/L2, K_u
with its multi-role caveat) reaches every listed mode; M6′'s knob K5′ carries
its stated accuracy risk honestly. One deliberate absence confirmed correct:
no mode for "basket self-cancellation" is needed — with w̄ ≥ 0 and trained
e_f, destructive interference of a mixed basket is the M5′/crowding geometry
already instrumented (angle + the F-DC05-01 norm instrument covers the norm
side).

**(d) Requirements delta (re-rating vs the doc's own §5).** Concur on all 13
rows — **no delta**. Contested rows double-checked: R5 "partial (user-side
re-scope)" is right against the adversary's "weak" (the requirement's own
text names the weak-signal case, which is measurable and instrumented; the
absent-signal half is honestly NO'd); R6 "partial" correctly carries the
geometric qualification post-5b-9; R2 "strong" survives the intrinsicness
question (M6.1b: fixed, data-derived, human-performable aggregation stated in
the explanation; shares are literal summands — the boundary notes on
CF-trained meanings and B(t) are present). R1's operative-core reading is the
ratified per-stage form (requirements doc clarification note, 2026-07-12).

**(e) Cold-start mechanism check, incl. bias + ID-keyed side channels.**
Walked all ID-keyed channels on this host: (1) e_ID(u) at cold/fold-in
scoring — drop convention recorded in the doc AND built+pinned (t07),
F-S0-07's dc01 lesson applied from day one; (2) per-user bias scalar —
fleet-off (C3) and structurally rank-inert for item ranking (3b-v a): strictly
milder than dc01's bias gap, correctly analyzed; (3) **B(t) = 1ᵀt is the
item-side popularity home** — identified, disclosed, instrumented (M6′);
for cold ITEMS the entire t vector is untrained, so fU inherits the host's
chance-level cold-item behavior — the doc's stated expectation is correct and
the S0.3 runner still runs the row for comparability. The S0.3 eval detects
what this design could hide: nothing ID-keyed survives at cold scoring
post-drop, and the B(t) channel is measured by its own committed instruments.
**Constraints recorded for the future cold-USER eval spec (scrutiny-stage,
M6.2):** it must define its own divisor (F-S0-03 lesson), fold removed
purchases into warm-eval exclusion CSRs (F-S0-09 analogue on the user axis),
and rebuild history weights from the VARIANT train file (§3.2 leakage rule,
already pinned at build for the item-side variant dirs).

**(f) Steck cosine-caveat applicability.** The user side trains through the
same shifted cosine it reads out — the dc01 F-DC01-02 protection argument
transfers to u* and its shares. The profile read-out cos(e_f, p^u_l) is a
*different pairing* than the trained one (q_u vs p^u_l) — same residual risk
class as dc01's profile read-out, and the doc correctly carries F-DC01-02's
discipline (§3.5 boundary note). t_l is not a cosine and makes no semantic
similarity claim; its printed-value epistemics are the gauge discipline
(3b-i), ratified. **What is missing is the committed validity check itself:**
dc01 committed an SC.8 profile-validity spot-check (F-DC01-02) and later
mechanized it as the dual-naming-route diagnostic (F-DC01-13). dc05 has no fU
mirror of either — and `pipeline.py:133–140` (verified this session) routes
fU user-prototype naming through the intrinsic route ONLY, the exact `if/else`
shape whose fI analogue was the one unjustified gap at dc01's SC.5. Without
the post-hoc route on the same checkpoint, the intrinsic word profiles cannot
be validated against what member users actually buy. → **F-DC05-03.**

**(g) First-party verifications of load-bearing doc claims.** `user_proto` IS
in the submitted S0.7 fleet (row 3, `s0_foundation.md`) — §3.6's "already in
the frozen reference fleet" holds as *submitted*; numbers pending (see
precondition declaration). Pipeline/`EXPLAINABLE_MODELS` already includes
`feature_user_proto` (no F-DC01-11 recurrence). Configs registered as two
fixed configs + debug (hyper_params.py:257–323), per C7/F-DC01-01's lesson.
The §4 measured numbers have **no committed generator** (grep over
dc_checks/dc05 + scripts: nothing) → **F-DC05-04.**

### 4. Findings (bundled with proposed fixes — decided at this gate)

**F-DC05-01 [minor]** — §3.1(b)'s "ID-row relative influence is
history-length-independent under the mean" overstates: constant weight ≠
constant influence; the metadata part's norm generically shrinks with basket
spread, so relative ID influence grows with |H_u| (geometric handover,
compounding M5′(b)'s training-quality handover).
**Proposal:** dated precision amendment to §3.1(b) (+ one-line cross-ref in
M5′(b)); extend the M5′ instrument set with metadata-part norm and
ID-share-of-‖q_u‖ vs history-length strata; optionally extend the C9 script
with the norm-vs-|H| curve if SC.2 runs a toy round anyway.

**F-DC05-02 [minor]** — The in-batch inclusion regularizers see an
interaction-weighted user cloud (multiplicity ∝ |H_u|), which under fU
over-represents exactly the crowded heavy users — prototype placement biased
toward the mean-basket cone; thin users (the bet's stratum) relatively
under-covered. Host-inherited sampling, fU-specific consequence; not covered
by §3.8's duplicate-basket note.
**Proposal:** dated amendment adding this to the §3.8 M5′/M4′ interlock; 
instrument precision: the committed cloud-spread/coverage instruments are
computed BOTH user-uniform and interaction-weighted (cheap — same arrays,
two weightings).

**F-DC05-03 [minor]** — No committed profile-validity check for the fU
intrinsic read-out: F-DC01-02's SC.8 spot-check and F-DC01-13's dual-route
mechanization have no fU mirror, and the pipeline runs intrinsic-only naming
for `feature_user_proto` (pipeline.py:133–140), so intrinsic word profiles
cannot be compared against post-hoc member-purchase profiling on the same
checkpoint.
**Proposal:** (a) dated amendment to §3.4 read-out 4: commit the SC.8
profile-validity spot-check (intrinsic cos(e_f,p^u_l) profiles vs the
post-hoc top-k-member-users purchase-lift profiles, same checkpoint, both
arms); (b) the mechanization — dual naming route for fU (post-hoc pass
alongside intrinsic, artifacts nested per scoring dir, exactly F-DC01-13's
shape) — lands at SC.5, this candidate's explanation step.

**F-DC05-04 [minor]** — The §4 measured V1 numbers (train-history stats,
D_max = 112, distinct-signature count 73,167 / 0.61% / largest class 5) have
no committed generator script — the exact failure mode the S0-build learning
records ("numbers measured during a spec session are unreproducible unless
the script is committed WITH the spec").
**Proposal:** commit `Master/temp/dc_checks/dc05/basket_stats.py`
regenerating every §4 number from the split's train file +
`item_features.csv` (canonical 5-field set), run once and confirm the quoted
numbers (one documented rerun round if any deviate); banner: scrutiny working
basis only, no thesis quotes (F-DC01-06 restriction).

**F-DC05-05 [trivial] [fixed]** — Claim-label inconsistency: 4b uses C1–C9,
the rest of the file cites C1′–C9′. One-line label-convention note added to
4b (this session, SC.1a trivial fix). Logged in the ledger.

### 5. Gate — CLOSED 2026-07-12

SC.1a complete. Four minor findings + one trivial fix; no blocker-class
concept flaw; the concept record's own three review rounds (4b, 5b, review
sitting) hold up under re-derivation. **Gate decisions (user):**

- **F-DC05-01 accepted re-scoped** — effect acknowledged, "warrants testing
  out": instruments committed (metadata-part norm + ID-share vs
  history-length strata), minimal precision amendment only, no mechanism
  response yet.
- **F-DC05-02 approved with extensions** — recorded conditional on M5′
  (precondition instrument: the angle-to-mean per history-length diagram,
  committed as a standing SC.8 figure); adopted framing "activity-weighted
  sampling = implicit prototype-placement regularizer"; two-weightings
  readout measures the size; routes to a **dc05-specific hidden-effects
  section** (fU-emergent effects); no fix attempt now.
- **F-DC05-03 approved as proposed** — (a) spot-check commitment at SC.2,
  (b) dual naming route at SC.5.
- **F-DC05-04 wontfix** — numbers are scrutiny working basis only; thesis
  treatment gets dedicated reproducible experiments at writing time
  (standing rule); re-check owed then.
- Protocol-entry prompt issued at the gate (M5′ precondition + implicit-
  regularizer framing + dc05 hidden-effects section decision).

Amendments land at SC.2 as dated, visible edits.

---

## SC.1b Adversarial pass (2026-07-12)

Subagent received ONLY the dc05 design doc + the requirements doc (Ground
rule 10); brief per protocol ("what did the previous adversary miss; which
rebuttals do not hold; what breaks at H&M scale"). Seven points returned.

### Critique (verbatim)

> 1. **The central bet's information story is broken: fU's "features" are a deterministic transform of the very interaction signal whose weakness they are supposed to compensate — and the LightFM precedent cited for it does not transfer.** R5 reads "represented from features alone when interaction signal is weak" (requirements §3). For a 3-train-row user, w̄_u is computed from exactly those 3 rows; the only information added beyond the interactions is *item* metadata about the same ≤3 purchases. The B5 §5 evidence retailed in §2.1 and 3b-v ("the tags+about finding transplanted to histories") came from profile text — user-side information *independent of interaction count* — so it is evidence for a mechanism dc05 does not have. What remains is a parameter-sharing/smoothing claim (well-trained e_f, noisy 3-draw weights), which is real but categorically weaker, and the R5 "partial" rating plus rebuttal 7's defense ("the requirement's own text covers the weak-signal half") both trade on reading "features" as if they were side information. Neither the previous adversary nor the author names this collapse of R5's features-vs-interactions dichotomy.
>
> 2. **Rebuttal 2 fails twice on its own citations.** (i) It cites C9 — angle-to-mean falls with |H| — as proof thin users are "the least squashed." But C9's toy draws all baskets from *one shared distribution* (the doc's own M5′ precision note says so), so 100% of a thin user's angular deviation in that toy is multinomial sampling noise; a large angle-to-mean there is evidence of noise, not of individuation. Citing a simulation containing zero taste heterogeneity to rebut a claim about taste individuation is invalid regardless of which direction the angles go. (ii) Its fallback — fU's estimate "is at least data" versus the host's "nearly-untrained free vector, i.e. pure init noise" — mischaracterizes the baseline: a 3-row user's free embedding receives 3 positive updates per epoch across up to 60 epochs (plus negatives) under the actual ranking loss; it is underdetermined and L2-shrunk, not init noise, and it is fitted to the *same* 3 interactions fU consumes. The rebuttal survives only by comparing fU's estimator to a strawman of the host's.
>
> 3. **Rebuttal 1's equalizer — "both play under B(t)" — is contradicted by the document's own M5′→M6′ interlock.** The M5′ note (c) states that crowded composed user vectors flatten activations, "making the user-independent B(t) channel the cheapest differentiator." B(t)'s trained magnitude is therefore *endogenous to the arm*: fU's compact composed cloud (common-mass squash, mean-regression) is predicted — by this document — to route more score mass into B(t) than the host's free user cloud needs to. So "host-inherited" names the channel's existence, not its size, and the rebuttal's pre-emptive reframe ("if B(t) dominates, that is a finding about the U-host class") assigns a potentially fU-aggravated pathology to the host by fiat. The instrument is measured on both arms, which is right; the exculpatory logic wrapped around it is not.
>
> 4. **"M5′ is now a demonstrated mechanic" (4b Outcome, 4c table) is an unearned status upgrade — C9 demonstrates the law of large numbers, not crowding.** The doc's own ratified reading of M5′ is "noise-for-crowding": the risk is that *distinct personal centroids* are crowded by the shared population mass. Exhibiting that requires heterogeneous per-user distributions and a between-user-separation vs within-user-noise measurement; C9 has one shared p and measures only angle-to-mean. What the toy exhibits is the benign half (noise shrinks with |H|) that nobody doubted; the risk half — heavy users' *centroids* being close together — was never simulated and remains pure conjecture. "Toy-exhibited" in the degeneration table thus launders an untested claim into a verified-sounding one.
>
> 5. **The R1 "strong" rating contradicts the document's own function-class analysis.** The R1 row asserts "the score has no path around the grounded activation vector u*" — but B(t) = 1ᵀt flows through u* only via its *constant* +1 coordinate, which no feature, no history, and no user can influence. Elsewhere the same document calls B(t) "a de-facto item bias living inside the 'bias-free' fleet" (§3.4), concedes "'bias-free' is structurally weaker here" (§3.7-4), and collapses the function class to "per-item scalar bias + MF" (3b-iv). A learned, feature-independent additive score channel is precisely the kind of around-the-features path R1's operative core excludes; the honest rating is partial-with-disclosure, matching the candidate's own B(t) treatment everywhere except the requirements table.
>
> 6. **H&M temporal structure is unexamined, and the adopted power fallback makes it worse.** The document never looks at transaction dates: in a 1-month window, a median-5 history plausibly concentrates in one or two same-day orders, yet the flagship per-purchase zoom ("your 12 dark-denim purchases, +0.31") presents purchases as independent taste evidences — 3b-ii disciplines word-clique *attribution noise* but not this presentation-level independence implication, and co-purchase lockstep (which 3b-ii itself predicts) is strongest exactly within orders. Worse, rebuttal 6's adopted fallback for thin-band resolution is `hm_3_month`: a 3.5× different user population whose 3-month train window spans a season change — the regime where the static train-window mean composition's declared taste-stationarity limitation binds hardest. That is not a like-for-like power upgrade for a V1 claim; it is a different experiment with a weaker modeling assumption.
>
> 7. **The S2 "strong (this candidate IS the S2 promotion)" rating rewrites the requirement it grades.** S2's text names its content explicitly: "Symmetric extension to user metadata (age, club membership, …)". dc05 contains zero user metadata — Seed 2 rejects it, and the metadata variant is deferred to a future "candidate prime" (thesis-outline step 7). Delivering a *different* user-side mechanism while deferring the one S2 names is at best "partial — S2-as-written deferred, re-interpreted in history vocabulary"; "strong — delivered" grades the candidate against a requirement quietly redefined to fit it. Given the document elsewhere polices exactly this kind of scope substitution (e.g., R5's cold honesty), the asymmetry is conspicuous.

### Response (concede / rebut / open risk, point by point)

**P1 — concede the precedent overstatement + the information-provenance
sentence; rebut the dichotomy-collapse conclusion.** Conceded: B5's
tags+about arm used profile text — side information independent of
interaction count — so 3b-v's "the tags+about finding transplanted to
histories" overstates; the precedent is *analogical* (user-side features
helped the thin stratum), not mechanistic. Also conceded: fU's user-side
information is item side-information reached through the same ≤3
interactions — the R5 row should say so in one sentence. Rebutted: the
dichotomy does not collapse — item metadata IS information the host cannot
see (the host's ID-granular embedding does not know two purchases share
"dark denim"); the actual mechanism claim is statistical sharing (word rows
trained by every user carrying them), which is the LightFM §2.3-1
generalization argument plus the SVD++/FISM history-model lineage, and the
noid arm/fold-in realize "no free per-user parameters" literally. Rating
stays partial; precision amendments → F-DC05-08.

**P2 — concede both wording failures; rebut the conclusion reversal.**
(i) Conceded: C9's homogeneous toy cannot distinguish individuation from
sampling noise; its legitimate role is refuting 5b-2's squash-into-the-cone
*geometry* (thin users are not angularly absorbed), not establishing
"least squashed = most individuated." The response's own "noise, not
collapse" half was the correct claim; the C9 citation gets scoped to the
refutation role. (ii) Conceded: "pure init noise" is a strawman — the
host's 3-row embedding is underdetermined and L2-shrunk, not untrained,
and it fits the same 3 interactions. The honest comparison: same
interactions, attribute granularity with cross-user sharing vs ID
granularity without it. The bet is unchanged; the words were wrong.
→ folded into F-DC05-07 (with P4).

**P3 — concede in full; adopt the comparative interpretation rule.** The
channel's *existence* is host-inherited; its trained *size* is
arm-endogenous, and M5′(c) predicts fU-aggravation. Adopted rule: the B(t)
variance share is read COMPARATIVELY (fU vs host, same instrument, both
checkpoints) — host-level share ⇒ host-class property; fU ≫ host ⇒
candidate-attributable pathology, priced into the R2 narrative. The
rebuttal-1 record's unconditional "finding about the U-host class" framing
is superseded. → F-DC05-06.

**P4 — concede the status precision; adopt the heterogeneous toy.**
"Toy-exhibited" is earned by the noise half (and the homogeneous case as
the all-centroids-coincide extreme); the crowding half — between-user
centroid separation vs within-user noise as |H| grows under a heterogeneous
population — was never simulated. Labels corrected (4b Outcome, 4c table);
SC.2's one-round re-check gains the heterogeneous-mixture toy (per-user
taste distributions around a population prior; measure separation-to-noise
ratio vs |H|). → F-DC05-07.

**P5 — concede the sentence; rebut the downgrade (user decides).** The R1
row's "no path around u*" clause is false as written: B(t) rides u*'s
constant +1 coordinate — arithmetically through u*, semantically around
the features. Corrected sentence discloses the host-inherited
feature-independent channel with cross-refs (3b-iv, M6′). Rating stays
strong on the operative core: R1 is architectural (features first-class in
the representation that drives the score; no parallel reranker/bolt-on) —
B(t) is not a module beside the mechanism but the host score form itself,
and eliminating it would break the like-for-like stage bar. → F-DC05-06.

**P6 — concede both halves.** (i) Within-order co-purchase is the
strongest co-purchase-lockstep source and 3b-ii should name it; the
planned re-based decoupling-mass instrument gains an order-aware dimension
(within-order vs cross-order pairs), and a cheap order-structure readout
(same-day multi-item share of train baskets) joins the SC.8 working-basis
instruments — no new committed script now, respecting the F-DC05-04
disposition. The per-purchase zoom's independence *implication* is
epistemics → routed to the dc05-specific hidden-effects section (figures
stay arithmetic-honest per the SC.5 gate scope). (ii) The `hm_3_month`
fallback is re-scoped: corroborative robustness probe under a declared
stationarity strain (season boundary, 3.5× population) — never a
substitute reading for the V1 claim; the charter C2 written-reason clause
carries this caveat if invoked. → F-DC05-09.

**P7 — concede the labeling asymmetry; keep the substance.** S2-as-written
names demographic metadata; dc05 rejected that deliberately (Seed 2,
recorded, user-ratified) and delivers the user-side slot in history
vocabulary. The row is re-labeled "strong (re-scoped)" with the honest
form R5 already uses, and "IS the S2 promotion" is softened to "delivers
the S2 slot's user-side feature-awareness in history vocabulary; the
literal demographic variant deliberately rejected and deferred to
candidate prime." → F-DC05-10.

### Fold list

Accepted points → new proposal bundles F-DC05-06..10 (ledger, pending this
gate); all amendments land at SC.2 together with the SC.1a bundle. SC.2's
toy re-check round is now non-optional in practice: it carries the
heterogeneous-crowding exhibit (P4) and the F-DC05-01 norm-handover curve.

### Gate — CLOSED 2026-07-12

**Decisions (user):**

- **P1 / F-DC05-08 → wontfix for dc05:** the precedent concern belongs to
  the later user-attribute variant (candidate prime, vault `2026-07-11_2351`
  step 7); mis-lands here; hypothesis tested empirically regardless.
- **P2+P4 / F-DC05-07 → approved** ("contextualize the sentence"), under the
  gate-stated user principle: **work empirically, not scholastically** —
  hypotheses + designed tests at this stage, no pre-emptive verdicts.
  Principle added to the learnings ledger.
- **P3 / F-DC05-06a → approved wholeheartedly** (comparative B(t) rule).
- **P5 / F-DC05-06b → approved** as contextualize-and-keep-strong.
- **P6 / F-DC05-09 → wontfix: explicitly out of scope** — per-event
  interaction context is the lineage's declared drop; already flagged as a
  future direction in vault `2026-07-12_0051`; revisit after the full
  lineage.
- **P7 / F-DC05-10 → approved** ("concede the label"), amendment cites the
  vault placement of demographics (`2026-07-11_2351` step 7).

Approved amendments (F-DC05-01/02/03a/06/07/10) land at SC.2 as dated,
visible edits; SC.2 runs one toy re-check round (C10′ heterogeneous
crowding, C11′ norm handover).

---

## SC.2 Concept fixes (2026-07-12)

### 1. Amendment list (all dated, visible edits in the design doc)

| # | Finding | Where | What |
|---|---|---|---|
| 1 | F-DC05-01 | §3.1 (normalization paragraph) | Precision amendment: "history-length-independent" holds for the ID row's weight, not its influence — geometric norm-handover channel named; instruments (metadata-part norm, ID-share vs strata) joined to the M5′ set; designed test C11′. |
| 2 | F-DC05-02 | §3.8 M5′ block | Interaction-weighted coverage cloud: conditional on M5′ (precondition figure: angle-to-mean per history-length band), implicit-regularizer framing, two-weightings size instrument, routing to the dc05-specific hidden-effects section. |
| 3 | F-DC05-03a | §3.4 read-out 4 | Committed SC.8 profile-validity spot-check (intrinsic vs post-hoc member-purchase lift, same checkpoint, both arms); dual naming route = SC.5 obligation. |
| 4 | F-DC05-06a | 5b response 1 | Superseding note: comparative B(t) rule (fU-vs-host on the same instrument; attribution by measurement, never by fiat). |
| 5 | F-DC05-06b | §5 R1 row | "No path around u*" contextualized: B(t) rides u*'s constant coordinate — feature-independent channel disclosed; rating kept on the operative core. |
| 6 | F-DC05-07 | 4b Outcome, 4c M5′ row, §5 R6 row, 5b response 2 | "Toy-exhibited" split into noise half (C9) vs crowding half (C10′); C9-citation scoped to the refutation role; "pure init noise" replaced by the honest host-baseline comparison. |
| 7 | F-DC05-10 | §5 S2 row | Re-labeled "strong (re-scoped)"; vault citation `2026-07-11_2351` step 7; re-interpretation flagged, not silent. |
| 8 | — | 4b (new closing note) | SC.2 re-check outcome recorded (C10′/C11′ all confirmed). |

Candidate index row updated: status → **in scrutiny — concept amended (SC.2)**.

### 2. Black-box toy re-check (one round, per protocol)

Spec: `Master/temp/dc_checks/dc05/claims_spec_sc2.md` (self-contained; C10′
heterogeneous crowding-vs-noise, C11′ norm handover). Subagent received ONLY
the spec; script: `Master/temp/dc_checks/dc05/check_claims_sc2.py`
(float64, 5 seeds, ~4 s CPU). **Report (verbatim, tables abridged to the
seed-mean rows — full tables print from the script):**

> **C10′(a) — noise half — CONFIRMED.** Mean angle to own centroid strictly decreasing in |H| for every α (e.g. α=2: 60.4° → 43.5° → 28.6° → 16.5° → 9.7° for |H| = 3→300), on the seed-mean and all 5 individual seeds — zero violations.
>
> **C10′(b) — individuation half — CONFIRMED.** Mean pairwise angle between composed vectors converges to the CENTROID pairwise angle, not 0 (α=2: 79.0° at |H|=3 → 39.6° at |H|=300 vs centroid 37.5°; α=0.2: → 71.8° vs 71.6°; α=20: → 20.0° vs 13.1°, convergence visibly slower when centroids are close).
>
> **C10′(c) — crowding half — CONFIRMED.** Asymptotic between-centroid separation strictly decreasing in α: 71.60° (α=0.2) / 37.49° (α=2) / 13.06° (α=20) — crowding is governed by the heterogeneity axis α, independent of |H|.
>
> **C10′(d) — joint SNR — CONFIRMED.** Separation-to-noise strictly increasing in |H| for every α (α=0.2: 1.82 → 12.55; α=2: 1.31 → 4.07; α=20: 1.25 → 1.85) — growth dramatic for idiosyncratic populations, flat for homogeneous ones, consistent with the crowding story.
>
> **C11′(a) — norm handover — CONFIRMED** (α=2). Mean ‖m_u‖ strictly decreasing: 7.86 → 5.37 → 4.48 → 4.10 → 3.98 across |H| = 3→300, asymptoting to mean ‖c_u‖ = 3.92 (within 1.6% at |H|=300); positive asymptote, not 0.
>
> **C11′(b) — ID share — CONFIRMED.** Mean ID energy share ‖e_ID‖²/(‖m‖²+‖e_ID‖²) strictly increasing: 0.344 → 0.522 → 0.608 → 0.647 → 0.660 (|H| = 3→300); norm ratio ‖m‖/‖e_ID‖ strictly decreasing 1.42 → 0.72.
>
> **Summary: all six claims CONFIRMED; no claim untestable; not a single monotonicity violation on any seed.** Ambiguities resolved: nested baskets (prefix coupling across |H|), p_pop shared across α within a seed, both C11′(b) measures reported, per-pair angle averaging, V_g fixed at [15,19,22,26,30].

**Interpretation for the record (first-party):** C10′ sharpens M5′'s
empirical question — |H| per se *improves* individuation relative to noise;
the crowding risk lives on the heterogeneity axis (how close real users'
taste centroids sit), which is exactly what the committed real-checkpoint
instruments (angle-to-mean per band, cloud spread in both weightings)
measure. C11′ confirms the F-DC05-01 geometric handover: in the toy, the ID
row's energy share grows from ~1/3 (|H|=3) to ~2/3 (|H|=300) — the ids-arm
division-of-labor reading (words carry the taste class, ID carries
idiosyncrasy, handover grows with history) now has a mechanical toy exhibit;
the noid arm has no such counterweight, reinforcing M5′-threatens-noid.

### 3. Ledger status updates

F-DC05-01, -02, -03 (part a), -06, -07, -10 → **fixed** (this artifact's
commit); F-DC05-03 part (b) = SC.5 obligation; F-DC05-04/-08/-09 wontfix
(gate decisions); F-DC05-05 trivial fixed at SC.1a.

### 4. Gate

SC.2 complete: all approved amendments applied as dated visible edits, index
updated, one-round black-box re-check all-confirmed with two genuinely new
exhibits (crowding = heterogeneity-axis risk; ID-share handover curve).
Next step: SC.3 (implementation alignment — patch the built implementation
to the amended concept where behavior is touched; the SC.2 amendments are
concept/instrument-level, so SC.3 is expected to be verification-heavy, not
change-heavy). Awaiting gate review.

---

## SC.3 Implementation alignment (2026-07-13)

Protocol §SC.3 re-read this session. Scope: the dc01/dc02 flavor (the candidate
was BUILT before scrutiny — patch/verify the implementation against the amended
concept) plus the interlude's consumption item: **fU-on-ml-1m through the
S0-build-extension builders** (bags layout, per-dataset fields), ml-1m basket
stats, and the H&M-specificity design-doc note. As predicted at the SC.2 gate,
the step is verification-heavy: the SC.2 amendments are concept/instrument-
level and owe the code nothing now.

### 1. Green baseline (current HEAD, 7d56051)

dc05 `t01–t10 + i06` all green re-run this session; S0-build-extension pins
`s0 t07/t08/t09/t10` all green. The build's keystones stand: C5′ bit-identity
(t05), A1 term-by-term objective equality (t06), cold-user ID-drop mirror
(t07), checkpoint roundtrip with non-persistent buffers (t08).

### 2. Amendment→code walk (nothing owed — verified, not assumed)

| SC.2 bundle | Code consequence now | Status |
|---|---|---|
| F-DC05-01 norm handover | Instruments (metadata-part norm, ID-share vs strata) = SC.8 real-checkpoint work; no train-path change | none owed ✓ |
| F-DC05-02 interaction-weighted cloud | Two-weightings readout = SC.8; sampling deliberately untouched (no mechanism response, gate decision) | none owed ✓ |
| F-DC05-03a profile-validity spot-check | SC.8 commitment; part (b) dual naming route = **SC.5 obligation, tracked** | none owed at SC.3 ✓ |
| F-DC05-06 comparative B(t) rule + R1 | Doc-level; B(t) instrument = SC.8 | none owed ✓ |
| F-DC05-07 M5′ evidence split | Doc-level; C10′ committed at SC.2 | none owed ✓ |
| F-DC05-10 S2 re-label | Doc-level | none owed ✓ |

### 3. Carried checklist (from SC.1a preamble + roadmap)

- **F-S0-06 NaN symmetry, vocab side:** inherited by delegation — the builder
  routes through `build_feature_ids` (NaN raise) or `build_feature_bags`
  ('nan'-token raise + `keep_default_na=False`). ✓
- **F-S0-06 class, train-file side:** probed with malformed toy CSVs — a NaN
  `user_id` row is **silently dropped** (pandas groupby drops NaN keys); a NaN
  `item_id` raises only accidentally (opaque IndexError). → **F-DC05-11**
  (below). Both real train files verified 0 NaN in both columns (hm_1_month
  476,394 rows; ml-1m 562,308 rows), so the proposed guard is behavior-neutral.
- **Explicit config keys:** `use_bias=0` explicit via `base_param` in all three
  fU configs (fleet parity); `use_id_feature` explicit True/False, never
  searched; noid differs ONLY in that flag (t04 pin). ✓
- **Leakage rule:** every injection seam passes its own directory; variant-train
  rebuild pinned by t02c. ✓
- **Factory conventions:** single-owner init (F-DC01-10 pattern), host-shape
  asserts, item table M×K_u (t03). Cold conventions: user-side drop + item-side
  declared no-op for fU (t07). ✓

### 4. fU-on-ml-1m — the second testbed's chain, exercised and pinned (t11)

New test `dc_checks/dc05/t11_ml1m_fu_chain.py` (19 checks, green): canonical
resolution ('ml-1m' AND 'ml-1m_cold' → genres+tags@0.8, bags); real builder on
`data/ml-1m`; **independent vote-mass recompute** — Σ_f w̄_{u,f} == mean
tokens/purchase per user, recomputed from raw CSVs by a separate pandas path
(maxΔ 7.6e-06 over all 6,034 users); padding discipline; injection discipline
(payload lands user-side, caller dict never carries tensors);
**hand-recomposed q_u** from raw CSVs vs module forward for 3 sampled users
(maxΔ ≤ 2.9e-06); forward/loss finite; gradients reach all four parameter
surfaces (word table, ID rows, prototypes, item table), none on buffers.

**ml-1m working numbers** (recorded as a dated §4 addendum in the design doc;
F-DC05-04 disposition extends — scrutiny working basis, no thesis quotes;
t11 pins the load-bearing identities): N=6,034 / M=3,125 / train 562,308
(dedup-stable); |H_u| mean 93.2, median 56, p10 15, p90 223, p99 493, min 3,
max 1,415; V=1,061 (18 genres + 1,043 tags@0.8); distinct tokens/user mean
449, median 430, **D_max=1,022** (~96% of the vocabulary in heavy baskets —
the omnipresence/M5′ end, exactly the regime the charter added ml-1m to
probe); per-user vote mass mean 24.27, range 11.0–48.8; **basket-signature
twins 0 of 6,034** (V1: 0.61%); padded buffers 74 MB (embedding_bag fallback
unneeded on both datasets).

**Design-doc dated notes applied (2026-07-13):** §3.1 dataset-scope note —
"each purchase contributes one value per field / total metadata mass exactly
F=5" is H&M/fixed-layout-specific; under ml-1m raw bags the per-user metadata
mass is the basket's mean tokens-per-movie, so the **metadata-vs-ID balance
becomes user-dependent** (a mass-heterogeneity channel beside F-DC05-01's
norm handover, measured by the same committed instruments; the USER-side face
of the ρ=0.672 popularity-coupling must-mention). Plus the §4 ml-1m
working-numbers addendum.

### 5. Findings

- **F-DC05-11 [minor] [fixed at gate]** — silent drop of NaN `user_id` train
  rows in `build_user_history_weights` (F-S0-06 class, train-file side).
  Guard + t02 extension approved at the gate and landed same day;
  behavior-neutral on in-scope data (both real train files 0 NaN).
- **F-DC05-12 [trivial] [fixed]** — `dc_checks/dc05/smoke_train.py` had been
  broken since the 'canonical'-sentinel migration (S0-build ext component c):
  it hand-feeds configs to `Trainer` without the resolution seam, so the
  injection guard killed every local smoke at model build. Fixed this session
  (one `resolve_feature_fields_in_conf` call, the same seam `start_hyper`
  runs); learnings entry added (side harnesses belong to a migration's
  regression set — the migration updated the pinned tests but not the harness).

### 6. Local smokes (repaired harness, both datasets)

- `feature_user_proto_debug × hm_1_month`: train 215 s, all test metrics
  finite, HR@10 0.2679 / NDCG@10 0.1184 — consistent with the build session;
  the H&M path is regression-clean.
- `feature_user_proto_debug × ml-1m`: train 1,018 s, all metrics finite,
  HR@10 0.4369 / NDCG@10 0.2331 — **the first end-to-end fU train on the
  second testbed** (builder → sentinel resolution → trainer → tester →
  results dir). 2-epoch debug sanity values, never reportable.
- Placement note: these ran on laptop CPU (in flight before the policy);
  from the next smoke onward the **LEO5 login-node A30 policy** applies
  (CLAUDE.md rule 2026-07-13).

### 7. Session policy/planning records (same sitting, user directives)

CLAUDE.md gained the smoke-test/small-workload placement rule (login-node
A30s when VPN up; compared rows stay SLURM — hard gate untouched); v3
snapshot in `Master/llm_usage/`. Early-submission planning principle recorded
(cluster congested → frozen-spec runs submit as early as gates permit;
candidates for pulling forward: ml-1m fleet rows, ml-1m_cold scp — proposals
belong to the SC.6/next gates).

### 8. Gate — CLOSED 2026-07-13

**Decisions (user):** (a) **F-DC05-11 approved → fixed same day** — guard
landed, t02 extended with both NaN cases, builder-touching tests re-run green
(t02, t05 keystone, t11, s0-t08). (b) **ml-1m working-numbers treatment
confirmed** (F-DC05-04-consistent: dossier + design-doc record; t11 pins the
identities; no standalone generator). (c) Planning clarification recorded:
**cold-variant retrains cannot be pulled forward on either dataset** — they
consume the best canonical config, which exists only after the canonical
hyperopt finishes (H&M: fleet in queue since 07-11 → cold retrains wait for
it, correctly; ml-1m: the pullable-forward item is the canonical hyperopt
rows themselves + the ml-1m_cold scp, its cold retrains then follow the same
pattern). The cold protocol is mechanically identical on both datasets.

Next step: SC.4 (black-box spec-vs-code + first-party audit, feature entry
end-to-end on BOTH datasets).

---

## SC.4 Implementation scrutiny (2026-07-13)

**Scope:** the fU implementation as landed at SC.3 (HEAD at session start; dc05
suite t01–t11 + i06 re-run green this session as the audit baseline). Protocol
SC.4 section re-read. Black-box spec-vs-code subagent (GR10) received ONLY the
corrected design doc (all dated amendments through 2026-07-13) + the
implementation files (extractors, factory, feature_ids, hyper_params, start,
rec_sys, cold_eval, and the read-out modules: history_readout, accessor,
breakdown ×2, naming, proto_cards, pipeline); blind to the dossier, ledger,
protocol, tests, and this session.

### 1. Black-box spec-vs-code report (verbatim)

> # Spec-vs-code audit — dc05 `feature_user_proto` (fU-ProtoMF)
>
> **Spec:** `Master/docs/design_candidates/dc05_history_composed_user_factors.md` (incl. all dated amendments through 2026-07-13).
> **Scope:** the 15 named implementation files, plus traced call paths (`rec_sys/trainer.py:67`, `rec_sys/tester.py:50`, `utilities/explanations/loader.py:46`, `utilities/explanations/naming/*`, `Master/scripts/run_combo.py:75-101`, `experiment_helper.py:202-203`).
>
> ## Findings (most severe first)
>
> ### 1. MAJOR — The committed intrinsic explanation surface is fixed-layout-only and non-functional on ml-1m, the second testbed the spec's dated amendments put in scope
>
> **Spec passages:** §3.1 dataset-scope note (2026-07-13) makes ml-1m an in-scope testbed under the bags layout ("each purchase contributes its TOKEN COUNT … measured on the real split"); §4's ml-1m working-numbers block presents the second testbed as live via "the real builder chain"; §3.4 commits read-outs 2 (per-word shares), 3 (per-purchase regrouping) and 4 (intrinsic profiles `cos(e_f, p^u_l)` for naming) plus the renderer slot as core candidate commitments, with no dataset qualifier; the header records "renderer/cards/naming slots … all green".
>
> **What the code does:** the *training* path fully supports bags (`feature_extraction/feature_ids.py:332-448`, `build_user_history_weights(layout='bags')`; resolution at `feature_ids.py:35-84`). The *read-out* path does not:
>
> - `utilities/explanations/breakdown.py:369` — the fU per-purchase zoom (the ratified figure default) calls `build_feature_ids(dataset_dir, feature_fields)`, the fixed-layout builder. On ml-1m this raises `ValueError("field 'tags' has 94 NaN/missing cells …")` — **verified live against `data/ml-1m/item_features.csv`** — so `compute_breakdown_feature_user_proto` aborts and no fU breakdown can ever be produced on ml-1m (in the pipeline the exception is caught per-explainer, so the artifact is silently absent).
> - `utilities/explanations/breakdown.py:147-155` — `feature_code_labels` rebuilds the vocabulary as per-field `sorted(unique(astype(str)))` of the raw cells. On bag columns that yields joined-string labels (verified: 297+2,966 = 3,263 labels vs. the model's 1,061 token codes), so the word-zoom labels at `breakdown.py:357-359` would be **silently wrong** on any path that bypasses the crash above (e.g. an empty-basket user, where lines 384-391 skip the per-purchase branch and the breakdown IS written with mislabeled `alt_zoom_lines`).
> - `utilities/explanations/naming/namer.py:122-129` — the same fixed-layout vocab reconstruction feeds `name_prototypes_intrinsic`; its size guard (`len(code_to_cv) != scores.shape[0]`) fires loudly on ml-1m, so intrinsic naming (read-out 4) fails, `naming_user` falls to `None`, prototype cards render nothing and breakdown labels degrade to "p0, p1, …". `intrinsic_naming_config` (`naming/config.py:100-108`) also builds `FeatureSpec(f)` without `multi_value_sep`.
> - `utilities/explanations/history_readout.py:48-77` — `per_purchase_rows` assumes a fixed `(n_items, F)` code tensor; there is no bag-layout regrouping (`r_i = (1/|H_u|)·Σ_{tokens of i} e_t`) at all.
>
> **Why it matters:** read-outs 2-4 are the candidate's headline deliverable ("this stage closes exactly the surface S0.2 left post-hoc", R2 row). On one of the two in-scope datasets they have no implementation. The failure modes are mostly loud (a mitigating credit — the guards prevent wrong numbers being written, except the empty-basket mislabeling corner), but nothing in the spec licenses "read-outs are H&M-only"; the dated ml-1m amendments qualify only the *mass-balance analysis*, not the explanation surface.
>
> ### 2. MINOR — Figure zoom can fold the user-ID row into the remainder, against the "its own is_id line" commitment
>
> **Spec:** §3.4 renderer slot: "**zoom** = the top community's per-purchase shares (read-out 3; top-6 purchases by |share| + stated remainder), **the user-ID row as its own `is_id` line**".
> **Code:** `utilities/explanations/breakdown.py:623-649` (`_barh`) selects the top-6 lines by |value| across *all* zoom lines, ID row included; a small ID share is folded into "(+ N smaller lines)" and loses its dedicated hatched line on the **figure** (the md companion at `render_breakdown_text` always lists it, tagged `*(ID row)*`).
> **Why it matters:** the ID line is the M3′/feature-explained honesty device; on the primary visual artifact it is only conditionally present. Arithmetic stays exact (the fold is summed and stated), hence minor, not major.
>
> ### 3. MINOR — The `_noid` ablation arm is unreachable by the explanations pipeline, and the renderer's own docstring claims otherwise
>
> **Spec:** §3.6 commits the ids/noid pair as the isolating ablation, M3′ is "measured by design via the ablation pair"; `utilities/explanations/breakdown.py:24` claims the slot covers "``feature_user_proto`` (fU, **incl. ``_noid``**)".
> **Code:** `utilities/explanations/explainers/breakdown.py:26-27` (`supports()`), `utilities/explanations/pipeline.py:36-37` (`EXPLAINABLE_MODELS`) and `utilities/explanations/accessor/__init__.py:11-18` all match the exact string `"feature_user_proto"`; a run whose `metadata.json` model is `feature_user_proto_noid` is skipped by the standalone pipeline ("not explainable — skipping") and `get_accessor` would raise. Only the breakdown CLI (`breakdown.py:866`, `startswith("feature_user_proto")` with a hard-coded accessor key) works for the noid arm. `Master/scripts/run_combo.py:83-91` deliberately keeps fU out of auto-explanations until the SC(dc05) gate (documented, licensed) — but directs ablation arms to "the standalone pipeline/breakdown CLI", of which the pipeline half refuses them.
> **Why it matters:** noid cards/naming/geometry artifacts — inputs to the M3′/R4 frontier reading — cannot be produced without hand-editing; and the module documentation overstates coverage. The metric-level ids-vs-noid comparison is unaffected.
>
> ### 4. MINOR — cold_eval's auxiliary diagnostics are H&M-hard-coded; on an ml-1m cold variant the run dies before writing its report
>
> **Spec:** §3.4 (M6.2 mirror (ii)): "the fU fleet row still runs it [the S0.3 cold-ITEM runner] for comparability", with ml-1m in scope per the charter amendment (and `ml-1m_cold` already registered in `start.py:26`).
> **Code:** `utilities/cold_eval.py:55-56` pins `CANONICAL_FIELDS` to the five H&M columns; `attr_knn_patch` (line 227) fails gracefully (ValueError caught at line 432), but `tie_block_diagnostic` → `_signature_classes` (lines 250-253) is **not** wrapped — on an ml-1m variant it raises (missing columns) at line 437, after the primary metrics are computed but **before `report.json` is written** (line 450). Workaround exists (`--skip-tie-diagnostic`), and no ml-1m cold variant exists yet, so minor; this is shared S0 runner code, not fU-specific, but the fU comparability commitment inherits it.
>
> ### 5. COSMETIC — F=0 keystone + `max_norm` aliases the padding row onto user 0's ID row
>
> `HistoryFeatureEmbedding` padding uses word id 0 / weight 0.0 (licensed and documented, `feature_extractors.py:269-270`). In the F=0 reduction, `n_features=0`, so padding id 0 *is user 0's ID row*; under `max_norm` that row would be renorm-clamped on every forward regardless of whether user 0 is in the batch — a train-dynamics perturbation the "contributes exactly nothing" wording doesn't cover and that could break C5′ bit-identity under `max_norm`. Unreachable in every committed config (no config searches or sets `max_norm`; the C5′ toy ran without it), hence cosmetic. (Same-family note: `max_norm` passed to the outer `PrototypeEmbedding` at `feature_extractor_factories.py:199` is inert when `embedding_ext` is supplied — dead parameter, harmless.)
>
> ## Areas checked and found conforming
>
> - **Composition (§3.1/§3.2):** `q_u = Σ_f w̄_{u,f}·e_f (+ e_ID(u))` with `w̄ = n_{u,f}/|H_u|`, mean over the **deduplicated** (keep-first, `(user_id,item_id)`) train basket — `feature_ids.py:397, 434-446` matches the spec's dedup disclosure exactly, including the defensive re-dedup and the "distinct purchased articles" semantics; on bags, `n_{u,f}` = items of H_u carrying token f (dup-token guard upstream), per-user vote mass = tokens/|H_u| as the dataset-scope note states. Users absent from train get all-zero rows. ID row at offset `n_features + u`, on/off arms clean in both table sizing and forward (`feature_extractors.py:312, 335-336`).
> - **Keystone (§3.6, C5′):** V=0+ID forward is exactly a per-user embedding lookup — verified live this audit; the F=0 builder path returns well-formed zero-weight tensors.
> - **Host invariance:** `PrototypeEmbedding` is untouched on the `embedding_ext=None` path (construction-order comment preserves RNG bit-identity); the fU item branch is the host's free `Embedding(n_items, K_u)` built through the same `create_model` path as `user_proto`'s (`feature_extractor_factories.py:203-204`); score is the unchanged `RecSys` dot; inclusion regularizers run byte-identically in-batch; `rec_sys.py`/trainer/dataset carry nothing fU-specific (the injection seam predates dc05 and is spec-licensed). Search space mirrors `user_proto_chose_original_hyper_params` exactly; `use_id_feature` is a fixed flag in both arms, never searched (charter C7 honored); `_debug` matches the build-header license.
> - **Leakage (M6.1a):** `build_user_history_weights` reads only `data_path`'s own `listening_history_train.csv` (+ static `item_features.csv`); every seam — `trainer.py:67`, `tester.py:50`, `loader.py:46`, `cold_eval.build_model` (variant dir) — passes its own directory, so variant weights rebuild from the variant train file. Nothing anywhere derives from val/test. The cold-user ID-drop mirror (`cold_eval.py:176-223, 375-385`) implements the §3.4 convention with the structure-drift abort, correctly dormant until a `cold_users.csv` exists (M6.2's declared state), and the item-side drop correctly does not apply to fU.
> - **Read-out arithmetic (§3.4, C1′/C2′/C4′):** per-word and per-purchase shares are exact regroupings of one sum, jointly `‖q_u‖`-normalized (`per_feature_shares` with pre-weighted rows; `history_readout.py` mirrors builder dedup); `B(t) = 1ᵀt` is separated with the mandatory item-intercept wording verbatim ("non-personalized — this item's own scalar, identical for every user"), never rank-inert-phrased; the host slot carries the identical panel math and B(t) line (GR7-fair) with its post-hoc nearest-users zoom declared; self-checks assert total and zoom-vs-parent before writing; the footer computes the feature-explained fraction of the *personalized* score as re-scoped; the intrinsic `cos(e_f, p^u_l)` naming route and the F-DC05-03a dual (intrinsic + post-hoc) user-side naming both exist (`pipeline.py:104-107, 133-138, 187-202`); empty-basket degeneracy renders the honest statement path. Cards extend per-side with declared routes and shared geometry.
>
> **Summary:** the mechanism core (composition, host invariance, keystone, leakage, configs) conforms tightly to the spec, including its amendments. The one substantive divergence is dataset asymmetry: everything the spec promises works on H&M; on ml-1m only the *model* works — the entire committed explanation surface (finding 1) does not, and the spec contains no license for that gap.

### 2. Response (point by point)

**P1 — ml-1m read-out surface: CONCEDED in full (convergent finding).** The
first-party audit found the same gap independently this session, plus one
surface the auditor's file set could see but did not name:
`feature_readout.item_feature_rows` is equally bags-blind (it gathers
`feature_ids[item_idx]` verbatim, so under bags it would include padding-id-0
rows at full weight — the fI slot's exactness self-check would fire). One
detail sharpened: the "empty-basket mislabeling corner" does not actually
produce wrong labels today — an empty train basket implies empty `word_codes`
(the same train file feeds both), so the alt zoom degenerates to the ID line
alone, and on the canonical ml-1m split no empty-basket user exists (min
|H_u| = 3). The real mislabel hazard is prospective: any partial fix that
removes the `build_feature_ids` crash without rebuilding labels token-aware
would start writing silently wrong labels (verified: 3,263 reconstructed
labels vs 1,061 token codes — indexing succeeds, labels wrong). Today's
failure modes are loud-or-absent, not wrong — but the committed explanation
surface has no implementation on one of two in-scope testbeds, and SC.5
(dataset list pinned there) builds directly on it. **Severity major stands.
→ F-DC05-13** (bundled fix below).

**P2 — figure-zoom ID-row fold: CONCEDED.** `_barh` verified: top-6 by
|value| across all zoom lines, ID row included; a small ID share loses its
hatched line on the figure (md companion always carries it). Spec commits
"the user-ID row as its own `is_id` line" — and the same commitment exists on
the fI side (shared renderer). **→ F-DC05-14** (minor, shared; fix: is_id
lines exempt from the top-N fold).

**P3 — noid pipeline-reachability: PARTIALLY CONCEDED.** The structural half
is rebutted by ratified convention: headline-only auto-explanations is the
host convention gate-ratified at dc01's F-DC01-11 ("headline model only; arms
CLI-reachable"), and the ids-vs-noid comparison is metric-level; fI's arms
have exactly the same reachability. The documentation half is conceded — the
breakdown docstring's "incl. `_noid`" and run_combo's "standalone
pipeline/breakdown CLI" pointer both overstate (the standalone pipeline keys
on headline names). **Fixed as trivial this session (F-DC05-15):** both
comments now state the slot-vs-pipeline distinction and route arms to the
breakdown CLI. Recorded for SC.8: if the instruments there want noid
naming/cards artifacts, the key-normalization is a 3-line change decided then.

**P4 — cold_eval auxiliary diagnostics H&M-hard-coded: CONCEDED** (shared S0
machinery, surfaced by the charter's ml-1m amendment — it predates ml-1m
being in scope; both prior black-box cold audits ran before the amendment).
The damaging half is the unwrapped `tie_block_diagnostic`: on a non-H&M
variant the runner dies AFTER computing primary metrics but BEFORE writing
`report.json`. **→ F-DC05-16** (minor, shared; minimal graceful-degradation
fix now, full ml-1m cold support explicitly deferred to the ml-1m_cold
enablement work).

**P5 — F=0 + max_norm padding alias: CONCEDED, document-only.** Real corner,
unreachable in every committed config (none sets `max_norm`; C5′/t05 run
without it). **Fixed as trivial this session (F-DC05-17):** docstring note in
`HistoryFeatureEmbedding` records the alias, its max_norm interaction, and
the re-verify obligation should a max_norm config ever appear. The inert
outer `max_norm` parameter (factory → PrototypeEmbedding with embedding_ext
supplied) is left as-is — byte-parity with the dc01 branch, same inertness.

### 3. First-party audit record

- **Green baseline:** dc05 t01–t11 + i06 all re-run green this session
  (keystone t05 bit-identity, A1 term-by-term t06, cold-user drop t07,
  checkpoint roundtrip t08, gradient paths t09, breakdown/read-out t10,
  ml-1m chain t11).
- **Contract conformance:** fU uses `PrototypeEmbedding` exactly as the host
  `user_proto` does (reg losses accumulate in forward, drained via
  `get_and_reset_loss()`; the nested `HistoryFeatureEmbedding` rides the
  base-class zero-loss contract — no unaccounted loss). Item side = host free
  `Embedding` via the same factory path. Factory single-owner init
  (F-DC01-10 pattern) with host-shape asserts. Non-persistent buffers move
  with device and stay out of checkpoints (t08).
- **Optimization parity:** forward = one batched gather
  (B[, N], D_max, d) × weights + sum — no Python loops, no full-catalog pass;
  user side is the 1-D index path (one user per row), so the transient is
  (B, D_max, d), bounded (~209 MB worst-case ml-1m at B=512, d=100). Accessor
  `user_embeddings` batches at 8,192 (≤ 3.3 GB transient on ml-1m — offline
  path, acceptable; noted, no finding). Builder is vectorized pandas
  (groupby/cumcount scatter), no per-user loops.
- **Test adequacy vs claims spec:** C1′/C2′/C3/C4′ (check_claims), C5′
  keystone (t05, bit-identical fwd + grads), A1 objective equality (t06),
  C10′/C11′ (check_claims_sc2) all committed and green; integration
  t01–t04/t07–t09 cover shapes, builder, factory, configs, cold drop,
  roundtrip, gradient reach; t10 pins the whole read-out/renderer/pipeline
  stack INCLUDING the dual naming route; t11 pins the ml-1m TRAIN chain
  (independent vote-mass recompute, hand-recomposed q_u). Gap: no test covers
  the ml-1m READ-OUT chain — exactly where F-DC05-13 lives; the fix bundle
  includes the pin (t12).
- **Feature entry end-to-end, BOTH datasets:** hm_1_month — canonical 5 →
  `build_feature_ids` → injection (caller-clean, t10) →
  `HistoryFeatureEmbedding` → renderer: complete chain, pinned. ml-1m —
  canonical genres+tags@0.8 bags → `build_feature_bags` →
  `build_user_history_weights(layout='bags')` → injection → module: complete
  and pinned (t11) for training; read-out half missing (F-DC05-13). All four
  injection seams (trainer, tester, cold_eval, explanations loader) verified
  passing their OWN directory — the M6.1a leakage rule is mechanically sound.
- **SC.5-obligation status found ahead of schedule:** the F-DC05-03(b) dual
  naming route (user side) is ALREADY implemented (`pipeline.py:190-202`,
  landed at the build commit d7a983c) and pinned (t10
  `fu_dual_route_user_side_written`) — SC.5 verifies it in context rather
  than building it. run_combo's `EXPLAINABLE_MODELS` exclusion of fU is
  verified as the documented, licensed staging ("auto-explanations stay
  opt-in until SC(dc05)'s explanation gate") — the flip is an SC.5 item, not
  a finding.
- **Ledger hygiene (trivial, fixed this session):** F-DC05-08/-09 header
  status tags still read `[open]` although both gate dispositions were
  wontfix (and SC.2's artifact records them as such) — headers corrected.

### 4. Findings bundle (decided at this gate)

**F-DC05-13 [major] [shared]** — the explanation read-out layer is
fixed-layout-only; on the bags layout (ml-1m) the committed intrinsic
surface has no implementation: breakdown per-purchase zoom crashes
(`build_feature_ids` on multi-value cells), intrinsic naming dies on its
vocab-size guard (3,263 cell-level labels vs 1,061 token codes),
`feature_code_labels` reconstructs the wrong vocabulary,
`per_purchase_rows` has no bag regrouping, and `item_feature_rows` (fI
side) would gather padding rows at full weight. Loud-or-absent today; the
prospective hazard is silently wrong labels under any partial fix.
**Proposal:** (a) ONE shared layout-aware vocab/label helper in
`feature_ids.py` (`code→(field, value)` in exactly the builders' order;
bags: verbatim split on the separator, per-field sorted unique tokens),
consumed by BOTH `feature_code_labels` and
`name_prototypes_from_score_matrix` (replacing both ad-hoc
reconstructions — single source of truth); (b) bags-aware
`per_purchase_rows` (consume the item bag tensors + weights;
r_i = (1/|H_u|)·Σ_{tokens of i} e_t); (c) bags-aware `item_feature_rows`
(filter padding via `feature_weights`); (d) pin test
`dc_checks/dc05/t12_ml1m_readout_chain.py`: naming vocab == n_features on
ml-1m, breakdown end-to-end on a bags toy + the real ml-1m dir, labels
verified against the token vocab. Apply at this gate (SC.5 builds on it).

**F-DC05-14 [minor] [shared]** — `_barh` can fold the ID row into
"(+ N smaller lines)" on the figure zoom, against the §3.4 "its own
`is_id` line" commitment (both fI and fU specs carry it; md companion
unaffected). **Proposal:** is_id lines exempt from the top-N cut (always
shown, remainder excludes them); extend t10's honesty-marker pins.

**F-DC05-15 [trivial] [dc05] [fixed]** — noid-arm reachability
overstatement in two docs (breakdown module docstring "incl. `_noid`";
run_combo comment routing arms to "the standalone pipeline"). Fixed this
session; SC.8 note recorded (key-normalization decided there if needed).

**F-DC05-16 [minor] [shared]** — `run_cold_eval`'s unwrapped
`tie_block_diagnostic` (H&M-hard-coded fields) kills a non-H&M variant run
after primary metrics but before `report.json` is written; `attr_knn_patch`
already degrades gracefully, the tie diagnostic does not. **Proposal:**
wrap the tie diagnostic in the same graceful try/except (report written,
diagnostic marked unavailable with the reason); full ml-1m cold support
(bags-aware kNN/tie machinery) explicitly deferred to the ml-1m_cold
enablement work, recorded here so no future audit re-flags the deferral.

**F-DC05-17 [trivial] [dc05] [fixed]** — F=0 padding-id-0 aliases user 0's
ID row under max_norm (unreachable in committed configs). Docstring note
recording the corner + the re-verify obligation applied this session.

### 5. Gate — CLOSED 2026-07-13

**Decisions (user):** F-DC05-13 approved after a requested plain-language
deep-dive on the fix design (root cause = two independent vocabulary
reconstructions; fix = builder-owned single source of truth, refactor variant
over mirror variant); then **"approve all"** — F-DC05-13/-14/-16 fixes plus
ratification of the P3/P5 dispositions and the trivial fixes.

### 6. Fix application (same day)

- **F-DC05-13** — refactor variant landed: per-field vocab construction
  extracted into `_fixed_field_vocab`/`_bag_field_tokens` (the builders now
  call them), `build_code_to_field_value` exposes the code→(field, value) map;
  `feature_code_labels` and `name_prototypes_from_score_matrix` consume it
  (legacy items_info reconstruction survives only as the fixed-layout fallback
  for toy dirs — identical recipe — and REFUSES bags rather than mislabel);
  `per_purchase_rows` gained the bags regrouping (item bag tensors + weights);
  `item_feature_rows`/`item_meta_codes` filter padding (fI/lightfm side);
  layout threaded from saved configs through pipeline → ExplainCtx → slots →
  CLI. One iteration during application: the first cut made the pipeline's
  vocab build unconditional-fatal, which broke dc01's toy-dir t12 — resolved
  by the graceful fixed-only fallback (bags stays fatal), matching prior
  behavior exactly where the old recipe was provably correct.
- **F-DC05-14** — `_barh` exempts `is_id` lines from the top-N fold.
- **F-DC05-16** — tie diagnostic wrapped (attr_knn parity); report always
  written; ml-1m-cold deferral recorded in the ledger entry.
- **New pin:** `dc_checks/dc05/t12_ml1m_readout_chain.py` (20 checks): label
  count == n_features (1,061), labels == independent raw-CSV token recompute,
  fixed-layout back-compat byte-equal, bags-without-dir refusal, intrinsic
  naming runs on ml-1m (descriptors ∈ token vocab), fU breakdown end-to-end
  on the real split (internal exactness self-checks; png+md written),
  per-purchase bags rows sum to q_u (maxΔ 9.5e-07), bags `item_feature_rows`
  identity for both ID arms. t10 extended with the F-DC05-14 fold pin.
- **Verification sweep, all green:** dc05 t01–t12 + i06; s0 t02/t06/t07/t08 +
  i02; dc01 t06/t08/t10/t11/t12 + i02 (builder refactor + renderer + cold_eval
  regression-clean across all three suites).

The candidate's implementation now stands **faithful to the amended spec on
both in-scope datasets** by independent audit + first-party verification +
the new e2e pin. Next step: **SC.5** (explanation scrutiny — comparability
matrix, attribution gates, dataset-list pinning; carried obligations: the
run_combo auto-explanations flip, dual-route verification in context).

---

## Interlude — dataset-scope charter amendment (2026-07-12, between SC.2 and SC.3)

User decision at the SC.2 gate follow-up (vault `2026-07-12_2114`; dated
amendments applied to charter C2/C5/C6): **ml-1m joins as a second
quantitative testbed** (complementary regime: heavy histories × coarse
vocabulary — probes the M5′/omnipresence/B(t) end, while H&M V1 probes the
thin-user end where the central bet lives); **amazon2014 discarded from the
feature-aware scope** (ratings-only as used by ProtoMF — no features, no
object for a feature-aware comparison; stays replication context). ml-1m
canonical set: `genres` (bag layout — LightFM-native, B5 §2.2/§4.1 verified)
+ `year` (banded); fleet rows added lazily. Consequences for this routine:
dc05's dataset list is pinned at SC.5/SC.6 (expected: hm_1_month primary +
ml-1m); the fU-side ml-1m support (basket builder over multi-valued genre
fields + the ml-1m `item_features.csv` build) folds into SC.3's scope; the
H&M-specific working numbers in the design doc (V=426, twin rates, basket
stats) remain V1-scoped — ml-1m gets its own at its steps.

---

## SC.5 Explanation scrutiny (2026-07-13)

**Scope:** protocol v1.2 SC.5 section re-read (incl. the thesis-weight
preamble — the artifacts here are the standing host-vs-variant comparison
instrument; for this stage the pair is **fU vs U-ProtoMF**, the two fixed
questions asked of both: (1) what do the user prototypes look like —
intrinsic word profiles vs post-hoc profiling; (2) what does a concrete
recommendation look like, broken down through the taste communities — the
shared renderer, same user/item, side by side). Manifest loaded: the full
`utilities/explanations/` tree (pipeline, loader, items_info, all six
accessors, all seven explainers, naming layer, `history_readout.py`,
`feature_readout.py`, `breakdown.py`), `utilities/explanations_utils.py`,
`run_combo.py` wiring, design doc §3.4/§3.5/§3.6 as amended, dc01's SC.5
section (the matrix precedent and its row-11 deferral notes). Ledger
re-check at session start: no `open` findings anywhere. Green baseline
re-established this session: dc05 **t10 + t12 ALL PASS** (the two
explanation-surface pins; the full suite ran green at SC.4, same HEAD
lineage, 6a7a60b).

**Verified enabling fact (re-checked for this stage):** `cosine_type` is
fixed `'shifted'` in every config in `confs/hyper_params.py` (incl. all
three fU configs) — every post-hoc similarity assumption in the pipeline
matches the scored model.

**Carried obligations, status found:**
- **F-DC05-03(b) dual naming route (user side)** — implemented at the build
  commit (`pipeline.py:213–225`: post-hoc `name_prototypes_from_weights`
  over `items_in_user_proto_space()` runs beside the intrinsic pass,
  artifacts nested under `explanations/<scoring>/`), pinned by t10
  `fu_dual_route_user_side_written`, green this session. **Verified in
  context** — with one precision finding on the committed SC.8 spot-check's
  post-hoc arm (F-DC05-20 below).
- **run_combo auto-explanations flip** — still excluded (licensed staging,
  verified at SC.4); the license expires at this gate → F-DC05-18 below.
- **Dataset-list pinning** — resolved in §4 below.

### 1. Comparability matrix

Columns: the original ProtoMF models vs fU. ✓ = produced by the live
pipeline; — = absent with justification. Item-side rows are carried
unchanged from dc01's SC.5 matrix (fI column omitted here — no
cross-candidate ranking; its cells are dc01's record).

| # | artifact | `user_proto` (host) | `item_proto` | `user_item_proto` | `feature_user_proto` (fU) |
|---|---|---|---|---|---|
| 1 | top-k items per **user** prototype (CSV) | ✓ | — (a) | ✓ | ✓ (post-hoc route on the t-coefficient space — kept deliberately as the R2 comparison point) |
| 2 | top-k items per **item** prototype (CSV) | — (a) | ✓ | ✓ | — (a) |
| 3 | post-hoc naming, user side (lift over top-k aligned items) | ✓ | — (a) | ✓ | ✓ (dual route, F-DC05-03b — nested under its own scoring dir) |
| 4 | intrinsic naming, user side (cos(e_f, p^u_l)) | — (c) | — (c) | — (c) | ✓ (builder-owned vocab on both layouts, F-DC05-13) |
| 5 | naming transparency dumps (names + full stats CSVs) | ✓ | ✓ | ✓ | ✓ |
| 6 | TSNE, user-prototype space (named) | ✓ | — (a) | ✓ | ✓ (composed q_u cloud, batched accessor) |
| 7 | TSNE, item-prototype space (named) | — (a) | ✓ | ✓ | — (a) |
| 8 | weight_viz (paper's per-recommendation bars) | — (d) | — (d) | ✓ | — (d) |
| 9 | feature_small_multiples (item side) | — (a) | ✓ | ✓ | — (a — needs item prototypes; a user-side mirror was considered and declined, §3) |
| 10 | per-word exact shares + per-purchase regrouping (`history_readout`) | — (c) | — (c) | — (c) | ✓ |
| 11 | user-perspective recommendation breakdown (shared renderer) | ✓ (host slot, built at the dc05 build — the delivery dc01's row 11(e) deferred to this stage) | ✓ (dc01 stage) | ○ deferred to the fUfI merge stage (same (e) logic) | ✓ (fU slot; per-purchase zoom = figure default, word zoom in the md companion) |
| 12 | prototype cards, user side | ✓ (post-hoc route, declared in subtitle) | — (a) | ○ deferred (fUfI) | ✓ (intrinsic word-profile bars, same card geometry) |

Justifications: **(a)** host-structural — that prototype side does not
exist on the model (fU's item-side absences exactly mirror the
`user_proto` column: host-parity, GR7-symmetric). **(c)** algorithmically
impossible — requires learned word rows co-embedded with the user
prototypes; only the history composition puts them there (the candidate's
novelty; declared, not a gap). **(d)** projection branches exist only on
the UI double-tie. **No unjustified gap in the artifact set itself** —
dc01's one gap (intrinsic-only naming routing) was preempted for fU at the
build (dual route landed day one). The three findings below sit in wiring
precision, not artifact absence.

### 2. Algorithmic-novelty attribution gates (GR7)

Per new/altered artifact: *could original ProtoMF have produced this exact
artifact?*

- **Intrinsic user-prototype naming** (cos(e_f, p^u_l) via
  `name_prototypes_intrinsic`, side="user"): **NO.** Requires learned
  per-word embeddings sharing the user-prototype metric space; only
  q_u = Σ w̄_f·e_f puts them there. *Enabling property (thesis sentence):
  because user representations are sums of shared attribute-word rows,
  words and taste communities are co-embedded, so a community's meaning is
  read from parameters alone — no member sample, no top-k proxy.* → novelty.
- **Per-word exact shares of a community activation** (read-out 2): **NO.**
  Exactness is the linearity of q_u in its rows under the cosine's joint
  normalization; a free user vector has no composing rows. → novelty.
- **Per-purchase regrouping** (read-out 3, `per_purchase_rows`): **NO — and
  no fI analogue either.** Requires the composition weights to be sums over
  the user's own purchases (w̄_{u,f} = Σ_{i∈H_u} 1[f∈f_i]/|H_u|); neither
  the host nor fI has purchase structure inside a representation. → the
  candidate's genuinely-new dimension (§3).
- **Global word profile / naming stats CSVs (user side):** **NO** (same
  co-embedding requirement). → novelty.
- **Host breakdown slot (`compute_breakdown_user_proto`) + host user-side
  cards + the B(t) disclosure line on both slots:** **YES** — post-hoc
  machinery over quantities every U-host exposes → shared-pipeline work,
  backported by construction (built at the dc05 build, GR7-fair; the B(t)
  wording is identical on both slots). **Retro-invalidation sweep:** dc01's
  SC.5 matrix walked — **still-valid.** The dc05-stage additions are purely
  additive (no dc01 artifact altered; row 11(e)'s deferral is hereby
  delivered for `user_proto`, exactly as anticipated: "adding them later is
  slot work only"); the F-DC05-13 builder-vocab refactor touched dc01's
  label sources and was regression-verified green at SC.4 (dc01 t06/t08/
  t10/t12/i02).
- **Bags-layout vocab/label machinery (F-DC05-13):** shared infrastructure
  (serves fI and lightfm label surfaces too) — landed as a backport at
  SC.4, swept there and above. No further sweep due.
- **Dual naming route, user side:** the comparison itself requires the
  intrinsic arm (candidate-enabled); the post-hoc arm is the host's own
  route. Scrutiny diagnostic per the F-DC01-13 precedent — thesis use
  deliberately open.

**Verdict: four genuine novelties; the backports are the host slot/cards/
B(t) disclosure (already landed, sweep clean). No stale matrix anywhere.**

### 3. New interpretability axis + visualization

**The axis (one name): behavioral-evidence attribution.** Two halves:
(i) the user-side mirror of dc01's axis — taste-community meaning read from
parameters alone (intrinsic word profiles), closing exactly the surface
S0.2 left post-hoc on the user side; (ii) the genuinely-new half, which not
even fI has: every community activation decomposes **exactly** over the
user's own purchases — "you activate *dark-denim menswear* because of
these 12 purchases, +0.31 of 0.42" — recommendation evidence stated in the
user's own behavior, not in model abstractions. The host structurally
cannot produce (i) or (ii); fI cannot produce (ii) (items have no basket).

**Visualization:** already the fU breakdown slot's **figure-default zoom**
(ratified spec defaults, §3.4) — it will feed SC.9 as the showcase's right
panel, side-by-side with the host's declared-post-hoc nearest-users zoom.
The intrinsic user cards are the question-1 standing figure. **Nothing new
to build.** A user-side `feature_small_multiples` mirror (users carrying a
word, colored by community activation) was considered and **declined**: it
would duplicate what cards + TSNE labels already show, and the v1.2
compactness directive (repeated-use figures) argues against a third
user-side geometry figure. Recorded as a proposal only, revisitable if
SC.9 finds a gap.

### 4. Dataset-list pin (v1.2 requirement)

**Pinned for dc05's explanation artifacts and SC.6 runs:
`hm_1_month` (primary; canonical 5 fields, V=426, fixed layout) +
`ml-1m` (second testbed; charter-canonical genres+tags@0.8, V=1,061, bags
layout).** Both chains are end-to-end verified and pinned (t10/t11/t12;
F-DC05-13 closed the read-out half on bags). `hm_3_month` is NOT in the
pin — it remains the F-DC05-09 corroborative-only fallback under the
charter C2 written-reason clause. amazon2014 excluded (charter C2
amendment: no features). `ml-1m_cold` is staged data work; cold-run
scoping belongs to SC.6.

### 5. Gating-consistency check

Walked for fU across all three gate layers:

- `pipeline.EXPLAINABLE_MODELS` includes `feature_user_proto` ✓; accessor
  registered ✓; routing order (attr → intrinsic_item → intrinsic_user) is
  safe for fU (both item flags False) ✓.
- Explainer `supports()` vs accessor flags: naming/tsne/top-k/cards/
  breakdown include fU ✓ (has_user_prototypes, has_intrinsic_user_grounding);
  weight_viz excludes ✓ (has_projections=False); feature_small_multiples
  excludes ✓ (has_item_prototypes=False); user-side artifacts on the host
  ✓ symmetric. Failure modes loud (per-explainer try/except prints; bags
  vocab failure stays fatal rather than mislabel) ✓.
- **`run_combo.EXPLAINABLE_MODELS` still excludes `feature_user_proto`** —
  the documented staging whose license expires at this gate → **F-DC05-18**.
  `_noid`/`_debug` arms stay out per the headline-only convention
  (F-DC01-11 / F-DC05-15); noid artifacts reachable via the breakdown CLI;
  SC.8 key-normalization note stands.
- One vocabulary-parity gap found in the shared naming layer on the second
  testbed → **F-DC05-19**.

### 6. Findings (bundled proposals — decided at this gate)

**F-DC05-18 [minor] [dc05]** — `run_combo.EXPLAINABLE_MODELS` excludes
`feature_user_proto`; the build-time staging ("auto-explanations stay
opt-in until SC(dc05)'s explanation gate") reaches its expiry here. A
trained fU headline run currently produces no explanation artifacts unless
the pipeline is invoked by hand — the exact F-DC01-11 shape.
**Proposal:** add `feature_user_proto` to the set (headline model only;
arms CLI-reachable, host convention); update the staging comment; extend
dc05 t10 with the run_combo membership pin (dc01's t10 pins its own the
same way).

**F-DC05-19 [minor] [shared]** — the ml-1m naming defaults exclude `tags`:
`NAMING_DEFAULTS["ml-1m"]` = `[genres]` predates the S0-build-extension
`item_features.csv`, which now carries the pipe-separated `tags` column —
while the fU intrinsic vocabulary on ml-1m is 98% tags (1,043 of 1,061
codes). Consequences: (a) the dual-route profile-validity comparison on
the second testbed compares tag-dominated intrinsic profiles against
genres-only post-hoc names — an 18-value vocabulary cannot validate a
1,061-token profile; (b) the host `user_proto`'s ml-1m naming and cards
are genres-only — an under-steel-manned baseline (GR7). No wrong output
today (genres-only names are correct as far as they go); the gap is
comparison power.
**Proposal:** add `FeatureSpec("tags", multi_value_sep="|")` to the ml-1m
naming defaults (shared route — serves every ml-1m model identically;
lift selection + max_descriptors already bound name length); pin in t12
(post-hoc naming stats on ml-1m contain tag descriptors). Retro-sweep
note: no ml-1m run artifacts exist yet anywhere (fleet rows not yet
submitted), so nothing is retroactively invalidated; dc01's matrix is
hm-pinned and unaffected.

**F-DC05-20 [minor] [dc05]** — spot-check post-hoc-arm precision: the
committed F-DC05-03a wording names "post-hoc top-k-member-users
purchase-lift profiles" as the SC.8 validation arm, but the landed
mechanization (the dual naming route) implements the **aligned-items**
lift profile (feature lift over the top-k items aligned with p^u_l in
t-space — the host's own naming route). These are two distinct post-hoc
readings with different strengths: aligned-items is the **host-parity**
naming comparison (identical route to how the host itself is named —
like-for-like by construction) but is mediated by the trained t
coefficients; member-users purchase-lift is the **data-grounded**
reading ("what members actually buy" — behavioral, not parameter-mediated)
and already exists as machinery in the host breakdown zoom
(`compute_breakdown_user_proto`: nearest users → consumed-item lift).
**Proposal (doc precision, no code now):** dated amendment to §3.4
read-out 4 distinguishing the two arms and assigning both to the SC.8
spot-check — primary mechanized arm = the dual naming route (aligned
items, landed); secondary data-grounded arm = member-user purchase-lift
profiles, computed at SC.8 from the existing breakdown-zoom machinery
(any small helper is decided there, where the checkpoint exists).

### 7. Observations recorded for SC.8 (working evidence, no action here)

- The SC.8 rendered-explanation spot-check (protocol SC.8.4) has its dc05
  shape ready: ≥5 user-prototype cards (both routes), ≥3 fU breakdowns,
  ≥3 side-by-side fU-vs-`user_proto` breakdowns (same user, same item) —
  all machinery pinned; plus the F-DC05-03a profile-validity comparison
  (both arms per F-DC05-20's split, both ID arms per the F-DC05-15 note).
- On ml-1m the per-purchase zoom will render movie titles over heavy
  baskets (median |H_u| = 56; top-6 + remainder discipline carries the
  fold) — the figure default was ratified on H&M thin baskets; eyeball the
  first real ml-1m render deliberately (learnings `[fU-cycle dc05 build]`:
  slot layout tolerances are part of the contract).
- B(t) magnitude expectations (M6′ instruments) and the M5′ angle-to-mean
  precondition figure are SC.8 items on the real checkpoints, unchanged.

### 8. Gate

**Status: OPEN — awaiting user review.** Decisions requested:

1. Accept the comparability matrix incl. absence justifications (§1).
2. Accept the attribution-gate verdicts: four novelties; host-slot/cards/
   B(t) backports already landed; dc01 matrix swept **still-valid** (§2).
3. Accept the axis ("behavioral-evidence attribution") + the decision that
   its visualization is already built (per-purchase zoom + intrinsic
   cards); user-side small-multiples declined (§3).
4. Ratify the dataset pin: **hm_1_month + ml-1m** (§4).
5. Approve F-DC05-18 (run_combo flip + pin), F-DC05-19 (ml-1m tags in the
   shared naming defaults + pin), F-DC05-20 (two-arm spot-check precision
   amendment).
6. On approval: fixes land supervised, tests green, candidate index →
   **hardened**, artifact finalized before SC.6.

### Gate closure + application round (2026-07-13, same sitting)

**Gate CLOSED — decisions (user, after a plain-language walkthrough of every
point):**

1. Comparability matrix + absence justifications: **accepted** (walkthrough
   covered the three absence classes and the fairness logic).
2. Attribution verdicts: **accepted**, with a recorded clarification at the
   user's probe: a "Yes" (backport) verdict governs **credit, not
   existence** — backported artifacts are kept, built more widely, and may
   be claimed in the thesis as the thesis's own methodological contribution
   (the shared fairness-contracted comparison framework); only attribution
   to the *candidate's algorithm* is restricted. Both piles are load-bearing;
   the B(t) disclosure line additionally carries host-class scientific
   content in its own right.
3. Axis + no-new-viz + small-multiples declination: **accepted**, with the
   user's scoping note: presentation-level tuning of these artifacts is
   deliberately deferred to the results stage and thesis writing.
4. Dataset pin **hm_1_month + ml-1m**: **ratified**.
5. F-DC05-18/-19/-20: **"approve all."**

**Fix application (same day):**

- **F-DC05-18** — `feature_user_proto` added to `run_combo.EXPLAINABLE_MODELS`
  (headline only; staging comment replaced by the wired-at-SC.5 record);
  t10 §7 pins membership + noid exclusion.
- **F-DC05-19** — `FeatureSpec("tags", multi_value_sep="|")` added to the
  ml-1m naming defaults (comment records the F-DC05-19 rationale); t12 §6
  pins defaults-include-tags, pipe separator, and a real-items_info post-hoc
  pass carrying tag descriptors (dtype note: the pin casts item_id to int,
  matching `load_items_info`).
- **F-DC05-20** — dated precision amendment applied to design doc §3.4
  read-out 4: both post-hoc arms committed to the SC.8 spot-check with
  declared roles (aligned-items = mechanized like-for-like, landed;
  member-users purchase-lift = data-grounded arm, computed at SC.8 via the
  existing breakdown-zoom machinery).

**Verification:** dc05 t10 (incl. new §7) + t12 (incl. new §6) ALL PASS;
regression dc01 t05 / t10 / t12 ALL PASS (their run_combo pins assert
membership, not the exact set — unaffected by the fU addition).

Ledger: F-DC05-18/19/20 → **fixed**. Candidate index → **hardened**.
→ Next step: **SC.6** (cluster-run plan — requires the S0.7 reference
fleet's frozen table, still in flight on LEO5; per the early-submission
directive the ml-1m canonical hyperopt rows + ml-1m_cold scp remain the
pull-forward candidates, proposals at the SC.6 gate).

---

## SC.6 Cluster-run plan (2026-07-15)

> Executed same-day as dc01's SC.6 (its 9-job fleet is PENDING on LEO5 at
> writing). Protocol v1.2, SC.6 section re-read. The dossier header's
> precondition declaration is DISCHARGED this step: S0.7 part 2 landed
> 2026-07-15 (S0 COMPLETE), re-check against S0 performed below.

### 0. Preamble — S0 re-check + open findings + learnings

- **S0.7 part 2 landed:** the frozen reference table exists
  (`s0_foundation.md`, 2026-07-15). The fU stage bar is **`user_proto`
  warm 0.5725 / 0.3322 (HR@10/N@10), cold-cold 0.0000 (tie floor, R-a),
  +kNN 0.2999**. Feature baselines: `lightfm_tags` 0.4582 warm / 0.4705
  cold-cold; `lightfm_tags_ids` 0.5087 / 0.5080; context row
  `user_item_proto` 0.6587 warm. Interpretation rules R-a..R-g accompany
  all readings; two-tier numbers policy applies throughout.
- **Open-findings re-check:** no `open` findings anywhere in the ledger.
  Constraining wontfix conventions for THIS step: F-S0-10 (explicit
  `--profile dev` on every submission line), F-S0-02/12 (LEO5-only eval
  regime), F-DC05-16 (ml-1m cold machinery deferred — not this routine's
  work), F-DC05-04 (§4 numbers = working basis only).
- **Learnings applied:** file-level dataset-presence checks before
  feature-aware submissions (`item_features.csv` explicitly — the ml-1m
  copy was scp'd + md5-verified during the dc01 SC.6 sitting);
  smoke-one-before-batch; one `/leo5-submit` per run (hard gate);
  config-schema side-harness blast radius (t04/t05/t09 pins current).
- **Walltime inputs:** `Master/temp/dc01_dc05_walltime_probes_2026-07-15.md`
  — **fU costs ×2.76 over its host on hm_1_month** (padded per-user history
  gather, D_max=112; free on ml-1m's bags layout, ×0.98). Anchors:
  `user_proto` hm dev fleet 3h22m, its cold retrain 2h04.
  **Correction recorded against the memo's blanket retrain sizing:** the
  memo recommends 4:00:00 for all dc01/dc05 cold retrains from the S0.7
  range (0h45–2h04), but fU's ×2.76 applies to retrains too —
  2h04 × 2.76 ≈ 5h42m > 4h. fU-arm cold retrains are sized **8:00:00**
  here (Wave B below).

### 1. Dataset pin (decision requested at this gate)

Per charter C2 (amended 2026-07-12) the per-candidate list pins at
SC.5/SC.6; dc05's SC.5 ratified **hm_1_month + ml-1m** for the explanation
instrument. **Proposed run pin: `hm_1_month` (+ `hm_1_month_cold`) primary;
`ml-1m` (warm) second quantitative testbed; `ml-1m_cold` NOT pinned.**

- `hm_1_month`: primary — the thin-history regime where the central bet
  lives (25.1% of V1 users at the 3-row train floor; D4 history-length
  strata are the committed readout).
- `ml-1m` (warm): the complementary regime (heavy histories × coarse
  vocabulary) — probes exactly the M5′ mean-regression / omnipresence /
  B(t) end the H&M testbed cannot reach (median |H_u| = 56 vs 5). Written
  reason per C2: the charter's own second-testbed amendment; the fU
  read-out chain is ml-1m-proven (F-DC05-13 bags support, t12).
- `ml-1m_cold`: not pinned, adopting the three grounds ratified at dc01's
  SC.6 gate verbatim (no valid CF referee column until the bags-aware kNN
  exists — F-DC05-16 deferral honored; the "interaction-hidden, not
  new-release" epistemic caveat; deferral closes no door). Additionally
  fU-specific: dc05's cold-item story is *host behavior preserved* (§R5 —
  the item side is untouched), so the H&M cold columns carry the whole
  claim; an ml-1m cold row would add nothing dc05-specific.

### 2. Run manifest

All runs: branch `feat/scrutiny` (commit hash recorded at submission), seed
38210573, uniform-99 negatives, `use_bias=0` (explicit), `feature_fields:
'canonical'` (hm → S0.5 five, V=426 fixed layout; ml-1m → genres+tags@0.8
bags, V=1,061), selection on val `hit_ratio@10`, dev profile (30/60/7/4),
explicit `--profile dev`, conc 5, num-workers 1. One run = one job = one
`/leo5-submit`. Fleet tag `dc05-sc6` (ml-1m fleet row additionally
`s0-fleet-ml1m`).

**Wave A — dc05 hyperopts, hm_1_month (primary):**

| # | Run | Config | Est. (A30) | `--time` |
|---|---|---|---|---|
| R1′ | fU headline | `feature_user_proto` (ids, canonical 5) | ~9h17m | 1-00:00:00 |
| R2′ | ablation | `feature_user_proto_noid` (C7 pair) | ≤R1′ | 1-00:00:00 |

**No f0-analogue arm (recorded reasoning):** dc05's design doc commits the
ids/noid pair ONLY (§3.6); the C5′ keystone fU(V=0,+ID) ≡ `user_proto` is
pinned bit-identically by tests (4b C5, dc_checks), not run as a fleet arm.
dc01's R3 noise yardstick (wrapper/init-path noise at identical
architecture, single seed) was ratified as measured ONCE on the primary
testbed; its reading transfers to dc05 as fleet-level calibration with the
declared limitation that it was measured on the I-host wrapper. Registering
an fU-f0 arm would exceed C7 (not a design-doc-committed ablation) — raised
at the gate as an option only if the user wants the yardstick re-measured
on the U-host wrapper.

**Wave B — cold retrains + evals, hm_1_month_cold (S0.3 §6 convention;
structurally dependent on Wave A best configs — submitted at Wave A
completion):**

| # | Run | Source config | Est. (A30) | `--time` |
|---|---|---|---|---|
| R3′ | fU cold retrain | R1′ best, `--retrain-config`, `--profile dev` | ~5h42m | 8:00:00 |
| R4′ | noid cold retrain | R2′ best, same convention | ≤R3′ | 8:00:00 |

Then cold evals (`python -m utilities.cold_eval`, login-node A30,
small-workload policy) with `--canonical-results-dir` at the R1′/R2′ dirs
(F-S0-11 warm-delta row). **Committed reading (mirror claim):** fU leaves
the item side untouched, so its native cold-item behavior is expected
host-class (tie floor per R-a; the kNN column is the referee — fU's item
branch is CF, the attr-kNN patch applies). The comparison R3′/R4′ vs the
frozen `user_proto` cold row *verifies the mirror claim empirically* rather
than asserting it — deviations are findings, not surprises to explain away
(empirical-not-scholastic). The item-side ID drop is a no-op on fU
(`config_expects_id_drop` false — item branch is not tags+ids); the
cold-USER drop mirror in the fU code is fold-in machinery, exercised by
tests, not by this eval (no cold-user testbed this cycle, §R5 honest-NO).

**Wave C — dc05 hyperopts, ml-1m (independent of Wave A):**

| # | Run | Config | Est. (A30) | `--time` |
|---|---|---|---|---|
| R5′ | fU headline | `feature_user_proto` | ~1h34m | 4:00:00 |
| R6′ | ablation | `feature_user_proto_noid` | ~1h34m | 4:00:00 |

**Wave D — ml-1m reference-fleet row (charter C6 lazy addition — left to
this plan by dc01's SC.6 gate decision 3, "no cross-candidate work"):**

| # | Run | Config | Role for dc05 | Est. (A30) | `--time` |
|---|---|---|---|---|---|
| R7′ | `user_proto` ml-1m | host search space | the fU stage bar on testbed 2 | ~1h36m | 4:00:00 |

`item_proto`, `mf`, `lightfm_tags(_ids)` ml-1m rows are ALREADY QUEUED under
dc01's plan (jobs 7041255/56/61/63) — fleet rows, computed once, consumed
here without re-run (C6). Popularity reference follows the ml-1m_cold
enablement (unchanged).

**Post-run measurements (SC.8 scope, named for completeness — login-node
small workloads on produced checkpoints):** D3 pass + **D4 history-length
strata** (the central-bet readout: thin-user bands vs `user_proto`, both
arms); the committed M5′ instrument set (angle-to-mean-direction per
history-length band — the standing precondition figure; metadata-part norm
+ ID-share vs |H| strata, F-DC05-01); cloud spread/coverage BOTH
user-uniform and interaction-weighted (F-DC05-02); B(t) variance share
fU-vs-host on the same instrument (F-DC05-06 comparative rule); gauge-mass
/ activation-spectrum decomposition both arms (§3.8/C7′); profile-validity
spot-check, BOTH post-hoc arms (aligned-items dual route, landed; +
member-user purchase-lift, F-DC05-20); rendered-explanation spot-check
incl. ≥3 shared-renderer breakdowns fU-vs-`user_proto`, same user/item
(protocol SC.8.4); same-day multi-item share readout (F-DC05-09 working
basis). Auto-explanations for R1′/R5′ ride run_combo (F-DC05-18); first
real ml-1m render eyeballed deliberately (SC.5 §7 note).

### 3. Charter compliance (clause by clause)

- **C1 single lineage:** all runs from `feat/scrutiny`; this artifact
  committed before submission so the recorded hash contains it.
- **C2 datasets:** hm_1_month + hm_1_month_cold primary; ml-1m per the
  second-testbed amendment with §1's written reason; ml-1m_cold not
  invoked; hm_3_month not invoked (F-DC05-09 stationarity caveat never
  triggered); amazon2014 out of feature-aware scope.
- **C3 eval protocol:** temporal LOO, 1+99 uniform, HR/NDCG@{1,3,5,10,50},
  val `hit_ratio@10` selection, `use_bias=0` explicit, cold eval per S0.3
  (cold-vs-cold primary, single-config retrain, training-negative
  exclusion, warm-val stopping); D3/D4 at SC.8.
- **C4 budget & seeds:** dev 30/60/7/4 on every hyperopt; single seed
  38210573; no multi-seed; one run = one job; envelope check §4. The 24h
  limit on Wave A stays within the >1-day soft envelope per run (9h17m
  estimate; the limit is margin, not expectation).
- **C5 features:** `'canonical'` sentinel → per-dataset canonical sets; no
  price_band, no product_code; no per-candidate deviation.
- **C6 baselines:** hm comparisons against the FROZEN table only — nothing
  re-run. R7′ enters under the lazy-addition amendment (this is its
  first-citing plan); dc01's queued ml-1m rows consumed as fleet property.
- **C7 ablation discipline:** exactly the design doc's committed pair
  (ids/noid, both testbeds); no f0-analogue (reasoning §2); no exploratory
  sweeps; ids-vs-noid interpretation pre-committed (design doc §3.6
  three-part rule, M5′/M3′ instruments joined at SC.8).
- **C8 manifests:** every run records model/config, resolved feature fields
  + layout, `use_id_feature`, negatives, seed, profile, dataset(+variant),
  branch+commit, `use_bias`. SC.8 audits against this list — **including
  the GPU-partition deviation below if ratified.**

### 4. Compute estimate vs the envelope

7 SLURM jobs total: Wave A ~18.6h + Wave B ~11.4h + Wave C ~3.2h + Wave D
~1.6h ≈ **35 GPU-hours (A30-anchored; A100 placement shrinks it), largest
single job ~9h17m** (limit 24h, 2.6× margin — fU's ×2.76 ratio was measured
solo at the top of the search ranges; 5-way GPU sharing may shift it, the
margin absorbs it). Envelope (vault 2026-06-07_2310): soft flag >1 day per
run — none (largest estimate 9h17m); hard limit >10 days — irrelevant.
Login-node evals + readouts: minutes each.

### 5. Declared deviations (decision requested at this gate)

**D-1 — GPU placement: agnostic queueing (fleet-geometry deviation;
user decision at this gate, superseding the A100 proposal).**
The S0.7 fleet and dc01's 9 queued jobs are pinned A30; at writing the A30
line is congested (live queue check this sitting: the only typed-GPU
pending jobs are dc01's own 9; pressure is cluster-wide — ~1,900 CPU jobs
competing for node cores). Decision: this fleet queues **untyped**
(`--gres=gpu:1`, any free card — verified in live use on LEO5; node
features are null so a same-family constraint is not expressible).
Rationale (user): hardware choice matters only for **runtime estimation**;
scores are card-independent within the fleet's proven geometry. Accepted
residuals, declared: jobs may scatter across card types, including the A40
(GA102/CC 8.6 — outside the A30/A100 silicon family); wall-time anchors
fragment into per-card families. Mitigations: (a) all resource requests
stay sized for the SMALLEST card (A30 24 GB benchmark geometry: conc 5,
num-workers 1, 10 CPUs, 64G — explicit flags, card-independent; bigger
cards only add headroom); (b) `--time` limits stay A30-anchored (margins
only grow); (c) **the landed card is recorded per job** (job log prints
hostname + `nvidia-smi` name; copied into the run-submission record) so
every future runtime estimate knows its anchor family. Mechanics: 3-line
extension to `slurm/run_combo.slurm` (`--gpu-type any` → untyped gres),
`--dry-run`-verified, committed with this artifact (our own script, not
upstream). Carried as a declared deviation in this dossier + the C8 audit
at SC.8; the one-job same-card probe remains the named remedy if any
comparison ever looks hardware-suspicious.

**D-2 — mid-fleet pull (dc01 hash-hold convention).** dc01's submission
record holds the LEO5 checkout at 4d33617 until its fleet completes.
Submitting this plan requires commit+push+pull while dc01's jobs are
PENDING → those jobs will record THIS plan's hash in their C8 manifests
instead of their own plan's. Delta 4d33617→HEAD is **docs-only** (verified
at this sitting: dossier/docs commits, no code), so run behavior is
untouched; the deviation is bookkeeping. Remedy: dated amendment to dc01's
run-submission record noting the recorded-hash shift + this justification,
applied in the same commit as this artifact.

### 6. Preconditions & submission discipline

1. **Commit + push this artifact; `git pull` on LEO5**; verify HEAD ==
   submitted hash. D-2 amendment rides the same commit.
2. **File-level data checks on scratch:** hm_1_month splits +
   `item_features.csv`; hm_1_month_cold variant files (Wave B); ml-1m 6
   split files + `item_features.csv` (md5-verified at the dc01 sitting —
   re-verify cheaply).
3. **`/leo5-load` before the first submission** — decides D-1's A100 vs
   fallback; recorded in the run-submission record.
4. **Smoke-one-before-batch (login-node A30, quarantined under
   `/scratch/c7031336/smoke_runs/dc05_sc6/`):** the fU *hyperopt* path has
   never run on the cluster (timing probes covered the retrain-config path
   only; dc01's smokes covered fI + lightfm). Two smokes: fU × hm_1_month
   (fixed layout + history-weights builder inside Ray trials) and
   fU × ml-1m (bags layout ditto), ~2 trials × 3 epochs each. Known
   toy-budget edge on record (dc01 smokes): nonzero exit at
   best-checkpoint extraction when no toy trial improves — same verdict
   template applies (path-clean = GO).
   **Resource-measurement duty (added at this gate, user directive):** fU
   has NO replication-report benchmark row — the submit skill's VRAM/RAM
   estimation has no measured anchor for it (nearest anchor: the host
   `user_proto`, S0.7-proven at conc 5 on A30, plus the architectural
   delta, expected small). The smokes therefore MEASURE, not just
   path-check: (a) peak GPU memory per trial at fleet batch settings
   (`torch.cuda.max_memory_allocated` / `nvidia-smi` sampling), pass bar
   ≤ ~3.8 GB/trial (5 × trial ≤ 80% of A30 24 GB — the smallest card the
   agnostic queue can land on); (b) peak node RAM vs the 64G request.
   **GO requires path-clean AND footprint-fits.** Contingency, decided at
   a gate, never improvised: if 5 trials cannot fit an A30, concurrency is
   NOT silently lowered (fleet-pinned — search-dynamics knob); the options
   returned to the user are typed bigger-card queueing vs a declared
   concurrency deviation. Measured numbers land in the run-submission
   record as fU's first resource anchors. R7′ (`user_proto` ml-1m) needs no
   dedicated smoke: CF model, no feature machinery, host hyperopt path
   proven on hm at S0.7 and the ml-1m dataset path exercised by dc01's
   queued R8 (`item_proto`).
5. **One `/leo5-submit` per run**, explicit `--profile dev` (F-S0-10).
6. **Sequencing:** after smokes GO → Waves A, C, D queue immediately
   (5 jobs); Wave B (R3′/R4′) at Wave A completion — same sitting as
   dc01's R4/R5 if the waves land together.

### 7. What SC.8 will consume

R1′/R2′ vs frozen `user_proto` (stage bar) / `lightfm_tags(_ids)` /
`user_item_proto`-as-context (warm, hm); R3′/R4′ vs the frozen cold columns
under R-a..R-e (mirror-claim verification); R5′/R6′ vs R7′ + dc01's
R8/R10/R11 rows (warm, ml-1m); D4 history-length strata joined per the §3.6
interpretation rule; the M5′/B(t)/gauge instrument set on both arms; dc01's
R3 noise-yardstick reading as fleet-level calibration (declared I-host
limitation). Anomaly handling: at most one documented rerun round (Ground
rule 6).

### Gate (SC.6)

**Status: OPEN — decisions one at a time (dc01-gate convention):**

1. **Dataset pin (§1):** hm_1_month(+cold) + ml-1m warm; ml-1m_cold not
   pinned (dc01's three grounds + the fU mirror-claim ground).
   **→ RATIFIED 2026-07-15**, with a recorded user expectation: the
   ml-1m cold machinery (bags-aware kNN/tie fallback, the F-DC05-16
   enablement work-package) **will likely be wanted for the thesis** —
   deliberately deferred until the current fleets are confirmed working,
   then scheduled as its own work-package (not inside any SC routine).
2. **Run manifest (§2):** R1′–R7′; no f0-analogue arm (option available on
   request); Wave-B mirror-claim reading; the 8:00:00 fU retrain sizing
   (memo correction).
   **→ RATIFIED 2026-07-15** after a plain-language walkthrough of the
   empty-control question (what the noise yardstick buys, when it matters —
   only if the fU-vs-host gap lands small). User decision: **no fU empty
   run** — dc01's R3 draw serves as the borrowed fleet-level yardstick
   (declared I-host limitation). If SC.8 finds the fU gap inside the
   measured wobble, adding the fU control then remains open as the one
   documented follow-up — deferred, not committed. The 8h fU-retrain
   sizing accepted (memo's blanket 4h corrected for the ×2.76 tax).
3. **Deviations (§5):** D-1 A100 placement (+ fallback rule), D-2 mid-fleet
   pull with dc01-record amendment.
   **→ RATIFIED 2026-07-15, D-1 SUPERSEDED at the gate (user decision):**
   after a walkthrough of queue mechanics (live check: only typed-GPU
   pending jobs are dc01's own; pressure is cluster-wide CPU jobs) and the
   smallest-resource/backfill question, the user chose **agnostic
   queueing** over the A100 pin — §5 D-1 rewritten in place to the
   ratified form (untyped gres, A30-sized requests, per-job card
   recording, A40/scatter residuals accepted, wrapper extension). D-2
   approved as proposed (plain-language "what is a checkout" walkthrough
   given; docs-only delta verified).
4. **Discipline (§6):** smoke set (fU hyperopt path ×2), then A/C/D queue
   immediately; B at Wave-A completion.
   **→ RATIFIED 2026-07-15, EXTENDED at the gate (user directive):** the
   smokes gain the resource-measurement duty (§6.4 as amended) — fU has no
   replication-report benchmark row, so the smokes measure peak GPU
   memory/trial + node RAM against the A30-sized requests; GO = path-clean
   AND footprint-fits; measured numbers become fU's first resource
   anchors. Login-node placement confirmed explicitly.

On approval: commit artifact (+D-2 amendment) → push → LEO5 sync →
`/leo5-load` → smokes → submissions, each via its own `/leo5-submit`.

**Gate CLOSED 2026-07-15 — all four decisions ratified one at a time
(dataset pin with the ml-1m-cold thesis expectation recorded; manifest
with no-empty-run + 8h retrain sizing; deviations with D-1 superseded to
agnostic; discipline with measuring smokes). Execution proceeds this
sitting; run-submission record follows below.**

### Run-submission record (2026-07-15, same sitting)

**Preconditions executed:** artifact + wrapper extension committed
(5936aae + 68bfe4f) and pushed (user authorization given in-session); LEO5
pulled to 68bfe4f (verified HEAD == local before every submission);
**cluster checkout now held at 68bfe4f until both fleets complete** (the
D-2 note in dc01's record covers the shift its pending jobs inherit);
file-level data checks passed (hm_1_month, hm_1_month_cold, ml-1m — all
splits + `item_features.csv` each); `/leo5-load` 19:21: A100 0/6 free,
A30 36/42 free, A40 4/4 free, 1,360 pending (only GPU-typed pending jobs
= dc01's own 9 → priority-bound, not resource-bound; the agnostic call
stands, the superseded A100 pin would have queued behind 6/6 busy cards).

**Wrapper `--gpu-type any` dry-run-verified** (untyped `gpu:1` emitted;
typed default regression-clean) before live use.

**Login-node smokes (measuring, per the §6.4 amendment), quarantined under
`/scratch/c7031336/smoke_runs/dc05_sc6/`:**
- One infrastructure lesson first: initial launch died at Ray startup —
  AF_UNIX socket paths cap at 107 bytes and the quarantine dir exceeded it;
  smoke-script-only issue (fleet jobs use short `/scratch/.../ray_tmp/`),
  fixed by pointing Ray's tmpdir at node-local `/tmp`.
- fU × hm_1_month (fixed layout, hyperopt path in Ray): **full PASS,
  exit 0** — 2 trials × 3 epochs trained, best-checkpoint extraction +
  results persistence worked end-to-end. **GPU peak 829 MiB total for 2
  concurrent trials (~415 MB/trial)** vs the 3.8 GB/trial pass bar (~9×
  headroom; host user_proto benchmarks in the same band, so worst-case
  search draws stay far under the bar). RAM: summed-RSS peak ~19.4 GB for
  2 trials (measure overcounts shared pages; benchmark-grade hm figure is
  22–27 GB at conc 5) — 64G request holds. **GO.**
- fU × ml-1m (bags layout, hyperopt path in Ray): **full PASS, exit 0** —
  same chain; GPU peak 895 MiB / 2 trials, RAM profile equivalent. **GO.**
- These are fU's **first resource anchors** (no replication-report row
  exists); banked here per the gate directive.

**Submissions (one `/leo5-submit` each, all dev profile, seed 38210573,
conc 5, num-workers 1, `--gpu-type any` (D-1), 64G, tag `dc05-sc6`; R7′
additionally `s0-fleet-ml1m`):**

| # | Job | Model × dataset | `--time` | gres |
|---|---|---|---|---|
| R1′ | 7050748 | feature_user_proto × hm_1_month | 1-00:00:00 | gpu:1 (untyped) |
| R2′ | 7050778 | feature_user_proto_noid × hm_1_month | 1-00:00:00 | gpu:1 (untyped) |
| R5′ | 7050802 | feature_user_proto × ml-1m | 4:00:00 | gpu:1 (untyped) |
| R6′ | 7050834 | feature_user_proto_noid × ml-1m | 4:00:00 | gpu:1 (untyped) |
| R7′ | 7050861 | user_proto × ml-1m (fleet row) | 4:00:00 | gpu:1 (untyped) |

Queue verified post-submission: all 5 PENDING with untyped `gres:gpu:1`
beside dc01's 9 A30-typed jobs. **Landed-card bookkeeping obligation:**
when each job runs, its log's `nvidia-smi` line names the card — to be
copied into this record (D-1 mitigation c) at SC.8 or at the Wave-B
sitting, whichever comes first.

R3′/R4′ (hm_1_month_cold retrains, 8:00:00 each per the corrected sizing)
follow at Wave A completion (`--retrain-config` from R1′/R2′ best
configs); cold evals on the login node with `--canonical-results-dir`.
→ Next protocol step: **SC.7** (flush & retrospective), separate sitting.

### Wave completion + landed cards + Wave B submission (2026-07-19)

**All 5 hyperopts COMPLETED, `Run status: OK` each** (R1′ 2:52:24,
R2′ 2:45:03, R5′ 1:18:32, R6′ 2:03:18, R7′ 2:00:01). **Landed cards
(D-1 mitigation c, from the job logs' nvidia-smi lines): ALL FIVE ON
A30** — the untyped requests resolved to the convention card on their
own; the D-1 fleet-geometry deviation is **empirically vacuous for this
fleet** (zero hardware asymmetry vs the frozen baselines; SC.8's audit
can record it as such). Walltime note for the anchor table: R1′ ran
2h52m on A30 vs the 9h17m probe-scaled estimate — the solo-measured
×2.76 gather tax largely does not materialize under dev-fleet hyperopt
conditions (ASHA truncation + sampled configs below the probe's
top-of-range point); the probe method over-forecasts fU fleet walltime
~3×, noted for thesis-tier sizing.

**Warm test numbers extracted** (working evidence, two-tier policy;
interpretation belongs to SC.8): fU 0.5832/0.3496 and fU-noid
0.5833/0.3507 vs the frozen `user_proto` bar 0.5725/0.3322 on hm (both
arms above the bar; ids-vs-noid gap ≈ 0 — the ID row buys nothing on
thin-history hm); ml-1m fU 0.5491, noid 0.5424 vs `user_proto` (R7′)
0.5668 — a modest tax in the heavy-history regime. dc01's R3 noise
yardstick read |f0 − item_proto| ≈ 0.003 HR@10. **SC.8 verification
obligation flagged:** confirm R1′/R2′ selected genuinely different best
configs (the near-tie is config-plausible but must be checked before
any "grounding is free" reading).

**Wave B submitted (2026-07-19, one `/leo5-submit` each; preconditions:
both source `config.json`s present, cold variant files on scratch,
cluster checkout verified still held at 68bfe4f):**

| # | Job | Run | `--time` | gres |
|---|---|---|---|---|
| R3′ | ~~7123801~~ → **7124943** | fU cold retrain (R1′ best config) | ~~8:00:00~~ → **12:00:00** | gpu:1 (untyped) |
| R4′ | ~~7123802~~ → **7125107** | fU-noid cold retrain (R2′ best config) | ~~8:00:00~~ → **12:00:00** | gpu:1 (untyped) |

**Sizing correction (same sitting, user prompt "does 8 hours make
sense?"):** evidence check found BOTH winning configs = dim 81 / K 76 /
batch 256 — exactly the walltime probe's measurement point (~368 s/epoch
solo, the full ×2.76 tax) — and the warm winner ran the complete 60-epoch
budget with no early stop. Worst-case retrain ≈ 7–7.5h vs the 8h limit
(<10% margin, convention wants 2×). `scontrol update TimeLimit` upward is
permission-denied for users on LEO5 (only reductions allowed), so both
never-started jobs were cancelled and resubmitted at 12:00:00 (one
`/leo5-submit` each; no compute lost, queue-age loss ~minutes). Identical
winning configs also partially explain the warm ids-vs-noid near-tie —
the SC.8 config-check obligation is hereby half-answered: same
architecture won both searches; the runs differ only in `use_id_feature`.

Queued beside dc01's R4/R5 (7121151/52, submitted separately at that
fleet's Wave-A completion). After completion: cold evals on the
login node with `--canonical-results-dir` at the R1′/R2′ dirs, then
`/scrutiny resume dc05` → SC.8.

### Wave B landed + cold rows (2026-08-15)

Both cold retrains **COMPLETED on 2026-07-19** — R3′ (7124943) 1:57:11 and
R4′ (7125107) 1:35:55, well inside the resized 12h (the resize was correct
insurance but ~6× oversized; the ×2.76 gather tax again did not materialize).
Both ran **wandb-online without incident**, which retires dc01's cross-noted
expectation that they would hit the same hang: that hang was CPU starvation
from `--gpu-per-trial 1.0` (2 CPUs requested), and dc05's runs requested 10.
See dc01's "Wave B resolution" section — dc05's geometry is the one that
proved correct, and dc01 was resubmitted to match it.

**Cold rows (login-node `cold_eval`, `--canonical-results-dir`, 2026-08-15):**
fU is a user-side mechanism, so its item side is a plain ID embedding and the
attr-kNN patch applies as it does to every CF row — the patched line is the
one that carries meaning; unpatched cold-vs-cold is 0.0000 for both arms,
matching `user_proto` / `item_proto` / `user_item_proto` exactly (untrained
cold ID embeddings; established shape, not a defect). *(Dated amendment
2026-08-20, F-S0-15: the honest value of these no-signal blocks is the
**chance floor (~0.10)**, not a structural zero — the 0.0000 point is a
sigmoid tie-fabrication artifact (distinct raw logits merged to bit-equal
0.5 in fp32, then resolved by the argpartition convention). Raw-logit
brackets measured 2026-08-20: fU 0.0955, fU-noid 0.1133. No claim rests on
these blocks; quoting rule F-S0-15(a) applies wherever they appear.)*

| block | fU | fU-noid |
|---|---:|---:|
| warm test (variant) | 0.6249 / 0.3783 | 0.6222 / 0.3769 |
| cold-vs-cold (unpatched) | 0.0000 | 0.0000 |
| **attr-kNN cold-vs-cold** | **0.3265 / 0.1733** | **0.3418 / 0.1796** |
| attr-kNN cold-vs-all | 0.2829 / 0.1139 | 0.2998 / 0.1192 |
| popularity floor | 0.1202 / 0.0562 | 0.1202 / 0.0562 |

Both arms clear their host `user_proto` (0.2999 patched) and the popularity
floor; both sit below the LightFM rows. The ids-vs-noid near-tie of the warm
fleet repeats here (0.3265 vs 0.3418) — consistent with the already-recorded
fact that both searches selected the same architecture (dim 81 / K 76) and
differ only in `use_id_feature`.

**F-S0-14 applies to these rows too.** dc05's tie-block exposure is moderate
rather than extreme (fU twin share 0.482 / 9.42 distinct sigs; noid 0.359 /
9.60) — far from dc01-noid's 0.989 — but the comparison these rows enter is
against a cold bar whose top rows are ~0.99 twin share, so the ordering claim
is blocked by the same finding until the bracket instrument exists.
