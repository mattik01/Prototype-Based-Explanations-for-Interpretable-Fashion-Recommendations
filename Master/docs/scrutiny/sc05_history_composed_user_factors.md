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
