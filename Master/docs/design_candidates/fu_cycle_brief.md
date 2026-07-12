# fU Design-Cycle Brief — directed cycle for the lineage's user-side stage

> v1.0, 2026-07-12. Authorized by the user at the dc01 SC.5 close-out sitting.
> **This brief is the per-cycle modification layer on top of the Design Candidate
> Protocol (`Master/docs/design_candidate_protocol.md`, v1.3), which otherwise
> governs unchanged** — ground rules, per-step artifacts, gates, one candidate per
> cycle. Where this brief and the protocol conflict, the brief wins (dated,
> user-authorized deviations, listed as M1–M7 below).
>
> **Invocation (fresh context):** `/design-candidate` and instruct the session to
> load THIS brief as mandatory input alongside the protocol. The brief's §4
> manifest is the context-load floor.

## 1. Strategic frame (why this cycle exists)

The thesis is committed to the **C1 lineage** (vault: lineage commitment,
2026-07-11): staged hosts mirroring ProtoMF's own U/I/UI split — **fI on
I-ProtoMF (done: dc01, hardened through SC.5), fU on U-ProtoMF (THIS CYCLE),
fUfI on UI-ProtoMF (the merge stage, later)**. This cycle designs the missing
user-side piece. It is exactly the promotion path the S0.2 side-reasoning
disposition anticipated: *"user-side features — promotion would be a new
design-candidate cycle."* That promotion is hereby called in.

**What pauses meanwhile (user decision, recorded in the scrutiny protocol
changelog + dc01 dossier):** S0.7 part 2 (reference-fleet results → frozen
table) and dc01's SC.6+ resume **based on cluster progress**, not on this cycle.
dc02/dc03/dc04 scrutiny is **deferred indefinitely** — those candidates become
licensed quarry (M4). Learnings from this cycle that touch the paused work
trickle back into it (M7).

**The null candidate.** The obvious seed is the **mirror of fI**: the user's
free embedding is replaced by a composed q_u = Σ_f e_f over user features,
feeding U-ProtoMF's user-prototype layer, score = t̂·u* with t̂ the free item
vector in R^{K_u}. The cycle's real job is to establish whether the mirror
**survives the user side's genuinely different physics** — or whether a
different mechanism (possibly cannibalized from dc02/03/04) is required. The
mirror is the default to beat, not the foregone conclusion. Known differences
the mirror must survive (§3, M6): thin native user features, history-derived
features (leakage + intrinsicness questions), the mirrored gauge, cold-user
reality.

## 2. Modifications to the protocol (M1–M7, user-authorized 2026-07-12)

**M1 — Directed cycle.** The design space is deliberately narrowed to the
lineage's user-side slot: host = **U-ProtoMF** (`user_proto`), stage bar =
**fU vs U-ProtoMF, like-for-like** (the staged-bar discipline from dc01 §G/R6′
applies verbatim). The protocol's distinctness-from-every-candidate rule is
re-scoped: distinctness from dc01 is NOT required (fU is dc01's lineage
sibling by design); distinctness within this cycle's own shortlist still
applies. Step 0's "open niche" question becomes: *which user-side mechanism
best serves the lineage*, not *which region of design space is unoccupied*.

**M2 — Literature de-emphasized, honesty unchanged.** Step 1's corpus sweep
becomes a **targeted pass** (user-side/CBF/user-modeling material + a re-check
of the §I ProtoMF-citers sweep), not an all-themes sweep; seeds may come
first from the lineage itself, the scratchpad, and dc02/03/04 cannibalization
(M4). Step 2's deep read is scoped to the **load-bearing claims** of whatever
the chosen mechanism actually leans on. Unchanged and binding: GR6 (no
from-memory citations — every claim that leans on a paper is verified against
the PDF), GR8 ("the paper says" vs "I propose"), and Step 2b (prior-art probe
— the novelty check is NOT relaxed).

**M3 — Mathematical soundness amplified.** The analyses dc01 earned through
three adversarial passes become **mandatory first-party work in this cycle**:

- **New Step 3b — hidden-effects pre-audit** (runs after Step 3, before
  Step 4; its artifact is `## 3b. Hidden-effects pre-audit`). Enumerate, with
  the dc01 record as the template:
  - *Gauge freedoms.* The F-DC01-08 anatomy mirrors: in fU the **free item
    vector t̂ ∈ R^{K_u}** probes the user-prototype slab — same frozen-at-init
    gauge component, same L2 canonicalization story, now on the item side.
    State the exact gauge dimension and which rendered quantities it touches.
  - *Lockstep/identifiability.* F-DC01-06's frozen-difference mechanism on
    **user-feature correlations** (and on history-derived features, where
    co-consumption creates near-lockstep). Same characterize→exhibit→remedy→
    delimit discipline; the decoupling-mass proxy is available prior art
    (`Master/temp/dc_checks/dc01/decoupling_mass.py`), scrutiny-basis-only.
  - *Norm/angle channels.* F-DC01-09's direction-only expressiveness on the
    user side; the **length-channel refund hypothesis** (vault 2026-07-12_1224,
    scratchpad entry + length-ablation control) predicts fU pays the same
    angular tax fI does — this cycle names the fU form of the tax and its
    instruments.
  - *Function-class honesty* (dc01 §I.3 style): what fU's ranking function
    class collapses to algebraically, and what keeps the factorization pinned.
  - *Operating-point / cold mismatches, popularity channels* (F-DC01-03/04
    analogues) — for USERS.
- **Step 4b claims spec (amplified floor):** must include the **keystone
  reduction claim** — fU(F=0, +ID) ≡ `user_proto`, bit-identity as the
  eventual implementation target (dc01's C5/C5′ and i02′ are the pattern) —
  plus gauge claims (C8/C9 analogues) and the pre-audit's checkable
  consequences. The dc01 check scripts (`check_claims*.py`, `check_claims_c9.py`)
  are the style reference. One-round rule unchanged.

**M4 — Cannibalization license.** dc02 (attribute-space prototypes /
concept-bottleneck), dc03 (visual exemplar prototypes / push), dc04
(attribute-anchored prototypes / live centroids) are **explicit quarry**: their
mechanisms, amendments, and check suites may be taken wholesale or in part,
transplanted to the user side, with attribution in the candidate file
("cannibalized from dc0N §X") and in the index row. R8 parsimony still binds —
transplant ONE clean idea, do not ensemble. Their index rows and files stay
untouched (they remain registered candidates; their scrutiny is deferred, not
cancelled).

**M5 — Comparison-instrument continuity (binding design context).** The
scrutiny protocol v1.2 SC.5 preamble governs the read-out design: the
explanation system is the thesis's **standing comparison instrument**, and the
fU stage answers the same two questions vs its host (`user_proto`), through the
**existing shared machinery** — the breakdown renderer
(`utilities/explanations/breakdown.py`; fU defines its slot, deferred there
2026-07-12 precisely for this), prototype cards, and the naming routes
(intrinsic user-prototype naming analogue vs post-hoc). SC.5's gate decisions
bind: rendered artifacts carry **arithmetic honesty only** (+1 baseline, signed
shares, ID-row line + explained fraction); **effect-epistemics go to the
thesis's hidden-effects section**, never onto the figures. Step 3's read-out
section must specify the renderer slot contents concretely.

**M6 — fU-specific mandatory confrontations** (Step 3 and Step 5 must address
these explicitly; silence is a gap):

1. **User-feature reality.** Native customer fields are thin (age, club
   status, news frequency, postal code — verify against `customers.csv` /
   `hm_feature_analysis.md`, never from memory). The alternative is
   **history-derived features** (user = aggregate of consumed items' metadata
   — the "history-composed user" scratchpad entry). If history-derived: (a)
   **leakage discipline** — computed from the train window only, S0.3-checklist
   spirit; (b) **intrinsicness question** — is a history-derived composition
   still R2-intrinsic, and what does its explanation line honestly say
   ("because you bought denim" is a different claim than "because you are
   34"); (c) the dynamic/static boundary (recompute cadence).
2. **Cold users, honestly.** S0.2 dispositioned cold users as structurally
   hard (thin features). Re-derive what fU can and cannot claim for R5 on the
   user side; the 3-scenario cold testbed scratchpad entry is input. A
   cold-USER eval spec is **scrutiny-stage work** — flag it, do not build it.
3. **Requirements per-stage reading.** R1's clarification note (2026-07-12)
   applies: operative core on single-branch hosts; the weight-sharing exemplar
   is assessed at the merge. Rate all R/S rows in their fU form.
4. **ft_type naming.** Propose the config name (suggestion:
   `feature_user_proto`, mirroring the reserved-names convention in CLAUDE.md)
   — reserved at registration, wired at build time.

**M7 — Learnings trickle-back.** Any finding this cycle surfaces that touches
the paused work (S0 instrument, dc01 concept/implementation, charter) is filed
immediately: scrutiny learnings ledger entries tagged `[fU-cycle]`, and — where
it names a concrete correction — a dated "resume note" appended to the relevant
dossier section. The paused sessions inherit these on resume; nothing waits in
conversation memory.

## 3. Knowledge-infusion manifest (context-load floor for the fresh session)

Protocol + this brief, then:

- `Master/docs/feature_aware_solution_requirements.md` — incl. the R1
  clarification note (2026-07-12).
- `Master/docs/design_candidates/dc01_feature_composed_factors.md` — the
  **governing fI record**: HOST REDIRECTION banner + re-host amendment §§A–I
  (mechanism, read-outs, G re-rating, H modes, I.1–I.5) + §3.4 as amended.
  This is the lineage's stage-1 exemplar; fU mirrors its structure.
- `Master/docs/design_candidates/candidate_index.md`, `scratchpad.md` (the
  captured user-side entries are priority Step-1 input), and the dc02/dc03/dc04
  candidate docs (quarry, M4).
- `Master/docs/scrutiny/learnings.md` (mandatory watch-outs),
  `findings_ledger.md` (F-DC01-06/08/09 anatomies — the pre-audit templates),
  and the dc01 dossier's SC.1b-delta + SC.5 sections
  (`sc01_feature_item_proto.md`).
- `Master/docs/scrutiny/s0_foundation.md` — S0.2 side reasoning (the user-side
  disposition this cycle promotes) + S0.6 charter (C1–C8) + S0.3 cold spec
  (leakage checklist as the discipline template).
- Vault: `2026-07-12_1224` (length-channel refund) and the lineage-commitment
  entry; `Master/docs/hm_feature_analysis.md` + `hm_dataset_notes.md`
  (user-feature reality); `Master/temp/dc01_feature_composed_bias_gap.md` is
  retired (folded into the design doc) — do not reload.
- Code (read, not modify, this cycle): `feature_extraction/feature_extractors.py`
  (`PrototypeEmbedding`, `FeatureEmbedding`), `feature_extractor_factories.py`
  ('prototypes' User-Proto path + the dc01 branch as the build pattern),
  `utilities/explanations/breakdown.py` (the renderer whose fU slot Step 3
  designs).

## 4. Cycle mechanics & what follows

- **Candidate id: dc05** (working title: fU-ProtoMF — feature-composed user
  factors on the U-ProtoMF host; the cycle may land elsewhere than the mirror).
  Naming note: the scrutiny protocol's out-of-scope list reserved "dc05" for a
  post-protocol *synthesis* candidate — that reservation is renumbered to
  "dcNN synthesis" (ids are sequential registration order, nothing more).
- Output: `Master/docs/design_candidates/dc05_<slug>.md` + index row
  (provenance likely `lineage`/`scratchpad+corpus`/`cannibalized-dc0N` — state
  it precisely; new provenance labels are fine if declared).
- Gates unchanged: per-step artifacts, user review at cycle end (Step 6 stop).
- **After the cycle (separate sessions, in order):** (1) user reviews dc05;
  (2) **build** — SC.3-style supervised implementation on `feat/scrutiny`
  (single code lineage), dc_checks/dc05 suite, keystone i02-style bit-identity
  vs `user_proto`, renderer/cards/naming slots wired; (3) **SC(dc05) scrutiny
  routine** under the scrutiny protocol. In parallel and independently:
  S0.7-part-2 → dc01 SC.6+ resume on cluster progress.
