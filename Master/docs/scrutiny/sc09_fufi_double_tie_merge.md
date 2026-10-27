# SC(dc09) — Scrutiny dossier: fUfI-ProtoMF (feature-composed user AND item factors under the UI double tie, `feature_user_item_proto`)

> Candidate: dc09 (`Master/docs/design_candidates/dc09_fufi_double_tie_merge.md`,
> produced 2026-10-26 by the directed merge cycle under dc05's M1–M7 by precedent;
> 4b + 5b absorbed as amendments A1–A16; user review decisions recorded in §6.2 the
> same day — **shared word table is the DEFAULT, `_sep` the arm; ID arms ids-both /
> noid-both only; `hm_kcore` deferred; pre-registration accepted**). NOT yet built.
> Protocol version in force: **v1.3** (2026-08-20). Branch: `feat/scrutiny`
> (HEAD 64f908c at session start, clean).
>
> **Precondition declaration (protocol SC preamble):** S0 is complete through S0.7
> (charter, build, frozen dev-tier reference table 2026-07-15). **Declared deviation
> for this candidate:** by the user's 2026-10-26 scoping decision every reported
> number is a **100-trial (`prod`) row, single seed 38210573** — the comparison basis
> is therefore the thesis-grade `_n100` tier, NOT charter C4's dev profile and NOT
> the S0.7 dev-tier table. Host rows at that tier: `user_item_proto` × ml-1m n100
> (May 21, pre-S0, user-accepted as the host row) and × hm_1_month n100 (wave-1 job
> 8153730, queued, cannot start before ~2026-10-28). SC.6 must cite this as a C4
> deviation under the two-tier numbers policy (2026-07-14) and refuse any comparison
> that mixes tiers.

---

## SC.1a Concept self-scrutiny (2026-10-26)

### 1. Preamble

**Manifest loaded:** protocol (full read + SC.1a re-read), findings ledger (full),
learnings ledger, requirements doc (R1 clarification note, R6 staging note),
candidate index, dc09 design doc (all 2,052 lines incl. A1–A16 and the §6.2
decisions), `dc_checks/dc09/claims_spec.md` + `report.md` + the verbatim 5b
critique, supervisor presentation, the four interpretability vault entries. First-
party verification reads: `feature_extractors.py` (FeatureEmbedding /
HistoryFeatureEmbedding layouts, PrototypeEmbedding.forward, Concatenate),
`feature_extractor_factories.py` (`prototypes_double_tie` branch), `rec_sys.py`
forward (LOO routing), `cold_eval.py` (ID-drop finders), `hyper_params.py` (host UI
space), charter C2/C3/C4 text, the S0-build-extension ml-1m record +
`data/ml-1m/build_item_features.py`, vault `2026-08-15_1721`, `2026-08-17_1823`,
`2026-07-12_1815`; all ten vault entries the design doc cites exist.

**Learnings watch-outs that apply here:** #1 (labels fused to numbers — the
`tags@0.8` provenance was verified against the build script, see F-DC09-05);
#11 (every host swap re-derives its hidden effects first-party — the shared-table
default is a *second* swap on top of the cycle's text, re-derived below); #12
(empirical, not scholastic — every new mode below ships with its instrument and
control arm); #7/#8 (rebuttal overclaim; grades carry their scope — applied to
A1's re-ratings); #14/#15/#16 (side harnesses, builder-owned vocab, renderer
tolerances — SC.3/SC.5 checklist); #17 (cluster-ops bundle — SC.6).

**Open/shared ledger re-check:** no `open` finding exists. Shared findings re-checked
against current code: F-S0-07/F-S0-13 ID-drop primitive present (`drop_cold_id_rows`,
`_find_feature_embedding` recursing Concatenate → PrototypeEmbedding →
FeatureEmbedding; the recursion does **not** reach the proposed
`TwoRoleComposedExtractor` — see F-DC09-01); F-S0-03 evaluator assert unchanged;
F-DC05-13 layout-aware labels (dc09 consumes both layouts on both sides — not
stale); F-DC05-16 ml-1m cold deferral stands (no ml-1m cold row for fUfI either);
F-DC05-19 ml-1m naming defaults stand. **Nothing marked stale.** dc06/dc07/dc08
findings live in their design docs/index, not this ledger (lightweight paths) —
F-DC08-02/03 (identity rows non-identified / not rank-inert) are inert here (dc08
rows = 0 in every dc09 arm).

**Where the record is:** the design doc's §3–§5 text describes the *separate-table*
candidate; the user's review flipped the default to the shared table with a
banner note (§3.1) and swapped names (§3.2, §6.1, §6.3) but did not re-derive
§3.2's module spec, 3b's gauge analysis, §3.6's mode list, 3b-v's per-arm
pre-registration, or the §5.1 grade texts for the new default. That re-derivation
is the bulk of this step (findings F-DC09-01..04).

### 2. Known-issues dispositions FIRST (steel-man before adversary)

**4b (three falsified sub-clauses).** C2(b) share ≠ removal counterfactual on the
user linear half — **resolved** (§3.4 read-out 4 amended; renderer renders shares,
counterfactual Δ_j disclosed). C10 monotonicity regime — **resolved** (3b-iii
qualified). C14(c) sign flip — **resolved** as wording; **and superseded by the
default**: K10's alignment penalty has no object when the word table is shared
(L_align ≡ 0); it is a `_sep`-arm knob only. The 4b round was run on the
separate-table mechanism (claims spec: "SEPARATE parameters from E_I"); every claim
C1–C13 is table-agnostic (none uses E^U ≠ E^I), the keystone C4 lives at V = 0 where
sharing is moot, so **no claim is invalidated by the default flip** — but the
sharing-specific identities (F-DC09-02 gauge collapse; the centroid identity in §3
below) were never toy-checked and go to SC.2.

**5b (23 points, A1–A16).** 1 (R3 positional labelling): resolved by A1's split grade;
the "stitched across two geometries" half is **resolved by the user decision**
(one geometry by default); positional labelling of the linear halves remains an
inherent, disclosed host property. 2 (gauge bars): resolved A1/A4/A6; effective-
gauge instrument owed at SC.8. 3 (R1 / LightFM precedent): resolved A7; the
cross-side tie is now the default, so the R1 grade text ("untied by default") is
**stale** → F-DC09-04. 4 (arms favour different requirements): **superseded by the
user decision** (ids-both / noid-both; mixed optional) — the user-ID-row risk on
ml-1m is carried into the ids arm's reading rule (F-DC09-04 restates 3b-v for the
2×2). 5 (S1/R8 grade a plan): S1 resolved (partial); `lightfm_tags_hist` is an SC.6
decision; R8 needs a second look under the new default → requirements delta.
6/7/8 (head inference, falsification, power): resolved A3/A8; ASHA-pruning and
search-box-ceiling are **open risks** for SC.6/SC.8 (instruments named). 9: resolved
A1/A9. 10 (deferred question, weak instrument): resolved A10 and **superseded** —
the user took the one-table decision directly; `_sep` measures its cost. 11: resolved
A8; the "one-vocabulary story lives in the ablation arm" sentence is now reversed.
12 (Steck on word rows): resolved A5/A14 as a mode, but the Steck attribution is
imprecise → F-DC09-06. 13 (ml-1m tag-count channel): resolved A5 (M13); its
flagged caveat is now **verified TRUE** → F-DC09-05. 14 (global scale gauge):
resolved A4 (3b-x) for separate tables; **wrong under the shared default** →
F-DC09-02. 15 (dataset asymmetry): resolved A13 as open risks; `hm_kcore`
deferred by the user. 16 (sampled eval): resolved A16 (D3 popularity-negatives
secondary read-out required — verified present in charter C3). 17 (2×2 not a
factorial): resolved A2/A11; fI-on-UI / fU-on-UI stay optional arms (user). 18:
resolved A15. 19: resolved A1/A3. 20: resolved A12. 21/22: resolved A6. 23: resolved
A1.

**Documented gaps (M7–M14, 3b-i..x, 5.5 table):** all remain open-by-design with
instruments; none re-discovered below. Two are affected by the default flip: M11
is removed by construction (as the doc says) — but its *replacement* mode is
unnamed → F-DC09-03; 3b-x is re-derived → F-DC09-02.

### 3. The scrutiny proper

**(a) Mathematical soundness.** The four-component identity, the W-gauge
(dimension d·(K−d−1), slab reading, adaptive-optimizer caveat), the LOO argument,
the keystone, and the collapsed function class (3b-iv) all re-derive first-party
and match 4b. Precision notes (no finding): the A4 "≈ 65 of 76 coordinates of t̂
softly invisible" is a 65-dimensional *subspace* of R^{K_u}, not 65 named
coordinates — every rendered coordinate t̂_l generically mixes identified and
gauge parts, which is exactly why the slab-projection reading is the identified
object. **New, under the shared default:** with words shared, the per-side scale
gauge of 3b-x no longer exists — (E^I, W^t) → (cE^I, W^t/c) rescales the user
composition's word part too, so û changes; what survives is one *joint* gauge
(c on the word block and both ID blocks, 1/c on both W). Consequence: the relative
scale of a word row as seen through W^t vs W^u is now identified (it was pure gauge
in `_sep`). → **F-DC09-02**. **Centroid identity (observation, thesis material):**
under sharing, q_u^{words} = (1/|H_u|) Σ_{j∈H_u} (q_j − e^I_ID(j)) exactly — the
user is the centroid of their purchased items' *feature-only* vectors in the
item's own geometry; in the noid arm q_u = mean_j q_j, so **fUfI-noid-shared is an
item-neighbourhood model in composed space** (û = W^u·centroid; the I-half
ûᵀt* is a centroid-vs-item bilinear). dc05's per-purchase regrouping becomes
literally a per-item-vector regrouping. Goes to SC.2 as a toy identity and to
SC.5 as explanation-axis input.

**(b) Expressiveness.** ids arms (shared or sep) = host function class
(reparameterisation: e_ID(i) = T[i] − Σ_f e_f on the item side, e_ID(u) = U[u] −
centroid on the user side), differently priced (partial pooling; vault
`2026-08-15_1721` §4). noid-sep = additive-span restriction (A9). noid-shared ⊂
noid-sep: one V×d table serves both compositions — hm 426·d ≈ 31 K vs the host's
6.36 M entity parameters (≈ 200× shrink); ml-1m 1,061·d ≈ 77 K vs 0.67 M. Ceilings
(S0.5 / S0-build-ext numbers): hm 67.5 % of items share a 5-field signature (max
class 97) → in noid arms twins are tied on **both** halves (F-S0-14 bracket quoting
applies to every cold row); ml-1m 3.8 % (≈ 1.3 % outside the genres-only tail).
Gradient paths: every word row has four — item cosine, item linear (W^t), user
cosine (via basket mean), user linear (W^u) — under sharing; two under `_sep`.
The model accesses strictly more information than the host (words on both sides)
and can leverage it on both halves (norm channel open, A1 Eq. 11 free W).

**(c) Degeneration-mode coverage.** M7–M14 + the 3b gauges cover the separate-table
design. Under the shared default two things change: (i) M11 disappears, but the
*role conflict* it hid is now on one table — each word row serves item-property
and taste-indicator roles under four gradient sources; (ii) M7's contamination
gains a direct user-side path: item-linear popularity pressure inflates shared
rows, which now enter q_u directly, tilting user vectors toward popular-word
directions (in `_sep` the user table was shaped by the user side alone). Neither
is named with an instrument → **F-DC09-03 (M15 + M7-shared)**. Control arm:
`_sep`. The remaining knobs reach every listed mode (K7–K11, `normalize_composed`);
K10 has no object under sharing (noted above).

**(d) Requirements delta (vs the A1 table).** R1 **strong** — agree on grade, grade
*text* stale: the within-side exemplar holds AND the cross-side word-table tie is
now the default (the lineage's first cross-side tie); `_sep` prices it. R2
agree. R3 agree. R4 **partial** — agree; prose stale ("the Seed B arm would make
… one geometry" is now the default — a genuine R4 gain to state, rating still
partial because grounding stays unenforced and profile validity is owed). R5
agree. R6 agree (partial; bet). R7 agree. R8 **strong → strong with disclosure**:
the doc's own Seed-B analysis said sharing "as a headline stacks two moves"; the
user chose that headline knowingly — the grade must carry it ("two moves; the
second isolated by the single-variable `_sep` arm"), not delete the reasoning.
S1–S5 agree. → folded into **F-DC09-04**.

**(e) Cold-start mechanism check, incl. the bias channel.** Fleet `use_bias=0` →
no bias scalar on either side (verified `rec_sys_param` in the host space). ID-keyed
side channels at cold inference: item ID row (both roles read it; its linear image
W^t e_ID feeds the live intercept's item-specific residual (W^tᵀ1)·e_ID — all
zeroed by ONE drop on the shared instance), user ID row (cold-user testbed absent;
`drop_cold_user_id_rows` mirror exists). **Hazard:** both finders recurse
Concatenate → PrototypeEmbedding → composition; the proposed
`TwoRoleComposedExtractor` is neither, so `_find_feature_embedding` returns None and
the runner would **silently skip the drop** — the exact F-S0-07 silent-depressant
class; `config_expects_id_drop` also does not know the new `ft_type`, so the
post-drop abort would not fire either → concept-level obligation recorded in
**F-DC09-01** (SC.3 implements; dc01 t11 pattern). S0.3 detects the design's cold
failure modes: cold magnitude handicap (cold-vs-all), twin ties (tie bracket),
intercept-only cold users (not measurable — 3-scenario testbed unmet, declared).
ml-1m cold: none (F-DC05-16).

**(f) Steck applicability.** The per-dimension-rescaling argument needs an
objective invariant to diag rescaling of the embedding; here diag D on a word
table is absorbed by W → W D⁻¹ for the linear halves **but changes every cosine
unit**, so the full objective is NOT invariant and Steck's exact arbitrariness does
not transfer — F-DC01-02's "trains through the cosine it reads" still pins the
geometry. What M14 correctly names is weaker: rows shaped by a *mixed*
cosine/dot objective, for which the semantic reading of cos(e_f, p) is
unvalidated, not provably arbitrary. The mitigation (profile-validity spot-check
fUfI vs fI/fU on the same vocabulary; K8 λ_W) stands → **F-DC09-06** (precision,
honest direction).

### 4. Findings (bundled with proposed fixes — decided at this gate)

- **F-DC09-01 [minor]** — The shared-table default is unspecified at the §3.2
  anti-vagueness level (module spec, parameter table, injection, init ownership,
  finders all describe `_sep`); a word *block* cannot be shared between two
  differently-sized `nn.Embedding`s by `.weight` re-pointing; cold finders miss the
  new wrapper. **Proposal:** dated §3.2 amendment — one word-table module (V×d)
  owned and initialised once by the factory, each side its own ID block,
  composition = word lookup + ID lookup (both layouts); `max_norm` double renorm
  idempotent (not searched — inert); wrapper exposes the composition so both cold
  finders reach it and `config_expects_id_drop`/`_user_id_drop` register the new
  `ft_type`; keystone unaffected (V = 0). SC.3 implements + t05.
- **F-DC09-02 [minor]** — 3b-x per-side scale gauge is false under sharing; only
  the joint gauge survives, and the cross-side relative row scale becomes
  identified. **Proposal:** dated 3b-x amendment; SC.2 toy claims: joint gauge
  exact; per-side gauge broken (S changes) under sharing; centroid identity.
- **F-DC09-03 [minor]** — Missing modes under the default: **M15 role conflict on
  shared rows** (four gradient sources per word row; instrument: per-role
  gradient-norm split per row, extending 3b-vii's two-role note) and **M7-shared**
  (item-side popularity norm inflation enters q_u directly; instrument: popularity-
  axis R² on the user-composition cloud, fUfI-shared vs `_sep`). **Proposal:** dated
  §3.6 + §5.5 rows; `_sep` named as the control arm for both.
- **F-DC09-04 [minor]** — Stale arm/default references after the user's decisions:
  3b-v names `fUfI-mixed`; §3.7/§4.5 keep the mixed-arm promotion and ladder; §3.8
  dev. 6 says "separate as default"; R1/R4/R8 grade texts predate the flip.
  **Proposal:** one dated consistency amendment — 3b-v restated per arm for
  {ids, noid} × {shared, sep} (expectations: fUfI-noid-* > fU-noid 0.6082 and
  > fI-ids 0.5628; fUfI-ids-* > fU-ids 0.5613; disconfirmation (a)(b)(c) unchanged;
  user-ID-row risk read into the ids arms); R1 "strong — within-side exemplar +
  cross-side word tie by default, `_sep` prices it"; R4 prose updated (one geometry
  now default, rating unchanged); R8 "strong — two moves, second isolated by
  `_sep`"; §3.8 dev. 6 inverted; mixed arm = optional (user).
- **F-DC09-05 [minor]** — M13's caveat verified TRUE: `tags@0.8` = Tag-Genome
  relevance ≥ 0.8 (S0-build ext §1, `build_item_features.py`, B5 §4.1 recipe);
  Tag-Genome relevance is a learned score predicted from user tagging/rating
  behaviour (Vig, Sen & Riedl 2012 — citation to verify at thesis RW; not in the
  reading list). Fleet-wide property of the ml-1m testbed (fI/fU/lightfm rows
  alike), not a dc09 pathology. **Proposal:** dated amendment to M13 + §3.5 + the
  R6 reading ("ml-1m content is partly behaviour-derived; hm's taxonomy is static
  catalog metadata — the testbeds differ in feature provenance"); M7 trickle-back:
  Ch 3 Data owes the sentence; learnings-ledger entry at SC.7.
- **F-DC09-06 [minor]** — M14/§5.3 overstate Steck: the cosine units break
  diag-rescaling invariance, so the arbitrariness argument does not transfer; the
  honest statement is "rows shaped by a mixed cosine/dot objective — semantic
  reading unvalidated, not arbitrary". **Proposal:** precision amendment to M14 +
  §5.3 Steck row; mitigation unchanged.

**Carried to later steps (no finding):** `lightfm_tags_hist` charter-amendment
decision (SC.6); fI-on-UI / fU-on-UI optional arms (SC.6); ASHA grace / search-box
ceiling instruments (SC.6/SC.8); n100-tier C4 deviation citation (SC.6); centroid
identity → SC.2 toy + SC.5 axis; `max_norm` unreachable (not searched) — re-verify
if ever added.

### 5. Gate — CLOSED 2026-10-26

User decision (plain-language walkthrough of all six; F-DC09-02's gauge effect
explained at the user's request — one joint scale gauge survives sharing, the
cross-side W^t-vs-W^u response ratio becomes identified, instruments normalise
jointly; centroid identity confirmed as thesis material): **"accept all six"** —
F-DC09-01..06 accepted as proposed. Amendments land at SC.2 (ledger status stays
`open` until the resolving edits; F-DC09-01's code half at SC.3). Next: SC.1b.

---

## SC.1b Adversarial pass (2026-10-26)

Subagent brief (protocol SC.1b.1, verbatim): *"The previous adversary's critique and
the author's rebuttals are in the file. What did the previous adversary miss? Which
rebuttals do not actually hold? What breaks at H&M scale?"* — plus the instruction to
attack the candidate as decided in §6.2 (shared table default). Inputs: ONLY the
design doc and the requirements doc (Ground rule 10). Output saved as
`Master/temp/dc_checks/dc09/sc1b_adversarial_critique.md` (3,889 words); reproduced
verbatim below. Launched during the SC.1a gate (independent of its outcome; its
artifact appended only after that gate closed).

### Critique (verbatim)

> # SC.1b adversarial critique — dc09 fUfI-ProtoMF as it stands after the §6.2 review decisions
>
> Input: read in full, and only, `Master/docs/design_candidates/dc09_fufi_double_tie_merge.md` (2,053 lines, incl. the §5b.1 critique, §5b.2 rebuttals, §5b.3 amendments A1–A16 and the §6.2 user decisions) and `Master/docs/feature_aware_solution_requirements.md` (R1–R8, S1–S5). No other file opened, no code run, no web. The candidate is attacked as decided in §6.2: shared word table = default, `_sep` = arm, ID arms = ids-on-both / noid-on-both.
>
> Notation: E ∈ R^{V×d} the shared word table; q_i = Σ_{f∈f_i} e_f + e^I_ID(i); q_u = Σ_g w_{u,g} e_g + e^U_ID(u) with w_{u,g} = n_{u,g}/|H_u| (so Σ_g w_{u,g} = F = 5 on hm for every user; the basket-mean bag size on ml-1m); A = P̂^uᵀW^t, B = P̂^tᵀW^u (d×d, §3b-iv); w_t = W^tᵀ1, w_u = W^uᵀ1; |H_u| on ml-1m ≈ 574,376/6,034 ≈ 95 (§2.1 Table 1 numbers).
>
> ## A. The shared-table default: consequences the separate-table text never re-derived
>
> **1. The default arm has no verification and no module design.** Every 4b claim (§4b.1 C1–C14) was specified and checked for two tables; the keystone C4 runs at V = 0 where sharing is vacuous, C5 reduces one side at a time. There is no claim of the form ∇_E L_shared = ∇_{E^I}L + ∇_{E^U}L at E^I = E^U (chain-rule sum over two roles), none on the shared-row LOO coupling §3b-vii itself flags, none on the two-role gradient balance (point 6). §3.2's factory builds `item_base = FeatureEmbedding(...)` and `user_base = HistoryFeatureEmbedding(...)` as two owners of two tables and calls `init_parameters()` on each; §2.5 describes those tables as monolithic (V+M)×d and (V+N)×d blocks and says no user-side sharing wrapper exists. A shared *word block* with separate ID blocks cannot be a sub-tensor of two Parameters of different height: both scrutinised modules must be split into (words, IDs) Parameters — touching the dc01/dc05 code that the t03 RNG/golden pin protects (§3.2 item 1, §6.3) — and a single-owner init rule for the shared block has to be written (two inits = a different RNG stream and a last-writer-wins table). Two per-side `max_norm` slots (§3.2 factory, `ip.get('max_norm')`, `up.get('max_norm')`) now target one table. §6.3's keystone list t01–t06 has no sharing test. The default arm of the run plan is the one arm the cycle never specified or checked; "4b: 11/14 confirmed" (§6.2) is a statement about the ablation.
>
> **2. Under one table the personalised score is a low-rank word–word affinity model with a quadratic overlap channel — unlisted, and it changes the prior-art answer.** From §3b-iv, pers(u,i) = q̂_uᵀA q_i + q̂_iᵀB q_u = q_uᵀ M_{ui} q_i with M_{ui} = A/‖q_u‖ + Bᵀ/‖q_i‖. Substituting the shared compositions: pers = Σ_{g,f} w_{u,g}·e_gᵀM_{ui}e_f + ID terms — a V×V word-affinity matrix G = E M Eᵀ of rank ≤ d between basket words and candidate words. For every word f in both the basket and the candidate the term is w_{u,f}·e_fᵀM_sym e_f, a quadratic form in one row: the model raises every basket∩candidate overlap by growing ‖e_f‖ along M_sym's positive eigendirections, and M7's norm inflation of frequent words feeds it (contribution ∝ ‖e_f‖²). Under separate tables the same term is e^U_fᵀM e^I_f, a bilinear in two independently decayed rows that must be grown *and* aligned; the unilateral norm channel exists only in the default. Consequences: (a) a fourth popularity/overlap home missing from §3b-v's enumeration (a)(b)(c); (b) the I-half becomes "repeat your words" weighted by word frequency, narrated per purchase by read-out 4 as taste; (c) the default is structurally the shared-feature-table factorisation machine — one table, pairwise basket-feature × candidate-feature inner products through a learned d×d core, 1/|H| implicit-feedback normalisation — i.e. SVD++/FISM's y_j sharing at word level and AFM's learned interaction, which §1a dismissed as "dc08's" (B12/B13) and "Axis-B scorer material, out of scope" (B8). §2b probed the separate-table seed; "nobody has put the two together" does not cover the default and must be re-run against the corpus's own B8/B12/B13 before the thesis positions the headline. Instrument: split both halves into overlap (f ∈ f_i ∩ basket) vs non-overlap word pairs; report the overlap share against word frequency.
>
> **3. A cross-unit popularity leak: the item intercept trains the user cosine geometry.** B_item(i) = w_tᵀq_i (§3.1). Under sampled softmax, words of the positive move by +η(1−p_pos)·w_t per step and words of negatives by −η·p_c·w_t; with uniform negatives, frequent words accumulate a +w_t component. Under separate tables that component stayed in E^I. Under the default it sits inside every q_u = Σ_g w_{u,g}e_g, so u*_l = 1 + cos(q_u, p^u_l) reads the basket's mean word popularity along ŵ_t, and a user prototype aligned with w_t is a "mainstream" community whose word profile is word frequency — dc05's popularity prototype re-entering through a channel dc05 never had. §3.6 M7 is an intra-table *norm* mode; this is a *direction* leak from one unit's intercept into the other unit's cosine, with no instrument in §3.6/§3b. A second leak of the same kind: B_user = w_uᵀq_u is rank-inert (§3.1) but loss-active under BCE (a per-user shift cancels under sampled-softmax and BPR, not under BCE), and `bce` is in the searched loss set (§2.5): under BCE the "inert" intercept pushes every basket word along w_u into rows the item side reads. Instruments: cos(ŵ_t, p^u_l) over l; projection of q_u on ŵ_t vs basket popularity; w_u-direction gradient mass by loss type.
>
> **4. Two prototype sets fitted to one point cloud: "comparable" collapses into "redundant".** On hm the user word vector is the mean of 2–10 item word vectors over a 5-field taxonomy; in the noid arm the user cloud lies inside the convex hull of the item cloud, in the ids arm inside it plus a user-ID row that partial pooling shrinks for thin users (point 12). The inclusion regularisers (§3.6 inheritance) place each prototype set near its batch cloud — the same cloud. Prediction for hm under the default: the cross-prototype matrix cos(p^u_l, p^t_k) that Seed B advertises as a free read-out (§1c, §3.4 item 7) has row maxima near 1 for most l, and the two cosine units return near-collinear readings of a given input — the "community vs archetype" narrative has no content on the fashion dataset. In the ids default the matrix is additionally contaminated: P^t absorbs item-ID directions and P^u user-ID directions the other side never sees (with 87,069 ID rows and 426 word rows in R^73 there is no "word subspace" — §3.6 M9's "ID subspace" is statistical, not geometric), so cos(p^u_l, p^t_k) is not a vocabulary comparison. Instruments: row-max distribution of the cross-prototype matrix; canonical correlation between u*(q) and t*(q) on the same q.
>
> **5. Stale text, a wrong gauge section, and a rating the author's own argument now contradicts.** Written for separate tables and not re-derived: §5.1 R1 row ("cross-side untied by default"), §5.1 R1 prose, §5.5 M11 row ("open risk in the default arm; removed in `_shared`"), §3.6 M11 ("Seed A specific"), §3.8 deviation 6 ("separate word tables as default"), §3.7's "Original cycle text", §3.2's code, §3b-x. §3b-x is wrong under the default, not merely mislabelled: the two per-side scale gauges do not exist (scaling one ID block alone rotates q); only the global (E, E^I_ID, E^U_ID, W^t, W^u) → (c·, c·, c·, ·/c, ·/c) survives, so ‖W^t‖_F/‖W^u‖_F becomes identified (useful for M8, unclaimed) and the prescription "normalise by the table's mean row norm or by ‖W‖" now has one table and two W's. R8 "strong" (§5.1) rests on "no new layer, loss, or table" and "knob and controls are arms"; §1c's own Seed B text says that as a headline the shared table "stacks two moves — compose both sides *and* tie them across sides" and was "not chosen as headline" for exactly that reason. The headline is now that candidate. Either §1c's R8 reasoning was wrong or R8 is partial; the document asserts both. Rebuttal 5b.2-5 ("the mechanism is one move") no longer holds.
>
> **6. Two masters at unequal weight, and the weight is dataset-dependent.** A shared row e_f receives item-role gradient with coefficient 1 per candidate containing f (B·N slots per step) and user-role gradient with coefficient w_{u,f} = n_{u,f}/|H_u| per batch user. On ml-1m (|H_u| ≈ 95) a tag in 5 of a user's movies enters with weight ≈ 0.05: the shared table is item-trained and the user side a passenger — the structural reason behind A13's "fUfI ≈ fI-on-UI + user-ID row". On hm (|H_u| ≈ 3–10) the weights are comparable and 426 rows serve two objectives with different geometric demands (discrimination among candidates vs basket-mean clustering). M11 "removed by construction" (§3.6, §5.5) traded divergence for competition; the competition's outcome is not named, and §3b-vii's "word-row gradient norms by role" is listed as a coupling to watch, not as the instrument for which role owns the geometry. Pre-registrable: on hm `_sep` beats the default on user-side profile sharpness and loses on parameter efficiency; on ml-1m they tie because the user role is diluted away.
>
> ## B. A gauge both the author and the previous adversary missed: translation of the word table in the ids arm
>
> **7. Derivation, and what it does to §3.4.** For each field φ pick v_φ ∈ R^d and set e_f ← e_f + v_{φ(f)}, e^I_ID(i) ← e^I_ID(i) − Σ_φ v_φ, e^U_ID(u) ← e^U_ID(u) − Σ_φ v_φ. On hm every item has exactly one word per field, so q_i is unchanged for every item and q_u^{LOO} for every user and every left-out positive (each remaining basket item contributes Σ_φ v_φ; the mean keeps it). Every score, every loss, all four inclusion regularisers (they read q only) are invariant; only L2 and `max_norm` break it. Dimension 5·d = 365 at d = 73. On ml-1m use one v per field with e_ID(i) ← e_ID(i) − Σ_φ |f_i∩φ|·v_φ and e_ID(u) ← e_ID(u) − mean_j Σ_φ |f_j∩φ|·v_φ: exact at inference, broken at training only by LOO's varying basket-mean bag size. This is neither §3b-i's W-gauge nor §3b-x's scale gauge: it is a symmetry of the ids arm's warm model in which the ID rows absorb any per-field centroid of the word table. Under it, in all four components the per-row shares move and only the sum is fixed: c_{f,k} ← c_{f,k} + v_φ·p_k/(‖q_i‖‖p_k‖) for every word of field φ with the ID share absorbing the negative; (W^t e_f)_l ← (W^t e_f)_l + (W^t v_φ)_l; the intercept zoom w_t·e_f ← w_t·e_f + w_t·v_φ. So in the ids default the absolute level of "Trousers Denim +0.95 · Menswear +0.35 · Dark Blue +0.20" (§3.4 rendered example), every *between-field* comparison of shares, the "item-specific" line, and the naming profiles cos(e_f, p) themselves (S4 "strong") are regulariser artefacts; identified are the within-field contrasts (Dark Blue − Black) and the totals. In the noid arm no row absorbs the shift, q moves, and the shares are identified: the ID ablation (§3.7) is also an identifiability ablation, and the identified arm is the non-default one. dc06's per-field-mean centring (§0.2, §1b-23) is exactly the gauge fix v_φ = −mean_φ; the document files it as a composition knob and says the merge takes "uncentred compositions on both sides as its default". §3.1's "B_item is NOT gauge" is true of the total and false of its per-word zoom. Lineage-wide (M7 trickle-back): dc01-ids and dc05-ids carry the same symmetry on hm.
>
> **8. Named coverage is gauge-dependent, so disconfirmation (c) and the M9/M10 detector measure the regulariser.** The U-half's named part Σ_{f∈f_i} u*ᵀW^t e_f shifts by u*ᵀW^t Σ_φ v_φ under point 7 while the whole half is fixed; the coverage ratio can be driven to any value, including outside [0,1], by a score-preserving translation. §3.4 item 6 calls the per-component split "the parameterization-invariant aggregate (dc08 4b C9)" — whatever C9 proved, it was not invariance under this symmetry. §3b-v (c) ("a win whose linear-half named coverage is below the cosine units' is M9 realised") therefore compares two unidentified numbers, and the M9/M10 rows of §5.5 name an undefined detector. Fix: state coverage in a gauge-fixed convention (per-field centred table, ID row = residual) or use an invariant: on fixed-field data the shift is the same for every item at fixed u, so the share of the *across-item variance* of a linear-half component carried by word rows is invariant and can replace the ratio for the linear halves (not for the cosine units, where ‖q_i‖ varies across items).
>
> **9. Cold-item scores of the ids arm are gauge-dependent; under the default the gauge is set by the user-ID population.** Dropping the ID row (§3.4 "Cold and thin cases", §3b-v) removes the row that absorbed −Σ_φ v_φ: q_i^{cold} = Σ_f e_f ← q_i^{cold} + Σ_φ v_φ. Two warm-identical ids models related by point 7 rank cold items differently; the cold operating point is whatever centroid L2 chose. Rebuttal 5b.2-4 ("not unmeasured — S0.3 is committed") holds only in the sense that a number will exist: it is a property of the regulariser's canonical gauge, not of the warm model. Under the default the canonical centroid is one vector for both ID blocks: minimising Σ_f‖e_f+v‖² + Σ_i‖e^I_ID(i)−Fv‖² + Σ_u‖e^U_ID(u)−Fv‖² gives v ≈ (pooled mean ID row)/F, with weight F²(M+N) ≈ 2.2·10⁶ against V = 426, and the pooled mean is 84 % user rows (N/(N+M) = 73,418/87,069). The merge's cold-*item* behaviour is thus set by the L2 balance of 73,418 user-ID rows — a cross-side coupling the shared decision created silently; under `_sep` each side's centroid was set by its own ID block. The noid arm has no such gauge (cold = warm representation). Cheap instrument (login-node class, runnable today on dc01's fI-ids checkpoints): apply v = −(pooled mean ID row)/F to a trained ids checkpoint, verify warm scores unchanged, re-run the cold protocol; the change is the gauge-dependence of the cold read-out.
>
> **10. "The gauge parts are init noise" (§3b-i) does not survive the author's own optimiser note.** A4(2) concedes that under Adam the W-gauge component is not conserved and calls it "score-inert, which is the identifiability-relevant property". It is worse than non-conserved: Adam's update is D_t·g_t with a diagonal preconditioner, and Proj_null(D_t g_t) ≠ 0 generically even though Proj_null(g_t) = 0 (C6b), so the gauge part accumulates a systematic, path-dependent drift decayed only by weight decay — not zero-mean noise, not init. The rendered per-prototype lines t̂_l, û_k (§3.4 read-outs 3/4, "slab-projected representatives") carry an optimiser-path artefact; rebuttal 5b.2-2(c) ("conservation was never load-bearing") is right for identifiability and wrong for the disclosure, which must say "optimiser-dependent", not "init noise".
>
> ## C. Hidden effects at scale the pre-audit does not reach
>
> **11. Hidden rank-active intercepts inside the "personalised" halves on ml-1m.** Write q_u = μ + δ_u + e^U_ID(u), μ the basket-mean of the item word cloud toward which heavy users shrink (F-DC05-01, §3b-viii). The I-half is (W^uμ)ᵀt*(i) + (W^uδ_u + W^u e^U_ID)ᵀt*(i); the U-half is u*(μ)ᵀt̂(i) + ε_uᵀt̂(i). The first term of each is a per-item scalar shared by every heavy user — rank-active, popularity-capable, and rendered by read-out 4 as "your 95 movies +0.4" when most of it is the same for everyone. §3.1's "the merge carries exactly one popularity-capable intercept" is a statement about the algebraic split, not about user-dependence: on the R6-load-bearing dataset there are three per-item scalars (B_item, (W^uμ)ᵀt*, u*(μ)ᵀt̂). Under the shared default μ lives in the item geometry, so (W^uμ)ᵀt*(i) is directly comparable to B_item and the two can trade mass invisibly. The M8 instrument "variance share of each component across (u,i) pairs" cannot see this — a component can have large variance and zero personalisation. Replace with a two-way decomposition per component (item main / user main / interaction on a user×item grid) and report the user-independent share of each "personalised" half; pre-register that on ml-1m it exceeds the ID-row share.
>
> **12. On hm the ids/noid ladder is a different experiment than on ml-1m, and the user side degenerates to item–item content matching.** Parameter mass: the ids arm on hm is 5.36 M user-ID entries of 6.36 M (§4.1); each row is hit |H_u| ≈ 3–10 times per epoch against decay every step, so thin users' ID rows shrink toward 0 and ids ≈ (item-ids, user-weak) for the thin majority. §6.2's "ids-on-both vs noid-on-both" isolates the user-ID row on ml-1m (≈ 95 hits per row per epoch) and barely on hm. For a 3-train-row user after LOO, q_u is the mean of two item word vectors on a twin-dense 5-tuple catalogue — often one item's vector; the I-half is then exactly (1/|H|)Σ_j (W^u q_j)ᵀt*(i), FISM through words, and "community l" is a cluster label on item vectors; in the noid default the model has no per-user parameter at all. A13 reads "25 % at three rows" as three *train* rows ("LOO removes a third"); §2.1's protocol is a temporal leave-one-out split holding out val and test, so if the 25 % counts raw interactions that stratum has one train row and q_u^{LOO} = 0 at every training step — the C11a case as the modal user, not an edge case. The document must state which; the explanation surface differs by three purchase lines vs none.
>
> **13. LOO hygiene is a plumbing truth, not a data truth, on hm.** C8 verifies both roles see q_u without the positive's slot (§2.5, §3b-vii). With twins dense enough to be "the known equivalence-class ceiling" (§3.9), the positive's exact word vector remains inside q_u^{LOO} whenever a twin is in the basket, and the overlap channel of point 2 then scores the positive against itself at training. Not leakage (the same happens at test) but a repeat-tuple channel the card narrates as taste, strongest for the thin, basics-heavy hm users; §3b-vii's "not leakage (the positive's own slot is removed)" is the sentence that misses it. Instrument: fraction of training positives with a basket twin; I-half overlap share for those vs the rest.
>
> ## D. The pre-registration clause (§3b-v) as accepted in §6.2(5)
>
> **14. Clause (a) has a cheap pass, the SE is the wrong one, and "best arm" over a 2×2 eats the margin.** (a) "best fUfI arm ≤ max(fI, fU) ⇒ adds nothing": the max is fU-noid 0.6082, whose host gap is −0.0060 (U host 0.6142). An fUfI arm at 0.615 passes (a) and (b) (bar 0.612) while sitting −0.0096 below UI — a worse stage-relative result than fU, failing the document's own registered expectation (1) (fUfI−UI < fU−U), which is listed as an expectation but not wired into the disconfirmation clause; the host delta UI−U = +0.0104 is never subtracted. (b) "best arm" is a max over the §6.2 2×2: with single-seed noise σ ≈ 0.006 the expected max of four draws exceeds the true level by ≈ 1.03σ, so "−2·SE" is effectively "−1·SE". And the quoted SE is one row's binomial, whereas the comparison is a per-user difference against the UI row on the same 6,034 users (paired, McNemar-style on discordant hits, if the evaluator's negative sample is seed-fixed across rows — seed 38210573 is shared). Three inflations, all in the candidate's favour, in a clause the user accepted as "the reading rule"; rebuttal 5b.2-8 conceded power and then wrote the wrong test.
>
> **15. The clause was written for an arm set that no longer exists and is silent on the knob the user flipped.** The per-arm expectations name "fUfI-mixed" (now optional) and say nothing about shared vs `_sep`: no prediction for the one designed decision the default embodies, and no rule for "the default loses to its arm" — is the headline row the default (a headline that lost its own ablation) or the best (selection, point 14)? Rebuttal 5b.2-10 ("naming the physics questions with their instruments *is* the deliverable") leaves this knob's deliverable empty.
>
> **16. The expectations contradict the document's own mechanism analysis and are parent-floor checks, not a bar.** A13 states fUfI on ml-1m ≈ fI-on-UI + user-ID row; the item composition costs ≈ −0.056 at dev tier on the I host (§5.1 R6) and the user-ID row −0.047 on fU (§3.7 A2). Additively the document's own numbers put the ml-1m point expectation near 0.52–0.57: clause (b) fails unless the linear-half refund (§3b-iii) returns ≥ +0.04, the one quantity nobody has measured. The registered bars (> 0.5613, > 0.5628, > 0.6082) are the lowest available parents; "fUfI-ids > fU-ids (0.5613)" is a bar 0.063 below the host. The honest pre-registration: "mechanism-derived expectation ≈ 0.57 on ml-1m; the bar is reached only if the refund delivers ≥ 0.04; the parent checks are secondary".
>
> ## E. Ratings, both directions
>
> **17. Too generous.** R8 (point 5). R6 "partial — the bet is finally assessable" should read "unassessed; mechanism-derived expectation negative at the bar on ml-1m, positive only through an unmeasured refund" (point 16). R5 "strong (cold items)" is strong for the noid arm only: in the ids default the cold representation is defined up to Σ_φ v_φ (point 9), and the default is ids. S4 "strong (conditional on R4)": in the ids arm the profile it names is gauge-dependent (point 7) — not an R4 condition but an identifiability defect of the naming input itself. S5 "strong": the default needs the table split of point 1, which §3.2/§4 price at zero.
>
> **18. Too harsh.** R2 was downgraded (A1) for the W-gauge of per-line bars; R2's text asks that explanations be "derivable from quantities the model computes at inference time … not reconstructed afterwards" — gauge-dependence is identifiability (requirements §4 "Identifiability of prototypes"), not post-hocness. R2 should return to strong with the gauge filed under R4/§4, where points 7–10 belong and where R4 "partial" is itself too mild for the ids arm. S1 "partial" penalises the absence of one specific twin while three decoupled rows (`lightfm_tags`, `lightfm_tags_ids`, `lightfm_hist`) sit in the fleet and S1 is a "may"; the fair grade is strong-with-a-gap.
>
> ## F. The explanation surface and the run plan
>
> **19. The item-side counterfactual is a phantom item; the natural exact counterfactual is substitution, and it is the identified quantity.** On a fixed-field vocabulary "remove row f" (§3.4 read-out 3, A6) yields a four-field item the catalogue cannot contain. The counterfactual a shopper can act on is "what if it were Black instead of Dark Blue": exact on the linear half, u*ᵀW^t(e_{f'} − e_f), and precisely the within-field contrast point 7 identifies; the document never offers it. From the same rendered example: "Menswear" (section) and the department words encode the target customer's gender and age segment, so §5.3's "gender is not in any vocabulary" is false on hm, and with department the strongest signal (5.2× lift, requirements R7) the predicted hm communities (point 4) are Ladieswear/Menswear/Divided/Children segments. The safety row should say user prototypes will be gendered by construction and auditable, not that the vocabulary is gender-free.
>
> **20. The run plan as decided is the one A2 says to refuse.** §4.5's minimum attributable set ranks `lightfm_tags_hist` first ("without the third row an fUfI−UI delta attributes to nothing … a cut that drops (1)'s control should be refused rather than reported as attributable"), the ID ladder second, `_shared` third. §6.2 commits the 2×2 {ids,noid}×{shared,sep} (ranks 2 and 3) and leaves the rank-1 control "to be decided at SC.6". If SC.6 declines the charter amendment, the plan is eight runs that, by the document's own rule, attribute to nothing.
>
> ---
> Rebuttals in §5b.2 that do not hold as written: 2(c) (point 10), 4 (point 9), 5 on R8 (point 5), 8 on the SE (point 14), 10 on "instruments are the deliverable" (points 8, 11, 15), 16 on LOO/twins being covered by the tie policy (point 13).

### Response (concede / rebut / open risk, point by point)

1. **Default arm unverified / no module design — concede.** Extends F-DC09-01 (the
   table split touches `FeatureEmbedding`/`HistoryFeatureEmbedding`, so the single-table
   path must stay bit-identical — dc01 t03 RNG/golden and the dc05 keystones are the
   guards; single-owner init of the shared block; the two `max_norm` slots become one)
   and F-DC09-02 (sharing keystone for SC.2/SC.3: the shared model ≡ the `_sep` model with
   E^U := E^I tied — scores equal and ∇_E L = ∇_{E^I}L + ∇_{E^U}L by the chain rule;
   LOO under sharing). §6.2's "4b: 11/14 confirmed" is a statement about the `_sep`
   mechanism — wording amended.
2. **Word–word affinity with a quadratic overlap channel — concede the mechanism,
   rebut the novelty consequence.** Derivation checked first-party: pers(u,i) =
   q_uᵀM_{ui}q_i with M_{ui} = A/‖q_u‖ + Bᵀ/‖q_i‖; a word in both basket and candidate
   contributes w_{u,f}·e_fᵀM_sym e_f — quadratic in ONE row under sharing, bilinear in two
   independently decayed rows under `_sep`. New mode **M16 (overlap / self-affinity
   channel)** with the instrument (overlap vs non-overlap word-pair split per half;
   overlap share vs word frequency), joined by point 13's basket-twin instrument →
   **F-DC09-10**. Positioning: the collapsed shared form is FM-shaped (pairwise
   basket-word × candidate-word inner products through a learned d×d core) and §2b must
   say so and distinguish it (FM/AFM: identity or attention core, no prototype units,
   no per-entity normalisation; here the core is prototype-mediated P̂ᵀW and the two
   normalisations are the cosine units'). Rebut: the prior-art verdict concerned the
   prototype double-tie mechanism, which no FM variant has — this adds a required
   related-work paragraph (B8/B12/B13 are in the corpus), it does not reopen the
   verdict. "Repeat your words narrated as taste": content overlap *is* the CBF signal;
   the pathology is its frequency weighting through M7, which the instrument separates.
3. **Intercept-direction leak into the user cosine; BCE loss-activity of B_user —
   concede both** (fold into F-DC09-03 as the precise mechanism of M7-shared: the leaked
   component lies along ŵ_t = W^tᵀ1/‖W^tᵀ1‖; instruments cos(ŵ_t, p^u_l), projection of
   q_u on ŵ_t vs basket popularity; w_u-direction gradient mass by loss — `bce` is in the
   searched loss set, verified). Host note: B_user's BCE loss-activity exists in host UI
   too (free u); new here is that the pushed rows are read by the item side.
4. **Two prototype sets on one cloud — open risk, instruments adopted** (new mode
   **M18 prototype-set redundancy** → **F-DC09-11**): row-max distribution of
   cos(p^u_l, p^t_k); canonical correlation between u*(q) and t*(q) over the same q;
   retained mass of each prototype set in the word rows' principal subspace (the "word
   subspace is statistical" point); `_sep` as control. Partial rebut: the two batch clouds
   differ (interaction-weighted basket means vs positives + uniform negatives ≈
   catalogue-uniform), K_u ≠ K_t — redundancy is a hypothesis with a test, not a
   prediction (learnings #12).
5. **Stale text / wrong gauge / R8 — concede; already F-DC09-02/04** (accepted at
   SC.1a). Added to F-DC09-02: ‖W^t‖_F/‖W^u‖_F as an M8 instrument. The R8 text
   formally withdraws the 5b.2-5 "one move" rebuttal.
6. **Two masters at unequal, dataset-dependent weight — partial rebut, concede the
   gap.** Weight counting: the mean normalisation gives every user's word part total
   weight F regardless of |H_u| (5 on hm; the mean bag size on ml-1m), so summed over a
   batch the user role carries ≈ B·ρ_f per word against the item role's ≈ B·N·ρ_f — the
   structural asymmetry is **N : 1 (candidate slots vs one user slot) on both datasets**,
   not |H_u|-driven; ml-1m's per-row dilution is the same total spread thinner. What does
   differ by dataset is the downstream factor (cosine gradient ∝ 1/‖q_u‖, ‖q_u‖ shrinking
   for heavy users — F-DC05-01) and A13's mean regression. Concede: "which role owns the
   geometry" had no instrument — M15's per-role gradient-norm split (F-DC09-03) is
   promoted to the owning-role instrument; the adversary's prediction (hm: `_sep` sharper
   user profiles; ml-1m: tie) is recorded as a hypothesis with its test, not pre-assigned.
7. **Translation gauge of the ids arm — concede fully; MAJOR → F-DC09-07.** Verified
   first-party: on hm (fixed five fields) e_f ← e_f + v_{φ(f)}, e^I_ID ← e^I_ID − Σ_φ v_φ,
   e^U_ID ← e^U_ID − Σ_φ v_φ leaves every q_i and every q_u^{LOO} unchanged (needs
   |H_u∖{i}| ≥ 1 — true on V1: 5-core, 25.1 % of users at the 3-train-row floor, so LOO
   baskets hold ≥ 2 items); exact except through L2/`max_norm`; dimension 5d. ml-1m:
   exact at inference with the bag-count correction, broken in training by LOO's varying
   bag mass. The lineage *knew* the gauge (dc06's "canonical within-field gauge", filed
   as a composition knob) but never drew the read-out consequence: in any ids arm every
   per-row share of all four components, the per-word intercept zoom, the ID line,
   every between-field comparison and the naming profiles cos(e_f, p) are
   gauge-dependent; identified are within-field contrasts and totals. Fix, read-out time
   only (no mechanism change — consistent with the user's "uncentred compositions"
   scoping): all per-row read-outs, coverage and naming reported in the **per-field-
   centred canonical gauge** (e_f − mean_φ over the field's vocabulary rows; ID line =
   e_ID + Σ_φ mean_φ, "item-specific + field baselines"); centred quantities are
   gauge-invariant functions of the trained model; totals unchanged; noid arms already
   identified. SC.2 toy claims: gauge exact on the fixed layout incl. LOO baskets;
   centred shares + totals invariant; ml-1m inference-exact. M7 trickle-back: dc01-ids
   and dc05-ids rendered shares on hm carry the same gauge — thesis hidden-effects
   section; dc06's record is the analysis to cite.
8. **Named coverage gauge-dependent — concede; folds into F-DC09-07.** Coverage is
   computed in the canonical gauge (identified); the across-item variance share of each
   linear-half component joins as a gauge-free cross-check; disconfirmation (c) refers to
   canonical-gauge coverage. dc08 4b C9's "parameterization-invariant" meant invariance
   to the word/identity re-partition, not to this symmetry — label corrected.
9. **Cold-item scores gauge-dependent under the ID drop — concede; MAJOR, shared →
   F-DC09-08.** Verified: after the drop q_cold = Σ_f e_f shifts by Σ_φ v_φ — a common
   shift that is rank-inert through t̂ and B_item on cold-vs-cold (per-user constants)
   but **rank-active through the cosine unit** (directions move) and on cold-vs-all. So
   the S0.3 §5 "drop the ID row" convention scores cold items in L2's canonical gauge
   in every ids arm, lineage-wide (fI-ids dev-tier cold rows included). Invariant
   convention proposed: **mean-ID imputation** — cold q = Σ_f e_f + ē_ID with ē_ID the
   mean of that side's warm ID rows (shifts by −Σ_φ v_φ exactly as the words shift by
   +Σ_φ v_φ; user-side mirror for cold users). Under L2 ē_ID is expected small, so the
   numerical change is expected small — which is exactly what the adversary's cheap
   instrument measures (gauge-shift a trained fI-ids checkpoint, verify warm
   bit-identity, re-run cold eval; login-node class, existing dc01 checkpoints). Timing:
   the thesis-tier cold wave (wave 2) has not run — adopting the invariant convention
   first makes every reported cold number convention-clean; landed dev-tier rows are
   disclosed, not re-run (user's call at the gate). Rebut the "84 % user rows set the
   item centroid" consequence: it describes L2's canonical gauge, which the invariant
   convention makes irrelevant.
10. **Adam drift ≠ init noise — concede wording** (fold into F-DC09-02): the gauge
    component is optimiser-path-dependent and score-inert, decayed only by weight decay;
    never "init noise".
11. **Common-mode mass inside the personalised halves — concede → F-DC09-09.** Three
    per-item scalars on ml-1m (B_item, (W^uμ)ᵀt*, u*(μ)ᵀt̂) with μ the basket-mean
    centroid; instrument: two-way decomposition (item main / user main / interaction)
    of each component on a sampled user×item grid, reporting the user-independent share;
    M8's variance-share instrument kept but demoted (blind to this). Renderer
    consequence deferred to SC.5 (a "common to all users" line is arithmetic, F-DC01-12
    category 1 — decided there). The adversary's "exceeds the ID-row share on ml-1m" is
    recorded as a hypothesis with its test.
12. **hm ladder ≠ ml-1m ladder / item–item matching / the 3-row question — (a) concede
    as a reading note** (A13's ID-share-of-‖q_u‖-by-band instrument, read per dataset:
    the ids/noid contrast isolates the user-ID row on ml-1m and weakly on hm); **(b)
    concede** — it is F-DC09-02's centroid identity in its hm form (q_u^{LOO} = mean of
    ≥ 2 item vectors; I-half = FISM-through-words); **(c) rebut**: V1 is 5-core and the
    25.1 % sit at the 3-TRAIN-row floor (dc05 §4, verified against the artifact
    2026-07-12; `hm_splitter.py` k-cores before the leave-one-out split), so q_u^{LOO}
    is never 0 on hm and C11a is not the modal case. A13 is correct; a clarifying
    parenthesis is added (F-DC09-04).
13. **LOO and basket twins — concede as instrument** (joins M16, F-DC09-10): fraction
    of training positives with a basket twin; overlap share for those vs the rest. Not
    leakage (identical at test); a repeat-tuple channel to disclose.
14. **Disconfirmation clause — concede all three → F-DC09-12.** (i) The **headline arm is
    pre-named**: `feature_user_item_proto` (ids, shared) — mirroring fI/fU, where ids is
    the headline — the bar applies to it and the arms are reported beside it (no post-hoc
    max; the ≈ 1.03σ selection inflation disappears). (ii) Clause (d) wires expectation
    (1): the stage-relative gap (UI − fUfI_headline) is compared with (U − fU) = 0.006
    and (I − fI; n100 host pending). (iii) The SC.8 test is **paired per user** (same
    6,034 users, seed-fixed negatives: McNemar on discordant hits) — per-user hit
    vectors must exist: SC.3 verifies the Tester exposes them or adds a login-node-class
    post-hoc script.
15. **Silent on shared vs sep; headline = default or best — concede** (F-DC09-12):
    headline = default; if an arm beats it, "the headline lost its own ablation" is
    reported with the arm's number and the selection caveat, never re-selected.
    Shared-vs-sep hypotheses pre-registered: accuracy open (no expectation);
    explanation side — shared predicts cross-prototype comparability (M18 is the
    counter-hypothesis), `_sep` predicts sharper hm user profiles (point 6).
16. **Expectations vs mechanism; parent floors — partial concede.** The honest
    sentence: "no additive argument predicts reaching the bar; the registered positive
    mechanism is the linear-half refund (3b-iii), unmeasured; the parent checks are
    secondary." Rebut the 0.52–0.57 point estimate: cross-host additivity (I-host fI
    cost + U-host user-ID cost) is not licensed; the fI −0.056 is a dev-tier number
    against an I-host n100 row still pending (wave 1), and the lineage's own record
    flags that regression as a budget artefact at matched budget (fI′ close,
    2026-09-08). Recorded as the adversary's estimate, not adopted.
17. **Too generous — R8 conceded** (SC.1a); **R6 wording conceded** ("partial —
    unassessed; mechanism-derived expectation not positive except through the
    unmeasured refund"); **R5 conceded conditionally** ("strong (cold items) under the
    invariant cold convention of F-DC09-08; gauge-dependent under the current drop");
    **S4 conceded** ("strong, conditional on R4, profiles read in the canonical gauge in
    ids arms"); **S5 — rebut the grade, concede the sentence**: S5 is the per-step
    pipeline profile (unchanged: lookups + two small matmuls); the table split is one-off
    build cost — stated in §4.
18. **Too harsh — R2: concede the re-scope** ("strong — derivability; per-line
    identifiability is an R4/§4 matter, now partly repaired (canonical gauge) and partly
    disclosed (W-slab)"); **S1: concede** ("strong-with-a-gap: three single-side
    decoupled rows in the fleet; the two-sided twin is the SC.6 decision"). Both fold
    into F-DC09-04.
19. **Phantom-item counterfactual; gender — concede both.** On fixed-field data the
    rendered counterfactual becomes **substitution within field** (u*ᵀW^t(e_{f'} − e_f):
    exact on the linear half, catalogue-realisable, and exactly the identified
    within-field contrast) — F-DC09-07; removal survives only on the bags layout. §5.3
    safety row corrected (F-DC09-04): section/department words (Menswear, Ladieswear,
    Divided, Children) encode the target customer's segment, so user prototypes will be
    segment-aligned by construction and auditable from the profiles — the intended
    property; "gender is not in any vocabulary" withdrawn.
20. **Run plan drops the rank-1 control — concede the tension, no new finding.** The
    `lightfm_tags_hist` charter amendment is an SC.6 gate decision taken before any
    submission; if declined, the attribution language is scoped (fUfI vs UI + the ID /
    table arms; no head-vs-composition attribution). Carried to SC.6 in this wording.

**Rebuttal re-check (the adversary's closing list):** 5b.2-2(c) withdrawn (point 10);
5b.2-4 narrowed (cold IS measured, but in a gauge — point 9); 5b.2-5 on R8 withdrawn
(SC.1a); 5b.2-8 on the SE withdrawn (point 14); 5b.2-10 holds in principle (instruments
are the deliverable) and the shared/sep knob now has them (point 15); 5b.2-16 holds for
the tie policy and is extended by the basket-twin instrument (point 13).

### Fold list

- **New findings:** F-DC09-07 [major] translation gauge + canonical read-out gauge +
  substitution counterfactual (points 7, 8, 19a); F-DC09-08 [major, shared] invariant
  cold convention (point 9); F-DC09-09 [minor] common-mode mass in the personalised
  halves (point 11); F-DC09-10 [minor] M16 overlap channel + FM positioning +
  basket-twin instrument (points 2, 13); F-DC09-11 [minor] M18 prototype-set
  redundancy (point 4); F-DC09-12 [minor] pre-registration repair (points 14–16).
- **Extended (accepted at SC.1a):** F-DC09-01 (+ table split guarded by dc01 t03 /
  dc05 keystones, single-owner init, `max_norm` unification, sharing keystone test);
  F-DC09-02 (+ chain-rule gradient claim, LOO-under-sharing claim, Adam wording,
  ‖W^t‖/‖W^u‖ as M8 instrument); F-DC09-03 (+ ŵ_t direction leak with instruments,
  BCE loss-activity of B_user, owning-role instrument with the hm/ml-1m hypothesis);
  F-DC09-04 (+ R2/S1/R5/R6/S4/S5 grade texts, §5.3 segment row, §2b FM-positioning
  note, A13 train-row parenthesis, §6.2 "11/14" scope).
- **Carried, no finding:** point 20 → SC.6 wording.

### Gate — CLOSED 2026-10-26

Walked through one finding at a time in plain language (Matteo asked to "go through
the points one by one … with a bit more context and detail"). Outcomes:

| Finding | Outcome |
|---|---|
| F-DC09-07 (read-out gauge) | **accepted as proposed** — canonical per-field-centred read-outs (vocabulary average) for ids arms; "Black instead of Dark Blue" substitution as the hm counterfactual; dc01/dc05-ids hm shares → hidden-effects note |
| F-DC09-08 (cold convention) | **accepted** ("yeah okay I guess") — mean-ID imputation as the shared S0.3 §5 convention; `cold_eval` change + dc01-checkpoint instrument before wave 2's cold jobs; dev-tier cold rows disclosed, not re-run |
| F-DC09-09 (common-mode mass) | **accepted, trimmed** — §3.1 wording fix; ONE one-off two-way decomposition per dataset; "common to all users" explanation line decided at SC.5; ranking angle = one disclosure line |
| F-DC09-10 (overlap channel) | **accepted, trimmed** — §2b FM/SVD++/AFM positioning; M16 description; ONE one-off overlap/twin measurement; frequency instrument dropped; **both shared and `_sep` arms confirmed** ("we definitely need to try out both here") |
| F-DC09-11 (prototype-set redundancy) | **accepted, trimmed** — M18 description; conditional caveat on read-out 7(b); ONE cross-prototype cosine-matrix measurement; CCA/subspace instruments dropped |
| F-DC09-12 (pre-registration repair) | **REJECTED** ("way too complicated, does not matter, let's just run the tests and look at them and reason about them, basta") — §6.2(5) clause stays as accepted, unamended; the inflations stay on record for the SC.8 reasoning only |
| Extensions on F-DC09-01..04 | **accepted**, with the three instruments on 03 dropped |

**Framing reminder recorded mid-gate (at 09):** "we are not optimizing a system, we
are proposing an algorithmic addition." Applied from 09 onward and retroactively
consistent with 07/08 (read-out and convention changes, no training or tuning
change): a finding counts for what it changes in the *description* of the addition
and in the *honesty of its explanations*; ranking-side consequences are one-line
disclosures; at most one one-off descriptive measurement per mechanism; no
instrument loops, knobs or mitigations. Saved as a standing memory.

**Discussion points worth keeping (thesis-relevant prose, captured per the
gate-deep-dive rule):**

1. *Thesis footprint of 07/08* (Matteo: "I don't know what of that will enter the
   thesis"): from 07 — one paragraph in the explanations chapter (word contributions
   on hm reported relative to the field average, because shifting a whole field
   leaves the model unchanged; the identity line includes the field baselines;
   counterfactual worded as substitution) + a short hidden-effects entry shared with
   dc01/dc05. From 08 — two sentences in the cold protocol (the convention and why)
   + one line on whether the instrument moved the numbers. The algebra, toy checks
   and ledger stay in the dossier → LLM-usage appendix. Placement decided at writing
   time via the placement ledger; the protocol entry is what makes it findable.
2. *Shared vs separate, intuitively* (Matteo: "intuitively which one is better?"):
   shared = the better *proposal* (one word, one meaning; "same kind again" handed
   over as one row scored against itself — two thirds of hm items have a
   word-identical twin; rare words trained from both roles; both prototype sets in
   one geometry, comparable in the same units). Separate = the better-*behaved*
   model (a basket word is evidence about the person, a candidate word is a property
   of the thing — "Baby/Children" in a basket means "shops for a child", on a
   candidate means "kids item"; no self-affinity shortcut; prototype sets less likely
   to become copies; it is the lineage's natural union of the dc01 item table and the
   dc05 user table). Expectation: hm shared wins or ties; ml-1m small gap either way
   (centroid shrinkage swamps both). Both conventions are respectable in the
   literature (FM shares one vector per feature across roles; SVD++ keeps history
   and candidate vectors separate). Shared is the committed design, `_sep` the
   honest control — exactly the pairing already chosen in §6.2.
3. *Number to re-verify at SC.2:* the fU-stage gap quoted in the F-DC09-12 proposal
   (0.006) disagrees with the thesis-tier fleet record (U host 0.613 vs fU 0.563 at
   n100); moot after the rejection of 12, but the figure must not propagate into any
   amendment text.

Next: SC.2 — apply the amendment bundle (F-DC09-01..11 as decided above, 12 closed-
rejected) to the design doc as dated visible amendments; update the candidate-index
row; one black-box toy re-check round (sharing keystone incl. chain-rule gradients
and LOO-under-sharing; centroid identity; gauge collapse — joint gauge exact,
per-side broken; translation gauge + canonical-gauge invariance on the hm fixed
layout incl. LOO baskets, ml-1m inference-exact; invariant cold convention);
`## SC.2 Concept fixes`, stop at its gate.

---

## SC.2 Concept fixes (2026-10-26)

*Protocol v1.3 SC.2 re-read before execution. Inputs: the SC.1a/SC.1b gate outcomes above (F-DC09-01..11 accepted, 09/10/11 trimmed under the "algorithmic addition, not system optimisation" framing; F-DC09-12 rejected); the design doc at 64f908c; the 4b claims spec (`claims_spec.md`, separate-table mechanism).*

### 1. Amendment bundle applied to the design doc (dated, visible; never a silent rewrite)

Applied by `apply_sc2_amendments.py` (session scratchpad; every anchor asserted unique, refuses a second run) — 16 dated markers in `dc09_fufi_double_tie_merge.md`, one in `s0_foundation.md`, index row updated. Index of the bundle = design doc **§6.4** (B1–B12); per finding:

| Finding | Where in the design doc | What changed |
|---|---|---|
| F-DC09-01 | §3.2 (new block after the config names), §3.2 item 6, §6.2 status, §6.3 | shared-table build spec: one word block `E_words` owned/initialised once by the factory, per-side ID blocks, `word_table=` on both compositions with a bit-identity guard on the single-table path, one `max_norm` key, cold finders reach the composition through the wrapper's `base`, `config_expects_*_id_drop` registration, sharing keystone t07 (chain rule); "4b 11/14" re-scoped to `_sep` |
| F-DC09-02 | 3b-x (appended), 3b-i wording | per-side scale gauge FALSE under sharing, joint gauge survives, ‖W^t‖_F/‖W^u‖_F identified; centroid identity (ids/noid/LOO forms; I-half = FISM-through-words); "init noise" → optimiser-path-dependent, score-inert |
| F-DC09-03 | §3.6 (M15, M7-shared incl. BCE-active B_user; M11 note), §5.5 rows | description only — instruments dropped at the gate; `_sep` is the control; descriptive hypothesis hm `_sep` sharper / ml-1m tie |
| F-DC09-04 | 3b-v, 3b-viii (A13 precision), §3.8 dev. 6/7, §4.5 (×3), §5.1 rows R1/R2/R5/R6/R8/S1/S4 + R4/R6 prose, §5.3 safety row, §6.2 | consistency with the §6.2 review decisions (shared default, 2×2 arms, mixed optional); grade texts; "gender not in any vocabulary" withdrawn (section/department words encode the segment); S5 build-cost sentence |
| F-DC09-05 | §3.6 M13, §3.5, §5.1 R6 prose | Tag-Genome provenance verified TRUE — ml-1m content partly behaviour-derived vs hm static taxonomy; fleet-wide disclosure (Ch. 3 Data) |
| F-DC09-06 | §3.6 M14, §5.3 Steck row | precision: cosine units break the diagonal-rescaling invariance Steck's argument needs → "mixed objective, semantic reading unvalidated", not "provably arbitrary" |
| F-DC09-07 | **3b-xi (new)**, §3.4 read-outs 3/5/7 + block before the renderer slot + example caption, §5.1 S4, §5.5 row | translation gauge of the ids arms (F·d on hm; inference-exact with bag-count correction on ml-1m, broken in LOO training); canonical per-field-centred read-out gauge (vocabulary average; ID line = item-specific + field baselines); substitution-within-field counterfactual; dc01/dc05-ids hm shares → hidden-effects section |
| F-DC09-08 | §3.4 cold paragraph (new block), §3.2 item 6, §3.8 dev. 7, §5.1 R5, §6.3; **S0.3 §5 (foundation doc, dated amendment)** | cold convention → mean-ID imputation on both sides; **empty-basket cold users keep q_u = 0** (imputing there would break invariance — a precision found while writing the amendment, tested as D7c); dc01-checkpoint instrument before wave 2; `cold_eval` change + s0 pin at SC.3; dev-tier rows disclosed |
| F-DC09-09 | §3.1 (wording), **3b-xii (new)**, §3.6 M8 note, §3.4 block (renderer line → SC.5), §5.5 row | "exactly one popularity-capable intercept" → algebraic statement about explicit intercepts only; common-mode mass described; one one-off two-way decomposition; ranking angle = one line |
| F-DC09-10 | §2b (positioning paragraph before the verdict), §3.6 M16, §5.5 row | FM/AFM/SVD++ positioning with the three distinguishing points; overlap (quadratic in one row) + twin channel; one one-off measurement; frequency reading folded into the M7 cross-link |
| F-DC09-11 | §3.6 M18, §3.4 7(b) caveat, §5.5 row | nested clouds; hypothesis with one cross-prototype cosine measurement; 7(b) conditional |
| F-DC09-12 | 3b-v (record only) | REJECTED by the user; the three inflations and the missing shared-vs-`_sep` expectation stay on record for the SC.8 reasoning |

Not amended (deliberately): the mechanism (§2.6/§3.1 equations), the arm set, the search space, the pre-registration clause's content, every grade not named above.

### 2. Black-box toy re-check (one round)

Spec: `Master/temp/dc_checks/dc09/claims_spec_sc2.md` (self-contained: both table layouts, both data layouts, LOO, three losses, canonical centring, both cold conventions). Claims D1–D11 cover every amendment that touches math: D1/D2 sharing keystone + chain rule + LOO routing (F-DC09-01/02); D3 centroid identity (02); D4 gauge collapse (02); D5 translation gauge incl. LOO baskets and the bag-layout breakage (07); D6 canonical-gauge invariance, totals, within-field contrasts, the vocabulary-vs-frequency convention (07); D7 cold conventions incl. the empty-basket precision (08); D8 common-mode shape (09); D9 overlap identity + quadratic-in-one-row (10); D10 nesting (11); D11 B_user loss-activity by loss (03). Subagent received only the spec; scripts to `check_claims_sc2.py`, report to `report_sc2.md`.

*(Run note: the first launch was interrupted by the user after reading the spec — no output; relaunched with the identical brief, same session day.)* Script ran in the `protomf` env, CPU, 49 s; 6 seeds × 3 shapes × layouts {fixed, bag, bag-with-repeats} × {ids, noid} × {sep, shared} × 7 scale sets (unit, 1e-3, 1e3, two mixed, all-negative, random per block). **Report (verbatim, abridged to the summary and the falsified sub-clauses — full numbers in `report_sc2.md`):**

> | claim | verdict |
> |---|---|
> | D1 -- Sharing keystone -- chain rule | CONFIRMED |
> | D2 -- LOO under sharing -- gradient routing | CONFIRMED |
> | D3 -- Centroid identity | CONFIRMED |
> | D4 -- Gauge collapse under sharing | FALSIFIED in (a) 'for any c > 0', (d) parenthetical 'identified quantity under sharing' |
> | D5 -- Translation gauge of the ids arms -- fixed layout | CONFIRMED |
> | D6 -- Canonical gauge makes per-row read-outs invariant | CONFIRMED |
> | D7 -- Cold conventions under the gauge | CONFIRMED |
> | D8 -- Common-mode mass in the personalised halves | FALSIFIED in (b) literal reading (ids arm, e_ID(u) per-user of fixed scale, only delta_u -> 0) |
> | D9 -- Overlap / self-affinity channel | CONFIRMED |
> | D10 -- Nesting of the user cloud | FALSIFIED in LS reconstruction 'gives exactly 1/\|H_u\| each, residual ~ 0' |
> | D11 -- B_user is rank-inert but loss-active under BCE only | FALSIFIED in (a) 'their gradients with respect to the direction w_u = W_u^T 1 in parameter space -- i.e. dL/db' |
>
> **D4(a) 'for any c > 0' — FALSIFIED.** With the spec's own guarded cosine (eps = 1e-8), once c·\|q_i\| < eps the cosine unit is no longer scale-invariant, so S changes. Correct statement: invariance holds for all c with c·min_i\|q_i\| ≥ eps. Edge-of-definition failure, not a substantive one.
>
> **D4(d) parenthetical 'identified quantity under sharing' — FALSIFIED.** Counter-map T_c in the shared IDS arm: E_w → E_w/c, E_Iid → E_Iid/c, W_t → c·W_t, W_u → W_u/c, e_ID(u) → c·e_ID(u) + (c − 1/c)·m_u with m_u = (1/\|H_u\|) Σ_j Σ_f e_f (full basket). Then q_i → q_i/c, q_u(full) → c·q_u, so S, all four components and both regularisers are unchanged at every full-basket (inference) score [1008 cases, max err 1.4e-15] while \|\|W_t\|\|/\|\|W_u\|\| is multiplied by c². What pins it is the training-time LOO: q_u^LOO(i) would need the per-user shift (c − 1/c)·m_{u,−i}, which varies with i, so T_c changes the LOO training scores [1008/1008]. Correct statement: 'identified under sharing' holds only for the noid arm or for the LOO training objective; for the ids arm's inference function it is a gauge quantity, as under sep.
>
> **D8(b) literal reading — FALSIFIED.** q_u = μ + δ_u + e_ID(u) keeps the per-user ID rows when σ = scale(δ_u) → 0, so users still differ by e_ID(u) and the user-main/interaction shares converge to the share produced by the e_ID(u) spread alone [0.164 at τ = 0.3; 5.6e-3 at τ = 0.05] — nonzero. Correct statement: the shares → 0 iff the WHOLE per-user part δ_u + e_ID(u) → 0 [then 2.8e-12 — CONFIRMED]; the noid limit (c) is CONFIRMED.
>
> **D10 LS reconstruction — FALSIFIED.** The residual is always ~0, but the minimum-norm least-squares coefficients equal 1/\|H_u\| only when the null space of [q_j]_{j∈H_u} is orthogonal to the all-ones vector (in practice: linearly independent, \|H_u\| ≤ d); 806/5535 random baskets fail; {g1},{g2},{g1,g2} gives α = (2/9, 2/9, 4/9). Correct statement: q_u = (1/\|H_u\|) Σ_j q_j always (convex hull — CONFIRMED, 2772 + 2772 cases), but LS recovers the weights only under linear independence.
>
> **D11(a) 'gradients w.r.t. the direction w_u — i.e. ∂L/∂b' — FALSIFIED.** Moving parameters along w_u is NOT a pure B_user shift — it also moves û (so the I-half changes through t\*) and u\*; the directional derivative is nonzero in 7190/7190 cases. Correct statement: ∂L/∂b = 0 for softmax/BPR [≤ 2.8e-16], equivalently the component of the gradient that flows THROUGH the B_user term, (Σ_i ∂L/∂S_ui)·w_u, is exactly 0.
>
> Also CONFIRMED (selected numbers): D1(a) bitwise in f64 and f32 over 504 configs, gradients ≤ 2e-16; D3 ≤ 2e-15; D5 incl. LOO baskets; D7 with rank flips under "drop" (6561 cold–cold, 122,430 cold–warm pairs) and invariance under mean-ID imputation incl. the empty-basket precision; D11(b) BCE ∂L/∂b ≠ 0, (c) rank-inert.

**First-party response.**

- **D4(d) — concede; a real error in the F-DC09-02 amendment → new finding F-DC09-13 [minor].** 3b-x as amended says the cross-side ratio "is an *identified* quantity under sharing … recorded as the one descriptive number". True in the noid-shared arm only. Verified with independent code (`dc_checks/dc09/verify_tc_firstparty.py`, fixed layout, ids-shared): full-basket |ΔS| ≤ 8e-14, LOO |ΔS| up to 96, ratio × c² exactly. The same map moves the **user-side split between word part and user-ID row** (word part ÷c, ID row absorbs) — so in every ids arm with a user-ID row that split is pinned by the leave-one-out training objective alone: the LOO residual is (c − 1/c)(m_u − m_{u,−i}) = O(1/|H_u|) — strong on hm (3–10 train rows), weak on ml-1m (|H_u| ≈ 95). Lineage: on the U host (dc05 fU-ids) q_u reaches the score only through the cosine unit (`feature_extractor_factories.py:200–244`: item side free in R^{K_u}, `use_weight_matrix=False`), so the freedom there is **per user** (e_ID(u) → λ_u·q_u − m_u, any λ_u > 0, inference-exact), again pinned only by LOO — the ID-share-by-history-band reading (F-DC05-01, A13) reads a quantity weakly pinned at long baskets. Description-level consequence only (framing 2026-10-26): no instrument, no read-out change (a scale freedom — F-DC09-07's canonical gauge does not touch it), no mechanism change; noid arms unaffected.
- **D4(a) — rebut as a spec edge.** The design doc never says "any c"; real composition norms sit far above the guard. No amendment.
- **D8(b) — the doc does not claim the falsified limit** (3b-xii states the common-mode terms, D8(a), confirmed), but its "heavy users shrink toward μ" invites the misreading. **Precision clause P1** proposed for 3b-xii: the common-mode share approaches the whole personalised half only when the entire per-user part δ_u + e^U_ID(u) is small — in ids arms the user-ID row stays personal even for heavy users, which is exactly what the ml-1m hypothesis compares.
- **D10 — spec-method error, no doc consequence.** The least-squares sub-clause was my verification recipe, not a design-doc statement; M18's nesting claim is the convex-hull identity, confirmed.
- **D11(a) — spec wording error, no doc consequence.** The M7-shared text says B_user "cancels within the user" under softmax/BPR — the confirmed ∂L/∂b = 0 / B_user-routed reading; the SC.1b phrase "pushes every basket word along w_u" is the routed gradient under BCE, also confirmed (D11(b)).
- **Observations (not falsifications), dispositions:** `_sep` carries a 2F·d translation gauge (each table its own shift) vs F·d shared — **precision clause P2** for 3b-xi; canonical centring is itself a D5 map, so it would move LOO scores of |H_u| = 1 users — already conditioned in 3b-xi (|H_u∖{i}| ≥ 1, true on V1), and the canonical gauge is a read-out transform on full baskets (D6 confirmed); D1's bit-identity holds only for identical op sequences — **carried to SC.3 t07** (assert bitwise only where the shared and tied-`_sep` paths share op order, else f64 machine precision); noid t\*-path into a positive-only word is 0 when q_u^LOO = 0 — unreachable on hm V1 (3-train-row floor) and ml-1m; D9(b)'s diagonal term carries the in-item multiplicity when words repeat — not reachable (distinct words per item on both datasets); BCE ∂L/∂b exponentially small in doubly-saturated rows — noted, no consequence.

### 3. Number re-verification (SC.1b gate, discussion point 3)

Resolved, no error: the doc's n100 table (§5) quotes **test** HR@10 from `summary.md` read from LEO5 2026-10-26 (U host 0.6142, fU-noid 0.6082, fU-ids 0.5613); the fleet memory record quotes the **stratified-readout** regime for the same runs (0.613 / 0.604 / 0.563). The disputed 0.006 is the noid-arm gap (0.6142 − 0.6082); the ids-arm gap is 0.053 — both appear only in the verbatim 5b critique and the SE sentence, never in SC.2 amendment text. **Carried to SC.6/SC.8:** every quoted row names its regime; comparisons use the readout regime (fleet record rule: never mix `summary.md` test numbers with readout numbers).

### 4. New finding + proposals at this gate

- **F-DC09-13 [minor]** (ledger): dated correction to the 3b-x amendment — the cross-side ratio is identified in the noid-shared arm only, pinned by LOO training alone in ids arms, quoted for noid-shared only; one disclosure sentence — in every ids arm with a user-ID row the user-side word/ID split is set by leave-one-out training alone (strongly on hm, weakly on ml-1m; per user on the U host), so user-side per-word shares and the ID-share-by-band reading on ml-1m partly reflect weight decay and the optimiser path → hidden-effects pointer (dc05 + dc09). Ledger F-DC09-02's "identified M8 instrument" withdrawn for ids arms.
- **P1** (3b-xii precision clause, D8) and **P2** (3b-xi: F·d shared / 2F·d `_sep`, D5) — wording only.

### 5. Ledger status updates (applied at gate close)

| Finding | Status | Note |
|---|---|---|
| F-DC09-01 | stays **open** | doc half resolved (D1/D2 confirmed); code half + t07 at SC.3 |
| F-DC09-02 | **fixed** | D3/D4(b,c) confirmed; the ratio sentence superseded by F-DC09-13 |
| F-DC09-03 | **fixed** | D11 confirmed (BCE activity); doc text unaffected by the D11(a) spec slip |
| F-DC09-04, -05, -06 | **fixed** | document-only |
| F-DC09-07 | stays **open** | doc half resolved (D5/D6 confirmed); canonical-gauge read-outs implemented at SC.5 |
| F-DC09-08 | stays **open** | doc half resolved (D7 confirmed incl. empty basket); `cold_eval` + s0 pin at SC.3; dc01-checkpoint instrument before wave 2 |
| F-DC09-09 | stays **open** | doc resolved (D8(a)/(c) confirmed; P1); "common to all users" renderer line at SC.5 |
| F-DC09-10 | **fixed** | D9 confirmed; one-off measurement → SC.6 run plan |
| F-DC09-11 | **fixed** | D10 hull confirmed; one-off measurement → SC.6 run plan |
| F-DC09-12 | **wontfix** | rejected at the SC.1b gate |
| F-DC09-13 | **open → fixed** at gate close if accepted | document-only |

### 6. Gate — CLOSED 2026-10-26

Decisions asked: (1) F-DC09-13 as proposed? (2) P1 + P2 wording? User: **"yes i accept"** — both. Applied the same day: 3b-x marker + dated correction paragraph (F-DC09-13), P1 in 3b-xii, P2 in 3b-xi, §6.4 B13 + toy-round result line, §6.2 status line, candidate-index row; ledger statuses set exactly as in §5 (01/07/08/09 open for their SC.3/SC.5 halves, 12 wontfix, the rest fixed). First-party check script persisted as `Master/temp/dc_checks/dc09/verify_tc_firstparty.py`.

**Carried to later steps:** SC.3 — t07 bitwise only under identical op order; F-DC09-01 code half; F-DC09-08 `cold_eval` + s0 pin, dc01-checkpoint instrument before wave 2. SC.5 — F-DC09-07 canonical-gauge read-outs; F-DC09-09 renderer line. SC.6 — one-off measurements of F-DC09-09/10/11; `lightfm_tags_hist` decision; n100-tier C4 deviation; regime named on every quoted row (§3). Hidden-effects section — F-DC09-07 (translation gauge, dc01/dc05-ids hm shares) and F-DC09-13 (user-side word/ID split pinned by LOO alone, dc05-ids per user).

Next: SC.3 — implementation from scratch (dc09 is unbuilt): verify preconditions, write a condensed implementation plan in `Master/temp/`, gate it, then build.

---

## SC.3 Implementation alignment (2026-10-26 —)

*Protocol v1.3 SC.3 re-read before execution. dc09 is unbuilt → the from-scratch path: preconditions, condensed plan, gate, build.*

### 1. Plan (gate CLOSED 2026-10-26 — G1–G5 accepted: "1. I agree, 2. yes agree. 3. yes it is. 4. yup")

Registered: `Master/temp/dc09_feature_user_item_proto_implementation_plan.md` — preconditions P1–P4 (datasets live only on LEO5 scratch, so real-data tests and smokes run on the login node in a detached worktree while wave 1 holds the fleet checkout); decisions G1–G6; build stages A–E; tests t01–t09 + an s0 pin. Decisions asked: **G1** shared table realised as one factory-owned `[words; item IDs; user IDs]` embedding passed to both compositions (deviation from SC.2's `word_table=` wording — same mechanism, existing op sequence kept, t07 bitwise); **G3** four 2×2 configs only; **G4** mean-ID imputation fleet-wide incl. `lightfm_tags_ids`, `drop` kept as an option; **G5** explanation stack split (accessor + naming at SC.3; breakdown slots + canonical gauge at SC.5). G6 (submit after SC.4/SC.6, SC.5 during the queue wait) is flagged for the SC.4 gate.

### 2. Build record

| stage | commit | what landed |
|---|---|---|
| A — core model + registration | `c9f3baa` | `FeatureEmbedding` / `HistoryFeatureEmbedding` accept a borrowed `embedding_layer` (shape-checked, not owned, not initialised) and HFE an `id_block_offset`; defaults = the old code path. `PrototypeEmbedding.sim_from_embed` (the old `forward` tail, split out). `TwoRoleComposedExtractor` (one composition read per side per step, fed to both roles). Factory branch `feature_user_item_proto` (G1 shared `[words; item IDs; user IDs]` table or two owned tables under `_sep`; config never mutated; asserts per plan). `inject_feature_ids` injects both payloads. Four 2×2 configs; `start.py` + `run_combo` registration. Modifications log updated. |
| B — cold convention | `3cb3cad` | `cold_eval`: mean-ID imputation is the default fleet-wide (G4, F-DC09-08), `--cold-id-convention drop` keeps the old rows reproducible; finders recurse through the wrapper and `config_expects_*` know the new ft_type (F-DC09-01 code half); user side uses `id_row_offset` (F-DC09-14, trivial). Pins: new `dc_checks/s0/t13`; dc01 t11 runs both conventions. |
| C — explanation accessor (G5, SC.3 half) | `af1d4b9` | `FeatureUserItemProtoAccessor` (both prototype sets, composed factors, both projections, per-side word rows); `_sep`/`_noid_sep` suffix normalisation; pipeline + generic explainers register the model; user-side intrinsic naming reads the user's own word rows. `weight_viz` stays excluded until SC.5 (same û_k·t*_k folding as dc02's exclusion). |
| D — cluster (no code change) | — | `feat/scrutiny` pushed 86f7a77..af1d4b9 (user OK, 2026-10-26); detached worktree `~/UIFProtoMF/ProtoMF_dev` at `af1d4b9` — the fleet checkout stays at `3ce563d`. Real-data test t08, eight smokes, explanations pipeline on one smoke checkpoint, cold-convention instrument. Scripts + evidence: `Master/temp/dc_checks/dc09/{t08_vocab_alignment_real.py, smoke_dc09.sh, d01_cold_convention_instrument.py, stage_d/}`. |
| E — regression sweep | — | After each of A, B, C: dc01 / dc05 / dc07 / dc08 / s0 suites equal the P1 baseline except the checks added on purpose. D added no code, so the stage-C sweep is final. The baseline's 8 pre-existing failures are unchanged: dc01 t03 + dc07 t06 (stale golden, environment drift 1.19e-6), dc01 t05 + t10 (`run_combo.EXPLAINABLE_MODELS` emptied 2026-08-17), dc05 t04 (`user_side_additions_exact`) + t06 (`l2_mass_equal`), dc08 t04 (import path), s0 t04 (needs data). |

**Tests** (`Master/temp/dc_checks/dc09/`, plus the s0 pin):

| test | laptop (torch 2.5.1, CPU) | LEO5 login node (torch 2.5.1, Linux) |
|---|---|---|
| t01 keystone (F=0 ≡ UI host) | pass | pass |
| t02 LOO reaches both user roles | pass | pass |
| t03 single-table path untouched | pass (bitwise vs the pre-change golden) | t03a fails as expected: 48/51 golden tensors differ at float32 rounding (≤ 1.2e-6 rel). The golden was captured on macOS and is bitwise only there; same class as the dc01 t03 drift. t03b passes. |
| t04 registration | pass | pass |
| t05 cold (finders, imputation, abort, offsets) | pass | pass |
| t06 four components | pass | pass |
| t07 sharing keystone | pass | pass |
| t08 vocabulary alignment, real splits | — (data only on LEO5) | pass on hm_1_month, ml-1m, hm_1_month_cold, ml-1m_cold (below) |
| t09 checkpoint round-trip | pass | pass |
| t10 accessor | pass | pass |
| s0 t13 cold convention pin | pass | pass |

**t08 on the real splits** (production seam: canonical resolution → `inject_feature_ids` → factory): both sides resolve the same fields and layout and one index — hm V = 426 (5 fixed fields), ml-1m V = 1061 (genres + tags, bags, D_max 72) — equal to the label table's length; the user builder's leave-one-out token table *is* the item code table (ids and weights); every code is a word. An independent rebuild of w = n_{u,f}/|H_u| from the item codes and the variant's own train pairs matches every (user, word) cell (1.37 M on hm, 2.71 M on ml-1m; max abs 3e-8, float32 storage). On both `_cold` variants no train pair touches a cold item. All four arms build on the real payloads with the planned table layout (shared ids: V + M + N rows, user-ID offset V + M; `_sep`: V + M and V + N); a manual float64 recomposition of q_i and q_u from the table rows matches the bases' outputs on 512 sampled objects (≤ 1.6e-7 rel); scores finite. 29 s, 2.6 GB RSS.

*Observation (no claim):* on hm_1_month_cold, 143 users lost their whole train basket to cold items. They appear in no cold-test row, but in 112 of 59,048 warm-test rows (0.19%), where every model scores them without training signal (fU/fUfI: empty history). This is a property of the variant, shared by all rows.

**Cold-convention instrument (F-DC09-08; one-off, descriptive).** `d01_cold_convention_instrument.py` on the two trained hm_1_month_cold checkpoints that carry item ID rows: dc01 fI-ids and S0.4 lightfm_tags_ids (dev tier, 2026-08-20 cold reports). Each checkpoint is read in three gauges of the same trained model. *Trained* is the checkpoint as saved. *Centred* centres each field's words on their vocabulary mean and moves the means into the ID rows. *Random* applies a random per-field shift at 3× the word-row std. Each gauge is scored under both conventions, with the fleet runner's datasets and seeds. The instrument writes only to the smoke folder, never into the checkpoints' `cold_eval/`.

| checkpoint | gauge | drop: cold-vs-cold / cold-vs-all HR@10 | impute: cold-vs-cold / cold-vs-all HR@10 |
|---|---|---|---|
| fI-ids | trained | 0.2876 / 0.2179 | 0.2874 / 0.2187 |
| fI-ids | centred | 0.1857 / 0.5604 | 0.2874 / 0.2187 |
| fI-ids | random | 0.2833 / 0.0246 | 0.2874 / 0.2187 |
| lightfm_tags_ids | trained | 0.5080 / 0.5082 | 0.5080 / 0.5082 |
| lightfm_tags_ids | centred | 0.5080 / 0.5109 | 0.5080 / 0.5082 |
| lightfm_tags_ids | random | 0.5080 / 0.5035 | 0.5080 / 0.5082 |

- **Integrity.** The gauge-shifted models score sampled warm test rows identically (≤ 4e-7 rel, float32). Trained × drop reproduces both fleet reports exactly (abs diff 0.0).
- **The old drop rule depended on the gauge.** Across three parameterisations that score every warm row the same, fI-ids reads 0.19–0.29 cold-vs-cold and 0.02–0.56 cold-vs-all. LightFM's cold-vs-cold is the exception: under a plain dot product the drop shifts every cold candidate's score by the same user constant, so ranks stay; its cold-vs-all still moves (0.504–0.511).
- **Imputation does not depend on the gauge.** It gives one number per checkpoint in all three gauges.
- **The published rows stand.** In the trained gauge, drop and imputation agree to ≤ 0.0008 HR@10, because the trained warm-ID mean is almost zero: ‖ē_ID‖ = 0.00055 against a mean row norm of 0.078 (fI), and 0.00018 against 0.060 (LightFM). This fits weight decay picking the minimum-norm point of the gauge orbit, where ē_ID = (n_φ/M)·(field-φ word mean) for every field φ, which is small when M = 13,651 ≫ n_φ. So the dev-tier cold rows landed under drop need only the one disclosure sentence F-DC09-08 asked for: they equal the amended convention to ≤ 0.001 HR@10 on the measured checkpoints. The convention change buys invariance, not a different number.
- **Not covered.** fU-ids (no cold-user variant exists) and dc09 (not trained yet).

**Smokes** (login-node A30, `smoke_dc09.sh`: `run_combo --profile smoke --num-samples 2 --n-epochs 3`, W&B off, results under `/scratch/c7031336/smoke_runs/dc09/`). 2 × 3 rather than the full smoke profile (5 × 15), as at the dc05/dc07 smokes: this is a pipeline check on a shared node, and the full profile would take ~1–2 h per hm arm. Test HR@10 after 3 epochs is pipeline evidence only, never a comparison.

| dataset | arm | trials | test HR@10 (3 epochs) | peak VRAM | wall |
|---|---|---|---|---|---|
| hm_1_month | ids, shared | 2/2 | 0.385 | 3.7 GB | 528 s |
| hm_1_month | noid, shared | 2/2 | 0.378 | 3.6 GB | 530 s |
| hm_1_month | ids, `_sep` | 2/2 | 0.402 | 3.9 GB | 526 s |
| hm_1_month | noid, `_sep` | 2/2 | 0.403 | 3.6 GB | 524 s |
| ml-1m | ids, shared | 2/2 | 0.362 | 3.6 GB | 365 s |
| ml-1m | noid, shared | 2/2 | 0.408 | 3.6 GB | 360 s |
| ml-1m | ids, `_sep` | 2/2 | 0.416 | 3.6 GB | 363 s |
| ml-1m | noid, `_sep` | 2/2 | 0.429 | 3.6 GB | 360 s |

All eight complete the hyperopt path (Ray trials, best checkpoint, test evaluation, results persisted). Validation HR@10 rises from 0.02–0.04 at initialisation to 0.2–0.4 within 3 epochs. Per-epoch time with two trials sharing one A30: ~125–135 s on hm, ~80 s on ml-1m. That figure is for SC.4's optimisation-parity check against the hosts; it is not judged here.

**Explanations pipeline** on the hm ids-shared smoke checkpoint (CPU): naming on both sides (intrinsic cosine on the word rows, plus post-hoc lift), t-SNE, top-k items and prototype cards on both sides, per-field small multiples on the item side. 19 files, exit 0; `weight_viz` correctly absent. The content is not read: it is a 3-epoch checkpoint, and SC.5 owns it.

### 3. Gate — CLOSED 2026-10-27

Decisions asked: (1) close F-DC09-01 and F-DC09-08 as fixed, with the instrument result added to S0.3 §5 and the one-sentence disclosure for the landed dev-tier cold rows; F-DC09-07/-09 stay open for SC.5; (2) accept two deviations from the plan: smokes at 2 trials × 3 epochs instead of the full smoke profile, and t03's bitwise golden holding only on the capture platform (cluster differs at float32 rounding, ≤ 1.2e-6). User: **"yes accept both"**. Applied the same day: ledger statuses (01, 08 → fixed with resolution notes; 14 logged trivial/fixed), S0.3 §5 result line, design-doc §6.2 status, candidate-index row.

**Carried to later steps:** SC.4 — independent spec-vs-code pass; optimisation parity incl. the per-epoch time (~125–135 s hm / ~80 s ml-1m with two trials per A30); G6 (submit after SC.4/SC.6, SC.5 during the queue wait) raised at the SC.4 gate. SC.5 — F-DC09-07 canonical-gauge read-outs, F-DC09-09 "common to all users" line, `weight_viz` decision, user-side small multiples. Disclosure — the dev-tier cold rows equal the amended convention to ≤ 0.001 HR@10 (measured above).

Next: SC.4 — implementation scrutiny.

---

## SC.4 Implementation scrutiny (2026-10-27 —)

*Protocol v1.3 SC.4 re-read before execution.*

### 1. Spec-vs-code, black-box

Subagent launched 2026-10-27 with ONLY the corrected design doc (`Master/docs/design_candidates/dc09_fufi_double_tie_merge.md`) and the implementation files (`feature_extractors.py` relevant classes, the factory branch + helpers, `feature_ids.py` injection, the four configs + host config, `rec_sys.py`, `cold_eval.py` convention + finders, the accessor, the registration lines); brief: "where does the code diverge from the spec?" — no dossier, ledger, tests or notes. Report attached verbatim below on return, then the point-by-point response.

#### Subagent report (returned 2026-10-27; every finding, location and number as returned — list layout condensed into paragraphs)

> **Scope.** I read only the spec (`Master/docs/design_candidates/dc09_fufi_double_tie_merge.md`, all 2,099 lines, with its amendments treated as binding) and the named code files. I ran nothing.
>
> **Bottom line.** I found **no blocker and no major divergence**. The score, the within-side tie, leave-one-out, payload injection, configs, cold-row convention and registration all match the amended spec. The findings below are minor or trivial. Two are missing pieces the spec asked for (finding 1, and the ladder blocked in finding 2). One is a documented deferral that affects how honest the explanations are (finding 3).
>
> **1. MINOR — `normalize_composed` (the length-ablation control) is not implemented, and a config that sets it is silently ignored.** Spec: §3.2 item 1 class spec (`normalize_composed: bool = False`, `q = q / q.norm(...)` in `forward`), the §3.2 factory sketch, §3.7 length-ablation control, §6.3 build notes (`_norm` arm, keystone t03 "`normalize_composed` no-op on the cosine units (C9)"). Code: `feature_extractors.py:1060-1061` (no such parameter), `:1096-1100` (`forward` never normalises), `feature_extractor_factories.py:260-333` (key never read or rejected); the string occurs in none of the named files. Effect: a future `_norm` config would train the unablated model with no error. Classification: spec requirement not implemented, omission undocumented; the 2026-10-26 run plan (2×2) makes the arm optional. Severity minor; cheapest fix an assert that rejects the key.
>
> **2. MINOR — the vocabulary-equality assert blocks the spec's single-side reductions and ladder rungs, even in `_sep`.** Spec: §0.1 M3 (the two single-side reductions), §3.7 isolating ladder (fI-on-UI, fU-on-UI per the A2 amendment), §4.5 priority (4). Code: `feature_extractor_factories.py:281-283` asserts `ip['n_features'] == up['n_features']` in both table modes. Arithmetic (hm): user `feature_fields=[]` → user `n_features = 0` vs item 426 → the assert fires (ml-1m 1,061 ≠ 0). In the shared layout the equality is necessary (user ID rows at `id_row_offset = n_features + id_block_offset`); in `_sep` nothing needs it. Classification: code choice where the spec leaves the arm optional. Severity minor.
>
> **3. MINOR — the accessor exposes raw word rows in ids arms; the canonical per-field-centred gauge is not applied yet (documented deferral).** Spec: 3b-xi + §3.4 SC.2 amendment (F-DC09-07). Code: `accessor/feature_user_item_proto.py:75-79` returns `weight[:n_features]` uncentred; lines 28-29 advertise intrinsic grounding on both sides. Classification: deliberate and documented (accessor docstring lines 12-13: "SC.5 work"). Severity minor today; an explanation-honesty problem if any naming/breakdown output on an ids-arm checkpoint is produced before SC.5 (cos(e_f + v_φ, p) ≠ cos(e_f, p)).
>
> **4. TRIVIAL — the shared table is one packed `nn.Embedding` [words ; item IDs ; user IDs], not a V×d word block plus per-side ID blocks via `word_table=`.** Code: factories 295-306; `FeatureEmbedding` 261-273; `HistoryFeatureEmbedding` 554-567, 599-607. Offset check ids-shared hm (V=426, M=13,651, N=73,418): total 87,495; item IDs [426, 14,077); user `id_row_offset` 14,077; user IDs [14,077, 87,495); 87,069 ID rows = the spec's M18 figure; noid-shared 426 rows; mixed layouts consistent. Functionally equivalent (disjoint ranges, per-element optimisers, no `max_norm`), conditional on the init scale ((c)1). Deliberate, documented (factory 267-271, docstrings).
>
> **5. TRIVIAL — the "warm" set for mean-ID imputation is "every non-cold ID", not "trained on the variant" (spec gap).** Spec §3.4 F-DC09-08 does not define "warm". Code: `cold_eval.py:299` (`warm = np.setdiff1d(np.arange(fe.n_objects), cold)`), `:320` for users; `attr_knn_patch` takes warm from the variant train file (`:365`). Gauge invariance unaffected; untrained non-cold rows would enter the mean.
>
> **6. TRIVIAL — the vocabulary guard compares counts, not identity.** `factories.py:281` checks only equal `n_features`; two literal field lists of equal total size would pass and misalign words under sharing. Unreachable with the 'canonical' sentinel.
>
> **7. TRIVIAL — knobs are asserted off instead of passed through** (`max_norm`, dc06 knobs; `bias=False` hard-coded where the sketch had `use_bias=False`). Documented in the assert messages; no effect (the host space searches none of them).
>
> **8. TRIVIAL — initialisation prose and order.** The prose says `RecSys.init_parameters` reaches the prototypes' `init_parameters`; the code reaches only the linear layers (matches the spec's own class sketch; `PrototypeEmbedding.init_parameters` is a no-op with `use_weight_matrix` off). The factory initialises the tables before building the prototypes; only the RNG stream order differs; the keystone is defined by weight copy.
>
> **No divergence found:** score and wiring (`[sims(K_u), lin(K_t)]` / `[lin(K_u), sims(K_t)]`, `rec_sys.py:121` = Eq. 12, `use_bias=0`); one q per side for both roles; LOO training-only and reaching both user roles; loss (L_rec + both prototype accumulators); factory (no config mutation, single-owner init, dc08 rows asserted 0); injection (own `data_path`, shallow copies); configs (host space mirrored, `share_word_table` never searched, four arm names); cold path (finders reach `.base`, both `config_expects_*` register the ft_type, item imputation at `n_features + i`, user at `id_row_offset`, empty baskets zeroed, imputation default); registration.
>
> **(a) Spec requirements not found in the named files:** 1. `normalize_composed` / `_norm` (finding 1). 2. Breakdown slots `compute_breakdown_feature_user_item_proto` / `compute_breakdown_user_item_proto`, `extra_baselines`, renderer self-check (§3.4; `breakdown.py`, SC.5). 3. Accessor helpers for read-outs 3–5, per-component coverage (6), the canonical gauge (SC.5). 4. `lightfm_tags_hist` (SC.6 decision). 5. fI-on-UI / fU-on-UI ladder configs (blocked by finding 2). 6. A16 popularity-sampled-negatives secondary read-out (configs inherit `eval_neg_strategy: 'uniform'`; may live in tester/eval code). 7. The one-off measurements F-DC09-09/10/11 and keystone tests t01–t10 (outside scope by rule).
>
> **(b) Code behaviour the spec does not cover:** 1. Packed-table aliasing — under sharing, `item_feature_extractor.parameters()` contains every user-ID row and vice versa; `state_dict` holds four aliases of one tensor; per-side parameter accounting (§4.1) double-counts if it walks one extractor's parameters; 3b-x's table-relative norm instruments would be dominated by ID rows (87,069 of 87,495 on hm) — instruments must slice `[:n_features]`. 2. The imputation "warm" definition (finding 5).
>
> **(c) Undecidable from these files:** 1. Init scale of the packed table vs `_sep`/host tables — if `general_weight_init` is shape-dependent (e.g. Xavier) the shared arm gets a different init std, a second variable in the shared-vs-`_sep` arm (decided by `utilities/utils.py:general_weight_init`). 2. Optimiser grouping with the aliased parameter (fine with deduplicated `model.parameters()`; decided by `rec_sys/trainer.py`). 3. Downstream ID-row addressing — any explanation/naming code addressing user ID rows as `n_features + u` would read item ID rows in ids-shared (decided by `breakdown.py`, `naming/`). 4. Device handling in the accessor (`_batched` builds CPU indices; a CUDA model would mix devices in `FeatureEmbedding.forward`; decided by `accessor/base.py` and the pipeline). 5. Eval mode during validation (without `model.eval()`, LOO would subtract non-basket words on both user roles; decided by `trainer.py`). 6. Untrained rows in the cold variants (decided by the variant train file vs `cold_items.csv`). 7. Whether t01/t07 exist and pass against this exact factory (excluded by the review rules).

#### Response (first-party, every point checked against the code or the data)

| # | verdict | evidence / action |
|---|---|---|
| 1 `normalize_composed` unbuilt, silently ignored | **concede** | True: not in the class, the factory never reads it. Not in the 2×2 run plan (user decision 2026-10-26), so not built; the silent-ignore hazard is real → **trivial fix applied**: the factory rejects the key (all three dicts), pinned by new t11. Design-doc note owed (§3.7 / §6.3: the control is unbuilt). |
| 2 vocab-equality assert blocks the single-side ladder rungs, even in `_sep` | **concede (scope fact, no change)** | True (user `feature_fields=[]` → 0 ≠ 426). The rungs (fI-on-UI / fU-on-UI) are outside the 2×2 run plan; building them would need the equality relaxed for `_sep` plus F=0 tests. Design-doc note owed next to the ladder. |
| 3 accessor exposes raw word rows in ids arms | **agree — the open SC.5 half of F-DC09-07** | Documented deferral. Exposure so far: one 3-epoch smoke checkpoint's naming output (never landed); `run_combo` auto-explanations are off (`EXPLAINABLE_MODELS` empty), so no fleet checkpoint gets ids-arm read-outs before someone runs the pipeline by hand. Open risk until SC.5; a raw-gauge stamp on pipeline output for ids arms is proposed at this gate. |
| 4 packed `[words; item IDs; user IDs]` table instead of `word_table=` | **agree — documented deviation (G1)** | Offsets re-derived by the reviewer match t08's real-data layout (87,495 rows on hm, user IDs from 14,077). Its open condition (c)1 resolved: `general_weight_init` draws `nn.Embedding` rows `normal_(0, 1)` regardless of shape, so shared and `_sep` start from the same per-row law. |
| 5 "warm" = every non-cold ID, not "trained on the variant" | **concede the spec gap; document-only** | Measured: 3 of 10,921 non-cold items on hm_1_month_cold never appear in its train file, 0 of 2,500 on ml-1m_cold; no cold-user variant exists. Invariance unaffected (the gauge shifts every ID row). Spec gap closed by stating the definition and these counts in §3.4; no code change. |
| 6 vocabulary guard compares counts only | **concede** | Unreachable through the canonical sentinel, cheap to close → **trivial fix applied**: `feature_fields` and `feature_layout` must match when both sides carry them (t11). |
| 7 knobs asserted off | **agree, no action** | Documented in the assert messages; none is in the host search space. |
| 8 init prose / order | **concede the prose** | `RecSys.init_parameters` reaches only the linear layers (the prototypes initialise at construction; `PrototypeEmbedding.init_parameters` is a no-op with the weight matrix off). The design-doc sentence is wrong → dated correction owed. Init order: RNG-stream only; keystones are weight-copy based. |
| (a)2–3 breakdown slots, read-out helpers, canonical gauge | **agree — SC.5** | As scheduled. |
| (a)4 `lightfm_tags_hist` | **agree — SC.6 decision** | As scheduled. |
| (a)6 A16 popularity-negative read-out | **resolved** | Lives outside the configs: `Master/scripts/popneg_readout.py` (D3 harness). |
| (b)1 packed-table aliasing | **agree — instrument rule** | `parameters()` of one extractor contains the other side's ID rows; `state_dict` has four aliases (t09 round-trips them). Rule for SC.6 instruments: count the shared table once; word-table norms slice `[:n_features]` (87,069 of 87,495 hm rows are ID rows). Design-doc note owed (§4.1 / 3b-x). |
| (c)2 optimiser with the aliased table | **resolved, clean** | `trainer.py:91/93` build Adam/Adagrad on `model.parameters()` (deduplicated): one parameter, weight decay once. |
| (c)3 ID-row addressing downstream | **resolved, clean** | Item IDs at `n_features + i` (`feature_readout.py:103`, `sc8_geometry_readout.py`) — correct in both layouts (items first); user IDs via `id_row_offset` (`history_readout.py:46,101`). |
| (c)4 accessor device | **resolved, clean** | The explanation loader keeps the model on CPU (`loader.py:60`, `map_location="cpu"`). |
| (c)5 eval mode in validation | **resolved, clean** | `trainer.val()` calls `model.eval()` (`trainer.py:234`); LOO never sees a validation target. |
| (c)6 untrained rows in the variants | **resolved** | See 5. |
| (c)7 keystones | **resolved** | t01 (F=0 ≡ UI host, bitwise) and t07 (sharing keystone) exist and pass (SC.3 §2; re-run after the SC.4 fixes: t01–t07, t09–t11 green). |

### 2. First-party audit

**Optimisation parity (measured 2026-10-27, login-node A30, lone trial; `Master/temp/dc_checks/dc09/sc4_epoch_bench.{py,sh}`, `sc4_profile.py`, evidence `sc4/profile.json`).** One matched config for every model (d = 64, K = 50 per prototype side, B = 256, 50 popular negatives, sampled softmax, Adam). Training epoch as the trainer logs it on hm_1_month: UI host 134/135 s, fU host 135/136 s, fI host 134/136 s, fUfI 133/135 s, fUfI `_sep` 136/135 s — identical, because every model waits on the same input pipeline. The profile splits the epoch:

| stage | hm_1_month | ml-1m |
|---|---|---|
| inherited popular-negative sampler, per sample | 264 µs | 121 µs |
| → DataLoader, 1 worker (fleet) / 2 workers (upstream's setting) | 135 s / 67 s per epoch | 78 s / 40 s per epoch |
| GPU step, UI host / fI / fU / fUfI / fUfI `_sep` | 15.2 / 14.2 / 3.6 / 16.5 / 15.9 ms (28 / 27 / 7 / 31 / 30 s per epoch) | 15.0 / 25.8 / 5.1 / 29.1 / 29.5 ms (33 / 57 / 11 / 64 / 65 s per epoch) |
| peak GPU memory, UI host / fUfI | 1,064 / 1,161 MiB | 1,005 / 1,097 MiB |
| trial set-up: train dataset + injection + factory (fUfI) | 0.7 + 0.9 + 0.06 s | 0.2 + 1.9 + 0.01 s |
| validation pass per epoch (99 uniform negatives per user; sampler-bound as well) | 18 s | 0.9 s |

Reading:
- **Data loading is the bottleneck for every model, hosts included.** The sampler is upstream's, unchanged apart from the cold-set mask: per sample it copies the full popularity vector, raises it to 0.75, renormalises, and calls `np.random.choice(n_items, replace=False, p=p)`. The fleet runs it in 1 DataLoader worker (S0 decision, F-S0-02: with more workers each worker repeats the same negative stream; 0 workers changes the eval regime), where upstream ran 2 workers with `prefetch_factor=5`. The GPU idles ~75–80% of each epoch, which the fleet's 5–8 trials per GPU absorb.
- **fUfI's model cost matches its hosts on hm** (+9% over the UI host per GPU step) and is hidden behind the loader.
- **On ml-1m the bag-layout item composition doubles the GPU step** (fI 1.7×, fUfI 1.9× the UI host): `FeatureEmbedding`'s bag path materialises (B × 51 × 73 × d) before summing, where the user side has a flat `embedding_bag` path (`pooling='bag'`, dc08). Still hidden behind the 1-worker loader (64 s vs 78 s), so no wall-time effect at the fleet's settings.
- **Per-trial set-up is negligible** (≤ 2.6 s against trial walls of minutes to hours); the factory itself is milliseconds.
- **No change proposed for the dc09 runs.** Any sampler change (vectorised or GPU sampling, seeded workers) alters the negatives and breaks equal treatment with the 100-trial fleet; an `embedding_bag` item path would save nothing while the loader is slower. Both are recorded as options for a future tier.

**Design-principle conformance.**
- `FeatureExtractor` contract: `TwoRoleComposedExtractor` subclasses it; `forward(o_idxs, **ext_kwargs)`; `get_and_reset_loss` delegates to the prototype layer; `init_parameters` initialises only its linear layer (prototypes initialise at construction, as on the host).
- Losses through `get_and_reset_loss` only: t01 (loss equal to the UI host at F = 0), t06.
- `(batch, 1+n_neg)` indexed lookups, no full-catalog pass per training step: the GPU step above is within 9% of the host on hm.
- Factory / config / registration conventions: t04; unlike the host's double-tie branch, the configs are never mutated.
- Device handling: payloads are non-persistent buffers that move with `.to()`; 8/8 CUDA smokes (SC.3).
- Checkpoint round-trip: t09 (the shared table is one Parameter after load; payloads rebuilt by injection).

**Test adequacy.** Claims covered: SC.2 toy round D1–D11; SC.3 suite t01–t10; SC.4 t11 (factory guards); s0 t13 (cold convention), s0 t14 (results-folder guard). Keystone F = 0 ≡ UI host present and bitwise (t01). The two spec keystones without a test: the `normalize_composed` no-op (the control is unbuilt and now rejected) and the single-side reductions (not buildable from this ft_type; finding 2).

**Feature entry.** The S0.5 canonical field sets reach the modules end to end through the production seam on both sides, both datasets and both cold variants (t08, SC.3 §2).

### 3. Fixes and findings

Applied as trivial (fix now + log): F-DC09-15 (`normalize_composed` rejected), F-DC09-16 (vocabulary identity, not only counts); both pinned by t11, dc09 suite t01–t07, t09–t11 green; modifications log updated. Proposed at the gate: F-DC09-17 (design-doc corrections bundle), F-DC09-18 (bag-path GPU cost on ml-1m, wontfix with reasons). F-DC09-07's SC.5 half stays the open explanation risk.

### 4. Gate — CLOSED 2026-10-27

Decisions asked: (1) F-DC09-17 design-doc corrections; (2) F-DC09-18 wontfix for the dc09 runs; (3) G6 — run SC.6 and submit the 2×2 × {hm_1_month, ml-1m} before SC.5, doing SC.5 during the queue wait. User: **"yes i agree"** (all three). Applied the same day: the four dated amendments in the design doc, ledger statuses (17 fixed, 18 wontfix; 15/16 fixed at SC.4).

**Declared step-order deviation (G6):** SC.6 precedes SC.5. Code frozen at submission; SC.5 touches only explanation code (read-outs, canonical gauge, breakdown slots). If SC.5 finds a defect that needs a model change, the affected runs are repeated. Submission from the fleet checkout at `c0c6ac1` while wave 2 holds the lock: it trains the dc09 configs exactly as HEAD does — the SC.4 commits add only factory guards that valid configs pass (t11) and documentation; stated in the SC.6 run plan.

Next: SC.6 — cluster-run plan.
