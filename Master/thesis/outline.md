# Thesis Outline — Living Document

**Created:** 2026-07-13 · **Status:** draft skeleton v1 (pre-supervisor, pre-results)
**LaTeX:** `main.tex` + `chapters/*.tex` (DBIS template, upstream commit 2042a67; original preserved as `template_original.tex`).
**Companions:** `placement_ledger.md` (protocol-entry → section placements) · `supervisor_questions.md` (SQ-N register: contested decisions parked for the supervisor — standing procedure, never force a hard call).

Sections carry **stable slugs** (in each chapter file's header and `\label`s). Skills and the ledger reference slugs, never chapter numbers — so reordering/renumbering never orphans a placement. Chapter files keep their historical numeric prefixes (e.g. `04_data.tex` compiles as Ch 3 since the 2+3 merge); compiled numbers come from the `main.tex` input order.

## Design inputs (why this shape)

1. **The committed lineage outline** (`protocol_vault/2026-07-11_2351_thesis-outline-c1-lineage.md`) — presentation order for part one; Interpretability Quantified strictly after.
2. **DBIS norms** (deep-research verified, 2026-07-13): structure set at group/supervisor level; sample-thesis skeleton Intro → Background → Related Work → Dataset → contribution chapters → Evaluation → Conclusion; **bonbon scheme** (every chapter: intro, main part, conclusion); problem statement + outline paragraph at end of Introduction; Related Work = argumentative gap analysis ending with the gap; hedge-free evidence-backed prose, opinion only in Conclusion; Conclusion ≠ summary (answer RQs, compare, limitations, future work). Multi-chapter contributions are the endorsed pattern (Wurzinger exemplar: Dataset/Features/Evaluation as separate chapters).
3. **ProtoMF paper mirror** (soft, per user 2026-07-13): Methodology ordered U→I→UI → our lineage I-side → U-side → merged; Results ordered accuracy → prototype interpretation → explanations. **LightFM mirror** (softer): Datasets as first-class chapter; cold-start protocol anchoring; tags/tags+ids arms.
4. **Deviations from the DBIS exemplar, deliberate:** (a) results live *inside* each contribution chapter (not one shared Evaluation chapter), because the committed narrative demands "present fI fully, demonstrate its explanation, then move on" — the shared, reusable evaluation machinery is factored into the eval-framework chapter instead (SQ-4); (b) Background and Related Work merged into one chapter (2026-07-14, SQ-7) — the RW middle overlapped the teaching material almost 1:1. Both flagged for supervisor conversation.
5. **Caveat from research:** no verified exemplar covers sequencing two heterogeneous blocks (lineage + methodology). The `\part{}` markers in `main.tex` (Part I/II) are prepared but commented out — decide with supervisor.

## Status legend

- 🟢 **ripe** — material rich AND upstream work closed; low expected influx; writable now
- 🟡 **partial** — material exists; upstream still moving or influx expected
- 🔴 **blocked** — waits on results (S0.7 fleet, SC.6+, merge cycle) or undesigned work
- ⚪ **last** — deliberately written at the end (intro, abstract, conclusion)

---

## Ch 1 — Introduction `intro` ⚪

Headline set course-corrected with Matteo (2026-07-13), final shape **five numbered sections**: Motivation → post-hoc-naming problem (host aimed at the same goal, stopped one step short) → attribute-aware prototypes (carries BOTH payoffs — second-Fliege headline cut, content folded in) → interpretability as a model objective (block-two working title, Matteo's phrasing) → Research Questions and Contributions (merged; RQ→contribution→chapter mapping) + unnumbered closing outline paragraph. Prose last; Matteo fills bullets first (bullet-first convention). Prose discipline: Rudin argued ONCE in 1.1, only applied in 1.2; soft budget 4–6 pages. Must address the Kaggle "feature engineering contribution limited" quote head-on. RQ formulation waits for stable results chapters.

**Section statuses after the 2026-07-13 revision walk:** 1.1–1.3 bulleted and course-corrected (done for now); 1.4 FROZEN until Part II progresses; 1.5 deliberately blank — RQs formalized WITH the supervisor (together with the two-part structure), and strategically only after **Part II is tangible**: a solid framework (define interpretable output → identify impacting knobs → how/to what extent they tune) plus proof of concept, so Block II survives the scope conversation despite Block I's size ("higher chance she lets me keep it"). Open meta-question for her: how many RQs may one thesis attempt.

**Open terminology decisions (Ch 1 owns them):**
- **Two lanes pending Matteo's S2/Rudin check:** goal-framing "*transparent* recommendation" (end) *through* "*interpretable* ML" (means) vs. staying on "interpretable" throughout. Working thesis title on the title page follows the same fork (intrinsically-interpretable variant primary, transparent variant commented).
- **attribute vs feature (agreed):** "attribute" = human-readable catalogue property (semantics, prose); "feature" = its encoded model-facing representation (mechanism); code naming untouched. One definitional sentence early. Global prose pass = future dedicated session; until then new text follows the convention. **Model names keep the `f` deliberately** (Matteo's justification, 2026-07-13): the composition mechanism sums arbitrary rows — attribute-derived, ID (not really an attribute), or engineered (e.g. bias-correcting) — "the model does not know and does not care"; *choosing* attribute rows is what makes it interpretable. Mechanism = feature-composed; contribution = attribute-aware.
- 1.3 headline variant open: "Attribute-Aware Prototypes" vs "…: Closing the Naming Gap".

## Ch 2 — Background and Related Work `background` + `related-work` 🟢/🟡

**MERGED 2026-07-14** (Matteo; SQ-7 records the possible split — would be quite hard to untangle; file `02_background_related_work.tex`, both slugs and all old `\label`s stay valid). Teach-then-position along the two axes; chapter intro absorbs the old scope section (axes + modality-light exclusions); chapter ends with **The Gap**. Sections: MF in recommender systems (implicit setting + ranking task inside) → interpretable ML (moved before the host; **subsection headers deliberately TBD** — Matteo decides them from the vault entries; content parked as [TBD] bullets: intrinsic-vs-post-hoc, x→c→y, concept-side positioning, additivity paragraph **[directive: linearity→interpretability chain must become a paragraph]**, family-wide naming sharpener slot, **closing block = prototype landscape as the transition into ProtoMF** — PPM family + recsys neighborhood at survey level only, verdicts reserved for The Gap) → ProtoMF family — PURE TEACHING since 2026-07-14 second restructure (U/I, double-tied model, regularizers — no longer claimed "pure" interpretability penalties, **[PLANNED: small ablation on their accuracy/legibility effects]**; the old closing positioning subsection dissolved: survey half → end of IML section as the transition-into-ProtoMF block, verdict half → The Gap, its sole owner) → feature-aware MF (short positioning pass) + LightFM (taught in depth, + evaluative passage: LightFM's interpretability gains over plain MF — partial, byproduct; grounds ingredients, lacks a read-out) → **The Gap** (axis failure modes → checkable claim → combination-not-ingredient → modality-light `2026-07-13_1817` → hand-off; marked slot for the family-wide naming-problem sharpener). Old eval-explanations RW section relocated to `interpretability-quantified` §taxonomy. Background half ripe (ProtoMF mechanics since Phase 2, positioning entries closed 2026-06-11/16); positioning passes 🟡 — literature influx (S2 re-read, parallel sessions) lands here. Anti-pattern to avoid (examiner-flagged): laundry list without explicit relate-to-ours moves.

## Ch 3 — Data `data` 🟢

Restructured 2026-07-14 (Matteo): dataset sections carry only source/contents/why-selected; every decision-shaped item is preprocessing. H&M (source, contents, why selected; Kaggle handled here fully & once) → ML-1M (source, contents, why: comparability to both seed papers + dense opposite-regime control) → Preprocessing & Splits (both datasets directly: dedup/5-core/LOO + implicitization + honest consequences; H&M V1/V2 versions; cold variants; canonical 5-field set **[directive: product_code exclusion reasoning is section-worthy]**; ml-1m genre bags **[directive: popularity coupling mentioned wherever ml-1m feature results appear]**; sparsity positioning as closing block — dense control + two flavors of sparsity, falsifiable expectations). Three flat sections; sub-headers only at prose time if needed. Feature analysis and splits frozen; small open follow-ups (product_code/colour-value extraction) don't touch the prose skeleton.

## Ch 4 — Evaluation Framework `eval-framework` 🟡

Restructured 2026-07-14 (granularity: headers follow mass; 6 sections → 3): **Computational Setup** first (LEO5/SLURM + laptop; Ray Tune, budgets 40 dev / 100 final matching the ProtoMF paper; W&B + manifests) → **Comparison Design** (blocks: metrics + citable justification + MAP@12 note; C1–C8 charter; reference fleet AND replication together — "same thing with extension", paragraph + table; cold-item protocol block — five deviations + cold 2×2 claim boundary kept explicit, split construction in Data ch) → **The Explanation System** (decided 2026-07-14: shared machinery lives here, demonstrated on the host ProtoMF — deliberately exposing the naming gap empirically; lineage chapters show DELTA ONLY, "focus on what is new", no host re-demos) → **Evaluating Part II [PLACEHOLDER]** (reserved — chapter serves both blocks; filled once Part II experiments are designed). Design frozen (S0.6 charter) — reference *numbers* wait on the S0.7 fleet.

## Ch 5 — Grounding the Item Side: fI-ProtoMF `fi-protomf` 🟡→🔴

Full-build chapter (delta baseline). Headers reworked 2026-07-14 (8→6; why-first and model swapped — motivate before building): why-item-first (taste revealed — load-bearing: fU's opening calls back to it) → model (blocks: construction/keystone/arms, terminology home, four-component inventory) → re-coordinatization (+ committed 3D figure) → read-outs & limits → **results [🔴 S0.7 + SC.6+]** → explanations, delta-only (closing block: route-1 projection lens, bridge to the fU chapter). Theory sections are the richest in the vault (dc01 dossier hardened through SC.5). Reminder: scrutiny numbers = working evidence only; results text from dedicated runs.

## Ch 6 — Grounding the User Side: fU-ProtoMF `fu-protomf` 🟡→🔴

**Delta chapter** (2026-07-14): fI is the full build; here only what differs — surprisingly so — and the new challenges; no mirrors of fI's inventory/why-first/re-coordinatization sections; deliberately shorter than Ch 5. Lineage step 5. Headers reworked 2026-07-14 (6→5): chapter intro carries the information-ordering punchline (`info(U) ⊂ info(fU) ⊆ info(fI)`, no own header) → model (delta focus + drift guard: no composition re-teach; mean normalization, B(t), ids/noid division of labor) → **"Users with Empty Histories: The Road to User-Attribute-Aware Prototypes" [directive: exact & prominent]** — zero-interaction path + the afU road moved here from the merge chapter (2026-07-14, "part of that line"; revises committed outline step 7's home; afU very likely WILL be built — announced in the title, not hedged) → per-purchase attribution (novelty axis) → results [🔴] → explanations (delta-only). Design hardened through SC.5 (2026-07-13); datasets pinned hm_1_month + ml-1m.

## Ch 7 — The Merge: (a)fUfI-ProtoMF `fufi-merge` 🔴

Lineage steps 6–7. PLANNED only — passes its own DC + scrutiny cycle first. **Delta chapter, second order** (2026-07-14): both sides fully built by then — only merge-specific phenomena; thinnest lineage chapter by design, skeleton deliberately least detailed (no candidate, zero results — fills in on contact); don't pad to sibling size. Headers reworked 2026-07-14 (4→3, then 3→2 + intro): chapter intro = the build-order mirror (no own header) → **The Tie** (title provisional — becomes Double/Triple/n-fold Tie once the candidate's tie structure is known) [PLANNED: double tie under grounded factors, which-side-is-free intercept; bullet: one-sentence acknowledgment that the goofy letter-string names ((a)fUfI etc.) are a versioning device/variant descriptor — all of these are really just *Attribute-Aware ProtoMF*] → Results [PLANNED: length-channel refund test; merged-model cold-user empirics if any]. Chapter title carries "(a)" — afU very likely lands before the merge. The afU road lives in the fU chapter's empty-histories section ("The Road to User-Attribute-Aware Prototypes", 2026-07-14).

## Ch 8 — Hidden Effects `hidden-effects` 🟡→🔴

**Structure deliberately unformed** (2026-07-14 walk: left untouched — expect regrouping of effects and discarding of unmeasured rungs once experiments run; current headers = inventory placeholders, don't restructure before measurements). The pedagogical challenge ladder (`2026-07-12_1224`): B(t) intercept **[3 directives: describe/measure/display]** → share identifiability/twins → ID–twins–popularity braid **[user: "absolutely should report"]** → token-mass coupling → length refund → **ids-arm dominance theorem [directive: meticulously checked]** → Steck cosine reliability. Mechanisms derived; measurements pending (login-node instruments + SC runs). Granularity flexible: compresses into Ch 7 (the merge) if thin.

## Ch 9 — Why Quantify Interpretability? `interpretability-objective` 🟡

*(Part II title: **"Towards Interpretability as a Model Objective"** — renamed 2026-07-14 from "Interpretability Quantified"; the block's working title promoted to the `\part{}`, "Towards" as honest hedge. First section renamed "Penalties Tuned Blind" to avoid the echo.)*

Part II split into three chapters (2026-07-14; old slug `interpretability-quantified` retired — ledger rows need a migration pass). **Rashomon scoping (hard decision 2026-07-14): the set is NEVER measured — device/guiding principle only; amends vault `2026-07-12_1412` (agreement-proxy experiment dies, Dacrema stays as citation).** Chapter A = the why: motivation (penalties tuned blind; explained-well collapses into reasons-well) → "Many Good Models" (Rashomon as device, early + headline-level, cited evidence only; tuning = navigating toward legible reasoners, ε = budget) → **accuracy-cost passage [directive: reviewer inoculation; numbers from Ch 11's experiments]**. Chapter intro = Part II roadmap (define output → identify knobs → how far they tune + proof of concept; pre-supervisor milestone). Fallback (Matteo): if too short at prose time, merge into Ch 10.

## Ch 10 — Interpretability Knobs and Metrics `knobs-metrics` 🟡

Chapter B = the what (vocabulary; no routes named yet, no experiments): knob/metric = same property on cause/effect side (diversity trap; surrogate bridge; surrogate-or-not stated per row) → the taxonomy (property→metric→surrogate table; Rudin families as citable spine + soft/hard tag; entry-point classification; four sparsity axes, crispness ≠ feature-sparsity; **per-knob adjustment class, higher abstraction, details TBD: model-structural vs training-adjustable {loss / validation / regularizer}**; **the VOCABULARY dimension (2026-07-14): explanation vocabulary ≈ internal vocabulary in truly interpretable models — constraining it = rarely-discussed interpretability dimension, citation C6 sleep scoring (cite the original Niknazar & Mednick publication, the poster is Matteo's own); doubles as expert-knowledge infusion, can even raise performance; UW-feature-use row = its novel instantiation, novelty highlighted; UW also the worked example of a CLASS-CROSSING knob — loss term OR preprocessing feature selection, multi-class representation TBD; **FIDELITY vs LEGIBILITY as the metric-family tag on every row (vault 2026-07-14_1220): fidelity = descriptor-gap measurements ("is it true?"), legibility = complexity measurements ("is it holdable?"), orthogonal, citable via XAI faithfulness-vs-comprehensibility**). **Role (2026-07-14 second pass): the taxonomy is the LOOKUP TABLE Ch 11's method dispatches on.** Relocated eval-explanations literature (S4/G1) cited here. Open thread: merge Matteo's own knob/metric list.

## Ch 11 — Interpretability Tuning `interpretability-tuning` 🟡→🔴

Chapter C = the how — the whole method, own chapter (Matteo), + **ALL Part II experiments**. Restructured 2026-07-14 (second pass): **method first, routes = its final dispatch step** (not a preceding philosophy); goal = tunably/MORE interpretable models (Part I delivered interpretable ones). Order: the method — explanation-backward tracing (formula → quantify via taxonomy metrics, sparsity first → backtrace every affecting component → per component desired direction + adjustment class from the taxonomy → dispatch) → **"Tunably More Interpretable Models through Two Channels"** (retitled 2026-07-14; "channel" = prose word for route, Route-1/2 stays as vault shorthand: model-structural → Rashomon-window search, Rashomon = one-sentence implicit assumption; differentiable → training loss; validation-type → regularizer and/or Route-1 selection signal; unification stands; principledness spectrum as Route-2 quality-ladder block) → **Demonstrating the Method [mandatory; experiments folded in, no separate title — 2026-07-14]**: movement 1 = the analytic walk (our own score formula; dispatch table as payoff; four sparsity axes + UW row fall out as products); movement 2 = the empirical proof [🔴 not designed: lineage instrumentation, Route-1 knob search, Route-2 studies incl. DPP lift, accuracy-cost numbers, the three honest unknowns as experiment design]. **UW feature use = a bullet in the channels section since 2026-07-14 (no own section — part of the method; dual-natured: loss term or preprocessing feature selection; novelty boundary vs Rudin carried exactly).** Title resonates with SQ-3 ladder C/D.

## Ch 12 — Bias and Adjustment `bias-adjustment` 🔴 CONDITIONAL (95% out of scope)

Matteo 2026-07-13: the grounded/composed models open a *control* avenue (addressable components → per-component bias measurement + targeted correction, incl. bias-correcting rows). Full possibility space = a Part III **or a future project** — deliberately not this thesis. Included as a chapter (with one small experiment) ONLY IF it proves genuinely cheap once Part II tooling exists; otherwise collapses into Conclusion→Future Work. `\part{Bias and Adjustment}` + chapter input exist commented in `main.tex`; file `chapters/11_bias_adjustment.tex`. Decision point: after Part II experiments are designed.

## Ch 13 — Conclusion `conclusion` ⚪

Answer RQs / compare / limitations (incl. the **visible deliberate omission**: bias/fairness parked) / future work (lineage step 8: images under rec loss; interaction-attribute prototypes; afU; dc02 contrast). Not a summary.

## Appendix `appendix`

Hyperparameters, full tables, extra figures, **LLM Usage Disclosure** (from `Master/llm_usage/`, per project policy).

---

## Standing research tasks (feed sections when done)

- **Naming problem across the prototype family** (Matteo, 2026-07-13): deep research whether the naming problem or a sibling appears in other prototype-based interpretable ML (PPMs first — S5 §4.4 "semantically annotated prototypes" is the preliminary signal). If it generalizes → 1.2's complaint becomes family-wide, and Related Work §prototypes gets its sharpest relate-to-ours move. Trigger deliberately (deep-research run or literature session), not en passant.

## Known unknowns / expected outline churn

- Supervisor pass pending (module agreement outranks all of this).
- Ch 7/8 boundary and Ch 8's (hidden effects) chapter-vs-section fate depend on how much measured material lands.
- RQs unformulated (deliberately — after results stabilize).
- `\part{}` split decision open.
- dc02 (lukewarm) could re-enter as an enforced-grounding contrast section if revisited after fUfI.
