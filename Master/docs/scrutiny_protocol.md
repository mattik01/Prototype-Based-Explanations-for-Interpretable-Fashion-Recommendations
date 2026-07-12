# Scrutiny Protocol

> v1.2, 2026-07-12.
> Executed via the `/scrutiny` skill. The skill is a thin wrapper; **this
> document is the source of truth** for the procedure.
>
> Purpose: ruthlessly audit **and repair** (a) the shared evaluation/comparison
> foundation and (b) each design candidate's concept, implementation,
> explanations, and cluster-run readiness — so that every candidate reaches a
> fairly-comparable, tested state and every claim that survives scrutiny is one
> the thesis can stand on. Sibling of the Design Candidate Protocol
> (`design_candidate_protocol.md`): that protocol *generates* candidates on
> paper; this one *hardens* them into evidence.
>
> Shape: one global foundation pass (**S0**, runs once), then one per-candidate
> routine (**SC(dcNN)**), in the fixed order **dc01 → dc02 → dc03 → dc04**.
> Every step ends at a user review gate.

## Relationship to the Design Candidate Protocol

This protocol **inherits** the following ground rules from
`design_candidate_protocol.md` verbatim (they are not restated in full):
one step at a time / re-read before each step (DC GR1–2), artifact before
progress (DC GR3), no test-shopping (DC GR5), no from-memory citations
(DC GR6), no cross-candidate ranking (DC GR7), "the paper says" vs "I propose"
(DC GR8), spillover capture to the design scratchpad (DC GR9).

Where the two protocols differ: the DC protocol forbids implementation; this
protocol **exists to touch code** — under the fix policy below.

## Ground rules (apply to every step)

1. **One step at a time; artifact before progress.** Before starting a step,
   re-read that step's section of this document (mandatory Read call). Each
   step ends by appending its section to the dossier and stopping at the user
   gate. A step without a written artifact did not happen.
   **Gate mechanics for fix-bearing steps:** the gate is the
   *proposal-bundle decision point* — findings + proposed fixes are presented
   and decided there; approved fixes are applied *after* the gate, and the
   step's artifact (incl. fix commits) is finalized before the **next** step
   begins. "Artifact before progress" means before the next step, not before
   the gate.
2. **Steel-man rule.** Always scrutinize the best current version. If a
   downstream step uncovers an upstream flaw (an implementation scrutiny finds
   a concept error; an explanation scrutiny finds an implementation error),
   pause the step, fix upstream (gated), then resume downstream against the
   corrected state. Never file findings against a state already known broken.
3. **Fix policy.** *Trivial* fixes are applied immediately and logged.
   Trivial = doc typos, dead/unused flags, or code diffs with **no behavior
   change to the scored path** that are covered by an existing test. Everything
   else becomes a finding **bundled with its proposed fix** (finding →
   severity → proposal in one block), decided at one gate, then applied.
   No finding is presented fix-less unless genuinely open.
4. **Single code lineage.** All scrutiny, fixes, S0-build, and dc03/dc04
   implementation happen on **one integration branch**: `feat/scrutiny`
   (forked from `feat/attr-item-proto`, the superset containing dc01+dc02
   code). No per-candidate fix branches — dc01's shared-infra fixes must be
   the same code dc02 is scrutinized on. Merge to `dev` at protocol
   completion.
5. **No cross-candidate ranking** (inherited DC GR7). Verdicts use
   requirement-anchored **absolute** phrasing ("dc02 rates partial on R6
   because …"), never comparatives ("better than dc01"). Applies to dossiers,
   the findings ledger, and the SC.10 verdicts. Cross-candidate judgement is
   the user's.
6. **No test-shopping / one-rerun rule.** A claim falsified stays falsified
   this pass. An anomalous cluster run gets **at most one** documented rerun
   round (SC.8), with the anomaly and the rerun rationale written down before
   the rerun is submitted.
7. **Algorithmic-novelty attribution gate** (explanations). For every new or
   altered explanation artifact a candidate ships, ask: *"could original
   ProtoMF have produced this exact artifact?"* If **yes** → it is shared-
   pipeline work: backport it to the baseline models so the comparison stays
   apples-to-apples, and do not present it as candidate novelty. Only if
   **no** — the artifact requires the candidate's algorithmic change — is it
   novelty. Every backport triggers a **retro-invalidation sweep**: list the
   comparability matrices (SC.5) of previously-scrutinized candidates and mark
   each still-valid or stale (stale ones get a re-run of SC.5's matrix only).
8. **Learnings capture.** Any step may append to `scrutiny/learnings.md`:
   watch-outs, recurring error patterns, checks that proved high-yield.
   *Design ideas* (new mechanisms, candidate variants) do NOT go there — they
   route to `design_candidates/scratchpad.md` as `[captured scrutiny dcNN]`
   entries (DC GR9 convention). Learnings are compacted at every flush step
   (merge duplicates, drop candidate-specific detail once generalized); hard
   cap ~one screen. The next candidate's SC.1a loads them as mandatory input.
9. **Protocol versioning.** Self-evaluation proposals (SC.7/SC.11) that the
   user accepts are edited into this document with a dated changelog entry and
   an **"applies from dcNN onward"** note. A candidate scrutinized under an
   older version keeps that version noted in its dossier header — asymmetries
   are declared, never hidden.
10. **Adversarial independence.** Subagent passes (SC.1b, SC.2 re-check,
    SC.4 spec-vs-code, SC.10 cold read) receive **only** the artifacts named in
    their step — never this conversation or the scrutinizing session's
    reasoning.
11. **Modifications-log boundary.** Fixes to files that exist in the original
    upstream repo are logged in `Master/docs/modifications_log.md` as usual
    (file-centric *what*), with a one-line cross-reference to the finding ID.
    The findings ledger owns the *why* (finding, severity, disposition,
    resolving commit). Rationale is written once, in the ledger.

## Files & artifacts

- **This protocol:** `Master/docs/scrutiny_protocol.md`.
- **Dossiers:** `Master/docs/scrutiny/s0_foundation.md` and
  `Master/docs/scrutiny/scNN_<slug>.md` (one per candidate, e.g.
  `sc01_feature_item_proto.md`) — one appended section per step.
  Dossier header records the protocol version in force.
- **Findings ledger:** `Master/docs/scrutiny/findings_ledger.md` — every
  non-trivial finding. Schema:
  - **ID:** `F-S0-NN` (foundation) / `F-DC0N-NN` (candidate-scoped; IDs use
    uppercase `DC` by convention, unlike the lowercase `dc0N` candidate
    names).
  - **applies-to:** `shared` | `dc0N` — `shared` findings fan out: each later
    candidate's SC.1a re-checks them against current code.
  - **severity:** `trivial` / `minor` / `major` / `blocker`.
  - **status:** `open` → `fixed` / `wontfix` / `superseded` / `stale`
    (+ resolving commit hash when fixed).
  - Trivial fixes get a one-line ledger entry (ID + what + commit), no
    proposal block.
- **Learnings ledger:** `Master/docs/scrutiny/learnings.md` — watch-outs and
  patterns only (Ground rule 8); compacted at every flush; capped.
- **The comparison-fairness charter** lives as a section of
  `s0_foundation.md` (written in S0.6) and is referenced, not copied, by every
  SC.6 run plan.
- **Candidate index:** `Master/docs/design_candidates/candidate_index.md` —
  status column updated at SC.2 (concept amended), SC.5 (hardened), SC.11
  (scrutiny complete / verdict recorded).

## Context-loading manifest

Load-at-start lists per session type. Paths are repo-relative. The manifest is
the *floor*, not the ceiling — steps may read more.

**Every scrutiny session (any step):**
- `Master/docs/scrutiny_protocol.md` (this file — the current step's section,
  re-read)
- `Master/docs/scrutiny/findings_ledger.md` — re-check `open` findings that
  apply to this session's scope
- `Master/docs/scrutiny/learnings.md`
- `Master/docs/feature_aware_solution_requirements.md` (R1–R8 / S1–S5 — the
  criteria spine; living document, never from memory)
- `Master/docs/design_candidates/candidate_index.md`
- For SC sessions: the candidate's design doc
  (`Master/docs/design_candidates/dcNN_*.md`) and its dossier.

**S0 sessions additionally:**
- Docs: `Master/docs/masterplan.md`, `Master/docs/replication_report.md`,
  `Master/docs/hm_feature_analysis.md`, `Master/docs/hm_dataset_notes.md`
- Code: `utilities/eval.py`, `utilities/consts.py`,
  `rec_sys/protomf_dataset.py`, `data/hm/hm_splitter.py`,
  `confs/hyper_params.py`, `feature_extraction/feature_ids.py`,
  `data/hm/price_band.py`
- Protocol vault (eval + feature themes), under `Master/protocol_vault/`:
  `2026-06-16_1631_protomf-item-ranking-basecase.md`,
  `2026-06-16_1704_eval-design-citable-justification.md`,
  `2026-06-07_2309_eval-scope-lightweight-map12.md`,
  `2026-06-07_2311_why-hm-sparse-feature-rich.md`,
  `2026-06-11_0843_hm-feature-analysis-dataset-suitability.md`,
  `2026-06-16_1824_feature-aware-baselines.md`,
  `2026-06-16_1807_attribute-source-positioning.md`

**SC.1a/1b (concept scrutiny) additionally:**
- The candidate's `Master/temp/dc_checks/dcNN/claims_spec.md` + check
  scripts/results
- `Master/temp/supervisor_progress_presentation.md` (current framing + honesty
  caveats)
- Protocol vault (interpretability theme):
  `2026-06-16_1808_intrinsic-over-posthoc-principle.md`,
  `2026-06-11_1220_posthoc-naming-vs-intrinsic-model-distinction.md`,
  `2026-06-16_2102_cosine-readout-caveat-steck.md`,
  `2026-06-16_1715_feature-injection-design-fork.md`
- dc01 only: `Master/temp/dc01_feature_composed_bias_gap.md`

**SC.3/SC.4 (implementation) additionally:**
- The candidate's implementation plan
  (`Master/temp/dc01_feature_item_proto_implementation_plan.md` /
  `Master/temp/dc02_attr_item_proto_implementation_plan.md`; dc03/dc04: none —
  SC.3 writes one)
- `Master/docs/modification_map.md` (advisory — verify against code)
- The actual model/factory/config/read-out code and the
  `Master/temp/dc_checks/dcNN/` test suite

**SC.5 (explanations) additionally:**
- `utilities/explanations/` (pipeline, loader, accessor/, naming/,
  explainers/), `utilities/explanations_utils.py` (the original-paper
  baseline), the candidate's read-out module(s)

**SC.6 / SC.8 (runs) additionally:**
- `Master/protocol_vault/2026-06-07_2310_compute-feasibility-envelope.md`
- The S0.6 charter section of `s0_foundation.md`
- `Master/docs/replication_report.md` numbers + S0.7 reference-run results
- Skills: `/leo5-load`, `/leo5-submit`

---

# S0 — Foundation scrutiny (runs once, before any candidate)

S0 audits and (via S0-build) repairs the shared measurement apparatus. The
principle: **candidates must be judged by an instrument that has itself been
scrutinized.** Throughout, "the two facets" means the thesis's two
contribution axes: **(A) intrinsic prototype interpretability via feature
grounding, (B) sparsity/cold-start robustness** (see the requirements doc
§2). Step order is deliberate — the side question (S0.2) shapes the
cold-start design (S0.3); the feature audit (S0.5) feeds the charter (S0.6).

## S0.1 — Eval protocol audit

**Goal:** know exactly what the current eval measures, and decide what changes.

1. Audit the live protocol against the ProtoMF paper's: leave-one-out
   temporal split, 1 positive vs `NEG_VAL=99` uniform negatives,
   HR/NDCG@{1,3,5,10,50}, `hit_ratio@10` selection, k-core, dedup (repeat
   purchases structurally invisible — discovery-only framing).
2. Assess: sampled-negatives caveat (uniform, popularity-blind); metric-set
   sufficiency for the thesis's two facets; seeds policy for scrutiny-phase
   runs (single seed vs `SEED_LIST`) — decide and record.
3. **"Gaps we will NOT fix" declaration (mandatory):** every recognized gap
   that stays open is listed with a one-line reason. Strong defaults:
   headline eval stays uniform-99 for replication comparability
   (popularity-sampled negatives at most a cheap secondary readout);
   beyond-accuracy metrics only if effectively free.
4. **Artifact (`## S0.1 Eval protocol audit`):** current-state summary,
   findings+proposals, the not-fixing list.

## S0.2 — Item-side vs user-side: fundamental reasoning

**Goal:** a written argument for *where* improvements can honestly be expected.

1. All four candidates are item-side-only (user branch = free CF everywhere).
   Reason through, for each facet: **interpretability** — which explanation
   surfaces become grounded (item prototypes) and which remain post-hoc (user
   prototypes, projection branches), and what that means for thesis claims;
   **cold-start** — which scenarios the designs can address (cold *items*) and
   which they structurally cannot (cold users; H&M user features are thin).
2. Disposition of user-side features: documented reasoning, **default
   out-of-scope for this protocol** — promotion would be a new
   design-candidate cycle (S2 in the requirements doc), not scrutiny work.
3. **Artifact (`## S0.2 Side reasoning`):** the argument, the expected-impact
   map (facet × side), the user-side disposition. This section is
   thesis-quotable material — write it accordingly.

## S0.3 — Cold-start evaluation design

**Goal:** a full, implementable spec for a cold-item eval all models run.

1. Define **cold item** under the temporal split (e.g. items first appearing
   after the train cutoff / held-out item set with interactions removed),
   consistent with the dedup + k-core pipeline. State strata (popularity
   tail vs true-cold) if used.
2. **Leakage checklist (mandatory):** attribute availability at scoring time
   (articles.csv is static — fine; anything derived from transactions, e.g.
   price bands, must be computed train-only); no future information in
   features; k-core must not silently delete the cold set; eval negatives
   drawn correctly for cold rankings.
3. Define **how every model scores cold items** — including CF-only baselines
   (a stated convention: e.g. zero/mean embedding, popularity fallback,
   attribute-kNN fallback per dc02's amendment) so the comparison is defined
   for all rows of the results table, not just feature-aware ones. Include
   dc02's full-catalog tie-block diagnostic where applicable. Note dc01's
   bias-channel caveat (`dc01_feature_composed_bias_gap.md`) as an interaction
   this eval must be able to *detect*.
4. Build on prior art in-repo: dc01 implementation plan §3.6 committed cold
   protocol; dc02 §3.6 amendments.
5. **Artifact (`## S0.3 Cold-start eval spec`):** the spec, ready to
   implement in S0-build (module touch-points named: splitter flags, dataset
   class, evaluator, tester).

## S0.4 — CBF-only baseline design

**Goal:** a pure-content baseline that isolates "features help" from
"prototypes help."

1. Design a content-based-filtering-only baseline (LightFM-inspired
   feature-factor model and/or attribute-kNN — pick the minimal set that
   serves the isolation argument; justify count). No prototype machinery.
2. Specify at implementation concreteness: where it slots
   (`feature_extraction/` extractor + factory entry vs standalone scorer),
   config/`ft_type` name, training/eval parity with the rest of the fleet
   (same splits, same negatives, same metrics, cold-start eval included).
3. **Artifact (`## S0.4 CBF baseline spec`).**

## S0.5 — Feature audit

**Goal:** the features are right, enter correctly, and are the same where
fairness demands it.

1. **Entry correctness:** audit `feature_ids.py` (`build_feature_ids`,
   `build_attr_multi_hot`, `inject_feature_ids`) — offsets, vocab
   construction, NaN policy, the `item_id` coverage guard, serialization
   boundary (tensor never in `config.json`).
2. **Parity decision:** dc01 configs use 5 fields, dc02 uses 9 — decide the
   canonical field set (or a justified per-candidate deviation) for all
   scrutiny-phase runs. Decide `price_band` (built but unwired) and
   `product_code` (strongest signal, identity-like, deliberately excluded)
   in writing.
3. **Recompute dc02's signature-collision rate under the chosen field set**
   (one groupby over `item_features.csv`) — the 68% tie-cap is a function of
   this decision and dc02's SC.1 must inherit a current number.
4. **Artifact (`## S0.5 Feature audit`):** findings+proposals, the canonical
   field set, the collision numbers.

## S0.6 — Comparison-fairness charter

**Goal:** the fixed contract every candidate's run plan must satisfy.

1. Write the charter: canonical feature set (from S0.5); eval protocol incl.
   cold-start eval (from S0.1/S0.3); dev-config budget definition (trials ×
   epochs — dev profile, NOT the paper's 100/100); seed policy; negative
   sampling; datasets (hm_1_month primary; others only with reason);
   **single-code-lineage clause** (Ground rule 4); baseline set (CF-only
   ProtoMF variants + CBF from S0.4 + popularity reference).
2. Name preconditions and pause-filler work: **dc03 go/no-go precondition** —
   the ~25 GB H&M image re-download + subdirectory verification (kick off in
   the background during S0; do not let it block dc01/dc02). **Download
   target: LEO5 scratch (or the GPU desktop), NOT the laptop** (no space; the
   images are only needed where pre-extraction/training runs) — check
   quota/scratch space via `ssh leo5` before starting the pull; kaggle CLI on
   the target machine. dc04's literature-support search (model-knowledge
   provenance) as pause-filler.
3. **Artifact (`## S0.6 Charter`):** the charter. Every SC.6 run plan cites
   compliance clause by clause.

## S0-build — Implement the gated designs

**Goal:** the foundation fixes exist in code before any candidate is judged.

Supervised build — **one gate per landed component** (cold-start evaluator
(S0.3), CBF baseline (S0.4), feature-parity fixes (S0.5), plus any S0.1
accepted proposals), then a closing gate on the combined artifact. Each component lands with tests in the style of
`Master/temp/dc_checks/` (new dir: `Master/temp/dc_checks/s0/`). Ledger
entries get resolving commits.

**Artifact (`## S0-build`):** component list, commits, test results.

## S0.7 — Foundation verification + reference runs

**Goal:** scrutinize the instrument, then freeze the baseline numbers.

1. **Scrutinize S0-build itself** — the same standard candidates get: a
   black-box subagent checks the cold-start evaluator and CBF baseline against
   their S0.3/S0.4 specs (spec-vs-code, Ground rule 10); leakage checklist
   walked against the actual implementation; smoke runs.
2. **Freeze reference runs on LEO5 under the charter:** CF-only ProtoMF
   variants (at minimum `user_item_proto`, `item_proto`, `mf`), the CBF
   baseline, popularity reference — computed **once**, on the integration
   branch, with the cold-start eval active. These are the numbers every
   candidate compares against; no SC.6 re-runs baselines without a charter
   change.
3. **Artifact (`## S0.7 Verification + reference runs`):** subagent report,
   fixes, the frozen reference-results table (+ result-dir paths).

---

# SC(dcNN) — Per-candidate scrutiny routine

Runs per candidate, order dc01 → dc02 → dc03 → dc04. **Precondition: S0 is
complete through S0.7** (charter, foundation build, and reference runs exist)
— a candidate routine started earlier must say so explicitly and re-check
against S0 once it lands. Steps SC.1a–SC.7 are the
pre-run block; the routine then **pauses** for cluster runs; SC.8–SC.11 are the
post-run block. For dc03/dc04 (paper-only), SC.3 means *implement from
scratch*; everything else is identical.

## SC.1a — Concept self-scrutiny

**Goal:** ruthless first-party audit of the corrected-to-date concept.

1. **Preamble (no separate gate):** load the manifest; review
   `learnings.md` (list which watch-outs apply here — for dc01 it may hold
   only S0-era entries or be empty; that is fine); re-check `open` and
   `shared` ledger findings against current code/docs (mark `stale` ones).
2. **Known-issues disposition FIRST (steel-man before adversary):** walk the
   candidate's own recorded weaknesses — its "5b. Devil's advocate" and
   "4b. Math sanity check" sections (labeled sections *inside the candidate's
   design doc*, named after the Design Candidate Protocol steps that produced
   them), and documented gaps (e.g. dc01 bias-channel,
   dc02 tie-cap, dc04 preimage non-identifiability) — and for each: still
   open / resolved / superseded, with reasoning. New findings must not
   re-discover what the file already admits.
3. **The scrutiny proper:**
   - *Mathematical soundness:* derivations, limit cases, invariants.
   - *Expressiveness:* does the model access **more information** than the
     host? Is it mathematically **able to leverage it** (gradient paths,
     capacity)? Where are the capacity ceilings (e.g. signature classes,
     rank constraints), stated quantitatively where cheap?
   - *Degeneration-mode coverage:* is the design-doc's mode list complete?
     Any mode the knobs can't reach?
   - *Requirements delta:* re-rate R1–R8/S1–S5 vs the doc's own claimed
     ratings; justify every delta (too generous AND too harsh both count).
   - *Cold-start mechanism check* — **including the bias channel** and any
     other ID-keyed side channels; does the S0.3 eval detect the failure
     modes this design could have?
   - *Steck cosine-caveat applicability:* does the read-out's semantic
     reading survive per-dimension rescaling arguments? What mitigation
     applies?
4. Findings bundled with proposed fixes (Ground rule 3).
5. **Artifact (`## SC.1a Concept self-scrutiny`):** dispositions, findings
   ledger entries, proposal bundles.

## SC.1b — Adversarial concept pass

**Goal:** an independent read that goes *past* the prior devil's advocate.

1. Spawn a subagent receiving ONLY: the candidate design doc (including its
   Step 5b critique + author responses) and the requirements doc. Brief:
   *"The previous adversary's critique and the author's rebuttals are in the
   file. What did the previous adversary miss? Which rebuttals do not
   actually hold? What breaks at H&M scale?"*
2. Attach the critique verbatim; respond point-by-point (concede / rebut /
   open risk). Fold accepted points into the SC.1a proposal bundles.
3. **Artifact (`## SC.1b Adversarial pass`):** critique + response.

## SC.2 — Concept fixes + re-verification

**Goal:** the corrected concept, verified where the fixes touch math.

1. Apply the gated amendment bundle to the design doc as **dated, visible
   amendments** (never silent rewrites of earlier sections); update the
   candidate index row (status/date).
2. **One black-box toy re-check** (DC Step-4b style, one round only) covering
   every claim the amendments touch. **Mandatory** where prior claims were
   falsified (dc03's CL7/CL8, dc04's CL7a). Subagent gets only the updated
   claims spec; scripts to `Master/temp/dc_checks/dcNN/`.
3. **Artifact (`## SC.2 Concept fixes`):** amendment list, re-check report,
   ledger status updates.

## SC.3 — Implementation alignment

**Goal:** code matches the corrected concept before it is judged.

- **dc01/dc02:** patch the existing implementation to the amended concept
  (supervised; per-fix commits referencing finding IDs; modifications-log
  cross-refs per Ground rule 11). Extend the dc_checks suite where behavior
  changed.
- **dc03/dc04:** verify preconditions first (dc03: image set re-downloaded
  and verified per S0.6 — if not met, STOP this candidate and record it;
  dc04: literature-support status). Then write a condensed implementation
  plan (register in `Master/temp/`), gate it, and implement — following the
  conventions the dc01/dc02 builds established (factory branch, config
  registration, accessor, read-out, dc_checks suite, local smoke via
  `Master/temp/dc_checks/dc02/smoke_train.py`-style harness).
- **Artifact (`## SC.3 Implementation alignment`):** change/build summary,
  commits, test results.

## SC.4 — Implementation scrutiny

**Goal:** the code is faithful, idiomatic, and optimized like the host.

1. **Spec-vs-code, black-box (the high-value independence point):** subagent
   receives ONLY the corrected design doc + the implementation (relevant
   files) — brief: *"where does the code diverge from the spec?"* Attach
   verbatim, respond.
2. First-party audit:
   - *ProtoMF design-principle conformance:* `FeatureExtractor` contract,
     losses via `get_and_reset_loss()`, `(batch, 1+n_neg)` indexed lookups,
     no full-catalog pass per step, factory/config/registration conventions,
     device handling, checkpoint round-trip.
   - *Optimization parity:* vectorization and memory behavior at the standard
     the original codebase sets (no per-item Python loops on hot paths,
     buffers vs parameters, precompute/cache where the design doc promised).
   - *Test adequacy:* dc_checks coverage vs the claims spec; keystone
     equivalence tests (e.g. F=0 reductions) present and green.
   - *Feature entry (per-candidate):* the candidate consumes the S0.5
     canonical field set correctly end-to-end (splitter → builder → module).
3. → gated fixes (bundled), applied before SC.5.
4. **Artifact (`## SC.4 Implementation scrutiny`):** subagent report +
   response, audit findings, fix commits.

## SC.5 — Explanation scrutiny

**Goal:** maximally comparable to the original pipeline; every claimed novelty
traceable to the algorithm; explanation **quality** demonstrated from the end
user's perspective, not just mechanism asserted.

**Why this step carries thesis weight (user directive, 2026-07-12):** the
artifacts built here are the thesis's **standing comparison instrument**, not
per-candidate decoration. At every lineage stage the variant is compared to
its host (fI vs I-ProtoMF now; later fU vs U-ProtoMF; fUfI vs UI-ProtoMF at
the merge) by the same two questions, asked of BOTH models: (1) *what do the
prototypes look like* — post-hoc interpretation on the host vs intrinsic
grounding on the variant; (2) *what does a concrete recommendation look like,
broken down through those prototypes* — the shared breakdown renderer, same
user/item, side by side. Both are answered in part **graphically**, across
the datasets in scope (exact dataset list per variant pinned at SC.5/SC.6).
Because these graphics recur at every stage × dataset, they are designed for
**repeated use**: refined to be as insightful as possible AND compact enough
to appear many times without bloating the thesis.

1. **Comparability matrix:** rows = every explanation artifact in the
   pipeline (naming routes, tsne, top-k, weight-viz, small-multiples,
   read-outs, the recommendation breakdown of step 4); columns = original
   ProtoMF models vs this candidate. Every
   absent cell gets a written justification (e.g. dc02's weight_viz exclusion
   and its §3.4 rationale). Unjustified gaps are findings.
2. **Algorithmic-novelty attribution gate** (Ground rule 7) per new/altered
   artifact: could the original have produced it? yes → backport +
   retro-invalidation sweep; no → record *which algorithmic property* enables
   it (this sentence is thesis material).
3. **New-interpretability-axis identification:** name the candidate's
   genuinely-new axis (e.g. intrinsic naming, exact shares, exemplar
   identity) and propose the visualization that shows it — one paragraph +
   sketch; **implement only if it will feed the chapter** (SC.9), else it
   stays a proposal.
4. **User-perspective recommendation breakdown (mandatory build).** The
   thesis aims for *better explanations*, not merely intrinsically-grounded
   ones — this step builds the artifact that demonstrates it. One **shared**
   breakdown renderer in `utilities/explanations/`: given (user, recommended
   item, model), produce a self-contained "why did I get this
   recommendation?" artifact — text + figure, thesis-figure quality,
   non-interactive — for **every** model in the comparison, baselines
   included.
   **Fairness contract — maximum shared principle, advantages from the
   algorithm only:** rendering, format, and presentation effort are identical
   across models; each model plugs its own mechanism into the inherently
   model-specific slots (base ProtoMF: post-hoc prototype naming;
   feature-grounded candidates: intrinsic read-outs). Inherent mechanism
   asymmetries are *kept and declared* — they are the object of comparison —
   but no quality difference may stem from unequal engineering effort.
   The baseline's post-hoc naming route is **shared-pipeline work under
   Ground rule 7**: steel-man it to the best of our ability (backport +
   retro-invalidation sweep apply). Built here pre-run; rendered on real
   checkpoints at SC.8.4; the side-by-side is the SC.9 showcase.
5. **Gating-consistency check:** explainer `supports()` sets vs accessor
   `has_*` flags vs `EXPLAINABLE_MODELS` — consistent for this candidate, no
   silent drops.
6. No adversarial subagent here (structural, pre-training — low yield).
7. → gated fixes; candidate index status → hardened.
8. **Artifact (`## SC.5 Explanation scrutiny`):** matrix, gate verdicts, axis
   + viz proposal, breakdown-renderer build summary (commits, per-model slot
   map), fixes.

## SC.6 — Cluster-run plan

**Goal:** the exact runs that take this candidate to fairly-comparable.

1. Run manifest under the S0.6 charter (cite compliance clause by clause):
   datasets (hm_1_month primary), dev-config budget, seeds, the candidate's
   config(s) + **minimal isolating ablations only** (e.g. dc01
   `use_id_feature` fork and F=0 keystone; dc02 knob-off base; per-candidate
   as the design doc's baseline gate demands), cold-start eval runs.
   Baselines come from S0.7's frozen reference runs — not re-run.
2. Compute estimate vs the envelope (>1 day soft, >10 days hard); flags with
   mitigations, not vetoes.
3. Output ready for `/leo5-submit`.
4. **Artifact (`## SC.6 Run plan`).**

## SC.7 — Pre-pause flush & retrospective → PAUSE

**Goal:** the next candidate inherits everything learned so far, now.

1. Flush concept/implementation/explanation learnings to `learnings.md`
   (compact per Ground rule 8) — do NOT wait for this candidate's runs.
2. Protocol self-evaluation, timeboxed: did any step misfire, miss, or add no
   value? Max ~3 proposals; accepted ones edited per Ground rule 9.
3. **PAUSE.** Submit/await cluster runs. The next candidate's routine may
   start during the pause. **Resume semantics:** when runs finish while
   another candidate's step is in progress, finish the current gated step
   first; every resume (`/scrutiny resume dcNN`) re-runs the context load
   (manifest + open-findings re-check).
4. **Artifact (`## SC.7 Flush & retrospective`):** learnings delta, protocol
   proposals + decisions, run-submission record.

## SC.8 — Results verification (post-runs)

**Goal:** numbers are trusted before they become prose.

1. **Charter-compliance audit of the actual runs:** configs as manifested
   (feature set, negatives, seeds, budget, branch/commit)? Deviations are
   findings.
2. **Anomaly triage:** diverged trials, NaN, suspicious patience exits,
   hardware faults — at most one documented rerun round (Ground rule 6).
3. Seed-variance check (if multi-seed); cold-start metric extraction against
   the S0.3 spec; comparison rows against S0.7 references assembled.
4. **Rendered-explanation spot-check on real checkpoints:** run the
   explanations pipeline, eyeball N real explanations (N ≥ 5 prototypes +
   ≥ 3 per-recommendation read-outs, **plus ≥ 3 shared-renderer
   recommendation breakdowns (SC.5.4) rendered side-by-side for the candidate
   and at least one CF-only reference model — same user, same recommended
   item**) — this is where Steck/R7 concerns
   actually manifest; record observations.
5. **Evidence-backed R1–R8/S1–S5 re-rating table** — ratings now cite run
   evidence, not design intent.
6. **Artifact (`## SC.8 Results verification`).**

## SC.9 — Chapter draft (results-memo)

**Goal:** the candidate laid out coherently, as if a mini-paper.

1. Draft, capped ~3–5 pages, non-verbose: mechanism (self-contained),
   motivation against the two facets, relation to the host, experimental
   setup (charter reference), results incl. cold-start, explanation showcase
   (the SC.5 axis, with a real rendered example, **plus the side-by-side
   user-perspective breakdown — candidate vs baseline, same user/item, from
   the shared renderer**), limitations & honest
   caveats (Steck, dev-profile provisionality, known open findings).
2. Full thesis prose comes later, only for candidates the user selects —
   this is a results-memo, not a chapter.
3. Store as `Master/docs/scrutiny/scNN_chapter_draft.md`.
4. **Artifact:** the draft itself + a dossier pointer.

## SC.10 — Cold-read critique

**Goal:** does the candidate survive being read as a whole?

1. Spawn a **genuinely context-free** subagent: receives ONLY the chapter
   draft + the requirements doc. Brief: *"Is this sound? Are the claims
   supported by the evidence presented? Could this enter a master's thesis?
   What is missing or overclaimed?"*
2. Attach verbatim; respond; fix the draft where conceded.
3. Verdict recorded **dev-profile-provisional**, absolute phrasing only
   (Ground rule 5).
4. **Artifact (`## SC.10 Cold read`):** critique + response + verdict.

## SC.11 — Final flush

**Goal:** close the candidate's scrutiny loop.

1. Flush drafting/results learnings; compact.
2. Protocol self-evaluation (as SC.7.2).
3. Candidate index: status → `scrutinized (dev-profile)` + date; dossier
   closed with a 5-line summary block.
4. **Stop.** Present summary; the user decides the next move (next
   candidate's post-run block, next candidate's start, or protocol exit).

---

## Out of scope (for this protocol)

- **User-side features** — documented in S0.2, deferred; promotion = a new
  design-candidate cycle.
- **Full-profile runs** (paper's 100 trials / 100 epochs) — all verdicts here
  are dev-profile-provisional; full runs happen after the user selects
  candidates.
- **Beyond-accuracy metric suite** (coverage/diversity/novelty) — unless S0.1
  finds one effectively free.
- **Four polished thesis chapters** — SC.9 is results-memo depth.
- **dc05 synthesis candidate** — explicitly *after* this protocol, seeded by
  the learnings ledger and the design scratchpad captures.

## Session-count expectation

~8 S0 gates + ~12 gates × 4 candidates ≈ **55 gated reviews** (SC.1a/SC.1b
often share a sitting). This is
deliberate (per-step supervision was chosen); multiple steps can share a
sitting when the user is present. The gates are the protocol's spine — do not
batch steps to save sessions (Ground rule 1).

## Re-run mode

`/scrutiny stepX dcNN` (or `stepX s0`) re-runs only step X against the current
state of all inputs (substeps by their full label, e.g. `step1a dc02`,
`step0.3 s0`): re-read this doc's section for X, redo it, update the
dossier section with a **dated edit note**, update ledger/index as applicable.
Used e.g. after a backport's retro-invalidation sweep marks an SC.5 matrix
stale.

## Changelog

- **Strategic re-scoping note (2026-07-12, user decision — not a procedure
  change):** after dc01's SC.5, the fixed candidate order is deviated from:
  dc02→dc03→dc04 scrutiny is **deferred indefinitely**; a directed
  design-candidate cycle for the lineage's user-side stage (fU, candidate id
  dc05) is interposed (`Master/docs/design_candidates/fu_cycle_brief.md`),
  followed by its build and its own SC routine. dc02/03/04 remain registered
  and become licensed cannibalization quarry for that cycle. dc01 resumes at
  SC.6 on cluster progress (S0.7 part 2 first). The "dc05 synthesis"
  reservation in Out-of-scope is renumbered to "dcNN synthesis" (ids are
  registration order). Procedure text above is unchanged.
- **v1.2 (2026-07-12)** — SC.5 gains a "why this step carries thesis weight"
  preamble (user directive at the dc01 gate-resume sitting): the explanation
  artifacts are the thesis's standing host-vs-variant comparison instrument —
  two fixed questions (prototype view: post-hoc vs intrinsic; recommendation
  breakdown through the prototypes), asked at every lineage stage (fI|I,
  fU|U, fUfI|UI), answered partly graphically, across the in-scope datasets;
  graphics refined for insight + compactness under repeated use. Emphasis and
  scope clarification — the v1.1 renderer build is the vehicle, unchanged.
  Applies from dc01 onward.
- **v1.1 (2026-07-12)** — SC.5 gains a mandatory **user-perspective
  recommendation-breakdown build** (step 4): one shared renderer for all
  models, fairness contract "maximum shared principle — advantages from the
  algorithm only", inherent mechanism asymmetries (post-hoc naming vs
  intrinsic read-out) kept and declared; baseline post-hoc naming
  steel-manned as shared work under Ground rule 7. SC.8.4 (spot-check) and
  SC.9.1 (showcase) consume it. Rationale: the thesis aims for *better*
  explanations, not merely intrinsic ones. Applies from dc01 onward (dc01
  had not yet reached SC.5). User-driven refinement.
- **v1.0 (2026-07-11)** — initial protocol. Applies from dc01 onward.
