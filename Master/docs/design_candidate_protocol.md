# Design Candidate Protocol

> v1.3, 2026-06-16 (v1.0 2026-06-11; v1.1 adds the interpretability-constraint
> audit in Step 3, the degeneration-protection gate in Step 5, and constraint
> claims in Step 4b; v1.2 adds the user scratchpad and its sweep in Steps 0/1/6;
> v1.3 adds Ground rule 9 — spillover capture: adjacent ideas/variants that
> surface mid-cycle are appended to the scratchpad instead of discarded, with a
> completeness flush in Step 6).
> Executed via the `/design-candidate` skill. One invocation =
> one full cycle = exactly **one** design candidate. The skill is a thin wrapper;
> **this document is the source of truth** for the procedure.
>
> Purpose: produce literature-grounded architecture candidates for the
> feature-aware ProtoMF extension (Phase 4), evaluated in depth against
> `Master/docs/feature_aware_solution_requirements.md` — **no implementation**.
> The user reviews each finished candidate before the next cycle runs.

## Ground rules (apply to every step)

1. **One step at a time.** Treat the current step as if it were the entire task.
   Do not look ahead, do not batch steps, do not draft later sections early.
2. **Re-read before each step.** Before starting Step N, re-read Step N's section
   of this document (mandatory Read call), even if read minutes ago.
3. **Artifact before progress.** Each step ends by appending its section to the
   candidate file. A step without a written artifact did not happen.
4. **No implementation, no training.** Toy math checks only, via Step 4b.
5. **No test-shopping.** A claim falsified in Step 4b stays falsified this cycle.
6. **No from-memory citations.** Every claim attributed to a paper is verified
   against the local PDF (or the web) and cited with section/equation/page.
7. **No cross-candidate ranking.** Per-requirement assessment only — LLM
   self-ranking of research ideas is documented as ~coin-flip reliable
   (Si, Yang & Hashimoto 2024). Ranking is the user's job.
8. **Separate "the paper says" from "I propose."** Creative deviations from the
   source papers are welcome but must be flagged as such, always.
9. **Capture spillover, don't discard (v1.3).** Research steps routinely surface
   *adjacent* material — a different variant of the in-progress idea, a seed for a
   future candidate, a cross-paper connection — that R8 (one idea per cycle)
   forbids pursuing now. When such an idea has standalone value, **append it to
   `scratchpad.md` the moment it arises**, as a new entry tagged `[captured dcNN]`
   (so it stays distinct from the user's own entries), rather than dropping it. It
   then competes on equal footing in the next cycle's Step 1 sweep. Keep capture
   cheap — one or two lines, no deep-dive; it must not pull focus from the current
   step. Not every stray thought qualifies: capture only what a future cycle could
   genuinely use, and never edit or delete the user's existing entries.

## Files

- **Requirements:** `Master/docs/feature_aware_solution_requirements.md` (R1–R8,
  S1–S5, §4 tensions, §5 baseline-to-beat).
- **Corpus:** `Master/literature/feature_aware_reading_list.md` + PDFs under
  `Master/literature/papers/`.
- **Candidates:** `Master/docs/design_candidates/dcNN_<slug>.md` (NN = 01, 02, …).
- **Index:** `Master/docs/design_candidates/candidate_index.md` — one row per
  candidate with its fingerprint. Loaded in Step 0, updated in Step 6.
- **Scratchpad (v1.2; capture v1.3):** `Master/docs/design_candidates/scratchpad.md`
  — the user's free-form ideas/extensions/considerations, jotted while reading,
  **plus protocol-captured spillover** (Ground rule 9), tagged `[captured dcNN]`
  to keep it distinct from the user's own entries. Read in Step 0, swept in
  Step 1, back-annotated in Step 6. The protocol may **append** new captured
  entries and disposition markers, but **never edits or deletes the user's entry
  content**.
- **Toy-check scripts:** `Master/temp/dc_checks/dcNN/` (temp; user may relocate).

## Mechanism fingerprint (dedup device)

Every candidate is fingerprinted on three axes:

1. **Features-entry** — where item features enter the architecture
   (e.g. "features → factors à la SVDFeature", "prototypes live in feature space",
   "regression prior on embeddings").
2. **Grounding** — how prototypes acquire feature meaning
   (e.g. "prototype = learned attribute profile", "projection/push to real items",
   "concept-bottleneck activation").
3. **Read-out** — what the intrinsic explanation is at inference
   (e.g. "per-prototype score contribution × attribute profile",
   "concept activations shown directly").

A new candidate must differ from **every** registered candidate on **at least one
axis**, stated explicitly in Step 0/1. "Same idea, different paper" is a duplicate.

## Provenance classes

- `corpus` — from papers in the reading list.
- `corpus+external` — corpus seed extended by an external paper (web-verified;
  add the external paper to the reading list in Step 6).
- `model-knowledge` — from my prior knowledge; flagged as requiring literature
  support before it can carry thesis weight (find support in Step 2b or say so).
- `scratchpad` / `scratchpad+corpus` (v1.2) — seeded by a user scratchpad entry,
  alone or grounded by a corpus paper; same literature-support expectation as
  `model-knowledge` when no paper backs it.

---

## Step 0 — Frame

**Goal:** know the requirements and the occupied design space before ideating.

1. Read `feature_aware_solution_requirements.md` in full (it is a living
   document — do not rely on a remembered version).
2. Read `candidate_index.md` and skim each registered candidate's fingerprint
   and Step-1 shortlist (rejected seeds are free leads).
3. Read `scratchpad.md` (v1.2): the user's open entries are first-class input —
   they may name seeds, constraints, extensions, or doubts that shape this cycle.
4. **Artifact (`## 0. Frame`):** 5–10 lines — which fingerprint regions are
   covered, which are open, any requirement that recent candidates serve
   poorly (an open niche), and which open scratchpad entries look relevant to
   this cycle.

## Step 1 — Corpus scan & seed selection

**Goal:** pick one distinct, parsimonious seed mechanism.

1. Sweep the reading list (all themes — B backbone, C/D prototypes, E concepts/
   attention, F disentanglement, H fashion; §I for what's already taken).
1b. **Scratchpad sweep (v1.2):** give every *open* scratchpad entry — the user's
   own **and** any `[captured dcNN]` spillover from prior cycles — a one-line
   disposition in the artifact — `seed this cycle` / `shapes this candidate
   (how)` / `not this cycle (why)` / `answered inline (answer)`. **No open entry
   silently ignored.** A user idea competes for the shortlist on equal footing
   with corpus seeds (provenance: `scratchpad`, or `scratchpad+corpus` when a
   paper grounds it).
2. Shortlist **2–4 seeds**. A seed = one paper or an explicit combination
   (e.g. "B3 SVDFeature × C1 ProtoPNet push step"). For each shortlisted seed:
   - **Motivation gate:** one sentence — *which requirement(s) it serves and why*.
   - **Distinctness:** one sentence per registered candidate — how it differs
     (≥1 fingerprint axis).
   - **R8 parsimony filter:** is it ONE idea at `user_item_proto` granularity?
     Ensemble-shaped seeds are rejected *here*, before any deep-read effort.
3. Pick exactly **one** seed. Non-prototype explanation vehicles (AFM attention,
   pure concept bottleneck) are admissible but capped at 1–2 across the whole
   candidate set, tagged `S1-contrast` (ablation/contrast material, not headline).
4. **Artifact (`## 1. Seed selection`):** the shortlist with all three checks per
   seed, the chosen seed, provenance class, and why it won.

## Step 2 — Deep read

**Goal:** extract the seed mechanism precisely, from the actual papers.

1. Read the seed paper(s) — the **local PDFs**, not my memory of them. For
   combinations, read every constituent paper.
2. Extract: the exact mechanism (equations, with numbering), loss terms, training
   procedure, datasets/scale the authors used, what they claimed and what they
   actually showed, known limitations they admit.
3. **Citation-fidelity gate:** every extracted statement carries its location
   (section / equation / table). No paraphrase without a pointer.
4. **Artifact (`## 2. Mechanism extraction`):** the mechanism written out
   self-contained (the user must be able to follow without opening the paper),
   with the provenance pointers.

## Step 2b — Prior-art probe

**Goal:** verify the *adaptation* (not the seed) is not already published.

1. Targeted web search (2–4 queries): seed mechanism + prototypes/recommendation/
   side-features. Re-check against the §I forward-citation sweep in the reading
   list (all 31 ProtoMF citers were CF-only as of 2026-06-11 — re-verify for the
   specific mechanism, citers may have grown).
2. **Novelty gate — "refine or distinguish":** if something close exists, either
   refine the candidate to differ, or write explicitly how ours differs. Either
   outcome is recorded; "found nothing close" is also recorded (with the queries
   used, so the search is reproducible).
3. **Artifact (`## 2b. Prior-art probe`):** queries, findings, verdict.

## Step 3 — Adaptation design

**Goal:** the transplant into ProtoMF, concrete enough to implement from.

1. Specify: which modules change (`feature_extraction/feature_extractors.py`,
   factory, new `ft_type` per the reserved names in CLAUDE.md, losses, data
   pipeline), the forward pass with **tensor shapes** at `(batch, 1+n_neg)`
   granularity, where parameters live, what is tied (R1).
2. Specify the **intrinsic explanation read-out**: exactly which quantities from
   the forward pass form the explanation, and what a rendered explanation would
   look like for one concrete H&M-style example.
3. **Interpretability-constraint design (mandatory; added v1.1).** The host's
   regularizers (`sim_proto`/`sim_batch` inclusion criteria) are interpretability
   machinery — they prevent prototype degeneration. Every candidate answers three
   questions, at the same concreteness standard as the rest of this step:
   - **Inheritance:** which host regularizers survive the transplant, and do
     their *semantics* still hold under the new design (not just "the code runs")?
   - **New degeneration modes:** which new ways can this design's
     interpretability degrade that the inherited regularizers do not cover?
   - **Constraint candidates:** for each uncovered mode, *design* the natural
     soft/hard constraint — actual loss formula or hard-constraint mechanism,
     hook point (`get_and_reset_loss()` or existing knobs like `max_norm`),
     expected effect, cost — but **do not stack it into the candidate** (R8).
     Documented as optional knobs; adding one is the user's mechanism decision.
     Note which knobs (inherited + proposed) are **Rashomon-tunable** (see the
     requirements doc, §4 identifiability note).
4. **Anti-vagueness gate:** real class/file names and shapes — "add a feature
   encoder" without a signature does not pass.
5. **Baseline gate:** name the natural baselines (CF-only ProtoMF always; plus
   e.g. LightFM/B5 where apt) and the **ablation that isolates the one idea**.
6. Mark every deviation from the source papers as **"I propose"**.
7. **Artifact (`## 3. Adaptation design`).**

## Step 4 — Cost & feasibility

**Goal:** honest analytic expectations; feasibility flags, not vetoes.

1. **Parameter delta:** analytic, in terms of embedding dim d, n_prototypes K,
   H&M feature cardinalities (use real numbers from the 4.0 feature analysis
   where available). Compare against the CF-only `user_item_proto` count.
2. **Training-time expectation:** per-step FLOPs multiplier vs. baseline; effect
   on epochs/convergence if arguable.
3. **S5 pipeline-compatibility checklist** (from the requirements doc): sampled
   `(batch, 1+n_neg)` scoring preserved? No full-catalog pass per step? Item/user
   branch still cheap indexed computation? Losses fit `get_and_reset_loss()`?
   Anything expensive per item → state the precompute/cache strategy.
4. **Budget gate** (compute rules): cluster-only for real runs; >10 days
   infeasible (hard); >~1 day soft-infeasible for dev iterations. A breach flags
   the candidate and demands a mitigation; it does not auto-kill (spectrum rule).
5. **Artifact (`## 4. Cost & feasibility`):** estimates labeled as Fermi-style,
   checklist verdicts, flags.

## Step 4b — Black-box math sanity check

**Goal:** falsify what is cheaply falsifiable; do not fool myself.

1. **Claims spec (always written):** distill the candidate's checkable
   mathematical claims (cold-item activation non-degenerate, loss minimized where
   intended, gradient flows through the tie, score decomposes per-prototype, …).
   Where Step 3 designed constraint candidates, their checkable properties (loss
   minimized at the intended configuration, gradient reaches the intended
   parameters, degenerate cases) are standard claim material (v1.1).
   The spec contains mechanism definition + claims — **not** my derivation, not
   expected outcomes.
2. **Viability self-assessment (mandatory):** for each claim, judge whether a toy
   test is meaningful. DL-heavy claims that only materialize through training
   dynamics → reduce to what IS checkable at toy scale (shapes, gradient flow,
   degenerate/limit cases) or skip **with written justification**. Skipping is
   legitimate; silent skipping is not.
3. **Black-box execution:** spawn a subagent that receives ONLY the claims spec
   (never my derivation or this candidate file). It designs and runs a toy
   script (numpy/torch, toy dims, seconds of runtime, no training pipeline) in
   `Master/temp/dc_checks/dcNN/`, and reports per-claim: confirmed / falsified /
   untestable-at-toy-scale.
4. **One round only.** No re-spec-and-re-run until pass. A falsified claim is
   carried into Step 5 as a known defect; any mechanism revision is for the user
   to see first, with the failed check attached.
5. **Artifact (`## 4b. Math sanity check`):** claims spec, viability notes,
   subagent report verbatim, script path.

## Step 5 — Requirements evaluation

**Goal:** the in-depth assessment the whole cycle exists for.

1. Re-read the requirements doc once more (not from memory).
2. For **every** point — R1, R2, R3, R4, R5, R6, R7, R8, S1, S2, S3, S4, S5 —
   write: a rating (**strong / partial / weak / conflicts / n-a**) **plus**
   genuine prose: how the candidate satisfies it, where it strains, open risks.
   No checkmark-only rows. Remember: every "must" is a strong default on a
   spectrum, not a veto — argue trade-offs, don't hide them.
3. Assess against the **§4 design tensions** (where features enter, fidelity vs.
   accuracy, attribution granularity, R7 perceptual alignment, identifiability).
4. Run the **part-prototype pitfalls checklist** (reading-list S5: prototype
   quality/quantity, collapse, semantic gap, …) — one line each.
4b. **Interpretable-AI placement (conditional):** if the thesis's principle
   framework has been fixed (see `Master/docs/notes/ideas.md` §4 — e.g. Rudin S1
   or Zhang & Chen S2), additionally place the candidate against it: which
   principles it covers, which not. Skip silently while the framework is
   undecided.
4c. **Degeneration-protection gate (added v1.1):** a table with one row per
   degeneration mode identified in Step 3, status ∈ {protected by inherited
   regularizer / protected by proposed (unstacked) constraint / unprotected —
   open risk}. The gate is completeness: **no identified mode may be silently
   unlisted** (same spirit as 4b's "skipping is legitimate, silent skipping is
   not"). Statuses flag, never veto (spectrum rule).
5. Fold in Step 4b falsifications as known defects, explicitly.
6. **Artifact (`## 5. Requirements evaluation`):** rating table + prose,
   tensions, pitfalls pass.

## Step 5b — Devil's advocate

**Goal:** an adversarial read this candidate's author cannot give it.

1. Spawn a subagent that receives ONLY the candidate file (not this
   conversation, not my private reasoning) with the brief: *"strongest case
   against this design — why it fails its requirements, what its author is too
   in love with to see, what breaks at H&M scale."*
2. Attach the critique **verbatim**. Then write my response: concede / rebut /
   mark as open risk, point by point. No silent edits of earlier sections in
   response to the critique — visible amendments only.
3. **Artifact (`## 5b. Devil's advocate`):** critique + response.

## Step 6 — Register, annotate, cold-read

**Goal:** make the candidate usable by the user, standalone, and close the loop.

1. **Fingerprint & index:** write the 3-axis fingerprint into the candidate file;
   add the index row (id, title, seed papers, provenance class, fingerprint,
   `S1-contrast` tag if applicable, status `draft — awaiting review`, date).
2. **Reading-list back-annotation:** for every paper the candidate draws on, add
   a one-line indented note under that paper's entry in
   `feature_aware_reading_list.md`:
   `  - ↪ **dcNN**: <super-short — what was taken and how it was adapted>`
   so the user, when reading that paper, already knows what was done with it and
   can judge the candidate *while* reading. One line per paper, no more. New
   external papers (provenance `corpus+external`) get added to the list itself.
2b. **Scratchpad back-annotation (v1.2):** under every scratchpad entry this
   cycle consumed or answered, append one indented marker line:
   `  - ↪ dcNN <date>: <disposition — seeded / shaped §X / answered: …>`.
   Entries dispositioned `not this cycle` stay untouched (they re-enter the next
   cycle's sweep). Never edit or delete the user's entry text.
2c. **Spillover flush (v1.3):** before stopping, confirm that every adjacent
   idea/variant surfaced during this cycle (Ground rule 9) is recorded in
   `scratchpad.md` as a `[captured dcNN]` entry. Completeness gate, same spirit as
   the others: a valuable idea raised in conversation but never written did not
   survive. Capturing nothing is legitimate (nothing of value came up); silently
   dropping something is not.
3. **Cold-read gate:** re-read the candidate file top-to-bottom, fresh, as if I
   were the user. Test: *could Matteo evaluate — and later implement — from this
   file alone, without our conversation?* Fix gaps now (visible edits).
4. **Stop.** Present a short summary to the user. Do not start the next cycle —
   the user reviews first and invokes the skill again.

## Stopping condition (across cycles)

The well is dry when Step 1 cannot produce a single shortlist seed that passes
the distinctness and R8 gates. That cycle produces no candidate; instead, write a
short closing note in the index ("design space saturated as of dcNN, date") and
tell the user. A few strong candidates are the goal — not an endless stream.

## Re-run mode

`/design-candidate stepN dcMM` re-runs only Step N for registered candidate dcMM
(e.g. after the requirements doc changed): re-read this doc's Step N section,
redo it against the current state of all inputs, update the candidate file
section **with a dated edit note**, and update the index row's date/status.
