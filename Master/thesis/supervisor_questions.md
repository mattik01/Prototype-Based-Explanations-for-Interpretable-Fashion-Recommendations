# Supervisor Questions — Open Decisions Register

**Procedure (Matteo, 2026-07-13):** whenever a decision is hard to settle or we genuinely can't choose, we do NOT force it — we mark the spot (`SQ-N` comment in the file), record the variants and our thoughts here, and bring it to the supervisor. Items get resolved top-down at supervision meetings; resolved items move to the bottom with the outcome.

## Open

### SQ-1 · Headline of §1.2 (introduction)
- **A (current in text):** "The Naming Problem in Prototype-Based Matrix Factorization" — compact noun phrase, full context; "naming problem" implies the prototypes as object.
- **B (Matteo's fuller version):** "Matrix Factorization with Prototypes and the Problem of Naming Them Post Hoc" — maximally explicit (names the object and the post-hoc aspect), but reads sentence-like.
- Our thoughts: both survive the no-filler/precision bar; A is tighter, B is more self-explanatory in the TOC.

### SQ-2 · Terminology lane: "transparent" vs "interpretable" recommendation
- **Updated 2026-07-13 (second pass):** Matteo demoted "transparent" as the goal word — transparency = visible workings (weak: parameter-transparent MF is not understandable; neighborhood CF is transparent *and* legible but weak), interpretability = understandable/legible reasoning (the actual, stronger goal). §1.1 now reads "Motivation: Interpretable Recommendation". Matches Lipton/Rudin usage.
- Remaining check: Matteo verifies against S2/Rudin definitions during the survey re-read; supervisor only if still ambiguous. The transparency/interpretability *distinction itself* gets one definitional moment in §1.1.
- **Upgrade 2026-07-14 (vault `2026-07-14_1223`):** the terminology story now has a precise ladder decomposing "explained well" — transparency (window exists, architectural) → fidelity (window doesn't lie, descriptor gap) → legibility (room is tidy, complexity). Transparency gets a subordinate rung instead of competing as goal-word; each term keeps its literature meaning (Lipton; G1/S4 fidelity). This likely settles the remaining check in favor of the current lane.

### SQ-3 · Working thesis title
- **A (current default):** "Intrinsically Interpretable Recommendation through Attribute-Aware, Prototype-Based Matrix Factorization"
- **B (transparent lane, weakened by SQ-2 update):** "Transparent Recommendation through Attribute-Aware, Prototype-Based Matrix Factorization"
- **C (cocky, Part-II-forward):** "Tuning Interpretability in Recommendation through Attribute-Aware, Prototype-Based Matrix Factorization"
- **D (cockier still — Matteo, 2026-07-13):** "Tunably Interpretable Recommendation through Attribute-Aware, Prototype-Based Matrix Factorization" — recorded per Matteo's instruction as the more ambitious title, **gated by (1) Part II's success and (2) the supervisor's tolerance towards my antics**. (She might chuckle reading this.)
- Coupled to SQ-2; A stays the working title until II lands; C/D promote the tuning claim to the headline act — only defensible if the knob experiments deliver.

### SQ-4 · Results placement: inside each lineage chapter vs. one shared Evaluation chapter
- **A (current skeleton):** per-stage results inside Ch 6/7/8 (narrative demands presenting fI fully — incl. numbers and explanation demo — before fU enters); shared machinery factored into Ch 5.
- **B (DBIS exemplar pattern):** single Evaluation chapter after all model chapters.
- **C (hybrid):** per-stage results + short cross-lineage comparison after the merge.
- Our thoughts: A preserves the committed storyline; the cost is no single home for cross-stage comparison (C repairs that cheaply).

### SQ-5 · Part I / Part II split
- `\part{}` markers for "The Attribute-Grounding Lineage" / "Towards Interpretability as a Model Objective" (Part II renamed 2026-07-14, was "Interpretability Quantified") are **ACTIVE** (Matteo's default, 2026-07-13: "truly split it in I and II unless she says otherwise"). Two heterogeneous contribution blocks — no verified exemplar covers this; supervisor can veto. Note: the Part II title now shares its key phrase with SQ-3 title-ladder variants C/D — coherence, not conflict.
- Open sub-question: Ch 1–5 currently sit before Part I (unparted front matter) — fine in LaTeX and common, but she may prefer a "Part 0/Foundations" grouping or the parts to start earlier.
- Open sub-question: the Conclusion chapter currently falls *inside* Part II (LaTeX has no "end of parts"); acceptable, but alternatives exist (unnumbered closing part, or Part II titled to cover both).

### SQ-6 · Authorial voice: "we" vs impersonal/passive — **RECLASSIFIED: deferred, Matteo's own call (not for supervisor ruling)**
- **A (working default):** first-person plural "we" (+ occasional deliberate "one") — matches the ProtoMF paper ("we describe our ProtoMF models"), Kula's LightFM (even "I"), Rudin's voiced style (Matteo's model), and Matteo's authored sentences.
- **B (DBIS exemplar practice):** fully impersonal/passive — the Wurzinger sample thesis contains zero authorial "we"/"I" (its only "we"s are song lyrics in the data).
- DBIS states no rule (verified 2026-07-13). Matteo defers the final decision to himself, later; until then everything is written in "we" (active→passive retrofit is mechanical, the reverse is not).

### SQ-7 · Merged "Background and Related Work" chapter vs. the DBIS two-chapter skeleton
- **A (current skeleton, Matteo 2026-07-14):** one merged chapter — teach-then-position along the two axes (interpretability axis, mechanism axis), ending with The Gap. Chosen because the old Related Work middle overlapped the Background teaching material almost 1:1 (feature-aware MF family ↔ the LightFM section; prototype/concept positioning ↔ the x→c→y section), and the genuinely novel neighborhood (prototypes in recommendation) is thin — separate chapters forced teaching the same material twice.
- **B (DBIS sample skeleton):** separate Background (teach, neutral) and Related Work (argue, end at the gap) chapters.
- Our thoughts: DBIS structure is group-level convention, not a rule; the gap-closing arc the norm cares about is preserved (the merged chapter still ends with The Gap, and every positioning passage keeps explicit relate-to-ours moves). **Note for the meeting: a later split would be quite hard to untangle** — positioning passes are interleaved with the teaching sections at paragraph level (that interleaving is the point of the merge), so if she wants two chapters, the call should come before the chapter is prose.

## Resolved

*(none yet)*
