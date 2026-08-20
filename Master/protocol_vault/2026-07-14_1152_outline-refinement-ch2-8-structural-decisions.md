---
date: 2026-07-14
time: "11:52"
phase: 4
tags: [thesis/structure]
placed: 2026-08-20
---
# Outline refinement walk (Ch 2–8): merge, granularity principle, delta principle, explanation system placement, afU road

Full-day outline-refinement session over the thesis skeleton (`Master/thesis/`); decisions batch-logged. All structural, all reflected in `outline.md` + chapter files; supervisor-relevant forks recorded in the SQ register.

1. **Ch 2+3 merged into "Background and Related Work"** (SQ-7): the old Related-Work middle overlapped the Background teaching material almost 1:1 (feature-aware MF ↔ LightFM section; prototype/concept positioning ↔ x→c→y section), and the genuinely novel neighborhood (prototypes in recommendation) is thin. Merged form = teach-then-position along two axes (interpretability axis / mechanism axis), chapter ends with **The Gap** (pincer close: each axis's failure mode → checkable claim → "the gap is the combination, not an ingredient" → modality-light → hand-off). A later split would be quite hard to untangle — flagged for the supervisor *before* prose.
2. **Granularity principle** (now standing, memory-saved): outline headers follow *content mass*, not chapter parity — few flat sections + content-block bullets; sub-headers added at prose time only if blocks demand dividers. Applied: Data ch → 3 flat sections (datasets = source/contents/why-selected only; ALL decision-shaped material incl. variants, cold splits, canonical feature set, ml-1m genre bags, sparsity positioning lives in Preprocessing & Splits); eval-framework ch → Computational Setup / Comparison Design / Part II placeholder.
3. **The explanation system lives in the eval-framework chapter**, demonstrated on the host ProtoMF (already taught in Background) — the demo deliberately exposes the naming gap empirically (centroid-read prototype names) right before the lineage closes it. **Lineage chapters show DELTA ONLY** ("focus on what is new") — fI: parameter-derived item-prototype names; fU: per-purchase regrouping; no host re-demos.
4. **Delta principle across the lineage:** fI = full-build chapter (what/why/how exactly); fU = delta chapter (the not-a-mirror surprises ARE the content; drift guard on the model section); merge = delta of second order, thinnest by design, don't pad. fI's why-section ("Taste Is Revealed, Not Stated") moved to open the chapter and is annotated as the lineage's justification hub (fU's intro calls back to it).
5. **afU road moved into the fU chapter** (revises the committed lineage outline's step-7 presentation home, `2026-07-11_2351`): the cold-start bell rings in fU's empty-histories section, and the argument (user attributes enter last, weakest, only where nothing else exists) needs only the taste argument, not the merge. Section title announces it: "Users with Empty Histories: The Road to **User-Attribute-Aware Prototypes**" — *user*-attribute because everywhere else "attribute" means item attributes. **afU very likely WILL be built** (Matteo) — written as announced road, not hedge. Merge chapter title now "(a)fUfI-ProtoMF"; its Design section provisionally titled "The Tie" (rename by actual tie count once the candidate exists); one-sentence bullet acknowledging the goofy letter-string names as a pure *versioning device* — all these models are really just Attribute-Aware ProtoMF.
6. **Regularizers demoted from "pure" interpretability penalties** (Matteo): like L1/L2 they plausibly also affect accuracy — an empirical question; **small ablation study planned** (accuracy AND legibility effects). Consistency edit in Part II's motivation.
7. Hidden-effects chapter deliberately left unformed (headers = inventory placeholders; expect regrouping + discarding of unmeasured rungs after experiments).

[[thesis-writing]] [[decisions]] [[phase-4]]
