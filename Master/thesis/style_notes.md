# Thesis Style Ledger — Matteo's revealed writing preferences

*Loaded by /thesis-review (Phase 0) and useful to /thesis-write Phase C. Append-only per session; each entry dated. These are decisions, not suggestions — apply without re-litigating.*

## Punctuation & register (2026-08-21, fi-why sessions)
- **No colons, semicolons, or em-dashes in prose** — not his natural style. Prefer sentence breaks or commas. Sweep them out as a closing step of any pass over his text.
- Plain register; no business-speak ("leverage"), no filler ("essentially"), no throat-clearing ("We want to emphasize that").
- Simple present over progressive ("We derive", not "We are deriving").

## Terminology (2026-08-21)
- **Three-layer hierarchy:** *characteristics* = everything an item presents to a person (incl. look, photo-capturable) ⊃ *attributes* = the recorded, coarse catalogue subset → *interpretable feature* = an attribute's data form in the model. Perception-level statements say characteristics; model-facing statements route through attributes/features. (Refines the Ch-1 attribute-vs-feature convention; Ch 1 + outline still need the third layer added.)
- **No capitalized coinages** ("Taste-groups", "Item-Characteristics-Groups" are out). Plain lowercase phrases: full form first ("groups of items with shared attributes" / "groups of users with shared taste"), short forms after ("attribute groups" / "taste groups").
- **"Anchor" is a role, never a noun** for ProtoMF prototypes ("the prototypes anchor the space") — the noun belongs to ACF's anchors. Background's ACF passage owes a one-clause kinship note.
- 5.1 title is "Prototype Grounding" (provisional); the section treats ProtoMF as UI-ProtoMF for the intro (both sides prototyped) — needs one reader signal eventually.
- British spellings where started: catalogue, colour.

## Mechanism-accuracy anchors (2026-08-21)
- fI grounds the **item embedding table** (q_i = Σ attribute rows); prototypes stay free vectors and inherit legibility — "grounding the prototypes" = payoff shorthand, mechanism stated precisely in 5.2. Whether prototypes stay necessary = the lightfm_hist control's question.
- ProtoMF's post-hoc interpretation = **ranked most-similar-items inspection**, not clustering.
- Niknazar & Mednick 2024 (niknazar2024sleep): vocabulary-constrained first layer, +10% ablation gain, trade-off shown false — supports "increased accuracy as a side effect".

## Matteo's recurring mechanical slips (fix on sight, Phase 1)
German-style comma before restrictive clauses ("occurs, when", "To the extent, that", "deciding, what"); "Lense"→"lens" (recurred 3×); doubled words ("be again be", "a the"); missing apostrophes ("an items", "the models internals"); mid-sentence capitalized common nouns ("Prototypes", "User Attributes"); comma splices; adress/increaded/emphazise-class typos; straight quotes; hardcoded section numbers instead of \ref.
