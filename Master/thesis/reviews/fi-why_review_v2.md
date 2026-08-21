# Review v2 — Section 5.1 `fi-why`

**Date:** 2026-08-21 · **Skill:** /thesis-review (fresh invocation after Matteo's mid-session edits + v1 Phase-3 decisions) · **Git state:** HEAD 9d8bdd4, draft uncommitted on `feat/scrutiny`
**Standing decisions applied as review frame (v1 Phase 3):** section essence = asymmetry of the two conceptual ingredients; implementation order = brief transition statement only; current title disliked (sequence-focused), retitle pending.
*LLM-usage-disclosure artifact — do not delete; copy into `Master/llm_usage/` at finalization.*

## Phase 1 — mechanical fixes applied (3)

1. "that shape, when they generate" → comma removed (German-style comma before restrictive "when")
2. "For the item ingredient if a user" → comma after "ingredient"
3. Final transition sentence: missing full stop added after "merge the models"
4. (Requested addition, 2026-08-21) Source lines rewrapped to a consistent 78-char width across the section's prose; layout-only, no word or paragraph-break changes; standalone `...` placeholder lines kept as-is.

**Post-edit note:** Matteo's own V2-5 repair ("captures exactly this, but it as an unlabeled vector…") landed during the session and replaced the no-alternative-vocabulary sentence — V2-5 resolved in substance; one verb missing in the new clause ("but it as"), left for Matteo (word choice: drop "it" or add "does so as"/"arrives as").

## Phase 2 — findings

No blocking findings. Majors first; (c/f) = carried forward from v1 with direction already agreed.

### V2-1 — MAJOR · red line: title (and framing) contradict the agreed reframe
> \section{Why the Item Side First: Taste Is Revealed, Not Stated}
Decision from v1: essence = ingredient asymmetry, ordering = consequence. The header still leads with sequence; outline.md's headline for this section ("why-item-first") will need the same touch. **Fix direction:** asymmetry-first title (the "Taste Is Revealed, Not Stated" half already is the asymmetry); update outline.md when set.

### V2-2 — MAJOR (c/f F5) · claim–support: the three placeholder gaps in the opening paragraph
> "... (Intro, duction into the chapter...)" · "...(makes them hard to explain (why is the collaborative signal never explained in itself))" · "... The two ingredients lens..."
Self-signaled unfinished. The hard-to-explain claim still rests on nothing; filling argument supplied in v1 Phase 3 (relational signal → who-bought encoding → no attached semantics → post-hoc naming). Sample formulation offered, not yet requested.

### V2-3 — MAJOR (c/f F4) · clarity: pivot comparison still implicit
> "The two ingredients lens however is more intuitive and will therefore inform how the prototypes should be grounded."
Referent agreed in v1 ("more intuitive than the collaborative signal") but not yet in the text; "two ingredients lens" still coined without setup.

### V2-4 — MAJOR (c/f F3) · clarity: item-ingredient sentence still packs condition + consequence
> "For the item ingredient, if a user has no prior external knowledge, an item's characteristics are necessarily constrained to what is shown to the user."
Direction agreed: split into two sentences keeping both elements (no-prior-knowledge perspective; characteristics = what is shown).

### V2-5 — MAJOR (new) · clarity / coverage: the embedding sentence stops one clause short
> "Instead, users reveal their taste through their interactions, which in practice is described by a CF user embedding."
Two issues. (a) Agreement/referent: "which … is" — grammatical antecedent is "interactions" (plural); the intended referent is presumably the revealed taste. Reader stumbles. (b) The load-bearing link is still missing: *why* the CF embedding does not itself solve the problem — it captures revealed taste but carries no labels/vocabulary (vault: "arrives without labels"). Without that clause, "There is no alternative vocabulary to properly ground the user side" arrives unmotivated. **Fix direction:** resolve the referent; add the captures-but-unlabeled clause between this sentence and the no-alternative-vocabulary claim.

### V2-6 — MAJOR (c/f F6) · terminology: "anchor points" collides with the ACF baseline
> "…where the prototypes act as anchor points, which add some structure to the embedding space."
"Anchor" is the Anchor-based CF baseline's term; the thesis runs and teaches ACF. Undiscussed in v1. **Fix direction:** different word.

### V2-7 — MAJOR (new) · structure: the transition sentence can't yet carry its callback role
> "The following section will detail the construction of the easier item side, and only after that introduce the adaptation for the harder user side and eventually merge the models."
(a) "The following section" — the user side and the merge are later *chapters* (Ch fU, Ch merge), only the item-side construction is the next section; (b) no \ref anchors — fU's opening is planned to call back to exactly this sentence, so it wants Ch.~\ref{ch:fu} (and optionally \ref{ch:fufi}); (c) subject drift: "the section will … merge the models" — the thesis/we merges, not the section. **Fix direction:** one sentence, correct granularity (next section vs later chapters), refs in, consistent subject.

### V2-8 — MINOR (c/f F7) · house style: double hedge + capitals
> "User Attributes might correlate with taste, but are often very thin…"
"might … often" hedges a claim the vault supports plainly; "User Attributes" mid-sentence capitals.

### V2-9 — MINOR (c/f F8) · citations: bilinear-shape and revealed-preference claims uncited
Sample carried `koren2009mf` and `samuelson1948consumption`; both in bibliography.bib. Decide whether they land here.

### V2-10 — MINOR (c/f F10) · structure: closing definitional passage restates itself three times
> "Let it again be made clear that … explain themselves."
Also "deciding what and how to integrate" still lacks its object; "again" points at nothing earlier.

### V2-11 — MINOR (c/f F11) · terminology: "Item-Characteristics-Groups" / "User-Taste-Groups"
Capitalized coinages; "characteristics" beside the agreed attribute-vs-feature convention.

### V2-12 — MINOR (new) · wording: "adapted as interpretable vocabulary"
> "Interaction histories need to be the primary signal, but adapted as interpretable vocabulary."
"adapted as" reads off; likely "adapted into" / "translated into". Also elliptical ("need to be … but adapted") — small expansion would smooth it.

## Phase 3 — discussion outcomes
*(to be filled during the session)*
