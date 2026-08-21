# Review v1 — Section 5.1 `fi-why` ("Why the Item Side First: Taste Is Revealed, Not Stated")

**Date:** 2026-08-21 · **Skill:** /thesis-review · **Scope:** fresh AI-free first draft (uncommitted prose at top of `sec:fi-why`, `06_fi_protomf.tex`) · **Git state:** HEAD 9d8bdd4, draft uncommitted on `feat/scrutiny`
**Reference material:** agreed P1/P2/P3 outline (in the .tex, 2026-08-20), vault `2026-07-11_2350`, Phase-C sample `samples/fi-why_v1_body.tex` (coverage checklist only).
*LLM-usage-disclosure artifact — do not delete; copy into `Master/llm_usage/` at finalization.*

## Phase 1 — mechanical fixes applied in-place (18)

1. "occours, when" → "occurs when" (spelling + comma)
2. "matrix factorizations models" → "Matrix factorization models" (capitalization + agreement)
3. "the Prototypes act" → "the prototypes act" (mid-sentence capitalization)
4. "collabarotaive" → "collaborative" (inside Matteo's own placeholder note)
5. "Lense" → "lens"
6. straight quotes around "Item ingredient" → LaTeX ``…'' (+ lowercase "item")
7. "an items characteristics" → "an item's characteristics"
8. "To the extent, that" → "To the extent that"
9. "as features interpretable features" → "as interpretable features" (duplicated-word repair)
10. double space "to  provide" → "to provide"
11. ", Which we should" → ", which we should"
12. "conencted" → "connected"
13. "(see section 6.2)" → "(see Section~\ref{sec:fu-cold-users})" (hardcoded number → label; compiled numbers shift with file order)
14. "Let it be again be made clear, that" → "Let it again be made clear that" (stray "be", comma)
15. "deciding, what" → "deciding what"
16. "the models internals" → "The model's internals" (sentence capital + apostrophe)
17. "trough" → "through"
18. "naturaly" → "naturally"

## Phase 2 — findings (no edits made)

### F1 — BLOCKING · red line / coverage: the "why FIRST" argument (P3) is missing entirely
> draft ends: "…Prototypes are grounded in something interpretable and explain themselves."
The section's title and job is *why the item side first*, and the fU chapter's opening is directed to call back to this section's closing strategy sentence. The draft argues the item/user vocabulary asymmetry but never draws the ordering conclusion: no "in pure CF both embeddings are behavior summaries," no "item-side legibility was always borrowed from the catalogue," no "the order is data-forced," and no closing strategy sentence with the fU pointer. **Fix direction:** add the P3 paragraph per the agreed outline.

### F2 — MAJOR · coverage: P2's "already captured" half is absent
> "Instead, users reveal their taste through their interactions, which we should adapt to be the appropriate vocabulary…"
The vault calls this "the refinement that completes it": the CF user embedding already IS learned revealed taste — capture was never the problem; the problem is it arrives without labels. The draft jumps from "taste is revealed" straight to "adopt interactions as vocabulary," skipping the mechanism that justifies it. (Source: vault 2026-07-11_2350; P2 bullet.) **Fix direction:** insert the no-capture-problem / no-labels step between reveal and vocabulary conclusion.

### F3 — MAJOR · clarity: the item-ingredient sentence is hard to parse
> "For the ``item ingredient'' if a user has no prior external knowledge, an item's characteristics are necessarily constrained to what is shown to the user."
Whose knowledge, and what is constrained for whom, takes rereading; the underlying point (item content is observable and recorded in the catalogue) is simpler than the sentence. **Fix direction:** state observability/recordedness directly, drop the no-prior-knowledge framing or make it one clear clause.

### F4 — MAJOR · clarity / structure: the pivot sentence dangles
> "The two ingredients lens however is more intuitive and will therefore inform how the prototypes should be grounded."
"More intuitive" than what is unstated (presumably: than the collaborative embeddings just described); "the two ingredients lens" is coined here without setup. This is the sentence that turns the section from setup to argument — it must carry its comparison explicitly. **Fix direction:** name both sides of the comparison; consider merging with the preceding hard-to-explain claim.

### F5 — MAJOR · claim–support (self-flagged placeholder): "hard to explain" rests on nothing yet
> "…learned from collaborative signals, often exclusively and often at massive scale, ...(makes them hard to explain (why is the collaborative signal never explained in itself))"
Matteo's own margin question. The missing reasoning exists in the vault: a CF embedding encodes who-bought-it, with no semantics attached. Note this is the same material P3 needs (borrowed legibility) — one passage could serve both (F1). **Fix direction:** ground the claim with the behavior-summary/no-labels reasoning, or lean on the Background chapter's ProtoMF section and keep only a pointer.

### F6 — MAJOR · terminology: "anchor points" collides with the ACF baseline
> "…where the prototypes act as anchor points, which add some structure to the embedding space."
"Anchor" is the term of the Anchor-based CF baseline (Barkan et al.) that this thesis runs and teaches; describing ProtoMF's prototypes as anchor points invites exactly the confusion the Background chapter separates. **Fix direction:** different word (e.g. reference points / structure the space around learned prototypes).

### F7 — MINOR · house style: double hedge on the attributes claim
> "User Attributes might correlate with taste, but are often very thin…"
"might … often" hedges a claim the vault supports strongly (loose proxy, far too unspecific to ground an explanation in). Also mid-sentence capitals "User Attributes". **Fix direction:** state it plainly per the vault; DBIS hedge-free norm.

### F8 — MINOR · coverage / citations: sample carried two cites, draft has none
Bilinear-shape claim (sample: `koren2009mf`) and revealed preference (sample: `samuelson1948consumption`); both keys exist in `bibliography.bib`. DBIS norm: every fact reasoned or cited. **Fix direction:** decide whether these two claims get their cites here.

### F9 — MINOR · wording: "adapt" for "adopt", overloaded sentence
> "…their interactions, which we should adapt to be the appropriate vocabulary to explain User-Prototypes."
Likely "adopt as"; and the sentence stacks reveal + vocabulary-adoption in one clause. **Fix direction:** split; "adopt … as the vocabulary".

### F10 — MINOR · structure: the closing definitional passage restates itself three times
> "Let it again be made clear that when we talk about vocabulary, we are deciding what and how to integrate into the model. The model's internals are constrained to a vocabulary, such that just through transparency an explanation emerges naturally. Prototypes are grounded in something interpretable and explain themselves."
Three sentences, one point; "deciding what and how to integrate" lacks an object; "again" refers back to nothing that defined vocabulary earlier. **Fix direction:** merge into one or two sentences; give "integrate" its object (which attribute information, in what form).

### F11 — MINOR · terminology: "Item-Characteristics-Groups" / "User-Taste-Groups"
Coined capitalized hyphen chains; also "characteristics" sits beside the agreed attribute-vs-feature convention (Ch 1: "attribute" = human-readable catalogue property in prose). **Fix direction:** consider "groups of items sharing attributes" phrasing or lowercase the coinages; align on "attribute".

## Phase 3 — discussion outcomes (2026-08-21)

- **Section reframe (Matteo):** the section's essence is the *asymmetry of the conceptual ingredients* of a recommendation (item content vs user taste); the implementation order is a natural consequence, mentioned only as a brief outline/transition statement. Current title (sequence-focused) disliked — retitle pending, asymmetry-first. Amends the 2026-08-20 P1/P2/P3 emphasis: P3 shrinks to a transition sentence.
- **F1:** largely dissolved by the reframe + Matteo's new closing transition sentence (added during session). Residual notes: sentence says "the following section" but fU/merge are chapters; keep it as the anchor for fU's opening callback (add \ref{ch:fu} at polish time); missing final period.
- **F2:** addressed by Matteo's edits ("which in practice is described by a CF user embedding", "no alternative vocabulary", histories-as-primary-signal sentence).
- **F3:** agreed too complicated; elements to keep: no-prior-external-knowledge perspective + characteristics-are-what-is-shown. Direction: split condition/consequence into two sentences.
- **F4:** referent supplied — "more intuitive than the collaborative signal"; Matteo to insert.
- **F5:** help requested; argument supplied in-conversation (relational signal → who-bought encoding → no attached semantics → post-hoc naming). Sample formulation offered, not yet requested.
- F6–F11: not yet discussed.
