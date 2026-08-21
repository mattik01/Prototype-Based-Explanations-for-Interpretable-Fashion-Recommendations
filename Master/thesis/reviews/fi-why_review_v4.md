# Review v4 — Section 5.1 `fi-why`

**Date:** 2026-08-21 · **Skill:** /thesis-review (fourth invocation) · **Git state:** HEAD 9d8bdd4, draft uncommitted on `feat/scrutiny`
**Scope:** Matteo's manual rework implementing the five-move architecture (v3) in his own words. **Added lenses this pass (Matteo):** paragraphing, thesis-wide term consistency, self-coherence.
*LLM-usage-disclosure artifact — do not delete.*

## Phase 1 — mechanical fixes applied

1. Removed a verbatim copy-paste duplication of the entire opening sentence ("An interaction occurs … item embedding." appeared twice back to back)
2. "learnedembedding,the" → "learned embedding; the" (missing spaces; comma splice repaired with semicolon — flagging the choice)
3. Rejoined "n othing" → "nothing" (mid-word line-break)
4. Double space after "scale."
5. "its Prototypes" → "its prototypes"
6. "(see previous)" → "(Section~\ref{sec:bg-protomf})" and "(previous chapter)" → "Section~\ref{sec:bg-iml-intrinsic}," (placeholder refs realized; verify targets)
7. "emphazise, that" → "emphasize that"
8. "interpretabilty" → "interpretability"
9. "a models internals" → "a model's internals"
10. "A Recomendation is a the Users taste" → "A recommendation is the user's taste" (spelling, duplicate article, apostrophe — sentence still flagged for rewrite, V4-3)
11. "A user, that approaches an item" → "A user approaches an item" (restores the sentence's grammar)
12. Source rewrapped to 78 chars (standing request)

## Phase 2 — findings

### Coherence

**V4-1 — MAJOR · move 1 layer order + the "every score" overclaim (c/f, unresolved).**
> "users and items are represented by their similarity to a small set of learned prototype vectors. Each item and user still owns a learned embedding; the prototypes just act as the structural anchors of the embedding spaces, and every score is composed from positions relative to them."
The similarity-representation claim comes BEFORE the embedding table exists for the reader, so "still owns a learned embedding" arrives as a correction of the previous sentence rather than a layer beneath it — locally self-contradictory on first read. And "every score is composed from positions relative to them" still overclaims for the single-sided host (the user factor is a free embedding reading prototype activations). **Fix direction:** order the layers embedding → similarity profile → score; scope the anchor claim to the prototyped side.

**V4-2 — MAJOR · move 2 first sentence misattributes prototypes to MF at large.**
> "Prototypes and Embeddings as is common in MF models are learned exclusively from the collaborative signal and often at massive scale."
Reads as prototypes being common in MF models; they are ProtoMF-specific. Also "exclusively … and often at massive scale" — the exclusivity is the point (keep), "massive scale" is decoration that our datasets do not instantiate. Capitals "Prototypes and Embeddings". **Fix direction:** e.g. structure as "As is common in MF, the embeddings — and with them the prototypes — are learned exclusively from the collaborative signal."

**V4-3 — MAJOR · move 4 payoff sentence is broken.**
> "A recommendation is the user's taste by similarity to Taste-groups, matched with an item's characteristics by similarity to Item-Characteristics-Groups."
"is the user's taste by similarity to" has no working verb structure; the intended content (score = taste-side profile matched with characteristics-side profile) doesn't parse. Also reintroduces the capitalized coinages (see V4-5).

**V4-4 — MAJOR · "the vocabulary is intuitive" undersells the asymmetry it must carry.**
> "For the item ingredient, the vocabulary is intuitive."
The asymmetry the section argues is recorded vs latent (availability), not intuitive vs unintuitive — user taste is perfectly intuitive as a concept, it is just not written down. The next paragraph's "not as straightforward" then contrasts against the wrong axis. **Fix direction:** topic sentence on availability/recordedness ("already written down" did this).

### Term consistency (internal + thesis-wide)

**V4-5 — MAJOR · coinages inconsistent within the section.** Move 4 says "Taste-groups" / "Item-Characteristics-Groups"; part two says "characteristic groups" / "groups of shared taste". Pick one form, lowercase, use it both places.

**V4-6 — MAJOR · "characteristics" vs the thesis-wide "attribute" convention.** Ch 1 owns the agreed convention: "attribute" = human-readable catalogue property (prose/semantics), "feature" = its encoded representation. This section runs "characteristics" in the attribute role throughout, while the chapter announces "grounds the item side in catalogue attributes" (outline, lead-in). Either switch to "attributes" here or the convention gains a third term. Matteo's call — flag only.

**V4-7 — OPEN DECISION (c/f) · "structural anchors" vs ACF's "anchors".** Still pending; now used twice ("structural anchors", "the property propagates" relies on the anchor role).

**V4-8 — MINOR · "grounding" definition placement vs outline.** Outline places the terminology home in the Model section (5.2); the definitional sentence ("Grounding, as used in this thesis, means…") now lives in 5.1. Likely fine — 5.1 is where the argument needs it — but outline/5.2 should not re-define it differently; note for the 5.2 writing session.

### Paragraphing

**V4-9 — MINOR · move 3 paragraph does three jobs** (definition; the forcing emphasis; the precedent). It holds together, but if it grows at all, the precedent sentence is the natural split point.
**V4-10 — GOOD:** the seven-paragraph shape now matches the five moves + two asymmetry paragraphs + transition; the two-sentence hinge paragraph ("Grounding, however, needs its vocabulary…") works.

### Carried minors
- Comma-splice repair choice in move 1 (semicolon inserted — approve or reword).
- Cites: sleep paper placeholder ("(cite sleep paper, maybe find other examples.)"); ProtoMF paper cite absent at the post-hoc-naming sentence (ref inserted, cite still warranted); koren/samuelson decision open.
- "at no cost to accuracy" must be verified against the sleep paper before it stands.
- "embedding spaces" (plural) vs "the embedding space" — pick one.
- "as is common in MF models" — MF abbreviation usage consistent with Ch 2? verify at Ch 2 prose time.

## Phase 3 — discussion outcomes (2026-08-21)

- **V4-6 RESOLVED (decision):** three-layer hierarchy adopted — *characteristics* = everything an item presents to a person (incl. look, photo-capturable); *attributes* = the recorded, coarse subset in the catalogue; *interpretable feature* = an attribute's data form in the model. Section stays on "characteristics" at the perception level; model-level statements route through attributes ("attributes enter the model as interpretable features"); one definitional pass at first use; acknowledge the characteristics⊃attributes gap in a clause. **Refines the Ch-1 attribute-vs-feature convention — Ch 1 definitional sentence + outline terminology note gain the third term (pending).**
- **V4-5 RESOLVED (Claude's pick, approved):** "groups of items with shared attributes" / "groups of users with shared taste", short forms "attribute groups" / "taste groups", lowercase, no coinages.
- **V4-7 RESOLVED (decision):** "prototypes" stays the only noun; *anchor* used strictly as the role/verb ("the prototypes anchor the embedding space"); ACF keeps the noun "anchors"; Background's ACF passage notes the kinship in one clause (pending, for Ch 2).
- **V4-1..V4-6 APPLIED DIRECTLY (Matteo's explicit instruction, overriding the no-paste default):** six passages rewritten by Claude in the chapter — move-1 layer rebuild (anchor-as-verb, "embedding space" singular), move-2 first sentence, move-4 payoff sentence (full group forms), "already written down" topic sentence, part-two attribute-layer insertion + short forms ("attribute groups"/"taste groups"). Mirrored to samples/fi-why_review_v4_result.tex.
- **REVERTED same session (Matteo):** moves 1, 2 and 4 back to his own wording — only the part-two refactors kept as AI-worded ("already written down", attribute-layer insertion + "attribute groups", "taste groups"). Consequences: V4-1 reopens but is RESOLVED BY DECISION instead — this intro treats ProtoMF as UI-ProtoMF (both sides prototyped), under which his original claims are correct; V4-2 (move-2 first sentence) and V4-3 (broken payoff sentence, capitalized coinages) REOPEN for his own rewrite, now to use the kept "attribute groups"/"taste groups" forms.
- Still open: cites (sleep paper + bib entry, ProtoMF paper at post-hoc naming, koren/samuelson decision), "at no cost to accuracy" verification, semicolon-splice approval, section title rename, deferred items (Ch 1 terminology third layer, ACF kinship clause, 5.2 grounding non-redefinition).
