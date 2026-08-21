# Review v5 — Section 5.1 `fi-why`

**Date:** 2026-08-21 · **Skill:** /thesis-review (fifth invocation) · **Git state:** HEAD 9d8bdd4, uncommitted on `feat/scrutiny`
**Scope (Matteo):** sentence-level flow + word choices across the whole section; structure accepted as sound; "would you write anything differently?" — candidate phrasings included as suggestions, his to apply.
*LLM-usage-disclosure artifact — do not delete.*

## Phase 1 — mechanical fixes applied
1. "Prototypes and Embeddings … the collaborative signals." → "Prototypes and embeddings … the collaborative signal." (mid-sentence capital; singular to match the very next sentence's "The collaborative signal")
2. Rewrap to 78 chars (standing request)

## Phase 2 — findings (sentence flow + word choice; suggestions in quotes are candidates, not edits)

**P1 (move 1)**
- "mirror that shape when they compute" — "when" reads conditional; suggest "mirror that shape by computing a recommendation score as the product…". MINOR
- "the prototypes **just** act as the structural anchors" — "just" deflates a load-bearing claim; drop it. Also missing link between the two sentences: the similarities are computed *from* the embeddings — one clause ("Each item and user still owns a learned embedding, from which these similarities are computed; the prototypes act as…") joins the layers. MINOR
- "embedding spaces" plural — fine under the UI reading (two spaces) but decide once; P4/P6 talk as if one mechanism. MINOR

**P2 (move 2)**
- "Most MF methods share this **characteristic**." — word now collides with the section's own terminology layer (characteristics = item-property level). Suggest "share this opacity" / "share this property". MAJOR (term consistency, self-inflicted by the new hierarchy)
- "To **leverage** its prototypes for interpretability, ProtoMF introduces post-hoc methods to **essentially** name them" — "leverage" is business-speak, "essentially" is filler; the concrete mechanism is more vivid: "To make its prototypes interpretable, ProtoMF names them after training, by inspecting their most similar items (Section~\ref{sec:bg-protomf})." MINOR
- "there are issues connected to post-hoc readings of opaque objects" — flabby existential; suggest "post-hoc readings of opaque objects are problematic (Section~\ref{sec:bg-iml-intrinsic}), and where possible, intrinsic methods are preferred." MINOR

**P3 (move 3)**
- "We aim to close **that step**" — "that step" has no antecedent since the one-step-short phrasing left the text; either restore a step antecedent at P2's end or say "that gap". MAJOR (flow)
- "We want to emphasize that the vocabulary might consist of additional information **about elements**" — "elements" vague (items? users?); "We want to emphasize that" is throat-clearing. Suggest: "The vocabulary may well add information the model would not otherwise see, but we impose it as a constraint for interpretability, not primarily as a contribution to predictive power." MINOR
- "where it was applied **likewise** for interpretability" — "likewise" sits oddly; suggest "where constraining a model's internals to an expert-defined vocabulary served interpretability at no cost to accuracy". (Accuracy claim still to verify vs the sleep paper.) MINOR

**P4 (move 4)**
- Payoff sentence still broken + capitalized coinages (c/f V4-3): "A recommendation is the user's taste by similarity to Taste-groups, matched with an item's characteristics by similarity to Item-Characteristics-Groups." Candidate with the agreed forms: "A recommendation becomes a match between the two ingredients: the user's affinities to taste groups meeting the item's similarities to attribute groups." MAJOR
- "and **the** property propagates to the score" → "and this property propagates". MINOR

**P5 (hinge)**
- "and **this is addressed** throughout the lineage of models in this thesis" — vague "this", passive "addressed" weakens the hinge; suggest "and this asymmetry shapes the entire lineage of models in this thesis." MINOR

**P6 (item)**
- "to them, an item simply is the characteristics **shown about it**" — "shown about" is off; suggest "the characteristics shown to them" or "presented". MINOR
- The catalogue sentence runs three clauses; natural split after "as attributes": "The catalogue records a coarse subset of these characteristics as attributes. To the extent that these attributes enter the model as interpretable features, they provide…" MINOR

**P7 (user)**
- "captures this, **but as stated, as** an unlabeled, uninterpretable taste vector" — double "as" stumble; "unlabeled, uninterpretable" is a redundant pair (the second is the consequence of the first). Suggest "captures exactly this — but, as established above, as an unlabeled taste vector." or plainer "…captures this, yet only as an unlabeled taste vector." MINOR
- "**connect to** privacy concerns" → "raise privacy concerns". MINOR

**P8 (transition)** — clean; no findings.

## Phase 3 — discussion outcomes (2026-08-21)

- P3 opener discussed separately; Matteo asked for a post-hoc→intrinsic replacement phrasing; option chosen (Claude's pick): "Instead of naming prototypes after training, we ground them before it."
- **P2–P7 suggestions applied on Matteo's instruction — 10 by Claude, 3 already made/reworked by Matteo concurrently** (his: "share this opacity"; P3 opener → "We aim to address these concerns by grounding the prototypes"; precedent sentence, now claiming "increased accuracy as a side effect" — verify vs the sleep paper). Claude-applied (AI-worded until reworked): post-hoc naming concretized; problematic-readings sentence; vocabulary-as-constraint sentence; "this property"; payoff sentence rebuilt with agreed group forms — V4-3 CLOSED; hinge "asymmetry shapes"; "shown to them"; catalogue sentence split; "yet only as an unlabeled taste vector"; "raise privacy concerns". P1 and P8 remain Matteo's wording untouched.
- **New (Matteo, during session): a bracketed planning-notes paragraph** after move 3 (performance/cold-flexibility discussion; grounding-grounds-the-whole-space / prototypes-possibly-obsolete question; vocabulary-as-entrypoint highlight) — drafting layer, not reviewed; spelling inside fixed mechanically (address, increased, decision, include, that, obsolete, potentially, ProtoMF's).
- Post-edit stumble to resolve (Matteo): "A user approaches an item...; to them, an item simply is the characteristics shown to them." — "to them" now appears twice (collision of his clause and the applied fix); drop one.
- Still open: P1 suggestions (drop "just", "by computing", linking clause, spaces singular/plural) — not applied by instruction; cites (sleep paper + bib, ProtoMF paper, koren/samuelson); "at no cost to accuracy" verification; section title rename; deferred Ch1/Ch2/5.2 items.
