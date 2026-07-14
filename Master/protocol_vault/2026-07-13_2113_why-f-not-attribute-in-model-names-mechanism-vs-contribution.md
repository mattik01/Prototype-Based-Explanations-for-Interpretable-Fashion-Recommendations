---
date: 2026-07-13
time: "21:13"
phase: 4
---
# Why the `f` in fI-/fU-ProtoMF stays "feature", not "attribute" — mechanism vs contribution naming

*(Verbatim-capture entry — terminology justification from the thesis-outline session, user requested protocol after adversarial check. Complements the attribute/feature convention: "attribute" = human-readable catalogue property (semantics, prose); "feature" = encoded, model-facing input row (mechanism).)*

**DIRECTIVE (Matteo, 2026-07-13): in the thesis this is a SINGLE SENTENCE** — one line in the definitional/terminology passage justifying the `f` in the model names, not a paragraph or section. The entry below preserves the full argument so that sentence can be written (and defended if asked); the thesis gets only the sentence.

## The justification (Matteo, near-literal)

fI-ProtoMF — meaning *feature*-Item ProtoMF — is a better name than *attribute*-Item ProtoMF, "because we can feed whatever we want in there. If we use the ID version, a database ID is arguable whether it is an attribute of the item. We could also feed something else in there entirely — something correcting bias in some way. **The model does not know and does not care.**"

## Why it holds (checked together)

**Names should describe contracts, not intentions.** The composition mechanism's contract is: sum whatever encoded input rows arrive (q = Σ rows). "Feature" in the pinned model-facing sense describes exactly that contract. "Attribute" would over-promise — it would claim the inputs are semantically meaningful catalogue properties, which the ID row already violates today (a database handle is not a catalogue property), and which any future engineered row (e.g. a bias-correcting input) would violate further.

**The altitude resolution:** mechanism = *feature-composed* (hence the `f` in model names, and the code's `feature_*` naming needs no apology); contribution = *attribute-aware* (hence Part I's title and §1.3). *Choosing* attribute-derived rows is what turns the agnostic mechanism into an interpretable model — the interpretability lives in the choice of rows, not in the summation.

**Retroactive confirmation from dc02:** the codebase already implicitly follows this rule — dc02 is `attr_item_proto`, and there "attr" is correct because that design anchors prototypes in attribute space *by construction* (enforced grounding); the semantics are baked into the mechanism, not chosen by the operator. Rule made explicit: **`attr` = semantics enforced by the mechanism; `feature` = semantics-agnostic composition.**

**One honest caveat for the prose:** the agnosticism is *architectural*, not behavioral — training shapes what each row comes to mean. The model doesn't care what you feed it, but it decides what to make of it.

**Future-proofing side effect:** if an engineered non-semantic row is ever added (hidden-effects work flirts with such ideas), the naming scheme accommodates it without a retcon.

[[phase-4]] [[decisions]] [[thesis-writing]]
