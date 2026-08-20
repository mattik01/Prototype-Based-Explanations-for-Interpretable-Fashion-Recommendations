---
date: 2026-07-13
time: "18:17"
phase: 4
tags: [thesis/background, thesis/conclusion]
placed: 2026-08-20
---
# Core-level, modality-light — the positioning claim, its price, and the specialization fork

*(From the S2 survey read (Zhang & Chen 2020), §3.2–3.3 — Matteo's observation, worked through interactively and agreed. Captured fully per the verbatim-capture rule: this is a thesis-positioning-section entry plus a future-directions hook, and the framing must regenerate without the conversation in context.)*

## Matteo's observation (the seed)

Reading the survey, method after method turns out to be built *for* a particular data situation. FLAME needs reviews to mine aspect-sentiments from; EFM/HFT/SULM/FacT need a review corpus plus extraction tooling; others need social graphs, knowledge graphs, or images. Our approach isn't like that: it plugs into the core factorization machinery itself and asks only for generic item attributes. This limits us a bit (we can't exploit the riches a specialized setting offers), but it also positions us at a more fundamental level of the stack than these tailored methods. Belongs in a positioning section; and in future directions, the question of whether committing to a specialized scenario would open up entirely new additions to the proposal, or even alternate ways of achieving what we set out to do.

## The agreed sharpening: "modality-light", NOT "dataset agnostic"

The claim an examiner would attack is "dataset agnostic" — and they'd win: we still need item features that exist, are reasonably clean, and are in a vocabulary a user can perceive (the R7 requirement). Give us IDs only and we have nothing — which is precisely why amazon2014 was dropped from the charter (no features). The defensible phrasing: **we require only what a catalogue already has** — structured item attributes exist almost everywhere because the shop needs them to operate, whereas the dominant explainable-rec strand requires a free-text review corpus that H&M and MovieLens (two of the most standard datasets in existence) simply don't have. The field bought its explainability with a modality that is often absent; we buy ours with one that is nearly always present.

## The dial (the positioning figure, essentially)

Not a binary but a **generality↔richness dial** over input modality:

- **IDs only** — universal, but no intrinsic explanation possible at all (the vanilla-ProtoMF end; naming stays post-hoc).
- **Structured attributes** — near-universal, modest vocabulary. **← us.**
- **Reviews / images / knowledge graphs** — rich vocabulary (sentiment, user-side aspect preferences, visual detail), narrow applicability.

## State the trade before someone states it for us

It is a trade, not a free win: reviews carry *sentiment* and *user-side* aspect preferences ("this user cares about battery life") — structured attributes carry neither, so review-based methods can explain something we genuinely cannot. Our item-first commitment already concedes this ground on principle (taste must be revealed, not stated — [[2026-07-11_2350_interactions-explained-by-item-features-taste-revealed]]), so presenting it as **applicability bought with explanation richness** costs nothing and makes the claim credible.

**The evaluation cost (name it ourselves):** dataset-tailored methods get a free ground-truth explanation signal — the review text *is* the ground truth, which is exactly how S2 §4.3 does offline explanation evaluation (BLEU/ROUGE against the user's actual review). We have no such signal, so S2 §6.2.1's "there still lacks a usable offline explainability measure" hurts us considerably more than it hurts them. Generality has an evaluation cost — and that cost is one of the standing reasons the quantified-interpretability block exists.

## Composition vs vocabulary (what is general, what is domain-specific)

The mechanism split that makes this more than a positioning paragraph: **the composition is the general thing; the vocabulary is the domain-specific thing.** An item is a sum of feature rows; nothing in that mechanism cares what kind of thing a row is. The C1-lineage entry already says this about dc03 — a CNN bringing images in is "one more row type in the sum." So the architecture slides rightward along the dial *without changing shape*. This upgrades the ml-1m charter amendment too: two domains (movies, fashion), disjoint attribute vocabularies, same machinery, no code path that knows which is which — that is the **empirical demonstration of the generality claim**, not merely a robustness check. Worth saying explicitly at write-up.

## Future-directions fork (two siblings, both kept)

1. **Narrow (new modality = new row type):** specialized inputs enter the existing design unchanged — image rows, review-derived rows. A *prediction*, hence testable, not a hope.
2. **Broad (Matteo's — specialization opens new design space):** committing to a specialized scenario might unlock qualitatively new extensions or *alternate routes* to grounded prototypes — e.g. FLAME-style aspect-sentiment mining could ground **user-side** prototypes in a way our current data never can. Not a new row type; a different road to the same goal.

**Richer grounding via extra modalities (Matteo's addition, same session):** the deeper promise behind both forks — given extra modalities *and the computational structures to take them in* (an aspect-extraction pipeline, a CNN for images), the same composition mechanism could ground prototypes **richer** than catalogue attributes allow: a prototype readable not just as "black + jersey + womenswear" but against mined aspect vocabulary ("comfort", "fit", "quality-for-price") that catalogue fields never contain. **Caveat to address before using it: aspects are a user-specific thing.** They are mined from what users wrote, so an aspect profile is partly a property of the *user population's attention* (what reviewers happened to care about and how they felt), not of the item alone — unlike a catalogue attribute, which is a fact about the item. Grounding prototypes in aspects therefore imports user-side subjectivity (and its sentiment, sparsity, and popularity skews) into what we currently treat as objective item-side vocabulary; any such extension must say which side of that line each grounding token lives on.

[[decisions]] [[phase-4]] [[paper-understanding]] [[explanations]]
