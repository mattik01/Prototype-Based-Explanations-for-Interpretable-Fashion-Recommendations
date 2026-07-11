---
date: 2026-07-11
time: "23:50"
phase: 4
---
# Why an interaction is explainable by item features but not by user features — taste is revealed, not stated

*(Verbatim-capture entry from the dc01 SC.1b gate detour — thesis-motivation material; the user asked for exactly this argument to be protocolled. It grounds the item-side-first strategy and the C1 lineage outline.)*

## The core argument

An interaction happens when **item content meets user taste** — the bilinear score literally has that shape: taste-vector · content-vector. The two causal ingredients have opposite epistemic status:

- **Item content is observable and recorded.** What the garment *is* — the catalog writes it down in 426 words (departments, colours, patterns…). Composing items from features (dc01) grounds that side in the actual ingredient of the interaction.
- **User taste is latent and unrecorded.** Users don't state their taste; mostly they *couldn't* state it if asked — nobody can articulate their own taste function. Stated personal attributes (age, club membership, newsletter frequency) are a very loose proxy: enough signal for cold-user popularity fallbacks (what the Kaggle field used them for), nowhere near enough to *ground an explanation in*. Grounding user prototypes in demographics would ground them in a proxy so loose the grounding would be a lie ("recommended because you are 34 and get the newsletter" — R7-wrong, privacy-adjacent).

**The refinement that completes it:** taste is not *unobtainable* — it is unobtainable as a **stated feature**, but richly **revealed** through behavior (revealed preference). The purchase history is a high-fidelity record of taste, and capturing it is exactly what CF already does: the user embedding IS learned, revealed taste. The user side's problem was never *capture* — it is that revealed taste arrives **without labels**: a latent vector with no vocabulary attached. Therefore: **don't hunt for taste in user attributes; express revealed taste in item vocabulary** — describe a taste community by what its members buy.

## The four-component honesty inventory (dc01 as it stands)

The UI score has four contributing values; in dc01, their explainability splits: **t\*** (item's prototype activations) — grounded, feature-decomposable; **û** (user's affinities over named item prototypes) — derived-legible for free; **t̂** (item's appeal profile over user communities) — feature-attributable (linear-half read-out) but pointing at ungrounded units; **u\*** (user's community memberships) — ungrounded, post-hoc-named as in the host. User principle, adopted: *whatever contributes to the score cannot be left unexplained* — the current honest answer is disclosure (S0.2's binding score-share reporting); the lineage outline (companion entry [[2026-07-11_2351_thesis-outline-c1-lineage]]) is the plan for closing it for real.

## The supporting data asymmetry (from the same discussion)

In pure CF, *both* embedding spaces are behavior summaries — an item's CF embedding encodes *who bought it*, not what it is; the item side's legibility was always **borrowed** from the catalog's external vocabulary. dc01 moves that borrowing inside the model. The user side has no equivalent rich vocabulary to move inside (H&M: ~3 thin fields, ~15 usable values, postal code hashed/identity-like) — hence item-side-first is not a preference but a data-forced strategy, and the user side's eventual grounding must route through item vocabulary (history-composed representations / projected community profiles).

[[decisions]] [[explanations]] [[phase-4]]
