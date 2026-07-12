---
date: 2026-07-12
time: "12:24"
phase: 4
---
# Length-channel refund hypothesis + the "hidden effects" thesis section

*Captured under the verbatim-capture rule — worked through at the dc01 SC.1b-delta
gate sitting (adversarial pass on the re-hosted fI concept) and explicitly adopted
by Matteo ("yup I like that framing, lets protocol and test the hypothesis"). Two
things live here: a registered, testable cross-stage hypothesis, and a thesis-
structure decision. Companion findings: F-DC01-08 (user-affinity gauge), F-DC01-09
(angular twin tax) in the scrutiny findings ledger.*

## The build-up: bottleneck vs. tax

In fI-ProtoMF the item side scores through a cosine, and the fleet is bias-free. A
cosine only sees *direction* — vector length is discarded entirely. So **direction
is the only channel the item side has to say anything at all**: popularity, taste,
individuality — everything must be encoded as "pointing in a slightly different
direction." Call this the **bottleneck**.

The bottleneck is shared with the host: I-ProtoMF also scores angle-only, and it is
a published model that works — a free d-dimensional direction is an enormous
budget. The bottleneck alone is survivable; it sets the rules of the game, the same
rules for both players. It never explains an fI-vs-host difference on its own.

What is fI-specific is the **tax**: fI's directions are not free — they are
anchored to shared ingredient words. Signature twins (67.5% of the V1 catalog
shares its 5-field signature) have compositions q = m + e_i and q = m + e_j: the
same big shared five-row mass m, differing only in their small personal ID rows.
Their angular separation goes like ‖e_i − e_j‖ / ‖m‖ — the shared mass drowns the
individual difference first-order. Cruel interaction: with `max_norm` on, the ID
row is capped while m sums *five* capped rows — the squashing becomes structural.

The tax is an **ids-arm phenomenon**: in the noid arm twins are literally the same
vector (the known equivalence-class ceiling — nothing to squash); in the ids arm
the ID row is the ingredient whose *job* is twin separation, and it must do that
job through the angle bottleneck, against the shared mass. Consequence for the
ablation reading: the ids-vs-noid gap *understates* what item-level memory would
buy in an unconstrained model, and in the popularity strata the shortfall wears
the same costume as popularity blindness → gap-interpretation rule sub-component
(1b), "ID signal present but angularly unrealizable."

One-liner: *the bottleneck sets the rules of the game — the same rules the host
plays by; the anchoring is fI's handicap under those rules; the ID row is the
ingredient that has to overcome it.*

## Matteo's question, and the answer that became the hypothesis

**"Does a fUfI model, once working, have the ability to leverage vector length?"**

Yes — and that is exactly what makes the merge stage more than a formality. The
double-tie score has two halves, u*·t̂ + û·t*. The cosine parts stay angle-only
forever (that is what makes them prototype-shaped). But t̂ = W·q_i is *linear*:
double the vector, double that half of the score. So at the merge, each side's
length flows into the score **through the other side's half**. For twins, the
linear half sees W(e_i − e_j) directly — no division by the shared mass; the
squashing disappears in that half. Popularity can again be carried by simply
growing the ID row's length.

So the angular tax is **stage-local**: fI pays it (and fU will pay its mirror on
the user side); the fUfI merge dissolves it.

But length is a channel already convicted twice, and both charges return with it
at the merge: (1) **the cold magnitude handicap** — in a linear half, warm items
carry a learned magnitude bonus that cold items structurally lack after the ID
drop (first-round adversary P5); length expressiveness warm = calibration handicap
cold. (2) **The Steck exposure** — the linear half is dot-product-trained, no
trains-through-cosine protection; the "unprotected projection half" risk that
dissolved in the re-host re-materializes at the merge, together with its antidote,
the parked read-out 4 (exact norm-free linear-half attribution, F-DC01-07).

## The registered hypothesis (length-channel refund)

The single-sided stages (fI, fU) buy maximal grounding at the price of angle-only
expressiveness — a price we **measure** (twin-pair angle distributions,
ids-vs-host gap, popularity strata). The merge restores the length channel on both
sides — so if the tax is real, **the merge should recover accuracy specifically on
twins and popular items**, and the per-stage measurements will already exist to
show that it did, and why. The tax measured at fI becomes the predicted refund at
fUfI: a mechanism-level narrative arc for thesis part one, not a defensive caveat.
(Sibling of the earlier ID/twins/popularity braid hypothesis, vault
2026-07-12_0006 — the braid predicts cold-side degradation from warm ID-reliance;
the refund predicts merge-side recovery from the measured angular tax.)

## The thesis-structure decision: a "hidden effects" section

Matteo's own words (lightly cleaned):

> "All of these open issues and questions, like the braid as well from the
> protocol entries, will likely become its own section — something like *model
> mathematics, hypothesis risks, and experiments* (we will find a way better
> structure and name) — a section that runs through all of them one by one and
> highlights them. So a reader **first gets presented with the algorithm, and when
> he thinks that he understood it, he gets challenged by us**: we show these
> hidden effects in the model, and how and when they occur, and what their effect
> is, positive or negative. And maybe a careful reader will read carefully, bring
> his understanding to a new level, and even come up with his own effects that are
> hidden in our numbers but not measured yet — and maybe he will experiment."

Design intent: the section is a *pedagogical challenge ladder* — algorithm first,
then the hidden mechanisms one by one (each with: what it is, when it occurs,
sign of its effect, how we measured or bounded it), inviting the reader to find
the next one. Current inventory of candidate members (grows as scrutiny
proceeds): the share-identifiability gauge (F-DC01-06 / vault 2026-07-11_2310),
the user-affinity gauge (F-DC01-08), the angular bottleneck & twin tax
(F-DC01-09), the rank-inert baseline + catalog common mode, loudness vs.
omnipresence (F-DC01-05), the ID/twins/popularity braid (2026-07-12_0006), the
length-channel refund (this entry), the cold magnitude handicap and Steck
exposure of linear halves (merge stage).

Standing restriction carried over: scrutiny-phase measurements are working
evidence only — the section's thesis treatment gets dedicated experiments
designed at writing time (learnings-ledger rule, 2026-07-12).

[[phase-4]] [[decisions]] [[explanations]] [[experiments]]
