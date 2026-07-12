---
date: 2026-07-12
time: "23:13"
phase: 4
---
# ml-1m raw-bag popularity coupling: first live magnitude — baskets carry ~2× the catalog's token mass

**Context.** S0-build extension, component (d) gate (model-grade ml-1m `item_features.csv`,
genres + Tag-Genome tags @0.8 as raw indicator bags — the C5 recipe ratified at the S0.5
addendum). A test expectation failed in the right way: the check assumed per-user vote mass ≈
the catalog mean token count (12.3 tokens/movie); the real basket-weighted value is **24.27**
(min 11.0, max 48.8 across users). Not a bug — the S0.5-addendum popularity-coupling channel
(token count vs train popularity, Spearman ρ = 0.672) showing up live in the exact tensors the
model will consume. Pinned as `dc_checks/s0/t10`'s check with the reasoning in-file.

**The explanation, as worked through at the gate (captured per the verbatim rule):**

*What the number is.* When fU composes a user, each rated movie drops its tokens into the sum,
and with raw bags every token weighs 1. So a movie's "vote mass" in your representation = how
many tags it has. The catalog average is 12.3 tokens per movie. But averaging over what users
*actually rated*, it's 24.3. Same recipe, two different populations: movies-as-they-exist vs
movies-as-they-get-watched.

*Why the gap exists.* The Tag Genome is built from community activity — tagging, rating,
reviews. Popular movies accumulated more of that, so they cross the 0.8 relevance threshold on
more tags. That's the ρ = 0.672 coupling. The new reading gives the *magnitude on the actual
model input*: the coupling roughly **doubles** the token mass flowing through baskets compared
to a popularity-neutral world.

*What it implies for the model.* Inside one user's basket, a blockbuster with 48 tags outvotes
an obscure movie with 3 tags sixteen-to-one. So q_u tilts toward the vocabulary of the user's
*popular* choices. Two users whose distinctive difference lies in their obscure tail picks will
look more similar than they are, because both representations are dominated by the well-tagged
mainstream they share. And this **compounds** with F-DC05-02: the training loss already sees
heavy users more often (interaction-weighted sampling), and now within every basket the popular
movies also speak louder. Both pressures push the learned prototype vocabulary toward the
popular region — a direct headwind for exactly the thing dc05 bets on: individuating users
beyond mainstream taste.

*What it does NOT imply.* Not a bug, not a reversal of the raw-bags gate decision. (1) Raw bags
are the published LightFM recipe — the anchor was the point. (2) Popularity weighting isn't
automatically bad for accuracy — popularity is a strong prior; the channel may well HELP HR@10
while hurting the interpretability/individuation story. Which happens is what the committed
SC.8 instruments decide (two-weightings readout, ESS/mass-share, per-history-band diagrams).
Per-movie normalization (option B) stays the evidence-triggered fallback if the instruments
show the channel dominating.

**Disposition.** The ~2× magnitude joins the dc05 hidden-effects record alongside the ρ = 0.672
correlation (the standing must-mention for every ml-1m fU presentation now has a measured size,
not just a correlation). Scrutiny working basis only — thesis quotes require the dedicated
experiment designed at writing time (standing rule).

[[data]] [[experiments]] [[phase-4]] [[decisions]]
