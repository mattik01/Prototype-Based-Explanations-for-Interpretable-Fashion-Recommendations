---
date: 2026-07-12
time: "21:14"
phase: 4
tags: [thesis/data]
placed: 2026-08-20
---
# Dataset scope for the feature-aware phase: ml-1m in, amazon2014 out (no features)

Decision (user, at the dc05 SC.2→SC.3 boundary): the feature-aware testing was
tailoring itself to H&M alone — **ml-1m joins as a second quantitative
testbed**, and **amazon2014 is discarded from the feature-aware scope,
justified honestly by missing features**: the dataset as used by ProtoMF is a
ratings-only file (`ratings_Video_Games.csv`); no item metadata exists in the
pipeline, so a feature-aware comparison has no object there. Amazon stays
exactly what it already is — replication context. This justification is
considered defensible in the thesis.

Reasoning for ml-1m, beyond "the paper used it": ml-1m and H&M V1 probe
**opposite ends of the same instrument set** — ml-1m has heavy histories
(every user ≥ 20 ratings) over a coarse vocabulary (18 genres + year), i.e.
the crowding/omnipresence/B(t) regime, while H&M V1 (median 5 purchases, a
quarter of users at the 3-row floor) probes the thin-user regime where fU's
central bet lives.

The "genre bag problem" (movies carry variable-size genre sets, breaking the
repo's fixed one-value-per-field `FeatureEmbedding` layout) was checked
against the LightFM paper (B5, verified in the PDF §2.2/§4.1): **LightFM
never had to handle it — variable-size feature bags are the model's native
input** (entity = sum of however many feature embeddings it has). The
fixed-width layout is our implementation artifact, optimized for H&M; the
faithful general implementation is the variable-length index+weight (bag)
layout that dc05's `HistoryFeatureEmbedding` already uses on the user side.
Also noted from B5 §4.1: LightFM's MovieLens arm used MovieLens-10M joined
with the **Tag Genome** (1,030 tags at relevance ≥ 0.8), not genres alone —
so our genres+year setting is deliberately the coarse-vocabulary regime, and
a Tag-Genome join onto ml-1m movieIds is recorded as an unstacked upgrade
option (external download + a relevance threshold, production-logic
flavored — out of scope now).

Execution route: gated **charter amendment** (S0.6 C2 datasets, C5
per-dataset field set — proposal: `genres` bag + `year` banded, title
excluded as identity-like; C6 ml-1m reference-fleet rows added lazily, i.e.
submitted only when a run plan first cites them), plus bag-layout support in
the feature builders as SC-stage implementation work with its own checks.
Dataset lists per variant remain pinned at each candidate's SC.5/SC.6.

[[decisions]] [[data]] [[phase-4]] [[evaluation]]
