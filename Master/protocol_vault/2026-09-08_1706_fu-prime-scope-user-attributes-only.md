---
date: 2026-09-08
time: "17:06"
phase: 4
---
# fU′ (candidate prime) scoped to user attributes only; past-event attributes deferred; query context parked with the merge

**Decision (Matteo, 2026-09-08).** After mapping the full idea space (user attributes: three
shapes, vault 2026-09-08_1649; past-event interaction attributes: taxonomy, 2026-09-08_1650;
query-time context at the tie: 2026-09-08 entry above), fU′ is limited to **user attributes
only** — a single-variable candidate per the dc06 discipline. Preferred shape: the two-layer
side term (shape 2). Precondition: a cold-user split (S0-level), because on the warm hm split the
expected gain is ≈0 and the candidate's whole story is new users.

**Why past-event attributes are deferred, not included.** They are buildable WITHOUT a
dataloader change — as per-event multipliers or as event words, both baked into the history
buffers at build time, since the builder already reads the full train file (date, price,
channel per row). But on hm_1_month: recency is nearly flat within one month, season is a
constant, weekday is noise; what remains is channel and discount as event words (carrying the
binding loss of the count bottleneck) and repeat purchase (already in the mean). On this split
the past-event part would be a no-op dressed as a feature. Revisit only if hm_3_month/full
enters play.

**Code implication if they ever come in (recorded now).** The leave-one-out pooling landed
2026-09-08 (commit 89a1818) subtracts the positive's contribution using the per-ITEM token table
and a uniform 1/|H| weight. With per-event multipliers the subtraction needs the weight of THAT
purchase → a per-(user,item) lookup (sparse table keyed by pair), not per item; event words as
tokens need the positive's event tokens subtracted too. Without this the hygiene fix becomes
wrong in exactly the runs where it matters.

**Data preparation implication for fU′ itself.** The processed folders carry no user-features
file. The hm splitter must emit one (age band, club status, newsletter frequency, activity flags;
postal code EXCLUDED as identity-like); the ml-1m side would emit age and occupation (gender
excluded, stated as excluded). Then a builder, an injection branch and the factory shape — the
fI item-feature pattern one level up. No dataloader change.

**Query-time context** (season/channel of the prediction event): the only family that needs the
dataloader (positive's context travels with the batch) and an evaluation-protocol change; parked
with the merge cycle, where the tie hosts it naturally.

**Ordering unchanged.** fU 100-trial wave (post leave-one-out) → cold-user split as scrutiny work
→ fU′ /design-candidate cycle, shape 2, both datasets, after or alongside the merge.

[[decisions]] [[phase-4]] [[evaluation]]
