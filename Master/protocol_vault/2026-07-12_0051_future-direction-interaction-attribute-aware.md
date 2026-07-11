---
date: 2026-07-12
time: "00:51"
phase: 4
---
# Future direction: interaction-attribute-aware prototypes (the next natural extension after item-attribute grounding)

**The thought (for the thesis Future Work chapter):** the current lineage grounds prototypes in *static item attributes* (dc01 `feature_item_proto` etc.). If that succeeds, the next natural *conceptual* step is **interaction-attribute-aware** prototypes — grounding/conditioning on features of the **interaction event itself**, not just the item. H&M carries rich per-event context (transaction date → season/time, `sales_channel_id`, price paid) that the current design **deliberately drops** (the H&M data-asymmetry stance: context is per-event, no entity branch). Moving from static item features → dynamic interaction context is the obvious escalation: same prototype-grounding idea, applied one level up to the (user, item, context) event.

**Why deferred, not now:** it adds a context branch / event-level modeling that is out of scope for the master's thesis and only makes sense *conditional on* the item-attribute grounding paying off first — hence a Future Work pointer, not a candidate. Keep it earmarked; don't let it bleed into the current scope.

[[decisions]] [[data]] [[phase-4]]
