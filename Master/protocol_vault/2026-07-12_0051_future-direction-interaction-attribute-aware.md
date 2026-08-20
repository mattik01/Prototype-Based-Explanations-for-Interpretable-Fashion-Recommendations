---
date: 2026-07-12
time: "00:51"
phase: 4
tags: [thesis/conclusion]
placed: 2026-08-20
---
# Future direction: interaction-attribute-aware prototypes (the next natural extension after item-attribute grounding)

**The thought (for the thesis Future Work chapter):** the current lineage grounds prototypes in *static item attributes* (dc01 `feature_item_proto` etc.). If that succeeds, the next natural *conceptual* step is **interaction-attribute-aware** prototypes — grounding/conditioning on features of the **interaction event itself**, not just the item. H&M carries rich per-event context (transaction date → season/time, `sales_channel_id`, price paid) that the current design **deliberately drops** (the H&M data-asymmetry stance: context is per-event, no entity branch). Moving from static item features → dynamic interaction context is the obvious escalation: same prototype-grounding idea, applied one level up to the (user, item, context) event.

**Why deferred, not now:** it adds a context branch / event-level modeling that is out of scope for the master's thesis and only makes sense *conditional on* the item-attribute grounding paying off first — hence a Future Work pointer, not a candidate. Keep it earmarked; don't let it bleed into the current scope.

**Prior-art pointer (added 2026-07-13, S2 read):** if this thread wakes, the citation for *readable conjunction units* is **TEM — Wang et al., "Tree-Enhanced Embedding Model", WWW 2018** (S2 §3.4, p43): GBDT-learned cross-feature rules (e.g. age×genre×price) become embedded tokens, each human-readable as a rule path; attention over rule tokens scores and explains. The design fork it names: **rule-tokens vs raw-attribute-tokens** as the composable unit. Caveats logged at triage: its explanation vehicle is attention (E4 contested), it grounds inputs rather than prototypes (no hybrid bridge), and its interesting crosses need a real user-feature side. Deliberately *not* fetched — cite-in-RW-level only.

[[decisions]] [[data]] [[phase-4]]
