---
date: 2026-06-16
time: "18:07"
phase: 2
tags: [thesis/data, thesis/fu-protomf]
placed: 2026-08-20
---
# Attribute-source positioning: item-first, user candidate-dependent, context dropped

For the feature-aware ProtoMF extension, prioritize **item attributes** as the primary integration target — rich on H&M, and item prototypes can ground directly in feature space, giving prototypes intrinsic meaning. **User attributes** are **candidate-dependent**: not ruled out from integration into the recommendation algorithm, but when a candidate uses them, prefer intrinsic integration over post-hoc so they shape explanations by-design (thin H&M user features may limit richness). **Context (rating-relevant) attributes** are dropped for now — they are per-interaction, not per-entity, and don't map cleanly onto entity prototypes — with the caveat that S7 §3.2.3/3.2.4 show context↔side-information conversion, so context could be re-evaluated per use case if it can be folded into a user/item entity (mention, don't hard-exclude). The prioritization is **interpretability-driven**, reinforced by (not solely based on) the H&M data asymmetry. See [[decisions]]; relates to the design-principle entry on intrinsic vs post-hoc interpretability.

[[decisions]] [[explanations]] [[phase-4]] [[phase-2]]
