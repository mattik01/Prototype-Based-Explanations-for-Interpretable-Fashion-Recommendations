---
date: 2026-06-16
time: "20:57"
phase: 2
---
# Positioning: ProtoMF vs part-prototype models (PPMs) — shared principle, differing axes (S5)

**Shared (the principle):** Both represent an object by its **similarity to a set of learned prototypes**, and both are **ante-hoc/intrinsic** — the similarity scores *are* the computation, not a post-hoc estimate ([[intrinsic-over-posthoc-principle]], Rudin S1). So ProtoMF **inherits S5's modality-independent semantic challenges** (Axis A): latent-similarity ≠ human-similarity, unsupervised prototypes lack stated meaning, the fewer-vs-more prototype trade-off, cosine unreliability, redundancy/collapse. This is the bulk of what we cite S5 for.

**Differ — Axis A (representation):** S5's PPMs are *part*-prototypes — one input is spatially decomposed into patches, a prototype = an *object part* (S5 fn. 4: prototype = whole object vs prototypical-part = object part). ProtoMF has **no spatial decomposition**: an entity is one vector, prototypes are **whole-object anchor directions** ("prototypical users/items"). → **ProtoMF is prototype-based but not a *part*-prototype model.**

**Differ — Axis B (readout/regime):** PPM = positive-linear "scoring sheet" over prototype activations → fixed **classes** (classification). ProtoMF = **dot product of two similarity vectors** (user-sims · item-sims) → open-vocabulary **item ranking** (implicit feedback). Different scoring machinery + different regime.

**Differ — grounding/visualization:** PPMs *project* prototypes onto literal training patches (a showable image). ProtoMF interprets prototypes via **nearest items** — no spatial "where," because the input is an **ID, not pixels**.

**Consequence for citing S5:** transfer the **Axis-A semantic challenges + ante-hoc framing** (modality-independent); **disclaim the part/spatial-specific challenges** (spatial misalignment, patch-visualization ambiguity, deformable prototypes, pixel attribution) as **n/a**. S5 itself flags the gap pointing at us — *"PPMs for non-vision modalities are rare / undefined."* In related work, frame ProtoMF as **a recsys instance of the shared prototype-similarity principle**, never as a "part-prototype model."

**Related-work anchor (S5 §4.4):** S5 independently proposes *"semantically annotated prototypes via CBM-style semantic features"* (combining [29]=Koh CBM with the part-of idea) as an open future direction — **facet-1 instantiates exactly this for recsys**. Caveats: we borrow CBM's *named-semantic principle*, not its concept-bottleneck *architecture*; and "part-of" maps only loosely, as **feature-composition** ([[dc01_feature_composed_factors]]), not spatial parts. Vanilla ProtoMF does *not* do this (prototypes unsupervised, meaning post-hoc) — the thesis is the move from prototypes → semantically annotated prototypes. Clean framing: "the field points here, we go here."

[[decisions]] [[paper-understanding]] [[intrinsic-over-posthoc-principle]] [[attribute-source-positioning]] [[phase-2]]
