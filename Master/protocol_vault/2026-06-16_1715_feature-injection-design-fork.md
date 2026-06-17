---
date: 2026-06-16
time: "17:15"
phase: 2
---
# Design fork: feature injection via prediction vs prototype-space prior

Identified a design fork for feature-aware ProtoMF from S7 Eq 25 (Chen et al. 2020, §4.1.1), which has two slots for attributes: the likelihood (additive terms α/β/γ entering the prediction) and the priors p(W|X)p(H|Y) on latent factors. Mapped to two candidate routes: **(A)** features feed into the entity representation/score directly; **(B)** features act as a prior/regularizer shaping the prototype space without touching the score — natural fit since ProtoMF already has structured-space regularizers (R_proto, R_batch). S7 flags route (B) as "not yet observed" in the linear category, making it a **citable novelty angle**. Not mutually exclusive. Logged to design-candidate scratchpad for a future design cycle; decision deferred. Relates to [[decisions]] and the existing scratchpad entry on attaching feature profiles to CF prototypes.

[[decisions]] [[paper-understanding]] [[phase-4]] [[phase-2]]
