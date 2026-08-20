---
date: 2026-08-17
time: "18:23"
phase: 4
---
# CORRECTION: the double-tie "prototypes as projection basis" mechanism is wrong (code- and paper-refuted); July's UI-user effective count is unreproducible

Two corrections to the record, found during explanation deep-dive 3.1 (2026-08-17) when Matteo asked where the "duplicating a prototype costs representational rank" sentence came from.

## Correction 1: prototypes do NOT hold a projection job at the double tie

The claim (born [[2026-07-15_1112_prototype-collapse-is-a-self-built-popularity-bias]] as "hypothesis-grade", restated as mechanism in [[2026-08-15_1651_spectral-scale-gauge-inversion-direction-diversity-metric]]): at `prototypes_double_tie`, each side's prototype matrix is reused as the projection basis of the other side's linear half, so prototype row norms stop being gauge and duplication costs rank that accuracy sees.

**Verified against BOTH sources of truth, 2026-08-17:**
- **Code** (`feature_extractor_factories.py` `prototypes_double_tie`, `EmbeddingW`): the projections are free `nn.Linear(d, out_dim)` layers; the ONLY tying is each side's embedding TABLE shared between its two branches (same u feeds cos(u,p_l) and W_u·u).
- **Paper** (§3.3): "UI-ProtoMF *defines two linear transformations*, û = W^u·u, t̂ = W^t·t" — free matrices; what is shared "across the two units" is the user/item embeddings.

**Corrected statement:** prototype norms are pure gauge in EVERY implemented variant, UI included — prototypes only ever enter through cosine (forward, both regularizers, intrinsic naming). One could unit-normalize all prototype rows in the UI checkpoint and every score/ranking/explanation would be bit-identical. What gains a second job at the double tie is the **entity embedding** (cosine leg + linear leg — its norm IS read by the linear branch); the Aug-15 entry's entity-norm bullet stays correct, its prototype-norm bullet does not. Instrument consequence: row-normalize-first direction metrics are exact (not approximate) for prototypes at UI too. Design-space caveat: a future variant that deliberately ties W := P (e.g. a Route-1-style construction) would create the projection job for real — then, and only then, the scale-aware-spectral discussion applies to prototypes.

**Gauge ≠ irrelevant-to-training footnote:** grad of cos w.r.t. p scales ~1/‖p‖, so norms modulate direction-update speed; weight decay keeps them uniform (~0.13 at S0.7), making this roughly symmetric.

## Correction 2: July's "UI-user ≈ 64 effective (healthiest)" is unreproducible

The July-15 table lists user_item_proto (user side) ≈ 64 effective directions. Measured 2026-08-17 with the committed instrument (row-normalized, duplicate threshold, cross-checked greedy dedup vs connected components — all agree):
- canonical cluster checkpoint (expl_deep_dive 3.1): **10** effective of 76 (clone components 24/9/7/6/6/6/5/5/4/4; 410 pairs >0.999), spectral 4.85, 23 distinct lift names;
- older local run (`results/user_item_proto_hm_1_month`, different md5): **8** effective.

Neither is near 64 → the July number was a measurement error or was taken on a since-replaced checkpoint. **The qualitative claim survives, weakened:** UI is the healthiest H&M host (10 > 5 > 2 effective; names 23 vs 5 vs 7; UI-item 8/11 with the least-wasteful ratio) — but "≈64" must not be quoted.

## Downgrade: WHY UI is healthier is an OPEN question

With the projection-basis mechanism dead, the measured UI health advantage (same near-zero coverage regularizer as the collapsed single hosts — so NOT the regs) has no confirmed mechanism. Leading candidates: (a) the entity-embedding second job — the linear branch forces embeddings to stay linearly discriminative, changing what the herding regularizer finds to herd prototypes into; (b) UI-item side drew small K=11 (+ 5× coverage weight, July's own second factor); (c) unknown. The cosbias2x2 UI arms add evidence; a dedicated probe would be needed for attribution.

[[phase-4]] [[explanations]] [[paper-understanding]] [[decisions]]
