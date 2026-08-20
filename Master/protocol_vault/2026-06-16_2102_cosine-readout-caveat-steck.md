---
date: 2026-06-16
time: "21:02"
phase: 2
tags: [thesis/hidden-effects, thesis/conclusion]
placed: 2026-08-20
---
# Caveat: ProtoMF's cosine prototype-readout inherits Steck's unreliability — and harder (S5 §4.4)

S5 §4.4 cites cosine similarity as an **unreliable** similarity metric (Steck, WWW 2024 [55]). The twist: Steck's result targets **matrix-factorization embeddings** specifically — cosine can be near-arbitrary because the training objective is invariant to per-dimension rescalings. **ProtoMF is MF**, so the caveat lands *harder* on us than on the vision PPMs it was cited against.

Scope it precisely: ProtoMF's **computation-fidelity is intact** — the per-prototype contribution decomposition is exact algebra, no estimation step. What's threatened is treating the **shifted-cosine score as a faithful *semantic* similarity** ("how much item X expresses prototype P"). That semantic reading is where interpretability claims live, so it's the one at risk.

Implication: when we make interpretability claims from prototype cosines, **validate the readout** rather than assume it; metrics beyond cosine are an open knob. Facet-1 mitigates by making the score's *drivers* nameable (feature-composed prototypes), but does not by itself fix cosine's metric unreliability.

## Possible future experiment — cosine-reliability / readout-metric probe
Test whether the prototype-similarity readout is a *stable, meaningful* similarity or an arbitrary artifact:
- **Metric-swap ablation:** swap shifted-cosine for alternatives (standard cosine, dot product, L2/Euclidean, learned metric) and measure both ranking accuracy (`hit_ratio@10`/NDCG) **and** interpretation stability (overlap/rank-correlation of each prototype's nearest-items). *Tell:* accuracy roughly invariant but interpretation shifts ⇒ the cosine value is doing arbitrary work.
- **Rescaling / regularization perturbation (direct Steck test):** apply per-dimension rescalings (or vary reg strength) that leave predictions unchanged; check whether prototype nearest-items / cosine rankings move. *Tell:* unstable interpretation under prediction-preserving perturbation ⇒ unreliable semantic readout.
- **Seed stability:** consistency of prototype interpretations across seeds (S5 notes PPM run-to-run inconsistency).
- **Feature-overlap proxy (facet-1 bridge):** in the feature-aware variant, test whether cosine-similar items share named features — a proxy for semantic faithfulness of the readout.

Scope: pick the minimal viable subset (metric-swap + seed stability are cheapest); the rescaling probe is the most direct evidence but needs care. Wire results into `Master/experiments/`.

[[decisions]] [[protomf-vs-ppm-positioning]] [[intrinsic-over-posthoc-principle]] [[explanations]] [[phase-2]] [[phase-4]]
