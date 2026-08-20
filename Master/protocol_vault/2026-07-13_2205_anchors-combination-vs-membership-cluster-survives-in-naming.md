---
date: 2026-07-13
time: "22:05"
phase: 4
tags: [thesis/background, thesis/intro]
placed: 2026-08-20
---
# Why ProtoMF beats MF on interpretability: anchors, combination over membership — and the cluster notion surviving in the naming

*(Verbatim-capture entry — worked through at Matteo's initiative during the outline session ("let's hammer one thing home"). **Explicitly NOT thesis prose in this form** (Matteo: "it touches on too many things at once") — protocol-grade material to be decomposed; the pieces land at the appropriate points, chiefly Background §2.2.)*

## The point (agreed restatement)

> ProtoMF is more interpretable than MF because it places anchors in the latent space — the space where both sides of an interaction already live. Anchoring that space is not new in spirit: one could simply cluster the entities, and a prototype is indeed something like a cluster centroid. But ProtoMF does not *assign* entities to prototypes — it expresses them as a graded combination of affinities to all of them. For a recommendation, the combination is the more sensible form: a taste is not membership in one cluster, and the score itself is built from that combination. The cluster notion still matters, though — it does not drive the scoring, but it drives the *naming*: a prototype is named by reading its neighborhood, exactly as one would summarize a cluster around its centroid.

## Matteo's original (near-literal)

"ProtoMF is better than MF in terms of interpretability because it places anchors in the latent space, where both parts of the interaction's elements live. One could also just cluster elements (prototype anchors as a sort of centroid of a cluster), but to express a recommendation as a combination is more sensible in terms of prototypes, because it does not imply membership to one cluster. The notion of a cluster is important though, because it is involved at least partially in naming the anchors in ProtoMF."

## Sharpenings from the discussion

- **Combination beats membership twice, not once:** (1) graded — an entity relates to every prototype somewhat; no forced membership; multi-interest taste survives (0.7 minimalist, 0.4 streetwear). (2) mechanically honest — the similarity profile IS the representation the score is computed from, and the score decomposes additively over prototypes → each prototype carries a quantifiable share of *this* recommendation (GAM-like in prototype space, [[2026-06-11_1504_protomf-not-a-gam-additive-only-in-proto-space]]). Cluster membership yields a label; the profile yields an attribution.
- **Precision caveat:** soft clustering exists (mixture models) — the clean contrast is not hard-vs-soft but *membership as description* vs *affinity profile as the scoring representation itself*.
- **Clustering returns twice:** (a) the two regularizers sculpt cluster-like geometry — coverage + separation ([[2026-06-15_1238_protomf-two-regularizers-geometric-effect]]); (b) post-hoc naming reads a prototype as a centroid — summarize the neighborhood, that is the name ([[2026-06-08_1149_prototype-naming-and-similarity-routes]]).
- **The setup punchline:** scoring already outgrew clustering; naming has not yet — until the attributes arrive. (Clustering = the lens for naming the anchors; combination = the mechanism for using them.)
- Related-work echo: ACF carries the "anchors" idea by literal name — natural lineage entry point in Background.

[[phase-4]] [[explanations]] [[paper-understanding]] [[thesis-writing]]
