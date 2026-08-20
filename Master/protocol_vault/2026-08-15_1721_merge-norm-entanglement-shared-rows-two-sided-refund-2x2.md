---
date: 2026-08-15
time: "17:21"
phase: 4
tags: [thesis/fufi-merge, thesis/hidden-effects]
placed: 2026-08-20
---
# Merge norm entanglement: shared rows weld the popularity channel to the steering — two-sided refund upgrade, host-tie separability contrast, the built-in 2×2, and the expressivity resolution

*(Verbatim-capture entry from the F5 reading session, 2026-08-15 — a five-step build-up worked through interactively at Matteo's questions, all specifically about the **fUfI merge stage**. Destined for the hidden-effects section and the merge chapter (`fufi-merge`). Everything here is hypothesis/mechanism derivation — no measurements; designed tests belong to the merge cycle. Siblings: [[2026-08-15_1639_pm-purity-port-groundtruth-target-discussion]], [[2026-08-15_1651_spectral-scale-gauge-inversion-direction-diversity-metric]].)*

## 0. Matteo's opening question and the factual correction

Question: at the merge, prototype latents are constructed from shared attribute embeddings, so a prototype's norm is "subject to the universe, not to itself," while the linear part of a side is more free — could that entanglement cause distorting or performance-inhibiting effects?

Correction that reframes it: **in our lineage the prototypes are free** `nn.Parameter`s at every stage (grounding is read off via cos(e_f, p_k), unenforced; constructed prototypes were dc02/dc04, both shelved; even under a centroid-*init* scheme they are free after step one). What is "subject to the universe" is the **composed entity embedding** (t_i = Σ e_v + id_i; q_u = mean of basket rows + id_u). And at the double tie "the linear part" is not a separate parameter at all — the tie reuses the same composed embedding, projected through the other side's prototype matrix. So the concern is real, but its locus is the **shared attribute rows**, not the prototypes. (Prototypes stay clean even at the merge: their projection job is norm-sensitive but their cosine job is norm-blind, so the norm serves the one job without compromise.)

## 1. The mechanism: at the merge, each shared row's norm holds two jobs with different masters

At fUfI, composed embedding norms become behaviorally meaningful (they feed the linear halves) *and* they are class-tied. A shared row's norm is one scalar serving:

- **cosine branch:** relative row norms = mixture weights of every carrying item's direction (which attributes dominate its identity — the interpretable structure);
- **linear branch:** row norm = score magnitude class-wide (e.g. "denim items get a boost" — a class-level popularity prior; LightFM's feature-composed bias mechanism arriving uninvited).

The optimizer must compromise. Concrete distortion risk: **popularity pressure on the linear branch inflates certain rows' norms, and the inflated norms drag the cosine-branch directions of every carrying item toward the popular attributes** — popularity contaminating the interpretable direction structure through the norm channel, with no bias term anywhere in sight. ("Removing the bias term didn't remove the bias," one model later, wearing a new coat.)

**Two-sided upgrade of the length-channel refund hypothesis** ([[2026-07-12_1224_length-channel-refund-hypothesis-hidden-effects-section]]): the registered refund (linear halves make norm/length expressive again at the merge) and this contamination are **the two signs of one mechanism** — the reopened norm channel can refund the angular tax *or* smuggle popularity into the directions, and which sign dominates is not decidable from the armchair. Designed-test material for the merge cycle, with knobs ready if the bad sign shows: explicit feature-composed bias term (give popularity a dedicated home — the dc01 bias-gap fix), ID-row division of labor, centering.

## 2. Host double tie: the same two-jobs structure, but SEPARABLE — the problem is tie × composition, not the tie

Host UI-ProtoMF has the identical two-jobs anatomy (score = u*·(tP^u) + (uP^t)·t*; every entity embedding: cosine direction + linear vector; every prototype: cosine anchor + projection axis). But the host is safe because embeddings are per-entity free:

- inflating ‖t_i‖ boosts item i's linear half, leaves i's cosine direction untouched (cosine is norm-blind), and moves **no other item** — no shared rows;
- popularity lands as clean **per-item intercepts** (the already-catalogued implicit item-intercept channel, [[2026-07-12_1808_bt-implicit-item-intercept-shift-asymmetry-directives]]); direction structure and norm structure optimize quasi-independently.

The host's genuine cross-job tension is on **directions** (cosine anchor vs projection axis) — and per the collapse entry's hypothesis that tension is plausibly a *benefit* (redundancy costs projection rank accuracy sees → anti-collapse pressure; coheres with F-DC01-07). Same structural feature, opposite sign, because it acts on directions of free vectors, not norms of shared ones.

|  | free embeddings | composed embeddings |
|---|---|---|
| **no tie** (single-sided) | norm gauge on proto side, free intercept on linear side — nothing to entangle | norms gauge → norm channel **closed** (the tax); angular effects only |
| **double tie** | host UI: channel open, per-entity → **separable, benign** | fUfI: channel open, class-tied, doubly employed → **non-separable** |

**The built-in 2×2 (measurement gift):** the four cells are four models the lineage has or will have — host U/I (neither), host UI (tie only), fI/fU (composition only), fUfI (both). Host UI is the free control arm: if popularity contamination of direction structure appears at fUfI but not in host UI (identical tie, free embeddings), the tie×composition interaction is isolated with no new architecture. The refund-vs-contamination test ships with its own control group because of how the lineage was staged.

## 3. What "class" means, precisely (Matteo's hammer-down)

The channel's primitive unit is one **attribute value** — one row = one value v; its class = the carrier set of v (all items with colour_group=Black). Hence:

- **classes overlap, not partition** — each H&M item sits in exactly 5 value-classes (+ its ID singleton); its linear-half exposure = the sum of its five value effects + ID effect;
- **conjunctions are not addressable** — no row for black∩denim; boosting the intersection necessarily boosts all black and all denim. The channel's vocabulary = additive combinations of single-value effects (the composition-vs-FM boundary again: no interaction terms, for better and worse);
- **cross-item coupling** = norm-weighted signature overlap: share nothing → independent; share all 5 → **signature twins, maximal coupling** (identical linear halves up to ID rows). The entanglement structure is a weighted overlap graph; twins are its cliques; the finest distinction the shared channel can draw is the full 5-value signature; below that, only the ID row differentiates. (The ID/twins braid's "feature rows carry only class-level signal," now with "class" made precise; F-DC01-06 continuum = the coupling graph's clique structure.)
- **fU footnote:** user-side membership is graded — exposure to e_v ∝ v's share of the basket (mean-normalized weights); classes are soft. **ml-1m footnote:** variable tag bags (3–72 tokens, raw-bag policy) → tag-rich items sit in more classes, coupling them to more of the catalog (token-mass coupling inside the entanglement graph).

## 4. The expressivity resolution (Matteo: "is norm a less powerful channel — like neurons sharing weights?")

Split by arm — the analogy is right in one and instructively wrong in the other:

- **noid arm: yes, strictly less expressive.** Linear halves confined to additive spans of V shared rows (~V·d DoF vs the host's N·d); twins forced identical; conjunctions unaddressable. Hard weight sharing in the CNN sense — one filter reused across positions.
- **ids arm: NO — the function class is identical to the host's.** t_i = Σe_v + id_i is a **reparameterization**, not a restriction (any target reachable via id_i = target − Σe_v). What changes:
  1. **Repricing (partial pooling, the better analogy).** Under weight decay, "boost these 500 denim items" costs the host 500 per-item norms and fUfI **one row**. Class-shaped signals become cheap, idiosyncratic ones cost full price through the ID row — a multilevel model: shared rows = group effects, ID rows = individual effects, weight decay = the pooling prior. For class-aligned popularity the channel is *more* powerful per parameter — simultaneously the cold-start feature ("denim sells well" survives for a new item) and the contamination risk. One repricing, two faces.
  2. **Orthogonality loss — the sharpest sentence of the thread.** The host owns a per-item **direction-preserving dilation** (scale t_i: norm moves, direction untouched — what made its popularity channel clean). In fUfI no such move exists: only components can be scaled — scaling a shared row re-steers every carrying item; scaling the ID row rotates the item toward its ID (eroding feature-explained fraction). For any item sharing even one value with anyone, **no parameter move changes norm without changing direction.**

**The corrected one-liner:** *noid: strictly less expressive (hard sharing). ids: equally expressive, differently priced — class-level cheap (partial pooling), idiosyncratic full price — and no longer separable: every norm move is also a direction move, because the host's per-entity direction-preserving dilation does not exist in the composed parameterization.* This *derives* the separability collapse from the parameterization rather than asserting it, and plugs into F-DC01-08 (L2-as-canonicalizer), the ids/noid division-of-labor entries ([[2026-07-12_1830_m5-noise-for-crowding-ids-division-of-labor-centering-option]]), and the feature-composed-bias idea as the friendly face of the same repricing.

## 5. Routing

- **Hidden-effects inventory** (`hidden-effects`): joins as the merge-stage sibling of the braid — "norm channel welded to the steering"; the refund hypothesis is now explicitly two-sided.
- **Merge chapter** (`fufi-merge`): the 2×2 as the experiment design skeleton for the refund/contamination test; host-dependence of the effect doubles as exposition structure (teach the host's clean channel, then show composition welding it shut).
- **Instrument tie-in:** the direction-contamination arm is measured with the direction-diversity/purity instruments of the two sibling entries (per-field P_m profile, normalized-spectrum measures) — popularity axis vs attribute axes on the same PCs (R²-style) is the natural adjudicator.
- Scrutiny-evidence-not-thesis applies; nothing here is scheduled before the merge cycle's own DC + scrutiny steps.

[[explanations]] [[evaluation]] [[decisions]] [[phase-4]]
