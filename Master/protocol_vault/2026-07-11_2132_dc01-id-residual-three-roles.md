---
date: 2026-07-11
time: "21:32"
phase: 4
tags: [thesis/fi-protomf]
placed: 2026-08-20
---
# dc01: the ID residual is information, not dead weight — three roles, one hard boundary

*(Rewritten same day under the verbatim-capture rule — the first version compressed a praised gate discussion into jargon; this version carries the explanation close to literally, so it can regenerate the argument without the conversation in context. From the dc01 SC.1b gate discussion.)*

## Setup: where the ID part shows up

In dc01's +ID variant, the item's vector is q_i = e_dept + e_colour + … + **e_ID(i)**, so every prototype activation itemizes exactly as

> t*_k = 1 + (feature shares: c_dept,k + c_colour,k + …) + (the ID share: c_ID,k)

The question decided here: what to do with that last term. Answer: it has three legitimate roles — none of which is "discard."

## Role 1 — the honesty meter (already designed in)

The ID share is rendered, not hidden: "item-specific: +0.3." What makes this genuinely valuable rather than an embarrassment: it *quantifies exactly how complete the feature explanation is*, per recommendation, per prototype. Most explanation methods cannot tell you how much they are NOT explaining — ours can, to the decimal, because the decomposition is exact. Aggregated over items this becomes the **feature-explained fraction**, a committed headline metric.

## Role 2 — the residual can be partially *labeled*, even if not explained

Think about what e_ID(i) *is*: everything behavior taught the model about this item that its five attributes couldn't express. Under our bias-free fleet, item popularity has nowhere else to live — so we expect ID shares to be substantially "this item is popular/unpopular." That is an empirical claim, and **the SC.8 test (feature-explained fraction stratified by item popularity) was deemed extremely important at this gate** — it will further our understanding here: if confirmed, the rendered residual line can honestly say "item-specific (largely how popular this exact item is): +0.3" — real information to a user, even though it is not a feature.

## Role 3 — a positive diagnostic use (mis-tag / latent-property probe)

Since e_ID(i) lives in the same space as the feature vocabulary, you can read it *against* that vocabulary. An item whose ID residual aligns strongly with a feature it does NOT carry is telling you: "this item is tagged basic but *behaves* like premium-section items." The catalog says one thing; 623,000 purchases taught the model another. Mis-tag detection, latent-property discovery, trend-spotting — genuinely using the opaque part positively.

**What "lives in the same space" means (user question, worked through):** none of these are single numbers — every ID and every feature value is a learned **vector** of d numbers (d ≈ 10–100), an arrow in a d-dimensional room. "Same space" means concretely: (1) they are rows of ONE shared embedding table; (2) they are **summed together** into q_i — and a sum is blind to labels: once added, the model cannot tell which part came from a feature and which from the ID; they are interchangeable currency (LightFM's original framing: the ID is just one more feature); (3) they face the same judges — all of q_i meets the same prototypes through the same cosine. Comparing two of them computes "do two arrows point the same way?" — and since each direction in the room encodes some latent behavioral trait learned from purchases, two arrows pointing the same way means **the model uses them identically when predicting behavior**. Honest footnote: "the model uses them identically" is guaranteed; "and that matches the human label" is validated at SC.8 (profile-validity check, F-DC01-02), not assumed.

**Refinement 1 (user challenge — the mixture problem):** the naive probe "check cos(e_ID, e_f) per single feature" is too narrow: e_ID will generally encode a *mixture* of vocabulary directions, so single cosines can read ≈0 on a richly structured residual. But the naive fix — "fit e_ID as a linear combination of features" — is vacuous: with 426 feature arrows in a ~64-dimensional room, the dictionary is overcomplete, so EVERY vector (noise included) is exactly representable; "fits some combination" carries zero information. The meaningful probe is a **sparse decomposition with a null baseline**: claim structure only if a *few* features explain the residual far better than they explain random vectors.

**Refinement 2 (user question — does e_ID transfer to prototype space?):** yes, and the transfer is already computed: the per-prototype ID shares (c_ID,1 … c_ID,K) ARE e_ID's image in prototype coordinates — readable, once prototypes are grounded, as "this item's idiosyncrasy leans +0.2 toward *dark menswear denim*." Crucially, the overcompleteness problem does NOT come along, because of a distinction worth engraving: **reading coordinates** (computing cos(e_ID, p_k) for each prototype — exact, canonical, zero degrees of freedom) vs **reconstructing identity from a dictionary** (a fit — ill-posed when the dictionary is overcomplete). The prototype-coordinate reading is therefore the natural FIRST probe (K numbers, each named, none arbitrary; caveat: near-duplicate prototypes (mode M4) make coordinates redundant, not arbitrary); the sparse vocabulary fit is the fine-grained second probe.

## The caveat and the hard boundary (both decided)

- **Caveat:** Roles 2 and 3 are post-hoc readings of a CF vector — they can never carry the intrinsic claim (R2). *But that does not exclude them from the thesis*: as diagnostics/appendix material they are interesting in their own right (user: 200% agree) and turn the opaque residual from dead weight into information.
- **Hard boundary (user 200%):** ID rows never enter prototype profiles/explanations. Prototype meaning stays metadata-only — otherwise "prototype 12 means: item #4711, item #892 …" and we would have rebuilt the post-hoc top-k-items explainer inside the parameters.
- At cold inference the question dissolves: the ID row is dropped (never trained for a cold item; the F-S0-07 drop, implemented at SC.3).

Cross-refs: scratchpad `[captured scrutiny dc01]` (probe design details); dossier `Master/docs/scrutiny/sc01_feature_item_proto.md` SC.1b P2 addenda.

[[decisions]] [[explanations]] [[evaluation]] [[phase-4]]
