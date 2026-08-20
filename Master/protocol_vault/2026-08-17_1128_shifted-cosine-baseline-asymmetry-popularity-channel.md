---
date: 2026-08-17
time: "11:28"
phase: 4
---
# Shifted-cosine +1 baseline asymmetry — U-side popularity channel vs I-side dead gauge

**Setting.** Explanation deep dive (observations file `Master/docs/scrutiny/explanation_observations.md`), section 1.1 = U-ProtoMF on hm_1_month, reading the regenerated breakdown artifact `breakdown_user42_item670` plus the naming/cards/t-SNE set. Marked ultra-important by Matteo; capture close to verbatim (gate-discussion rule).

## The decomposition, plainly

U-ProtoMF scores an item as S(u,i) = Σ_l t_l · u*_l, where t is the item's embedding (one coordinate per user prototype) and u*_l = 1 + cos(u, p_l) is the shifted cosine similarity of the user to prototype l. Splitting out the "+1":

- **Item baseline B(t) = 1ᵀt** — the sum of the item's embedding coordinates. Comes purely from the "+1" in shifted cosine. It is user-independent (every user gives this item the same baseline) **but it differs between items, so it fully participates in item ranking.** It is a structural popularity channel the model can (and did) build itself.
- **Personalized part per prototype: t_l · (u*_l − 1) = t_l · cos(u, p_l)** — a two-factor product: (how much the user resembles prototype l) × (the item's learned coordinate on axis l). There is no separate "prototype value" factor; the item's coordinate IS the prototype's weight for that item.

**Measured on the real artifact:** S = +5.0889 = baseline **+4.0636** + personalized **+1.0253**. Roughly **80% of this recommendation's score is "this item's own scalar"; only 20% is about the user.** Combined with the fully collapsed prototypes (the 76-line personalized table is five blocks of near-identical numbers — 37× +0.036 for the Loungewear clones, 17× −0.0058 sport, 10× −0.019 Mama, etc., i.e. ~5 effective degrees of freedom), the honest description is: **on H&M, U-ProtoMF is close to a pure popularity ranker with a small ~5-dimensional personalization residue.** The prototype collapse (vault 2026-07-15_1112, self-built popularity bias) and the baseline dominance are two faces of the same mechanism, now both measured on explanation artifacts.

## The side asymmetry (why this is a USER-prototype-side defect)

On I-ProtoMF the roles swap: the item representation is the similarity vector [1 + cos(t, q_k)] and the user embedding u multiplies it. The "+1" mass becomes **1ᵀu — a per-USER scalar, identical for every item that user scores.** A constant added to all of a user's item scores cannot change which items rank on top → rank-inert.

Where could the I-side scalar still act? Worked through case by case:

| | U-side (1ᵀt, per-item) | I-side (1ᵀu, per-user) |
|---|---|---|
| Item ranking | **alive** — it IS the popularity channel | dead |
| BPR | alive for ranking | dead (cancels in the pairwise difference) |
| Sampled softmax | alive for ranking | dead (cancels in the per-user normalization) |
| BCE | alive | **alive** — acts as a classic per-user bias b_u, carrying *user activity* |

All three H&M host reference runs trained with **sampled softmax** → in our runs the I-side channel is dead in ranking AND in the loss. Corollary: since it cancels in the loss, **no gradient ever shapes it** — 1ᵀu is an unconstrained leftover drifting wherever other gradients incidentally push it. A gauge freedom. When reading I-ProtoMF breakdowns, the disclosed per-user baseline line must be treated as **uninterpretable noise** (disclosed for arithmetic honesty only, never narrated as meaningful).

**"User activity", precisely:** under BCE the scalar would encode the user's base interaction rate — "how likely this user is to interact with a random item at all" — the intercept of a per-user logistic regression. It is item-blind by construction: it can express "this user buys more, period", never anything item-conditional like "active users prefer unpopular items" (that would require the personalized term).

**Subtractability:** post-hoc, the per-user scalar is removable with zero effect — subtract 1ᵀu (or add a million) per user and every recommendation list and every per-user metric (NDCG/HR) is bit-identical. Only procedures comparing scores ACROSS users (global thresholds, cross-user candidate pooling, probability calibration) could ever see it; our offline pipeline has none. BUT: "subtract at inference" ≠ "remove the shift from the model" — training with unshifted cosine is a different model (the shift keeps similarities in [0,2], non-negative, which changes gradient dynamics and reachable geometries on both hosts; the U-side popularity channel is the loudest consequence of that design choice).

## Thesis-figure commitment

For the hidden-effects section: **"effective personalized vs unpersonalized component average"** — over all recommendations made (top-K lists), average the personalized vs unpersonalized score components, where **"effective" counts only rank-relevant mass** (mass that could change recommendations). By construction the I-side unpersonalized component is ≡ 0 effective; the U-side baseline counts in full. With these numbers plus a few selected graphics at the end of section 2.3, the effect can be explained very concisely. The per-recommendation baseline share (80% in the example above) is the same instrument at N=1.

## Resolution of an apparent contradiction: if the baseline already carries popularity, why do the prototypes ALSO collapse onto it? Shouldn't they be free?

Raised by Matteo while writing his 1.1 note ("80% baseline … also explains the collapsed prototypes to an extent" felt internally contradictory). Resolution in three forces:

**1. The channels share parameters and multiply — the baseline isn't "used up", it's amplifiable.** There is no separate popularity parameter: the baseline is 1ᵀt, the same coordinates the personalized term uses. Item i's coordinate l contributes t_l · (1 + cos(u, p_l)) — per unit of t, factor 1 from the shift plus cos from prototype alignment. A prototype pointing at the bulk-user direction makes popular items earn ~2× per unit of coordinate (1+cos ≈ 2 for the majority), while the anti-aligned minority pays **0, never negative** (1+cos bottoms out at 0). Inflating a popularity coordinate is therefore RISK-FREE: aligned users pay double, misaligned users pay nothing. The optimizer recruits prototypes INTO the popularity channel as amplifiers — without the shift, a popularity coordinate would punish popular items for misaligned users (cos < 0); the "+1" removes the downside and makes the collapse strategy strictly profitable.

**2. The gradient mass points that way anyway.** Prototypes only receive gradients through users; most gradient volume comes from bulk users buying head items (sampled softmax, uniform negatives — the easiest positive/negative separation IS popularity). Every prototype is pulled toward the user-mass centroid; only coherent, segregated user groups with distinctive co-purchases (sport, socks, Mama, plus-size — exactly the five surviving prototype groups) generate enough opposing gradient to hold a niche.

**3. Nothing pushes back — this run's regularizers actively help.** ProtoMF has NO prototype–prototype repulsion term: duplicating a useful prototype is free. Of its two regularizers, this run weighted the "each prototype near some batch user" term strongly (sim_proto_weight ≈ 1.80, max-type) — maximally satisfied by parking prototypes in the densest user region, i.e. it HERDS them into the bulk — while the spreading term ("each user near some prototype") is at sim_batch_weight ≈ 0.0018, effectively off. Hyperopt chose these weights on validation hit-ratio@10, which a popularity ranker wins on H&M. The empty prototype-less regions in the 1.1 t-SNE are the visible print of that 0.0018.

**Punchline for the note/thesis:** the baseline dominance and the prototype collapse are not two effects where one explains the other — they are the same optimization pull, and the shift's [0,2] range makes prototype-alignment-with-the-bulk a risk-free amplifier of the popularity channel, which is precisely what recruits the prototypes into collapsing.

## Addition: why BCE sees the per-user constant and softmax/BPR do not (thesis inclusion uncertain — Matteo undecided)

The two losses ask fundamentally different questions of the score. **Sampled softmax asks a RELATIVE question** — "does the positive beat these negatives?" — probabilities are exp(s_i)/Σ exp(s_j) within one user; adding a constant c to all of a user's scores gives exp(s_i+c) = e^c·exp(s_i), and e^c factors out of numerator and denominator — cancels exactly. The loss depends only on score differences within the user; a competition only cares about gaps. (BPR even more directly: σ(s⁺ − s⁻) is literally a function of the difference.) **BCE asks an ABSOLUTE question, one pair at a time** — "is this (user, item) pair a purchase, yes or no?" Each score goes through a sigmoid on its own, no cross-item normalization to absorb a constant, and σ(s+c) ≠ σ(s) — the offset changes every pair-probability, hence the loss, hence receives gradients.

Numeric contrast (positive = first item): scores [2, 0, −1] vs [12, 10, 9]. Softmax: identical probabilities (0.84, 0.11, 0.04) — the +10 vanished. BCE on the positive: σ(2) = 0.88 vs σ(12) = 0.999994 — the +10 mattered.

Why BCE fills the slot with user activity specifically: the offset raises all of a user's pair-probabilities uniformly, so its optimal value matches the user's base rate of positives among training pairs — heavy buyers have more positives, their offset drifts up until the average predicted probability agrees. The per-user intercept of a logistic regression finding marginal purchase propensity.

One-liner: softmax grades items against each other within a user (differences only), BCE grades each pair against an absolute yes/no (levels matter) — a per-user constant lives entirely in the level, so softmax is blind to it and BCE calibrates with it.

## Closing 1.1: the shift is UNNECESSARY for the explanation machinery it was introduced to serve

The paper's entire stated justification for the shifted cosine is one sentence (u* positive in [0,2]) plus the decomposability remark; no ablation, no discussion of side effects. Worked through at the 1.1 close:

- **Post-hoc top-k naming never consults the shift.** User side: top-k items per prototype are ranked by the raw item coordinate t_l — no cosine appears in the computation at all. Item side: 1+cos is strictly monotone in cos → identical orderings → identical top-k, names, lifts, cards. At a fixed checkpoint, every explanation artifact comes out the same under regular cosine.
- **The shift buys "non-negative memberships", NOT "non-negative explanations".** The representation u* has no negative entries, but score contributions s_l = u*_l·t_l still go negative whenever t_l < 0 (user 42's breakdown has negative Mama/sport bars). A truly subtraction-free decomposition would need non-negative t (NMF-style), which the paper does not do.
- **The synthetic max-activation probe, translated:** shifted (0,…,2,…,0) ≡ regular-cosine (+1 at l, −1 elsewhere) — semantically "member of l AND polar opposite of every other community", not "neutral". Yet score-wise the zero-background probe is the CLEANER one: it ranks by 2·t_l alone — mathematically identical to the top_k_items_per_prototype CSV. In regular cosine, the natural neutral-background probe (+1 at l, 0 elsewhere) gives the SAME ranking — semantics and arithmetic finally coincide. The −1-background variant is a different, interesting probe: ranking by 2t_l − 1ᵀt = the axis readout with the popularity baseline subtracted (a lift-flavored, popularity-corrected axis diagnostic — potential future artifact).
- **Probe feasibility caveat:** no real user can have a one-hot u* (with 37 clone prototypes, every actual user has near-identical similarities across the block); the paper's interpretation method extrapolates far off the reachable user manifold (Steck-family validity caveat).

**Punchline:** display-level positivity aside, the shifted cosine is inert for explanations and consequential only through the side effect nobody stated — the U-side popularity channel that shapes the learned geometry the explanations then faithfully report.

## Deferred follow-up entry

At the END of deep-dive section 2.3: a second protocol entry inventorying (a) how well we found concrete examples of this effect across sections 1.1–2.3 (Matteo is watching for it throughout), and (b) what else in the explanations is messed up and so-far unaccounted.

[[phase-4]] [[explanations]] [[paper-understanding]] [[thesis-writing]]
