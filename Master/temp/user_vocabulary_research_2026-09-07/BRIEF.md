# Shared brief for the user-vocabulary deep research (2026-09-07)

## The question (Matteo, verbatim)
"right now it averages interactions for user embeddings, which brings it back to the attribute
vocabulary which is fine, but truly lets think about what appropriate and interpretable vocabulary
would be based on interactions for a user. I am not sure how much we should impact the way the model
uses interactions, but maybe we can find way better solutions."
"forcing to view a users interactions history as only an average of attribute words is extremely
limited, I feel like we could come up with ways that the model could use that are semantically
meaningful that go beyond that. something like: we recommend this, because these two interactions in
your history strongly co-occur — or any other different idea."

## Scope (fixed)
- Input is ONLY *which items a user interacted with* in the train window. Event context (date, season,
  sales channel, price paid, order/basket structure) is OUT. Demographics are OUT, not even as fallback.
- "Combination/interaction of both" = item-identity-level history and attribute-level history may be
  combined or interact inside the user representation.
- Every idea must be priced on TWO axes: (1) is exact additive attribution to nameable units preserved?
  (2) how much does it alter what the host does with the interaction signal
  (none / re-weighting / new additive terms / new object)?

## The model as it stands (fU-ProtoMF, dc05, `feature_user_proto`)
User vector q_u = Σ_f (n_{u,f}/|H_u|) e_f + e_ID(u). Every train interaction is exploded into its
item-attribute word codes (hm: 5 categorical fields, V=426 values; ml-1m: genres+tags bags,
3–72 tokens per movie); words are mean-pooled over the history; a free per-user ID row is added
(`_noid` variant drops it). No timestamps, no cap, no recency, no IDF, no centring. The vocabulary
and codes are the item side's (shared with fI-ProtoMF), the embedding table is fU's own.
q_u then goes through U-ProtoMF's prototype layer: score = Σ_l t_{i,l} (1 + cos(q_u, p_l)) with a free
item vector t_i in prototype space. Explanations: prototype bars, zoom per purchased item, zoom per
attribute word, user-ID row disclosed; prototypes named intrinsically by cos(e_f, p_l).
Decoupled control `lightfm_hist`: identical q_u, plain dot product with a free item vector, no prototypes.

## Givens (do not re-derive)
- Constant per-word or per-field scalar weights add NO capacity (absorbed in row norms); per-user
  scalars (e.g. 1/|H_u|, softmax normalisers) are killed by the cosine or absorbed by the free ID row.
  Only context-dependent weights, pair/higher-order terms, or SHIFTS (centring) change the function class.
- Already recorded alternatives: demographic mirror (rejected), attribute-anchored user prototypes,
  count-bottleneck user, raw counts / |H_u|^{-1/2} normalisation, field-mean centring on the user side
  (fU′, captured not pursued), shared vs separate word tables fI/fU, TEM rule-tokens pointer,
  Balog et al. 2019 scrutable tag-set user models.
- Measured limitations of the mean-of-words user: activation-cloud effective rank 1.8–2.6;
  prototype profiles near-uniform entropy, p99 pairwise overlap 1.0 (≈5–6 distinct clusters of 76);
  metadata norm shrinks with history length so the ID row's share grows mechanically (norm handover);
  on the item side 85% of the mean composition is a catalog-constant vector; hm median history = 5
  purchases (94% of purchases share their date with another purchase by the same user), ml-1m median 56.
- Dev-tier accuracy (HR@10): fU ≈ host on hm (+0.011), below host on ml-1m (−0.018); the decoupled
  control lightfm_hist beats BOTH by 0.06–0.09. So the main loss sits at the prototype layer, not in
  the composition. Say whether a form would plausibly move that, or only the composition.

## Interpretability contract that any form must be judged against
Exact additive attribution of the score to nameable units (today: purchases and attribute words);
intrinsic prototype naming (prototype ↔ vocabulary rows, no post-hoc probing); explanation object
stays legible for a shopper; cold items (no ID) must still be scoreable via attributes.

## Output rules
Write `Master/temp/user_vocabulary_research_2026-09-07/angle_<X>.md`, ≤ 200 lines:
1. Findings (numbered, decision-grade, each with its source).
2. Candidate forms table: form · attribution unit · exact attribution preserved? · host alteration
   (none/re-weighting/new terms/new object) · cold-item behaviour · thin-history (5 purchases) behaviour ·
   the explanation sentence it licenses (write it out).
3. Citations: verified (you opened the abstract/full text) vs `[unverified]` (memory only). Use
   WebSearch/WebFetch (arXiv, ACM DL abstracts, Semantic Scholar pages). Save any PDF only to the
   scratchpad dir, never into the repo.
Verify every claim; do not invent results or page numbers.
