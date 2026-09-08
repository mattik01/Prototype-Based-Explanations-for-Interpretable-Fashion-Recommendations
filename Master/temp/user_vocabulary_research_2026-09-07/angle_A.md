# Angle A — the purchased items themselves as the user vocabulary (2026-09-07)

Scope of this angle: item-granular history composition. The user is a bag of *purchases*; each purchase
contributes either its composed attribute rows (today), a free per-item history row (SVD++ `y_j`, FISM `q_j`),
or both. Notation: `H_u` train history, `c_j = Σ_f e_f` the word-composition of item j, `y_j` an item-ID
*history* row (distinct from any target-side item vector), `e_ID(u)` the free user row.
Code fact checked (`feature_extraction/feature_extractors.py::HistoryFeatureEmbedding`): q_u is a static
per-user buffer pooled over the FULL train history, no target exclusion; user-ID rows live in the same
embedding table as the words (offset `n_features`).

## 1. Findings (decision-grade)

1. **The canonical literature form of "user = sum of history items" is SVD++ / NSVD / FISM, and it always
   uses a *separate* history table.** SVD++: `r̂_ui = μ+b_i+b_u + q_iᵀ(p_u + |R(u)|^{-1/2} Σ_{j∈R(u)} y_j)`
   (Koren & Bell chapter eq. 3, verified); the `y_j` are "a second set of item factors" distinct from the
   target-side `q_i`. NSVD (Paterek) drops `p_u` entirely and "models users based on the items that they
   rated" (Koren, Bell, Volinsky 2009, verified). FISM: `ŷ_ui = p_iᵀ(|R_u^+|^{-α} Σ_{j∈R_u^+\{i}} q_j)`,
   "each item has two embedding vectors p and q to differentiate its role of a prediction target or a
   historical interaction, which can also increase model expressiveness" (NAIS §2, eq. 3, verified).
   → For fU the item-ID history row `y_j` must be a table of its own anyway: the host's item vector `t_i` lives
   in prototype space R^L, `y_j` in word space R^d — they cannot be shared. The shared-vs-separate question
   therefore only concerns the *word* table (fI vs fU) and is unchanged by this angle; the item-ID row adds no
   new sharing decision. (Precedent for asymmetric tables being the more expressive choice: FISM/NAIS above.)

2. **Attribution unit and exactness.** In every dot-product member of the family the score decomposes exactly
   per history item: `ŷ_ui = Σ_j n_u^{-α} ⟨p_i, q_j⟩` — one nameable term per purchase ("because you bought j").
   Koren & Bell: "the neighborhood framework allows identifying which of the past user actions are most
   influential on the computed prediction" (verified). In fU the same holds one level down: `q_u·p_l` splits
   exactly into per-purchase terms, and each purchase term splits further into per-word terms (and a per-item-ID
   term if `y_j` is added). The cosine only introduces the shared per-user factor `1/‖q_u‖`, so shares of a
   prototype activation still sum to one — identical status to today's word attribution. Adding `y_j` therefore
   changes the attribution unit from {purchase → words} to {purchase → words + "this item itself"}; exactness
   is preserved. The `+1` shift term remains attributable to nothing (unchanged).

3. **History normalisation (1/|H|, |H|^{-α}) — literature vs the givens.** FISM tunes α∈[0,1] because its score
   is a plain dot product with no user row (Pan's FISM slides state the rule with `0≤α≤1`; NAIS tests α from
   0 to 1, verified). NAIS then notes the `|R_u^+|^{-α}` coefficient "is aborted into the attention weight
   without affecting the representation power" (verified) — i.e. any per-user scalar is absorbed as soon as a
   learned per-purchase weight exists. In fU: (a) `_noid` + cosine: α is exactly irrelevant; (b) with `e_ID`:
   the function class is unchanged (e_ID rescales itself), only the regularised solution moves. Adding `y_j`
   rows does NOT change this: `y_j` is scaled by the same `1/|H|` as the words, so α is still a user-constant
   on the history block. Verdict: the normalisation question is closed for this angle by the givens; do not
   spend a run on it. The Rendle-FM encoding `x_j = 1/√|N_u|` (Rendle 2010 §V-B, verified) is the same fact
   seen from the FM side.

4. **Norm handover: yes, item-ID rows inherit it, and they add a second, worse handover.** The mean of words
   shrinks with |H| because idiosyncratic word directions average out (given). A mean of free `y_j` rows shrinks
   at least as fast: `y_j` has no catalog-constant component unless it learns one, so `‖(1/|H|)Σ y_j‖ ~ |H|^{-1/2}`
   in the generic case. Consequences: (i) against `e_ID(u)` the handover is the same as today (long histories
   → ID dominates); (ii) *inside* the history block, no handover between words and `y_j` — both carry the
   same `1/|H|`; (iii) but `y_j` is free and per-item, so the optimiser can grow `‖y_j‖` for frequently
   bought items until the item-ID part dominates the word part → the word vocabulary is silently
   demoted at popular items. SVD++ regularises `y_j` with the same λ as `q_i`; fU would need a norm read-out
   (share of `‖Σ y_j‖` vs `‖Σ c_j‖` per user) as a disclosed diagnostic, not a design assumption.

5. **Is a per-purchase item-ID row a "covert" per-user ID?** Only in a rank sense. The history block is
   `Y_hist = D^{-1} H Y` with `H` the (n_users × n_items) train matrix; it can emulate arbitrary user vectors
   iff `rank(H) = n_users`, which requires n_items ≥ n_users *and* linearly independent baskets. ml-1m
   (6 040 users, 3 706 items) fails this outright; hm_1_month must be checked (users vs items count) — if
   users ≫ items it is not covert. Even where rank allows it, weight decay prefers the shared solution:
   two users with the same basket get the same vector. Literature reading of the same question: Koren 2008
   kept BOTH `p_u` and `Σ y_j` in SVD++ and reported gains over either alone, i.e. history-ID rows are not
   redundant with a free user row in practice [unverified numeric detail; the model structure is verified
   via Koren & Bell]. Practical risk for fU is the opposite: with `e_ID` present the free user row can absorb
   the fit and `y_j` never learns beyond the words → any `+y_j` run must be paired with its `_noid` twin.

6. **Target exclusion (FISM's `\{i}`) is a live issue for fU today, and becomes acute with item-ID rows.**
   FISM's defining property: "the value being estimated is not used for its own estimation" (abstract,
   verified); NAIS: `\{i}` exists "to avoid the modeling of self-similarity of the target item" (verified).
   fU pools the full train history into a static buffer, so during training the positive item's own words
   are inside `q_u` — at hm median |H|=5 that is 20 % of the history mass, and at test time the target is
   never in the history (train/test mismatch). With words alone this is a mild content-leak; with a free
   `y_i` row it lets the model learn `y_i ∥ (direction that lights t_i's prototypes)` = memorisation of the
   training positive. Any item-granular form needs leave-one-out pooling in training (drop item i's rows and
   renormalise) — a dataset/forward change, not a vocabulary change; it also applies to the current fU and to
   `lightfm_hist` and could by itself move numbers on hm (cheap ablation, high information).

7. **Thin history (5 purchases).** The item-to-item lineage is the one built for this regime: Amazon's
   item-to-item CF "performs well with limited user data, producing high-quality recommendations based on as
   few as two or three items" (Linden et al. 2003, verified). Per-purchase item-ID rows give a 5-purchase user
   five strong shared rows instead of ~20 catalog-dominated word rows (85 % constant, given). Attention over
   5 items (NAIS) has little to discriminate and its long-history motivation (softmax "may overly punish the
   weights of active users with a long history", verified) is an ml-1m concern (median 56), not an hm one.

8. **Cold items.** User side: a cold item never appears in a train history, so `y_j` is never missing where
   it is needed; if a warm-trained model is later fed a history containing an unseen item, the natural rule is
   "words only, no ID row" (LightFM's linear-combination-of-features model handles exactly this on both sides,
   abstract verified; Koren & Bell: "for items new to the system we do have to learn new parameters",
   verified). Item side: in the fU host the candidate is a free `t_i` — cold candidates are a host problem
   this angle cannot fix; only the fUfI merge (composed item side) does. Any item-item term (form A5 below)
   must be *optional per candidate*: absent `t'_i` → term = 0, score falls back to the prototype path.

9. **Does the "two purchases co-occur" sentence have an exact additive item-granular form?** Rendle 2010
   shows the FM on SVD++ features produces, beyond SVD++, "interactions between pairs of movies in N_u"
   (verified) — but those pair terms do not involve the target item, so they cancel in ranking. A pair term
   that affects the ranking must be third-order (target × history-pair), which is DeepICF's "interaction
   among all interacted item pairs … which historical itemsets in a user's profile are more important"
   (abstract, verified; nonlinear there, no exact attribution). The additive item-granular version is a
   pair-row vocabulary `y_{jj'}` (form A6): exact per pair, but it is a new object with O(|H|²) rows per
   user, per-pair support is tiny, and "co-occur" would mean set co-membership, since basket structure is out
   of scope. Honest verdict: the sentence Matteo wants is affordable only as a *readout* on an item-item
   term (A5: "two of your purchases both point at this item"), not as a learned pair vocabulary.

10. **Where the accuracy sits.** The given says lightfm_hist (plain dot) beats fU by 0.06–0.09 with the same
    `q_u`. Adding `y_j` rows changes only the composition, so it would lift lightfm_hist (SVD++ evidence that
    implicit item factors help, Koren & Bell verified) but leaves the prototype-layer gap. The one form here
    that plausibly moves the fU number is A5: an additive FISM/SVD++-style item-item term *beside* the
    prototype score (Koren 2008's integrated neighborhood+factor model is the precedent) — it imports exactly
    the dot-product piece that wins, while the prototype explanation stays whole and the new term is
    per-purchase transparent. Price: a "new additive term" in the score, a second explanation object.

11. **User-facing value of per-purchase explanations is real but not the top rung.** Herlocker et al. 2000
    Table 1 (verified): "Similarity to other movies rated" mean 4.97 (rank 5 of 21), below the neighbors'
    rating histogram (5.25) and past performance (5.19). Koren & Bell: item-item methods "are more amenable to
    explaining … because users are familiar with items previously preferred by them" (verified). Caveat for
    A4: attention weights are coefficients, so attribution is exact by construction, but Jain & Wallace 2019
    show attention weights are not faithful *importance* explanations in general (verified) — disclose them
    as weights, not as "why".

12. **Negative precedent worth recording.** Meta-Prod2Vec does NOT sum ID and metadata rows; it injects
    metadata "as side information to regularize the item embeddings" (abstract, verified) — a side-loss, not
    a vocabulary, so no additive attribution. De Pauw, Ruymbeek & Goethals 2022 build "user profiles within the
    domain of item metadata" with recommendations "computed as a linear function of item metadata and the
    interpretable user profile" (abstract, verified) — the closest published cousin of fU's contract; it
    argues the metadata-domain user is a legitimate end point, not a stopgap.

## 2. Candidate forms

| form | attribution unit | exact attribution? | host alteration | cold item | thin history (5) | explanation sentence |
|---|---|---|---|---|---|---|
| **A1 words + item-ID history row** `q_u=(1/|H|)Σ_j(c_j+y_j)+e_ID` (SVD++ shape) | purchase → {its words, "this item itself"} | yes (per-prototype dot; shared 1/‖q_u‖ as today) | new additive term in q_u (free row per item); prototype layer untouched | user side: words only if item unseen; item side: host problem | 5 strong shared rows; ID-row norm growth at popular items must be read out (F4) | "Prototype *Casual knitwear* lights up 40 % because of the 2 cardigans you bought — 25 % from their attributes, 15 % from those exact articles." |
| **A2 A1 without e_ID** (NSVD / FISM shape) | purchase → words + item row | yes | as A1, minus the free user row | as A1 | history carries everything; no ID to absorb (F5) | "This prototype is lit only by what you bought: article X (30 %), article Y (20 %) …" |
| **A3 leave-one-out pooling in training** (FISM `\{i}`; applies to today's fU too) | unchanged | unchanged | none at inference; training-time forward/dataset change | none | removes the 20 % self-term at |H|=5 (F6) | (no new sentence — fixes a train/test mismatch) |
| **A4 target-conditioned per-purchase attention** (NAIS, smoothed softmax β) | purchase, with a disclosed weight a_{ij} | yes, given the weights (coefficients of an exact sum) | re-weighting; q_u becomes q_{u,i} → prototype bars differ per candidate | as A1 | little to weight among 5 items; β issue is ml-1m's | "For this item we weighted your purchase of X 3× more than Y; X contributes 45 % of the score." (weights disclosed as weights, F11) |
| **A5 prototype score + additive item-item term** `s = Σ_l t_{il}(1+cos(q_u,p_l)) + |H|^{-α}Σ_j⟨t'_i, y_j⟩` (Koren 2008 integrated shape) | second term: purchase; first term: as today | yes — two exact sums, disclosed side by side | new additive term in the score; prototypes unchanged | second term = 0 for candidates without t'_i; prototype path scores them | each purchase is one direct term; the piece lightfm_hist wins with (F10) | "Because you bought X and Y (direct-item part +0.31) and because you match prototypes *Basics* and *Denim* (+0.52)." |
| **A6 co-purchased pair rows** `y_{jj'}` in q_u or in A5's term | purchase pair | yes per pair, but only ranking-relevant if the pair term meets the target (F9) | new object (pair vocabulary, O(|H|²)) | pairs with a cold member: none | 10 pairs per user, each with tiny support | "Because you bought both X and Y." — affordable only as an A5 readout, not as a learned vocabulary |

Not tabled: post-hoc Amazon-style "customers who bought X also bought" lookups (Linden 2003) — violates the
intrinsic-naming clause; DeepICF-style nonlinear pair pooling — no exact attribution.

## 3. Citations

Verified (abstract or full text opened):
- He, X. et al. *NAIS: Neural Attentive Item Similarity Model for Recommendation*, TKDE 2018, arXiv:1809.07053 — full text (eq. 3, eq. 8, §3.2 smoothing, §4.2.1).
- Xue, F. et al. *Deep Item-based Collaborative Filtering for Top-N Recommendation* (DeepICF), TOIS, arXiv:1811.04392 — abstract.
- Kabbur, Ning, Karypis. *FISM: Factored Item Similarity Models for Top-N Recommender Systems*, KDD 2013 — abstract (experts.umn.edu record); prediction rule via NAIS eq. 3 and W. Pan's FISM slides (csse.szu.edu.cn). The KDD PDF itself could not be opened (mirrors dead).
- Koren, Y., Bell, R. *Advances in Collaborative Filtering* (Recommender Systems Handbook chapter) — full text: SVD++ eq. 3, explainability/new-ratings passages, "items new to the system".
- Koren, Bell, Volinsky. *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer 2009 — full text (NSVD/Paterek description, eq. 9 normalisations).
- Linden, Smith, York. *Amazon.com Recommendations: Item-to-Item Collaborative Filtering*, IEEE Internet Computing 2003 — full text.
- Rendle, S. *Factorization Machines*, ICDM 2010 — full text (§V-B SVD++ encoding and the extra pair terms).
- Chen, T. et al. *SVDFeature: A Toolkit for Feature-based Collaborative Filtering*, JMLR 13 (2012) — full text (model eq., user/item = Σ feature factors).
- Kula, M. *Metadata Embeddings for User and Item Cold-start Recommendations* (LightFM), arXiv:1507.08439 — abstract.
- Herlocker, Konstan, Riedl. *Explaining Collaborative Filtering Recommendations*, CSCW 2000 — full text, Table 1.
- Jain, S., Wallace, B. *Attention is not Explanation*, NAACL 2019, arXiv:1902.10186 — abstract.
- Vasile, Smirnova, Conneau. *Meta-Prod2Vec*, RecSys 2016, arXiv:1607.07326 — abstract.
- De Pauw, Ruymbeek, Goethals. *Modelling Users with Item Metadata for Explainable and Interactive Recommendation*, arXiv:2207.00350 (2022) — abstract.
- Covington, Adams, Sargin. *Deep Neural Networks for YouTube Recommendations*, RecSys 2016 — abstract only (the averaged-watch-embedding detail is not in the abstract; not relied on above).

`[unverified]` (memory only):
- Koren, Y. *Factorization Meets the Neighborhood*, KDD 2008 — the Asymmetric-SVD variant and the "both p_u and Σy_j beat either alone" claim; PDF mirrors returned 403/410. Structure cross-checked through the Koren & Bell chapter.
- Sarwar et al. *Item-based Collaborative Filtering Recommendation Algorithms*, WWW 2001 — title/venue only (Semantic Scholar record, no abstract); cited nowhere load-bearing.
