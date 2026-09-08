# Angle C — the user's attribute profile as a distribution / estimator relative to the population

Scope: forms that treat p_u = (n_{u,f}/|H_u|)_f as a *statistical object* — re-weighted (IDF/BM25/SIF), contrasted
with the crowd (centring, lift, PMI), shrunk (Dirichlet, taxonomy), or summarised (entropy). Every form is priced
against the brief's theorem: per-value constants → absorbed in row norms; per-user scalars → killed by the cosine
or absorbed by the free ID row; only context-dependent weights, pair terms, or shifts change the function class.
Notation: m_u = Σ_f p_{u,f} e_f (metadata part), e_u = ID row, μ_vec = Σ_f p̄_f e_f (the crowd's composed vector,
p̄_f = population share of value f), q_u = m_u + e_u.

## 1. Findings

1. **IDF / BM25 / SIF weights are redundant (class i).** All three are a per-value constant α_f multiplying
   p_{u,f}: TF-IDF w = tf·log(N/df) [Pazzani & Billsus 2007, eq. 10.1; verified], BM25 the saturating variant,
   SIF α_f = a/(a+p̄_f) [Arora, Liang, Ma 2017 via offconvex.org; verified]. With fU's own free table,
   α_f e_f is just a re-parameterised row; identical function class. Two residual effects only: (a) optimisation
   path / weight-decay geometry changes (a reparameterisation, not capacity); (b) if the word table were ever
   *shared* with fI, a user-side α_f would break the tie and become a real (but still constant) asymmetry.
   Corollary from Levy & Goldberg 2014 [verified abstract]: SGNS-style embeddings already implicitly factorise a
   shifted-PMI matrix, i.e. learned rows internalise frequency corrections; putting IDF/PMI on the input duplicates
   what the table learns.
2. **Population-mean centring (fU′, all-but-the-top) is a shift that the ID row absorbs in function class but
   not in attribution or optimisation.** Centred: q_u = Σ_f (p_{u,f} − p̄_f) e_f + e_u = m_u − μ_vec + e_u. With
   a free e_u, e_u' = e_u − μ_vec reproduces the uncentred model exactly → class (i) for the *scored function* of
   the ID-bearing model. In `_noid` it is a genuine shift (one shared bias vector; class ii). Yet it matters
   three ways even with the ID row: (a) **attribution**: today the shared component μ_vec is carried either
   inside every ID row ("unnameable") or inside the metadata sum where it is credited to whichever words the user
   happens to hold; centred, it is credited to a nameable unit, "the crowd baseline", and word credit becomes
   *deviation* credit; (b) **geometry**: an uncentred mean-of-words cloud shares one dominant direction μ_vec
   (the item side's 85 % catalog-constant vector is the same phenomenon) — exactly Mu & Viswanath's finding that
   the common mean vector and top directions dominate and hurt [verified abstract]; the measured effective rank
   1.8–2.6 and near-uniform prototype profiles are what a cosine layer sees when every input points at μ_vec.
   Centring removes that direction from what the prototype layer receives, so this is the one form in this angle
   that plausibly moves the *prototype-layer* loss, not just the composition; (c) **regularisation**: the ID rows
   no longer pay L2 for storing −μ_vec, and at init (random e_u) the composed cloud is no longer rank-≈1.
   SIF's second step (remove the projection on the top singular direction of the user matrix) is a fixed linear
   map P applied to m_u; P Σ p e_f = Σ p (P e_f) is absorbed into the table if P is frozen, and acts as a rank
   constraint if recomputed — a regulariser, not new capacity. Field-mean vs global-mean: per-field centring
   subtracts one constant per field; since every hm user holds exactly one value per field per purchase, the five
   field means sum to the global mean and the two are the same shift.
3. **Lift and PMI relate to centring as ratio vs difference; only the clamp is new.** Lift p_{u,f}/p̄_f =
   p_{u,f}·(1/p̄_f) is IDF-shaped → class (i). PMI = log p_{u,f} − log p̄_f: the −log p̄_f term is a per-value
   constant but *added*, so it multiplies the presence indicator, not the count; and log p_{u,f} = log n_{u,f}
   − log|H_u| makes the weight a nonlinear function of the user's own count vector. First-order Taylor around p̄:
   PMI ≈ (p_{u,f} − p̄_f)/p̄_f, i.e. centring times an absorbable per-value constant. The genuinely new content
   of PMI is therefore (a) the log curvature for large ratios and (b) the −∞ at zero, which forces PPMI =
   max(0, ·) — a data-dependent *sparsifier* that keeps only words the user over-indexes on. Class (iii):
   additive in words still holds (one scalar per word), but the scalar depends on |H_u| and the crowd, and words
   the user buys at or below crowd rate contribute nothing ("under-indexing" is unrepresentable). Discontinuous
   at the threshold; at 5 purchases every word has share ≥ 0.2, so PPMI keeps essentially all words whose crowd
   share is < 0.2 — i.e. nearly all of them — and the form degenerates to a presence-set with IDF (class i).
4. **Dirichlet shrinkage toward the crowd is invisible to the score once the ID row exists; it is a
   direction-changing shift only in `_noid`.** p̂_{u,f} = (n_{u,f} + κ p̄_f)/(|H_u| + κ) = λ_u p_{u,f} + (1−λ_u) p̄_f,
   λ_u = |H_u|/(|H_u|+κ) (Zhai & Lafferty's Dirichlet prior; existence verified via TOIS listing, formula
   [unverified, textbook]). Composed: q_u = λ_u m_u + (1−λ_u) μ_vec + e_u. λ_u is a per-user scalar (absorbed);
   (1−λ_u) μ_vec is per-user scalar × constant vector (absorbed by e_u). With the ID row: class (i). In `_noid`:
   the mix λ_u m_u + (1−λ_u) μ_vec changes *direction* with |H_u|, so the cosine sees it: class (iii) shift with
   history-dependent weight — thin users are pulled toward the crowd prototype profile, long users keep their own.
   **Centring and shrinkage are complementary only if μ_vec stays in the representation**: centred + shrunk =
   λ_u (m_u − μ_vec), a pure per-user scalar, dead under the cosine.
5. **The cosine host cannot express evidence mass.** Any honest estimator's confidence (posterior concentration,
   |H_u|, bias-corrected entropy) is a per-user scalar; the prototype layer normalises it away. A 1-purchase user
   activates prototypes as decisively as a 100-purchase user. The only surviving channels for "how sure" are the
   metadata:ID norm ratio (which the free row controls, and which the norm handover already shows drifting
   mechanically) and the item side t_i. Statistical honesty about thin histories therefore has to live in the
   *explanation object*, not in the score.
6. **Taxonomy shrinkage is already built in for hm; hierarchical shrinkage proper is a cross-field term.**
   Ziegler, Lausen & Schmidt-Thieme 2004 [verified full text]: profile score per product s/|R_i|, per descriptor
   s/(|R_i|·|f(b_k)|), propagated to each ancestor with sco(p_m) = κ·sco(p_{m+1})/(sib(p_{m+1})+1), Σ_k v_ik = s.
   In our terms: the per-product and per-user normalisers are dead (theorem); the per-descriptor 1/|f(b)| is a
   per-*purchase* weight — constant on hm (fixed 5 fields) but real on ml-1m bags (3–72 tokens), where it is a
   context-dependent re-weighting (class iii: a purchase with 60 tags stops outvoting one with 4); ancestor
   propagation = emitting parent words alongside leaf words, which hm's field set (department ⊂ section, type
   ⊂ garment family) already does with κ = 1. Genuine hierarchical shrinkage — p̂(type|u) pulled toward the
   population conditional p̄(type | u's section mix) — makes the weight of "type=Skirt" depend on n_{u,section};
   class (iii)/pair term, and its interpretability cost is that a word's credit is no longer its own count.
7. **Identifiability at 5 purchases.** hm fields ≈ department ~250 / type ~130 / section ~56 / colour 50 /
   appearance 30 values in the raw catalog [approximate, unverified counts; the 1-month V = 426]. With N = 5, a
   share vector has support ≤ 5 and granularity 0.2; the MLE and the multiset of words are the same object. Only
   fields with |F| ≲ N distinct plausible values (section at best) are estimable as *distributions*; type,
   department, colour are estimable only as *sets*. Dirichlet MAP with κ ≈ 5–10 gives λ_u ≈ 0.33–0.5, i.e. the
   honest posterior is mostly the crowd. Entropy from 5 draws over K values is biased low by ≈ (K−1)/(2N) nats
   (Miller–Madow) [unverified, textbook]: a 5-purchase shopper looks "focused" by construction. Honest objects at
   this depth: (a) the 5 purchases and their words, verbatim; (b) per word, a *deviation-from-crowd* flag
   (p_{u,f} − p̄_f with its binomial standard error √(p̄(1−p̄)/N) ≈ 0.1–0.2 — so only large deviations are
   real); not a distribution, not an entropy score.
8. **Entropy / concentration as taste descriptors.** Anderson et al. 2020 define a generalist–specialist score
   as average pairwise similarity of a user's streamed tracks and find specialists benefit more from
   personalised recommendations [verified via Spotify Research page + S2 abstract]. As a *global* per-user
   scalar the descriptor cannot enter the additive score (theorem, finding 5). It survives only as a
   **per-user-per-field** gate c_{u,F} = 1 − H_{u,F}/log|F| multiplying the words of field F (class iii): "you are
   consistent about colour, eclectic about type → colour words get more say". Additive attribution per word is
   kept, but the word's weight now cites the whole field. Noisy at N = 5 (finding 7). As pure *display*
   ("focused shopper: 4 of 5 items black") it is free and honest if quoted as counts, not entropies.
9. **Precedent for deviation-as-unit in explanations.** Vig, Sen & Riedl 2009 [verified full text]: tag
   preference tag_pref(u,t) = (Σ_i r_{u,i}·tag_share(t,i) + r̄_u·k)/(Σ_i tag_share(t,i) + k) is a *shrunk*
   estimate (k = 0.05 toward the user's own mean r̄_u), and tag relevance correlates users' (tag_pref − r̄_u)
   deviations; their study reports tag preference as the stronger element for justification. Balog, Radlinski &
   Arakelyan 2019 [verified abstract]: set-based user models presented in natural language, scrutable/editable
   — already recorded in the brief. Both present the profile as a small nameable set with a signed deviation,
   which is exactly the object finding 7 says is honest at thin depth.
10. **Rocchio is the closest classic to "centred mean-of-words".** Q_{i+1} = αQ_i + β Σ_rel D/|rel| − γ Σ_nonrel
    D/|nonrel| [Pazzani & Billsus 2007 eq. 10.2; verified]. With "non-relevant" = the catalog (or the population
    profile), the γ term *is* population centring, with β/γ setting how far below the crowd the baseline sits.
    Framing fU′ as Rocchio with the crowd as negative prototype gives it a 1971 citation trail and a knob (β/γ)
    that is itself a per-user-constant scalar pair → still absorbed by the ID row, still a shift in `_noid`.

## 2. Candidate forms

| Form | Attribution unit | Exact additive attribution preserved? | Host alteration | Cold-item behaviour | Thin-history (5) behaviour | Explanation sentence it licenses |
|---|---|---|---|---|---|---|
| C1 IDF / BM25 / SIF per-value weight | words | yes (weight folds into e_f) | none (redundant, class i) | unchanged (user side) | unchanged | none new — same sentence as today |
| C2 Population-mean centring (fU′; = all-but-the-top step 1; = Rocchio with crowd negative) | words as *deviations* + one nameable "crowd baseline" unit | yes: score = Σ_f (p_{u,f}−p̄_f)·⟨e_f,·⟩ + crowd term + ID term | shift: class (i) with ID row (re-attribution + geometry), class (ii) in `_noid` | unchanged | deviations large and noisy (SE ≈ 0.1–0.2); baseline term dominates — honest if shown | "Compared with the average shopper you buy far more *black* and *skirts*; that is what matches this item. The rest is what everyone buys." |
| C3 SIF top-direction removal on user matrix | words (projected) | yes (linear) | regulariser / absorbed if frozen (class i) | unchanged | unchanged | none new |
| C4 PPMI user weights max(0, log p_{u,f} − log p̄_f) | over-indexed words only | yes per word, but weight depends on |H_u| and crowd; under-indexed words vanish | re-weighting, class (iii); discontinuous | unchanged | degenerates to presence-set + IDF (≈ class i) | "You buy *satin* 6× more often than average — this item is satin." |
| C5 Dirichlet shrinkage toward crowd (λ_u m_u + (1−λ_u) μ_vec) | words + crowd unit, mixed by history length | yes; crowd share (1−λ_u) is itself a named term | class (i) with ID row; class (iii) shift in `_noid` | unchanged | crowd term carries ≥ 50 % of the profile at N = 5 — the honest statement | "With only 5 purchases we lean half on what shoppers like you buy; the other half is your own *black* and *jersey*." |
| C6 Hierarchical (taxonomy-conditional) shrinkage, p̂(type\|u) toward p̄(type\|u's sections) | leaf words, credited via parent words | no longer exact per word: a leaf's weight cites its parent's count (pair term) | new terms / class (iii) | unchanged | pulls fine fields toward what the user's coarse fields predict — plausible, but credit crosses fields | "You buy *Ladieswear → Dresses*; dresses shoppers like you also buy *midi skirts*." |
| C7 Per-purchase 1/|f(b)| normalisation (Ziegler descriptor split; ml-1m bags only) | purchases and words | yes (weight per purchase) | re-weighting, class (iii) on ml-1m; dead on hm | unchanged | equalises long-tag movies vs short-tag movies | "Each film you watched counts equally, however many tags it carries." |
| C8 Per-user-per-field concentration gate c_{u,F} | words, gated by field consistency | yes per word; weight cites the field's spread | re-weighting, class (iii) | unchanged | gate biased toward "focused" at N = 5; needs bias correction or a floor | "You are very consistent about *colour* (4 of 5 black), so colour drives this match." |
| C9 Entropy / GS-style diversity as display only | none (score untouched) | n/a | none | n/a | quote counts, not entropy | "A focused shopper: 4 of your 5 items are black jersey tops." |
| C10 Deviation-from-crowd *explanation layer* on the current model (present today's word credits ordered/annotated by p_{u,f} − p̄_f and its SE) | words + "crowd" annotation | yes (presentation only) | none | unchanged | shows which of the ≤ 25 words are actually unusual; the rest labelled "typical" | "Of your purchases' 25 attribute words, 3 are unusual for a shopper here: *satin*, *sequin*, *party dress*. This item shares two of them." |

Ranking within this angle: C2 (fU′) is the only form with a mechanistic case for touching the prototype-layer
loss (removes the shared direction the cosine cannot ignore; Mu & Viswanath / SIF precedent) *and* it improves
attribution semantics; C10 is free and should ship regardless; C5 is the honest thin-history story but is
score-invisible with the ID row; C4/C6/C8 buy new behaviour at the cost of word credit depending on context.

## 3. Citations

Verified (abstract or full text opened):
- Pazzani & Billsus 2007, "Content-Based Recommendation Systems", The Adaptive Web, LNCS 4321 ch. 10 — full text
  (cs.fit.edu mirror): eq. 10.1 tf·idf, §10.6 Rocchio eq. 10.2, structured-attribute profiles §10.2.
- Ziegler, Lausen & Schmidt-Thieme 2004, "Taxonomy-driven Computation of Product Recommendations", CIKM — full
  text (aminer mirror): §3.2 eqs. 1–3, κ = 0.75, per-descriptor score split.
- Mu & Viswanath 2018, "All-but-the-Top", arXiv:1702.01417 — abstract.
- Arora, Liang & Ma 2017, "A Simple but Tough-to-Beat Baseline for Sentence Embeddings", ICLR — method verified
  via the authors' offconvex.org post (a/(a+p_w), top-singular-direction removal, c_0 discourse vector);
  OpenReview page itself blocked by verification wall.
- Levy & Goldberg 2014, "Neural Word Embedding as Implicit Matrix Factorization", NIPS — abstract (papers.nips.cc).
- Vig, Sen & Riedl 2009, "Tagsplanations", IUI — full text (grouplens.org): tag_pref formula, k = 0.05,
  tag_rel via Pearson on (tag_pref − r̄_u).
- Balog, Radlinski & Arakelyan 2019, "Transparent, Scrutable and Explainable User Models", SIGIR — abstract
  (Semantic Scholar API + research.google).
- Anderson, Maystre, Anderson, Mehrotra & Lalmas 2020, "Algorithmic Effects on the Diversity of Consumption on
  Spotify", WWW — abstract (Semantic Scholar API) + GS-score description (research.atspotify.com).
- Lops, de Gemmis & Semeraro 2011, "Content-based Recommender Systems: State of the Art and Trends", RecSys
  Handbook ch. 3 — existence/venue verified (Springer listing); content claims not drawn from it.

[unverified] (memory / textbook only):
- Zhai & Lafferty 2001/2004 Dirichlet-prior formula p(w|d) = (c(w,d) + μ p(w|C))/(|d| + μ) — paper existence and
  "smooths longer documents less" verified via secondary listings; formula from memory.
- Miller–Madow entropy bias (K−1)/(2N).
- hm per-field cardinalities (~250/130/56/50/30) — approximate recollection of the raw Kaggle catalog.
