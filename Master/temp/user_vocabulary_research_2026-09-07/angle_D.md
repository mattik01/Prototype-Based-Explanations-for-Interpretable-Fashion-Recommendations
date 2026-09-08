# Angle D — multi-vocabulary, hybrid and scrutable user models in the transparency literature

Scope reminder: input = which items were bought; no context, no demographics. Every form priced on (1) exact additive
attribution to nameable units, (2) host alteration. "fU" below = `feature_user_proto`, q_u = Σ_f (n_{u,f}/|H_u|) e_f + e_ID(u).

## 1. Findings (decision-grade)

1. **Balog, Radlinski & Arakelyan (SIGIR 2019) is a frequency-weighted mean, exactly fU's family, plus one thing fU lacks: a
   pair vocabulary.** User model W = weighted set of tags; tag preference r̂_t = mean rating over the user's rated items carrying
   tag t (Eq. 1, binary item–tag relevance); weight w_t = r̂_t − r_µ (Eq. 2, deviation from a neutral rating, r_µ = 0 objective,
   user-mean tried and worse). Pairs of co-occurring tags are added as **pseudo-tags** with the *residual* weight
   w_{ta,tb} = r̂_{ta,tb} − w_{ta} (Eq. 3) — the pair row carries only the excess over the first tag. Pairs are **within-item
   co-occurrence** (both tags on the same movie), never across interactions. Source: paper §3.2–3.4 [verified, full text].
2. **Their item score is log-linear in the tag weights, i.e. linear-editable like fU.** Final ranking (Eq. 22):
   log O = log(n_i^+ + 1) − log(|U| + 1 − n_i^−) + Σ_{t∈T+} w_t log P(t|θ_i) − Σ_{t∈T−} w_t log P(t|θ_i), with P(t|θ_i) a
   Beta-smoothed tag presence (Eq. 17). Only tags in the *selected* top-k statements S_k enter ("fully transparent"); using all
   candidates C while verbalising only k = "partially transparent". Scrutinising = overwrite w_t (to ±1) or delete the tag from W
   (§3.4). Popularity priors make it a hybrid; they carry most of the accuracy (Fig. 5, Table 5). [verified]
3. **Their profile is sparsified, not averaged over everything.** Candidate statements require ≥Θ_k covered items and a
   Wilcoxon-significant mean; utility U = coverage · significance · |w_t| (Eqs. 4–6), pairs U(ta,tb) = U(ta) + cov·sig·|w_{ta,tb}|
   over I_ta ∩ I_tb (Eqs. 7–8); greedy MMR-style selection by incremental *item coverage* (Eqs. 9–11) to k statements. Accuracy
   grows from k=5 to 15 then flattens (Fig. 5). Every statement is rendered with a **representative rated item** ("…such as
   The Day After Tomorrow") because pilots showed the example disambiguates the tag (§4.4). [verified]
4. **Human evidence on which unit users endorse (Balog 2019, Table 6, 122 crowd users, top-5 statements).** Two-tag
   statements: full statement + example accepted 27 %, statement w/o example 7 %, first tag + example 23 %, first tag only 10 %,
   **example item only 31 %**, none 2 % (n=143). Single-tag: tag + example 37 %, tag only 20 %, **example only 36 %**, none 7 %
   (n=417). So only 34 % / 57 % agree with the whole two-/one-tag statement, and the *purchased item* is the single most
   reliably endorsed unit; pair statements are endorsed *less* than single-tag ones. 19 %/27 % even rejected that the example
   belonged to the statement. Removing rejected statements ("Corrected") moved MRR 0.835→0.855 with priors, NDCG@10 0.643→0.645,
   not significant (Table 7). [verified]
5. **Tagsplanations (Vig, Sen & Riedl, IUI 2009) — also a frequency-weighted mean, used for justification only.** tag_pref(u,t)
   = (Σ_i r_{u,i}·tag_share(t,i) + k·r̄_u)/(Σ_i tag_share(t,i) + k): ratings weighted by how often the item was tagged t, shrunk
   toward the user's mean by a smoothing constant k for thin evidence; explicitly a *justification* that does not describe the
   underlying algorithm (§Design Space). User study (within-subject, 4 interfaces): RelSort (show relevance+preference, sort by
   relevance) best on justification/effectiveness/mood (p≤.005/.02/.02); RelOnly worst for justification (p≤.001) — **the
   user↔tag preference is what justifies, the tag↔item relevance is the organising order**. Subjective tags beat factual
   (p≤.02/.001/.001), but given the same idea users prefer the plainer factual word (violence 73.8 % vs violent 57.9 %; sexuality
   64.9 % vs sexy 15.4 %); best factual tags are genre/theme level, worst are highly specific ones. [verified]
6. **EFM (Zhang et al., SIGIR 2014) — aspect profile from mention *counts* with a saturating map, then a top-k linear score.**
   X_ij = 0 if never mentioned, else 1 + (N−1)·(2/(1+e^{−t_ij}) − 1) with t_ij the mention count (Eq. 2); ranking
   R_ij = α·Σ_{c∈C_i} X̃_ic Ỹ_jc /(kN) + (1−α)Ã_ij over the user's k most-cared features (Eq. 7; α=0.85, k=10 tuned). Explanation
   template: "You might be interested in [feature], on which this product performs well" (feature = argmax Ỹ among C_i; a
   *dis*-recommendation template uses argmin). Online A/B: CTR 4.34 % (feature explanations) vs 3.22 % ("people also viewed")
   vs 3.20 % (none), p=.033/.041. Score is linear in X̃ (editable in principle; no edit UI offered). [verified]
7. **YouTube DNN (Covington, Adams & Sargin, RecSys 2016) — industrial baseline: mean pooling, but per vocabulary then
   concatenated.** "simply averaging the embeddings performed best among several strategies (sum, component-wise max, etc.)";
   watch history = average of watched-video-ID embeddings, search history = average of unigram+bigram token embeddings, the two
   averages **concatenated** (not summed) with other features before the MLP (§3.2, Fig. 3). No attribution (MLP). [verified]
8. **LACE (Mysore et al., SIGIR 2023) — concept bottleneck with item-level evidence *under* each concept.** Profile = top-P
   concepts retrieved by min L2 distance from the user's document sentences; personalised concept value V_i = Σ_j Q_ji S_j /
   Σ_j Q_ji, a soft-assignment (optimal-transport plan Q) weighted mean of the user's own sentence vectors (Eq. 4); scoring is an
   OT/Wasserstein alignment (not additive). Edits: select/deselect, add, delete, rename a concept → Q and V recomputed. User study
   (20 CS researchers): 20–47 % gains on MRR/NDCG after ≈2.65 tuning rounds; users deleted 5.15 and added 1.25 concepts on average
   and strongly preferred selection over free edits (7.2 selections vs 0.68 additions); initial concept precision 74 %. [verified,
   arXiv HTML]
9. **The NL-profile line (Radlinski, Balog, Diaz, Dixon, Wedin SIGIR 2022; Ramos et al. ACL 2024; TEARS, Penaloza et al. 2024;
   Sanner et al. RecSys 2023).** Radlinski 2022 defines scrutable = "short and clear enough for someone to review and edit
   directly", decomposes into (1) scrutable user-model generation and (2) recommendation *exclusively* from that model, and calls
   the 2019 tag-pair templates a "restricted representation" whose editing "was limited to removing parts"; recommends letting
   users *select/validate* pre-written phrases rather than write. Ramos 2024: LLM-generated NL profiles reach warm-start parity
   with MF/NeuMF on rating prediction; profiles are ~all-positive so negative edits are hard. TEARS: z = α z_summary + (1−α)
   z_blackbox (convex mix, VAE decoder); controllability rises with α, NDCG peaks at intermediate α (inverse-U). Sanner 2023:
   language-only preferences are competitive with item-CF near cold start. None of these preserve exact attribution. [verified:
   abstracts/full text; TEARS formula from PDF]
10. **Balog & Radlinski (SIGIR 2020)**: measured across seven goals, **scrutability is the goal least correlated with the
    others** (satisfaction correlates with many); item-wise evaluation more sensitive than list-wise. Implication: a form can be
    more scrutable without being felt as better. [verified]
11. **Chang, Harper & Terveen (CSCW 2015)**: eliciting preferences over tag-described *groups* of items halves elicitation time
    and raises satisfaction vs rating 15 items — the reason the set-based/tag vocabulary is treated as the natural one. [verified,
    ACM abstract only]
12. **Synthesis for fU.** (a) Every scrutable profile in this literature is frequency-weighted-mean-of-attributes *plus a
    deviation/shrinkage step* (Balog: −r_µ; Vig: shrink toward r̄_u; EFM: saturating counts) *plus sparsification to k units*; fU
    has the mean only. (b) The only pair/co-occurrence unit in the literature is Balog's within-item tag pair, defined as a
    residual; nobody uses cross-interaction pairs. (c) Users most reliably endorse *item examples* and *single genre-level
    words*; multi-clause statements lose agreement. (d) Linear-editable like fU: Balog 2019 (log-linear), EFM (linear top-k);
    not linear: LACE (OT), TEARS/Ramos (LLM), YouTube (MLP). (e) Only EFM-style additive attribute terms touch the score layer;
    everything else is composition-only and would not by itself move the prototype-layer loss the brief identifies.

## 2. Candidate forms

| # | Form | Attribution unit | Exact attribution? | Host alteration | Cold item | Thin history (5) | Explanation sentence licensed |
|---|------|------------------|--------------------|-----------------|-----------|------------------|-------------------------------|
| D1 | **Within-item pair pseudo-words** (Balog Eq. 3): add rows e_{(a,b)} for attribute pairs co-occurring *on one purchased item*, pooled like words; residual-trained (row only carries excess over singles) | word + word-pair | yes (sum over words and pairs) | new additive terms (vocabulary grows; hm: pairs across 5 fields ≤ 10 per item) | fine — item side gets the same pair rows | fine; pairs are per item, 5 purchases → ≤50 pair codes | "Because 3 of your 5 purchases were *denim trousers* — the combination, beyond denim and trousers alone." |
| D2 | **Cross-interaction attribute pairs**: rows for (word a from purchase i, word b from purchase j≠i), i.e. attribute-level co-occurrence *across* the history (Matteo's sentence, made attributable) | history word-pair | yes if q_u sums pair rows; item side must expose the same pairs (via its own attribute pairs) | new object (user-only pair rows; item scoring needs a matching pair readout) | scoreable only via item's own attribute pairs, i.e. D1 on the item side | 5 purchases → C(5,2)=10 item pairs × field pairs; sparse, noisy | "Because you bought *black* and *jersey basics* together in your history, and this item is both." |
| D3 | **Top-k sparsified profile** (Balog S_k; coverage × significance selection, MMR): keep k words, zero the rest | k named words | yes over the kept words | re-weighting with data-dependent 0/1 gate (changes function class; not absorbed) | fine | k ≥ #distinct words usually → collapses to plain mean; coverage criterion trivial at |H|=5 | "Your profile is *ladieswear, black, trousers* — covering 4 of your 5 purchases." Edit: drop a word → next candidate enters. |
| D4 | **Deviation-from-base-rate weights** (Balog w_t = r̂_t − r_µ; Vig shrink toward r̄_u): n_{u,f}/|H| − p_f (catalog share) with count-dependent shrink n/(n+k) | word, signed | yes | SHIFT (centring) + context-dependent shrink → changes function class (= fU′ plus a confidence term) | fine | shrink dominates: 1–2 mentions ≈ neutral; honest under thin data | "You buy *black* about 3× more often than shoppers overall (4 of 5 purchases)." |
| D5 | **Saturating counts** (EFM Eq. 2): replace n_{u,f}/|H| by g(n_{u,f}) = 1 + (N−1)(2/(1+e^{−n}) − 1) | word | yes | context-dependent per-(user,word) weight (sublinear in count) → function class changes | fine | almost linear at n ≤ 5 → indistinguishable from fU on hm; matters on ml-1m (|H|=56) | "You have bought *trousers* repeatedly — repeated purchases count more than a one-off, but with diminishing returns." |
| D6 | **Item examples under words** (Balog §4.4; LACE V_i) — explanation-layer only: each word attribution names the purchase contributing most of that word | word ⟶ purchase | yes (unchanged) | none | n/a | works better: with 5 purchases each word has 1–2 natural examples | "You like *jersey basics*, such as the black T-shirt you bought." |
| D7 | **Concept-with-evidence profile** (LACE additive variant): q_u = Σ_w v_{u,w}, v_{u,w} = mean over purchases carrying w of a *purchase* vector (word-conditioned item vectors) | (word, purchases under it) | yes if purchase vectors are additive; LACE's OT scoring is not | new object (item vectors on the user side, tied or free) | recommended item side unchanged; cold *history* items fall back to words | 5 purchases → each concept backed by 1–3 items | "Under *trousers* your profile is these 3 purchases; the recommended item is close to that group." |
| D8 | **Two-vocabulary concat, YouTube/SVD++ style**: q_u = [mean of history item-ID vectors ; mean of words] (+ e_ID), prototypes block-structured | purchase (by ID) + word | yes per block if the prototype cos numerator is read block-wise | new object (history item-ID table on the user side; block-diagonal prototypes = sum in orthogonal subspaces) | recommended cold item: fine; history items without ID contribute only words | median 5 → ID block is a 5-item mean, high variance | "Because you bought *this exact item* (beyond its attributes) and items like it." |
| D9 | **EFM-style attribute score term**: score += α Σ_{c∈top-k(u)} x_{u,c} y_{i,c}, bypassing prototypes for the user's k most-cared words | word (direct user–item match) | yes | new additive term at the score layer (moves toward `lightfm_hist`, which beats both) | fine (y_{i,c} from item attributes) | top-k = all words at |H|=5 | "You might be interested in *black jersey*, which this product is." |
| D10 | **Disclosed ID-share cap** (TEARS α made static): constrain ‖e_ID‖ ≤ α‖q_words‖ and report the realised share | word vs ID | yes (already) | re-weighting by constraint (learned per-user α is absorbed by the ID row; a *fixed* cap restricts the class) | fine | caps the norm-handover mechanically strongest at |H| small | "At most 30 % of your profile is your personal ID row; the rest is these words." |
| D11 | NL / LLM summary bottleneck (Radlinski 2022, Ramos 2024, TEARS) | free text | **no** | new object | fine | n/a | not admissible under the contract — listed as the literature's direction, not a candidate |

Ranking against the brief's two axes: D6 (none) < D4, D10 (shift/constraint) < D3, D5 (data-dependent re-weighting) < D1, D9
(new additive terms) < D2, D7, D8 (new object). Only D9 plausibly addresses the prototype-layer loss; D1–D8 act on the composition.
Human evidence favours D6 and D3/D4 (item examples + few plain words) and warns against multi-clause pair statements (D1/D2).

## 3. Citations

Verified (full text or abstract opened this session):
- Balog, Radlinski, Arakelyan. Transparent, Scrutable and Explainable User Models for Personalized Recommendation. SIGIR 2019.
  doi 10.1145/3331184.3331211 — full text (Google Research PDF).
- Balog, Radlinski. Measuring Recommendation Explanation Quality: The Conflicting Goals of Explanations. SIGIR 2020.
  doi 10.1145/3397271.3401032 — full text (author PDF).
- Radlinski, Balog, Diaz, Dixon, Wedin. On Natural Language User Profiles for Transparent and Scrutable Recommendation. SIGIR 2022.
  doi 10.1145/3477495.3531873 — full text (author PDF) + arXiv 2205.09403.
- Vig, Sen, Riedl. Tagsplanations: Explaining Recommendations Using Tags. IUI 2009 — full text (GroupLens PDF).
- Zhang, Lai, Zhang, Zhang, Liu, Ma. Explicit Factor Models for Explainable Recommendation Based on Phrase-level Sentiment
  Analysis. SIGIR 2014. doi 10.1145/2600428.2609579 — full text (author PDF).
- Covington, Adams, Sargin. Deep Neural Networks for YouTube Recommendations. RecSys 2016 — full text (Google PDF).
- Mysore, McCallum, Zamani. Editable User Profiles for Controllable Text Recommendation. SIGIR 2023. arXiv 2304.04250 — HTML full text.
- Ramos, Rahmani, Wang, Fu, Lipani. Transparent and Scrutable Recommendations Using Natural Language User Profiles. ACL 2024 —
  full text (ACL Anthology PDF).
- Penaloza, Gouvert, Wu, Charlin. TEARS: Textual Representations for Scrutable Recommendations. arXiv 2410.19302 — full text PDF.
- Sanner, Balog, Radlinski, Wedin, Dixon. Large Language Models are Competitive Near Cold-start Recommenders for Language- and
  Item-based Preferences. RecSys 2023. arXiv 2307.14225 — abstract only.
- Chang, Harper, Terveen. Using Groups of Items for Preference Elicitation in Recommender Systems. CSCW 2015 — ACM abstract only.
- Balog, Radlinski, Karatzoglou. On Interpretation and Measurement of Soft Attributes for Recommendation. SIGIR 2021 — title/venue
  only via search; not used for any claim.

`[unverified]` (memory only): Koren 2008 SVD++ (implicit-feedback item factors summed into the user vector, the ancestor of D8);
Guesmi et al. IUI 2021 open/scrutable interest models (not used).

PDFs saved only under the session scratchpad (`scratchpad/pdfs/`), none in the repo.
