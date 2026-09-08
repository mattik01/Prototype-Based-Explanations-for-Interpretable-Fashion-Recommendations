# Angle B — structure *within* a user's history: pairs, co-occurrence, sets, rules (2026-09-07)

Motivating sentence (Matteo): "we recommend this, because these two interactions in your history strongly co-occur."
Host recap (brief): q_u = Σ_f w̄_{u,f} e_f + e_ID(u), mean-pooled words, then score = Σ_l t_{i,l}(1+cos(q_u,p_l)).
Sizes used below (repo facts): hm 5 fields (department, product_type, section, colour_group, graphical_appearance;
`feature_extraction/feature_ids.py`), V=426, one value per field; ml-1m bags, 1,061 token codes (`sc01_chapter_draft.md` l.34);
hm median history 5, 94.3% same-day (`sc05_chapter_draft.md` l.340); ml-1m median 56.

## 1. Findings (decision-grade)

**F1. The motivating sentence is, in the strict sense, never faithful for a single hm user — "strongly" is a population word.**
With only *which items* as input, a pair of things in one history is present once or absent; there is no within-user
frequency from which "strongly co-occur" could be read (5 purchases → 10 item pairs, each seen once). Any strength weight
attached to a pair is either learned across users (an FM-type factor ⟨v_a,v_b⟩, an association-rule confidence, a co-purchase
edge weight) or a product of the user's own marginal frequencies (see F3). The faithful split is therefore always two-part:
*presence* is the user's ("you bought A and B"), *weight* is the population's ("shoppers who buy A with B tend to …"). Any
form tabled below is graded on whether it says both halves out loud. (Sources: FM/NFM/AFM pair terms, F3; triple2vec F6;
ItemRank F8.)

**F2. Word-count vector = information ceiling for cross-item word pairs; the only genuinely new pair information in a history
is *within-item* conjunction (and item identity).** The mean-of-words user is a function of the history's word-count vector
n_u ∈ N^V. A cross-item word pair (a from item j, b from item k) with weight n_a·n_b is a *deterministic quadratic function of
n_u* — it adds a nonlinearity, zero new information about the history. What n_u loses is (i) which words sat together in the
same item (conjunctions: black×dress vs black×skirt) and (ii) item identity. So on the user side, "pairs" that carry new
signal are exactly the item-level conjunctions — which are item attributes, computable on the item side too (fI's E3 rival
explanation in `dc06_…md` R6 row). Consequence: the informationally honest "pair" candidate is a **vocabulary extension with
conjunction tokens** (TEM-style cross features as tokens), inherited by fU's mean unchanged — not a pair term over the history.

**F3. FM / NFM / AFM second-order terms restricted to the user's basket: there *is* a term indexed by each pair, but its weight
is the independence expectation.** FM's pairwise term Σ_{i<j}⟨v_i,v_j⟩x_i x_j (Rendle 2010, restated in AFM §2; verified) and
NFM's Bi-Interaction pooling f_BI(V_x)=Σ_{i}Σ_{j>i} x_i v_i ⊙ x_j v_j, computable via ½[(Σ x_i v_i)² − Σ (x_i v_i)²] (He & Chua
2017, Eqs. 3–4; verified) give, for x = the user's word frequencies, a per-pair vector w̄_a w̄_b (e_a⊙e_b). The pair *is* a score
term (faithful attribution unit), but its coefficient w̄_a w̄_b is what you would expect under independence — it measures
"both present", not "co-occur strongly". AFM replaces sum-pooling with learned attention a_{ij} over pairs and claims this
"greatly enhances the interpretability and transparency of FM" (Xiao et al. 2017, abstract; verified): the weights become
context-dependent (function class changes) but come from an opaque one-layer MLP — an exact multiplier, not a nameable
reason. Two host consequences: (a) the NFM identity shows the pair mass ≈ ½ q_meta² − diag; with 85% of the item composition a
catalog-constant vector c, the pair term ≈ ½ c⊙c + c⊙δ — it *amplifies* the common component (D3 anisotropy, eff. rank 1.8–2.6)
unless the words are centred first (fU′ prerequisite; scratchpad l.121); (b) with a free ID row, q_u ∈ R^d is already an
arbitrary per-user vector — a pair term cannot enlarge the warm ids-arm function class at all; it can only change inductive
bias, the noid arm, and the explanation. This is the D2 lesson in its general form: **any pair structure that is a fixed
function of the train history is absorbable by e_ID(u), and the optimizer has no reason to route it through the pair table.**
Faithfulness risk: the explanation shows pair shares the model did not need.

**F4. Pair-vocabulary sizes.** hm, 5 purchases × 5 words: 25 word instances → 300 instance pairs = 50 within-item (5×C(5,2))
+ 50 cross-item same-field + 200 cross-item cross-field; item pairs 10. Pair *types*: cross-field value pairs ≤ C(426,2)=90,525
(minus same-field pairs) — an explicit pair-row table of ~10⁴–10⁵ × d is feasible but each row is seen by few users at 5
purchases. ml-1m, 56 movies × 3–72 tokens (≈20 avg): ≈1,100 token instances → ≈600k cross-movie instance pairs per user,
≈10⁴ within-movie; pair types ≤ C(1061,2)≈562k. Cross-item pair enumeration on ml-1m is not renderable and must be pooled to
pair-type counts (which by F2 is again a function of n_u). Explicit pair rows only make sense for *within-item* conjunctions,
support-pruned (TEM prunes by tree leaves; association rules by min-support).

**F5. Attribute-value pairs and item pairs are different beasts for the contract.** Attribute pairs (colour×garment) can have
their own row → prototypes stay intrinsically nameable (cos(e_{ab},p_l)), cold items are scoreable (their conjunctions are
known), and the sentence is shopper-legible ("black dresses"). Item pairs (j,k ∈ H_u) need either an |I|² table (infeasible)
or a factorization e_j⊙e_k over *item ID rows* — outside the shared vocabulary, so prototype naming falls back to post-hoc
nearest-items (the ProtoMF default the lineage rejects), the noid arm has no item rows at all, and every item-pair term is a
pure function of identity → maximally D2-absorbable. Item pairs are the fUfI-merge's business, not fU's.

**F6. Basket-completion models define pair vocabularies *within a basket*, and their item–item term is population-learned.**
triple2vec trains on triples T={(i,j,u): i,j ∈ I_b, i≠j, b ∈ B_u} and scores s_{i,j,u} = f_iᵀg_j + f_iᵀh_u + g_jᵀh_u, the first term
labelled "item-to-item complementarity", the others "user-to-item compatibility" (Wan et al. CIKM 2018, Eqs. 2–3; verified).
DREAM builds a basket vector by max- or average-pooling item embeddings (Yu et al. SIGIR 2016, Eq. 1; verified) — average
pooling is exactly fU's ingredient, and the paper argues max pooling wins because average pooling treats items "in an
independent way" (verified). Sceptre infers substitute/complement links *between product pairs in the population* from review
text, as supervised link prediction (McAuley et al. KDD 2015, abstract; verified); P-Companion conditions on the query
product only, not on the user's history (Hao et al. CIKM 2020, Amazon Science page; verified). None of these has a term
indexed by a pair *inside one user's history with user-specific strength*; the pair weight is always a shared parameter.

**F7. Same-day structure makes hm "pairs in your history" ≈ "pairs in your order" by accident.** 94.3% of hm train purchases
share their date with another purchase by the same user; the median user's purchases all fall in same-day orders (1.73
purchase days). So a cross-item pair term on hm is de facto a within-basket complementarity term (F6's f_iᵀg_j) even though
order structure is out of scope — the model would learn basket co-purchase and the explanation would call it taste. On ml-1m
the same term means "two movies you both rated" with no basket semantics. One form, two meanings: a disclosure obligation,
not a design win.

**F8. Graph views of a single history: personalised PageRank / ItemRank have NO within-history pair term.** ItemRank is
IR_u = α·C·IR_u + (1−α)·d_u with d_u the user's own rated items, C the population co-rating correlation matrix (Gori & Pucci
IJCAI 2007, Eq. 4; verified). Closed form IR_u = (1−α)(I−αC)⁻¹ d_u is *linear in d_u*: seeds never interact, so each purchase's
contribution is exactly additive (good for attribution) but "these two purchases co-occur" is not a statement about anything
the score used. RecWalk keeps the same restart-linear structure (Nikolakopoulos & Karypis WSDM 2019; abstract only, [unverified]
for the linearity claim). The only graph form with a real within-history pair is the *induced co-purchase subgraph*: weight
each history item by its population co-purchase edges to the other history items (F-B6 below) — a context-dependent
re-weighting, sentence "we counted X more because it goes with three other things you bought."

**F9. Deep Sets / Janossy k=2.** Deep Sets: any permutation-invariant set function is ρ(Σ φ(x)) (Zaheer et al. 2017, Thm 2;
verified). Janossy k-ary pooling averages f over all k-subsequences; Thm 2.1: F_{k−1} ⊊ F_k, and Corollary 2.1: DeepSets (k=1)
"pushes the modeling of k-ary relationships to ρ" (Murphy et al. ICLR 2019; verified). Mapping to fU: φ = word-row lookup
(linear), Σ = the mean, ρ = cosine-prototype layer. So fU is k=1 with a *linear* φ — the count bottleneck of F2 — and the
prototype layer is the only place pairs could be reconstructed, which it structurally cannot from a mean of linear rows.
k=2 Janossy with f = elementwise product is F3; with f = MLP the pair contribution stays exactly additive but is a black-box
vector, no vocabulary row → fails intrinsic naming. Janossy is the theory frame; it does not add a form beyond F3.

**F10. Association-rule / itemset profiles are the one family where "strongly" is honest — as a population statement.** Rules
{A,B}→C mined over users carry confidence/lift; a user profile = the antecedents the history satisfies; score contribution per
fired rule is additive. Sentence: "you bought A and B; N% of shoppers who bought both also bought C" — presence the user's,
strength the population's, both said. Costs: rules with item consequents cannot fire on a cold target (attribute-level rules
{black,dress}→{skirt-family} fix that); at 5 purchases few antecedents fire; rules are a fixed function of history → D2-absorbed
in the warm ids-arm. TEM is the learned version: GBDT leaves = cross-feature rules used as tokens, embedded and attention-pooled,
"generalize to unseen cross features on user ID and item ID" (Wang et al. WWW 2018, abstract via ACM/GitHub; formula details
[unverified], full text not reachable this session). Balog et al. SIGIR 2019 is the *scrutable* version: set-based tag
preferences with **pairwise tag statements encoded as pseudo-tags** ("You don't like 'adventure' unless tagged 'thriller'"),
significance computed over I_{ta}∩I_{tb}, and users can zero a pair's weight (verified, §4–5, user study §7). Note Balog's pairs
are *tag conjunctions on items* — F2's within-item case — not cross-item pairs.

**F11. Would any of this move the accuracy loss?** The measured loss sits at the prototype layer (lightfm_hist +0.06–0.09 over
both). Pair terms act on the composition; via F3(a) they likely *worsen* the common-component problem uncentred. Only F-B4
(conjunction vocabulary) and F-B6 (induced-subgraph re-weighting) plausibly help the *noid* arm; none addresses the prototype
layer. Honest expectation: explanation-side gains, accuracy ≈ 0 in the ids-arm.

## 2. Candidate forms

| # | Form | Attribution unit | Exact additive attribution preserved? | Host alteration | Cold-item behaviour | Thin history (5 purchases) | Licensed sentence |
|---|---|---|---|---|---|---|---|
| F-B1 | Basket-restricted FM/NFM pair term: q_u += λ Σ_{a<b} w̄_a w̄_b (e_a⊙e_b) over history words (cross-item) | word pair (a,b) | Yes (pair vectors sum into q_u; same zoom as today), but pair weight = product of marginals; no own row → prototype naming stays via unary rows only | new additive terms (quadratic in n_u; zero new params) | Unaffected (user side); target scored as today | 250 cross-item pairs, weights ≤ 1/25 each; amplifies common component (F3a) | "You bought both black items and dresses" — NOT "co-occur strongly" |
| F-B2 | AFM-style attention over history word pairs | word pair with learned weight a_ab | Yes as multiplier; the weight's origin is opaque | re-weighting + new terms; context-dependent (survives cosine) | Unaffected | Attention over 250 pairs from 25 tokens; overfits at 5 purchases (AFM itself needs dropout on the pair layer) | "Of your purchases, the pair black×dress counted most" — weight unexplained |
| F-B3 | Explicit pair-row table for cross-field value pairs (colour×garment) over the *history* (cross-item) | pair token e_{ab} | Yes; pair rows are vocabulary → prototypes nameable by pairs | new terms + new vocabulary object (≤90k rows hm; ≈560k ml-1m, must prune) | Unaffected | Sparse: most pair rows seen by few users; D2-absorbed in ids-arm | "You often combine black with dresses" — but cross-item pairing ≠ combining (F2/F7) |
| F-B4 | **Conjunction vocabulary** — within-item cross-field tokens (black∧dress) added to the shared item vocabulary, support-pruned (TEM-style); fU inherits via the mean, unchanged | conjunction token per purchase | Yes, exactly as today (it is only more words) | none for fU (vocabulary change, shared with fI; addresses fI's E3 too) | Yes — cold item's conjunctions known | 5 purchases × ≤10 conjunctions = 50 tokens; pruned vocabulary keeps rows warm | "Because you bought black dresses and dark denim trousers" |
| F-B5 | Association-rule / itemset profile: attribute-level rules {A,B}→C mined over users; fired antecedents add a term (rule tokens) | rule (antecedent pair, consequent) | Yes; population confidence disclosed as the weight | new additive terms (outside or inside q_u); rule mining is a preprocessing object | Yes if consequents are attribute sets; No for item consequents | Few rules fire; D2-absorbed warm | "You bought A and B; N% of shoppers who did also bought C-type items" |
| F-B6 | Induced co-purchase subgraph re-weighting: w̄_{u,j} ∝ 1 + Σ_{k∈H_u} pop-co-purchase(j,k), fixed (not learned) | purchase j (weight from its pairs) | Yes; per-purchase weight is a disclosed multiplier | re-weighting only (context-dependent, so not killed by cosine) | Unaffected | With 5 items, weights are coarse; on hm equals "in the same order" (F7) | "We counted X more: it goes with 3 other things you bought" |
| F-B7 | FISM/complement pair (history item j, target i): score += Σ_j comp(j,i), attribute-composed comp for cold | (purchase, target) pair | Yes, exactly (linear over history) | new additive term *outside* the prototype layer (new object) | Yes via attribute composition of comp | 5 terms, very legible | "Because you bought X, which is usually bought with items like this" — not a within-history pair |
| F-B8 | Janossy k=2 with MLP f over history word/item embeddings | pair, black-box vector | Additive per pair, but pair content has no row → intrinsic naming lost | new object | Unaffected | 300 pairs; heavy for the signal | none legible beyond "these two purchases together" |

Recommendation for the parent synthesis: F-B4 is the only pair form that is informationally new (F2), contract-clean
(exact attribution, intrinsic naming, cold OK), host-neutral for fU and simultaneously answers fI's E3; F-B5 is the honest home
for the word "strongly"; F-B6 is the only genuinely within-history pair mechanism that is a pure re-weighting. F-B1/B2/B3/B8
are tabled to be rejected with reasons (F2, F3, F7). F-B7 is adjacent (item-neighbourhood), recorded for the fUfI merge.

## 3. Citations

Verified (abstract or full text opened this session):
- He & Chua 2017, *Neural Factorization Machines*, SIGIR — arXiv:1708.05027 (PDF; Eqs. 3–4, §3.1.1 NFM-0).
- Xiao et al. 2017, *Attentional Factorization Machines*, IJCAI — arXiv:1708.04617 (PDF; §2 FM Eq., pair-wise interaction layer, abstract claim).
- Blondel et al. 2016, *Higher-Order Factorization Machines*, NIPS — arXiv:1607.07195 (PDF; ANOVA-kernel multilinearity Eq. 7, O(dm) DP).
- McAuley, Pandey, Leskovec 2015, *Inferring Networks of Substitutable and Complementary Products*, KDD — arXiv:1506.08839 (abstract).
- Hao et al. 2020, *P-Companion*, CIKM — amazon.science publication page (abstract/summary).
- Wan, Wang, Liu, Bennett, McAuley 2018, *Representing and Recommending Shopping Baskets with Complementarity, Compatibility and Loyalty*, CIKM — cseweb.ucsd.edu/~jmcauley/pdfs/cikm18a.pdf (Eqs. 2–4).
- Yu et al. 2016, *A Dynamic Recurrent Model for Next Basket Recommendation* (DREAM), SIGIR — cseweb.ucsd.edu course copy (Eq. 1, §4 pooling comparison).
- Kabbur, Ning, Karypis 2013, *FISM*, KDD — via W. Pan's FISMauc slides (prediction rule Eqs. 1–2); original PDF not opened.
- Murphy et al. 2019, *Janossy Pooling*, ICLR — arXiv:1811.01900 (PDF; Def. 2.2, Thm 2.1, Cor. 2.1).
- Zaheer et al. 2017, *Deep Sets*, NeurIPS — arXiv:1703.06114 (PDF; Thm 2).
- Gori & Pucci 2007, *ItemRank*, IJCAI — ijcai.org/Proceedings/07/Papers/444.pdf (Eqs. 4–5, d_u definition).
- Balog, Radlinski, Arakelyan 2019, *Transparent, Scrutable and Explainable User Models*, SIGIR — krisztianbalog.com/files/sigir2019-transrec.pdf (Fig. 1, §4.1 pairwise pseudo-tags, §5.1, §7 scrutiny options).
- Wang, He, Feng, Nie, Chua 2018, *TEM*, WWW — abstract only (ACM DL snippet + author GitHub README); mechanism details **[unverified]**.

Memory only [unverified]: Rendle 2010 FM original (formula taken from AFM's restatement instead); RecWalk restart-linearity;
association-rule recommender literature (generic; no specific paper claimed); Haveliwala 2002 topic-sensitive PageRank
(cited inside ItemRank). Repo facts cited by file/line above; no PDFs saved into the repo (scratchpad only).
