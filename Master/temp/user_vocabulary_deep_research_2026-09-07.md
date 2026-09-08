# Deep research: how should a user's interaction history be represented in fU-ProtoMF, beyond "mean of attribute words"? (dc05 → possible fU′)

> **Question (Matteo, 2026-09-07, verbatim):** "right now it averages interactions for user embeddings, which
> brings it back to the attribute vocabulary which is fine, but truly lets think about what appropriate and
> interpretable vocabulary would be based on interactions for a user. I am not sure how much we should impact the
> way the model uses interactions, but maybe we can find way better solutions." — "forcing to view a users
> interactions history as only an average of attribute words is extremely limited, I feel like we could come up with
> ways that the model could use that are semantically meaningful that go beyond that. something like: we recommend
> this, because these two interactions in your history strongly co-occur — or any other different idea."
>
> **Scope (his decisions):** input = *which items* were interacted with in the train window, nothing else. Per-event
> context (date, season, channel, price, order structure) OUT. Demographics OUT, not even as fallback. "Maybe the
> embedding could be a combination or interaction between both" read as: item-identity-level and attribute-level
> history may be combined inside the user representation (assumption, stated here).
>
> **Method:** seven-agent sweep, 2026-09-07. Top-down angles A (items as vocabulary), B (within-history pairs /
> co-occurrence / sets / rules), C (profile as distribution / estimator), D (scrutable & hybrid user models); then
> E (prototype layer + explanation contract per form) and F (adversarial critic + ranking) over the fixed menu
> A–D produced; G (bottom-up: surveys' own taxonomies, recsys + EHR/CLV/IR/NLP) added at Matteo's request to check
> the top-down framing. Raw notes: `Master/temp/user_vocabulary_research_2026-09-07/angle_A…G.md`, `BRIEF.md`,
> `MENU.md`. The `paper-search` MCP was down; all retrieval via WebSearch/WebFetch. Claims are verified against
> abstracts/full text unless marked `[unverified]`. **Tier: working evidence / design input, not thesis text.**
> Angles E, F, G were cut once by a rate limit; E and F had written their files, G was re-run twice (the second
> attempt died with a killed session; the third completed in a fresh session the same evening).

## 0. Headline verdict

1. **"Strongly co-occur in your history" is never a faithful sentence about one hm user.** With 5 purchases every
   pair is present once; "strongly" can only be a population weight (FM factor, rule confidence, co-purchase edge).
   The faithful form is two-part: *presence* is the user's, *strength* is the population's (B-F1). Any form is graded
   on whether it says both halves.
2. **Cross-item word pairs add no information; the only new pair signal in a history is within-item conjunction and
   item identity.** A cross-item pair term with weight n_a·n_b is a deterministic quadratic of the word-count vector
   (B-F2). Via the NFM identity it is ≈ ½ c⊙c with c the catalog-constant, i.e. it *amplifies* the common-direction
   collapse unless centred first (B-F3). Conjunction tokens (black∧dress) are an item-side vocabulary extension that
   fU inherits through the mean — and they are fI's E3 rival, which today's ID-row probe found *cold* on the item side.
3. **With a free ID row, almost nothing on the menu changes the function class.** Every form that is a fixed function
   of the user's own train row is absorbable by e_ID(u) — the D2 lesson generalised (F-O1; realised today on the
   item side, D2 0.07→0.49). Only target-conditioned attention (M3) escapes. So in the ids arm the menu changes
   inductive bias and attribution, not capacity; the honest test bed for composition forms is `_noid`.
4. **The measured loss is at the prototype layer, and exactly one composition form has a mechanism there:
   population-mean centring (fU′).** It is the only form that *removes* the shared input direction the cosine layer
   cannot ignore (Mu & Viswanath; SIF step 2); M6/M7/M10/M16-alone *feed* it; the rest leave it (E-1). Forms that
   move the score more (M4, M8, M15) do so by adding the winning dot-product path *beside* the prototypes, which makes
   the prototype bars decorative — a contract-widening, not a repair.
5. **Decision-grade code finding: during training the scored positive's own words are inside q_u.** The history
   buffer is the full train file and the user extractor sees only the user index (verified: `feature_ids.py:496-510`,
   `feature_extractors.py:459-470`, `rec_sys.py:103`, `protomf_dataset.py` `__getitem__`). At hm median |H|=5 that is
   20 % of the metadata mass; at test the target is never in the history — a train/test mismatch of the FISM `\{i}`
   kind. `lightfm_hist` shares it (same module), so the fU-vs-control comparison is unbiased, but every absolute hm
   number of both is inflated by an unknown amount, and any item-ID history row (M1/M14) turns it into memorisation of
   the positive. Leave-one-out pooling in training (M2) is hygiene owed before any composition claim.
6. **What the transparency literature actually does** (D): every scrutable profile = frequency-weighted mean +
   deviation/shrinkage + sparsification to few units, grounded with *item examples*; the human evidence (Balog 2019,
   122 users) says the single most-endorsed unit is the example item, single-tag statements beat pair statements
   (57 % vs 34 % agreement), and Tagsplanations found plain genre-level words beat specific ones. The cheapest
   interpretability gains are explanation-layer only (X1 item examples under words, X2 deviation-from-crowd display).

## 1. Pathologies of the mean-of-words user (each tied to a dc05 measurement)

| # | Pathology | Measured where | Which forms act on it |
|---|---|---|---|
| P1 | Shared input direction: every q_u contains the population mean μ_vec (item side: 85 % of the mean composition is the catalog constant) → cosine layer sees a near-rank-1 cloud | eff. rank hm fU 1.79 / noid 2.20; ml-1m 2.63 / 2.46 (host 4.44); profile entropy ≈ uniform; p99 overlap 1.0 (`sc05_chapter_draft.md` l.281–296) | removes: M9. feeds: M6, M7, M10, M16-alone. leaves: rest |
| P2 | Norm handover: ‖mean of words‖ shrinks with |H| → e_ID share grows mechanically | hm ID-energy share 0.009→0.050 thin→heavy; ml-1m 0.003→0.070 (F-DC05-01) | M16 (cap), M9 (removes the part of the shrink due to μ), disclosure otherwise |
| P3 | Count bottleneck: q_u is a function of the word-count vector n_u; item identity and within-item conjunction are lost (Janossy k=1 with linear φ; B-F2, B-F9) | structural | M1/M14 (identity), M5 (conjunction) |
| P4 | Evidence mass is inexpressible: a 1-purchase user activates prototypes as decisively as a 100-purchase user (cosine kills every per-user scalar) | structural (C-5) | none in score; X2 in the explanation |
| P5 | Training self-inclusion of the positive (20 % at |H|=5) | verified this session (§0.5) | M2 |
| P6 | Dataset meaning flip: hm = 5 same-day purchases × 5 fixed fields; ml-1m = 56 movies × 3–72 tokens; most forms mean different things on the two (F-O3; B-F7: on hm any cross-item pair term is a within-basket term by accident) | 94.3 % same-day (F-DC05-09); bag-size channel 0.67 | disclosure obligation for every form |
| P7 | Free rows choose their own norm: exact shares can be optimizer-chosen shares (E, contract fact 2) | ID norm handover above; would recur for y_j rows | norm read-outs mandatory for M1/M14 |

## 2. The ladder of history-representation forms

Columns: unit of attribution · exact attribution preserved? · host interaction-use altered · common direction (R removes /
L leaves / F feeds) · cold-item · thin-history (5) · the sentence it licenses. Full audits in angle_E.md §1 and angle_F.md §1.

### Tier A — same units, re-weighted / centred / constrained (contract intact)

| # | Form | Unit | Exact? | Host altered | Dir. | Cold item | 5 purchases | Sentence |
|---|---|---|---|---|---|---|---|---|
| M2 | Leave-one-out pooling in training (FISM `\{i}`) | unchanged | yes | none (training hygiene) | L | n/a | trains on 4 of 5 | none new |
| M9 | Population/field-mean centring (fU′); **drop the "count shrink" half** — a per-user scalar on (m_u−μ), dead in both arms (F) | signed word deviations + one named "crowd baseline" | yes | shift (function-class redundant in ids arm, real in noid) | **R** | fine (attributes known) | deviations at 0.2 granularity, SE 0.1–0.2 → collapse the ~400 tiny negatives into the crowd unit | "Compared with the average shopper you buy far more *black* and *skirts*; that is what matches this item." |
| M16 | Disclosed ID-share cap ‖e_ID‖ ≤ α‖q_words‖ | unchanged (+1 number) | yes | constraint | **F alone** (e_ID is today the only μ-free component); neutral with M9 | n/a | bites where histories are short — the wrong end on hm | "At most 30 % of your profile is your personal row." |
| M12 | Top-k sparsified profile (coverage × significance) | ≤ k words | yes | data-dependent 0/1 gate | R if significance-driven, F if coverage-driven | fine | no-op (≤ 20 distinct words) | "Your profile is *ladieswear, black, trousers*, covering 4 of 5 purchases." |
| M13 | Saturating counts (EFM) | unchanged | yes | re-weighting (noid-only) | L | fine | no-op (n ≤ 5) | "Repeats count more, with diminishing returns." |
| M11 | Per-user-per-field concentration gate | words + 5 field gates | yes; gate's reason is nameable ("4 of 5 black") | re-weighting (noid-only) | L; on hm upweights the gender-typed section field | fine | entropy from 5 draws biased → gates ≈ 1 | "You are very consistent about *colour*, so colour drives this match." |
| M10 | Dirichlet shrink toward the crowd | words + crowd share | yes | shift (noid-only) | **F by construction** — the anti-M9 | fine | half crowd at κ≈5 → deepens collapse | "With only 5 purchases we lean half on shoppers like you." |
| M7 | Co-purchase-subgraph re-weighting of own purchases | purchases × disclosed multiplier | yes | re-weighting with population edges as input (**breaks info-nesting**) | F | fine | on hm ≈ "same order" | "We counted X more: it goes with 3 other things you bought." |

### Tier B — new additive, nameable units (contract intact, object grows)

| # | Form | Unit | Exact? | Host altered | Dir. | Cold item | 5 purchases | Sentence |
|---|---|---|---|---|---|---|---|---|
| M5 | Conjunction vocabulary (within-item cross-field tokens, support-pruned, TEM/Balog pseudo-tag style; shared with fI) | words + phrases | yes; naming *richer* | none for the host (vocabulary) | L/F | fine | 25 words + up to 50 conjunctions = 75 units → illegible without collapsing; ml-1m C(72,2) explodes | "Because you bought black dresses and dark denim trousers." |
| M1 | Words + item-ID history rows y_j (SVD++/NSVD shape), with `_noid` twin | purchase → words + "this article itself" | yes; naming *incomplete* (unlabelled rows in the prototype space) | new rows in q_u (builder-only) | L→F (popular rows can grow a shared direction) | falls back to words (y_j untrained for tail) | 30 units; on ml-1m y_j out-competes words → NSVD with a word-shaped explanation | "25 % from your cardigans' attributes, 15 % from those exact articles." |
| M6 | Attribute-level association-rule tokens | fired rules with confidence | yes | new terms; population confidences as input (**breaks nesting**) | **F** (high-support antecedents = the constant) | fine (attribute consequents) | 10–50 fired rules flood q_u with crowd words | "You bought A and B; N % of shoppers who did also bought C-type items." |
| M14 | Two-vocabulary concat (item-ID mean ‖ word mean, block prototypes) | per block | numerator only | new object (sub-case of M1 with a block constraint) | word block L, ID block free | as M1 | 30 | "Because you bought *this exact item* and items like it." |

### Tier C — a second scoring path beside the prototypes, or a per-candidate profile (contract widens)

| # | Form | Unit | Exact? | Host altered | Dir. | Cold item | 5 purchases | Sentence |
|---|---|---|---|---|---|---|---|---|
| M4 | Prototype score + additive FISM item-item term | two objects (direct purchase terms + prototype bars) | yes, two sums | new score path (new ft_type); prototype path risks becoming decorative | L | cold candidate has no t′_i → term 0 → warm/cold scored on different scales | 5 direct terms | "Because you bought X and Y (+0.31 direct) and you match *Basics* (+0.52)." |
| M15 | EFM-style direct attribute-match term | words(i) ∩ profile | yes, two objects | new score path; **not** ID-absorbable; is P5 turned into a feature | L | cold-safe | ≤ 5 units, the most legible sentence | "You might be interested in *black jersey*, which this product is." |
| M8 | (purchase, target) complement term, attribute-composed complement | per (purchase, target) | yes | new score path | L | composable | 5 | "Because you bought X, usually bought with items like this." |
| M3 | Target-conditioned attention over purchases (NAIS) | 5 weights + words, *per candidate* | yes given weights; weights unnameable (Jain & Wallace) | new ft_type; the user profile stops being a per-user object | L | fine | nothing to discriminate among 5 | "For this item we weighted your purchase of X 3× more." |

**Rejected by the angles, agreed by the critic:** basket-restricted FM/NFM pair term (weights = independence expectation,
amplifies the constant), AFM attention over word pairs, explicit cross-item pair-row table, Janossy k=2 MLP (no
vocabulary row), IDF/BM25/SIF constant weights (redundant), PPMI weights (degenerates to presence + IDF at N=5),
NL/LLM profile bottleneck (no exact attribution), post-hoc Amazon-style lookups (breaks intrinsic naming).

**Explanation-layer only, ship regardless:** X1 item examples under each word/prototype (the most-endorsed unit in
the human evidence); X2 deviation-from-crowd display with standard errors (honest at |H|=5: most deviations are noise).

### Critic's ranking (exact attribution first; host alteration none → re-weighting → new terms → new object; ID-redundancy tiebreak)
1 M2 · 2 M9 (no shrink) · 3 M16 (only beside M9) · 4 M5 · 5 M1 · 6 M12 · 7 M13 · 8 M11 · 9 M7 · 10 M10 · 11 M15 · 12 M6 · 13 M4 · 14 M14 · 15 M3 · 16 M8.

### Answer to the two framing questions
- *Is there a semantically richer vocabulary than mean-of-words that stays inside the contract?* Yes, three: signed
  deviations from the crowd (M9), within-item conjunction phrases (M5), and the purchased items themselves (M1). Of
  these only M9 has a mechanism at the layer where the accuracy is lost; M5 has negative item-side evidence today; M1
  answers "use the interactions, not just their words" most literally but expects a lightfm_hist-style lift with the
  prototype-layer gap untouched, and needs M2 first.
- *How much should the host's use of interactions be altered?* The evidence says: as little as a shift (M9) plus
  hygiene (M2) before anything else, because (a) the ID row absorbs anything bigger in the ids arm, (b) the loss is not
  in the composition, and (c) every larger form flips meaning between hm and ml-1m (P6).

## 3. Points that feed the thesis text

- Limitations chapter / hidden-effects section: P3 (count bottleneck, Janossy k=1 framing), P4 (evidence mass
  inexpressible under cosine), P5 (training self-inclusion — disclose, with the control sharing it), P6 (the same form
  means two things on the two datasets), and the two-part faithfulness rule for any co-occurrence sentence (§0.1).
- The "why not attention / pairs" paragraph: cross-item pairs are information-free given n_u (B-F2); attention weights
  are exact multipliers with unnameable reasons (Jain & Wallace; W&P's uniform baseline *is* fU).
- Scrutability: fU is linear-editable like Balog 2019; the ID row is the non-scrutable residual (vault 2026-07-14_1511
  already says this); M9 makes the edit unit "deviation from the crowd", which is the unit the human studies endorse.
- Fairness: word-named user prototypes make the ProtoMF §5.3 gender proxy *auditable by name* on hm (section/department
  labels `[unverified: labels not re-checked this session]`); M1/M14/M11/M6/M7 make it less auditable; M9 makes a
  gendered deviation a signed, readable coordinate. The §5.3 representative-user test transfers unchanged.
- fUfI merge outlook: item pairs, complement terms (M8) and shared-vs-separate tables are merge-cycle business, not fU's.

## 4. Cheap instruments (login-node class, working evidence)

1. **User-side H1 / D2**: ‖mean_u q_meta‖ / mean_u ‖q_meta‖ and ‖mean_u e_ID‖ / mean_u ‖e_ID‖ on the fU hm and ml-1m
   checkpoints — the item-side numbers (85 %, 0.07) have no user-side twin yet. Reuse `id_row_probe.py` Section B.
2. **Self-inclusion cost**: re-evaluate the saved fU and lightfm_hist checkpoints with the positive's words removed
   from q_u at *test* time is meaningless (target absent anyway); the real instrument is a single M2 retrain of the
   winning config for fU-noid and lightfm_hist on hm (≈1 h each) and reading the delta. This is the only item here
   that trains.
3. **Within-history pair signal exists at all?** For hm users with |H| ≥ 5: lift of their item pairs and cross-field
   value pairs vs chance, split same-day vs cross-day, to quantify P6/B-F7 (does "pair" reduce to "same order").
4. **User-side conjunction encoding**: regress e_ID(u) on within-item conjunction indicators of the user's purchases,
   controlling for the unary counts — the user-side twin of today's item probe; the F verdict on M5 hinges on it.
5. **Name-coverage share** per user prototype (Σ|word-row contributions| / Σ|all|), on today's checkpoints as the
   baseline any M1/M14 run must be compared against.

## 5. Key citations (verification status as reported by the agents; full lists in the angle files)

Verified (abstract or full text opened): He et al. NAIS (TKDE 2018); Kabbur, Ning & Karypis FISM (KDD 2013, via
abstract + slides); Koren & Bell, Advances in CF (Handbook chapter, SVD++ eq. 3); Koren, Bell & Volinsky (IEEE Computer
2009); Linden, Smith & York (IEEE IC 2003); Rendle FM (ICDM 2010); Chen et al. SVDFeature (JMLR 2012); Kula LightFM
(2015); Herlocker, Konstan & Riedl (CSCW 2000, Table 1); He & Chua NFM (SIGIR 2017); Xiao et al. AFM (IJCAI 2017);
Blondel et al. HOFM (NIPS 2016); McAuley, Pandey & Leskovec Sceptre (KDD 2015); Wan et al. triple2vec (CIKM 2018); Yu et
al. DREAM (SIGIR 2016); Murphy et al. Janossy pooling (ICLR 2019); Zaheer et al. Deep Sets (NeurIPS 2017); Gori & Pucci
ItemRank (IJCAI 2007); Balog, Radlinski & Arakelyan (SIGIR 2019, full text incl. §4.1 pairwise pseudo-tags, §7 user
study); Balog & Radlinski (SIGIR 2020); Radlinski et al. NL user profiles (SIGIR 2022); Vig, Sen & Riedl Tagsplanations
(IUI 2009); Zhang et al. EFM (SIGIR 2014); Covington et al. YouTube DNN (RecSys 2016); Mysore et al. LACE (SIGIR 2023);
Ramos et al. (ACL 2024); Penaloza et al. TEARS (arXiv 2024); Pazzani & Billsus (2007); Ziegler et al. taxonomy-driven
(CIKM 2004); Mu & Viswanath all-but-the-top (ICLR 2018); Arora, Liang & Ma SIF (ICLR 2017, via authors' post); Levy &
Goldberg (NIPS 2014); Gao et al. representation degeneration (ICLR 2019); Jain & Wallace (NAACL 2019); Wiegreffe &
Pinter (EMNLP 2019); Zhang & Chen survey (FnTIR 2020); Tintarev & Masthoff (2007); Melchiorre et al. ProtoMF §5.3 (local
PDF).
`[unverified]` (memory only, flagged): Koren KDD 2008 Asymmetric-SVD numbers; Sarwar et al. 2001; Wang et al. TEM (WWW
2018) mechanism details (abstract only); Zhai & Lafferty Dirichlet formula; Miller–Madow bias; RecWalk restart-linearity;
hm per-field cardinalities and gender-typed section labels (not re-read from `articles.csv` this session).

## 6. Shortlist for a possible fU′ `/design-candidate` cycle (no decision taken here)

Single-variable rule from dc06 applies: one change per candidate, knobs not a new ft_type where possible.

1. **M2 first, as hygiene, not a candidate** — leave-one-out pooling in training for fU *and* lightfm_hist (small
   forward/RecSys change; the user extractor must receive the positive's word codes in the training branch). Re-run the
   two winning hm configs. Every later read-out is contaminated until this is done.
2. **fU′ = population-mean centring (M9), noid-arm primary, ids arm as pre-registered D2 read-out with a magnitude
   floor** — the mirror of the fI′ cycle already queued; the one form with a prototype-layer mechanism and an
   interpretability *gain* (deviation-from-crowd unit). Expect the ids arm to re-house μ in e_ID as fI′-ids did.
   Pair with M16 only as an α-control, never alone. Cheap: module flag mirroring dc06 + builder-supplied p̄.
3. **M1 (item-ID history rows) only after 1 and 2, with its `_noid` twin, norm read-out ‖Σy‖/‖Σc‖ and per-prototype
   name-coverage share pre-registered** — the literal answer to "use the interactions", builder-only, provably not a
   covert user ID on either dataset (items < users), but the honest expectation is a lightfm_hist-style lift with the
   prototype-layer gap untouched.

Not shortlisted: M5 (conjunction vocabulary) until instrument §4.4 shows non-zero user-side conjunction encoding — it
is an item-side vocabulary paper and today's item probe was cold; M4/M15 (second scoring path) because they answer
the accuracy question by bypassing the prototypes; M3 because the user profile stops being a per-user object.

## 7. Bottom-up check (angle G)

Source: `angle_G.md` (191 lines; surveys Wang IJCAI'19, Wang CSUR'21, He'23 user-behaviour modelling, Purificato 2024
user-profiling, Zhang & Chen 2020 explainable-rec grid; EHR (Med2Vec, RETAIN, Deepr, Si 2021 review), IR search
personalisation (Bennett SIGIR'12), CLV/RFM (Fader & Hardie 2005), Deep Sets).

**What the surveys' own taxonomies say.**
- The recsys surveys are *order-first*: their axes are Markov / RNN / CNN / attention / memory / GNN. Mean-pooling appears
  only as the pre-2016 baseline, and He'23 lists "interpretable user representations" as an open problem. The brief's
  scope (an unordered set of items, no time) is the surveys' least-populated cell; that cell is filled by pattern/rule
  mining, session-KNN, MLP-over-set and generative topic models — *not* by a different pooling operator.
- User-profiling taxonomy (Purificato 2024): term-based vector space, ontology/semantic, concept hierarchy, graph/KG,
  universal, holistic; the root is Rich 1979 stereotypes — U-ProtoMF's prototypes are a learned stereotype model.
- Zhang & Chen 2020 allow exactly three user-side explanation styles: relevant users/items, features, topic word-clusters.
  That is the ceiling on what "semantically meaningful" can mean for a history-derived profile.
- Cross-domain lesson (EHR, IR, NLP): attribution survives by keeping *the sum and linear rows*; structure beyond the mean
  comes from **grouping** (visit, session, interest cluster, topic), not from a new operator. Deep Sets: every order-free
  form is ρ(Σφ), so the real design space is what φ sees, sum vs max, and whether ρ is the identity. RETAIN shows that
  context-dependent attention *can* keep exact per-(purchase, word) attribution if rows are linear and pooling is a
  weighted sum.

**Families the top-down menu (M1–M16) missed.**

| # | Family | Exact? | Host alteration | Verdict for fU |
|---|---|---|---|---|
| F3 | Neighbour-user additive term (user/session-KNN; Herlocker "relevant users" style) | yes | new additive term beside the prototype score | Genuinely absent from the menu; thin-history tolerant; licenses "12 shoppers with baskets like yours bought this". Same structural cost as M4/M15: a second scoring path beside the prototypes. |
| F4 | Linear multi-hop propagation inside q_u (LightGCN-style) | yes (linear) | new terms inside q_u | Reject with reason: injects other users' words into "your history"; the contract's *own-history* clause breaks. |
| F5 | Mixed-membership (simplex) user over word-topics, score Σ_z θ_{u,z} φ_{z,i} | yes | new object — **replaces the prototype layer** | The only bottom-up candidate aimed at where the accuracy loss sits, and topics are nameable by top words. It is not an fU′ single-variable change; it is a different host. Record as a future-work / part-two idea, not a dc05 knob. |
| F6 | Multi-interest K sub-profiles (MIND/ComiRec) | within a sub-profile only | new object | ml-1m-only (K≈1 at 5 purchases); loses additivity across sub-profiles. Not shortlisted. |
| F10 | RETAIN exactness constraint on attention | — | requirement, not a form | Amend M3/M11: linear rows + weighted sum ⇒ attribution stays exact with weights frozen; angle E's attention-faithfulness objection (Jain & Wallace) then applies to the *weights*, not the decomposition. |
| F2′ | Item-level rule *confidence as an additive term* | yes, per rule | new additive term | Uncovered slice of A/B; needs item IDs, so no cold-item path; fUfI-merge business like M1. |

Present in another guise: F7≈M3 hard-gated, F8≈C1/M13, F9≈C6, F11≈M13 limit (presence profile), F14≈M14/M1, F15≈intrinsic
naming. Out of scope but surfaced by every source: order (F1), basket level (F12; hm 94 % same-day makes it near no-op),
recency/monetary (F13).

**Effect on the verdict and shortlist.** None of the missed families overturns §0 or §6. F5 is the one substantive
addition: it is the bottom-up literature's answer to "the loss is at the prototype layer", but it swaps the host rather
than repairing it, so it belongs on the fUfI/part-two horizon and in the thesis's future-work text, not in an fU′ cycle.
F3 strengthens the case that any accuracy gain from a *new term* comes from a path beside the prototypes (same as M4/M15).
F10 is a free improvement to how M3/M11 are specified if they are ever pursued. The shortlist in §6 stands unchanged.

Verification caveats carried from G: Lops/Pazzani chapter bodies paywalled (metadata only); RUM, Hofmann 2004, LDA,
Rich 1979, Sontag 2012, Herlocker 2000 `[unverified]`.
