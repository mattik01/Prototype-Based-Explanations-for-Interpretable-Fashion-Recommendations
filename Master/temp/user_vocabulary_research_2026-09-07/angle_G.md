# Angle G — bottom-up: how the literature taxonomises "user from interaction history" (2026-09-07)

Method: started from surveys/taxonomies (recsys: sequential, session-based, user-behaviour-modelling, user-modelling/
profiling, explainable-rec; then EHR, IR search personalisation, CLV/RFM, NLP set functions/topic models), extracted the
taxonomy *as the source states it*, then mapped every family onto the brief's axes and diffed against MENU.md.
PDFs + text extracts in scratchpad/pdf/ (not in repo). "verified" = I opened the full text or abstract page.

## §1 The taxonomies as the sources state them

### 1.1 Recommender systems
**G1. Wang et al., "Sequential Recommender Systems: Challenges, Progress and Prospects", IJCAI 2019 survey track
(arXiv 2001.04830) [verified full text].** Fig. "SRS approaches from the technical perspective" has three top classes:
*Traditional sequence models* (sequential pattern mining; Markov chain models), *Latent representation models*
(factorization machines; embedding), *Deep neural network models* → *basic* (RNN, CNN, GNN) and *advanced* (attention
models, memory networks, mixture models). The user is never a first-class object: it is whatever the sequence encoder
emits. All three classes are order-based except FM/embedding — i.e. the survey's own frame is mostly OUT of the brief's
scope (no timestamps/order).

**G2. Wang et al., "A Survey on Session-based Recommender Systems", ACM CSUR 2021 (arXiv 1902.04864) [verified full
text].** Much richer bottom layer. Sessions are classified as *ordered / unordered / flexibly-ordered* ("shopping sessions
are sometimes unordered since users may pick up a basket of items … without following an explicit order", §4). Approach
taxonomy (Fig. 5 / Table 5): *Conventional* = pattern/rule mining (frequent-pattern → rule confidence P(v̂|ŝ) over a
threshold), K-nearest-neighbour (item-KNN: item = binary vector over sessions, cosine; session-KNN: score(v̂) =
Σ_{s∈N(c)} sim(c,s)·1_s(v̂), Eq. 3), Markov chain, generative probabilistic model (infer a latent taxonomy/topics of items,
learn transitions among topics); *Latent representation* = latent factor model, distributed representation
(skip-gram/CBOW-style item2vec); *Deep* = RNN, MLP ("mainly suitable for unordered session data", Eq. 15 MLP fuses
sub-context representations), CNN, GNN, attention, memory networks, mixture models, generative models, RL. Table 5
grades every class on four dependency axes: sequential vs non-sequential, short- vs long-term, first- vs higher-order,
pointwise vs collective. Pattern mining and KNN are the only "intuitive, simple and effective" classes (Table 6, pros).
Relevant reading for fU: the *non-sequential, collective, higher-order* cell — the one hm's unordered 5-item baskets
actually live in — is filled by pattern mining, session-KNN, MLP-over-set and generative topic models, not by pooling.

**G3. He et al., "A Survey on User Behavior Modeling in Recommender Systems", 2023 (arXiv 2302.11087) [verified full
text].** Four branches (Table 1): *Conventional UBM* (RNN GRU4Rec/HRNN; CNN Caser/NextItNet; attention SASRec, DIN
target-aware, DIEN, DSIN, BST, BERT4Rec), *Long-sequence UBM* (memory-augmented UIC/HPMN; *retrieval-based* UBR "retrieve
relevant behaviors from long histories"), *Multi-type UBM* (DMT, MBGCN, NMTR), *UBM with side information* (TiSASRec,
p-RNN, S3Rec). Stated progression: "unidirectional dependencies (RNNs) → skip behaviors (CNNs) → arbitrary pairwise
dependency (Attention) or multiple relationships (GNNs)". Explicit open problem §8: "the learned user interest
representations are not well interpretable … interpretable UBM is another promising future research direction."
Nothing in this survey is mean-pooling; pooling is treated as the pre-2016 baseline it replaced.

**G4. Purificato, Boratto & De Luca, "User Modeling and User Profiling: A Comprehensive Survey", 2024 (arXiv 2402.09660)
[verified full text].** §4.2.5 "User profile representation" variants: *Term-based (vector-space model)* — "a set of
terms weighted by vectors of keywords … Boolean, TF, or TF-IDF", "a single vector encompassing all interests or multiple
vectors reflecting interests in various domains"; *Semantic-based (ontology-based)* — weighted concepts in a semantic
network (Sieg et al. 2007); *Concept-based (hierarchy-based)* — interest weights on nodes of a taxonomy (Nanas et al. 2003,
LCS hierarchy + TF-IDF edge weights); *Graph- and knowledge-graph-based*; *Universal user representation*; *Holistic user
modeling*. Historical root: Rich 1979 *stereotype user modelling* — "predefined sets of characteristics … associated with
certain types of users" (a learned stereotype = a user prototype; U-ProtoMF is literally this). §4.3.1 "Explainable user
modeling" names Balog et al. 2019 set-based scrutable models as the reference point.

**G5. Zhang & Chen, "Explainable Recommendation: A Survey and New Perspectives", FnTIR 2020 (arXiv 1804.11192) [verified
full text].** Table 1.1 is a 2-D grid: *model* (neighbour-based, MF, topic modeling, graph-based, deep learning,
knowledge-based, rule mining, post-hoc) × *explanation information* (relevant user or item; user/item features; textual
sentence; visual; social; word cluster). The *user-side* information styles are exactly three: relevant users/items
(Herlocker 2000 histograms of neighbours), features (Vig 2009 tagsplanations; EFM), and word clusters / topics (HFT,
McAuley & Leskovec 2013: "projecting each user's latent vector into the latent topic space in LDA … helps to understand
why a user made a particular rating"; Wu & Ester 2015). Nothing else exists as a user-side explanation vocabulary in this
grid — a useful ceiling on what "semantically meaningful" can mean for a shopper.

**G6. Content-based filtering chapters.** Lops, de Gemmis & Semeraro, "Content-based Recommender Systems: State of the Art
and Trends", RecSys Handbook 2011 [metadata verified via Semantic Scholar; body unverified — Springer paywalled]: user
profile = learned classifier over item content — probabilistic (naive Bayes over word counts), Rocchio centroid, kNN,
plus semantic (WordNet-synset / ontology) profiles. Pazzani & Billsus 2007 "Content-Based Recommendation Systems" [metadata
verified; body unverified]: same classifier framing (decision trees, nearest neighbour, Rocchio, linear classifiers,
naive Bayes). The 2015 handbook chapter "Semantics-Aware CBRS" [abstract page fetched, body unverified] splits semantics
into *top-down* (ontologies, Linked Open Data) vs *bottom-up* (distributional word embeddings).

**G7. Two exemplar families the surveys single out that no A–F angle names.** (a) *Multi-interest / multi-vector
user*: MIND (Li et al., CIKM 2019, arXiv 1904.08030) [verified] — "dynamic routing can be viewed as soft-clustering,
which groups user's historical behaviors into several clusters. Each cluster … is further used to infer the user
representation vector corresponding to one particular interest"; ComiRec (Cen et al., KDD 2020) [verified abstract] adds a
controllable aggregation over the K interests. (b) *Neighbourhood / propagation as user model*: session-KNN (G2 Eq. 3)
and LightGCN (He et al., SIGIR 2020, arXiv 2002.02126) [verified abstract] — user = linear "weighted sum of the
embeddings learned at all layers" of neighbourhood aggregation, "feature transformation and nonlinear activation …
contribute little", i.e. the user is a (multi-hop) mean over items and other users' items, purely linear.
(c) *Memory networks* (RUM, Chen et al. WSDM 2018 — existence verified via He 2023 reference list; content [unverified]):
item-level memory slots read with target-aware attention — a slotted form of M3.

### 1.2 Other domains
**G8. EHR / patient representation.** Si et al., "Deep representation learning of patient data from EHR: a systematic
review", JBI 2021 (arXiv 2010.02809) [verified full text]: input layer — "Among 37 studies learning from structured codes,
the majority (n=27) generate one-hot vectors … the frequency of a given code is used in 5 studies"; hierarchy "from
event-level, to visit-level, to patient-level"; Deep Averaging Network (mean-of-code-embeddings) appears as a listed
architecture [62, 90]; attention "in 13 studies … to provide a degree of interpretability". Med2Vec (Choi et al., KDD 2016,
arXiv 1602.05568) [verified]: visit = binary bag-of-codes x_t → ReLU(W_c x_t + b_c) with a *non-negativity* constraint
"to enable interpretability" — "each coordinate … can be readily interpreted … top k codes that have the largest values
for the i-th coordinate … view each coordinate as a disease group" (§3.5); codes within a visit are explicitly *unordered*.
RETAIN (Choi et al., NeurIPS 2016, arXiv 1608.05745) [verified]: linear embedding v_i = W_emb x_i, visit-level α_j and
variable-level β_j attention from two RNNs; because the context is a weighted *sum* of *linear* embeddings, the prediction
"can be completely deconstructed to the variables at each input": ω(y, x_{j,k}) = α_j W(β_j ⊙ W_emb[:,k]) x_{j,k} (Eq. 5) —
exact additive attribution to every (visit, code) pair *despite* context-dependent weights. Deepr (Nguyen et al. 2016,
arXiv 1607.07519) [verified]: record → "sentence" of codes, CNN detects "clinical motifs" = "co-occurrences of diseases"
in a sliding window, max-pooled; explanation = the motifs with strong response. Atom: the code (ICD/ATC), grouped by visit.

**G9. IR — search personalisation.** Bennett et al., SIGIR 2012 [verified full text]: one framework, Eq. §3.1 — feature =
Σ_{related queries} Σ_{results} weight × sim(d, d') × action, rewritten as ⟨φ(d), Σ_{past clicks} w·φ(d')⟩, i.e. the user is an
*aggregated weight vector* whose representation φ is either identity (sparse URL indicator → "number of times the user
has clicked on d", the re-finding feature) or *ODP topic probabilities* ("top two levels of the ODP hierarchy … restricted
to the three most probable classes"), compared by cosine; temporal views Session/Historic/Aggregate + decay c^t. The
*target-conditioned selection* of history ("related queries": identical / superset / subset of the current query) is the
IR ancestor of M3/UBR. Sontag et al. WSDM 2012 (per Bennett §2) build generative ODP-category user models. Atom: ODP category.

**G10. Customer analytics / CLV.** Fader, Hardie & Lee, "RFM and CLV", J. Marketing Research 42 (2005) [verified full text]:
"it is common practice to summarize a customer's past behavior in terms of their 'RFM' characteristics: recency (time of
most recent purchase), frequency (number of past purchases), and monetary value"; under the Pareto/NBD + gamma-gamma
model "recency, frequency, and monetary value are sufficient statistics for an individual customer's purchasing history".
Atom: the *transaction count/time*, not the item — the customer is a set of parameters of a stochastic process. All three
are out of the brief's scope (dates, price) or a per-user scalar (|H_u|) the cosine kills. Contribution: the idea of user =
*sufficient statistics of a generative model*; the cosine host has no channel for evidence mass (brief given).

**G11. NLP / set functions.** Zaheer et al., Deep Sets, NeurIPS 2017 (arXiv 1703.06114) [verified]: Theorem 2 — any
permutation-invariant set function on a countable universe is ρ(Σ_{x∈X} φ(x)). So *every* order-free user representation
is "sum-pool then transform"; the design space is (i) what φ sees (word, item, pair), (ii) sum vs max (max-pool is the
Deepr/CNN choice), (iii) whether ρ is identity (keeps additivity) or not. Topic models: LDA/pLSA user = mixed-membership
simplex over topics, topics = distributions over items/words, read by "top-n words" (Zhang & Chen G5, HFT) [LDA itself
unverified/classic]. Bag-of-words with TF-IDF = the term-based profile of G4.

## §2 Cross-domain atoms table
| Domain | Interpretable atom | Grouping level | Dominant aggregation | Interpretability device |
|---|---|---|---|---|
| RecSys CF/sequential (G1–G3) | item ID | sequence (ordered) | RNN/CNN/attention; pooling = baseline | none / attention weights (survey: open problem) |
| RecSys session-based (G2) | item ID; item set | session (often unordered) | rule confidence; KNN neighbour sum; MLP over set; topic transitions | rules & neighbours are "intuitive" (Table 6) |
| RecSys content-based (G4, G6) | keyword / concept / tag | flat bag over consumed items | TF-IDF weighted vector; NB counts; Rocchio centroid; ontology node weights | inspect the weighted term list |
| RecSys explainable (G5) | relevant user/item · feature · topic word-cluster | — | MF / topic model / neighbours | the three user-side styles |
| EHR (G8) | medical code | code → visit → patient | one-hot/multi-hot (27/37), counts (5/37), DAN mean, RNN over visits | non-negative coordinates (Med2Vec); exact linear attention attribution (RETAIN); motifs (Deepr) |
| IR search (G9) | ODP topic category; URL | click → query → session/historic | Σ weight·φ(clicked doc), cosine; decay | topic-profile similarity features |
| CLV/RFM (G10) | transaction (count, time, amount) | customer | sufficient statistics R, F, M | iso-value curves over R×F |
| NLP set/topic (G11) | word / token | document | ρ(Σφ); max-pool; simplex over topics | top-n words per topic |
Convergent lesson: every domain that wants attribution keeps the *sum* and makes φ linear (RETAIN, LightGCN, Bennett,
Med2Vec's non-negativity); every domain that wants structure beyond the mean does it through *grouping* (visit, session,
interest cluster, topic) rather than through a different pooling operator.

## §3 Mapping every family onto the brief's axes
Host recap: q_u = Σ_f (n_{u,f}/|H_u|) e_f + e_ID; score = Σ_l t_{i,l}(1+cos(q_u,p_l)). "Exact" = score decomposes additively
into nameable units without approximation. Alteration: none / re-weighting / new terms / new object.
| # | Family (source) | Attribution unit | Exact? | Host alteration | Cold item | Thin history (5) | Licensed sentence |
|---|---|---|---|---|---|---|---|
| F1 | Sequence encoders: Markov/RNN/CNN/attention (G1–G3) | none (hidden state) | no | new object | ok via t_i | poor; needs order | — (OUT of scope: no order/time) |
| F2 | Item-level pattern/rule mining, confidence P(v̂\|ŝ) (G2 §6.1.1) | (purchase-set → target) rule | yes, per rule | new additive term | none (rule needs item ID; attribute-composed head = M8) | good — built for baskets | "Shoppers who bought your A and B usually also buy this" (A/B partly cover as co-purchase; item-level rule *confidence as an additive term* not on menu) |
| F3 | Session/user-KNN: score = Σ_{n∈N(u)} sim(u,n)·1_n(i) (G2 Eq. 3; Herlocker 2000 in G5) | neighbour user | yes | new additive term (or new object if replaces prototypes) | none for cold items (only via attribute sim) | fine (sim on 5-item sets is noisy but defined) | "12 shoppers with baskets very like yours bought this" — the *relevant-user* style, the one user-side style in G5 not on the menu |
| F4 | Linear multi-hop propagation, user = Σ_k α_k A^k (LightGCN) | own purchases + co-purchasers' purchases | yes (linear) | new terms inside q_u | 2-hop reaches only train items | strong smoothing at 5 | "Because people who bought your X also buy [Y-type] items" — crowd words enter q_u; contract cost: rows no longer the user's own |
| F5 | Mixed-membership user θ_u ∈ simplex over topics; topic = distribution over words/items (pLSA/LDA/HFT, G2 generative, G5 topic column) | topic (named by top words) | yes if score = Σ_z θ_{u,z} φ_{z,i}; θ from counts (Dirichlet) | new object (replaces prototype layer) — *this is the family that competes with the prototype layer itself* | via φ_z over words | posterior shrinks to prior at 5 | "Your history is 60% [topic: black/jersey/trousers], and this item is typical of that topic" |
| F6 | Multi-interest K sub-profiles (MIND soft-clustering; ComiRec) | (sub-profile, its purchases) | exact within the winning sub-profile; max/softmax over K breaks additivity across them | new object (K q_u's) | ok | K collapses to ≈1–2 at 5 purchases | "This matches the *sportswear* side of your history (3 of your 5 purchases)" |
| F7 | Retrieval-based UBM (UBR) / memory slots (RUM) | retrieved purchases | yes if pooled linearly after hard selection | re-weighting (0/1, target-conditioned) | ok | irrelevant at 5; only ml-1m (56) | "We looked only at your 10 purchases most like this item" — hard-gate variant of M3 |
| F8 | Term-based TF-IDF / naive-Bayes profile (G4, G6) | word | yes | none (constant weights, given) or re-weighting via log(n+α) | ok | ok | = C1 (rejected) / M13 saturating counts; NB log-count is one specific curve |
| F9 | Concept/hierarchy profile (G4 concept-based; ontology) | taxonomy node | yes | new rows (parent-field rows) | ok | helps (shrink to parent) | = C6 on the menu |
| F10 | RETAIN two-level linear attention: α per purchase × β per word, linear embedding, ω exact (G8 Eq. 5) | (purchase, word) | **yes** — exact despite context-dependent weights, because context = weighted sum of linear rows | re-weighting | ok | fine | "Purchase 3 mattered most (α), and within it *colour: black* (β)" — a *design constraint* for M3/M11: keep rows linear and attribute with weights frozen |
| F11 | Max-pool / presence profile (Deepr; Deep Sets ρ∘max) | word (or purchase owning each coordinate) | yes (each coordinate ← one owner) | re-weighting (frequency → presence) | ok | good — kills 5-item count noise | "You have bought [word] at least once" — limit case of M13, not new |
| F12 | Hierarchical grouping event→visit→patient (G8) | basket | yes if linear | new object | ok | hm: 94% same-day → baskets ≈ history | OUT (basket structure excluded) |
| F13 | RFM sufficient statistics (G10) | transaction count/time | — | — | — | — | OUT (dates, price; \|H_u\| killed by cosine) |
| F14 | Identity ⊕ topic dual representation, temporal views (G9) | URL/ID + topic | yes | new rows | topic half | ok | = M14 / M1; decay OUT |
| F15 | Non-negativity on word rows for coordinate readability (Med2Vec) | coordinate of q_u | yes | constraint, no new capacity | ok | ok | "Dimension 7 = {black, jersey, trousers}" — redundant with intrinsic cos(e_f,p_l) naming; note only |
Where the accuracy lever sits (brief given: loss is at the prototype layer, lightfm_hist +0.06–0.09): only F5 (replace
the prototype layer by a mixed-membership layer with the same word vocabulary) and F3/F4 (add a neighbour term beside
the prototype score) plausibly move that number; F6/F7/F10/F11 change the composition only. F2 adds a second, item-ID
channel (like M4) and would move accuracy on warm items but not through the prototype layer.

## §4 Families missing from MENU.md
Checked against M1–M16, X1–X2 and the rejected list. Genuinely absent (not a variant of an existing entry):
- **F3 Neighbour-user term ("relevant users" explanation style).** Every survey grid (G2 KNN class, G5 "relevant user or
  item" row, G4 collaborative profiling) lists it; MENU has no user-neighbour object at all (M7 uses population co-purchase
  edges to re-weight the user's *own* purchases, which is a different thing). Exact, additive, thin-history-tolerant, and
  the sentence is the one Herlocker et al. 2000 found most persuasive [A already cites Herlocker; check its verdict].
- **F4 Linear multi-hop propagation inside q_u (LightGCN-style).** Exact and linear, but it injects other users' words into
  q_u, so the "your history" contract weakens; worth listing explicitly as *rejected with reason* rather than silent.
- **F5 Mixed-membership (simplex) user over word-topics.** The topic-model column of G5 and the generative class of G2. It is
  the only family that replaces the prototype layer with a *nameable* alternative (topic = top words) and so is the only
  bottom-up candidate aimed at where the accuracy loss sits. Relation to MENU: "count-bottleneck user" and M10 Dirichlet
  shrinkage are pieces of it, but the mixed-membership *score* Σ_z θ_{u,z} φ_{z,i} is not on the menu.
- **F6 Multi-interest sub-profiles (MIND/ComiRec).** No K-vector user anywhere in A–F. Cost is additivity across
  sub-profiles; at hm's 5 purchases K≈1 so it is an ml-1m-only idea.
- **F10 RETAIN-style exactness constraint for attention.** Not a new form but a missing *requirement* on M3/M11: linear
  rows + weighted sum ⇒ attribution stays exact per (purchase, word) with weights frozen; the attention-faithfulness
  objection in angle E (Jain & Wallace) applies to the weights, not to the decomposition.
Present on MENU in another guise (no action): F2≈A/B co-purchase + M6/M8 (item-level rule confidence as an additive term
is the only uncovered slice), F7≈M3 hard-gated, F8≈C1/M13, F9≈C6, F11≈M13 limit, F14≈M14/M1, F15≈intrinsic naming.
Out of the brief's scope but surfaced by every bottom-up source: F1 order (all three recsys surveys are order-first),
F12 basket level (EHR's visit level; hm's 94% same-day purchases would make it nearly a no-op), F13 recency/monetary.

## Citations
Verified (full text or abstract page opened): Wang et al. IJCAI 2019 arXiv 2001.04830; Wang et al. ACM CSUR 2021 arXiv
1902.04864; He et al. 2023 arXiv 2302.11087; Purificato, Boratto, De Luca 2024 arXiv 2402.09660; Zhang & Chen FnTIR 2020
arXiv 1804.11192; Li et al. MIND CIKM 2019 arXiv 1904.08030; Cen et al. ComiRec KDD 2020 arXiv 2005.09347; He et al.
LightGCN SIGIR 2020 arXiv 2002.02126 (abstract); Si et al. JBI 2021 arXiv 2010.02809; Choi et al. Med2Vec KDD 2016 arXiv
1602.05568; Choi et al. RETAIN NeurIPS 2016 arXiv 1608.05745; Nguyen et al. Deepr 2016 arXiv 1607.07519; Bennett et al.
SIGIR 2012 (ryenwhite.com PDF); Fader, Hardie & Lee JMR 2005 (brucehardie.com PDF); Zaheer et al. Deep Sets NeurIPS 2017
arXiv 1703.06114. Metadata-only verified (Semantic Scholar; body paywalled): Lops, de Gemmis, Semeraro, RecSys Handbook
2011 ch. 3; Pazzani & Billsus, The Adaptive Web 2007; Lops et al. "Semantics-Aware CBRS" Handbook 2015.
[unverified]: Chen et al. RUM WSDM 2018 content; Hofmann 2004 pLSA-CF; Blei et al. LDA 2003; Rich 1979 (cited via G4);
Sontag et al. WSDM 2012 (cited via Bennett §2); Herlocker et al. 2000 (cited via G5).
