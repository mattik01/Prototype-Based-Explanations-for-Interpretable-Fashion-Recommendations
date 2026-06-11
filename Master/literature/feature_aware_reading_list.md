# Reading List — Feature-Aware (Latent-Factor) Recommendation & Intrinsic Explanations

> Iteration 4, 2026-06-11. Goal: ground the design choices in
> `feature_aware_solution_requirements.md`.
> ✅ **Citations verified 2026-06-11** — titles/authors/venues/years web-checked against
> arXiv/ACM/publisher pages; corrections applied inline. Remaining soft spots flagged
> with ⚠️. Still re-check page numbers/DOIs when building the BibTeX.

## ⚡ Immediate prioritization — evaluating dc01 (2026-06-11)

Minimal path to judge `dc01_feature_composed_factors.md`; defer the rest.

1. **S2** — only the intrinsic-vs-post-hoc taxonomy chapters (adjudicates dc01's 5b dispute #6).
2. **S7** (short) — the two feature-integration strategies = the axis dc01 picked a point on.
3. **S5** — pitfalls checklist used in dc01 Step 5; verify it was applied honestly.
4. **C3** (short) — frames the R2 intrinsic-vs-post-hoc stakes. Optional: **E1** = the road dc01 didn't take (Seed 2 fork).
5. Then the seed papers: **B5 → B3 (10 min) → A1 re-skim §3+§5.2 → D1 (positioning skim)**.

Deprioritize for now: S3, S4, S6, S8, S9, F2, I3, B10 — needed for breadth / later candidate comparison, not for dc01.

## 📁 Local PDF copies (2026-06-11)

Nearly all papers are downloaded to **`Master/literature/papers/`** (PDFs gitignored,
~182 MB, 46 PDFs), in subfolders mirroring this list's sections, filenames prefixed
with the list IDs (e.g. `B_feature_aware_mf/B4_rendle_2010_factorization_machines.pdf`).
**Not obtained:** F3 (IEEE Xplore bot-blocks scripted downloads — grab manually in a
browser on uni VPN: https://ieeexplore.ieee.org/document/9720218), B4's libFM *TIST*
companion (closed access), and the C5 depth-on-demand variants ProtoPNeXt/SESM (not
attempted). Notes: S2 is the arXiv preprint (1804.11192; official FnTIR version is
paywalled — content matches, cite the published one), A1 is a copy of the existing
`Master/Protomf-paper.pdf`, and the paywalled ACM/Springer ones (S9, A2, B2, I5) came
through the uni VPN.

## How to use this list (survey-first)

Skim the **§0 surveys** to build the map and steal taxonomy/vocabulary for Related
Work, *then* go deep on the Tier-1 core (see reading path at the end).
**★ = core** · **◦ = depth-on-demand / cite-in-RW.**

### The thesis has two axes — read for both

ProtoMF is just **an MF model (user·item dot product) + a prototype explanation
layer**. The thesis improves both axes, and the literature splits the same way:

- **Q1 — Backbone:** *how do side-features enter a latent-factor / MF model?*
  → **Theme B** (feature-aware / hybrid MF). This is the part that is **not**
  prototype-specific and is a first-class study area in its own right.
- **Q2 — Explanation:** *how does that model explain itself intrinsically?*
  → Themes **C–F**. Prototypes (C, D) are **one vehicle among several**: concept
  bottlenecks (E), **attention over feature interactions** (E / AFM in B), and
  disentanglement (F) are the alternatives — study them as design options and as
  comparison points for the deliberately-decoupled variant (requirements S1).

---

## §0 — Orientation surveys (read/skim first)

- **★ S1. Rudin, C. Chen, Z. Chen, Huang, Semenova, Zhong — "Interpretable Machine
  Learning: Fundamental Principles and 10 Grand Challenges", *Statistics Surveys* 16:1–85,
  2022 (arXiv:2103.11251).** GC#4 = case-based
  reasoning (prototypes); GC#5–6 = disentanglement. The intrinsic-interpretability
  framing for the whole thesis.
- **★ S2. Zhang & Chen — "Explainable Recommendation: A Survey and New Perspectives",
  *Found. & Trends IR* 2020.** Recsys-explainability map; intrinsic-vs-post-hoc
  taxonomy (motivates R2).
- **★ S3. Zhang, Yao, Sun, Tay — "Deep Learning based Recommender System: A Survey and
  New Perspectives", *ACM Computing Surveys* 2019.** Best single map of **feature-aware
  / hybrid deep recommenders** — where FM/NFM/DeepFM/Wide&Deep/CDL sit relative to each
  other. Primary survey for Theme B (Q1).
- **★ S4. X. Chen, Y. Zhang, J.-R. Wen — "Measuring 'Why' in Recommender Systems: a
  Comprehensive Survey on the Evaluation of Explainable Recommendation",
  arXiv:2202.06466 (2022).** Catalogue of explanation-quality metrics
  → pick the eval protocol ([[ideas]] §2).
- **◦ S5. Elhadri, Michalski, Wróbel, Schlötterer, Zieliński, Seifert — "This looks
  like what? Challenges and Future Research Directions for Part-Prototype Models",
  arXiv:2502.09340 (2025; accepted at XAI-2026).** Taxonomy of prototype **failure modes** — the
  design-pitfalls checklist for the prototype vehicle (requirements §4).
- **◦ S6. M. Wang et al. (12 authors; last: X. Zhao) — "Embedding in Recommender
  Systems: A Survey", arXiv:2310.18608 (2023).** How
  embeddings — including **side-feature embeddings** — are constructed; useful menu for
  the backbone.
- **◦ S7. W.-H. Chen, Hsu, Lai, Liu, Yeh, Lin — "Attribute-Aware Recommender System
  Based on Collaborative Filtering: Survey and Classification", *Frontiers in Big Data*
  2:49 (2019).**
  The two integration strategies (content-into-CF vs. extend-CF-with-attributes) —
  vocabulary for "where do features enter" (requirements §4).
- **◦ S8. W. Zhang, Bei, et al. — "Cold-Start Recommendation towards the Era of Large
  Language Models: A Comprehensive Survey and Roadmap" (arXiv:2501.01945, 2025)** + the
  **Awesome-Cold-Start-Recommendation** repo. Side-info / meta-learning / generative
  taxonomy for R5 (LLM-era framed — use the taxonomy, skip the LLM specifics).
- **◦ S9. Ding, Lai, Mok, Chua — "Computational Technologies for Fashion
  Recommendation: A Survey", *ACM Computing Surveys* 56(5):121 (2023,
  10.1145/3627100).** Domain anchor.

---

## A. The model we extend

- **★ A1. ProtoMF — Melchiorre, Rekabsaz, Ganhör, Schedl, RecSys 2022, pp. 246–256.**
  *(Corrected: previous iteration listed a non-existent author "Penz".)* Re-read with the
  feature-aware + intrinsic-explanation lens. See `[[ProtoMF Allesandro]]`.
  - ↪ **dc01**: host architecture kept verbatim (UI double-tie, Eqs. 1–12); only the item ID embedding is replaced; their §6 names our direction as future work.
- **◦ A2. Anchor-based CF (ACF) — Barkan, Hirsch, Katz, Caciularu, Koenigstein,
  CIKM 2021, pp. 2877–2881.** Repo baseline
  (`AnchorBasedCollaborativeFiltering`); conceptual predecessor to prototypes.

## B. Feature-aware / hybrid matrix factorization — the backbone (Q1)

*How side-features enter a latent-factor model. Lead with surveys S3 + S7. This whole
theme is prototype-agnostic — it's the modeling foundation.*

**B.1 — Folding features into the latent factors (the classic templates):**
- **★ B1. Singh & Gordon — "Relational Learning via Collective Matrix Factorization"
  (CMF), KDD 2008.** Jointly factorise the user–item matrix **and** the item–feature
  (and user–feature) matrices with *shared* latent factors. The foundational "fold a
  feature matrix into MF" idea.
- **★ B2. Agarwal & Chen — "Regression-based Latent Factor Models" (RLFM), KDD 2009**
  ✓ verified. Features act as **regression priors on the latent factors**, so a
  cold item's factor is predicted from its features. Foundational cold-start-via-
  features mechanism (R5).
- **★ B3. T. Chen, W. Zhang, Lu, K. Chen, Zheng, Yu — "SVDFeature: A Toolkit for
  Feature-based Collaborative Filtering", *JMLR* 13 (2012)** ✓ verified. A linear model **constructs the user/
  item latent factors directly from features**. This is the cleanest architectural
  template for "ProtoMF-with-features" — features → factors → dot product. Won KDD Cup
  twice; battle-tested formulation.
  - ↪ **dc01**: the generalised features→factors template (§2 equation); justifies real-valued feature weights if ever needed beyond binary fields.
- **★ B4. Rendle — "Factorization Machines", ICDM 2010** (+ "FM with libFM", *ACM TIST*
  2012). The general model subsuming MF + arbitrary side-features via pairwise
  interactions. The reference all later feature-aware models extend.
- **★ B5. Kula — "Metadata Embeddings for User and Item Cold-start Recommendations"
  (LightFM), CBRecSys workshop @ RecSys 2015 (arXiv:1507.08439).** User/item = **sum of feature embeddings** → graceful cold-start.
  Simplest concrete R5 realisation and a likely **baseline**.
  - ↪ **dc01**: primary seed — item embedding = Σ feature embeddings (+ID row, "tags+ids") swapped in under the prototype layer; also the no-prototype ablation baseline.

**B.2 — Content-aware MF (unstructured content → factors):**
- **◦ B6. Wang & Blei — "Collaborative Topic Modeling for Recommending Scientific
  Articles" (CTR), KDD 2011.** PMF + LDA topics on item content; the topics are an
  **interpretable** content representation. The probabilistic ancestor of feature-aware
  explanation.
- **◦ B7. Wang, Wang, Yeung — "Collaborative Deep Learning for Recommender Systems"
  (CDL), KDD 2015.** Deep successor to CTR: an autoencoder on item content aligned with
  the MF embeddings.

**B.3 — Deep feature-interaction descendants (positioning + one that's interpretable):**
- **★ B8. Xiao et al. — "Attentional Factorization Machines" (AFM), IJCAI 2017.**
  Attention learns the **weight of each feature interaction** → an FM that is *both*
  more accurate **and interpretable** (you can read which interactions drove the
  score). This is an **alternative intrinsic-explanation vehicle to prototypes** —
  squarely the "doesn't have to be prototypes" direction. Pair with the caveat E4.
- **◦ B9. NFM (He & Chua, SIGIR 2017), DeepFM (Guo et al., IJCAI 2017), Wide & Deep
  (Cheng et al., DLRS@RecSys 2016).** Deep content-aware CF for positioning / external
  baselines; DCN / xDeepFM if explicit feature-crossing is wanted.
- **★ B10. Volkovs, Yu, Poutanen — "DropoutNet: Addressing Cold Start", NeurIPS 2017.**
  Train with interaction input dropped → forces reliance on content. A **training
  trick** for R5 that composes with any backbone here.

## C. Prototype / case-based interpretability — the paradigm (explanation vehicle 1)

*Caveat: this literature is overwhelmingly **vision**; our features are tabular/
categorical. Treat as conceptual templates to port, not drop-in methods.*
- **★ C1. Chen et al. — "This Looks Like That" (ProtoPNet), NeurIPS 2019.** Prototype
  layer mechanics, projection/push step, cluster + separation losses → analogues for
  feature-aware prototype regularisation (R4).
- **★ C2. M. X. Li, Rudolf, Mattes, Blank, Lioutikov — "An Overview of Prototype
  Formulations for Interpretable Deep Learning", arXiv:2410.08925 (2024; ⚠️ preprint,
  no venue yet as of 2026-06).** The *menu* of prototype-layer formulations; design-relevant
  when deciding how features parameterise prototypes.
- **◦ C3. Rudin — "Stop Explaining Black Box ML Models…", *Nature Mach. Intell.* 2019.**
  Punchy manifesto for intrinsic-over-post-hoc (motivates R2).
- **◦ C4. Prototypes beyond vision** — e.g. "Language Model Meets Prototypes…"
  (Ximing Wen, arXiv:2412.03761, 2024 — ⚠️ only a 2-page AAAI-25 Doctoral Consortium
  abstract; pick a fuller text/tabular-prototype paper if this point needs to carry
  weight). Evidence the paradigm ports to non-image (text/tabular) data — our regime.
- **◦ C5. Variants** — ProtoTree (Nauta et al., CVPR 2021; composing activations in a
  tree), Deformable ProtoPNet (Donnelly et al., CVPR 2022), ProtoPNeXt ("This Looks
  Better than That", Willard et al., arXiv:2406.14675, 2024; training stability); SESM
  ("Learning to Select Prototypical Parts for Interpretable Sequential Data Modeling",
  AAAI 2023; sequential). Pull only if relevant.

## D. Prototypes **in recommendation** (closest prior art)

- **★★ D1. Sankar, Wang, Krishnan, Sundaram — "ProtoCF: Prototypical Collaborative
  Filtering for Few-shot Recommendation", RecSys 2021** ✓ verified. Prototypes as the cold-start /
  long-tail mechanism **in recommendation** (60–80 % Recall@50 on tail items <20
  interactions). **Positioning:** ProtoCF composes prototypes from *interaction
  few-shot*, **not** item metadata — your wedge is grounding prototypes in item
  **features**. Use as template + baseline; contrast explicitly.
  - ↪ **dc01**: distinguished as closest recsys prior art (interaction few-shot, not metadata); its tail-item stratified evaluation adopted into dc01's cold-item protocol (§3.6 amendment).

## E. Concept / feature grounding & attention (explanation vehicles 2–3)

- **★ E1. Koh et al. — "Concept Bottleneck Models", ICML 2020.** Predict concepts, then
  target from concepts only. Cleanest formalism for "the prototype/factor *is* its
  attribute profile" (R4). Fork: concepts-as-prototypes vs. concepts-feeding-factors.
- **★ E2. Alvarez-Melis & Jaakkola — "Self-Explaining Neural Networks" (SENN),
  NeurIPS 2018.** Prediction = Σ(concept × relevance) with a **fidelity/stability
  regulariser** — the precedent for a *tied* intrinsic explanation objective (R1+R2).
- **◦ E3. CBM survey (2025 — ⚠️ unpinned; pick a concrete one when needed); Schrodi,
  Schur, Argus, Brox — "Concept Bottleneck Models Without Predefined Concepts"
  (arXiv:2407.03921, 2024).** Concept leakage + accuracy/interpretability trade-off; learning concepts vs.
  using given (messy) H&M attributes.
- **★ E4. Jain & Wallace — "Attention is not Explanation" (NAACL 2019)** + **Wiegreffe &
  Pinter — "Attention is not *not* Explanation" (EMNLP 2019).** Essential caveat **if**
  you pursue the AFM (B8) attention-as-explanation route: attention weights are
  contested as faithful explanations. Read before claiming attention = explanation.

## F. Disentangled / interpretable representations (grounding for R4 / §4 identifiability)

- **★ F1. Ma et al. — "Learning Disentangled Representations for Recommendation"
  (MacridVAE), NeurIPS 2019 (arXiv:1910.14238).** Micro-disentanglement forces each
  latent dim onto an isolated factor — their example is **"size or colour of a shirt"**.
  The mechanism behind prototype/factor identifiability.
- **★ F2. Dervishaj, Ruotsalo, Maistro, Lioma — "Are Representation Disentanglement
  and Interpretability Linked in Recommendation Models? A Critical Review and
  Reproducibility Study", ECIR 2025 (arXiv:2501.18805).** Cautionary keystone: tests whether disentanglement
  *actually delivers* interpretability in recsys. Read before assuming feature-grounded
  factors are interpretable "by construction".
- **◦ F3. "Disentangled Representation Learning for Recommendation" (IEEE TPAMI 2022;
  title/venue verified — ⚠️ pull exact author list when building BibTeX; companion repo:
  SEM-MacridVAE).**

## G. Explainable-recommendation evaluation (eval method)

*(Surveys S2, S4 lead.)* **◦ G1. Barkan, Schein, Elisha, Bogina, Baklanov, Koenigstein —
"Fidelity-Aware Recommendation Explanations via Stochastic Path Integration", AAAI 2026
(arXiv:2511.18047)** — modern take on the fidelity metric that makes R2 measurable.
Same Barkan/Koenigstein group as ACF (A2).

## H. Fashion domain & visual/attribute features (S3 images)

*(Survey S9 leads.)* **◦ H1. McAuley et al. — visually-aware rec lineage** (Styles &
Substitutes, SIGIR 2015; VBPR, AAAI 2016). **◦ H2. Packer, McAuley, Ramisa —
"Visually-Aware Personalized Recommendation using Interpretable Image Representations",
AI-for-Fashion workshop @ KDD 2018 (arXiv:1806.09820)** — fashion-native
feature-grounded explanation; design inspiration for R4 if images re-enter
(requirements S3).

## I. Forward citations of ProtoMF (added 2026-06-11)

*Semantic Scholar lists **31 papers citing ProtoMF** (DOI 10.1145/3523227.3546756) as of
2026-06-11. Classified by full-text "ProtoMF" mention counts (via ar5iv / extracted
PDFs) where available, abstracts otherwise. PDFs of I.1 are in
`Master/literature/papers/I_protomf_forward_citations/`.*

### I.1 — Directly build on / improve / analyze ProtoMF (read alongside D1)

- **★ I1. "A Plug-and-Play Model-Agnostic Embedding Enhancement Approach for
  Explainable Recommendation" (RVRec), *IEEE Trans. Multimedia* 2025
  (arXiv:2509.03130).** 65 full-text mentions — by far the deepest engagement; enhances
  embeddings for explainability with ProtoMF as a central subject/baseline.
- **★ I2. "Advancing Cultural Inclusivity: Optimizing Embedding Spaces for Balanced
  Music Recommendations", 2024 (arXiv:2405.17607).** 24 mentions. Identifies
  popularity/cultural **bias in ProtoMF's prototype-based embedding space** and fixes it
  there — evidence that prototype spaces are auditable (a selling point for R2), and a
  known-weakness checklist for our variant.
- **★ I3. "UIPC-MF: User-Item Prototype Connection Matrix Factorization for Explainable
  Collaborative Filtering", PAKDD 2024 (arXiv:2308.07048).** 16 mentions. The most
  direct successor: adds a learned **connection matrix between user and item
  prototypes**, explicitly positioned against ProtoMF. Closest competitor on the
  CF-only axis — must-cite, candidate baseline.
- **★ I4. "Modular Debiasing of Latent User Representations in Prototype-Based
  Recommender Systems", ECML-PKDD 2024 (10.1007/978-3-031-70341-6_4).** 13 mentions
  (verified in PDF). Debiases ProtoMF-style user prototype representations — same JKU
  orbit as A1.
- **◦ I5. "Hierarchical Matrix Factorization for Interpretable Collaborative Filtering"
  (HMF), *Pattern Recognition Letters* 2024 (arXiv:2311.13277).** 9 mentions.
  Clustering-hierarchy alternative to prototypes; compares against ProtoMF as an
  interpretable-MF baseline. Useful as a "non-prototype intrinsic MF" comparison point.

### I.2 — Cite ProtoMF only in passing (RW fodder / context, 0–3 mentions)

Prototype-flavoured but independent: PPA (CIKM 2024) + PPA++ (2026) preference
prototypes for cross-domain; "Contrastive Prototype Framework for Calibrating Video
Recommendation" (ACM MM 2025); "Prototype learning based hierarchical decoupling for
multimodal recommendation" (*ESWA* 2026 — ⚠️ paywalled, no abstract; could not verify
depth, possibly I.1 material); fraud-detection user-behavior prototypes (IEEE TBD
2025). Explainability context: SPINRec (**= G1**, 1 mention), LXR (WWW 2024 + TORS
2025), "Extracting Interaction-Aware Monosemantic Concepts" (AAAI 2026,
arXiv:2511.18024), "Can Offline Metrics Measure Explanation Goals?" (TORS),
explainable conversational RS (SIGIR 2023), music-RS explainability (RecSys 2024 DS, 3
mentions — related-work only). JKU/Schedl-group methods papers using ProtoMF as
baseline/context: single-branch architectures (RecSys 2024: arXiv:2409.17864 — **also
relevant to Theme B/R5 multimodal cold-start**; RecSys 2025: arXiv:2508.03518; TORS
2025: arXiv:2509.18807), ACT-R sequential music (RecSys 2023), emotion-based music
(UMAP 2024). Generic cites: temporal MF (IEEE TBD 2026), SMSBPR (*Neurocomputing*
2025), calibrated recommendations (TIST 2025), cross-domain structural alignment (TKDE
2024), top-n metric optimization (ECIR 2024), random seeds (Perspectives@RecSys 2023),
WellFactor patient profiling (2023), fuzzy cold-start music (2025).

**Takeaway for the gap claim:** none of the 31 citing papers grounds prototypes in
**item side-features / content** — the closest (I3, I4, I5) all stay CF-only. The
feature-aware wedge still looks open even in ProtoMF's own forward-citation
neighbourhood.

---

## Tiered reading path

- **Tier 0 — lay of the land (~2–3 days):** S1, S2, S3, S4. (Surveys only.)
- **Tier 1 — design core (~2 weeks):**
  - *Backbone (Q1):* A1 ProtoMF → **B3 SVDFeature** + B4 FM → B1 CMF / B2 RLFM →
    B5 LightFM → B8 AFM → B10 DropoutNet.
  - *Explanation (Q2):* C1 ProtoPNet + C2 formulations → E1 CBM + E2 SENN →
    **D1 ProtoCF** → **F1 MacridVAE + F2 disentanglement critical review**.
- **Tier 2 — depth-on-demand:** everything marked ◦, pulled when a concrete design
  candidate needs it.

## Appropriateness / honesty notes

- **Two axes, deliberately.** Theme B (backbone) is prototype-agnostic and stands on
  its own — it answers "how to make MF feature-aware" independent of explanations.
  Themes C–F are the explanation menu; **prototypes are not the only option** (AFM/
  attention, concept bottlenecks, disentanglement are live alternatives).
- **Modality gap.** Prototype/CV literature (C) is vision-centric; our features are
  tabular/categorical. Mechanics port (C4); part-localisation tricks mostly don't.
- **Position, don't just cite,** the bracketing works: ProtoCF (prior art), AFM
  (alternative explanation vehicle), and the disentanglement critical review (validity
  threat).

## Gaps for the next iteration

- A model that grounds **prototypes/latent factors in item side-features** *and*
  reads them out as explanations — the union of B and C/E — still looks **open**; the
  thesis's specific wedge. **Now corroborated by the §I forward-citation sweep
  (2026-06-11): all 31 ProtoMF citers stay CF-only.** Search terms: *"content prototype recommendation"*,
  *"attribute prototype"*, *"concept-based recommendation"*, *"feature-based latent
  factor explanation"*.
- Prototype/factor **diversity & identifiability** regularisers (link Theme C losses to
  Theme F disentanglement penalties).
- H&M-specific baselines — Kaggle write-ups already in `Master/docs/kaggle_sources/`.

---
### Source links (web-search seeds — verify before citing)
- Rudin et al. 2022, 10 Grand Challenges: https://arxiv.org/abs/2103.11251
- DL-based Recommender Systems survey (CSUR 2019): https://arxiv.org/abs/1707.07435
- Embedding in RecSys survey (2023): https://arxiv.org/pdf/2310.18608
- Part-Prototype Models survey (2025): https://arxiv.org/pdf/2502.09340
- Prototype Formulations overview (2024): https://arxiv.org/html/2410.08925v3
- Prototypes for text classification (2024): https://arxiv.org/pdf/2412.03761
- CMF (Singh & Gordon, KDD 2008): https://dl.acm.org/doi/10.1145/1401890.1401969
- SVDFeature (JMLR 2012): https://wnzhang.net/papers/svdfeature-jmlr.pdf
- Factorization Machines (Rendle, ICDM 2010): https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
- Attentional Factorization Machines (IJCAI 2017): https://www.ijcai.org/proceedings/2017/0435.pdf
- Collaborative Topic Regression (KDD 2011): https://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf
- Collaborative Deep Learning (KDD 2015): https://www.researchgate.net/publication/280105148
- Attention is not Explanation (NAACL 2019): https://arxiv.org/abs/1902.10186
- Attention is not not Explanation (EMNLP 2019): https://arxiv.org/abs/1908.04626
- ProtoCF (RecSys 2021): https://aravindsankar28.github.io/files/ProtoCF-RecSys2021.pdf
- MacridVAE (NeurIPS 2019): https://arxiv.org/abs/1910.14238
- Disentanglement ↔ interpretability critical review (2025): https://arxiv.org/pdf/2501.18805
- Cold-start survey, LLM era (2025): https://arxiv.org/pdf/2501.01945
- Awesome-Cold-Start-Recommendation: https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation
- Attribute-aware CF survey: https://www.frontiersin.org/articles/10.3389/fdata.2019.00049/full
- Explainable Rec survey (Zhang & Chen): https://dl.acm.org/doi/abs/10.1561/1500000066
- Measuring "Why" eval survey: https://arxiv.org/pdf/2202.06466
- Fashion recommendation survey (CSUR 2023): https://dl.acm.org/doi/10.1145/3627100
- Interpretable visual representations: https://arxiv.org/pdf/1806.09820
