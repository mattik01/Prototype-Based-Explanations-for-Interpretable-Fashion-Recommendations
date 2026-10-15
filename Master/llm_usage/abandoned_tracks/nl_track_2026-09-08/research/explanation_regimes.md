# Explanation regimes for nonlinear composition below a frozen prototype layer — literature evaluation

*nl track, sub-problem 3 of 3 (PICKUP_2026-09-09). Written 2026-09-09. Companion files: `item_side_algorithms.md`, `user_side_algorithms.md` (other agents). Read-only inputs: research plan §0/§4–7, sp0_w1 memo §3, vault 2026-07-12_1200 and 2026-07-14_1504, taxonomy rows 8/13/16.*

**Verification status (GR6).** Every citation below was checked this session against its arXiv abstract page, publisher/Semantic Scholar record, or the local PDF text. Tag legend: ✅ arXiv/DOI page fetched; ✅ᴾ bibliographic record verified via Semantic Scholar API (ACM DL 403), content read from local PDF or the record's abstract; ⚠️ unverified. Quantitative claims carry table/section pointers; where only the abstract was read the claim is limited to what the abstract says.

---

## 0. The question, sharpened

The model is `score(u,i) = Σ_l t_l (1 + cos(q_u, p_l))` with q_u composed from named categorical rows (item attribute fields on the fI side; purchases on the fU side). Three composition families are now admissible below the frozen layer:

| regime | composition | attribution object | what is exact | what needs a story |
|---|---|---|---|---|
| **1 — exact-linear** | `q = Σ_r c_r e_r` | per-row share of each `cos(q,p_l)` given ‖q‖ | the split of *each activation* into row terms | the normaliser ‖q‖ is a global nonlinearity: shares are exact *given ‖q‖* (memo §3.1: Shapley disagrees about the top purchase for 26 % of hm users) |
| **2 — exact-given-weights** | `q = Σ_r w_r(x) c_r e_r` (attention / RETAIN) | weighted row shares | the split, once the weights are read as data | whether `w_r` are *faithful* importances — the whole attention debate (§B) |
| **3 — post-hoc** | `q = ρ(Σ_r φ(e_r))`, MLP, pairs, target-conditioned | Shapley / IG / deletion over rows | nothing structurally; exact Shapley is *enumerable* here (≤ 32 field coalitions, ≤ 4,096 purchase coalitions at |H| ≤ 12) | choice of absent-player baseline; disagreement between attribution methods; efficiency holds by construction but additivity of the *function* does not |

Regime 1 is the lineage's current object (vault 2026-07-12_1200, caveat (b): "cosine is a tolerated nonlinearity"). The memo's Shapley read-out turns caveat (b) from a sentence into a number, and that number is what §E's *generalised Score Fidelity* has to carry. The vault 2026-07-14_1504 position ("fidelity is exact by construction at the contribution level; the problem relocates to semantics") stays true for regime 1 *given ‖q‖*, becomes conditional in regime 2, and is simply false in regime 3 — the thesis must say which regime each form lands in.

---

## A. Attribution for additive-with-nonlinear-upstream models

### A.1 The family and where each member is exact

| model | form | exact attribution? | anchoring / identifiability | accuracy cost reported | verified |
|---|---|---|---|---|---|
| **GA²M** — Lou, Caruana, Gehrke, Hooker 2013, *Accurate intelligible models with pairwise interactions*, KDD 2013 (DOI 10.1145/2487575.2487579) | `g(y) = Σ_i f_i(x_i) + Σ_{(i,j)∈S} f_ij(x_i,x_j)`, pairs chosen by FAST | **yes** — every unary and pair term is a summand; pair term is a 2-D heat map | none needed: terms are indexed by named fields | Sec. 6, Tables 2–3: GA²M-FAST "nearly as accurate as full-complexity random forest models", sometimes better; GAM→GA²M gap is the interaction share | ✅ (Cornell PDF fetched, text read) |
| **NAM** — Agarwal et al. 2021, *Neural Additive Models: Interpretable ML with Neural Nets*, NeurIPS 2021 spotlight, arXiv 2004.13912 | `Σ_i f_i(x_i)`, each `f_i` a small net | **yes** per feature (shape functions) | none (one net per named feature) | abstract: "more accurate than … logistic regression", competitive with SOTA on tabular | ✅ |
| **NODE-GAM / NODE-GA²M** — Chang, Caruana, Goldenberg 2022, ICLR 2022 spotlight, arXiv 2106.01613 | differentiable oblivious trees per unary/pair term | **yes** (same additive skeleton) | none | abstract: better than other GAMs on large data, remain interpretable; self-supervised pretraining possible | ✅ |
| **HONAM** — Kim, Choi, Kim 2025, *Higher-order Neural Additive Models*, ICDM 2025, arXiv 2209.15409 | additive over interaction terms of arbitrary order | **yes** per term ("each interaction term remains individually interpretable") | none; legibility drops with order | abstract: accuracy up over NAM | ✅ |
| **NA²M / NB²M** — Kishimoto, Yamanishi, Matsuda, Shirakawa, *Neural Additive and Basis Models with Feature Selection and Interactions*, arXiv 2606.19850 (2026 posting; abstract page says "accepted at PAKDD 2024") | NAM/NBM + feature-selection layer + two-input pair nets | **yes** per term | none | abstract: maintains/improves baseline accuracy at much lower cost; plain NA²M does not scale past ~100 features (search-result note) | ✅ |
| **InstaSHAP** — Enouen & Liu 2025, arXiv 2502.14177 | trains a GAM whose components *are* the Shapley values | bridge result: on a GAM the additive components coincide with SHAP; the GAM's representational limits are *inherent to SHAP itself* | — | — | ✅ |
| **RETAIN** — Choi et al. 2016, NeurIPS 2016, arXiv 1608.05745 | two-level attention (visit `α_t`, variable `β_t`) over *linear* embeddings; prediction linear in `Σ_t α_t β_t ⊙ v_t` | **yes given weights**: the paper derives a per-(visit, variable) contribution as a product of the visit weight α_t, the variable weight β_t and the linear embedding/output weights (RETAIN §3 interpretation; formula paraphrased, not re-read this session) — the regime-2 prototype | the weights are the nonlinear step; RETAIN itself offers no faithfulness test beyond case studies | abstract: "comparable to RNN" on 14 M visits / 263 k patients | ✅ |
| **SENN** — Alvarez-Melis & Jaakkola 2018, NeurIPS 2018, arXiv 1806.07538 (local E2) | `f(x) = g(θ(x)_1 h(x)_1, …, θ(x)_k h(x)_k)`, `g` additive; concepts `h(x)` and coefficients `θ(x)` both learned | **yes given `θ(x)`** (a locally linear model); *not* exact globally — coefficients drift with x | **anchoring loss required**: robustness term makes `θ` locally Lipschitz (Sec. 3, Def. 3.1–3.3, §4 "robustness loss … behave locally as a linear function"); trades accuracy for stability (§3 end: "trades off performance against stability") | Sec. 5.3 / Fig. 3: faithfulness measured as corr(relevance, prob-drop-on-removal); removal is exact because `θ_i := 0` (E2 text §"Faithfulness") | ✅ |
| **CBM** — Koh et al. 2020, ICML 2020, arXiv 2007.04612 (local E1) | `x → ĉ (named concepts) → y`, `y` any function of `ĉ` (linear in their experiments) | attribution to *concepts* exact when the second stage is linear; attribution to raw inputs is post-hoc | concepts are anchored by *supervision* (independent / sequential / joint training) — the anchoring loss is the concept loss itself | **Table 1** (±2 SD): OAI RMSE indep 0.435 / seq 0.418 / joint 0.418 vs standard 0.441 (no cost); **CUB error indep 0.240 / seq 0.243 / joint 0.199 vs standard 0.175** — i.e. +2.4 pts (joint) to +6.8 pts (seq) error for the bottleneck; multitask 0.162 but no intervention | ✅ |
| **Leakage critique** — Mahinpei et al. 2021, *Promises and Pitfalls of Black-Box Concept Learning Models*, arXiv 2106.13314 | — | — | concept representations "encode information beyond the pre-defined concepts"; standard mitigations do not fully remove it → the concept read-out over-states what the named channel carries | — | ✅ |
| **Identifiability critique** — Margeloiu et al. 2021, *Do Concept Bottleneck Models Learn as Intended?*, ICLR 2021 RAI workshop, arXiv 2105.04289 | — | — | post-hoc probing shows concepts "do not correspond to anything semantically meaningful in input space" (esp. joint training) | — | ✅ |
| **CEM** — Espinosa Zarlenga et al. 2022, *Concept Embedding Models*, NeurIPS 2022, arXiv 2209.09056 | high-dimensional concept embeddings instead of scalars | attribution to concepts becomes *per-embedding*, no longer scalar-exact | claims to lift the accuracy–explainability trade-off; leakage is the price of the extra capacity | abstract: competitive with unconstrained nets | ✅ |
| **UCBM** — Schrodi et al. 2024, *CBMs Without Predefined Concepts*, arXiv 2407.03921 (local E3) | unsupervised concept discovery + input-dependent sparse concept selection | exact w.r.t. discovered concepts; naming is post-hoc | none supervised | Table 1: "narrows the gap to the original black-box model" (top-1, 3 seeds) | ✅ (local) |

### A.2 What this says for the nl track

1. **Exactness is a property of the *last mile*, not of the upstream nets** — GA²M/NAM/HONAM are exact because the score is a sum over named terms whatever computes them. This is the plan's §0 rule restated by the tabular-GAM literature. For the fI side, an *explicit pair unit* (SP1(a)) is GA²M with categorical `f_ij`: exact, no anchoring, legibility cost = number of pairs; FAST-style selection (Lou 2013 §4) is the literature's answer to "which pairs".
2. **RETAIN is regime 2 in the literature's own words**: the contribution formula is exact *once the attention values are treated as data*. RETAIN never tests whether the weights are faithful — that is exactly the gap §B fills. The fU attention form (SP2/SP3) inherits both the formula and the gap.
3. **Bottlenecks need an anchoring loss and pay for it.** CBM's supervised concept loss *is* the anchor; without it (joint training, Margeloiu) or with too much capacity around it (Mahinpei leakage) the named channel stops meaning what it says. Reported cost on CUB: +2.4–6.8 error points (Koh Table 1). SENN anchors differently — a Lipschitz penalty on the coefficient generator — and states the trade explicitly. **For SP7 (named bottleneck over feature words)** the identifiability lesson is dc04's: the re-weighting map must be pinned (sparsity + supervision-free anchoring such as tying the bottleneck basis to the linear rows), and the *leakage* to measure is "how much of the score is carried by the re-weighting rather than the words" — this is the §E *bottleneck re-weighting divergence*.
4. **InstaSHAP closes the loop with §C**: on a purely additive function the Shapley values *are* the additive components. So regime 1 ≡ regime 3 for an additive game — the memo's unit test (φ_fixed = linear share to 1e-6 on the dot models) is exactly this theorem. Under the cosine layer the function is *not* additive in the rows, so the two must differ, and they do (memo §3.1).

---

## B. Attention faithfulness — the test battery

### B.1 Sources

| paper | claim | method | verified |
|---|---|---|---|
| **Jain & Wallace 2019**, *Attention is not Explanation*, NAACL 2019, arXiv 1902.10186 (local E4a) | attention weights "frequently uncorrelated with gradient-based measures"; alternative attention distributions give the same output | Kendall τ between α and gradient (τ_g) / leave-one-out (τ_loo) importances (§4.1, Alg. 1, Table 2); permuted attention (Alg. 2) and *adversarial* attention maximally far from α̂ with ε-equal output (§4.2). Headline example: τ_g = 0.29, median output change under permutation 0.006 (intro example, Fig. 1) | ✅ |
| **Wiegreffe & Pinter 2019**, *Attention is not not Explanation*, EMNLP 2019, arXiv 1908.04626 (local E4b) | J&W's adversary is per-instance and detached from training, so it does not disprove attention | four tests: (1) **uniform-weights baseline** (§3.2, Table 2 — on the two News datasets uniform ≈ trained ⇒ "attention is not explanation if you don't need it"); (2) **seed variance** calibration (§3.3, Fig. 2); (3) **diagnostic MLP with frozen weights** — a non-contextual MLP fed the trained attention as fixed weights must beat uniform (§3.4); (4) **trained adversary** — a whole model trained to keep outputs but move attention, judged on the diagnostic (§4) | ✅ |
| **Serrano & Smith 2019**, *Is Attention Interpretable?*, ACL 2019, arXiv 1906.03731 | attention is "a noisy predictor" of importance; gradient-based rankings often better | erase attended items in attention order until the decision flips; count how many erasures needed vs. gradient ordering | ✅ |
| **Bastings & Filippova 2020**, *The elephant in the interpretability room*, BlackboxNLP 2020, arXiv 2010.05607 | when the question is "which inputs mattered", use input-saliency methods, not attention | position paper; asks authors to state the explanation's purpose and audience | ✅ |
| **Liu et al. 2022**, *Rethinking Attention-Model Explainability through Faithfulness Violation Test*, ICML 2022, arXiv 2201.12114 | high attention can sit on features that *suppress* the prediction; most attention explanations (raw, attention×gradient) violate polarity | faithfulness = importance correlation **and** polarity consistency; the violation test checks the sign | ✅ |
| **Li et al. 2024**, *Attention Is Not the Only Choice: Counterfactual Reasoning for Path-Based Explainable Recommendation*, TKDE 2024, arXiv 2401.05744 | recsys-specific: path-attention weights "are unstable for different reproductions, and they may not always align with human intuition" | replaces attention explanations with counterfactual path weights; proposes new quantitative tests | ✅ |
| Recsys attention-faithfulness evaluations 2021–2026 beyond Li 2024 | search turned up *no* dedicated recsys study running the J&W/W&P battery on history attention (SASRec-style attention maps are visualised, not tested) | — | (negative finding of this session's search; ⚠️ may exist) |

### B.2 Distilled battery for a regime-2 recommender (attention over linear rows)

The linear-rows setting makes the battery *cheaper and sharper* than in NLP: the uniform model is exactly today's linear fU/fI, and the diagnostic MLP is unnecessary because the readout is already linear.

| # | test | construction here | pass criterion | fail reading |
|---|---|---|---|---|
| T1 | **uniform baseline** (W&P §3.2) | `w_r ≡ 1/|H|` = the existing linear arm, same hyperopt budget | attention arm beats uniform by > noise yardstick (0.0026 HR@10 hm, plan §6) | "attention not needed" — weights carry no information worth explaining (W&P: on such data the debate is moot) |
| T2 | **seed variance** (W&P §3.3) | ≥3 seeds; per-user Spearman / top-1 agreement of `w` across seeds | agreement well above the *between-user* baseline and above the adversary's distance | weights are not a stable property of the data; do not print them |
| T3 | **frozen-weight retrain** (W&P §3.4) | freeze `w` from seed A, retrain rows; also feed `w` to a fresh linear model | retrained model ≈ original and ≫ uniform | weights are an optimisation artefact rather than a summary of the data |
| T4 | **adversarial same-output weights** (J&W §4.2 per-instance; W&P §4 trained) | maximise JSD(w', w) s.t. |Δscore| < ε for the recommended item — **cheap here**: the score is bilinear in w, so the feasible set is an explicit polytope | adversary cannot move mass far without changing the score (small attainable JSD) | large attainable JSD ⇒ the weights are one of many equivalent stories (Rashomon at the weight level) |
| T5 | **agreement with gradient / LOO / Shapley** (J&W §4.1; plan §7 "attribution agreement") | Kendall τ / Spearman between `w_r·share_r` and exact Shapley `φ_r` (enumerable) and leave-one-out Δrank | τ clearly above the regime-1 baseline (memo: linear-share vs Shapley Spearman 0.96 hm) | weights explain the *composition* but not the *score* |
| T6 | **polarity consistency** (Liu 2022) | sign of `w_r·share_r` vs sign of `φ_r` / Δscore on removal | few high-weight rows with negative Shapley | high attention on evidence that *lowers* the score — memo already shows 40–48 % of purchases carry negative φ, so this test will bite |
| T7 | **erasure-to-flip** (Serrano & Smith) | remove rows in attention order vs Shapley order vs random until the target leaves top-10 | attention order needs no more erasures than Shapley order, far fewer than random | attention ordering is a noisy proxy |

Reporting rule: T1 decides whether the regime-2 arm is *worth* explaining; T2–T4 decide whether the weights are *a property of the model*; T5–T7 decide whether they are *importances*. A form passes regime 2 only if all three groups pass; otherwise its weights are demoted to "composition parameters" and attribution falls back to regime 3.

---

## C. Post-hoc attribution for set / history encoders

### C.1 Sources

| paper | what it fixes for us | verified |
|---|---|---|
| **Lundberg & Lee 2017**, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017, arXiv 1705.07874 | SHAP = the unique additive attribution with local accuracy, missingness, consistency; unifies LIME/DeepLIFT/etc. Efficiency: Σφ = f(x) − f(baseline) by construction — additivity of the *explanation*, not of the *model* | ✅ |
| **Castro, Gómez, Tejada 2009**, *Polynomial calculation of the Shapley value based on sampling*, Computers & Operations Research 2009, DOI 10.1016/j.cor.2008.04.004 | permutation-sampling estimator with polynomial cost; the memo's `renorm`-game estimator on ml-1m is this | ✅ᴾ (S2 record; abstract not exposed by publisher — error-bound details ⚠️ not re-read) |
| **Sundararajan, Taly, Yan 2017**, *Axiomatic Attribution for Deep Networks*, ICML 2017, arXiv 1703.01365 | IG = path integral of gradients from a *baseline*; axioms Sensitivity + Implementation Invariance; completeness Σ IG = f(x) − f(x'). Baseline choice is the free parameter | ✅ |
| **DeYoung et al. 2020**, *ERASER*, ACL 2020, arXiv 1911.03429 | faithfulness metrics **comprehensiveness** (drop when rationale removed) and **sufficiency** (performance with rationale only); AOPC-style curves | ✅ |
| **Ghazimatin et al. 2020**, *PRINCE*, WSDM 2020, arXiv 1911.08378 | counterfactual explanation = "minimal set of user actions that, if removed, changes the recommendation to a different item"; polynomial algorithm on RWR graphs; evaluation: explanation size + user study | ✅ |
| **Tran, Ghazimatin, Saha Roy 2021**, *ACCENT*, SIGIR 2021 short, arXiv 2105.05008 | influence-function extension of PRINCE to neural recommenders (NCF, RCF), ML-100K; explanation = small action set flipping the top-1 | ✅ |
| **Tan et al. 2021**, *Counterfactual Explainable Recommendation* (CountER), CIKM 2021, arXiv 2108.10539 | minimal *aspect* perturbation flipping the recommendation; metrics **PN / PS** (S4 §4 definitions: PN = item leaves list when explanation features ignored; PS = item stays when only they remain) + explanation complexity/strength | ✅ |
| **Zhong & Nègre 2022**, *Shap-enhanced counterfactual explanations for recommendations* (SHAP4Rec), ACM SAC 2022, DOI 10.1145/3477314.3507029 | SHAP surrogate + counterfactual reasoning for recsys; perturbation grounded in Shapley | ✅ᴾ |
| **Barkan et al. 2024**, *A Counterfactual Framework for Learning and Evaluating Explanations for RS* (LXR), WWW 2024, DOI 10.1145/3589334.3645560 | learned post-hoc explainer with self-supervised counterfactual losses; introduces the AUC-perturbation fidelity metrics G1 reuses | ✅ᴾ |
| **Baklanov et al. 2024**, *CEERS*, RecSys 2024, DOI 10.1145/3640457.3688015 | counterfactual evaluation protocol for RS explanations (CV-style deletion/insertion transplanted to RS) | ✅ᴾ |
| **Baklanov et al. 2025**, *Refining Fidelity Metrics for Explainable Recommendations*, SIGIR 2025, DOI 10.1145/3726302.3730242 | metrics that separate *supporting* from *contradicting* features and evaluate at **fixed explanation length** rather than fixed perturbation % | ✅ᴾ |
| **Barkan et al. 2026**, *SPINRec*, AAAI 2026, arXiv 2511.18047 (local G1) | stochastic baselines for path integration on binary histories; metrics POS@K_r,K_e, DEL@K_e, INS@K_e, CDCG@K_e + AUC (G1 §"Evaluation", Tables 1–2) across MF/VAE/NCF × ML1M/Yahoo/Pinterest | ✅ |
| **Nguyen et al. 2026**, *From Top-1 to Top-K: Reproducibility & Benchmarking of Counterfactual Explanations for RS*, SIGIR 2026 repro, arXiv 2604.19663 | benchmarks 11 methods (LIME-RS, SHAP, PRINCE, ACCENT, LXR, GREASE, GNN explainers) on effectiveness / sparsity / cost; item-level vs list-level scope | ✅ |
| **Pliatsika et al. 2025**, *ShaRP*, VLDB 2025, arXiv 2401.16744 | Shapley for *rankings*: profit functions = rank, top-k membership, pairwise preference — the right "game payoff" vocabulary for a ranked recommender | ✅ |
| **Huang & Marques-Silva 2023/24**, *A Refutation of Shapley Values for Explainability*, arXiv 2309.03041 | Shapley can rank an irrelevant feature above a relevant one for Boolean functions of any arity — a caution against treating φ as ground truth | ✅ |
| **Hwang et al. 2025**, *SHAP-based Explanations are Sensitive to Feature Representation*, FAccT 2025, arXiv 2505.08345 | encoding/binning choices change SHAP rankings — for us: field vs. tuple vs. one-hot "players" give different attributions | ✅ |

### C.2 The three design decisions, answered

**(i) Absent-player baseline for a history.** SHAP, IG and SPINRec all make the baseline the explanation's hidden free parameter (Sundararajan: a "neutral" input; G1 §Intro: "all-zero 'cold user' vectors fail to reflect" real behaviour, hence sampled baselines). For our composition three baselines are *meaningful*, not arbitrary, because the model literally has a value for each:

| baseline | game | when it is the right one |
|---|---|---|
| **empty coalition = zero row** (`q = 0` → shifted cosine undefined; use `q = e_ID` in ids arms, or the +1 constant channel only) | memo's `fixed` game on the noid arms | noid arms; measures "what the history adds to nothing" |
| **ID-only** (`q = e_ID`) | `fixed` game on ids arms (memo: ID share ≤ 2.2 %) | ids arms — the ID row is the *disclosed* residual, so it is the honest absent-player state |
| **coalition mean** (renormalise over present rows) | memo's `renorm` game | the counterfactual-faithful game: what the model *would* compute for the user with that purchase removed — this is what a "without your X" sentence asserts |
| population-sampled histories (SPINRec) | — | only for regime-3 forms whose ρ has no natural zero (Deep Sets with a bias-heavy ρ); report the baseline sensitivity as G1 does (Fig. 2, κ sweep) |

Rule: **the renorm/leave-one-out game is the default for regime 3** (it is what counterfactual recsys evaluates, PN/PS and ACCENT/PRINCE all remove actions), the fixed game is the *bridge* to regime 1 (where it is the linear share exactly, memo §3.1 row 1). Always state which.

**(ii) Shapley ≠ linear shares for non-additive functions.** This is not a bug of either: for an additive game φ = component (InstaSHAP; Lundberg's local accuracy), for any other game Shapley averages marginal contributions over coalitions and *must* differ. The memo measured it *inside regime 1*: fixed-game φ vs linear share on the cosine models differ by L1 0.44–0.63 of the attributed mass (hm) and the top purchase changes for 26–28 % of hm users (Spearman 0.96); on ml-1m L1 > 1 and top-1 agreement 50–59 % — with the caveat that ml-1m φ are permutation estimates. The reason is the normaliser: cos(q_S, p) rescales with ‖q_S‖ for every coalition S, so the game is additive in direction but not in the score. Consequence for the taxonomy: the regime-1 share is exact *given ‖q_H‖*, and its distance to Shapley is the disclosed price of the cosine.

**(iii) How recsys evaluates counterfactuals.** Three metric families, all deletion/insertion over the user's own actions, none needing ground truth:
- *flip / necessity*: PN (Tan 2021 via S4 §4), POS@K_r,K_e (G1 = "item drops out of top-K_r when top-K_e features removed"), PRINCE/ACCENT explanation = minimal flipping set (size reported);
- *retention / sufficiency*: PS (Tan 2021), INS@K_e (G1, confidence recovered from top-K_e features alone), ERASER sufficiency;
- *graded rank change*: CDCG@K_e (G1, rank degradation after removal), DEL@K_e (confidence drop), AUC of the perturbation curve (LXR/G1), "performance shift" (S4, Liu 2020); Baklanov 2025 adds fixed-length comparison and the support-vs-contradict split.
The memo's counterfactual read-out (mean |Δrank| per removed purchase 1.29–1.51 under fU vs 3.09–3.17 under the dot; 12 % vs 28 % of hits with a *necessary* purchase) is already POS@(10, 1) and a CDCG-like quantity; the SPINRec/Baklanov naming should be adopted so the numbers are comparable to the literature.

---

## D. Fidelity metrics for recommender explanations

| source | metric | what it measures | how it relates to Score Fidelity | verified |
|---|---|---|---|---|
| **Peake & Wang 2018**, *Explanation Mining: Post Hoc Interpretability of Latent Factor Models for RS*, KDD 2018, DOI 10.1145/3219819.3220072 | **Model Fidelity** = \|explainable ∩ recommended\| / \|recommended\| (S2 §4.3 restatement) | *coverage* of the recommended list by post-hoc association rules — counts items, not score mass | Score Fidelity generalises the same ratio from "items that have *an* explanation" to "share of the *score* that named units carry" (vault 2026-07-15_1137) | ✅ᴾ (S2 API record; formula from local S2 §4.3 text) |
| **Chen, Zhang, Wen 2022**, *Measuring "Why" in RS*, IJCAI 2022 survey, arXiv 2202.06466 (local S4) | taxonomy of evaluation; **PN / PS** (from Tan 2021) and **performance shift** (Liu 2020) as the model-perspective, ground-truth-free metrics (S4 §4) | necessity / sufficiency of the explanation set; performance change after removing key history items | supplies the *counterfactual* leg; nothing in S4 measures the *explained share of a score* — the gap the thesis fills | ✅ |
| **Barkan et al. 2026 SPINRec** (G1) | POS@K_r,K_e, DEL@K_e, INS@K_e, CDCG@K_e, AUC variants (G1 §Evaluation, Tables 1–2) | counterfactual fidelity of a *ranking* of user actions | the standard for regime-3 comparisons; also the protocol for reporting baseline sensitivity | ✅ |
| **Baklanov et al. 2025** (SIGIR 2025) | refined counterfactual metrics: support vs contradict, fixed explanation length | fixes the "contradicting evidence counted as explanation" flaw — directly relevant to the memo's 40–48 % negative-φ purchases | Score Fidelity should be split into supporting and contradicting named mass | ✅ᴾ |
| **Zanon, Manzato, Rocha 2025**, *Can Offline Metrics Measure Explanation Goals?*, ACM TORS 2025, arXiv 2310.14379 | offline diversity/popularity/recency metrics of path explanations vs user perception | offline metrics need refinement to track user goals | reminder that fidelity is a *model-perspective* claim; do not oversell it as user benefit | ✅ |
| **Steck, Ekanadham, Kallus 2024**, *Is Cosine-Similarity of Embeddings Really About Similarity?*, WWW 2024 Companion, arXiv 2403.05440 | analytic: under regularised linear MF, cosine similarities can be "arbitrary and therefore meaningless" — non-unique, controlled by the regulariser | the *normaliser* is not just a nonlinearity for attribution (memo) but a semantically fragile step | motivates reporting Score Fidelity both with and without the normaliser (regime "exact-given-‖q‖") and keeping taxonomy row 10 (cosine reliability) attached to every regime tag | ✅ |
| **A standard for "explained share of a score"** | — | none found in the recsys literature this session; the closest are ERASER sufficiency (share of *performance* retained) and Lundberg's local accuracy (Σφ = f − f₀, a *completeness* identity, not a share on named units) | Score Fidelity remains the thesis's own construct; anchor it to Peake & Wang (ratio form), Lundberg (completeness) and SPINRec/Baklanov (counterfactual validation) | (negative finding) |

---

## E. Synthesis — generalised Score Fidelity and the per-form table

### E.1 Definition

For user u, item i, prototype activations `a_l = 1 + cos(q, p_l)`, score `s = Σ_l t_l a_l`. Let `U` be the set of *named units* (attribute rows, feature words, purchases; never the ID row, never the +1 constant). An attribution `A` assigns each unit r a share `φ_r` of s with a *completeness identity* `Σ_r φ_r + φ_ID + φ_const + ε_A = s`.

> **SF_A(u,i) = Σ_{r∈U} |φ_r| / (Σ_{r∈U∪{ID,const}} |φ_r| + |ε_A|)**, reported with its **regime tag** and **ε_A**, split into **SF⁺ (φ_r > 0)** and **SF⁻ (φ_r < 0)** (Baklanov 2025 support/contradict).

| regime tag | attribution A | ε_A | mandatory companion number |
|---|---|---|---|
| **exact** | linear shares, function additive in rows (plain dot: `lightfm_hist`) | 0 (float) | none — φ = Shapley (memo row 1: max Δ 1.6e-6) |
| **exact-given-‖q‖** | linear shares of each `cos(q,p_l)` with ‖q_H‖ held fixed (today's fI/fU) | 0 *conditional* on ‖q‖ | **cosine-normaliser disagreement**: L1(φ_fixed-Shapley, linear)/Σ\|share\|, top-1 agreement, Spearman (memo: 0.63 / 74 % / 0.96 fU hm). Report per dataset; ml-1m carries MC error |
| **exact-given-weights** | `w_r · share_r` | 0 conditional on w | battery T1–T7 (§B.2) results; agreement τ(w·share, φ) is the faithfulness number |
| **Shapley-with-error** | exact enumeration where \|H\| ≤ cap, permutation sampling above (Castro 2009) | MC standard error (0 when enumerated) | baseline named (ID-only / empty / renorm / sampled), coverage (% users enumerated exactly), and POS/INS/CDCG at K_e = 1 |

**How to report the measured cosine-normaliser disagreement.** As a *property of regime 1*, not as a failure: "Per-row shares are exact given the full-history normaliser. Letting the normaliser move with the coalition (exact Shapley over purchases, fixed-weight game) changes the attributed mass by 44–63 % L1 on hm and flips the top purchase for 26–28 % of users, at rank correlation 0.96; on the plain-dot control the two coincide to float precision." Quote hm as the clean number, ml-1m with the sampling caveat (456/2000 users skipped, permutation estimates), and pin the +1 constant channel and the ID share as separate rows. Steck 2024 supplies the second sentence: the normaliser is also the step whose semantics the regulariser can make arbitrary, so the *exact* and *exact-given-‖q‖* rows are not interchangeable even when their SF values agree.

### E.2 Per-form table

| form (plan SP) | regime | mandatory tests | typical fidelity / accuracy trade in the literature | R2 compliance (intrinsic vs instrument) |
|---|---|---|---|---|
| **linear sum** (today: fI, fU, lightfm_hist) | exact (dot) / exact-given-‖q‖ (proto) | normaliser-disagreement read-out (done, memo §3.1); ID share | GA²M/NAM: exact attribution at "nearly RF" accuracy on tabular (Lou 2013 Tables 2–3); here dot beats proto by 0.05–0.09 HR@10 — the layer, not the sum, pays | **intrinsic** — the explanation is the computation; Score Fidelity is a read-out, no training knob needed |
| **attention-weighted sum** (SP2/SP3, RETAIN form) | exact-given-weights | full battery T1–T7; τ(w·share, φ); polarity test especially (40–48 % negative-φ purchases exist already) | RETAIN: "comparable to RNN" (abstract) with case-study-only faithfulness; NLP battery: W&P — uniform often within noise on easy tasks, adversaries exist but fail the frozen-weight diagnostic; Liu 2022 — raw attention usually violates polarity. Expect: small accuracy gain on ml-1m bags (72 tokens), null on hm (5 purchases), weights pass T2–T3, mixed on T5–T6 | **intrinsic for the decomposition, instrument for the weights**: the weights are printed as model output only after passing the battery; a sparsity/entropy regulariser on w is a legitimate R2 knob (taxonomy row 4 analogue), a "faithfulness loss" is not |
| **named bottleneck** (SP7, feature-word re-weighting) | exact-given-weights *downstream of the bottleneck*; leakage above it | bottleneck **re-weighting divergence** (KL between re-weighted and linear word masses); Mahinpei-style leakage probe (can the bottleneck code predict something the words cannot?); Margeloiu-style semantic check (nearest words of each bottleneck unit stable across seeds, taxonomy row 11) | CBM: +2.4–6.8 error pts on CUB, none on OAI (Koh Table 1); UCBM/CEM narrow the gap by adding capacity — i.e. by leaking (Mahinpei). SENN: explicit stability-vs-accuracy trade via the Lipschitz weight | **intrinsic only with an anchoring loss** (dc04 lesson; CBM concept loss / SENN robustness term). Anchoring = R2 knob; leakage probe = R1 instrument |
| **capped disclosed residual** (SP8) | exact-given-‖q‖ for the named part + one **opaque unit** of bounded mass | residual share (by construction ≤ cap) vs accuracy at each cap — the frontier; Shapley of the residual unit vs the named units to confirm the cap is honoured under the cosine | no direct precedent; closest is ERASER sufficiency (performance retained with rationale only) and Peake & Wang's Model Fidelity as coverage; CBM "no bottleneck"/multitask rows (Koh Table 1) are the uncapped end of the same frontier | **instrument** — a Part-II frontier tracer, not a candidate; R2 knob is the cap itself |
| **Deep Sets / PMA set encoder** (SP5/SP6) | Shapley-with-error (enumerable at \|H\| ≤ 12; sampled above) | baseline declaration (renorm default, ID-only for ids arms); coverage; POS@(10,1), INS, CDCG (G1 naming); IG cross-check with SPINRec-style baseline sensitivity; positive-leakage seam (drop φ(pos) before ρ) | model-agnostic recsys explainers benchmark (Nguyen 2026: effectiveness–sparsity trade varies by method; LXR/SPINRec best on AUC fidelity); Huang & Marques-Silva: φ can misrank — report top-k overlap with LOO, not φ alone. Accuracy: plan expects hm null at \|H\|≈5 | **instrument only** — nothing intrinsic survives; prototype naming degrades to post-hoc top-k (plan §5). Fails taxonomy row 16 by construction; must be declared as the comparison arm, not a candidate |
| **target-conditioned attention** (SP9, NAIS/DIN form) | exact-given-weights *per candidate* (weights depend on i) | battery T1–T7 run per (u,i); plus **explanation-object caveat**: shares not comparable across candidates for the same user; ShaRP-style rank/top-k payoff for any Shapley cross-check; leakage: positive must not attend to itself | Li 2024 (TKDE): attention paths unstable across reproductions in recsys — expect T2 to be the binding test; no recsys study runs the full battery (negative finding §B.1) | **intrinsic decomposition, instrument weights** as for SP2, with the extra disclosure that the "profile" is per-candidate (plan §6) — the only form that escapes P5, and the one whose explanation object changes most |

### E.3 Rules to carry into the design-candidate cycles

1. Every arm reports **SF with a regime tag and ε_A**; the tag is part of the metric name (taxonomy row 13 gets a regime column; row 8 gets the normaliser-disagreement number as its "interaction-term magnitude" entry for the cosine).
2. The **plain-dot twin is the unit test** for every attribution instrument: φ must equal linear shares there (InstaSHAP / Lundberg local accuracy). If it does not, the instrument is wrong, not the model.
3. **Baseline is declared, never defaulted**: renorm (counterfactual-faithful) for the shopper sentence, fixed/ID-only for comparability with regime 1, sampled (SPINRec) only when ρ has no natural zero — and then with the κ-sensitivity plot.
4. **Split supporting from contradicting mass** (Baklanov 2025); the memo shows the contradicting half is ~40–48 % of purchases in every arm.
5. **Attention weights are printed only after the battery** (§B.2); until then they are composition parameters. T1 against the existing linear arm is free and comes first.
6. **Bottlenecks come with an anchor and a leakage probe** or they are not named (Mahinpei, Margeloiu, dc04).
7. **Regime 3 forms are instruments/controls, not candidates** under R2 (taxonomy row 16); they are still worth running because they bound the headroom (plan §0, tuple-ID logic) and because exact enumeration makes their attribution *exact-post-hoc* — a phrase the thesis can use precisely, once, with the coalition counts.
8. Steck 2024 travels with every cosine-bearing row: exactness of the split says nothing about whether the cosine *means* similarity; keep taxonomy row 10 as a separate read-out.

---

## F. Open items / not done this session

- No recsys paper was found that runs the W&P battery on history attention; if one exists it would change §E.2 row 2's "typical trade" from NLP evidence to domain evidence (⚠️ search gap, one query budget).
- Castro 2009 error bounds not re-read (publisher blocks abstract); the estimator's use in the memo is unaffected.
- GA²M's accuracy claim is read from the paper's §6 text ("nearly as accurate as full-complexity random forest"); exact per-dataset RMSE/error deltas (Tables 2–3) were not transcribed.
- CBM cost numbers are Koh Table 1 only (CUB/OAI); no recsys bottleneck accuracy cost exists in the literature read — dc02/dc04's own numbers are the thesis's evidence there.
