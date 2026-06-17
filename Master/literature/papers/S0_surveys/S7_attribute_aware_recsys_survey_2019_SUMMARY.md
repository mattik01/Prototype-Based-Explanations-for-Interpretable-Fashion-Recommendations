# S7 — Chen et al. 2020, "Attribute-Aware Recommender System Based on Collaborative Filtering: Survey and Classification"

Mini-summary, thesis-relevant only. Survey (Frontiers in Big Data, Vol 2, Art 49). PDF in this folder (annotated copy: `*-annotated.pdf`). Full model catalog + classification reproduced in `S7_attribute_aware_recsys_survey_2019_TABLES.md`. Insights in `protocol_vault/`.

## One line
A survey that classifies ~103 attribute-aware CF models into four mathematical families — discriminative MF, generative MF, generalized factorization, heterogeneous graphs — and compares six of them empirically (RMSE).

## Why it matters to us
- It is the **map of "how to put features into a CF model,"** which is exactly the design space of the feature-aware ProtoMF extension. We use its taxonomy for *positioning* and its findings for *motivation* — see [[2026-06-16_1858_thesis-intro-storyline-feature-motivation]] and [[2026-06-16_1807_attribute-source-positioning]].
- Its **four-category taxonomy is the map of the design space**; on our assessment the promising region for prototypes is **DMF (§4.1, esp. similarity/bilinear)**, while GMF (empirically weak), GF (mainly useful as baselines), and HG (paradigm-mismatched + doesn't scale on H&M) are back-shelf for now — see [[2026-06-16_1838_fm-gmf-suitability-hybrid-bridge]].

## Most citable for (★ = strongest)
- ★ **Attribute-aware models beat their plain counterparts** (§5.3: FM > MF, NCF+ > NCF) — the motivation that adding features should also help accuracy. (RMSE evidence — cite directionally.)
- ★ **Four-category taxonomy** DMF / GMF / GF / HG (§4, highlighted p2 & p11) — the canonical way to classify attribute integration.
- **AUC penalizes all rank positions equally** → position-aware metrics (NDCG/MRR/MAP) better for top-N (§3.5.2) — justifies reporting hit_ratio@10 / NDCG and *not* RMSE/AUC. See [[2026-06-16_1704_eval-design-citable-justification]].
- **Attribute sources are interconvertible** (user/item/rating; §3.2.3–3.2.4, Eqs 11–12) and **user vs item need not be distinguished** (MF symmetry, §3.2.1, highlighted p7) — license for source-agnostic feature integration.
- **DMF & GF > GMF**, and **user attributes most beneficial** (§5.3) — both with caveats below.

## Coverage map (relevance-flagged)
§2 Preliminaries · **§3.2 sources** · **§3.3 types** · **§3.4 rating types** · **§3.5 goals (ranking vs prediction)** · **§4.1 DMF (linear/bilinear/similarity)** · §4.2 GMF · **§4.3 GF (FM/TF)** · §4.4 HG · **§5.3 findings** · model catalog + classification → `…_TABLES.md`.

## Our insights from reading it (protocol links)
- ProtoMF is an **item-ranking base case**; the survey is rating-prediction — [[2026-06-16_1631_protomf-item-ranking-basecase]]
- **Attribute-source positioning** (item-first, user candidate-dependent, context dropped) — [[2026-06-16_1807_attribute-source-positioning]]
- **Intrinsic > post-hoc** interpretability principle (prototype grounding currently post-hoc) — [[2026-06-16_1808_intrinsic-over-posthoc-principle]]
- **Feature-injection fork** (add / interact / regularize) — [[2026-06-16_1715_feature-injection-design-fork]]
- **FM/NFM/DeepFM/NCF+ as baselines** — [[2026-06-16_1824_feature-aware-baselines]]
- **FM/GMF not a core mechanism; hybrid needs a grounding bridge** — [[2026-06-16_1838_fm-gmf-suitability-hybrid-bridge]]
- **FM×prototype = richer explainable scoring (Axis-B experiment idea, not feature grounding)** — [[2026-06-16_1837_fmxprototype-richer-interaction]]

## Caveats when citing
- **Rating-prediction / RMSE survey.** Its taxonomy and all experiments live on the pointwise side (§3.5.1); ProtoMF is item-ranking. Cite for the *map* and for *directional* motivation, not as proof in our regime.
- **"User attributes most beneficial"** is RMSE-based on datasets with rich user demographics; does not transfer cleanly to H&M (thin user features). Reconciliation: user signal lives in CF; item features add grounding — see [[2026-06-16_1858_thesis-intro-storyline-feature-motivation]].
- **Attribute-aware ≠ hybrid** (the paper's own caveat, highlighted p2): it covers attributes-as-side-information, not content+CF hybrids.
