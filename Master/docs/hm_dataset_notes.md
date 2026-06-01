# H&M Dataset Notes

Synthesis of findings from the H&M Personalized Fashion Recommendations Kaggle competition (Feb–May 2022) — discussions, top solutions, public notebooks, post-mortems, external write-ups.

Per-source notes live in [`kaggle_sources/`](kaggle_sources/) (one .md per substantial source).
Source index: [`kaggle_sources/_register.csv`](kaggle_sources/_register.csv).

Source IDs referenced inline below: [01] aji_samudra (89th), [02] wp_zhang (45th), [03] fujikawa (12th gold), [04] jingsailian (Chinese summary of 1st/3rd), [05] mike_chesnokov (142nd), [06] altimetrik (top 1.6% claimed), [07] future_architect (46th gold), [08] FLAIRS paper (43rd ≈ top 2%), [09] haradai1262 (code-only), [10] kuto5046 (23rd silver).

---

## 1. Approach Taxonomy

**Dominant architecture (≥9/10 sources): two-stage retrieval → GBDT reranker.**

- **Two-stage CF retrieval + LightGBM reranker** — `[01] [02] [05] [06] [08] [10]` and inferred for 1st place `[04]`. The default top-solution recipe.
- **Two-stage CF retrieval + CatBoost reranker (GPU, YetiRank loss)** — `[09]` and reportedly **1st place** `[10]`. Outperformed LightGBM at the top.
- **Two-stage embedding retrieval + Transformer reranker** — `[07]` (46th gold). Cited by `[03]` as +0.007 LB gain — the architectural regret of the 12th-place team.
- **Two-tower NN with activity-gating** — 4th place (cited only via `[10]`). Outlier architecture; gating compensates for heavy-tailed customer activity.
- **NN→GBDT stacking** — `[03]` (12th). NN predictions fed into GBDT as features, plus rank-averaging with Optuna-tuned weights.
- **Ensemble of multiple CF strategies → ensemble of ranker models** — `[02]` (2 strategies × 3 rankers), `[05]` (4 models equal-weight), `[10]` (6 models with rank-position weighting).

**Rare or absent:**
- Pure end-to-end deep learning (without GBDT post-ranking) — explicitly *underperformed* per `[07]`.
- Pure prototype-based or anchor-based methods — **not found in any extracted source**. Direct opening for thesis contribution.
- Sequence-only transformers (no retrieval-first stage) — not seen at competitive ranks.

---

## 2. Feature Consensus

### Item features (`articles.csv`) — what gets used

| Feature | Sources using it | Notes |
|---|---|---|
| `product_code` | `[03] [09]` | Variant-agnostic re-purchase signal |
| `product_type_no` | `[01] [09] [10]` | Universally used |
| `product_group_name` | `[09]` | Often label-encoded |
| `graphical_appearance_no` | `[09]` | Pattern/print attribute |
| `colour_group_code` | `[01] [09]` | **Critical — feeds "other colors" candidate channel** |
| `perceived_colour_value_id`, `perceived_colour_master_id` | `[09]` | Lighter/darker tone signals |
| `department_no` | `[01] [09]` | Universally used |
| `index_code`, `index_group_no`, `section_no` | `[09]` | Higher-level taxonomy |
| `garment_group_no` | `[09] [10]` | Higher-level garment category |
| price | `[01] [06] [09] [10]` | Multi-stat aggregations (mean/median/max/min) |

### User features (`customers.csv`)

| Feature | Sources | Notes |
|---|---|---|
| `age` | `[01] [05] [06] [09] [10]` | Universal. Often bucketed into `age_id`. |
| `postal_code` | `[06] [09] [10]` | **2nd place insight**: postal + age popularity grouping `[03]`. Rare codes collapsed in `[09]`. |
| `club_member_status` | `[09]` | Binarized |
| `fashion_news_frequency` | `[09]` | Binarized |
| `FN`, `Active` | `[09]` | NaN-filled to 0 |

### Derived / engineered features (consensus)

- **Multi-scale temporal counts (≥4 sources):** purchases/clicks aggregated at 1-day, 1-week, 30-day, 90-day, 366-day windows — `[03] [07] [08] [09]`. **This is the single most consistent feature-engineering pattern.**
- **Time decay on interactions:** power-law decay wins `[08]`; time-weighted counts `[05]`; `label_time_decay=0.05` in `[09]`. Linear decay also tried `[01]`.
- **Re-purchase rate (smoothed):** `buy2 / (buy1 + 5)` at article + product_code + customer level `[09]`. Similar repeat-purchase signal universal.
- **Cross-attribute user × item-attribute counts:** `n_buy_hist_{ptype, graph, col, dep, idx, idxg, sec}` — `[09]` (most explicit), echoed in `[01] [03]`.
- **Recall-channel similarity scores as ranker features:** **+0.00407 CV gain on 1st-place ablation** `[04]`. Critical recurring lesson.

### Image / text features

| Used? | Sources | Modality |
|---|---|---|
| **Image** (EfficientNet, transformer) | `[02] [03] [06] [07]` | EfficientNet + PCA gave **only +0.0007 CV** `[03]` — small. Multi-source `[06]`. |
| **Text** (Word2Vec, BERT, descriptions) | `[02] [03] [07] [10]` | BERT + UMAP-6D `[10]`. Word2Vec `[02] [03]`. 6th place: TF-IDF+SVD on item text → *user embedding* `[10]`. |
| **Neither (structured only)** | `[01] [05] [08] [09]` + 1st place per `[04]` | Including the **1st place winner** and the FLAIRS paper. |

**Headline finding:** image + text features are **adopted but rarely the headline gain**. The winning solution `[04]` (and FLAIRS `[08]`) used neither.

### Graph embeddings (cross-cutting)

- ProNE (1st place per `[10]`), DeepWalk + Node2Vec + LINE + ProNE (1st per `[04]`), DeepWalk (`[03]`), LightGCN via RecBole (`[10]`), Item2Vec (`[07]`). **At least 5 sources use graph embeddings.** Strong recurring CF-augmentation pattern.

→ Direct feeds for Phase 4.0 (H&M Feature Analysis) and Phase 4.3 (feature-aware architecture design).

---

## 3. Candidate Generation Playbook

Frequency of candidate-generation channels across the 10 sources:

| Channel | Sources | Notes |
|---|---|---|
| **Repeat purchases (user's own history)** | `[01] [03] [04] [05] [06] [07] [09] [10]` | **9/10 sources — universal.** Single strongest signal in fashion. |
| **Popularity top-N (global, last week/month)** | `[01] [03] [04] [05] [06] [10]` | Often 600–1000 items. 2nd-place strategy `[04]`. |
| **Item-item CF (cosine / TF-IDF / RP3β)** | `[04] [05] [08] [10]` | Cosine vs RP3β: RP3β +0.0015 MAP@12 in `[08]`. ItemCF in `[04]`. |
| **"Other colors" / product variants** | `[03] [04] [10]` | **Highly interpretable** — directly maps to prototype explanations. |
| **Demographic / age cold-start popular** | `[01] [05] [06] [10]` | Age + postal-code segmentation for new users. |
| **Graph embeddings (DeepWalk/Node2Vec/ProNE/LightGCN/Item2Vec)** | `[03] [04] [07] [10]` | Plus 1st place per `[10]`. |
| **User-user KNN (TF-IDF on item history)** | `[05]` | 100 candidates/user. |
| **Two-tower / DSSM neural retrieval** | `[01] (TF-Recommenders)` `[02] (DSSM)` | Soft adoption. |
| **Logistic regression coarse-ranking from top-1k** | `[04]` (1st place) | Pre-rank step before LightGBM. |
| **Whitelisting alive items (last 30 days)** | `[07] [08]` | **20% of catalogue → 90% coverage** of holdout transactions `[08]`. |
| **Section-ID popular items (user's prior sections)** | 2nd place `[04]` | Section-conditioned popularity. |

Typical candidate pool sizes: 30 `[01]`, ~50 `[02]`, 100–300 `[10]`, 130 `[04]` (2nd), 200 short-list `[09]`, 600 `[06]`, top-1k `[04]` (1st).

---

## 4. Pitfalls Consensus

Recurring lessons explicitly flagged by top solutions:

1. **"Retrieve → Ranking insufficient recall" is the biggest failure mode** — `[03]` flagged as their biggest weakness. Confirmed by 1st place quote: *"candidate strategy is key, feature engineering contribution limited"* (`[03] [10]`).
2. **Add recall-channel similarity scores as ranker features, not just as a candidate filter.** Without this, 1st place CV would have been 0.02855 instead of 0.03262 (`[04]`).
3. **Single-fold validation is risky** but `[01]` reports last-7-days holdout *"still gave good correlation with leaderboard"*. Multi-fold weekly CV used by `[03]`. Holdout-with-leak-prevention in `[10]`. Standard split = train ≤ week W-1, validate on week W.
4. **Dedup transactions on (customer, article, date)** — `[09]` does this explicitly. Implicit elsewhere.
5. **Cold-start is small but non-zero (0.7% users, 0.9% items per `[10]`).** Rule-based age-based popularity post-processing gives minor lift.
6. **Hardware constraints force compromises** — `[01]` skipped images (50GB RAM); `[02]` capped at 4 weeks training. Top solutions often used GPU + cudf `[09]` or RTX 3090 `[07]`.
7. **Fashion items have lives** — only ~35k of 106k items "alive" at any time `[08]`. Recommending obsolete items wastes recall budget.
8. **Pure end-to-end DL underperforms hybrid logic** — `[07]` quote: *"Effective recommendations require skillfully blending human logic with ML algorithms rather than pursuing pure neural solutions."*
9. **Negative sample quantity matters enormously** — 1024 → 4096 negatives doubled LB from 0.010 to 0.020 in `[07]`.
10. **Categorical features in LightGBM** (`department_no`, `product_type_no` declared as `categorical_feature`) lifted CV by 0.0005–0.0008 (2nd place via earlier search).

---

## 5. Metric Landscape

### Competition metric
MAP@12 (Mean Average Precision at 12). The official leaderboard metric. Public/private 50/50 split.

### MAP@12 distribution across leaderboard

| Rank tier | MAP@12 public | MAP@12 private | Source |
|---|---|---|---|
| **1st** (senkin13 + 30CrMnSiA) | 0.0372 | **0.0379** | `[03] [04]` |
| **2nd** (wht1996 / paweljankiewicz) | 0.0368 | 0.0374 | Per search-result snippets |
| **3rd** | — | — | BPR-based, +0.0015 LB lift quoted `[04]` |
| **12th** gold (Fujikawa) | 0.0339 | n/a | `[03]` |
| **23rd** silver (kuto5046) | LB14 → priv23 | — | `[10]` |
| **43rd** (FLAIRS/D'Amico) | 0.0343 (best CV) | — | `[08]` |
| **46th** gold (Future Architect) | ~0.0410 est. | — | `[07]` (inferred from +0.007 over `[03]`) |
| **45th** silver (wp_zhang) | 0.0292 | 0.02996 | `[02]` |
| **89th** silver (aji_samudra) | not reported | not reported | `[01]` |
| **142nd** bronze (mike_chesnokov) | 0.02530 | 0.02554 | `[05]` |

### CF-only ceiling

Mike Chesnokov `[05]` (142nd bronze) used CF + LightGBM, no images/graph embeddings, reached **LB 0.02530**. Indicates the "pure structured CF" ceiling lies near LB 0.025–0.026 — top-5% is reachable with this alone.

### Pop-only baseline

cdeotte baseline (cited via `[03]`): **CV 0.0264, LB 0.021** — a strong popularity-based baseline. Top solutions get to ~0.038 (about +0.017 above pure-popularity).

### Cross-metric reports

- **MAP@12, MAP@100, MAP@200, Recall@12, Recall@100, Recall@200** reported in `[08]` (FLAIRS) — only source with multiple cutoffs.
- **No HR@K / NDCG@K reported in any source.** Competition standard was MAP@12. **Gap to our replication report** (which uses HR@{1,3,5,10,50} + NDCG@{1,3,5,10,50}): we must report MAP@12 alongside HR/NDCG to compare against Kaggle solutions.
- **No Recall@K besides FLAIRS, no precision@K, no MRR.**

---

## 6. Thesis-Specific Q&A

### 6.1 Cold-start handling

Cold-start was **small (0.7% users, 0.9% items)** but every competitive solution treats it as a separate retrieval lane:
- **Demographic conditioning** (age + postal_code): `[01] [05] [06] [10]` + 2nd place `[03]`.
- **Last-week popular for new users**: `[01] [05] [10]`.
- **Rule-based post-processing for cold users only** `[10]`.

**Thesis angle:** prototype-based methods could absorb cold-start *inside* the model (assign new users/items to prototypes via demographic features) — none of the extracted competitors did this; it stays a clean differentiation point.

### 6.2 Prototype-like or explainable methods

**No top solution used prototype-based, anchor-based, or otherwise explicitly interpretable architectures.** Closest analogues:
- **"Other colors" + "Re-order" candidate channels** `[03] [04]` — interpretable *retrieval rationale*, not interpretable ranking.
- **LightGCN graph embeddings** `[10]` — has *implicit* interpretability via neighborhood paths.
- **6th place TF-IDF+SVD on item text → user embedding** `[10]` — user representation is interpretable as text-content centroid.

**Thesis position is clean:** explicit prototype-based explanations are *missing* from the H&M competition landscape. Our work fills a gap, not a crowded space.

### 6.3 CF-only ceiling

- Pure CF + LightGBM (no images, no graph): **LB ≈ 0.0253** (`[05]` 142nd bronze).
- CF + temporal modifications + LightGBM (FLAIRS WI+TW+Reranker): **MAP@12 0.0343** (`[08]` 43rd).
- The gap from pure popularity (LB 0.021 per `[03]`) to top-1 (LB 0.0372): **+0.016 absolute** is the total room for ranker engineering.
- A clean prototype-based ProtoMF baseline on H&M should target the LB 0.025–0.030 band to be competitive.

### 6.4 Feature ROI ranking (most thesis-citable ablations)

From Fujikawa's `[03]` quantified ladder (CV deltas):
| Addition | CV gain |
|---|---|
| Metadata embeddings introduced | +0.0044 (largest single lift) |
| Top2-5 similar-item features | +0.0018 |
| Likelihood dropout | +0.0008 |
| DiceLoss scaling | +0.0021 |
| Image (EfficientNet+PCA) + graph (DeepWalk) embeddings | +0.0007 (smallest) |

From `[04]` (1st place): recall-channel features as ranker features: **+0.00407 CV (0.02855 → 0.03262)** — biggest single lift documented for any 1st-place solution.

From `[07]` (46th): 1024→4096 negatives: **LB 0.010 → 0.020, +0.010 absolute (2× lift)**.

From `[08]` (FLAIRS): Cosine+R → Cosine+WI+TW+R: **MAP@12 +54.7% relative** (0.0212 → 0.0328).

**Take-away for Phase 4.0:** structured metadata embeddings (+0.0044) and recall-channel features (+0.00407) dwarf image features (+0.0007). Prioritize tabular/structured prototype features over image prototypes.

### 6.5 Temporal / seasonality patterns

- **5 windows are canonical:** 1d / 1w / 30d / 90d / 1y (`[03] [07] [08] [09]`).
- **Power-law time decay wins** over linear / exponential / constant (`[08]`).
- **Fashion items have lives:** k-means on sales profiles gives 2 clusters — *seasonal* (sales→0 in 4 months) and *long-lived* (`[08]`).
- **Whitelist last-30-days alive items at prediction:** 20% of catalog → 90% transaction coverage (`[08]`).
- **Repeat-purchase rate dominates** all temporal patterns; competition was 2 years (2018-09-20 → 2020-09-22), validation week 2020-09-23.

**Implication for hm_3_month vs hm_full:** the 30-day whitelist effect suggests a 3-month dataset already captures most of the "alive" item structure. Going to full 2 years mostly adds dead/seasonal items.

### 6.6 Image / text feature usage

- **Image used by 4/10 sources** `[02] [03] [06] [07]`. **+0.0007 CV** in cleanest ablation `[03]`. Small but non-zero.
- **Text used by 4/10 sources** `[02] [03] [07] [10]`. BERT+UMAP `[10]` and Word2Vec `[02] [03]` are the recipes.
- **1st place used neither image nor text directly** (graph embeddings + structured features only).
- **Effort vs. payoff is bad for images, OK for text** when text is heavily compressed (UMAP-6D, SVD).

### 6.7 Ensemble vs. single-model

- **2-model ensemble** at minimum (`[02]`); 4-model `[05]`; 6-model `[10]`; 3 GBDT-groups + NN + Optuna-weighted rank-averaging `[03]`.
- **Single-model competitive ceiling:** Fujikawa's NN alone reached LB 0.0298–0.0303 (`[03]`); GBDT alone reached LB 0.0324. Final ensemble: LB 0.0339. **Single-model ceiling ≈ 0.0324; ensemble lift ≈ +0.0015.**

**Implication for our thesis:** a strong single-model ProtoMF result around LB 0.030 would be defensible — most of the field's headroom above that comes from ensembling, which is orthogonal to our prototype contribution.

---

## 7. Citable Quotes & Numbers for Thesis

### Quotes

> **"candidate strategy is key, feature engineering contribution limited"** — 1st place (senkin13/30CrMnSiA), cited via Fujikawa `[03]` and kuto5046 `[10]`. **Directly challenges feature-aware thesis premise — must be addressed head-on in introduction.**

> *"In addition, if I only increase the recall num and don't adding the recall features, the CV score is very very poor"* — 1st place via Chinese summary `[04]`. Recall-feature ablation **0.02855 → 0.03262 (+0.00407)**.

> *"Negative sample quantity and quality proved critical — 4,096 negatives achieved LB 0.020 vs. 0.010 at 1,024."* — Future Architect 46th `[07]`.

> *"Repeat purchase behavior dominated — users frequently repurchased identical items or color variations."* — Future Architect 46th `[07]`.

> *"Effective recommendations require skillfully blending human logic with ML algorithms rather than pursuing pure neural solutions."* — Future Architect 46th `[07]`.

> *"Going back 30 days from the start of a given holdout week and selecting the whitelisted items as the ones transacted within this timeframe leads to 90% coverage over the items transacted in the holdout week."* — FLAIRS `[08]`.

> *"Only 20% of the total items in the catalogue are responsible for 90% of the transactions in the holdout week."* — FLAIRS `[08]`.

> *"It's important to have good user-item feature to get well performing model"* — aji_samudra `[01]`.

> *"Candidates from the two strategies are quite different so that ensembling the ranking results can help to improve the score."* — wp_zhang `[02]`.

### Numbers

| Quantity | Value | Source |
|---|---|---|
| Dataset transactions | 31.8 M | `[08]` |
| Unique customers | 1.3 M | `[08]` |
| Unique articles | 106 K | `[08]` |
| Item features available | 24 | `[08]` |
| User features available | 6 | `[08]` |
| Context features available | 3 | `[08]` |
| Training period | 2018-09-20 → 2020-09-22 | `[07] [08]` |
| Validation week (canonical) | 2020-09-23 → 2020-09-29 | `[03] [07] [08]` |
| "Alive" items at any time | ~35K (of 106K) | `[08]` |
| Items responsible for 90% of holdout transactions | 20% | `[08]` |
| Cold-start users | 0.7% | `[10]` |
| Cold-start items | 0.9% | `[10]` |
| Total competition teams | 2,952–3,006 | various |
| 1st place private MAP@12 | 0.0379 | `[03] [04]` |
| 12th place public MAP@12 | 0.0339 | `[03]` |
| 43rd place MAP@12 (FLAIRS) | 0.0343 | `[08]` |
| 45th place private MAP@12 | 0.02996 | `[02]` |
| 142nd place private MAP@12 | 0.02554 | `[05]` |
| Popularity baseline (cdeotte) LB | 0.021 | `[03]` |
| Recall-feature ablation gain (1st place) | +0.00407 CV | `[04]` |
| BPR-similarity feature gain (3rd place) | +0.00147 CV | `[04]` |
| Image+graph embedding gain (12th place) | +0.0007 CV | `[03]` |
| Metadata embedding gain (12th place) | +0.0044 CV (largest single) | `[03]` |
| FLAIRS full-pipeline gain over plain CF | +54.71% relative MAP@12 | `[08]` |
| Hard-negative count lift (46th place) | LB 0.010 → 0.020 (1024→4096 neg) | `[07]` |
