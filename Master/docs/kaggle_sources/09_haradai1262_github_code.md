# haradai1262 — GitHub solo solution (code-only, no writeup)

**Source ID:** `09_haradai1262_github_code`
**Type:** writeup (GitHub repo — code only, no README)
**URL:** https://github.com/haradai1262/h-and-m-personalized-fashion-recommendations
**Author / Team:** haradai1262 (solo)
**Date posted:** 2022
**Audience signal:** Public GitHub repo (2 stars), no README — extracted directly from source code
**Final score:** Rank not stated in repo. Files suggest serious effort: 256KB notebook `exp120.ipynb` + 1576-line CatBoost feature engineering script.
**Thesis tags:** `repeat-purchase` `temporal-split` `cf-only` `feature-importance` `ensemble`

---

## Architecture summary

Two-stage pipeline. Candidate generation feeds **CatBoostRanker** trained with **YetiRank loss** on `MAP:top=12` eval metric. GPU-accelerated (cudf, CatBoost GPU). Sliding-window training across 10 weeks with 4-week stride.

## Features used (directly from `cat_v4-3.py`)

- **Item features (raw)**:
  - `product_code`, `product_type_no`, `product_group_name`, `graphical_appearance_no`, `colour_group_code`, `perceived_colour_value_id`, `perceived_colour_master_id`, `department_no`, `index_code`, `index_group_no`, `section_no`, `garment_group_no`
- **User features (raw)**:
  - `age`, `age_id` (bucketed), `FN`, `Active`, `club_member_status` (binarized), `fashion_news_frequency` (binarized), `postal_code` (rare codes collapsed to 0), `postal_code_ce` (count encoding)
- **Image / text features**: **NOT used** — pure structured/tabular features.

## Feature engineering specifics (~80+ engineered features)

**Temporal windows** (multi-scale aggregations across all features):
- `yesterday` = 1 day
- `recent` = 1 week (7 days)
- `hist_short` = 30 days
- `hist_mid` = 90 days
- `hist` = 366 days

**Per-item temporal counts:**
- `art_buy_{hist,hist_short,hist_mid,recent,yesterday}` — purchase count per article at each window
- `art_days_after_buy`, `art_days_from_oldest_buy`, `art_days_from_mode_buy` — recency stats
- `art_rate_sales_channel_{hist,recent}` — channel mix per item

**Per-item × channel:** `art_buy_hist_ch1`, `art_buy_hist_ch2` — separate counts per sales channel (online vs in-store).

**Re-purchase rate** (custom feature, smoothed): `rebuy_rate(article_id) = count_of_users_buying_2x / (count_of_users_buying_1x + sm=5)`. Computed at article level (`rebuy_rate`), product-code level (`code_rebuy_rate`), and per-user level (`cust_rebuy_rate`).

**Price aggregations** (per article, per user, per age-group):
- mean / median / max / min over `hist` window

**Age-buyer aggregations** (per article):
- `art_age_hist_{mean,median,max,min}` — typical buyer's age distribution per article

**User × item count features (sparse):**
- `n_buy_{hist,hist_short,hist_mid,recent}` — how often this user bought this article
- `days_after_buy` — days since last user-article transaction

**User × item-attribute features:**
- `n_buy_hist_ptype`, `n_buy_hist_graph`, `n_buy_hist_col`, `n_buy_hist_dep`, `n_buy_hist_idx`, `n_buy_hist_idxg`, `n_buy_hist_sec`
  - Times user bought each `product_type / graphical / colour / department / index / index_group / section`
  - Each has a matching `_recent` (1 week) and `days_after_buy_{ptype, graph, ...}` companion
- `n_buy_hist_prod` — at the `product_code` level (variant-agnostic)

**Age-group × item features:**
- `age_id_n_buy_{hist,recent}` — typical demographic affinity per item
- `age_id_n_buy_hist_all` — total purchases per age bucket
- `age_id_price_hist_{mean,median,max,min}` — typical prices for age group

**Postal-code encoding trick:** postal codes with ≤10 transactions are collapsed into single bucket "0"; common codes get integer-encoded.

## Candidate generation

Hyperparameters in code suggest:
- `N = 15000` total candidate space size
- `N_div = 300` divisions (likely batch processing)
- `tmp_top = 200` short-list size per user
- Filter on `len_hist = 366` days of history

Exact candidate channels (popularity / re-purchase / ItemCF) need notebook-level inspection — script only handles feature gen.

## Ranking / model

- **CatBoostRanker** with:
  - Loss: `YetiRank` (listwise ranking objective)
  - Eval metric: `MAP:top=12`
  - Depth 7, LR 0.05, Bernoulli bootstrap (subsample 0.69)
  - L2 reg ~7e-4
  - Up to 10,000 iterations with `od_type=Iter`, `od_wait=30` (early stopping)
  - GPU training (`task_type=GPU`, `gpu_ram_part=0.95`)
  - Random seed 46
- **`label_time_decay = 0.05`** — time-decayed labels (recent positives weighted higher)
- Training: `n_week=10`, sliding window step `slide_week=4` — many overlapping training windows

## Validation strategy

- `day_start_val = 2020-09-23` (competition validation week).
- `mode = "cv"` flag suggests CV evaluation against this week.

## Stated pitfalls / lessons

No README, no written commentary. Code patterns suggest:
- **Multi-scale temporal aggregation is non-negotiable** — same feature computed at 5 windows.
- **Cross-attribute user×item features (color/department/section)** are first-class features, not engineered as an afterthought.
- **Re-purchase rate at multiple granularities** (article / product_code / customer) is treated as a key signal.

## Notable insights / quotes

No quotes — code-only source. The implicit architecture statement: **a single, heavily-engineered CatBoostRanker with multi-scale temporal features + cross-attribute user×item counts can be a competitive top-tier ranker.**

## Reported metrics

- MAP@12 public/private: **not stated** anywhere visible. Repo lacks README; final rank unknown.
- Eval metric configured in CatBoost: `MAP:top=12`.

## Code / notebooks

- `exp/cat_v4-3.py` (1576 lines) — full feature pipeline + CatBoost training
- `exp/stacking_nb_6.ipynb` — likely the final ensemble / stacking step
- `exp/exp120.ipynb` (256KB) — experiment notebook

## Thesis relevance

- **Phase 4.0 / 4.3 (feature ROI):** **most explicit feature taxonomy** of any source extracted — gives us a concrete "what the kaggle field considers core features" list:
  - 12 raw item attributes from `articles.csv`
  - 8 user attributes from `customers.csv`
  - ~80+ engineered features dominated by **temporal counts** and **cross-attribute user×item counts**
- **Direct blueprint for thesis Phase 4 feature engineering:** if our feature-aware prototype loses to plain ProtoMF, this list shows what *minimum* feature engineering would be expected of a serious H&M benchmark.
- **The multi-scale temporal window pattern** (1d / 1wk / 30d / 90d / 1yr) is recurring across sources (also in fujikawa, future_architect, FLAIRS). Strong candidate for *built-in inductive bias* of feature-aware prototypes.
- **YetiRank + MAP@12 eval metric in CatBoost** is the practical training loss choice — relevant if we extend our trainer beyond BCE/BPR/sampled_softmax in Phase 4.
- **Re-purchase rate smoothing formula (`buy2 / (buy1 + sm)`)** is a clean prior-like feature design — applicable beyond H&M.
