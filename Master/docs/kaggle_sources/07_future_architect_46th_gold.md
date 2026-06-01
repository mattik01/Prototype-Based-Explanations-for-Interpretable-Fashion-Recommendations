# Future Architect — 46th place gold (Japanese tech blog post-mortem)

**Source ID:** `07_future_architect_46th_gold`
**Type:** blog (corporate tech blog, Japanese)
**URL:** https://future-architect.github.io/articles/20220602b/
**Author / Team:** Future Architect Inc. engineer (uncredited individually in summary)
**Date posted:** 2022-06-02
**Audience signal:** Public corporate engineering post-mortem
**Final score:** 46/2952 — Gold. Public/private MAP@12 not extracted (Fujikawa cites this team's approach as **+0.007 better than his own** → likely ~0.034 public).
**Thesis tags:** `image-features` `text-features` `repeat-purchase` `temporal-split` `ensemble`

---

## Architecture summary

**Two-stage embedding-retrieval → transformer-ranking** pipeline — the approach Fujikawa explicitly cites as outperforming his own (+0.007 LB gain) and regretted not adopting.

1. **Retrieval:** Item2Vec embedding model that fuses CF signal with content (images, text descriptions, categories). User embeddings = averaged history-item embeddings. Inner-product similarity for fast filtering.
2. **Ranking:** Transformer with attention scoring user history × candidate pairs.

## Features used

- **Item features**: product images, natural-language descriptions, category metadata — fused into the Item2Vec retrieval embedding.
- **User features**: history-derived (averaged item embeddings; no demographic features highlighted).
- **Engineered (user × item)**:
  - Past-purchase indicator (in retrieval candidate set)
  - Sequence attention scores in the ranker
- **Image / text features**: **YES, prominently** — both image and text features feed the Item2Vec retriever. Notable departure from many top solutions that skip image features.

## Feature engineering specifics

- **Item2Vec embeddings** trained on transaction sequences with content-feature fusion.
- **User embedding = mean of historical item embeddings** — simple but effective.
- **Sequence-length-aware attention** in the transformer ranker via `tf.tensor_scatter_nd` partitioning to handle variable-length histories efficiently.

## Candidate generation

Retrieval candidate pool defined by three filters:
1. Items previously purchased by the user
2. Top **20,000** items by interaction volume in past **90 days**
3. Only items with ≥1 purchase in past **week** (removes discontinued products)

True positives averaged rank **32 among 128 retrieved candidates** — challenging signal for the ranker.

## Ranking / model

- **Transformer ranker** with attention.
- **ApproxNDCGLoss customization**: replaced nDCG weighting term with **positive-ranking differences** — adapted TensorFlow Ranking's standard loss for this task.
- **Hard-negative sampling via Gumbel-Max Trick**: sampling difficult negatives from purchase-frequency distribution segmented by age group and prediction date.
- **Negative sample count = 4,096** (vs. typical 5–15 in Word2Vec).

## Validation strategy

- **Training window:** 2018-09-20 → 2020-09-22 (matches Fujikawa, same competition split).
- **Prediction window:** 2020-09-23 → 2020-09-29.
- Hardware: RTX 3090, inference time ~20 min for 1.37M users.

## Stated pitfalls / lessons

- **Negative sample quantity is critical:** 1,024 negatives → LB 0.010; 4,096 negatives → LB **0.020** (2× lift purely from increasing negative count).
- **Neural networks using images/text underperformed expectations** because of repeat-purchase dominance and sporadic user histories. Pure end-to-end DL can't beat human-rule + ML hybrids.
- **History is sporadic** — purchases concentrated in narrow categories with month-scale gaps. Standard transformer pretrain heuristics don't transfer cleanly.

## Notable insights / quotes

> "Negative sample quantity and quality proved critical — 4,096 negatives achieved LB 0.020 vs. 0.010 at 1,024."

> "Repeat purchase behavior dominated — users frequently repurchased identical items or color variations."

> "Effective recommendations require skillfully blending human logic with ML algorithms rather than pursuing pure neural solutions."

## Reported metrics

- MAP@12 public/private: not extracted explicitly. Fujikawa positions this team as **+0.007 over his 0.0339** → ~0.0410 likely.
- Negative-sample ablation: LB 0.010 (n=1024) → LB 0.020 (n=4096), +0.010 absolute.
- No HR@K / NDCG@K reported in summary; ApproxNDCGLoss used as training loss only.

## Code / notebooks

Not linked in the summary blog. May exist on Future Architect's internal repo.

## Thesis relevance

- **Phase 4.3 (architecture):** **The clean two-stage neural pipeline (Item2Vec retrieval → Transformer ranking) is the most architecturally aligned to our prototype-based thinking** of all sources extracted. Item embeddings + sequence attention map naturally to ProtoMF's prototype-distance framing.
- **Phase 4.0 (feature ROI):** image + text features used here in *retrieval* (not ranking), suggesting their value is in widening the candidate frontier rather than discriminating between close candidates. Strong design hint for feature-aware prototypes.
- **Phase 4.4 (negative sampling):** **Direct evidence** that ProtoMF's negative-sample count + difficulty matters enormously (+0.010 absolute from 4× more negatives). Our negative sampling strategy in feature-aware variants should be ablated carefully.
- **Phase 4.5 (explanations):** sequence-attention scores from the ranker are inherently interpretable — "this candidate was scored highly because it attended to your purchase X 5 weeks ago." That's a direct analog to ProtoMF's prototype-similarity weights.
- **Citable to thesis Ch. 6:** the negative-sample ablation table is unusually clean — good supporting evidence for the "hard-negative sampling matters more than feature engineering" line of argument.
