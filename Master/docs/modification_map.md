# Modification Map — Code Extension Points

> Living document mapping every file and line that needs modification for Phases 3–4.
> Created during Phase 2.4. Updated as understanding deepens.

---

## Table of Contents

1. [Adding a New Dataset (Phase 3)](#a-adding-a-new-dataset-phase-3)
2. [Adding a New Model / Feature Extractor (Phase 4)](#b-adding-a-new-model--feature-extractor-phase-4)
3. [Extending Explanations (Phase 4)](#c-extending-explanations-phase-4)
4. [File-by-File Reference](#d-file-by-file-reference)

---

## A. Adding a New Dataset (Phase 3)

Adding H&M requires **3 changes** to existing files and **1 new file**. The data pipeline is intentionally dataset-agnostic — `ProtoRecDataset` reads from a folder without caring which dataset produced the files.

### A1. New file: `data/hm/hm_splitter.py`

Follow the pattern from `data/ml-1m/movielens_splitter.py`. Both existing splitters share this pipeline:

```
Load CSV → Filter positive interactions → [Dedup if needed] →
K-core filter → Re-index user/item IDs (0-indexed) →
Temporal leave-1-out split → Save 5 CSVs
```

**Required output files** (names are hardcoded in `protomf_dataset.py:68-74`):
- `listening_history_train.csv` — columns: `user_id`, `item_id`
- `listening_history_val.csv` — columns: `user_id`, `item_id`
- `listening_history_test.csv` — columns: `user_id`, `item_id`
- `user_ids.csv` — columns: `user_id`, original ID
- `item_ids.csv` — columns: `item_id`, original ID

**H&M-specific additions:**
- Input is `transactions_train.csv` (implicit feedback — purchases, not ratings, so no rating threshold needed)
- Deduplication: keep first purchase per (user, item) pair
- K-core: 10-core (per masterplan design; higher than paper's 5-core due to H&M scale)
- Temporal window: last 6 months (configurable)
- Extra output: `item_features.csv` — joined from `articles.csv`, mapping `item_id` → categorical feature columns. Saved alongside splits, consumed in Phase 4 by `FeatureEncoder`.

### A2. `start.py` — line 16

```python
# Current (line 16):
choices=['amazon2014', 'ml-1m', 'lfm2b-1mon']

# Change to:
choices=['amazon2014', 'ml-1m', 'lfm2b-1mon', 'hm']
```

### A3. `experiment_helper.py` — lines 92–95

The ASHA scheduler is currently only enabled for `lfm2b-1mon` (the largest paper dataset). H&M will be similarly large and needs ASHA for efficient early stopping of bad trials.

```python
# Current (lines 92-95):
if dataset == 'lfm2b-1mon':
    scheduler = ASHAScheduler(grace_period=4)
else:
    scheduler = None

# Change to:
if dataset in ('lfm2b-1mon', 'hm'):
    scheduler = ASHAScheduler(grace_period=4)
else:
    scheduler = None
```

### A4. No changes needed in:

| File | Why not |
|------|---------|
| `rec_sys/protomf_dataset.py` | Dataset-agnostic. Reads any folder with the 5 expected CSVs. Builds sparse matrices from `user_id`/`item_id` columns. |
| `utilities/consts.py` | `DATA_PATH` is the parent; dataset subfolder is appended dynamically in `experiment_helper.py:106`. |
| `confs/hyper_params.py` | Search spaces are model-specific, not dataset-specific. Same configs work for any dataset. |
| `rec_sys/trainer.py` | Receives data loaders; doesn't know which dataset produced them. |
| `rec_sys/tester.py` | Same — dataset-agnostic. |

---

## B. Adding a New Model / Feature Extractor (Phase 4)

Adding feature-aware prototypes requires changes across **5 existing files** and **new classes** in 1 existing file. The architecture follows a clean factory pattern: configs describe what to build, the factory builds it, `RecSys` combines it.

### B1. `feature_extraction/feature_extractors.py` — Add new classes

**Where:** After `ConcatenateFeatureExtractors` (line ~371).

**New class: `FeatureEncoder(nn.Module)`**
- Purpose: Encode categorical item features into a dense vector.
- Input: `item_idxs` (tensor of item indices)
- Internal: `self.feature_table` (non-learnable `nn.Parameter` or registered buffer) — loaded from `item_features.csv` at construction time. Per-feature `nn.Embedding` layers for each categorical column. MLP to combine.
- Output: `feature_embedding` of shape `[..., feature_dim]`
- No regularization loss (or optional L2).

**New class: `FeatureAwarePrototypeEmbedding(FeatureExtractor)`**
- Purpose: Prototypes that operate on combined CF + feature representations.
- Composition: wraps a `PrototypeEmbedding` (CF branch) and a `FeatureEncoder` + separate prototype set (feature branch).
- Forward: compute CF prototypes and feature prototypes independently, concatenate similarity vectors.
- `get_and_reset_loss()`: aggregate regularization from both prototype sets.

**Key design constraint:** The `forward(o_idxs)` signature must stay unchanged — takes only entity indices, returns embedding tensor. This is enforced by `RecSys.forward()` which calls `self.user_feature_extractor(u_idxs)` and `self.item_feature_extractor(i_idxs)` at lines 103/111. The feature lookup must happen *inside* the extractor using stored `self.feature_table`.

**Existing class to study:** `PrototypeEmbedding` (lines 191–329) — the core component:
- `forward()` (line 299): embeds via internal `Embedding`, computes cosine `sim_mtx` to prototypes (line 311), accumulates regularization losses (lines 321–322).
- Regularization: `_entropy_reg_loss()` (line 278), `_inclusiveness_constraint()` (line 284), max-pool reg (line 267).
- Loss returned in `get_and_reset_loss()` (lines 326–329): weighted sum of proto and batch reg terms.
- `sim_mtx` is the explanation signal — shape `[batch, n_prototypes]`.

### B2. `feature_extraction/feature_extractor_factories.py` — Add new `ft_type` branches

**Two levels of dispatch:**

**Level 1: `create_models()` (lines 12–117)** — Creates paired user + item extractors.

Add new branch after `ft_type == 'acf'` block (after line 115):

```python
elif ft_type == 'feature_item_proto':
    # Item side: FeatureAwarePrototypeEmbedding (CF protos + feature protos)
    # User side: standard Embedding (dim = total item output dim)
    ...
```

**Level 2: `create_model()` (lines 119–173)** — Creates individual extractors.

Add new branch after `ft_type == 'acf'` block (after line 167):

```python
elif ft_type == 'feature_aware_prototypes':
    # Extract feature-specific params: feature_dim, n_feature_prototypes, data_path
    # Instantiate FeatureAwarePrototypeEmbedding
    ...
```

**Weight tying consideration:** Study the `prototypes_double_tie` branch (lines 69–99) carefully. It ties embedding weights between proto and embed branches (lines 93–94). The feature-aware variant may need analogous tying if using dual prototype spaces. The `data_path` parameter must be threaded through to enable loading `item_features.csv`.

### B3. `rec_sys/trainer.py` — Pass `data_path` to factory

**Line 54–55:** Currently:
```python
user_fe, item_fe = FeatureExtractorFactory.create_models(
    conf.n_users, conf.n_items, conf.ft_ext_param)
```

Needs `data_path` added so the factory can load `item_features.csv`:
```python
user_fe, item_fe = FeatureExtractorFactory.create_models(
    conf.n_users, conf.n_items, conf.ft_ext_param, data_path=conf.data_path)
```

Same change in `rec_sys/tester.py` lines 42–46.

**No other trainer changes needed.** The batch format `(u_idxs, i_idxs, labels)` stays the same. Feature lookup happens inside the extractor, not the dataloader.

### B4. `confs/hyper_params.py` — New hyperparameter configs

Add new config dicts (e.g., `feature_item_proto_hyper_params`) following the pattern of existing configs. New tunable parameters:

```python
feature_item_proto_hyper_params = {
    **base_hyper_params,
    'ft_type': 'feature_item_proto',
    'embedding_dim': tune.randint(10, 100),
    'item_ft_ext_param': {
        'ft_type': 'feature_aware_prototypes',
        'n_prototypes': tune.randint(10, 100),       # CF prototypes
        'n_feature_prototypes': tune.randint(10, 50), # Feature prototypes
        'feature_dim': tune.choice([16, 32, 64]),     # Feature encoder output dim
        'sim_proto_weight': tune.loguniform(1e-4, 1),
        'sim_batch_weight': tune.loguniform(1e-4, 1),
        # ... regularization type choices
    },
    'user_ft_ext_param': {
        'ft_type': 'embedding',
    },
}
```

### B5. `start.py` — Register new model CLI names

**Line 15:** Add new choices (e.g., `'feature_item_proto'`).

**Lines 34–45:** Add import and routing:
```python
elif model == 'feature_item_proto':
    conf_dict = feature_item_proto_hyper_params
```

### B6. Unchanged files:

| File | Why not |
|------|---------|
| `rec_sys/protomf_dataset.py` | Feature lookup is inside extractor, not dataloader. Batch format unchanged. |
| `rec_sys/rec_sys.py` | Calls extractors via `forward(o_idxs)` → gets embeddings → dot product. As long as user and item output dims match, RecSys doesn't care what's inside. |
| `utilities/eval.py` | Metric computation is model-agnostic. |

---

## C. Extending Explanations (Phase 4)

### C1. `utilities/explanations_utils.py` — New explanation functions

**Existing functions to build on:**
- `tsne_plot()` (line 7): Visualize prototype spaces with cosine TSNE. Extend to color by item category (from `articles.csv` metadata).
- `get_top_k_items()` (line 37): Find representative items per prototype. Extend to show feature profiles (e.g., "Prototype 3: Trousers, Dark, Menswear").
- `weight_visualization()` (line 65): Decompose a recommendation into per-prototype contributions. Extend to show CF vs. feature prototype contributions separately.

**New functions to add:**
- `get_prototype_feature_profile()` — For each feature prototype, aggregate the dominant feature values of its closest items. Produces interpretable labels like "Dresses + Red + Womenswear".
- `explain_recommendation_dual()` — Like `weight_visualization()` but with two explanation streams: "recommended because users like you interact with it (CF) AND because of its attributes (features)".
- `visualize_prototype_items_with_metadata()` — Table/grid of representative items with their feature columns displayed.

**Key internal tensor for explanations:**
- `sim_mtx` — computed in `PrototypeEmbedding.forward()` at line 311. Shape `[batch, n_prototypes]`. This is the similarity of each entity to each prototype — the raw explanation signal.
- For dual prototype spaces: two separate `sim_mtx` tensors (CF and feature), concatenated in final output but separable for explanation purposes.

---

## D. File-by-File Reference

Quick-reference table of every file's role and what changes it needs.

| File | Role | Phase 3 | Phase 4 |
|------|------|---------|---------|
| `start.py` | CLI entry, model/dataset dispatch | Add `'hm'` to dataset choices (L16) | Add new model choices (L15), routing (L34–45) |
| `experiment_helper.py` | Ray Tune orchestration, data loading | Add ASHA for H&M (L92–95) | — |
| `confs/hyper_params.py` | Search spaces per model | — | Add new model configs |
| `feature_extraction/feature_extractors.py` | All model architectures | — | Add `FeatureEncoder`, `FeatureAwarePrototypeEmbedding` |
| `feature_extraction/feature_extractor_factories.py` | Factory: config → model instances | — | Add new `ft_type` branches in both `create_models()` and `create_model()` |
| `rec_sys/rec_sys.py` | Combine extractors, compute loss | — | — (unchanged) |
| `rec_sys/trainer.py` | Training loop, checkpointing | — | Pass `data_path` to factory (L54–55) |
| `rec_sys/tester.py` | Test evaluation, checkpoint loading | — | Pass `data_path` to factory (L42–46) |
| `rec_sys/protomf_dataset.py` | Dataset + negative sampling | — | — (unchanged) |
| `utilities/eval.py` | NDCG/HR metrics | — | — (unchanged) |
| `utilities/consts.py` | Global constants | — | — (unchanged) |
| `utilities/explanations_utils.py` | Prototype interpretation, TSNE, vis | — | Add feature-aware explanation functions |
| `data/hm/hm_splitter.py` | **New file** — H&M preprocessing | Create (template: `movielens_splitter.py`) | — |

---

## Data Flow Diagram

```
CLI (start.py)
  │
  ├─ model name ──→ hyper_params.py ──→ conf dict (ft_type, search space)
  │                                         │
  ├─ dataset name ──→ DATA_PATH/dataset/ ───┤
  │                                         │
  └─ experiment_helper.py                   │
       │                                    │
       ├─ Ray Tune scheduler (ASHA for large datasets)
       │                                    │
       └─ Trial function ──────────────────→│
                                            │
                                    ┌───────┴────────┐
                                    │   Trainer      │
                                    │                │
                                    │  ProtoRecDataset(data_path)
                                    │    → (u_idxs, i_idxs, labels)
                                    │                │
                                    │  FeatureExtractorFactory.create_models(ft_ext_param)
                                    │    → user_fe, item_fe
                                    │                │
                                    │  RecSys(user_fe, item_fe, loss_func)
                                    │    .forward(u_idxs, i_idxs) → logits
                                    │    .loss_func(logits, labels) → loss
                                    │      └─ rec_loss + user_fe.get_and_reset_loss()
                                    │                  + item_fe.get_and_reset_loss()
                                    │                │
                                    │  Evaluator (HR@k, NDCG@k)
                                    │    → validation metrics → early stopping
                                    │    → best checkpoint saved
                                    └────────────────┘
```

---

## Prototype Architecture Detail

```
PrototypeEmbedding.forward(o_idxs):
  │
  ├─ self.embedding_ext(o_idxs)           → entity_embed [batch, embed_dim]
  │
  ├─ cosine_similarity(entity_embed,       → sim_mtx [batch, n_prototypes]
  │                    self.prototypes)       ← THIS IS THE EXPLANATION SIGNAL
  │
  ├─ (optional) sim_mtx @ weight_matrix   → projected [batch, out_dim]
  │
  ├─ accumulate reg losses:
  │   ├─ reg_batch(sim_mtx)  × sim_batch_weight
  │   └─ reg_proto(sim_mtx)  × sim_proto_weight
  │
  └─ return sim_mtx (or projected)
```

**For Phase 4 (Dual Prototype Spaces):**
```
FeatureAwarePrototypeEmbedding.forward(item_idxs):
  │
  ├─ CF branch:
  │   └─ PrototypeEmbedding(item_idxs)     → cf_sim [batch, n_cf_protos]
  │
  ├─ Feature branch:
  │   ├─ self.feature_table[item_idxs]      → raw features
  │   ├─ per-feature Embeddings + MLP       → feature_embed [batch, feat_dim]
  │   └─ cosine_sim(feature_embed, feat_protos) → feat_sim [batch, n_feat_protos]
  │
  └─ concat(cf_sim, feat_sim)              → [batch, n_cf_protos + n_feat_protos]
      Two separable explanation streams:
        "Users like you interact with it" (CF)
        "Because of its attributes" (Features)
```
