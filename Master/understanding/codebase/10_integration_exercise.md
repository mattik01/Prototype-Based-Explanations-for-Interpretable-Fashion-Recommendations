# Session 10: Integration Exercise — Guided Walkthrough

**Files:** All modules — this is a synthesis session

No new code to read. This session connects everything from sessions 01–09 by tracing complete paths through the system. We'll also identify the exact extension points for Phases 3–4.

At each location, modifiable parameters are listed in *italics* below, tagged as:
- **DECLARED** — the value is defined/set here (with `[search]` if it's a Ray Tune search space)
- **CONSUMED** — the value was declared higher up in the hierarchy but becomes important at this point

---

## Part A: End-to-End Traces

### Trace 1: From CLI to Test Metric

Let's follow `python start.py -m user_item_proto -d ml-1m` through the entire system.

**1. `start.py`** — Parses CLI arguments. `-m user_item_proto` maps to `ft_type='prototypes_double_tie'`. `-d ml-1m` sets the dataset. Calls `start_hyper()` (or `start_multiple_hyper()` with `-mp`).

> *Parameters at this location:*
> - `model` (`-m`): which model config dict to load — **DECLARED** (CLI argument, choices: `mf`, `acf`, `user_proto`, `item_proto`, `user_item_proto`, `debug`)
> - `dataset` (`-d`): which dataset to use — **DECLARED** (CLI argument, choices: `amazon2014`, `ml-1m`, `lfm2b-1mon`)
> - `multiple` (`-mp`): single-seed vs multi-seed run — **DECLARED** (CLI flag, default `False`)
> - `seed` (`-s`): random seed override — **DECLARED** (CLI argument, default `SINGLE_SEED` from consts)

**2. `experiment_helper.py: start_hyper()`** — Sets up Ray Tune with HyperOpt search algorithm. The search space comes from `hyper_params.py` for the `user_item_proto` model. Optionally creates an ASHA scheduler (only for `lfm2b-1mon`). Launches `tune.run()` with `start_training` as the training function.

> *Parameters at this location:*
> - `search_alg`: HyperOptSearch with seed — **DECLARED** (hardcoded to HyperOptSearch)
> - `scheduler`: ASHA early stopping — **DECLARED** (only for `lfm2b-1mon`; `grace_period=4` hardcoded)
> - `num_samples`: how many hyperparameter trials to run — **CONSUMED** from config dict, falls back to `NUM_SAMPLES` from consts (default 100)
> - `resources_per_trial`: GPU/CPU per trial — **CONSUMED** from `GPU_PER_TRIAL` (0.2) and `CPU_PER_TRIAL` (1) in consts
> - `metric`: which metric to optimize — **CONSUMED** from `OPTIMIZING_METRIC` in consts (`hit_ratio@10`)
> - `mode`: optimization direction — **DECLARED** (hardcoded `'max'`)
> - `data_path`: path to dataset files — **DECLARED** (composed from `DATA_PATH` + dataset name)
> - `seed`: injected into config dict — **CONSUMED** from CLI or `SINGLE_SEED`

**3. `experiment_helper.py: start_training(config)`** — Called by Ray Tune for each trial. Converts the config dict to a Namespace. Calls `load_data()` to build DataLoaders. Sets random seeds for reproducibility. Creates a `Trainer` and calls `trainer.run()`.

> *Parameters at this location:*
> - `seed`: passed to `reproducible()` — **CONSUMED** (sets `random`, `torch`, `numpy`, `cudnn` seeds)

**4. `experiment_helper.py: load_data()`** — Calls `get_protorecdataset_dataloader()` three times (train/val/test) with different parameters. Train loader has `n_neg` from config and shuffling; val/test loaders have `n_neg=99` and no shuffling.

> *Parameters at this location:*
> - `neg_train` (train n_neg): number of negatives per positive during training — **CONSUMED** from config `[search: tune.randint(1, 50)]`
> - `train_neg_strategy`: negative sampling strategy for training — **CONSUMED** from config `[search: tune.choice(['popular', 'uniform'])]`
> - `eval_neg_strategy`: negative sampling strategy for val/test — **CONSUMED** from config (fixed `'uniform'` in `base_param`)
> - `batch_size`: training batch size — **CONSUMED** from config `[search: tune.lograndint(64, 512, 2)]`
> - `val_batch_size`: validation/test batch size — **CONSUMED** from config (fixed `256` in `base_param`)
> - `NEG_VAL`: number of negatives for val/test — **CONSUMED** from consts (hardcoded `99`)
> - `NUM_WORKERS`: dataloader workers — **CONSUMED** from consts (hardcoded `2`)
> - `prefetch_factor`: train loader prefetch — **DECLARED** (hardcoded `5`, only for train)
> - `shuffle`: whether to shuffle batches — **DECLARED** (hardcoded `True` for train, omitted/`False` for val/test)

**5. `protomf_dataset.py`** — `ProtoRecDataset.load_data()` reads the CSV splits, builds COO matrix (for iteration) and CSR matrix (for negative sampling exclusion). The DataLoader wraps this and provides batches of `(user_idx, item_idxs, labels)`.

> *Parameters at this location:*
> - `data_path`: directory with CSV files — **CONSUMED** from config
> - `split_set`: `'train'`/`'val'`/`'test'` — **CONSUMED** from `load_data()` caller
> - `n_neg`: negatives per positive — **CONSUMED** from config (`neg_train` for train, `NEG_VAL` for val/test)
> - `neg_strategy`: `'uniform'` or `'popular'` — **CONSUMED** from config
> - Popularity smoothing exponent: — **DECLARED** (hardcoded `0.75` in `_neg_sample_popular`, line 143)

**6. `trainer.py: _build_model()`** — Gets `n_users/n_items` from the dataset. Calls `FeatureExtractorFactory.create_models()`.

> *Parameters at this location:*
> - `n_users`, `n_items`: derived from dataset — **CONSUMED** (read from `train_loader.dataset`)
> - `ft_ext_param`: full feature extractor config dict — **CONSUMED** from config (passed to factory)
> - `rec_sys_param`: RecSys-level config — **CONSUMED** from config
> - `loss_func_name`: which loss — **CONSUMED** from config `[search: tune.choice(['bce', 'bpr', 'sampled_softmax'])]`
> - `loss_func_aggr`: loss aggregation — **CONSUMED** from config (fixed `'mean'` for most models)

**7. `feature_extractor_factories.py: create_models()`** — Sees `ft_type='prototypes_double_tie'`. Builds four sub-extractors: user_proto (PrototypeEmbedding), user_embed (EmbeddingW), item_proto (PrototypeEmbedding), item_embed (EmbeddingW). Ties embedding weights. Wraps each side in ConcatenateFeatureExtractors.

> *Parameters at this location:*
> - `ft_type`: top-level model architecture — **CONSUMED** from config (e.g. `'prototypes_double_tie'`, `'detached'`, `'prototypes'`, `'acf'`)
> - `embedding_dim`: shared raw embedding dimension — **CONSUMED** from config `[search: tune.randint(10, 100)]`
> - `user_ft_ext_param.n_prototypes`: number of user prototypes — **CONSUMED** from config `[search: tune.randint(10, 100)]`
> - `item_ft_ext_param.n_prototypes`: number of item prototypes — **CONSUMED** from config `[search: tune.randint(10, 100)]`
> - `use_weight_matrix`: must be `False` for double tie — **CONSUMED** from config (fixed `False`, asserted on line 76)
> - `out_dimension`: output size for EmbeddingW linear layers — **DECLARED** (set here to the *other* side's `n_prototypes`, lines 82 and 89)
> - `invert`: concatenation order flag — **DECLARED** (hardcoded `False` for user, `True` for item, lines 96–97)

**8. `feature_extractors.py: PrototypeEmbedding.__init__()`** — Creates embedding table, prototype vectors, cosine function, regularization functions.

> *Parameters at this location:*
> - `n_prototypes`: how many prototype vectors to learn — **CONSUMED** from config `[search: tune.randint(10, 100)]`
> - `embedding_dim`: dimensionality of embedding and prototype vectors — **CONSUMED** from config `[search: tune.randint(10, 100)]`
> - `cosine_type`: type of cosine similarity — **CONSUMED** from config (fixed `'shifted'`; options: `'standard'`, `'shifted'`, `'shifted_and_div'`)
> - `reg_proto_type`: regularization on prototype axis — **CONSUMED** from config (fixed `'max'`; options: `'max'`, `'soft'`, `'incl'`)
> - `reg_batch_type`: regularization on batch axis — **CONSUMED** from config (fixed `'max'`; options: `'max'`, `'soft'`)
> - `sim_proto_weight`: weight on prototype regularization loss — **CONSUMED** from config `[search: tune.loguniform(1e-3, 10)]`
> - `sim_batch_weight`: weight on batch regularization loss — **CONSUMED** from config `[search: tune.loguniform(1e-3, 10)]`
> - `max_norm`: max L2 norm for embeddings — **CONSUMED** from config (default `None`, unused in prototype configs)
> - `use_weight_matrix`: optional internal linear projection — **CONSUMED** from config (fixed `False` for double tie)

**8b. `feature_extractors.py: EmbeddingW.__init__()`** — Creates embedding table + linear layer for projection.

> *Parameters at this location:*
> - `embedding_dim`: input dimension (shared with proto branch) — **CONSUMED** from config
> - `out_dimension`: output dimension of linear layer — **CONSUMED** from factory (set to other side's `n_prototypes`)
> - `use_bias`: whether linear layer has a bias term — **CONSUMED** from config (default `False`)
> - `max_norm`: max L2 norm for embeddings — **CONSUMED** from config (default `None`)

**8c. `feature_extractors.py: Embedding.__init__()`** — Base embedding lookup (used inside PrototypeEmbedding and EmbeddingW).

> *Parameters at this location:*
> - `n_objects`: number of users or items (table size) — **CONSUMED** from factory
> - `embedding_dim`: vector dimensionality — **CONSUMED** from config
> - `max_norm`: max L2 norm constraint on embedding vectors — **CONSUMED** from config (default `None`)
> - `only_positive`: force non-negative embeddings via `abs()` — **CONSUMED** from config (default `False`, not used in standard configs)

**9. `rec_sys.py: RecSys.__init__()`** — Receives the two ConcatenateFeatureExtractors. Selects the loss function via `functools.partial`. Sets up bias embeddings if enabled.

> *Parameters at this location:*
> - `use_bias`: whether to add user/item/global bias to dot product — **CONSUMED** from config `rec_sys_param` (fixed `0` = disabled in `base_param`; uses `> 0` check for Ray Tune compatibility)
> - `loss_func_name`: selects BCE/BPR/sampled_softmax — **CONSUMED** from config `[search: tune.choice(['bce', 'bpr', 'sampled_softmax'])]`
> - `loss_func_aggr`: `'mean'` or `'sum'` reduction — **CONSUMED** from config (fixed `'mean'` for proto models, `'sum'` for ACF)

**10. `trainer.py: _build_model()` continued** — Calls `rec_sys.init_parameters()`, wraps in `DataParallel`, moves to device.

> *Parameters at this location:*
> - `device`: `'cuda'` or `'cpu'` — **CONSUMED** from config (auto-detected in `base_param`)
> - Weight initialization strategy — **DECLARED** in `utils.py:general_weight_init()`: Xavier/Kaiming for Linear layers, normal for Embedding layers, constant 0 for biases. Not configurable.

**10b. `trainer.py: _build_optimizer()`** — Creates the optimizer.

> *Parameters at this location:*
> - `optim`: optimizer type — **CONSUMED** from config `optim_param` `[search: tune.choice(['adam', 'adagrad'])]`
> - `lr`: learning rate — **CONSUMED** from config `optim_param` `[search: tune.loguniform(1e-4, 1e-1)]`
> - `wd`: weight decay (L2 regularization) — **CONSUMED** from config `optim_param` `[search: tune.loguniform(1e-4, 1e-2)]`

**11. `trainer.py: run()`** — Validates with random weights (baseline). Enters epoch loop: train batches → validate → report to Ray Tune → save checkpoint if improved → early stop after patience=10.

> *Parameters at this location:*
> - `n_epochs`: maximum training epochs — **CONSUMED** from config (fixed `100` in `base_param`)
> - `max_patience`: epochs without improvement before early stop — **CONSUMED** from `MAX_PATIENCE` in consts (hardcoded `10`)
> - `optimizing_metric`: which metric to track for improvement — **CONSUMED** from `OPTIMIZING_METRIC` in consts (`hit_ratio@10`)

**12. `experiment_helper.py: start_hyper()` continued** — After all trials complete, `analysis.get_best_trial()` finds the best hyperparameters based on `hit_ratio@10`. Calls `start_testing()` with the best checkpoint path.

> *Parameters at this location:*
> - `metric_name`: metric for selecting best trial — **CONSUMED** from `OPTIMIZING_METRIC`
> - `scope`: how to select best (`'all'` = across all epochs) — **DECLARED** (hardcoded)

**13. `tester.py`** — Rebuilds the same architecture, loads saved weights via `load_state_dict()`, runs test evaluation, logs metrics to W&B.

> *Parameters at this location:*
> - `model_load_path`: checkpoint path — **CONSUMED** from `start_hyper()`
> - All architecture params (ft_ext_param, rec_sys_param, loss_func_name, etc.) — **CONSUMED** from best trial config (must match training exactly for `load_state_dict` to work)
> - Note: Tester does NOT wrap in `DataParallel` — accesses `self.model.loss_func()` directly (vs Trainer's `self.model.module.loss_func()`)

**14. `eval.py: Evaluator`** — Computes HR@K and NDCG@K for K∈{1,3,5,10,50}. Accumulates across batches, divides by `n_users`. Final numbers are the paper's results.

> *Parameters at this location:*
> - `K_VALUES`: which K values to evaluate — **CONSUMED** from consts (hardcoded `[1, 3, 5, 10, 50]`)
> - `n_users`: divisor for averaging metrics — **CONSUMED** from dataset

---

### Trace 2: A Single Training Batch

Let's follow one batch through the forward-backward pass. Assume batch_size=64, n_neg=10, user_n_prototypes=20, item_n_prototypes=30, embedding_dim=50.

**DataLoader yields**: `u_idxs [64]`, `i_idxs [64, 11]`, `labels [64, 11]`

> *Parameters shaping this output:*
> - `batch_size` — **CONSUMED** `[search: tune.lograndint(64, 512, 2)]`
> - `neg_train` (n_neg) — **CONSUMED** `[search: tune.randint(1, 50)]`
> - `neg_strategy` — **CONSUMED** `[search: tune.choice(['popular', 'uniform'])]`
> - Popularity exponent `0.75` — **DECLARED** in `protomf_dataset.py:143` (hardcoded, only used when strategy=`'popular'`)

**Move to device**: same shapes, now on GPU

> *Parameters:*
> - `device` — **CONSUMED** from config

**RecSys.forward()**:
1. `user_feature_extractor(u_idxs [64])`:
   - ConcatenateFeatureExtractors runs both sub-extractors:
     - user_proto: lookup → `[64, 50]` → cosine with prototypes → `[64, 20]` (accumulates reg losses)
     - user_embed: lookup (shared weights) → `[64, 50]` → linear → `[64, 30]`
   - Concatenate (no invert): `[64, 20+30]` = `[64, 50]`

> *Parameters active in user proto branch forward:*
> - `embedding_dim` (50) — **CONSUMED** (determines lookup vector size and prototype vector size)
> - `n_prototypes` (20) — **CONSUMED** (number of prototype vectors, output dimension)
> - `cosine_type` (`'shifted'`) — **CONSUMED** (determines `1 + cos` vs plain `cos` vs `(1+cos)/2`)
> - `reg_batch_type` (`'max'`) — **CONSUMED** (how batch regularization is computed on `sim_mtx`)
> - `reg_proto_type` (`'max'`) — **CONSUMED** (how prototype regularization is computed on `sim_mtx`)
> - `sim_batch_weight` — **CONSUMED** `[search]` (multiplied onto accumulated batch reg loss)
> - `sim_proto_weight` — **CONSUMED** `[search]` (multiplied onto accumulated proto reg loss)
>
> *Parameters active in user embed branch forward:*
> - `embedding_dim` (50) — **CONSUMED** (input to linear layer)
> - `out_dimension` (30 = item_n_prototypes) — **CONSUMED** from factory (output of linear layer)
> - Linear layer weight matrix `W [50, 30]` — **DECLARED** as `nn.Linear` (learned parameter)

2. `item_feature_extractor(i_idxs [64, 11])`:
   - ConcatenateFeatureExtractors runs both sub-extractors:
     - item_proto: lookup → `[64, 11, 50]` → cosine with prototypes → `[64, 11, 30]` (accumulates reg losses)
     - item_embed: lookup (shared weights) → `[64, 11, 50]` → linear → `[64, 11, 20]`
   - Concatenate (invert=True): `[64, 11, 20+30]` = `[64, 11, 50]`

> *Parameters active in item proto branch forward:*
> - Same parameter types as user proto branch above, but with item-side values
> - `n_prototypes` (30) — **CONSUMED**
> - `sim_batch_weight`, `sim_proto_weight` — **CONSUMED** `[search]` (item-side has independent search ranges)
>
> *Parameters active in item embed branch forward:*
> - `out_dimension` (20 = user_n_prototypes) — **CONSUMED** from factory

3. Dot product:
   - `u_embed.unsqueeze(1)` → `[64, 1, 50]`
   - `* i_embed` → `[64, 11, 50]` (broadcast)
   - `sum(dim=-1)` → `[64, 11]` — the logits

4. Add biases if enabled → still `[64, 11]`

> *Parameters:*
> - `use_bias` — **CONSUMED** (if enabled: adds `user_bias[u] + item_bias[i] + global_bias` to each logit)

**RecSys.loss_func(logits [64, 11], labels [64, 11])**:
1. `rec_loss` = selected loss function (BCE/BPR/softmax) on logits and labels

> *Parameters active in loss computation:*
> - `loss_func_name` — **CONSUMED** `[search: tune.choice(['bce', 'bpr', 'sampled_softmax'])]`
> - `loss_func_aggr` — **CONSUMED** (fixed `'mean'` for proto models; `'sum'` default for softmax)
> - BCE-specific: positive weight = `n_neg` — **DECLARED** in `rec_sys.py:139` (automatically derived from logits shape)
> - BPR-specific: labels replicated per negative — **DECLARED** in `rec_sys.py:160-161`

2. `item_feat_ext_loss` = ConcatenateFeatureExtractors drains both sub-extractors:
   - item_proto's accumulated `sim_proto_weight * proto_reg + sim_batch_weight * batch_reg`
   - item_embed (EmbeddingW) returns 0

> *Parameters:*
> - `sim_proto_weight` (item side) — **CONSUMED** `[search]` (multiplier on proto regularization)
> - `sim_batch_weight` (item side) — **CONSUMED** `[search]` (multiplier on batch regularization)

3. `user_feat_ext_loss` = same pattern for user side

> *Parameters:*
> - `sim_proto_weight` (user side) — **CONSUMED** `[search]` (independent from item side)
> - `sim_batch_weight` (user side) — **CONSUMED** `[search]` (independent from item side)

4. Total = rec_loss + item_feat_ext_loss + user_feat_ext_loss

**loss.backward()**: Gradients flow back through everything — the rec_loss gradients reach all parameters, and the regularization gradients reach the prototype matrices and embedding weights.

**optimizer.step()**: All parameters updated.

> *Parameters:*
> - `optim` — **CONSUMED** `[search: tune.choice(['adam', 'adagrad'])]`
> - `lr` — **CONSUMED** `[search: tune.loguniform(1e-4, 1e-1)]`
> - `wd` (weight decay) — **CONSUMED** `[search: tune.loguniform(1e-4, 1e-2)]`

---

### Trace 3: How Regularization Gradients Flow

Starting from `PrototypeEmbedding.forward()`:

1. During forward, `sim_mtx` is computed as cosine similarity between embeddings and prototypes

> *Parameters:*
> - `cosine_type` — **CONSUMED** (determines the similarity function applied)
> - `embedding_dim` — **CONSUMED** (dimensionality of both embeddings and prototypes for cosine computation)

2. `self._acc_r_batch += self.reg_batch_func(batch_proto)` — batch regularization computed on `sim_mtx`

> *Parameters:*
> - `reg_batch_type` — **CONSUMED** (selects `'max'`: negative max similarity per object; or `'soft'`: entropy of softmax over prototypes per object)

3. `self._acc_r_proto += self.reg_proto_func(batch_proto)` — prototype regularization computed on `sim_mtx`

> *Parameters:*
> - `reg_proto_type` — **CONSUMED** (selects `'max'`: negative max similarity per prototype; or `'soft'`: entropy of softmax over objects per prototype; or `'incl'`: inclusiveness constraint like ACF)

4. These accumulated values retain their computation graph (they're tensors with `requires_grad=True`)

5. In `RecSys.loss_func()`, `get_and_reset_loss()` returns the weighted sum

> *Parameters:*
> - `sim_proto_weight` — **CONSUMED** `[search]` (multiplier on `_acc_r_proto`)
> - `sim_batch_weight` — **CONSUMED** `[search]` (multiplier on `_acc_r_batch`)

6. This gets added to `rec_loss`
7. `loss.backward()` differentiates through the regularization → gradients flow to:
   - `self.prototypes` (the prototype vectors) — **learned `nn.Parameter`**, shape `[n_prototypes, embedding_dim]`
   - `self.embedding_ext.embedding_layer.weight` (the object embeddings) — **learned `nn.Embedding`**, shape `[n_objects, embedding_dim]`
   - Both are learnable parameters, so the regularization shapes them during training

The regularization gradients push:
- Embeddings to have clear prototype affinities (batch reg)
- Prototypes to each "own" some objects (proto reg)
- The main rec_loss gradients push everything toward better recommendations

The final learned prototypes are a balance between these forces.

---

## Complete Parameter Summary

### Search space parameters (tuned by Ray Tune per trial)

| Parameter | Declared in | Search range | Consumed in |
|-----------|-------------|-------------|-------------|
| `neg_train` | `hyper_params.py` | `tune.randint(1, 50)` | `load_data()` → `ProtoRecDataset` |
| `train_neg_strategy` | `hyper_params.py` | `tune.choice(['popular', 'uniform'])` | `ProtoRecDataset.__getitem__()` |
| `loss_func_name` | `hyper_params.py` | `tune.choice(['bce', 'bpr', 'sampled_softmax'])` | `RecSys.__init__()` |
| `batch_size` | `hyper_params.py` | `tune.lograndint(64, 512, 2)` | `load_data()` → DataLoader |
| `optim` | `hyper_params.py` | `tune.choice(['adam', 'adagrad'])` | `Trainer._build_optimizer()` |
| `wd` | `hyper_params.py` | `tune.loguniform(1e-4, 1e-2)` | `Trainer._build_optimizer()` |
| `lr` | `hyper_params.py` | `tune.loguniform(1e-4, 1e-1)` | `Trainer._build_optimizer()` |
| `embedding_dim` | `hyper_params.py` | `tune.randint(10, 100)` | `FeatureExtractorFactory` → all extractors |
| `user n_prototypes` | `hyper_params.py` | `tune.randint(10, 100)` | `PrototypeEmbedding` (user), `EmbeddingW` out_dim (item) |
| `item n_prototypes` | `hyper_params.py` | `tune.randint(10, 100)` | `PrototypeEmbedding` (item), `EmbeddingW` out_dim (user) |
| `user sim_proto_weight` | `hyper_params.py` | `tune.loguniform(1e-3, 10)` | `PrototypeEmbedding.get_and_reset_loss()` (user) |
| `user sim_batch_weight` | `hyper_params.py` | `tune.loguniform(1e-3, 10)` | `PrototypeEmbedding.get_and_reset_loss()` (user) |
| `item sim_proto_weight` | `hyper_params.py` | `tune.loguniform(1e-3, 10)` | `PrototypeEmbedding.get_and_reset_loss()` (item) |
| `item sim_batch_weight` | `hyper_params.py` | `tune.loguniform(1e-3, 10)` | `PrototypeEmbedding.get_and_reset_loss()` (item) |

### Fixed parameters (hardcoded or set in base_param, modifiable but not searched)

| Parameter | Value | Declared in | Consumed in |
|-----------|-------|-------------|-------------|
| `n_epochs` | `100` | `hyper_params.py` (base_param) | `Trainer.run()` |
| `val_batch_size` | `256` | `hyper_params.py` (base_param) | `load_data()` → DataLoader |
| `eval_neg_strategy` | `'uniform'` | `hyper_params.py` (base_param) | `ProtoRecDataset` (val/test) |
| `use_bias` | `0` (disabled) | `hyper_params.py` (base_param → rec_sys_param) | `RecSys.__init__()` |
| `loss_func_aggr` | `'mean'` | `hyper_params.py` (per-model config) | `RecSys.__init__()` |
| `cosine_type` | `'shifted'` | `hyper_params.py` (per-extractor) | `PrototypeEmbedding.__init__()` |
| `reg_proto_type` | `'max'` | `hyper_params.py` (per-extractor) | `PrototypeEmbedding.__init__()` |
| `reg_batch_type` | `'max'` | `hyper_params.py` (per-extractor) | `PrototypeEmbedding.__init__()` |
| `use_weight_matrix` | `False` | `hyper_params.py` (per-extractor) | `PrototypeEmbedding.__init__()`, factory assert |
| `max_norm` | `None` | default in extractors | `nn.Embedding` |
| `only_positive` | `False` | default in `Embedding` | `Embedding.forward()` |
| `device` | auto (`cuda`/`cpu`) | `hyper_params.py` (base_param) | everywhere (`.to(device)`) |
| `MAX_PATIENCE` | `10` | `consts.py` | `Trainer.run()` |
| `NEG_VAL` | `99` | `consts.py` | `load_data()` (val/test n_neg) |
| `NUM_SAMPLES` | `100` | `consts.py` | `tune.run()` fallback |
| `K_VALUES` | `[1, 3, 5, 10, 50]` | `consts.py` | `Evaluator.eval_batch()` |
| `OPTIMIZING_METRIC` | `'hit_ratio@10'` | `consts.py` | `Trainer.run()`, `start_hyper()` |
| `GPU_PER_TRIAL` | `0.2` | `consts.py` | `tune.run()` |
| `CPU_PER_TRIAL` | `1` | `consts.py` | `tune.run()` |
| `NUM_WORKERS` | `2` | `consts.py` | DataLoader |
| `SINGLE_SEED` | `38210573` | `consts.py` | default seed |
| `SEED_LIST` | `[38210573, 9491758, 2931009]` | `consts.py` | `start_multiple_hyper()` |
| `prefetch_factor` | `5` | `load_data()` | train DataLoader only |
| `popularity_exponent` | `0.75` | `protomf_dataset.py:143` | `_neg_sample_popular()` |
| `grace_period` (ASHA) | `4` | `experiment_helper.py:93` | ASHA scheduler (lfm2b-1mon only) |
| Weight init | Xavier/normal | `utils.py:general_weight_init()` | `init_parameters()` calls |

### Learned parameters (updated by gradient descent)

| Parameter | Shape | Declared in |
|-----------|-------|-------------|
| User embedding table | `[n_users, embedding_dim]` | `Embedding` (shared via tie) |
| Item embedding table | `[n_items, embedding_dim]` | `Embedding` (shared via tie) |
| User prototype vectors | `[user_n_proto, embedding_dim]` | `PrototypeEmbedding` (`nn.Parameter`) |
| Item prototype vectors | `[item_n_proto, embedding_dim]` | `PrototypeEmbedding` (`nn.Parameter`) |
| User embed linear weights | `[embedding_dim, item_n_proto]` | `EmbeddingW` (`nn.Linear`) |
| Item embed linear weights | `[embedding_dim, user_n_proto]` | `EmbeddingW` (`nn.Linear`) |
| User bias (if enabled) | `[n_users, 1]` | `RecSys` (`nn.Embedding`) |
| Item bias (if enabled) | `[n_items, 1]` | `RecSys` (`nn.Embedding`) |
| Global bias (if enabled) | `[1]` | `RecSys` (`nn.Parameter`) |

---

## Part B: Extension Points for Phase 3 (H&M Dataset)

### Files That Need Changes

| File | What to Change |
|------|---------------|
| `data/hm/hm_splitter.py` | **New file.** Must produce same output format as `movielens_splitter.py`: 5 CSVs (train/val/test listening histories + user_ids + item_ids) with contiguous 0-based IDs. |
| `confs/hyper_params.py` | Add `hm` dataset entry with search spaces. Can start as a copy of ml-1m's config. |
| `start.py` | Add `hm` to the dataset choices in argparse (the `-d` flag). |
| `experiment_helper.py` | Add `hm` to the dataset path mapping (where it resolves dataset name → data directory). Possibly add ASHA scheduler if the dataset is large. |
| `utilities/consts.py` | `DATA_PATH` is auto-detected, so no change needed. But verify `K_VALUES` and `OPTIMIZING_METRIC` make sense for H&M. |

### Constraints the H&M Splitter Must Satisfy

1. **Output CSVs must match the format** that `ProtoRecDataset.load_data()` expects: columns `user_id`, `item_id`, space-separated, integers
2. **IDs must be contiguous 0-based integers** — remapping from H&M's original article/customer IDs
3. **Train/val/test split**: temporal leave-k-out (sorted by timestamp, last → test, second-to-last → val)
4. **K-core filtering**: iterative removal of users/items with fewer than k positive interactions
5. **No information leakage**: popularity statistics, if used, must come from training data only
6. **User/item ID mapping files** must be saved (for later feature lookup in Phase 4)

---

## Part C: Extension Points for Phase 4 (Feature-Aware Prototypes)

### Where Features Would Enter

The most natural entry point is in a **new FeatureExtractor subclass**. Features (article category, color, etc.) modify the embedding before prototype similarity is computed. Three options:

**Option A: Feature-augmented embedding** — Concatenate or add feature vectors to the learned embedding before computing cosine similarity with prototypes. The `PrototypeEmbedding.forward()` would be modified (or subclassed) so that `o_embed` includes feature information.

**Option B: Feature-augmented prototypes** — Each prototype gets a feature-space component. The similarity computation considers both latent and feature similarity.

**Option C: Separate feature pathway** — A new pathway parallel to the prototype computation, combined via `ConcatenateFeatureExtractors` or a new combiner.

### What Would Change

| Component | Changes Needed |
|-----------|---------------|
| `feature_extractors.py` | New subclass (e.g., `FeaturePrototypeEmbedding`) that accepts feature tensors in `forward()` |
| `feature_extractor_factories.py` | New `ft_type` branch (e.g., `feature_item_proto`) in `create_models` and `create_model` |
| `hyper_params.py` | New config with feature-specific hyperparameters |
| `rec_sys.py` | Possibly modify `forward()` to pass feature data through (or handle via the extractor) |
| `protomf_dataset.py` | Load and serve feature vectors alongside user/item indices |
| `start.py` | New model option in argparse |

### The Interface Contract

Any new `FeatureExtractor` subclass must implement:
1. `forward(o_idxs)` — return a representation tensor (must match the other side's dimensions for dot product)
2. `init_parameters()` — initialize any new parameters
3. `get_and_reset_loss()` — return and reset any regularization losses

The existing `RecSys` doesn't need to change if features are handled entirely within the extractor. This is the beauty of the modular design.

---

## Part D: Design Strengths and Risks

### Strengths
- **Clean modularity**: RecSys doesn't know what extractors it's using → easy to add new types
- **Consistent interfaces**: `forward`, `init_parameters`, `get_and_reset_loss` work uniformly
- **Factory pattern**: config → model translation is centralized
- **Loss composition**: rec_loss + extractor losses is cleanly separated

### Potential Pitfalls for Extension
- **Config dict mutation** in `prototypes_double_tie` factory code — be careful not to reuse configs
- **Dimension matching** — the cross-matching constraint between prototype output and embedding dim is implicit, not validated until runtime
- **Feature data flow** — the current `RecSys.forward()` only receives indices, not feature tensors. Adding features may require changing the interface
- **The positive-at-index-0 convention** — it's undocumented except in code comments. Every new component must respect it
- **`DataParallel` vs `.module`** — the Trainer/Tester inconsistency is a paper cut that could cause bugs when modifying

### What to Test First
1. The H&M splitter output matches the expected format (smoke test with ProtoRecDataset)
2. Standard MF trains on H&M without errors (simplest model, isolates data issues)
3. Prototype models train on H&M (validates dimension matching with new dataset sizes)
4. Feature-aware extractor produces correct shapes (unit test forward pass with mock data)

---

## Check Your Understanding

Before considering this codebase walkthrough complete, make sure you can:

- [ ] Trace a complete experiment from `python start.py` to final test metrics
- [ ] Name every file touched and what each does
- [ ] Explain the shape transformations in the user_item_proto forward pass
- [ ] List the files that need changes for a new dataset
- [ ] Identify where feature information would enter the architecture
- [ ] Describe the extractor interface contract (3 methods)
- [ ] Explain the score decomposition used for explanations
- [ ] For any parameter, say where it's declared and where it's consumed
