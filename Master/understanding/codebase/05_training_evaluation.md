# Session 5: Training & Evaluation — Guided Walkthrough

**Files:** `rec_sys/trainer.py` (~166 lines), `rec_sys/tester.py` (~123 lines), `utilities/eval.py` (~98 lines)

Open all three files. We'll go through Trainer first, then Tester, then Evaluator.

---

## The Big Picture

These three files handle the full train/evaluate/test cycle:
- **Trainer** builds the model, runs the training loop with early stopping, saves checkpoints
- **Tester** rebuilds the same architecture, loads saved weights, runs final evaluation
- **Evaluator** computes Hit Ratio and NDCG metrics efficiently

---

## Part A: Trainer (`trainer.py`)

### Step 1: Constructor (lines 16–48)

The Trainer receives train/val DataLoaders and a config namespace. Look at **lines 40–41**:
```python
self.model = self._build_model()
self.optimizer = self._build_optimizer()
```
The model and optimizer are built immediately during construction — not lazily.

**Lines 37–38**: `OPTIMIZING_METRIC` and `MAX_PATIENCE` come from `utilities/consts.py`. The optimizing metric is `hit_ratio@10` — this is what determines "best model." Patience is 10 — after 10 epochs without improvement, training stops.

### Step 2: Building the Model (`_build_model`, lines 50–64)

Follow this carefully — it's where config becomes a live model.

```python
n_users = self.train_loader.dataset.n_users
n_items = self.train_loader.dataset.n_items
user_feature_extractor, item_feature_extractor = \
    FeatureExtractorFactory.create_models(self.ft_ext_param, n_users, n_items)
```

**Lines 52–55**: The dataset knows how many users/items exist. The factory creates the extractor pair based on the config (we'll cover this in Session 8).

**Line 60**: `rec_sys.init_parameters()` — remember from Session 4, this initializes the weights and sets the `initialized` flag.

**Line 61**: `rec_sys = nn.DataParallel(rec_sys)` — this wraps the model for multi-GPU training. Even on a single GPU (or CPU), this wrapper works fine — it just passes through. The reason it's always applied is simplicity: one code path regardless of hardware. But it has an important consequence...

**Line 62**: `rec_sys = rec_sys.to(self.device)` — moves everything to GPU (or keeps on CPU).

> **Pause and check:** `DataParallel` wraps the model. So `self.model` is now a `DataParallel` object, not a `RecSys`. The original `RecSys` is accessible via `self.model.module`. This is why later, on line 112, you see `self.model.module.loss_func(out, labels)` — you need `.module` to access the RecSys's loss function. Keep this in mind.

### Step 3: Building the Optimizer (lines 66–83)

Straightforward — reads `optim` name from config, creates Adam or Adagrad with learning rate and weight decay. Both are extracted with defaults: `lr=1e-3`, `wd=1e-4`.

### Step 4: The Training Loop (`run`, lines 85–135)

This is the heart. Let's walk through it.

**Lines 89–92** — Pre-training validation:
```python
metrics_values = self.val()
best_value = metrics_values[self.optimizing_metric]
tune.report(metrics_values)
```
Before any training happens, the model is validated with random weights. This establishes a **baseline** — if the model can't beat random initialization, something is wrong. It also gives Ray Tune an initial data point.

**Lines 94–99** — The epoch loop with early stopping:
```python
patience = 0
for epoch in range(self.n_epochs):
    if patience == self.max_patience:
        print('Max Patience reached, stopping.')
        break
```
Simple counter: if `best_value` doesn't improve for 10 consecutive epochs, stop.

**Lines 101–120** — The inner training loop for one epoch:

```python
self.model.train()                     # Line 101: enable dropout/batchnorm training mode
for u_idxs, i_idxs, labels in self.train_loader:
    u_idxs = u_idxs.to(self.device)    # Move to GPU
    i_idxs = i_idxs.to(self.device)
    labels = labels.to(self.device)

    out = self.model(u_idxs, i_idxs)   # Line 110: forward pass
    loss = self.model.module.loss_func(out, labels)  # Line 112: total loss

    loss.backward()                     # Line 116: compute gradients
    self.optimizer.step()               # Line 117: update weights
    self.optimizer.zero_grad()          # Line 118: clear gradients
```

Notice the gradient order: `backward → step → zero_grad`. Most tutorials show `zero_grad → backward → step`. Does it matter? Not really — `zero_grad` just needs to happen before the *next* `backward()`. Putting it after `step` is slightly more memory-efficient in some PyTorch versions because it can free gradient memory sooner. On the very first iteration, gradients start at zero anyway, so it's safe.

**Line 112**: Note `self.model.module.loss_func` — this is the `DataParallel` consequence. `self.model` is a `DataParallel` wrapper, so you need `.module` to reach the underlying `RecSys` and its `loss_func` method.

**Lines 123–135** — Post-epoch validation and checkpointing:
```python
metrics_values = self.val()
curr_value = metrics_values[self.optimizing_metric]
tune.report({**metrics_values, 'epoch_train_loss': epoch_train_loss})

if curr_value > best_value:
    best_value = curr_value
    with tune.checkpoint_dir(0) as checkpoint_dir:
        torch.save(self.model.module.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
    patience = 0
else:
    patience += 1
```

After each epoch: validate, report metrics to Ray Tune, and if it's the best so far, save a checkpoint. The `tune.checkpoint_dir(0)` asks Ray Tune for a directory to save into — the `0` is just a step index (always 0 here since we overwrite the same checkpoint). `state_dict()` saves just the weights, not the architecture. And again, `.module` is needed to get past `DataParallel`.

### Step 5: Validation (`val`, lines 137–165)

```python
@torch.no_grad()
def val(self):
    self.model.eval()
    eval = Evaluator(self.val_loader.dataset.n_users)

    for u_idxs, i_idxs, labels in self.val_loader:
        out = self.model(u_idxs, i_idxs)
        val_loss += self.model.module.loss_func(out, labels).item()
        out = nn.Sigmoid()(out)
        out = out.to('cpu')
        eval.eval_batch(out)

    metrics_values = {**eval.get_results(), 'val_loss': val_loss}
```

Key details:
- `@torch.no_grad()` — disables gradient computation for efficiency (no backprop during validation)
- `self.model.eval()` — sets the model to evaluation mode (affects dropout/batchnorm, though this codebase doesn't use them)
- `nn.Sigmoid()` is applied to get probabilities before metric computation
- Results are moved to CPU for the Evaluator (which uses numpy)

---

## Part B: Tester (`tester.py`)

### How Tester Differs from Trainer

The Tester rebuilds the exact same model architecture from config, then **loads saved weights** into it. Look at `_build_model` (lines 41–59):

```python
# Steps 1-2: identical to Trainer — build extractors, build RecSys, init_parameters
rec_sys.init_parameters()

# Step 3: Load saved weights
params = torch.load(self.model_load_path, map_location=self.device)
rec_sys.load_state_dict(params)
```

> **Pause and check:** Why rebuild the architecture and load weights, instead of just saving/loading the entire model with `torch.save(model)` / `torch.load(model)`?

The `state_dict` approach is the PyTorch best practice because:
1. **Portability** — `torch.save(model)` uses Python's pickle, which embeds class definitions and file paths. If you rename a file or change a class, loading breaks. `state_dict` is just a dict of tensor names → tensor values.
2. **Flexibility** — you can load weights onto a different device (`map_location`), or load partial weights for transfer learning.
3. **Reproducibility** — the architecture is defined by code + config, not by a binary blob.

**Important difference on line 79**: The Tester uses `self.model.loss_func(out, labels)` — no `.module`! That's because Tester does **not** wrap in `DataParallel` (compare Trainer line 61 vs Tester — no `nn.DataParallel` call). The Tester only runs inference, so multi-GPU parallelism isn't needed.

### `test()` vs `get_test_logits()` (lines 62–122)

`test()` computes metrics and logs to W&B — this is what runs after hyperparameter search finds the best trial.

`get_test_logits()` returns the raw per-user metric values without aggregation (`sum=False`, `aggregated=False`). This is for **post-hoc analysis** — when you want to know the NDCG for each individual user, not just the average. You'd use this for the explanation tools in Session 9.

---

## Part C: Evaluator (`utilities/eval.py`)

### Hit Ratio (lines 7–24)

```python
idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
hrs = np.any(idx_topk_part[:] == 0, axis=1).astype(int)
```

**Why `argpartition` instead of `argsort`?** Performance. `argpartition` is O(n) — it finds the k largest elements without fully sorting. `argsort` is O(n log n). When you have thousands of candidates, this matters. We negate logits (`-logits`) because `argpartition` finds the *smallest* k, so negating gives us the largest k.

**Why `== 0`?** This is the convention from the Dataset — the positive item is always at index 0. If the positive item's index (0) appears anywhere in the top-k partition, it's a hit. `np.any(..., axis=1)` checks per user.

> **Pause and check:** For a user with 100 candidate items where the positive item has the 3rd highest score, what is HR@5? HR@10?

Answer: The positive item is in the top-5, so HR@5 = 1. HR@10 = 1 too. HR is binary — either the item is in the top-k or it isn't.

### NDCG (lines 27–53)

NDCG is more nuanced — it cares *where* in the ranking the positive item appears.

```python
idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]  # Get top-k (unordered)
topk_part = logits[dummy_column, idx_topk_part]               # Get their actual scores
idx_part = np.argsort(-topk_part, axis=1)                     # Sort the top-k by score
idx_topk = idx_topk_part[dummy_column, idx_part]              # Get original indices in sorted order
```

The two-step process (partition then sort) is clever: first grab the k largest elements in O(n), then sort only those k elements in O(k log k). Much faster than sorting all n elements.

```python
rows, cols = np.where(idx_topk == 0)
ndcgs[rows] = 1. / np.log2((cols + 1) + 1)
```

`cols` is the rank position (0-indexed) of the positive item within the top-k. The NDCG formula: `1 / log2(rank + 2)` where rank is 0-indexed. So:
- Rank 0 (1st place): `1/log2(2)` = 1.0
- Rank 1 (2nd place): `1/log2(3)` ≈ 0.631
- Rank 2 (3rd place): `1/log2(4)` = 0.5
- Not in top-k: 0.0

### The Evaluator Class (lines 56–97)

The `Evaluator` is a **stateful accumulator**. It can't compute metrics on a single batch because metrics need to be averaged over *all users*, not per-batch.

```python
def eval_batch(self, out, sum=True):
    for k in K_VALUES:
        for metric_name, metric in zip(['ndcg@{}', 'hit_ratio@{}'], ...):
            self.metrics_values[metric_name.format(k)] = \
                self.metrics_values.get(metric_name.format(k), 0) + metric(out, k)
```

Each batch adds its sums to the running totals. Then:

```python
def get_results(self, aggregated=True):
    if aggregated:
        for metric_name in self.metrics_values:
            self.metrics_values[metric_name] /= self.n_users
```

At the end, divide by `n_users` to get the average. `K_VALUES` comes from `consts.py` — it's `[1, 3, 5, 10, 50]`, so you get metrics at multiple cutoff points.

---

## Key Takeaways

1. **Trainer wraps in DataParallel**, Tester doesn't — hence `.module` in Trainer but not Tester
2. **Pre-training validation** establishes a random baseline
3. **Early stopping** with patience=10 on `hit_ratio@10` prevents overfitting
4. **State dict saving** (not full model) is the checkpoint strategy — portable and flexible
5. **argpartition** gives O(n) top-k selection vs O(n log n) for full sort
6. **Evaluator accumulates** across batches, then divides by `n_users` at the end

---

## Check Your Understanding

Before moving on, make sure you can answer these:

- [ ] Why does Trainer use `self.model.module.loss_func` but Tester uses `self.model.loss_func`?
- [ ] What happens on the very first `self.val()` call before any training?
- [ ] Why is `state_dict` preferred over `torch.save(model)`?
- [ ] What's the difference between `argpartition` and `argsort`, and why does it matter?
- [ ] How does the Evaluator handle the fact that data comes in batches but metrics are per-user averages?
- [ ] For a positive item ranked 3rd out of 100 candidates: what's the HR@5? The NDCG@5?
