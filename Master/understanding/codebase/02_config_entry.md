# Session 2: Config & Entry

**Files:** `utilities/consts.py` (~23 lines), `confs/hyper_params.py` (~165 lines), `start.py` (~51 lines)

## Overview

These three files form the experiment configuration and launch pipeline. `consts.py` is the central constants file — it defines seeds, resource allocation (GPU/CPU per trial), training patience, evaluation K values, the optimizing metric (`hit_ratio@10`), and loads the W&B API key from a sensitive file. `hyper_params.py` defines Ray Tune search spaces for each model variant. Each config inherits from `base_hyper_params` (which sets shared tunable ranges for negatives, loss, batch size, optimizer) and adds model-specific parameters like `n_prototypes` or regularization weights. `start.py` is the CLI entry point — it parses arguments (`-m model`, `-d dataset`, `-s seed`, `-mp` for multi-seed), maps the model name to the right hyperparameter dict, and dispatches to the experiment helper.

Open these files side-by-side while answering.

---

## Part A: Constants (`consts.py`)

### Q1: What is `NEG_VAL` and where is it used vs `neg_train`?
`consts.py` defines `NEG_VAL = 99`. How does this differ from the `neg_train` parameter in `hyper_params.py`? Why would you want different numbers of negatives for training vs evaluation?

**Your answer:**
please explain

**Evaluation:** `NEG_VAL = 99` is the **fixed** number of negative samples used during validation and testing. `neg_train` (a `tune.randint(1, 50)`) is the **tunable** number used during training. They're different because:
- **Training:** fewer negatives = faster epochs, and the optimal number is model-dependent (hence tunable). Too many wastes compute, too few under-trains.
- **Evaluation:** you always want the same number (99) so metrics are comparable across trials and models. With 99 negatives + 1 positive = 100 candidates, the model must rank the positive item among 100 options. This is a standardized evaluation protocol from the RecSys literature.

Look at `experiment_helper.py` lines 35–36 vs 22–24 to see where each is used.

---

### Q2: Why is `OPTIMIZING_METRIC` set to `hit_ratio@10` rather than, say, `ndcg@10`?
This is the metric used for model selection during hyperparameter optimization. What does this choice mean for what the model is optimized toward? Would switching to NDCG change what kinds of models are preferred?
this means that it the model optimized for generally positive in the top 10, not better the higher they are. it would change it. I dont know why the authors thought hit ratio was more important
**Your answer:**

**Evaluation:** ✅ Good intuition. Hit Ratio@10 is binary — "is the positive item in the top 10?" (1 or 0). NDCG@10 is graded — it rewards placing the positive item *higher* within the top 10 (position 1 is worth more than position 8). So HR optimizes for *presence* in the list, NDCG for *ranking quality*.

As for why the authors chose HR: it's a common choice in the leave-one-out evaluation protocol (which this codebase uses). With only 1 positive per user, HR@10 is a natural binary success metric. NDCG would also work but adds sensitivity to position — the authors likely found HR@10 gave more stable model selection across trials. It's a practical choice, not a strong theoretical one.

---

### Q3: What would happen if you changed `GPU_PER_TRIAL` from 0.2 to 1.0?
Think about what Ray Tune does with this value and how it affects parallelism during hyperparameter search.

**Your answer:**
it means per trial it would use full gpu instead of just 20 percent of it. This means essentially only 1 trial can happen at a time, instead of now 5 I assume?

**Evaluation:** ✅ Correct. Ray Tune uses this as a resource allocation signal — with 0.2, it can schedule up to 5 trials concurrently on one GPU. At 1.0, only 1 trial at a time. These RecSys models are small enough that 20% GPU memory is plenty per trial, so 0.2 is a smart choice to maximize parallelism during the 100-sample hyperparameter search. On CPU (`GPU_PER_TRIAL` effectively ignored), parallelism is governed by `CPU_PER_TRIAL=1` instead.

---

## Part B: Hyperparameter Search Spaces (`hyper_params.py`)

### Q4: Explain the inheritance pattern — how does `mf_hyper_params` relate to `base_hyper_params`?
What does the `**base_hyper_params` unpacking do, and what does `mf_hyper_params` add on top? Why is this a reasonable design?

**Your answer:**
base parameter is basically the static ones, base hyper parameter, are those speccified in ranges, where the best values are supposed to be found.

**Evaluation:** ⚠️ Partially right — you're confusing `base_param` with `base_hyper_params`. There are actually two levels:
- `base_param` (line 5): **static** values — device, n_epochs, eval strategy, val_batch_size, use_bias. No `tune.*` calls.
- `base_hyper_params` (line 13): includes `**base_param` **plus tunable ranges** — neg_train, loss function, batch_size, optimizer params. These are shared across all models.
- `mf_hyper_params` (line 26): includes `**base_hyper_params` **plus model-specific** config — `loss_func_aggr` and the `ft_ext_param` dict (which specifies `ft_type: "detached"` and both extractors as plain embeddings with tunable `embedding_dim`).

The `**` dict unpacking merges all parent keys into the child. Later keys override earlier ones. It's a simple inheritance pattern — shared config lives in one place, model-specific additions are minimal.

---

### Q5: In the `user_proto` config, the item extractor is just `"embedding"` with no `embedding_dim`. Where does its dimension come from?
Look at `user_proto_chose_original_hyper_params`. The user side has `n_prototypes` and `embedding_dim`, but the item side only says `ft_type: "embedding"`. How does the item embedding know what dimension to use, and why must it match a specific value?

**Your answer:**
it must match the amount of prototypes on the user side

**Evaluation:** ✅ Correct. The dimension matching happens in the **factory**, not in the config. Look at `feature_extractor_factories.py` lines 43–50: it reads `user_n_prototypes` and passes that as the `embedding_dim` when creating the item embedding. This is because the dot product in `RecSys.forward()` requires both sides to output vectors of the same length. User prototypes output a vector of length `n_prototypes` (similarity to each prototype), so the item embedding must also have dimension `n_prototypes`. The config doesn't need to specify it because the factory handles the cross-matching.

---

### Q6: What is the difference between `tune.randint`, `tune.loguniform`, and `tune.lograndint`?
The config uses all three. Why is `batch_size` sampled with `lograndint` while `neg_train` uses `randint`? Why are regularization weights sampled with `loguniform`?

**Your answer:**
I do not know please explain.

**Evaluation:** These control how Ray Tune samples hyperparameter values:
- **`tune.randint(1, 50)`**: uniform random integers. Every value from 1–49 is equally likely. Used for `neg_train` because there's no reason to think 5 negatives is more likely optimal than 25.
- **`tune.loguniform(1e-4, 1e-2)`**: continuous values sampled uniformly on a **log scale**. This means 1e-4 to 1e-3 gets as many samples as 1e-3 to 1e-2. Used for learning rates, weight decay, and regularization weights because these parameters typically matter across orders of magnitude (0.001 vs 0.01 is a bigger deal than 0.051 vs 0.052).
- **`tune.lograndint(64, 512, 2)`**: integers sampled on a log scale (base 2). This favors values like 64, 128, 256 over 100, 200, 300. Used for batch size because powers of 2 are standard (GPU memory alignment) and the difference between batch 64 and 128 matters more than between 400 and 464.

Key intuition: use **log** sampling when the parameter spans orders of magnitude and relative changes matter more than absolute changes.

---

### Q7: Compare `mf_hyper_params` and `anchor_hyper_params` — why does ACF use `loss_func_aggr: 'sum'` while MF uses `'mean'`?
What effect does changing the loss aggregation have on the gradient magnitudes, and why might ACF need `sum`?

**Your answer:**
Also please explain.

**Evaluation:** `'mean'` divides the total loss by the number of elements; `'sum'` doesn't. So `sum` produces larger gradients by a factor of ~batch_size.

Why ACF uses `sum`: ACF has its own regularization losses (exclusiveness and inclusiveness) that are already summed over the batch and weighted by `delta_exc`/`delta_inc`. If you used `'mean'` for the recommendation loss, the rec loss gradients would be ~batch_size times smaller than the regularization gradients, making the regularization terms dominate. Using `'sum'` keeps the recommendation loss and regularization loss on comparable scales.

MF has no regularization loss from the extractors (plain embeddings return 0), so `'mean'` works fine and produces more stable, size-independent gradients.

This is a practical balancing act — the loss aggregation choice interacts with the regularization weights.

---

## Part C: Entry Point (`start.py`)

### Q8: What do the two environment variables at the top of `start.py` do?
Lines 9–10 set `WANDB_START_METHOD` and `OMP_NUM_THREADS`. Why are these set here and what would go wrong without them?

**Your answer:**
I dont know please explain

**Evaluation:**
- **`WANDB_START_METHOD = 'thread'`**: W&B normally starts a background **process** for logging. But Ray Tune already spawns multiple processes (one per trial). Launching sub-processes inside Ray Tune workers can cause deadlocks or crashes on some systems. Setting this to `'thread'` makes W&B use a thread instead, which plays nicely with Ray Tune's process model.
- **`OMP_NUM_THREADS = '1'`**: Limits OpenMP (used by PyTorch/NumPy for CPU parallelism) to 1 thread per process. Without this, each Ray Tune trial would try to use all CPU cores for internal operations (matrix multiplies, etc.), and with 5 trials running simultaneously, they'd fight over cores — causing *slower* execution due to thread contention. Setting it to 1 lets Ray Tune manage parallelism at the trial level instead.

Both are set early in `start.py` (before any imports that would read them) to ensure they take effect globally.

---

### Q9: If you wanted to add H&M as a new dataset, what changes would `start.py` need?
Walk through the code and identify every place that would need modification to support `-d hm`.

**Your answer:**
it would jsut need to be added to choices. and matching with a splitterfile, such that it implements the protodataset interface

**Evaluation:** ✅ Correct for `start.py` specifically — the only change is adding `'hm'` to the `choices` list on line 18. The splitter is a separate concern (runs independently before training), and the rest of the pipeline is dataset-agnostic — it just needs the CSVs in `data/hm/` with the right format.

One nuance you might have missed: `experiment_helper.py` line 92 has a special case `if dataset == 'lfm2b-1mon'` for the ASHA scheduler. You'd need to decide whether H&M needs this too (it would if training is slow, since H&M is a large dataset).

---

## Synthesis

### Q10: Trace how a CLI command becomes a Ray Tune config.
Starting from `python start.py -m item_proto -d ml-1m -s 42`, list each transformation the configuration undergoes before it reaches the training function. What format is it in at each step?

**Your answer:**
also please explain these questions are too detailed. not relevant i need to understand these things, and know them but i cannot look all that up in detail so much, I do not have the time

**Evaluation:** Fair point — I'll keep synthesis questions shorter in future sessions. Here's a quick summary for reference:

1. **CLI** → `argparse` extracts `model="item_proto"`, `dataset="ml-1m"`, `seed=42`
2. **start.py** → if/elif maps `"item_proto"` → `item_proto_chose_original_hyper_params` (a **dict** with `tune.*` objects as values)
3. **experiment_helper.start_hyper()** → adds `data_path` and `seed` keys to that dict, passes it as `config` to `tune.run()`
4. **Ray Tune** → for each of 100 trials, **resolves** all `tune.*` objects to concrete values (e.g., `tune.randint(10,100)` → `47`), producing a plain dict
5. **start_training()** → converts the plain dict to `argparse.Namespace` for dot-access (`config.batch_size` instead of `config['batch_size']`)
6. **Trainer** → reads from the Namespace to build the model and optimizer

---

## Session Summary

| Question | Rating | Notes |
|----------|--------|-------|
| Q1 NEG_VAL vs neg_train | — | Explained: fixed eval vs tunable training negatives |
| Q2 Optimizing metric | ✅ | Good intuition on HR vs NDCG distinction |
| Q3 GPU_PER_TRIAL | ✅ | Correct — 5 parallel trials down to 1 |
| Q4 Config inheritance | ⚠️ | Confused `base_param` and `base_hyper_params` — two levels |
| Q5 Dimension matching | ✅ | Correct — factory handles cross-matching |
| Q6 Tune distributions | — | Explained: uniform vs log-scale sampling |
| Q7 sum vs mean | — | Explained: loss scale balancing with regularization |
| Q8 Env variables | — | Explained: thread safety and CPU contention |
| Q9 Adding H&M | ✅ | Correct for start.py; noted ASHA scheduler nuance |
| Q10 CLI to config trace | — | Explained (simplified); feedback noted on question depth |

**Key takeaways:**
- You understood the high-level design well (metric choice, GPU allocation, dimension matching, dataset extension)
- Gap on `base_param` vs `base_hyper_params` — worth a quick re-read of lines 5–24 in `hyper_params.py`
- The log-scale sampling concept is important for Phase 4 when you design your own search spaces
- Environment variable setup is "set and forget" but good to know why it's there
