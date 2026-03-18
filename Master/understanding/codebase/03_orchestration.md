# Session 3: Orchestration

**Files:** `experiment_helper.py` (~161 lines)

## Overview

This file is the glue between the CLI entry point and the actual training/testing. It has four functions: `load_data()` creates train/val/test DataLoaders by calling `get_protorecdataset_dataloader` with appropriate parameters per split. `start_training()` is the function Ray Tune calls for each trial — it unpacks the config into a Namespace, loads data, sets reproducibility, creates a Trainer, and runs it. `start_testing()` does the same but for evaluation with a saved checkpoint. The main orchestration function `start_hyper()` sets up Ray Tune with HyperOpt search algorithm and optional ASHA scheduler, runs the hyperparameter sweep, extracts the best trial, and tests it. `start_multiple_hyper()` loops over 3 seeds, calls `start_hyper()` for each, and averages results to W&B. This file wires together Ray Tune, W&B logging, DataLoaders, and the Trainer/Tester classes.

Open this file side-by-side while answering.

---

### Q1: What is the difference between how `load_data` configures the train loader vs the val/test loaders?
Compare the parameters passed to `get_protorecdataset_dataloader` for training vs validation. What differs and why?

**Your answer:**
Use different split obviously. Neg_train, one the 99 and one a tunable parameter. Then the neg strategy, batch size and such, we have different args, but i am not sure we actually pass different values though.

**Evaluation:** ⚠️ Partially correct — good instinct but imprecise. Let's be specific about what actually differs:

| Parameter | Train | Val/Test |
|-----------|-------|----------|
| `split_set` | `'train'` | `'val'` / `'test'` |
| `n_neg` | `conf.neg_train` (tunable, typically 1–10) | `NEG_VAL` (fixed at 99 from consts) |
| `neg_strategy` | `conf.train_neg_strategy` | `conf.eval_neg_strategy` |
| `batch_size` | `conf.batch_size` | `conf.val_batch_size` |
| `shuffle` | `True` | not passed (defaults to `False`) |
| `prefetch_factor` | `5` | not passed (uses default) |

The key insight you got right: `n_neg` differs — training uses fewer negatives (tunable hyperparameter) because each batch needs to be fast, while val/test use 99 negatives to simulate a more realistic ranking task (rank the positive among 99 negatives). The neg strategies **can** differ — training might use popularity-based sampling while evaluation uses uniform. `shuffle=True` for training ensures random batch composition; val/test don't shuffle because order doesn't affect evaluation metrics.

---

### Q2: Why does `start_training` convert the config dict to `argparse.Namespace`?
Line 57 does `config = argparse.Namespace(**config)`. What does this enable, and why not just use the dict directly?

**Your answer:**
this one I do not know please hep me understand

**Evaluation:** ❌ Gap — here's the explanation:

Ray Tune passes the config as a **dict** (e.g., `config['batch_size']`). But the rest of the codebase (Trainer, load_data, etc.) accesses config values as **attributes** (e.g., `conf.batch_size`).

`argparse.Namespace(**config)` converts a dict into an object with attribute access:
```python
config = {'batch_size': 64, 'lr': 0.001}
config = argparse.Namespace(**config)
# Now: config.batch_size == 64, config.lr == 0.001
```

Why not just use the dict? Two reasons:
1. **Convenience** — `conf.batch_size` is cleaner than `conf['batch_size']`, and you get `AttributeError` for typos instead of `KeyError`
2. **Compatibility** — the Trainer and other components were written to expect attribute-style access (likely matching how argparse originally produces CLI arguments). Converting to Namespace bridges Ray Tune's dict format to the codebase's attribute format.

It's a small adapter pattern — nothing deep, just a convention mismatch between Ray Tune and the existing code.

---

### Q3: What is the ASHA scheduler and why is it only used for `lfm2b-1mon`?
Lines 92–95 conditionally create an `ASHAScheduler` with `grace_period=4`. What does ASHA do, what does `grace_period` mean, and why would only one dataset need it?

**Your answer:**
asha is a scheduler tht kills trials if they are unpromising. early. even if patience is not affected. just if they are not promising enough.

**Evaluation:** ✅ Correct direction, but let's sharpen it.

ASHA (Asynchronous Successive Halving Algorithm) is a **trial-level** early stopper, separate from the epoch-level patience inside the Trainer. It works across trials: after each epoch is reported to Ray Tune, ASHA compares each trial's metric against other trials' metrics at the same epoch. Unpromising trials are killed to free resources for new ones.

`grace_period=4` means: don't kill any trial before it has run for at least 4 epochs. This gives every trial a fair chance to warm up.

The distinction you made is important: **patience** (inside Trainer) stops a single trial that plateaus. **ASHA** kills trials that are performing worse *relative to other trials*. A trial might still be improving (patience not triggered) but be so far behind the best trials that continuing it is wasteful.

Why only `lfm2b-1mon`? This is the largest dataset — each trial takes much longer, so wasting GPU time on bad hyperparameter combinations is costly. For smaller datasets (ml-1m, amazon2014), trials are fast enough that it's cheaper to just let them all run to completion.

---

### Q4: Walk through the W&B logging structure.
The code creates W&B runs in two places — inside `start_hyper` (lines 98–99 and 130–134). What are these two logging contexts, and why are they separate?

**Your answer:**
please explain that to me

**Evaluation:** ❌ Gap — here's the explanation:

There are **two separate W&B logging contexts** in `start_hyper`:

**Context 1 (lines 98–99)** — `WandbLoggerCallback`:
```python
callback = WandbLoggerCallback(project=PROJECT_NAME, ..., job_type='train/val',
                                tags=[model, str(seed), dataset])
```
This is a Ray Tune callback. It creates a **separate W&B run for each trial** during hyperparameter search. Each trial logs its per-epoch metrics (train loss, val metrics) to its own W&B run. This lets you compare all trials' learning curves on the W&B dashboard. The `job_type='train/val'` tag helps you filter these runs.

**Context 2 (lines 130–134)** — manual `wandb.init`:
```python
wandb.init(project=PROJECT_NAME, group='test_results', config=best_trial_config,
           name=group_name, job_type='test', tags=[model, str(seed), dataset])
metric_values = start_testing(best_trial_config, best_trial_checkpoint)
wandb.finish()
```
After all trials complete and the best is selected, this creates a **single W&B run** for the **test results** of the winning trial. It logs the final test metrics under `job_type='test'` and `group='test_results'`.

Why separate?
- Trial runs are many (one per hyperparameter sample) and logged automatically by Ray Tune
- The test run is one per seed, logged manually with the best config
- On the W&B dashboard, you can filter by `job_type` to see only training curves or only final results
- The test run also stores `best_trial_config` so you can see which hyperparameters won

There's also a **third context** in `start_multiple_hyper` (lines 156–160): `group='aggr_results'` logs the mean across seeds — one run summarizing the entire experiment.

---

### Q5: How does Ray Tune know which metric to optimize?
Find where the metric name is specified, how it gets reported from the training loop, and how Ray Tune uses it to select the best trial.

**Your answer:**
OPTIIZER_METRIC and it uses it to slelect the best model for the best perofrming trial. it comes from the consts we looked at before

**Evaluation:** ⚠️ Right idea, but missing the chain. Let's trace it:

1. **Definition**: `OPTIMIZING_METRIC = 'hit_ratio@10'` in `utilities/consts.py`
2. **Reporting**: In `Trainer.run()` line 126: `tune.report({**metrics_values, 'epoch_train_loss': ...})` — after each epoch, the metrics dict (which includes `hit_ratio@10`) is reported to Ray Tune
3. **Ray Tune receives it** with a `_metric/` prefix, so it becomes `_metric/hit_ratio@10`
4. **Selection** in `start_hyper`:
   - Line 122: `metric='_metric/' + OPTIMIZING_METRIC` — tells `tune.run()` what to optimize
   - Line 123: `mode='max'` — higher is better
   - Line 126: `analysis.get_best_trial(metric_name, 'max', scope='all')` — finds the trial that achieved the highest `_metric/hit_ratio@10` across all epochs

So the full chain is: `consts.py` defines it → `Trainer` reports it per epoch → `tune.run()` uses it to guide search → `get_best_trial()` uses it to pick the winner.

---

### Q6: What happens between finding the best trial and testing it (lines 125–134)?
After `analysis.get_best_trial()`, several things happen before `start_testing()`. Walk through each step and explain why it's needed.

**Your answer:**
I seem to be confused what is the difference between trials and tests.

**Evaluation:** ❌ Important gap — let's clarify the terminology first, then walk through the code.

**Trial vs Test:**
- A **trial** is one run of the training loop with a specific set of hyperparameters. Ray Tune launches many trials (e.g., 50), each with different hyperparameters sampled from the search space. Each trial trains a model, validates it, and saves a checkpoint. Trials are about **finding the best hyperparameters**.
- **Testing** happens once, after the best trial is identified. You take the best trial's saved model and evaluate it on the **test set** (held-out data never seen during training or validation). This is the final performance number you report in the paper.

Now the code (lines 125–134):

```python
# Line 125-126: Find the winning trial
metric_name = '_metric/' + OPTIMIZING_METRIC
best_trial = analysis.get_best_trial(metric_name, 'max', scope='all')

# Line 127: Extract its hyperparameter config
best_trial_config = best_trial.config

# Line 128: Find the path to its saved model checkpoint
best_trial_checkpoint = os.path.join(
    analysis.get_best_checkpoint(best_trial, metric_name, 'max'), 'best_model.pth')

# Lines 130-132: Create a W&B run specifically for test results
wandb.login(key=WANDB_API_KEY)
wandb.init(project=PROJECT_NAME, group='test_results', config=best_trial_config,
           name=group_name, job_type='test', tags=[model, str(seed), dataset])

# Line 133: Run the actual test — rebuild model, load weights, evaluate on test set
metric_values = start_testing(best_trial_config, best_trial_checkpoint)

# Line 134: Close the W&B run
wandb.finish()
```

Each step is needed because:
- You need the **config** to rebuild the exact same model architecture
- You need the **checkpoint path** to load the trained weights
- The **W&B run** logs the final test metrics separately from training metrics
- `start_testing` creates a `Tester` that reconstructs the model and evaluates on test data

---

### Q7: In `start_multiple_hyper`, how are results aggregated across seeds?
Look at lines 147–153. What exactly is averaged, and what gets logged to W&B? Is there any information lost in this aggregation?

**Your answer:**
we take the mean of the metric over the trials

**Evaluation:** ⚠️ Correct idea but imprecise — "over the trials" should be "over the seeds."

Each seed runs a **complete** `start_hyper()` — its own hyperparameter search + testing. So `metric_values_list` contains 3 dicts (one per seed), each with test metrics from that seed's best trial.

```python
for key in metric_values_list[0].keys():      # Iterate over metric names
    _sum = 0
    for metric_values in metric_values_list:   # Sum across 3 seeds
        _sum += metric_values[key]
    _mean = _sum / len(metric_values_list)     # Average
    mean_values[key] = _mean
```

What's averaged: **all** test metrics — `hit_ratio@1`, `hit_ratio@3`, ..., `ndcg@50`, `test_loss`. Each is the mean of 3 independent runs.

What's lost: **variance**. You know the mean HR@10 across seeds, but not the spread. If one seed got 0.7 and another got 0.3, the mean 0.5 hides that instability. For a paper, you'd typically want to report mean ± std. The individual seed results are still in W&B (from the per-seed test runs), but the aggregated run only logs the means.

---

### Q8: If `start_training` raises an exception mid-trial, what happens?
Consider: does Ray Tune retry? Does W&B get a dangling run? Does the next trial start? What about `wandb.finish()` on line 68?

**Your answer:**
Please give me the answer

**Evaluation:** ❌ Gap — here's what happens:

1. **`wandb.finish()` on line 68 is skipped** — the exception exits `start_training` before reaching it. This leaves a **dangling W&B run** in "running" state (it'll eventually time out and show as "crashed" on the dashboard).

2. **Ray Tune catches the exception** — by default, Ray Tune marks the trial as `ERROR` and continues to the next trial. It does not retry by default (you'd need `max_failures > 0` in `tune.run()` for retries, which isn't set here).

3. **The next trial starts normally** — Ray Tune's trial runner is isolated. One trial crashing doesn't affect others.

4. **The errored trial can't be selected as best** — `get_best_trial` ignores errored trials.

5. **Overall risk**: if many trials error out, you end up with fewer valid trials to choose from, potentially a worse hyperparameter search. The dangling W&B runs are cosmetic clutter but not harmful.

This is why robust error handling matters less here — Ray Tune provides trial-level fault tolerance out of the box.

---

## Synthesis

### Q9: To add a new dataset, which parts of `experiment_helper.py` would need changes?
Assume the data files are already prepared. Would any function signatures change? Any new parameters needed?

**Your answer:**
*(unanswered)*

**Evaluation:** Here's the answer:

**Good news: almost nothing needs to change in this file.** The design is already dataset-agnostic.

- `load_data()` — **No changes.** It uses `conf.data_path` which is assembled on line 106 as `DATA_PATH/dataset_name`. As long as the new dataset's files are in `data/hm/` and follow the same CSV format, `get_protorecdataset_dataloader` will find them.
- `start_hyper()` — **Possibly one change:** the ASHA scheduler condition on line 92 checks `if dataset == 'lfm2b-1mon'`. If H&M is large enough to benefit from ASHA, you'd add `or dataset == 'hm'` here. Otherwise, no changes.
- Function signatures — **No changes.** The `dataset` string flows through as a tag for W&B and for path construction, both of which work with any string.

The changes needed are in **other files**: `start.py` (add to argparse choices), `hyper_params.py` (add search space), and possibly `consts.py`. But `experiment_helper.py` is effectively dataset-agnostic already.

---

## Session Summary

| Question | Rating | Notes |
|----------|--------|-------|
| Q1 Train vs val loader | ⚠️ | Right direction on n_neg, imprecise on other differences |
| Q2 Namespace conversion | ❌ | Didn't know — dict → attribute access adapter |
| Q3 ASHA scheduler | ✅ | Good understanding, needed sharpening on grace_period and "relative to other trials" |
| Q4 W&B structure | ❌ | Asked for explanation — two contexts: per-trial callback + per-seed test run |
| Q5 Metric optimization | ⚠️ | Knew the source, missed the reporting chain |
| Q6 Best trial to test | ❌ | Confused trials vs tests — key conceptual gap resolved above |
| Q7 Multi-seed aggregation | ⚠️ | Right idea, imprecise on "seeds" vs "trials", missed variance loss |
| Q8 Error handling | ❌ | Asked for answer — dangling W&B, Ray Tune continues, trial marked ERROR |
| Q9 Adding a dataset | ❌ | Unanswered — almost nothing changes in this file |

**Key gaps to revisit:**
- The trial/test distinction (Q6) is fundamental — make sure this is clear before moving on
- The W&B logging structure (Q4) matters for when you start running experiments
- The Namespace conversion (Q2) is minor but comes up often in Python codebases
