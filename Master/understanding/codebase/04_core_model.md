# Session 4: Core Model — Guided Walkthrough

**File:** `rec_sys/rec_sys.py` (~194 lines)

Open this file now. We'll walk through it top to bottom.

---

## The Big Picture

`RecSys` is the central PyTorch module — the "brain" of the system. It doesn't know or care whether it's doing plain matrix factorization or prototypes. It just takes a user extractor and an item extractor (pluggable), runs them, and computes a dot product. Simple and clean.

The file also defines three standalone loss functions at the bottom: BCE, BPR, and sampled softmax.

---

## Part A: The RecSys Module

### Step 1: Constructor (`__init__`, lines 11–60)

Look at lines 11–12. The constructor takes `user_feature_extractor` and `item_feature_extractor` as arguments — both typed as `FeatureExtractor`. This is the key design choice: RecSys is **agnostic** to what kind of extractor it gets. It could be a plain embedding, a prototype model, or ACF — RecSys doesn't care.

Now look at **line 36**:
```python
self.use_bias = self.rec_sys_param["use_bias"] > 0 if 'use_bias' in self.rec_sys_param else True
```
This checks `> 0` rather than a simple boolean. Why? Because in the hyperparameter search, `use_bias` comes from Ray Tune as a sampled value (e.g., `tune.choice([0, 1])`). Using `> 0` lets the hyperparameter be an integer that Ray Tune can search over, rather than a boolean which is harder to express in search spaces.

**Lines 38–41** conditionally create bias embeddings. If bias is enabled, every user gets a scalar bias, every item gets a scalar bias, and there's one global bias. These are additive corrections on top of the dot product — a standard MF technique.

**Lines 43–50**: The loss function is selected at init time using `functools.partial`. This is a clean pattern — `partial(bce_loss, aggregator='mean')` creates a new function that's just `bce_loss` with the aggregator already filled in. The training loop can then call `self.rec_loss(logits, labels)` without knowing which loss it's using.

**Line 52**: `self.initialized = False`. This is a safety guard.

> **Pause and check:** Look at line 62–73 (`init_parameters`). What does it do, and why is it separate from `__init__`? Think about *when* this gets called in the pipeline.

The answer: `init_parameters()` initializes the actual weight values (biases to 0, extractors get their own init). It's separate because PyTorch modules need to exist first (created in `__init__`), and then weights can be initialized. In this codebase, the `Trainer._build_model()` calls `init_parameters()` after constructing the RecSys (see `trainer.py:60`). The `initialized` flag on line 52 prevents anyone from accidentally running `forward()` on an uninitialized model — the assert on line 99 catches this.

### Step 2: The Forward Pass (lines 89–120)

This is where the magic happens. Let's trace it with concrete shapes.

Assume batch_size=64, n_neg=10 (so 1 positive + 10 negatives = 11 items per user).

**Line 103**: `u_embed = self.user_feature_extractor(u_idxs)`
- `u_idxs` shape: `[64]` (just user IDs)
- `u_embed` shape: `[64, D]` where D depends on the extractor (could be `n_prototypes` or `embedding_dim`)

**Line 111**: `i_embed = self.item_feature_extractor(i_idxs)`
- `i_idxs` shape: `[64, 11]` (each user has 11 candidate items)
- `i_embed` shape: `[64, 11, D]` (each item gets a D-dimensional vector)

**Line 114** — the critical line:
```python
dots = torch.sum(u_embed.unsqueeze(1) * i_embed, dim=-1)
```

Let's break this down:
1. `u_embed.unsqueeze(1)` → shape `[64, 1, D]` (adds a dimension for broadcasting)
2. `* i_embed` → broadcasting multiplies `[64, 1, D]` × `[64, 11, D]` = `[64, 11, D]` (element-wise)
3. `torch.sum(..., dim=-1)` → sums over the last dimension (D) → shape `[64, 11]`

This is a **batched dot product**. Each user's vector is dotted with each of their 11 candidate items. The `unsqueeze(1)` is necessary because the user has one vector but needs to be compared against 11 items — broadcasting handles this.

> **Pause and check:** The output `dots` has shape `[64, 11]`. What does `dots[3, 0]` represent? What about `dots[3, 7]`?

Answer: `dots[3, 0]` is the predicted score for user 3 and their **positive** item (always at index 0, the convention from the Dataset). `dots[3, 7]` is the score for user 3 and their 7th negative item.

**Lines 116–118**: If bias is enabled, add user bias + item bias + global bias to the dot products. These are learned scalar offsets — a user who rates everything high gets a positive user bias, a popular item gets a positive item bias.

### Step 3: The Loss Function (`loss_func`, lines 75–87)

```python
rec_loss = self.rec_loss(logits, labels)
item_feat_ext_loss = self.item_feature_extractor.get_and_reset_loss()
user_feat_ext_loss = self.user_feature_extractor.get_and_reset_loss()
return rec_loss + item_feat_ext_loss + user_feat_ext_loss
```

This is where the total loss is assembled. Three components:
1. **Recommendation loss** (BCE, BPR, or softmax) — how well the model ranks positive over negative
2. **Item extractor loss** — regularization from the item side (0 for plain embeddings, non-zero for prototypes/ACF)
3. **User extractor loss** — same for the user side

The `get_and_reset_loss()` pattern is important: during `forward()`, extractors like `PrototypeEmbedding` *accumulate* regularization losses internally. Then `loss_func` retrieves them and resets the counter to 0. The "reset" prevents double-counting across batches.

For a plain `Embedding` extractor, `get_and_reset_loss()` always returns 0 (it never modifies `cumulative_loss`), so the total loss is just the recommendation loss. For `PrototypeEmbedding`, it returns the weighted sum of prototype regularization losses.

---

## Part B: The Three Loss Functions

All three follow the same convention: logits shape `[batch_size, 1+n_neg]`, positive item always in column 0.

### BCE Loss (lines 123–143)

```python
weights = torch.ones_like(logits)
weights[:, 0] = logits.shape[1] - 1
```

The **weights** tensor solves a class imbalance problem. With 10 negatives per positive, the loss would be dominated by negatives (10× more negative terms). By giving the positive column weight `n_neg` (=10), the positive item counts as much as all negatives combined. This is equivalent to the formula in the docstring: `-n_neg * log(σ(x_pos)) - Σ log(1 - σ(x_neg))`.

Then it flattens everything and uses PyTorch's built-in `BCEWithLogitsLoss` (which applies sigmoid internally).

### BPR Loss (lines 146–167)

BPR (Bayesian Personalized Ranking) doesn't predict absolute scores — it predicts **pairwise preference**.

```python
pos_logits = logits[:, 0].unsqueeze(1)   # [batch_size, 1]
neg_logits = logits[:, 1:]               # [batch_size, n_neg]
diff_logits = pos_logits - neg_logits     # [batch_size, n_neg] via broadcasting
```

The model is trained to make `score(positive) - score(negative) > 0`. The `BCEWithLogitsLoss` on the difference acts like: "the probability that the positive is preferred should be high." The labels are all 1s (we always want positive > negative).

> **Pause and check:** What's the fundamental philosophical difference between BCE and BPR? BCE optimizes absolute predictions ("this item has score 0.9"), while BPR optimizes relative rankings ("this item is better than that one"). BPR doesn't care about the absolute scores, only the ordering.

### Sampled Softmax Loss (lines 170–193)

```python
pos_logits_sum = - logits[:, 0]
log_sum_exp_sum = torch.logsumexp(logits, dim=-1)
sampled_loss = pos_logits_sum + log_sum_exp_sum
```

This is the formula: `-x_pos + log(Σ exp(x_j))` where j goes over all candidates (positive + negatives).

The negation of the positive term is because this derives from cross-entropy. In standard softmax cross-entropy for the correct class: `-log(exp(x_pos) / Σ exp(x_j))` = `-x_pos + log(Σ exp(x_j))`. It's asking the model to make the positive item's score dominate the softmax distribution.

Notice the default aggregator is `'sum'` here, while BCE/BPR default to `'mean'`. This is because softmax already normalizes across the candidate set internally, so using `mean` on top would double-normalize and make gradients very small. The `sum` preserves the gradient scale.

---

## Key Takeaways

1. **RecSys is extractor-agnostic** — it just does dot products on whatever vectors the extractors produce
2. **The loss is composite** — recommendation loss + extractor regularization losses, cleanly separated
3. **Three loss functions** exist to give flexibility during hyperparameter search — BCE (pointwise), BPR (pairwise), softmax (listwise)
4. **The positive-at-index-0 convention** flows from Dataset through loss functions — it's a system-wide contract
5. **`functools.partial`** decouples loss selection from the training loop

---

## Check Your Understanding

Before moving on, make sure you can answer these:

- [ ] Why does `use_bias` use `> 0` instead of a boolean check?
- [ ] What are the shapes at each step of the dot product computation (line 114)?
- [ ] What would `get_and_reset_loss()` return for a plain Embedding extractor?
- [ ] What's the conceptual difference between BCE, BPR, and sampled softmax?
- [ ] If you wanted to add a new loss function (e.g., triplet margin loss), what would you change? (Hint: add the function, add an `elif` in the constructor, add the option in `hyper_params.py`)
