# Session 6: Feature Extractors (Base) — Guided Walkthrough

**File:** `feature_extraction/feature_extractors.py` — first ~189 lines (up to `AnchorBasedCollaborativeFiltering`)

Open the file now. We'll cover the abstract base class, then the two simple extractors, then ACF.

---

## The Big Picture

This file defines the **feature extractor hierarchy** — the pluggable components that turn user/item IDs into vector representations. Think of it as a menu of options for how to represent users and items. The simplest option is a raw embedding lookup; the most complex (covered next session) is the prototype model.

---

## Part A: The Abstract Base Class (`FeatureExtractor`, lines 9–39)

Look at lines 9–17:
```python
class FeatureExtractor(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.cumulative_loss = 0.
        self.name = "FeatureExtractor"
```

This is the interface contract. Every extractor must be a PyTorch module and implement three things:

1. **`forward(o_idxs)`** (line 34–39, abstract) — takes object indices, returns vectors
2. **`init_parameters()`** (lines 19–23) — initializes weights (default: do nothing)
3. **`get_and_reset_loss()`** (lines 25–32) — returns accumulated regularization loss, then resets to 0

The `cumulative_loss` field (line 16) is the key design element. Simple extractors like `Embedding` never touch it — they have no regularization, so `get_and_reset_loss()` always returns 0. Complex extractors like `PrototypeEmbedding` and ACF accumulate regularization losses during `forward()`, and this method drains them.

> **Pause and check:** Why "accumulate and drain" instead of just returning the loss directly from `forward()`? Because `forward()` returns the embeddings — the loss is a side channel. The caller (`RecSys.loss_func`) retrieves it separately. And accumulation handles the case where `forward()` is called multiple times before the loss is collected (e.g., in DataParallel).

---

## Part B: Embedding (lines 42–77)

The simplest extractor. Look at the constructor:

```python
self.embedding_layer = nn.Embedding(self.n_objects, self.embedding_dim, max_norm=self.max_norm)
```

**Line 62**: One `nn.Embedding` — a lookup table of shape `[n_objects, embedding_dim]`. Each user or item gets a learnable vector.

**`init_parameters` (line 69–70)**:
```python
def init_parameters(self):
    self.embedding_layer.apply(general_weight_init)
```
`apply()` is a PyTorch method that recursively applies a function to all submodules. For an `nn.Embedding`, `general_weight_init` (from `utilities/utils.py`) applies Xavier normal initialization to the weights. Using `apply` instead of calling init directly makes it work uniformly across module hierarchies.

**`forward` (lines 72–77)**:
```python
def forward(self, o_idxs):
    embeddings = self.embedding_layer(o_idxs)
    if self.only_positive:
        embeddings = torch.absolute(embeddings)
    return embeddings
```

Just a lookup. If `only_positive=True`, all embedding values are forced non-negative via absolute value. This constrains the representation space — useful when cosine similarity needs non-negative inputs to have a predictable range. In practice, this flag isn't commonly used in the default configs.

Notice: no `cumulative_loss` is ever modified here. So `get_and_reset_loss()` from the base class always returns 0. A plain embedding has no regularization.

---

## Part C: EmbeddingW (lines 80–114)

`EmbeddingW` extends `Embedding` by adding a linear projection after the lookup:

```python
class EmbeddingW(Embedding):
    def __init__(self, ...):
        super().__init__(n_objects, embedding_dim, max_norm)  # Creates the embedding
        self.linear_layer = nn.Linear(self.embedding_dim, self.out_dimension, bias=self.use_bias)
```

**Line 102**: A linear layer projects from `embedding_dim` to `out_dimension`.

**`forward` (lines 112–114)**:
```python
def forward(self, o_idxs):
    o_embed = super().forward(o_idxs)      # Embedding lookup → [batch, embedding_dim]
    return self.linear_layer(o_embed)       # Linear projection → [batch, out_dimension]
```

> **Pause and check:** When would you need this instead of a plain `Embedding`? The answer is the `prototypes_double_tie` variant (user_item_proto model).

Here's why: In user_item_proto, each side has both a prototype branch and an embedding branch, and they share the same underlying embedding weights. The prototype branch outputs `n_prototypes` dimensions. The embedding branch needs to output a *different* number of dimensions (the other side's `n_prototypes`). But if they share weights, the raw embedding dimension is fixed. The linear layer in `EmbeddingW` projects from the shared `embedding_dim` to whatever output dimension is needed. Without it, dimension mismatch.

For example: user prototypes = 20, item prototypes = 30, shared embedding dim = 50.
- User proto branch: lookup → [50] → cosine with 20 prototypes → [20]
- User embed branch: lookup → [50] → linear → [30] (must match item's 30 prototypes)
- Both use the same [50]-dimensional embedding weights, but output different dimensions.

---

## Part D: AnchorBasedCollaborativeFiltering (lines 117–188)

ACF is the baseline from Barkan et al. (CIKM 2021). It's a precursor to prototypes — understanding it makes ProtoMF's innovation clearer.

### Constructor (lines 122–150)

```python
self.anchors = anchors                    # Shared nn.Parameter, shape [n_anchors, embedding_dim]
self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)
self._acc_exc = 0.
self._acc_inc = 0.
```

**Line 136**: The `anchors` parameter is an `nn.Parameter` passed in from outside — shared between user and item ACF instances (both reference the same tensor). This is different from prototypes, where each side has its own.

### Forward Pass (lines 156–183)

Let's trace with batch_size=64, embedding_dim=50, n_anchors=20:

```python
o_embed = self.embedding_ext(o_idxs)        # [64, 50] — standard embedding lookup
o_dots = o_embed @ self.anchors.T            # [64, 20] — dot product with each anchor
o_coeff = nn.Softmax(dim=-1)(o_dots)         # [64, 20] — soft assignment weights (sum to 1)
o_vect = o_coeff @ self.anchors              # [64, 50] — weighted sum of anchor vectors
```

The conceptual flow:
1. Look up the object's raw embedding
2. Compute how similar it is to each anchor (dot product)
3. Turn similarities into weights via softmax (each object is a soft mixture of anchors)
4. Reconstruct the object as a **weighted sum of anchor vectors**

The output `o_vect` is back in the original embedding space (`embedding_dim`), not in anchor space. This is a key difference from prototypes.

### Regularization: Exclusiveness and Inclusiveness (lines 173–181)

```python
# Exclusiveness — entropy of each object's assignment
exc = - (o_coeff * torch.log(o_coeff)).sum()

# Inclusiveness — entropy of the aggregate assignment distribution
q_k = o_coeff.reshape(-1, self.n_anchors).sum(axis=0).div(o_coeff.sum())
inc = - (q_k * torch.log(q_k)).sum()
```

**Exclusiveness** computes the entropy of each object's softmax distribution over anchors. Low entropy = each object strongly associates with one anchor. The `get_and_reset_loss` on line 188 adds `+delta_exc * exc`, which *minimizes* high entropy → encourages **peaky assignments** (each object "belongs to" one anchor).

**Inclusiveness** computes the entropy of the *aggregate* anchor usage. `q_k` is the average softmax weight per anchor across the batch. High entropy = all anchors are used equally. The loss adds `-delta_inc * inc`, which *maximizes* entropy → encourages **even load across anchors** (no anchor goes unused).

These two forces work in tension:
- Exclusiveness: "each object should have ONE clear anchor"
- Inclusiveness: "all anchors should be used equally"

> **Pause and check:** Line 188: `return - self.delta_inc * acc_inc + self.delta_exc * acc_exc`. Why the negative on `delta_inc`? Because `acc_inc` is already negative entropy (`-(q_k * log(q_k)).sum()`), and we want to *maximize* entropy (even usage). So `-delta_inc * (-entropy) = +delta_inc * entropy` — gradient descent will push entropy higher.

Note: `delta_exc` and `delta_inc` are only non-zero for the **item** extractor (see line 126–127 docstring). The user ACF instance passes `delta_exc=0, delta_inc=0`, so its regularization contributes nothing. The constraints still compute (the code runs), but they're multiplied by zero. This is a design choice from the paper — regularizing only one side.

---

## ACF vs Prototypes: The Key Comparison

Both ACF and ProtoMF learn reference vectors (anchors/prototypes) and compute similarities. But the representations are fundamentally different:

| | ACF | ProtoMF |
|---|---|---|
| Reference vectors | Shared between user & item | Separate per side |
| Output | Weighted sum of anchors → back in embedding space | Cosine similarity vector → in prototype space |
| Output dimension | Same as `embedding_dim` | `n_prototypes` (different space) |
| Similarity | Dot product + softmax | Cosine similarity |
| Representation | "Object = mixture of archetypes" | "Object = similarity profile to prototypes" |

ProtoMF's innovation: instead of collapsing back to embedding space, the prototype similarity vector **is** the representation. This is what enables explainability — each dimension of the output directly corresponds to "how similar is this object to prototype k?"

---

## Check Your Understanding

Before moving on, make sure you can answer these:

- [ ] What does `cumulative_loss` do, and which extractors actually modify it?
- [ ] Why does `EmbeddingW` exist? When is a plain `Embedding` insufficient?
- [ ] In the ACF forward pass, what does the output `o_vect` represent conceptually?
- [ ] What do exclusiveness and inclusiveness encourage, and why are their signs opposite?
- [ ] What's the fundamental difference between ACF's output and ProtoMF's output?
