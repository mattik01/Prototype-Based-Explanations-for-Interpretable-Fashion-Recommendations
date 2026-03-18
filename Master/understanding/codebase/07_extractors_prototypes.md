# Session 7: Feature Extractors (Prototypes) — Guided Walkthrough

**File:** `feature_extraction/feature_extractors.py` — lines 191–372 (`PrototypeEmbedding` and `ConcatenateFeatureExtractors`)

Open the file and scroll to line 191. This is the core innovation of the paper.

---

## The Big Picture

`PrototypeEmbedding` represents each user or item not as a raw embedding, but as a **vector of cosine similarities to learned prototypes**. Think of it this way: instead of describing a movie by an opaque 50-dimensional vector, you describe it by how similar it is to 20 "archetypal movies" (the prototypes). This is what makes ProtoMF explainable — each dimension of the output has a clear meaning.

---

## Part A: PrototypeEmbedding

### Step 1: Constructor (lines 196–276)

**Lines 226–232** — The prototype matrix:
```python
if self.n_prototypes is None:
    self.prototypes = nn.Parameter(torch.randn([self.embedding_dim, self.embedding_dim]))
    self.n_prototypes = self.embedding_dim
else:
    self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]))
```

`self.prototypes` is an `nn.Parameter` of shape `[n_prototypes, embedding_dim]`. Each row is one prototype — a learnable vector in the same space as the embeddings. With 20 prototypes and embedding_dim=50, this is a `[20, 50]` matrix. These prototypes are learned end-to-end during training — they move to where they're most useful for the recommendation task.

**Lines 234–235** — Optional weight matrix:
```python
if self.use_weight_matrix:
    self.weight_matrix = nn.Linear(self.n_prototypes, self.embedding_dim, bias=False)
```

If enabled, a linear layer projects from prototype space (`n_prototypes`) back to embedding space (`embedding_dim`). This is for when you need the output dimension to differ from `n_prototypes`. In the standard user_proto/item_proto models, this is **off** — the similarity vector is the final representation directly.

**Lines 237–245** — Three cosine similarity variants:
```python
'standard':        cos(x, y)             # range [-1, 1]
'shifted':         1 + cos(x, y)         # range [0, 2]
'shifted_and_div': (1 + cos(x, y)) / 2   # range [0, 1]
```

> **Pause and check:** Why is `shifted` the default? Think about what happens downstream.

The output of this extractor gets dot-producted with the other side's representation in `RecSys.forward()`. If cosine values are in [-1, 1], negative similarities would flip the sign of the dot product contribution — a prototype the user is *dissimilar* to would contribute *negatively* to the score for items similar to that prototype. With `shifted` (range [0, 2]), all values are non-negative, so every prototype-similarity contributes positively (just more or less). This is more intuitive for recommendation: "user somewhat likes prototype 5 and item is very close to prototype 5 → positive contribution."

**Lines 247–263** — Regularization functions. There are two axes of regularization, each with two options:

**Batch regularization** (`reg_batch_type`, lines 248–253):
- `max`: `lambda x: - x.max(dim=1).values.mean()` — for each object in the batch, look at its highest prototype similarity. Minimizing the negation = maximizing the max = each object should have at least one prototype it's very similar to.
- `soft`: entropy regularization along axis 1 — each object's similarity distribution should be peaky (low entropy), meaning each object "belongs to" a clear prototype.

**Prototype regularization** (`reg_proto_type`, lines 255–263):
- `max`: `lambda x: - x.max(dim=0).values.mean()` — for each prototype, look at the most similar object in the batch. Maximizing this = each prototype should have at least one object strongly associated with it → prevents "dead" prototypes.
- `soft`: entropy regularization along axis 0 — each prototype's similarity distribution across the batch should be peaky.
- `incl`: inclusiveness constraint (same idea as ACF's) — ensures all prototypes are used equally.

The key insight: **batch reg** operates across prototypes for each object (axis=1), while **proto reg** operates across objects for each prototype (axis=0). Together they shape the similarity matrix from both directions.

### Step 2: Forward Pass (lines 299–324)

Let's trace with batch_size=64, embedding_dim=50, n_prototypes=20, and `u_idxs` shape `[64]`:

**Line 308**: `o_embed = self.embedding_ext(o_idxs)` → shape `[64, 50]`

**Line 311**: The critical line:
```python
sim_mtx = self.cosine_sim_func(o_embed.unsqueeze(-2), self.prototypes)
```

Let's unpack the broadcasting:
1. `o_embed.unsqueeze(-2)` → shape `[64, 1, 50]` (adds a dimension for broadcasting against prototypes)
2. `self.prototypes` has shape `[20, 50]`
3. PyTorch's `CosineSimilarity(dim=-1)` broadcasts: `[64, 1, 50]` vs `[20, 50]` → broadcasts to `[64, 20, 50]` → cosine along last dim → `[64, 20]`

So `sim_mtx` has shape `[64, 20]` — for each of the 64 objects, the cosine similarity to each of the 20 prototypes.

**Lines 313–316**: Output selection:
```python
if self.use_weight_matrix:
    w = self.weight_matrix(sim_mtx)   # [64, embedding_dim] — projected back
else:
    w = sim_mtx                       # [64, 20] — similarity vector IS the output
```

When `use_weight_matrix=False` (the standard case), **the output dimension equals `n_prototypes`**. This has a crucial constraint: the other side's embedding must also be `n_prototypes`-dimensional for the dot product in RecSys to work. If user has 20 prototypes, the item embedding must be 20-dimensional.

**Lines 318–322** — Computing regularization losses:
```python
batch_proto = sim_mtx.reshape([-1, sim_mtx.shape[-1]])  # Flatten to [batch_total, n_prototypes]
self._acc_r_batch += self.reg_batch_func(batch_proto)
self._acc_r_proto += self.reg_proto_func(batch_proto)
```

The `reshape` flattens any extra dimensions (when items have shape `[64, 11, 20]` — 11 candidates per user — it becomes `[704, 20]`). Then both regularizations are computed on this 2D matrix and accumulated. The accumulation happens every `forward()` call — over multiple batches in an epoch.

### Step 3: Loss Retrieval (lines 326–329)

```python
def get_and_reset_loss(self):
    acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
    self._acc_r_proto = self._acc_r_batch = 0
    return self.sim_proto_weight * acc_r_proto + self.sim_batch_weight * acc_r_batch
```

The weights (`sim_proto_weight`, `sim_batch_weight`) are hyperparameters searched by Ray Tune. They control how much the regularization matters relative to the recommendation loss. Too high → model focuses on regularization at the expense of accuracy. Too low → prototypes may collapse or go unused.

### Step 4: The Entropy Regularization in Detail (lines 278–282)

```python
@staticmethod
def _entropy_reg_loss(sim_mtx, axis):
    o_coeff = nn.Softmax(dim=axis)(sim_mtx)
    entropy = - (o_coeff * torch.log(o_coeff)).sum(axis=axis).mean()
    return entropy
```

This is the `soft` regularization option. Softmax normalizes along the given axis to get probabilities. Then it computes the entropy: `H = -Σ p * log(p)`. Low entropy = one element dominates = peaky distribution. The loss returns the entropy, and since it gets *minimized* during training, the model is pushed toward peaky distributions.

When `axis=1` (batch reg): softmax across prototypes for each object → each object should have a clear "best" prototype.
When `axis=0` (proto reg): softmax across objects for each prototype → each prototype should have a clear "best-matching" set of objects.

---

## Part B: ConcatenateFeatureExtractors (lines 332–372)

This wrapper is used for the `user_item_proto` model where each side has **both** a prototype branch and an embedding branch.

```python
class ConcatenateFeatureExtractors(FeatureExtractor):
    def __init__(self, model_1, model_2, invert=False):
        self.model_1 = model_1
        self.model_2 = model_2
        self.invert = invert
```

**`forward` (lines 355–362)**:
```python
def forward(self, o_idxs):
    o_repr_1 = self.model_1(o_idxs)
    o_repr_2 = self.model_2(o_idxs)
    if self.invert:
        return torch.cat([o_repr_2, o_repr_1], dim=-1)
    else:
        return torch.cat([o_repr_1, o_repr_2], dim=-1)
```

It runs both sub-extractors on the same input and concatenates their outputs. The `invert` flag swaps the order.

> **Pause and check:** Why does the order matter?

Because the dot product in RecSys is element-wise. If the user side outputs `[user_proto_similarities | user_embeddings_projected]`, then the item side must output `[item_embeddings_projected | item_proto_similarities]` — the user's prototype output must align with the item's embedding output and vice versa. The `invert` flag on one side ensures this alignment.

In practice (see factory code in Session 8): the user side is `ConcatenateFeatureExtractors(user_proto, user_embed, invert=False)` and the item side is `ConcatenateFeatureExtractors(item_proto, item_embed, invert=True)`.

Let's trace with user_n_prototypes=20, item_n_prototypes=30:

**User side** (invert=False): `[user_proto(20) | user_embed_projected(30)]` → total dim = 50
**Item side** (invert=True): `[item_embed_projected(20) | item_proto(30)]` → total dim = 50

The dot product then computes:
- `user_proto_sim(20) · item_embed_proj(20)` — how similar is the user's prototype profile to the item's projected embedding?
- `user_embed_proj(30) · item_proto_sim(30)` — how similar is the user's projected embedding to the item's prototype profile?

Both directions are captured. The total dimensions must match (20+30 = 20+30 ✓).

**`get_and_reset_loss` (lines 364–367)**:
```python
def get_and_reset_loss(self):
    loss_1 = self.model_1.get_and_reset_loss()
    loss_2 = self.model_2.get_and_reset_loss()
    return loss_1 + loss_2
```

Drains both sub-extractors' regularization losses. For user_item_proto, both the proto and embed sub-extractors contribute (though Embedding/EmbeddingW always return 0, so in practice only the proto branches contribute).

---

## Key Takeaways

1. **Prototypes are learned `nn.Parameter` vectors** — they're not pre-defined, they emerge from training
2. **The output IS the similarity vector** — each dimension has a clear meaning ("similarity to prototype k")
3. **Shifted cosine** keeps values non-negative for cleaner dot-product scoring
4. **Two axes of regularization** shape the similarity matrix: batch reg (each object → peaky) and proto reg (each prototype → used)
5. **ConcatenateFeatureExtractors** enables the dual-prototype model by aligning dimensions via `invert`
6. **Without `use_weight_matrix`**, output dim = `n_prototypes`, which constrains the other side's dimensions

---

## Check Your Understanding

Before moving on, make sure you can answer these:

- [ ] What is the shape of `self.prototypes` and what does each row represent?
- [ ] Why does `unsqueeze(-2)` enable broadcasting against the prototype matrix?
- [ ] Why does shifted cosine (range [0,2]) make more sense than standard cosine (range [-1,1]) for recommendations?
- [ ] What's the difference between `max` and `soft` regularization, and what does each encourage?
- [ ] For user_item_proto with 20 user prototypes and 30 item prototypes: what are the output dimensions of each side's two sub-extractors, and why must they cross-match?
