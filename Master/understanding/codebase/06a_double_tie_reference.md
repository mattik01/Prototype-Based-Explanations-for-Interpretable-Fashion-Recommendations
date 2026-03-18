# 06a — Weight Tying in `user_item_proto` (Double Tie)

*Companion reference to [06_extractors_base.md](06_extractors_base.md) and [08_factory_assembly.md](08_factory_assembly.md) — explains the `prototypes_double_tie` variant in detail.*

## The Problem

In the single-prototype variants (`item_proto`, `user_proto`), only one side has prototypes. The other side is a plain embedding sized to match:

```
item_proto:
  Item side:  embedding → cosine with 30 prototypes → [30]
  User side:  plain embedding of size [30]            → [30]
  Dot product: [30] · [30] ✓
```

No sharing needed. But what if we want prototypes on **both** sides?

```
  User side:  embedding → cosine with 20 user prototypes  → [20]
  Item side:  embedding → cosine with 30 item prototypes  → [30]
  Dot product: [20] · [30] ✗  — dimension mismatch!
```

The outputs live in different prototype spaces with different sizes. The dot product breaks.

---

## The Solution: Two Branches Per Side

Give each side a **second branch** that outputs in the other side's dimension, then concatenate.

### User side (two branches):

1. **Proto branch**: user embedding → cosine with 20 user prototypes → `[20]`
2. **Embed branch**: user embedding → linear layer → `[30]` (to match item's 30 prototypes)

Concatenated output: `[20 | 30]` = `[50]`

### Item side (two branches):

1. **Proto branch**: item embedding → cosine with 30 item prototypes → `[30]`
2. **Embed branch**: item embedding → linear layer → `[20]` (to match user's 20 prototypes)

Concatenated output (inverted order): `[20 | 30]` = `[50]`

### Dot product alignment

```
User:  [ user_proto_sim (20 dims) | user_embed_proj (30 dims) ]
                ↕ dots with                  ↕ dots with
Item:  [ item_embed_proj (20 dims) | item_proto_sim (30 dims) ]
```

- First 20 dims: user's prototype profile × item's learned response to user prototypes
- Last 30 dims: user's learned response to item prototypes × item's prototype profile

The `invert=True` flag on `ConcatenateFeatureExtractors` for the item side flips the concatenation order to achieve this alignment.

---

## What is a Linear Layer?

A linear layer is a learned matrix multiplication that converts a vector from one size to another:

```
input: [50]  →  multiply by learned [50 × 30] matrix  →  output: [30]
```

The matrix values are parameters updated by gradient descent, just like embeddings. It's a **learned resize** — no non-linearity (no activation function like ReLU).

### Why `item_proto` doesn't need one

In `item_proto`, the user embedding table is simply created with `n_prototypes` dimensions from the start. No conversion needed — the raw embedding is already the right size.

### Why `user_item_proto` needs one

The user's embedding table is shared between two branches. The proto branch needs it at `embedding_dim` (e.g. 50) to compute cosine similarity against 50-dim prototype vectors. The embed branch needs to output a different size (e.g. 30, to match item prototypes). You can't have the same table be both sizes, so the embed branch uses a linear layer as an adapter: `[50] → [30]`.

---

## The Weight Tie Itself

Each side has two branches, and each branch has its own embedding table internally. Without tying, user #7 would have **two separate 50-dim vectors** — one in the proto branch's table and one in the embed branch's table. That's wasteful and incoherent.

The tie (factory lines 93–94):
```python
user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight
item_embed.embedding_layer.weight = item_proto.embedding_ext.embedding_layer.weight
```

This is just a Python variable reassignment. `user_embed`'s table is thrown away. Now both names point to the **same tensor in memory**. One table, two references to it — like two shortcuts pointing to the same file.

This makes both branches point to **the same tensor in memory**. Now user #7 has **one** 50-dim vector, and both branches read from it:

- Proto branch: takes the vector → cosine similarity with prototypes → `[20]`
- Embed branch: takes the **same** vector → linear projection → `[30]`

**Same input, different transformations.**

### Why this matters

1. **Parameter efficiency** — without tying, you'd have 2× the embedding parameters for no benefit.

2. **Coherent learning** — when the proto branch's loss says "user #7's embedding should move closer to prototype 3," that gradient updates the shared embedding. The embed branch **immediately sees this change** because it reads the same table. Both branches pull on the same underlying representation, forcing it to be good for both tasks simultaneously.

3. **The `use_weight_matrix` assert** (factory line 76) — `PrototypeEmbedding` can optionally add its own internal linear projection. But for tying to work, we need direct access to the raw embedding. An extra projection inside the proto branch would break the symmetry of what's being shared.

---

## Full Mathematical Example

This section traces a complete recommendation score computation through the double-tie architecture, naming every learned parameter and every operation.

### Setup

- 1000 users, 500 items
- `embedding_dim` = 4 (kept small for clarity)
- `user_n_prototypes` = 2
- `item_n_prototypes` = 3
- Cosine type: `shifted` (1 + cos)

### All Learned Parameters

These are the tensors that gradient descent updates during training:

| Parameter | Shape | Lives in | What it is |
|-----------|-------|----------|------------|
| **E_user** | `[1000, 4]` | Shared user embedding table | Each user's 4-dim identity vector |
| **E_item** | `[500, 4]` | Shared item embedding table | Each item's 4-dim identity vector |
| **P_user** | `[2, 4]` | User proto branch | 2 user prototype vectors, each 4-dim |
| **P_item** | `[3, 4]` | Item proto branch | 3 item prototype vectors, each 4-dim |
| **W_user** | `[4, 3]` | User embed branch (linear layer) | Projects user embedding from 4-dim to 3-dim |
| **W_item** | `[4, 2]` | Item embed branch (linear layer) | Projects item embedding from 4-dim to 2-dim |

Total: 6 learned parameter tensors. (Bias terms in RecSys omitted for clarity.)

### Step-by-Step Computation for User 7 and Item 42

#### Step 1: Look up raw embeddings

```
e_u = E_user[7]  = [0.3, -0.5, 0.8, 0.1]     (user 7's shared embedding)
e_i = E_item[42] = [0.6, 0.2, -0.4, 0.7]      (item 42's shared embedding)
```

These are the **same vectors** used by both branches on each side (that's the tie).

#### Step 2: User proto branch — cosine similarity with user prototypes

The 2 user prototypes:
```
P_user[0] = [0.5, -0.3, 0.7, 0.2]    (user prototype 0)
P_user[1] = [-0.1, 0.8, 0.3, -0.6]   (user prototype 1)
```

Cosine similarity (shifted = 1 + cos):
```
cos(e_u, P_user[0]) = dot(e_u, P_user[0]) / (‖e_u‖ × ‖P_user[0]‖)

dot = 0.3×0.5 + (-0.5)×(-0.3) + 0.8×0.7 + 0.1×0.2
    = 0.15 + 0.15 + 0.56 + 0.02 = 0.88

‖e_u‖    = √(0.09 + 0.25 + 0.64 + 0.01) = √0.99 ≈ 0.995
‖P_user[0]‖ = √(0.25 + 0.09 + 0.49 + 0.04) = √0.87 ≈ 0.933

cos = 0.88 / (0.995 × 0.933) ≈ 0.948
shifted_cos = 1 + 0.948 = 1.948
```

Similarly for prototype 1:
```
dot = 0.3×(-0.1) + (-0.5)×0.8 + 0.8×0.3 + 0.1×(-0.6)
    = -0.03 + (-0.40) + 0.24 + (-0.06) = -0.25

‖P_user[1]‖ = √(0.01 + 0.64 + 0.09 + 0.36) = √1.10 ≈ 1.049

cos = -0.25 / (0.995 × 1.049) ≈ -0.239
shifted_cos = 1 + (-0.239) = 0.761
```

**User proto branch output: `[1.948, 0.761]`**

Interpretation: User 7 is very similar to user-prototype 0 (1.948 out of max 2.0) and less similar to user-prototype 1 (0.761).

#### Step 3: User embed branch — linear projection

Linear layer W_user (shape `[4, 3]`, projecting 4-dim → 3-dim):
```
W_user = [[0.2, -0.1, 0.4],
           [0.5,  0.3, -0.2],
           [-0.3, 0.6, 0.1],
           [0.1, -0.4, 0.7]]
```

Multiply e_u × W_user:
```
output[0] = 0.3×0.2 + (-0.5)×0.5 + 0.8×(-0.3) + 0.1×0.1
          = 0.06 - 0.25 - 0.24 + 0.01 = -0.42

output[1] = 0.3×(-0.1) + (-0.5)×0.3 + 0.8×0.6 + 0.1×(-0.4)
          = -0.03 - 0.15 + 0.48 - 0.04 = 0.26

output[2] = 0.3×0.4 + (-0.5)×(-0.2) + 0.8×0.1 + 0.1×0.7
          = 0.12 + 0.10 + 0.08 + 0.07 = 0.37
```

**User embed branch output: `[-0.42, 0.26, 0.37]`**

Interpretation: Opaque. These are learned projections — each number is a mix of all 4 embedding dimensions.

#### Step 4: Concatenate user branches

```
User output = [user_proto | user_embed] = [1.948, 0.761, -0.42, 0.26, 0.37]
                                            ^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
                                            2 dims (user   3 dims (matches
                                            prototypes)    item prototypes)
```

#### Step 5: Item proto branch — cosine similarity with item prototypes

The 3 item prototypes:
```
P_item[0] = [0.4, -0.6, 0.1, 0.3]    (item prototype 0)
P_item[1] = [0.7, 0.1, -0.5, 0.2]    (item prototype 1)
P_item[2] = [-0.2, 0.4, 0.8, -0.1]   (item prototype 2)
```

Computing shifted cosine for each (abbreviated):
```
sim(e_i, P_item[0]) = 1 + cos(e_i, P_item[0]) ≈ 1.310
sim(e_i, P_item[1]) = 1 + cos(e_i, P_item[1]) ≈ 1.476
sim(e_i, P_item[2]) = 1 + cos(e_i, P_item[2]) ≈ 0.883
```

**Item proto branch output: `[1.310, 1.476, 0.883]`**

Interpretation: Item 42 is most similar to item-prototype 1, somewhat to prototype 0, less to prototype 2.

#### Step 6: Item embed branch — linear projection

Linear layer W_item (shape `[4, 2]`, projecting 4-dim → 2-dim):
```
W_item = [[0.3, -0.2],
           [0.1,  0.5],
           [-0.4, 0.3],
           [0.6, -0.1]]
```

Multiply e_i × W_item:
```
output[0] = 0.6×0.3 + 0.2×0.1 + (-0.4)×(-0.4) + 0.7×0.6
          = 0.18 + 0.02 + 0.16 + 0.42 = 0.78

output[1] = 0.6×(-0.2) + 0.2×0.5 + (-0.4)×0.3 + 0.7×(-0.1)
          = -0.12 + 0.10 - 0.12 - 0.07 = -0.21
```

**Item embed branch output: `[0.78, -0.21]`**

Interpretation: Opaque. Learned projections.

#### Step 7: Concatenate item branches (INVERTED)

Because `invert=True`, the embed branch comes first:
```
Item output = [item_embed | item_proto] = [0.78, -0.21, 1.310, 1.476, 0.883]
                                           ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
                                           2 dims       3 dims (item
                                           (matches     prototypes)
                                           user protos)
```

#### Step 8: Dot product

```
User:  [1.948,  0.761, -0.42,  0.26,  0.37 ]
Item:  [0.78,  -0.21,  1.310,  1.476, 0.883]
```

Multiply element-wise and sum:

```
Term 0:  1.948 × 0.78  =  1.519    ← user_proto_sim[0] × item_embed_proj[0]
Term 1:  0.761 × -0.21 = -0.160    ← user_proto_sim[1] × item_embed_proj[1]
Term 2: -0.42  × 1.310 = -0.550    ← user_embed_proj[0] × item_proto_sim[0]
Term 3:  0.26  × 1.476 =  0.384    ← user_embed_proj[1] × item_proto_sim[1]
Term 4:  0.37  × 0.883 =  0.327    ← user_embed_proj[2] × item_proto_sim[2]
                          ------
score                   =  1.520
```

### Decomposing the Score

The score 1.520 is the sum of **two groups**:

#### Group A: User prototypes × Item projections (terms 0–1)

```
Σ_k  user_proto_sim[k] × item_embed_proj[k]

= 1.948 × 0.78  +  0.761 × (-0.21)
= 1.519 + (-0.160)
= 1.359
```

- **user_proto_sim[k]** is interpretable: it's the cosine similarity (shifted) of user 7 to user-prototype k. We can inspect the prototype vectors and find representative users for each prototype.
- **item_embed_proj[k]** is opaque: it's a linear combination of all of item 42's embedding dimensions. We can't say what it "means" for a specific prototype without tracing through W_item.

#### Group B: User projections × Item prototypes (terms 2–4)

```
Σ_k  user_embed_proj[k] × item_proto_sim[k]

= (-0.42) × 1.310  +  0.26 × 1.476  +  0.37 × 0.883
= (-0.550) + 0.384 + 0.327
= 0.161
```

- **item_proto_sim[k]** is interpretable: it's the cosine similarity of item 42 to item-prototype k. We can find representative items for each prototype.
- **user_embed_proj[k]** is opaque: a linear combination of user 7's embedding.

#### The full decomposition

```
score = [Group A: user proto profile × item response]  +  [Group B: user response × item proto profile]
      =  1.359                                          +  0.161
      =  1.520
```

### Comparison: Same Example in `item_proto` (Clean Case)

If this were `item_proto` instead, with 3 item prototypes:
- User 7 would have a plain embedding `[e_0, e_1, e_2]` of size 3 (one per prototype)
- Item 42 would have `item_proto_sim = [1.310, 1.476, 0.883]` (same as above)

```
score = e_0 × 1.310  +  e_1 × 1.476  +  e_2 × 0.883
```

Each term: "how much user cares about prototype k" × "how similar item is to prototype k."
Both factors are directly interpretable. No linear layer mixing. No opaque numbers.

---

## The Compromised Embedding Problem

In `item_proto`, the user embedding is trained with a **single job**: represent the user's preferences over item prototypes. Each dimension maps cleanly to one prototype.

In `user_item_proto`, the user embedding (e.g. `e_u = [0.3, -0.5, 0.8, 0.1]`) is shared between two branches and trained with **two jobs simultaneously**:

1. **Proto branch** pulls on it: "this vector should have high cosine similarity to user-prototype 0" → gradients push it toward `P_user[0]`
2. **Embed branch** pulls on it: "when multiplied by W_user, it should produce values that work well in the dot product with item prototype similarities" → gradients push it in a different direction

The final embedding is a **compromise** between these two pressures. This means:

- `user_proto_sim[k]` (cosine similarity to user-prototype k) is computed from an embedding that was **also** shaped by the embed branch's needs. The similarity is still mathematically cosine similarity, but the embedding it's computed from isn't purely optimized for this comparison.

- The item side has the same problem: `item_proto_sim[k]` comes from an item embedding that was also shaped by the item embed branch.

In `item_proto`, the user embedding is purely about prototypes → clean signal.
In `user_item_proto`, the user embedding is about prototypes AND linear projection → muddied signal.

The prototype similarities are still real cosine similarities — the math is the same. But the embeddings they're computed from encode a richer, less focused representation. The explainability isn't broken, but it's diluted.

---

## What is Trained vs What is Computed

| Thing | Trained or Computed? | Description |
|-------|---------------------|-------------|
| E_user `[1000, 4]` | **Trained** | User embedding lookup table |
| E_item `[500, 4]` | **Trained** | Item embedding lookup table |
| P_user `[2, 4]` | **Trained** | User prototype vectors |
| P_item `[3, 4]` | **Trained** | Item prototype vectors |
| W_user `[4, 3]` | **Trained** | User embed branch linear layer weights |
| W_item `[4, 2]` | **Trained** | Item embed branch linear layer weights |
| user_proto_sim `[2]` | **Computed** | Shifted cosine similarity of user embedding to each user prototype |
| item_proto_sim `[3]` | **Computed** | Shifted cosine similarity of item embedding to each item prototype |
| user_embed_proj `[3]` | **Computed** | User embedding × W_user (linear projection) |
| item_embed_proj `[2]` | **Computed** | Item embedding × W_item (linear projection) |
| score (scalar) | **Computed** | Dot product of concatenated user and item outputs |

All 6 trained parameter tensors are updated by gradient descent. The 5 computed values are deterministic functions of the trained parameters for a given (user, item) pair.

---

## Comparison Across Variants

| Variant | Sharing | Linear layers | Explainability |
|---------|---------|---------------|----------------|
| `mf` (detached) | None | None | None (opaque embeddings both sides) |
| `item_proto` | None | None | Full (each prototype term is interpretable) |
| `user_proto` | None | None | Full (symmetric to item_proto) |
| `user_item_proto` | Embedding tied within each side | One per side (embed branch) | Partial (prototype sims clean but from compromised embeddings; projected terms opaque) |
| `acf` | Anchors shared across sides | None | Indirect (output is back in embedding space) |

## Code References

```
feature_extractor_factories.py:69-99   — prototypes_double_tie construction
feature_extractor_factories.py:93-94   — the actual weight tie (assignment)
feature_extractor_factories.py:76      — use_weight_matrix assert
feature_extractor_factories.py:96-97   — ConcatenateFeatureExtractors with invert
feature_extractors.py:80-114           — EmbeddingW (embedding + linear layer)
feature_extractors.py:191-329          — PrototypeEmbedding
feature_extractors.py:332-372          — ConcatenateFeatureExtractors
```
