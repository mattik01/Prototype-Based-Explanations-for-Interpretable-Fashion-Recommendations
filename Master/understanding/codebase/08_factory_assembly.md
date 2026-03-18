# Session 8: Factory & Assembly — Guided Walkthrough

**File:** `feature_extraction/feature_extractor_factories.py` (~174 lines)

Open this file now. This is where config dicts become live PyTorch architectures.

---

## The Big Picture

`FeatureExtractorFactory` is the translation layer between hyperparameter configs (dicts from `hyper_params.py`) and actual model components. It has two methods: `create_models` (plural, builds a user/item pair) and `create_model` (singular, builds one extractor). The tricky part is `prototypes_double_tie` — four sub-extractors with shared weights.

---

## Step 1: The Two-Level API (lines 12–117 vs 119–173)

**`create_models(ft_ext_param, n_users, n_items)`** — the entry point. Called by `Trainer._build_model()`. It reads `ft_type` from the config and branches into four cases:

1. `detached` — two independent extractors
2. `prototypes` — one prototype side, one embedding side
3. `prototypes_double_tie` — both sides have prototype + embedding branches
4. `acf` — anchor-based collaborative filtering

**`create_model(ft_ext_param, n_objects, embedding_dim)`** — builds a single extractor. Called by `create_models` for each branch. Routes on `ft_type` to instantiate `Embedding`, `EmbeddingW`, `PrototypeEmbedding`, or `AnchorBasedCollaborativeFiltering`.

The two levels exist because `create_models` handles the **relationships** between user and item extractors (dimension matching, weight sharing), while `create_model` handles **individual construction**.

---

## Step 2: The `detached` Case (lines 29–35)

```python
if ft_type == 'detached':
    user_feature_extractor = FeatureExtractorFactory.create_model(
        ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)
    item_feature_extractor = FeatureExtractorFactory.create_model(
        ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)
    return user_feature_extractor, item_feature_extractor
```

Simplest case: build user and item independently. Both use the same `embedding_dim`. This is what standard MF (`mf` model) uses — two plain `Embedding` extractors with no connection between them.

---

## Step 3: The `prototypes` Case (lines 37–67)

This handles `user_proto` and `item_proto` — models where ONE side has prototypes and the other has plain embeddings.

Look at the dimension logic carefully (lines 43–50 for user_proto):

```python
user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']

user_feature_extractor = FeatureExtractorFactory.create_model(
    ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)
item_feature_extractor = FeatureExtractorFactory.create_model(
    ft_ext_param['item_ft_ext_param'], n_items, user_n_prototypes)  # ← KEY
```

The user prototype extractor outputs a vector of length `user_n_prototypes` (the similarity to each prototype). For the dot product in RecSys to work, the item embedding must also be `user_n_prototypes`-dimensional. So the item embedding is created with `embedding_dim = user_n_prototypes`, **not** the global `embedding_dim`.

> **Pause and check:** This is the cross-matching constraint. If user has 20 prototypes, the item embedding must be 20-dimensional. If item has 30 prototypes, the user embedding must be 30-dimensional. The dimensions must match across sides because of the dot product in `RecSys.forward()`.

The same logic applies symmetrically for `item_proto` (lines 52–62), where the item side has prototypes and the user embedding dimension becomes `item_n_prototypes`.

---

## Step 4: The `prototypes_double_tie` Case (lines 69–99)

This is the most complex case — the `user_item_proto` model. Let's walk through it step by step.

### What we're building:

Each side (user, item) gets TWO sub-extractors wrapped in `ConcatenateFeatureExtractors`:
- A **prototype branch** (PrototypeEmbedding)
- An **embedding branch** (EmbeddingW — embedding + linear projection)

And the embedding weights are **shared** between the two branches on each side.

### The assembly (lines 71–99):

**Lines 71–76** — Read config and verify:
```python
item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']
user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']
assert not user_use_weight_matrix and not item_use_weight_matrix, \
    'Use Weight Matrix should be turned off to tie the weights!'
```

`use_weight_matrix` must be off because if the prototype branch used a weight matrix, the output wouldn't be in prototype space anymore, breaking the explainability and the dimension logic.

**Lines 78–83** — Building user branches:
```python
ft_ext_param['user_ft_ext_param']['ft_type'] = 'prototypes'
user_proto = FeatureExtractorFactory.create_model(
    ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)

ft_ext_param['user_ft_ext_param']['ft_type'] = 'embedding_w'
ft_ext_param['user_ft_ext_param']['out_dimension'] = item_n_prototypes
user_embed = FeatureExtractorFactory.create_model(
    ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)
```

Watch what happens here: the code **mutates** the config dict. It temporarily changes `ft_type` from `prototypes_double_tie` to `prototypes` to build the proto branch, then changes it to `embedding_w` for the embed branch. It also injects `out_dimension = item_n_prototypes`.

> **Pause and check:** Is this mutation safe? It's modifying the input dict, which means after this function returns, `ft_ext_param` has been permanently changed. In this codebase it works because `create_models` is only called once per model build. But it's a subtle source of bugs — if you called `create_models` twice with the same dict, the second call would see the mutated values. Something to be aware of when extending.

**Lines 85–90** — Same for item branches:
```python
item_proto = FeatureExtractorFactory.create_model(
    ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)
item_embed = FeatureExtractorFactory.create_model(
    ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)
```

Item embed gets `out_dimension = user_n_prototypes`.

**Lines 92–94** — The weight tying:
```python
user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight
item_embed.embedding_layer.weight = item_proto.embedding_ext.embedding_layer.weight
```

This is a PyTorch parameter assignment. After this, `user_embed.embedding_layer.weight` and `user_proto.embedding_ext.embedding_layer.weight` point to the **same tensor in memory**. When gradients update one, the other is automatically updated too. This means each user has ONE underlying embedding vector used in both the prototype branch and the embedding branch.

Why tie? It's an inductive bias: the raw embedding captures an object's position in latent space, and both branches should agree on where an object "is." Without tying, the two branches could learn conflicting representations of the same user.

**Lines 96–97** — Wrapping in ConcatenateFeatureExtractors:
```python
user_feature_extractor = ConcatenateFeatureExtractors(user_proto, user_embed, invert=False)
item_feature_extractor = ConcatenateFeatureExtractors(item_proto, item_embed, invert=True)
```

The user side outputs `[proto_sim(user_n_proto) | embed_proj(item_n_proto)]`.
The item side, with `invert=True`, outputs `[embed_proj(user_n_proto) | proto_sim(item_n_proto)]`.

The dot product then aligns: `user_proto_sim · item_embed_proj` + `user_embed_proj · item_proto_sim`.

---

## Step 5: The `acf` Case (lines 101–115)

```python
anchors = nn.Parameter(torch.randn(n_anchors, embedding_dim))

user_feature_extractor = AnchorBasedCollaborativeFiltering(
    n_users, embedding_dim, anchors, max_norm=max_norm)
item_feature_extractor = AnchorBasedCollaborativeFiltering(
    n_items, embedding_dim, anchors, delta_exc, delta_inc, max_norm=max_norm)
```

**Line 109**: The anchors are created **once** and passed to **both** extractors. Same tensor, shared gradients. Unlike prototypes (where each side has its own), ACF anchors are global reference points.

Notice: `delta_exc` and `delta_inc` are only passed to the item extractor. The user extractor gets the default (0, 0), so its regularization is inactive.

---

## Step 6: The `create_model` Builder (lines 119–173)

This method instantiates a single extractor. Look at the pattern used for optional parameters:

```python
max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
only_positive = ft_ext_param['only_positive'] if 'only_positive' in ft_ext_param else False
```

Every optional parameter uses `if 'key' in dict else default`. This means the config can be sparse — you only need to specify parameters you want to change from defaults. The trade-off:
- **Pro**: Flexible configs, easy to add new parameters without breaking existing configs
- **Con**: No schema validation — a typo in a parameter name is silently ignored, falling through to the default. This can cause hard-to-debug issues.

---

## Key Takeaways

1. **Two-level API**: `create_models` handles relationships (dimension matching, weight tying), `create_model` handles individual construction
2. **Cross-dimension matching**: prototype output dim on one side = embedding dim on the other side
3. **Weight tying** in double_tie: same embedding tensor in memory, gradients update both branches
4. **Config mutation** in double_tie is a subtle pitfall — the dict is modified in place
5. **ACF shares anchors** globally; ProtoMF keeps prototypes per-side
6. **Extension point**: to add a new `ft_type`, add a branch in `create_models` and (if needed) a branch in `create_model`

---

## Check Your Understanding

Before moving on, make sure you can answer these:

- [ ] Why are there two separate methods (`create_models` vs `create_model`)?
- [ ] In user_proto, why does the item embedding dimension become `user_n_prototypes`?
- [ ] In prototypes_double_tie: what are the four sub-extractors, their types, and their output dimensions?
- [ ] What does `user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight` actually do in PyTorch?
- [ ] Why is `invert=True` on the item side but `invert=False` on the user side?
- [ ] If you wanted to add `feature_item_proto` as a new `ft_type`, where would you add code?
