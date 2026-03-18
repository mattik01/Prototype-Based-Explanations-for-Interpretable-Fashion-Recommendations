# Session 9: Explanations — Guided Walkthrough

**File:** `utilities/explanations_utils.py` (~152 lines)

Open this file. These tools are not part of training — they're for post-hoc analysis of trained prototype models.

---

## The Big Picture

ProtoMF's selling point is **explainability**. After training, you can:
1. **Visualize** where prototypes sit relative to objects (t-SNE)
2. **Interpret** what each prototype represents (closest items)
3. **Explain** why a specific item was recommended to a specific user (score decomposition)

These three functions do exactly that.

---

## Step 1: `tsne_plot` (lines 7–34)

```python
tsne = TSNE(perplexity=perplexity, metric='cosine', init='pca',
            learning_rate='auto', square_distances=True, random_state=42)

tsne_results = tsne.fit_transform(np.vstack([prototypes, objects]))
tsne_protos = tsne_results[:len(prototypes)]
tsne_embeds = tsne_results[len(prototypes):]
```

**Line 21**: Prototypes and objects are **stacked together** before t-SNE, then **split after**. This is crucial.

> **Pause and check:** Why not run t-SNE on prototypes and objects separately?

t-SNE creates a 2D layout that preserves neighborhood structure. If you run it separately on prototypes and objects, the two resulting 2D spaces have **no relationship to each other** — you can't overlay them meaningfully. By running t-SNE on the combined set, the algorithm places prototypes and objects in the **same 2D space**, so you can see which objects cluster around which prototypes.

**Line 18**: `metric='cosine'` — this matches the model's similarity measure. If you used Euclidean distance, the visualization would show a different notion of "closeness" than what the model actually uses, potentially misleading.

The plot shows objects as small blue dots and prototypes as larger red dots. You'd expect to see each prototype surrounded by a cluster of objects — its "territory." If a prototype is isolated with no nearby objects, it might be a dead prototype (the regularization should prevent this).

---

## Step 2: `get_top_k_items` (lines 37–62)

```python
def get_top_k_items(item_weights, items_info, proto_idx, top_k=10, invert=False):
    weights_proto = item_weights[:, proto_idx]
    top_k_indexes = np.argsort(weights_proto if invert else -weights_proto)[:top_k]
    top_k_weights = weights_proto[top_k_indexes]
    item_infos_top_k = items_info.set_index('item_id').loc[top_k_indexes]
    item_infos_top_k['item weight'] = top_k_weights
    return item_infos_top_k
```

This function answers: **"What does prototype k represent?"** by finding the items most similar to it.

**`item_weights`** has shape `[n_items, n_prototypes]` — but what it contains depends on which prototype space you're interpreting:

**For item prototypes**: Pass the **item-to-item-prototype similarity matrix** (each item's cosine similarity to each item prototype). The top-k items for prototype 5 are the items most similar to prototype 5 → tells you "prototype 5 represents action movies" (if the top items are all action movies).

**For user prototypes**: Pass the **full item embedding matrix** (all item embeddings). This is conceptually different — you're finding which items would score highest for a hypothetical user who is maximally close to user prototype 3. It answers: "what would user prototype 3 recommend?"

**Line 57**: The `invert` flag sorts in ascending order instead of descending — giving you the *farthest* items. You'd use this to understand what a prototype does NOT represent, or as a sanity check.

**Lines 60–61**: Looks up item metadata (title, genre, etc.) from a DataFrame. This is how you go from "prototype 5 has high similarity to items [42, 107, 233]" to "prototype 5 represents action movies from the 1990s."

---

## Step 3: `weight_visualization` (lines 65–151)

This is the most complex function — it creates the explanation plots from the paper. It decomposes a single recommendation score into visible components.

### The Mathematics

For the `user_item_proto` model, the score for user u and item i is:

```
score = u_sim_mtx · i_proj + i_sim_mtx · u_proj
```

Where:
- `u_sim_mtx`: user u's similarity to each **user** prototype (shape `[user_n_protos]`)
- `i_proj`: item i's projected embedding in user-prototype space (shape `[user_n_protos]`)
- `i_sim_mtx`: item i's similarity to each **item** prototype (shape `[item_n_protos]`)
- `u_proj`: user u's projected embedding in item-prototype space (shape `[item_n_protos]`)

Each dot product is an element-wise multiply followed by a sum. The element-wise products show **which prototypes contributed most to the score**.

### The Visualization (lines 80–151)

**Lines 82–83**:
```python
u_prods = u_sim_mtx * i_proj    # Element-wise: contribution of each user prototype
i_prods = i_sim_mtx * u_proj    # Element-wise: contribution of each item prototype
```

These are the "why" vectors. `u_prods[k]` tells you: "user prototype k contributed this much to the recommendation because the user is similar to it AND the item projects onto it."

The function creates **6 bar charts** in two groups of 3:

**User-side (3 panels)**:
1. `u_prods` — contribution of each user prototype to the score (the "why")
2. `i_proj` — how the item projects onto user-prototype space (item's perspective)
3. `u_sim_mtx` — user's similarity to each user prototype (user's perspective)

**Item-side (3 panels)**:
1. `i_prods` — contribution of each item prototype to the score
2. `u_proj` — how the user projects onto item-prototype space
3. `i_sim_mtx` — item's similarity to each item prototype

The top panel (products) is the element-wise product of the bottom two panels. You can visually see: "user is very similar to user prototype 5 AND this item projects strongly onto user prototype 5 → big contribution."

**Line 73** — The rescale function:
```python
rescale = lambda y: 1 - ((y + np.max(y)) / (np.max(y) * 2))
```

This maps values to [0, 1] for the `coolwarm` colormap. High positive values get cool (blue), high negative values get warm (red), zero gets white. It's a visual aid — you can immediately see which prototypes contribute positively vs negatively.

**Lines 92–93** — Figure width scaling:
```python
i_vis_ratio = i_n_prototypes / (i_n_prototypes + u_n_prototypes)
u_vis_ratio = 1 - i_vis_ratio
```

If there are 20 user prototypes and 30 item prototypes, the item figure is wider (30/50 of the space) and the user figure is narrower (20/50). This keeps bar widths consistent so they're visually comparable.

**Lines 112–118** — Annotation of top contributors:
```python
u_annotate_protos = np.argsort(-u_prods)[:annotate_top_k]
```

The top-k most contributing prototypes get labeled with their index, so you can trace back to `get_top_k_items` to see what that prototype represents.

---

## Putting It All Together: Explaining a Recommendation

Here's the workflow to explain "why was item B recommended to user A":

1. **Load the trained model** using `Tester`
2. **Extract the internal representations** — run user A and item B through the model, capturing `u_sim_mtx`, `u_proj`, `i_sim_mtx`, `i_proj`
3. **Call `weight_visualization`** — see which prototypes drove the score
4. **Identify the top contributing prototypes** — say user prototype 5 and item prototype 12
5. **Call `get_top_k_items`** for those prototypes — learn that user prototype 5 represents "users who like sci-fi" and item prototype 12 represents "critically acclaimed dramas"
6. **Optionally call `tsne_plot`** — visualize where user A and item B sit relative to prototypes

The explanation becomes: "User A was recommended Item B because User A is similar to the 'sci-fi enthusiast' user archetype, and Item B projects strongly onto that archetype. Additionally, Item B is close to the 'acclaimed drama' item archetype, and User A's embedding projects onto that archetype too."

This is fundamentally different from standard MF where the only explanation is "the dot product was high" — with prototypes, you can point to specific, interpretable components.

---

## Key Takeaways

1. **Joint t-SNE** is essential — separate runs produce incomparable spaces
2. **Cosine metric in t-SNE** matches the model's similarity measure
3. **`get_top_k_items`** interprets prototypes by their nearest items — different input matrices for user vs item prototypes
4. **Weight visualization** decomposes the score into per-prototype contributions
5. **The explanation chain**: score → contributing prototypes → representative items → human understanding

---

## Check Your Understanding

Before moving on, make sure you can answer these:

- [ ] Why must prototypes and objects be in the same t-SNE run?
- [ ] What's the difference in `item_weights` input when interpreting item prototypes vs user prototypes?
- [ ] In the weight visualization, what do the three panels on each side represent?
- [ ] How would you explain a specific recommendation to a non-technical stakeholder using these tools?
