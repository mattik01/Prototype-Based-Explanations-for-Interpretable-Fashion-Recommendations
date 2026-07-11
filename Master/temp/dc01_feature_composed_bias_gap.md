# Note: dc01 half-inherits LightFM's cold-start guarantee (bias channel is still ID-keyed)

> **RETIRED 2026-07-12 (dc01 SC.2, F-DC01-03):** this note's content is folded into
> the design doc — `dc01_feature_composed_factors.md` §3.4 amendment block
> ("Cold-inference amendments") — with its S0 dispositions (fleet `use_bias=0`,
> runner refusal, F-S0-07 guard). Kept only as the historical record; the design
> doc is authoritative.

*Temp note, 2026-07-07. Surfaced while re-reading B5 (LightFM). Move to scratchpad/protocol if it earns its place.*

## The finding
LightFM composes **both** halves of its score from features, so **both** are cold-start-safe:
- taste vector `q_i = Σ_f e_f` (feature embeddings)
- popularity term `b_i = Σ_f β_f` (feature **biases**)

dc01 (`feature_item_proto`) composes only the **first** half from features. The bias
is left as the stock ProtoMF per-ID scalar.

| | embedding | bias |
|---|---|---|
| **LightFM** | Σ feature embeddings | Σ feature **biases** — both cold-safe |
| **dc01** | Σ feature embeddings ✅ | per-item-**ID** scalar ❌ (cold item → row init 0, never trained) |

## Code evidence (verified 2026-07-07)
- `rec_sys/rec_sys.py:38–41` — `user_bias = nn.Embedding(n_users,1)`, `item_bias = nn.Embedding(n_items,1)`, `global_bias`; optional via `use_bias` (default on).
- `rec_sys/rec_sys.py:114–118` — `dots = Σ(u_embed·i_embed); dots += u_bias + i_bias + global_bias`. Bias is a flat offset added **outside** the prototype-similarity geometry (so popularity stays out of prototype space — good).
- `feature_extraction/feature_extractors.py:171–180` — `FeatureEmbedding.forward` returns `embedding_layer(feat).sum(dim=-2)` = Σ feature-value embeddings (+ optional per-object ID row). Representation is feature-composed; the bias is not.

## Consequence
A cold item in dc01 gets a **placeable taste vector** but a **dead (zero) popularity term** —
its `item_bias` row was never trained. LightFM's cold-start property is therefore only
**half-inherited**.

## Cheap fix (design seed)
Feature-compose the bias too: `b_i = Σ_f β_f` where each feature-value row carries a scalar
bias, mirroring the embedding path already built (same `feature_ids` gather, `.sum`). Near-zero
cost, closes the asymmetry, keeps popularity out of the prototype geometry. Optional per-object
ID-bias row (like the `use_id_feature` embedding row) preserves exact F=0 reduction to stock MF bias.

## Caveats / open questions
- Check whether the ProtoMF prototype variants actually run with `use_bias=True` in
  `confs/hyper_params.py` before claiming the bias matters at all for dc01.
- Is a strong popularity prior even *wanted* here, given the popularity-bias-in-prototype-space
  concern? ([[2026-06-16_2014_popularity-bias-prototype-interpretability-ablation]]) — the bias
  channel is arguably the *right* place to isolate popularity, but it's a design choice, not a free win.
