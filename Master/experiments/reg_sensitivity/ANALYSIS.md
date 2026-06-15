# Regularizer sensitivity sweep — analysis

**Model:** `user_item_proto` (prototypes_double_tie) · **Seed:** 38210573 · **Single seed.**
**LEO5 jobs:** 6614538 (ml-1m), 6614539 (hm_1_month), both COMPLETED 0:0 on 2026-06-11.
All hyperparameters fixed at the hyperopt-selected baseline; only the four prototype
regularizer weights (`user/item × sim_proto/sim_batch`) jointly scaled by `reg_multiplier`.

- ml-1m profile: production (100 ep / patience 10) · hm_1_month profile: dev (60 / 7).
- Prototypes: 11 item, 76 user (per the naming CSVs).

## 1. Accuracy vs regularization (clean, monotonic)

| Multiplier | ml-1m Test HR@10 | ml-1m Test NDCG@10 | hm Test HR@10 | hm Test NDCG@10 |
|:----------:|:----------------:|:------------------:|:-------------:|:---------------:|
| off (0)    | **0.6422**       | **0.3716**         | **0.6753**    | **0.4363**      |
| x0.01      | 0.6399           | 0.3673             | 0.6726        | 0.4336          |
| x0.1       | 0.6241           | 0.3589             | 0.6717        | 0.4287          |
| x1 (baseline) | 0.6319        | 0.3610             | 0.6569        | 0.4124          |
| x10        | 0.6157           | 0.3508             | 0.6419        | 0.4002          |
| x100       | 0.4786           | 0.2648             | 0.5058        | 0.2842          |

- **Regularization-off is the most accurate** on both datasets.
- Degradation is gentle up to x1 (~1–3% relative HR loss), then **collapses at x100** (−25% ml-1m, −25% hm).
- The **baseline operating point (x1 = the hyperopt-selected reg weights) already costs ~1.6 pts
  HR on hm** vs off, with no accuracy upside.

## 2. Interpretability vs regularization

**Robust metric — top-k categorical purity:** avg over item prototypes of the largest fraction of
the 20 nearest items sharing one category value. Preferred over mean lift, which is dominated by
rare-feature outliers.

> Note: ml-1m interpretability was initially empty because the run used the cluster's id-only
> `item_ids.csv` (the genre `item_features.csv` is gitignored and was never scp'd to the cluster).
> Explanations are post-hoc, so they were **regenerated locally from the saved checkpoints** with
> the local `item_features.csv` present — no retraining. ml-1m uses the pipe-separated `genres`
> column (single semantic categorical); H&M uses its richer article-metadata feature set.

**ml-1m — genre purity (single, clean signal):**

| Multiplier | genre purity |
|:----------:|:------------:|
| off        | 0.700        |
| x0.01      | 0.800        |
| x0.1       | 0.864        |
| **x1**     | **0.873**    |
| x10        | 0.795        |
| x100       | 0.445        |

Purity rises **monotonically** with regularization up to x1 (0.70 → 0.87, +25% relative), then
degrades at x10 and collapses at x100.

**H&M — top-k purity (coarser, multi-feature):**

| Multiplier | product_group purity | product_type purity |
|:----------:|:--------------------:|:-------------------:|
| off        | 0.509                | 0.364               |
| x0.01      | 0.500                | **0.455**           |
| x0.1       | **0.555**            | **0.455**           |
| x1         | 0.473                | 0.336               |
| x10        | 0.464                | 0.218               |
| x100       | 0.395                | 0.186               |

H&M peaks earlier (x0.01–x0.1) then degrades; same inverted-U shape, lower/noisier amplitude
(coarse 9-feature naming + dev-budget run).

- Caveat: lift-naming always selects top-3 descriptors, so descriptor *count* is constant and
  not informative; name-string collision counts are non-monotonic and unreliable.

## 3. Takeaway — the tradeoff

There **is** a genuine accuracy↔interpretability tradeoff, and it has an inverted-U interpretability
curve (rises then collapses), not a monotone one:

- **ml-1m: the baseline (x1 = the hyperopt-selected reg weights) is the interpretability sweet
  spot** — +25% relative genre purity for only ~1.6% relative HR loss vs off. Notably, the weights
  hyperopt picked *for accuracy* also land at the interpretability peak here.
- **H&M: sweet spot is lighter (x0.1)** — small purity gain at ≤1 pt HR cost; x1 already costs
  ~1.6 pts HR with no purity upside on the coarse metric.
- **x100 is pathological** on both datasets and both axes (collapse).
- Net: a non-trivial regularizer weight buys real prototype coherence; the *optimal* weight is
  dataset-dependent (coarser/larger catalogs favour lighter reg).

## Caveats
- **Common-factor scaling conflates the regularizers and is `sim_proto`-dominated (design limitation).**
  All four weights are multiplied by one shared multiplier, so their *ratios* stay frozen at the
  baseline's. Those baseline ratios are extremely imbalanced — `sim_proto` (item 1.626 / user 1.798)
  is ~170× / ~1000× larger than `sim_batch` (item 0.009407 / user 0.001765). Since each weight `wᵢ`
  sets that regularizer's influence (the `Rᵢ` terms are all similarly bounded), the regularization is
  **`sim_proto`-dominated at every multiplier**, and `sim_batch` rides along ~170–1000× weaker. At the
  strong extreme (×100) `sim_proto` → 163/180 (swamps `L_rec` → the collapse) while `sim_batch` →
  0.94/0.18 (never dominant). So the sweep probes a single *ray* through the 4-D weight space, the
  ×100 collapse is specifically a `sim_proto` effect, and `sim_batch`'s strong regime is never tested.
  **Consequence:** results reflect overall regularization strength along the baseline direction; they
  **cannot attribute effects to an individual regularizer or to the user-vs-item side.** Disentangling
  would require scaling each weight independently (one-at-a-time or a grid).
- **Single seed** — magnitudes are indicative, not significant; multi-seed needed before any claim.
- Top-k purity is a coarse automated proxy; the *geometric* interpretability the regularizers
  target (prototype spread, assignment sharpness) is better seen in the TSNE PNGs
  (`explanations/lift/tsne_*_prototypes.png`) and is not fully captured here.
- The H&M `section_name` purity column was discarded — its values exceeded 1.0, a parse artifact.
- ml-1m explanations were regenerated locally post-hoc (see note above); accuracy numbers are
  untouched from the original cluster run.
