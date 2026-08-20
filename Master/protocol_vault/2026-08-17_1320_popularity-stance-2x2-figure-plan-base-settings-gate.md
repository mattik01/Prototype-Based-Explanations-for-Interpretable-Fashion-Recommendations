---
date: 2026-08-17
time: "13:20"
phase: 4
---
# Popularity stance, 2×2 thesis-figure plan, and the fUfI base-settings gate

Follow-up to [[2026-08-17_1128_shifted-cosine-baseline-asymmetry-popularity-channel]] — Matteo's closing thoughts on 1.1 of the explanation deep dive, discussed and refined.

## Stance: popularity in recommendations is NOT a negative thing

Bias/popularity signal is legitimate — often it IS the right recommendation. The requirement (Matteo's criterion): popularity must enter through a **disclosed, separable channel** (an explicit item bias b_i is the textbook form: transparent, measurable, removable at will) and must NOT corrupt the embedding space, which this thesis builds to be **mathematically interpretable** (see later thesis sections) and, since it is already transparent, ideally **practically interpretable** too. The failure mode is not "the model uses popularity" — it is "the prototype space becomes semantically meaningless because it is so distorted and bent toward the popularity channel". Primarily an interpretability concern, possibly also a performance concern (to be seen — the 2×2 metrics will say).

Operationalization (so the stance is measurable, not aesthetic): corruption indicators = **effective prototype count** (distinct names / clone blocks, e.g. 1.1's "76 prototypes, 5 distinct names") and the **baseline-vs-personalized score share** (e.g. 1.1's 80/20). These are exactly what the cosbias2x2 arms move.

## Thesis figure plan (Popularity Channel section)

Grid figure (probably one dataset for the thesis render; both generated):
- **Rows: U-Proto, I-Proto only** (2 rows). UI tiles are included only if they DEVIATE from the trend; if they just follow it, UI is reported as numbers in the text, not tiles.
- **Columns:** cosine b-off · cosine b-on · shifted-cosine b-off · shifted-cosine b-on
- **Tile:** tsne_prototypes.png, **prototype names removed entirely** (irrelevant for this figure).
- **Per tile, two reported numbers:**
  1. **Effective bias percentage** — the unpersonalized component share, normalized so it counts ONLY rank-relevant mass: U-side c·1ᵀt counts, I-side c·1ᵀu counts as 0 (dead gauge) even though it exists, and b_i counts on BOTH sides when bias is on (so I-side bias-on cells are nonzero — the metric differentiates all 8 cells). Exact normalization (signed components; aggregate over actual top-K recommendations) needs one deliberate spec paragraph at instrument-build time.
  2. **HR@10 (test).**
  3. **Degeneration number: effective direction count** (row-normalized prototypes, pairwise-cosine duplicate threshold — instrument menu in [[2026-08-15_1651_spectral-scale-gauge-inversion-direction-diversity-metric]]), **flagged clearly as a proxy/approximation, NOT a reliable textbook metric** (threshold-dependent; captures clone-collapse, blind to milder concentration; Matteo directive 2026-08-17). Spectral entropy of the normalized prototype matrix as the threshold-free robustness backup if challenged. **Name-collapse count (distinct lift names) stays OUT of the tiles** — it is an explanation-surface indicator confounded by naming vocabulary/filters (H&M's 9 coarse columns vs ml-1m's tag genome), dataset-specific; report it in prose as the geometry→explanation bridge ("N effective directions → N distinct names"). Note from 1.1 vs 1.2: baseline share and geometry corruption DISSOCIATE (H&M 80%/5 names vs ml-1m 87%/20 names, both n=1 breakdowns) — the two tile numbers are genuinely independent gauges; channel usage is architectural, prototype recruitment is data-dependent (weak taste structure required).

Two-tier consequence (Matteo, 2026-08-17): the current 18 cosbias2x2 arms (jobs 7426702–7426722) are the **mechanism tier** — same winning config, knob flipped — clean for geometry/effect claims but UNFAIR for metric comparison (hyperparameters were optimized for the shifted/b-off cell only). **For the actual thesis graphic: run full hyperopt combos at full thesis-level config for every cell** (probably one dataset; incumbent cell included — the current reference is fleet/dev-tier). Queues after model freeze per the standing two-tier numbers policy.

Figure-time caveat retained: each t-SNE is its own embedding → within-tile qualitative comparison only (clustered vs spread), one caveat sentence in the caption.

## Commitment: fUfI base-settings gate

At the end of the cosbias2x2 experiment: **re-evaluate what the base settings (cosine_type × use_bias) should be for building fUfI-ProtoMF**; if the winner is not the incumbent (shifted, bias-off), queue the necessary reruns. At the end of the thesis's popularity-channel section, **justify exactly that decision** in writing. The 2×2 is thus not only a thesis exhibit — it is the deciding experiment for the merge stage's foundation.

[[phase-4]] [[explanations]] [[thesis-writing]] [[experiments]] [[decisions]]
