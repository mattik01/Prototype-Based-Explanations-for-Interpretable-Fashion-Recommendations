---
date: 2026-08-17
time: "19:37"
phase: 4
tags: [thesis/background, thesis/hidden-effects, thesis/fufi-merge]
placed: 2026-08-20
---
# DECISION: the lineage stays on (shifted cosine, bias off) — now evidenced, not inherited. cosbias2x2 results + grid drafts

Closes (working-tier) the base-settings question opened in [[2026-08-17_1320_popularity-stance-2x2-figure-plan-base-settings-gate]]. All 18 special-experiment arms landed (jobs 7426702–7426722, mechanism tier: reference winning config, only cosine_type/use_bias flipped, same seed).

## The 2×2 results in brief

- **I-side placebo PASSED:** item_proto ref vs cosstd — HR@10 0.4938/0.4947 (hm), 0.5711/0.5709 (ml); effective directions and user-cone statistic identical (0.661 = 0.661). The bias pairs are twins too (0.5807/0.5814, b_i spread 0.368 both). The cosine flip does nothing on the item host, in both bias settings — the loss-invariance theorem, now empirical.
- **U-side cosstd: the cone displacement.** Denied the free per-item baseline, the model rebuilt popularity by herding the USER cloud into a cone (mean resultant length 0.07→0.50 hm, 0.11→0.62 ml). Prototypes were NOT freed (hm 5→7 eff, still collapsed; ml 15→11, spectral 5.2→3.2 — worse). Metrics −1 to −3 pts. **The +1 shift is partially protective: it gives popularity a parameter-free outlet; removing it displaces the distortion into the embedding geometry itself.**
- **Bias sink is host-dependent, exactly as theory says:** I-host (no covert channel) uses b_i heavily (spread 0.15–0.37), +2.6 pts hm, geometry 2→7 eff. U-host/UI (covert channel available) leave b_i nearly untouched (0.006–0.016) until the shift is ALSO removed, then b_i partially substitutes (0.067–0.12).
- **b_u spread = 0.0000 in all nine bias arms** — the user bias never trains under sampled softmax (loss-inert), ninefold direct confirmation of the dead-gauge theory.
- **fp-twin nugget:** item_proto-hm bias vs cosstd_bias are loss-equivalent models; metrics twin to the 4th digit (0.5204/0.5205) yet geometry lands at 7 vs 2 effective directions — geometry underdetermined by the metric even WITHIN one config (floating-point lottery). Sibling of [[2026-08-17_1407_geometry-underdetermined-by-selection-metric-rashomon]].
- UI is where the shift is most load-bearing: cosstd costs −5 pts (hm 0.659→0.608) — relevant for the fUfI merge host.

## Why we stay on (shifted, bias off)

1. **By Matteo's own criterion** (popularity is fine if disclosed and non-corrupting): cosstd achieves the opposite — popularity relocates INTO the geometry (cone), prototypes stay collapsed, and the cleanly separable, measurable B(t) disclosure line is lost. The shift's channel is the most honest place for popularity to live: visible, separable, measurable to the 4th decimal. Manage and disclose, don't amputate.
2. **Bias-on wins only on the I-host**; neutral-to-negative on U/UI (sink unused while the covert channel exists), and it imports the cold-bias-policy problem (untrained b_i = silent cold-item depressant; cold_eval refuses such configs) into the thesis's central cold story. dc02 requires bias-off; comparability across the lineage survives only with the incumbent.
3. **Archaeology: the incumbent IS the paper's setting, and the cosine half was author-tested.** The authors' original upload has use_bias:0 in every shipped config (base_param, first commit), and contains user_proto_{standard,shifted,shifted_and_div}_hyper_params — an unpublished three-way cosine ablation; their final "chose" configs all settled on shifted. Our cosstd result independently reproduces their unpublished choice, mechanism now explained.
4. Frozen S0 table, all dossiers, both candidates were built on the incumbent — zero comparability cost.

**Decision wording:** (shifted, bias-off) confirmed as the lineage standard — evidenced, not inherited. Final ratification with equal-treatment numbers at thesis-section writing, per the standing commitment. What we take instead of a switch: the effective-bias-share instrument as standing disclosure; the coverage regularizer (sim_batch) as the actual promising geometry lever (next candidate arm if any); bias-on documented as the I-host-specific alternative.

## Addendum (same day): the effective-bias instrument is a LOWER BOUND — popularity can still hide inside the prototype structure

The annotated grids expose the instrument's blind spot, and the blind spot is itself evidence: the U-host cosstd tile reads **eff. bias 0%** while HR barely moves — popularity did not vanish, it went dark (into the user cone, inside the "personalized" mass where no additive decomposition can see it). The instrument counts only the DISCLOSED additive channels (c·1ᵀt terms + b_i); popularity carried by the geometry itself — the cone, and even in shifted models the clone-amplified cos-part (the recruitment mechanism) — is invisible to it. So:

- **eff. bias share = a lower bound on popularity mass**, never "the popularity content";
- the geometry indicators (cone statistic, effective direction count) are the complementary detectors for concealed popularity — the two families must always be reported together;
- this is the transparency argument self-demonstrated: shifted keeps popularity where the instrument can measure it to the percent; standard conceals it. Caption sentence for the thesis figure.

## Thesis placement (Matteo, 2026-08-17)

In the section that PRESENTS ProtoMF, at its end: bring this up — **what is concealed in the shifted cosine** (the structural popularity channel and its side asymmetry) **and how it does not seem to impact much** (the 2×2: metrics robust, placebo on the item side, incumbent not dominated). The two grid sheets carry the visual argument.

## Grid drafts (proto versions, mechanism tier)

![[cosbias2x2_grid_hm_PROTO.png]]
![[cosbias2x2_grid_ml_PROTO.png]]

(Also at `Master/temp/cosbias2x2_grid_{hm,ml}_PROTO.png`; generator scripts in session scratchpad.)

**Open questions for the final graphics:** (a) chosen dataset still open (hm shows the pathology + mechanisms, ml shows the healthy contrast — possibly both, possibly one + numbers); (b) **fair evaluation is probably needed: fully trained runs, i.e. full hyperopt at thesis-level config per cell** (the current arms are away-game mechanism tier — fine for geometry/effect claims, unfair for metric comparison), per the two-tier consequence already recorded in the figure-plan entry.

[[phase-4]] [[decisions]] [[explanations]] [[experiments]] [[thesis-writing]]
