---
date: 2026-08-17
time: "14:07"
phase: 4
---
# Prototype geometry is underdetermined by the selection metric — a Rashomon-flavored observation

Implication of the coverage-regularizer discovery ([[2026-08-17_1407_coverage-regularizer-uncollapses-hm-prototype-space]]): hyperopt, selecting purely on val hit_ratio@10, produced a **collapsed-geometry winner** on hm_1_month (sim_batch ≈ 0.002) and a **spread-geometry winner** on hm_3_month (sim_batch ≈ 5.7) — the objective was evidently near-indifferent between configurations whose *interpretability-relevant* property (5 effective prototypes vs 56) is night and day.

**The observation, stated carefully:** the property this thesis cares about — a semantically meaningful, non-degenerate prototype space — is only weakly determined by the selection objective. Directions in config space that barely move the metric can swing the explanation quality completely. Accuracy does not buy interpretability, and hyperopt on accuracy alone leaves interpretability to chance (or to whatever the regularizer weights happen to land on).

**Thesis placement:** this is Rashomon-flavored evidence landing in block two's territory (Rashomon = **device only, never measured** — the hard decision stands, see the thesis-block-two design). The current pair is cross-dataset, so it is an *illustration*, not a Rashomon measurement; a same-dataset pair (hm_1_month winning config ± sim_batch, the isolating rerun) would make the illustration crisp while still being device-grade, not a Rashomon-set exploration. Also feeds the knobs-metrics chapter: sim_batch_weight is a knob whose effect lives almost entirely in the interpretability axis, invisible to hit_ratio@10 — exactly the kind of knob/metric mismatch that chapter catalogs.

**Practical consequence for our own pipeline:** whenever we (or a future practitioner) hyperopt a prototype model for accuracy and then *sell the prototypes as explanations*, the explanation quality was never selected for — it must be separately measured (effective direction count et al., [[2026-08-15_1651_spectral-scale-gauge-inversion-direction-diversity-metric]]) and, if needed, separately enforced (coverage regularization). This is an argument the thesis can make concrete with our own artifacts.

[[phase-4]] [[evaluation]] [[explanations]] [[thesis-writing]]
