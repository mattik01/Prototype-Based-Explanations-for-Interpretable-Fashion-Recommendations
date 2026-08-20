---
date: 2026-07-12
time: "14:12"
phase: 4
tags: [thesis/interpretability-objective]
placed: 2026-08-20
---
# Correction: the Rashomon set IS partially measurable — and the tractable instrument for our regime

*(Correction/update entry. **Amends [[2026-06-11_1722_rashomon-set-recsys-positioning]]** and the caution carried in [[2026-07-12_1208_giving-up-predictive-power-sparsity-flips-the-trade]], both of which say flatly "the Rashomon set can't be measured, keep it as a conceptual lens only." A close read of R22 (Rudin et al. 2022) §9 shows that blanket claim is too strong. From the same read that produced [[2026-07-12_1411_interpretability-knobs-map-to-rudin-families]].)*

## What the earlier notes got wrong

The vault (and the S1 mini-summary) said the Rashomon set "can't be measured here." The compressed summary dropped the fact that **R22 §9 actually supplies a whole measurement toolkit.** Rudin gives, as measurable quantities:
- **Rashomon volume** — volume of the near-optimal set in parameter space (Semenova et al. 2019);
- **Rashomon ratio** — that volume over the whole hypothesis-space volume;
- **pattern Rashomon ratio** — counted over distinct prediction *patterns* rather than functions (fixed denominator, more comparable across model classes);
- **ε-flatness / ε-sharpness** — flat-minima-style variants;
- a **closed-form solution for linear regression** (from the data matrix's singular values);
- **Model Class Reliance (MCR)** and the **variable-importance cloud** as measurable 2-D projections of the set.

So "cannot be measured" is simply wrong as a general statement.

## What is *actually* true for our regime (the precise version)

The honest, regime-specific picture:
- **Exact volume/ratio in deep-MF parameter space is intractable.** The closed form is linear-regression-only; for anything deep, Rudin's own fallback is sampling or brute force. So the *original caution was right about deep-MF exact measurement* — just wrong as a blanket claim.
- **The tractable instrument here is Rudin's practical proxy** (R22 p.45, near-verbatim): *"run many different types of machine learning algorithms; if they generally perform similarly, it correlates with the existence of a large Rashomon set."* This is directly executable in recommendation, is cheap, and ties straight to the empirical hook we already use — **Dacrema et al. 2019's flat leaderboards** (well-tuned simple baselines matching neural methods) plus our own ProtoMF≈MF replication. Flat leaderboards *are* the many-algorithms-agree signal.

## Consequence for the block-two Route-1 fork

Earlier I had posed Route 1 as a hard fork: (a) honest-but-weak "Rashomon-*style* selection within a restricted model family" vs (b) ambitious "actually approximate the Rashomon set in the ranking regime." The measurability correction **tilts this toward (b)**: the agreement-proxy makes (b) honest, cheap, and citable rather than a research dead-end. Recommended stance: use the multi-model-agreement proxy (grounded in Dacrema) as the recsys-native evidence that the set is large, and keep exact volume/ratio claims out of the deep-MF setting. This does *not* resurrect over-claiming — we still never quote a numeric set-size for our own trained models; we quote agreement across a diverse model zoo as the proxy.

**Net wording guidance going forward:** replace "the Rashomon set can't be measured" with "exact volume is intractable for deep MF, but the multi-algorithm-agreement proxy is measurable and is what we use." The double-stakes recsys framing (low- and high-stakes at once) remains the licence for importing the apparatus at all — see [[2026-07-12_1409_two-route-rashomon-methodology]].

[[evaluation]] [[explanations]] [[decisions]] [[phase-4]]
