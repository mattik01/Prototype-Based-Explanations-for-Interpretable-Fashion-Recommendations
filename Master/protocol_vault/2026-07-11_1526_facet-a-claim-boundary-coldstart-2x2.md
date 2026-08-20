---
date: 2026-07-11
time: "15:26"
phase: 4
tags: [thesis/eval-framework, thesis/fi-protomf]
placed: 2026-08-20
---
# Facet A claim boundary + cold-start 2×2 (S0.2 scrutiny)

Two structural insights from the S0.2 side-reasoning step (scrutiny dossier `Master/docs/scrutiny/s0_foundation.md` §S0.2), both following from the double-tie score decomposing into **Block I** (user affinities · item-prototype sims) and **Block U** (user-prototype sims · item projection), with all four candidates touching only the item branch. **(1) Facet A claim boundary:** the candidates ground *what item prototypes mean* (intrinsic, feature-grounded vocabulary — R2/R3/R4), but the personalization weights û_k stay learned CF — claims must say "explanations are stated in intrinsic feature vocabulary," never "the model's reasoning is feature-transparent." Binding methodological rule adopted: every per-recommendation read-out must report the **score share of Block I vs Block U** on real checkpoints (SC.5/SC.8), else intrinsicness could describe a minority of the score. **(2) Cold-start 2×2:** in a bilinear score one warm side suffices — warm-user × cold-item works with item features alone (learned taste meets attribute-derived placement in the shared prototype space); cold users are structurally out of reach for item-side designs, would need user-side features, and don't exist in our split anyway. Thesis cold-start claims are scoped to cold items + warm-but-thin users; H&M is an item-cold domain (catalog churn), so the scoping matches the domain. Follow-up design ideas (dual-side attribute-aware candidate; 3-scenario cold testbed) captured in `design_candidates/scratchpad.md`.

[[explanations]] [[evaluation]] [[decisions]] [[phase-4]]
