---
date: 2026-07-07
time: "16:51"
phase: 4
---
# Extension point: tie visual processing to the recommendation loss (end-to-end), not two-step

Surfaced while re-reading **B5 (LightFM, Kula 2015, §8)**. LightFM cannot ingest raw visual/audio features; at Lyst they use a **two-step** pipeline — CNN → binary image tags → recommendations — and *conjecture* that "substantial improvements could be realised if the CNNs were trained with recommendation loss directly." That end-to-end framing is the interesting part for us.

**The point.** dc01 currently sits on the **two-step** side: pre-extracted categorical/tag features → feature-composed item factors → prototypes. The alternative is **end-to-end** — let a visual encoder (CNN or otherwise) be shaped *by the recommendation objective itself*, so the learned visual representation is optimised for what actually drives recommendations rather than for a generic tagging task. This is also the **modality-native form of the prototype paradigm**: ProtoPNet-lineage models (Theme C) learn image prototypes end-to-end under task loss, exactly what LightFM/Lyst can't do. Grounding prototypes in *pixels under rec loss* would fuse the ProtoPNet vision lineage with recsys (reading-list **H1 VBPR / H2 Packer–McAuley**), and could be immensely beneficial if visual features are ever incorporated.

**Scope decision (for now: out).** This is **outside the scope** of an already-ambitious thesis, which is deliberately **item-first, tabular/categorical, two-step**. Not a V1 candidate — logged as a **future extension point / Seed-2 direction**, not current work.

**Data note.** The earlier blocker (partial H&M image set, ~86 of ~105k subdirs — CLAUDE.md known issue) is **not** actually a constraint: the missing images can be re-obtained easily from Kaggle if/when this is pursued. So the brake is *scope*, not *data availability*.

Pairs with **R7** ([[2026-06-11_0905_r7-user-perceivable-feature-vocabulary]] — "the model should roughly see what the user sees"; the image itself as a user-perceivable feature) and the intrinsic-over-post-hoc principle ([[2026-06-16_1808_intrinsic-over-posthoc-principle]]).

[[decisions]] [[phase-4]] [[explanations]] [[compute]]
