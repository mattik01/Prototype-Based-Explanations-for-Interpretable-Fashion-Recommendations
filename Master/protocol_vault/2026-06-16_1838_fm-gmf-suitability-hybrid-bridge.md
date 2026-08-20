---
date: 2026-06-16
time: "18:38"
phase: 2
tags: [thesis/background]
placed: 2026-08-20
---
# FM/GMF not a core feature-integration mechanism for prototypes; hybrid only with a grounding bridge

Verdict from reading S7 §4.2–4.3. The FM/GMF families are **not well-suited as THE mechanism** to make prototypes feature-aware: they integrate features in their own feature-centric (FM) or reconstruction (GMF) ways that do **not** ground prototypes. They stay **interesting** because FM and its same-family extensions (NFM, DeepFM) have **proven themselves for feature integration** — hence their role as baselines. For a **hybrid** (a separate feature module alongside ProtoMF, which already handles the interaction side), there may be **even better approaches that use features alone**. Key desideratum: any hybrid is only worthwhile if there is a **bridge between the added feature module and the grounding of the prototypes** — otherwise the module just bolts features on without advancing intrinsic prototype interpretability (the actual goal). Without such a bridge, a hybrid is unattractive.

[[decisions]] [[paper-understanding]] [[phase-4]]
