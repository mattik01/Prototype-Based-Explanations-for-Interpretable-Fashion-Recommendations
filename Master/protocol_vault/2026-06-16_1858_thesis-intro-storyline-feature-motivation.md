---
date: 2026-06-16
time: "18:58"
phase: 2
tags: [thesis/intro]
placed: 2026-08-20
---
# Thesis intro storyline: interpretability + accuracy/flexibility facets (S7 as feature-motivation citation)

Introduction "storyline" for the thesis. Start from ProtoMF's strength — **interpretability via prototypes** (Rudin interpretable-ML / intrinsic angle; our Axis-A grounding). Then motivate adding features: it is **expected to also improve accuracy** — cite **Chen et al. (2020) [S7] §5.3 Finding 1**: attribute-aware models outperform their plain counterparts (matched pairs FM > MF, NCF+ > NCF). This frames the **two facets of the thesis**: (1) **more interpretability** through more grounded prototypes; (2) **more accuracy + more flexibility** because the algorithm can also leverage features.

**Citation caveat:** S7's evidence is **RMSE / rating-prediction** (§3.5.1 branch), not item-ranking — cite as *directional motivation*, not proof in our regime (ProtoMF is item-ranking, hit_ratio@10 / NDCG).

**Rebuttal-ready reconciliation** (S7 §5.3 also finds *user* attributes most beneficial, again RMSE, on datasets with rich user demographics): does **not** contradict our item-first positioning. On H&M the user-preference signal lives in purchase history (CF, which ProtoMF keeps); thin user demographics add little, while item features add the content grounding CF lacks. Honest caveat to state once: with *rich* user features an item-first interpretability focus might leave some accuracy on the table — moot on H&M. Pairs with [[decisions]] attribute-source positioning entry.

[[decisions]] [[paper-understanding]] [[explanations]] [[evaluation]] [[phase-4]]
