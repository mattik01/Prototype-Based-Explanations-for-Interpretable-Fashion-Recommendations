---
date: 2026-06-11
time: "18:06"
phase: 4
---
# Why ProtoMF over plain CF for interpretability — the positioning argument

The core "why not just use CF?" argument for the thesis. Memory-based collaborative filtering is *already* inherently interpretable ML — a fully transparent pipeline whose explanatory units are named items/users reached by a short bounded lookup. So interpretability alone is not a reason to invent ProtoMF. The reason to move past CF is that **matrix factorization buys advantages over memory-based CF** (better scaling, better performance on sparse data) — but raw MF *loses* interpretability (latent factors are meaningless). **ProtoMF is built on MF: it keeps MF's performance/scaling tier and re-adds the interpretability MF threw away.** So the chain is CF (transparent, weaker) → MF (strong, opaque) → ProtoMF (strong + interpretable again). The honest trade: ProtoMF is *not* fully transparent end-to-end the way CF is — the prototype abstraction introduces a naming hop (see [[posthoc-naming-vs-intrinsic-model-distinction]], naming = tax on abstraction) — but it still produces very intuitive explanations, just a different **flavor**: abstracted prototype *concepts* ("you like minimalist basics") vs. CF's concrete *item lists* ("similar to these items you bought"). The abstraction is the point, not a side effect. **Accuracy caveat (do not overclaim):** ProtoMF is "comparable to / sometimes better than MF," not strictly superior — our replication showed ProtoMF ≈ MF.

[[phase-4]] [[explanations]] [[decisions]] [[paper-understanding]]
