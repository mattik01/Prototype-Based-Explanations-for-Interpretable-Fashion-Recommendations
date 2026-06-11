---
date: 2026-06-11
time: "17:22"
phase: 4
---
# Rashomon set & RecSys — thesis-positioning note (use cautiously)

"Hard problem" splits into two independent axes: **low achievable ceiling** (RecSys *is* hard — noisy user behaviour caps absolute accuracy) vs. **small Rashomon set** (RecSys looks *easy* — many diverse models tie near the ceiling). Only the second axis justifies interpretable-by-design: a large Rashomon set ⇒ interpretability is ~free. Empirical hook (recsys-native, avoids the abstract construct): **Dacrema, Cremonesi & Jannach 2019, "Are We Really Making Much Progress?"** — well-tuned simple baselines match recent neural methods, i.e. flat leaderboards = evidence of a large recsys Rashomon set (plus our ProtoMF≈MF replication). **Caution:** the set can't actually be measured (any size claim is over-reach), and leaderboard flatness is partly confounded by noisy/insensitive eval metrics — so keep Rashomon as a *light conceptual lens / one-paragraph justification* for affordable interpretability, never a method or a measured quantity. (Consistency: downgrade the 12:30 "regularizer tuning = Rashomon search" entry to "Rashomon-style selection within a restricted model family" if Rashomon is kept.)

[[phase-4]] [[explanations]] [[evaluation]] [[decisions]]
