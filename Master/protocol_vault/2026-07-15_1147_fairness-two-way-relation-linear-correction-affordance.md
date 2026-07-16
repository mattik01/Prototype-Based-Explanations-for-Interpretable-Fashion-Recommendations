---
date: 2026-07-15
time: "11:47"
phase: 4
---
# Fairness future-work positioning: our approach relates to fairness in two ways (S2 §6.3)

*(From the S2 read, §6.3 p82, Matteo's point, agreed with one hedge and one strengthening. Feeds the fairness future-work outline — fairness itself stays parked per [[2026-07-12_1251_bias-fairness-parked-out-of-scope]]; this entry is what the future-work section says when the topic is resurfaced.)*

**Way 1 — field level (claim positioning, not fact):** interpretable ML is reasonably expected to help fairness research, and S2 §6.3 (p82) supplies both the supporting bridge (transparency shown to increase fairness in economic, political, and legal systems — Cowgill & Tucker 2019; Fine Licht 2014; Burke & Leben 2007) and the honest hedge to quote: "we are yet to see if and how transparency and fairness relate with each other in information systems." The thesis claims we are *positioned to test* an open question the field's own survey names — not that the link is established in recsys.

**Way 2 — algorithm level (the concrete claim): the linear correction affordance.** The same linearity that gives the scrutability affordance ([[2026-07-14_1511_balog-scrutability-fu-linear-edit-affordance]]: subtract a readable row → scores change immediately, no retraining) gives fairness machinery for free, in two grades:

- **Audit:** components are named and additive, so per-component score shares are directly computable *per user group* — "how much does feature X drive scores for group Y" is a measurement, not a modeling project.
- **Correct:** interventions are surgical — down-weight or zero a specific feature row or prototype contribution and observe the effect, the system-side edit to scrutability's user-side edit. **Same affordance, one level up.**

**Lineage making this credible:** the original ProtoMF paper itself ran bias-correction interventions on prototype activations (its trustworthy-AI motivation, continued by the I2/I4/I5 debiasing citers), and [[2026-06-11_1111_protomf-within-interpretable-ai]] already noted that feature-grounding makes prototype meaning auditable from explicit attribute profiles. dc01 sharpens exactly the property those interventions need: named components.

**Status:** future-work section content only — enabled-not-built, like the scrutability affordance and the route-1 projection. No fairness experiments enter the thesis scope.

[[decisions]] [[fairness]] [[explanations]] [[phase-4]]
