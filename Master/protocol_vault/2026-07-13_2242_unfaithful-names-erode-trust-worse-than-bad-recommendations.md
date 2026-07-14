---
date: 2026-07-13
time: "22:42"
phase: 4
---
# When explanations become interfaces, unfaithful names erode trust worse than bad recommendations

*(Verbatim-capture entry — Matteo's argument during the 1.2 outline revision, protocolled after discussion and approval. The strongest "why the naming problem matters" escalation so far; lands in §1.2 with an echo into §1.1's benefits discussion.)*

## The argument (Matteo, near-literal)

"Since we have the situation where a user wants to look at an explanation and tune the recommendation if possible: if he does it based on prototypes that are post-hoc named, and the naming does not truly capture what they actually affect in the system (rather trivial in our datasets, but not necessarily in real applications with many more features), then trust truly erodes and dissatisfaction sets in. If an explanation is demonstrably not perceived as trustworthy, that's worse than the recommendation."

## Agreed restatement

The moment an explanation becomes an *interface* — the user reads it and acts on it, tuning what they get — the names carry operational weight. If those names are post hoc and do not truly capture what the prototypes actually affect inside the system, the user's action misfires: they steer by the label, the system responds by the true semantics, and the mismatch is *visible to the user*. That erodes trust beyond what a bad recommendation ever could — a bad recommendation is an expected event (the ceiling is low, [[2026-07-13_2225_low-ceiling-makes-explanation-routine-bias-adjustment-part-iii]]), but a demonstrably untrustworthy explanation breaks the very contract the explanation established. Dissatisfaction from broken explanations is worse than dissatisfaction from imperfect recommendations.

**Scale caveat (Matteo, honest):** in our datasets the naming risk is rather trivial (few, clean attributes); in real applications with many more features it is not — the argument's weight grows with the feature space.

## Sharpenings (assistant phrasing — remodel before thesis use)

- This is the **faithfulness** argument from the XAI literature (Rudin: misleading explanations are dangerous; S2's trust/persuasiveness benefits run in reverse when the explanation is wrong) — but the steering scenario makes it sharper: unfaithfulness becomes *detectable by the user through their own action*, not just theoretically present under audit.
- Loop closure with intrinsic grounding: a composed name cannot lie about what it affects — the name IS the input row that computed the score share. **Faithfulness by construction, not by audit.** This is the §1.2→§1.3 hinge.
- Forward link: the steering scenario is exactly the interaction the conditional Part III (bias/adjustment, addressable components) would enable — explanation-as-interface is the shared premise.

[[phase-4]] [[explanations]] [[thesis-writing]]
