---
date: 2026-07-13
time: "22:25"
phase: 4
---
# Low accuracy ceiling makes explanation a routine need — and the grounded model opens a bias-adjustment avenue (conditional Part III)

*(Verbatim-capture entry — Matteo's thought during the outline session, protocolled on request after discussion. Partially supersedes [[2026-07-12_1251_bias-fairness-parked-out-of-scope]]: bias/fairness graduates from "parked, visible omission" to "outlined conditional chapter". Structural placeholder exists: `Master/thesis/chapters/11_bias_adjustment.tex` + commented `\part{Bias and Adjustment}` in `main.tex`.)*

## The motivation argument (Matteo, near-literal)

"Interpretability is particularly valuable in recsys, because the highest achievable scores are far from perfect, and the situation where a user does not understand what is presented to him and why WILL occur (not like 99.5% classification on MNIST) — thus it is super helpful to be able to explain it to him."

Sharpening (assistant phrasing — remodel in Matteo's voice before any thesis use): in a 99.5%-accuracy regime, explaining errors serves rare tail cases; in recommendation, a HR@10 of ~0.3 is a *good* result, so the user-confusion situation is a routine event and explanation capability is standing operational equipment rather than an add-on. Coherence bonus: the same low ceiling is what makes the Rashomon set large ([[2026-06-11_1722_rashomon-set-recsys-positioning]]) — one structural fact about recsys feeds both the §1.1 motivation and Part II's licence.

## The bias-adjustment avenue (Matteo, near-literal)

"Our model in theory — and this is the III block — does open an avenue for bias correction and adjusting. 100% it does, and strongly so."

Mechanism: grounded/composed models have *addressable components* (ID rows, the B(t)/popularity channel, per-attribute shares) → bias becomes measurable per component and correctable by targeted intervention — including feeding dedicated bias-correcting rows, which the composition mechanism accepts by construction ([[2026-07-13_2113_why-f-not-attribute-in-model-names-mechanism-vs-contribution]]).

## Scope ruling (Matteo, 2026-07-13)

- **95% out of scope.** The full possibility space in this department warrants a Part III **or a future project** — deliberately not this thesis.
- BUT: it might warrant a **dedicated chapter with one small experiment** instead of a bare Future Work listing — include only if, once Part II tooling exists, the experiment turns out to be genuinely cheap. Decision point: after Part II experiments are designed.
- Until then: chapter skeleton exists but is commented out of the build; fallback home is Conclusion → Future Work.

[[phase-4]] [[explanations]] [[fairness]] [[decisions]] [[thesis-writing]]
