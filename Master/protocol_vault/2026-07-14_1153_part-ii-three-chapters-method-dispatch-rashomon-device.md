---
date: 2026-07-14
time: "11:53"
phase: 4
tags: [thesis/interpretability-objective]
placed: 2026-08-20
---
# Part II restructured: three chapters (why/what/how), method-first with taxonomy as lookup table, Rashomon demoted to unmeasured device

Part II was one chapter ("Interpretability Quantified"); restructured during the outline walk into **three chapters** under the renamed part **"Towards Interpretability as a Model Objective"** (the block's working title promoted to the `\part{}`; "Towards" = honest hedge — tailored framework, bare-regularizer landing acceptable, reconciliation = future work; coheres with SQ-3 title-ladder C/D).

**The three chapters (why / what / how):**
1. **"Why Quantify Interpretability?"** — motivation: penalties tuned blind; explained-well collapses into reasons-well; "Many Good Models" (Rashomon, early + headline-level); accuracy-cost inoculation passage (Matteo: goes here, pre-emptive). Fallback: merges into the next chapter if too thin at prose time.
2. **"Interpretability Knobs and Metrics"** — the vocabulary. Knob/metric = same property on cause/effect side; the taxonomy (Rudin families spine, four sparsity axes, entry points) **now including, per knob, its adjustment class at a higher abstraction level (details TBD): model-structural vs training-adjustable {into the training loss / into validation / plain regularizer}**. Role: **the taxonomy is the LOOKUP TABLE the method dispatches on.** No routes named, no experiments here.
3. **"Interpretability Tuning"** — the method, own chapter (Matteo: "not just taxonomy — a whole framework, a whole method"). Goal reframed: Part I delivered interpretable models; this chapter is about **tunably / MORE interpretable** models. Order: **the method first** — explanation-backward tracing (fix the explanation formula → quantify interpretability via the taxonomy's metrics, sparsity first → backtrace every affecting component → per component: desired direction + adjustment class → dispatch) — **then the two routes as the method's final dispatch step** (section "Tunably More Interpretable Models through Two Channels"; "channel" = prose word, Route-1/2 stays vault shorthand): model-structural → Rashomon-window search; differentiable → training loss; validation-type → regularizer and/or Route-1 selection numbers. **A "Demonstrating the Method" section is mandatory** (walk the pipeline on our own score formula; dispatch table as payoff; the four sparsity axes + the understandability-weighted row fall out as *products*, proving the method generates knobs rather than assuming them). ALL Part II experiments pooled in this chapter.

**Rashomon scoping — hard decision (Matteo):** the Rashomon set is **never measured** in this thesis — "so wildly out of scope of my thesis it's comical." It is a *device*: a motivating and guiding principle — headline-level early in the motivation chapter, then a one-sentence **implicit base assumption** for Route 1. **This amends [[2026-07-12_1412_rashomon-set-is-partially-measurable-correction]]**: that entry's recommendation to use the multi-algorithm-agreement proxy *as our instrument* is dead; Dacrema 2019 flat leaderboards survive as citation-level evidence only. The ε accuracy-tolerance window survives inside Route 1 as protocol, not measurement.

Old slug `interpretability-quantified` retired (files `10a/10b/10c_*.tex`, slugs `interpretability-objective` / `knobs-metrics` / `interpretability-tuning`); placement-ledger rows referencing the old slug need a migration pass.

[[thesis-writing]] [[decisions]] [[explanations]] [[phase-4]]
