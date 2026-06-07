---
date: 2026-06-07
time: "23:12"
phase: 3
---
# Two-version H&M dataset strategy + 1-month/5-core sizing

Decided two H&M variants. **V1 = `hm_1_month`** (last month, iterative **5-core**; ~623K interactions / 73K users / 13.7K items) — the small efficient dev/judgment testbed and the source of all *quantitative* claims. **V2 = `hm_3_month`** — repurposed as the later *qualitative* explanation showcase (showing, not proving; its metrics must not become headline claims). Guiding principle: shrink levers must **preserve H&M's defining sparsity** — window length and random user subsampling are acceptable; high k-core and most-active-user sampling are rejected because they raise density and bias toward power-shoppers (10-core keeps only ~5% of users, behaviorally unrealistic for fashion; the lfm2b-1mon 10-core analogy is numerically but not behaviorally valid). 5-core is paper-faithful; window is the only real deviation (best-justified). Recorded in `Master/temp/hm_1month_plan.md`, not yet executed.

[[data]] [[decisions]] [[explanations]]
