---
date: 2026-07-15
time: "13:59"
phase: 4
tags: [thesis/eval-framework, thesis/intro]
placed: 2026-08-20
---
# Metric-walkthrough sharpenings at the S0 close

Full metric-by-metric walkthrough of the frozen reference table at the S0.7
closing gate (all columns: warm val/test HR/NDCG, D3, D4, cold-vs-cold, +kNN,
cold-vs-all, warm-delta, twin-share). Two claims got sharpened enough to record:

**1. The D4 history-flatness claim is window-compressed.** On `hm_1_month` the
user-history quartiles span almost nothing (Q1 = exactly 3 purchases, Q4 = 9–89
with median ~11) — one month of fashion purchases + the 5-core produces no deep
histories. The honest claim is "history strata don't differentiate *on the
one-month window*", never "history length doesn't matter"; on longer windows
(hm_3_month/hm_full) the strata genuinely separate and flatness could break.
Same scope discipline for datasets: every D3/D4 number is hm_1_month-only;
ml-1m is the deliberate opposite regime (heavy histories × coarse vocabulary)
and gets its own readouts free with the SC.6 fleet rows — no transfer assumed.

**2. The cold columns yield the thesis's motivation gap: "features inside vs
bolted on".** On cold items there is a ~0.09–0.27 HR@10 gap between post-hoc
attribute patching of CF models (attr-kNN fallback: 0.24–0.42, and the patched
ranking mirrors the warm ranking — patch quality rides on the warm embedding
space) and native feature integration (lightfm rows: 0.47–0.51, nearly closing
their own warm–cold gap). Raw CF sits at the artifact floor (interpretation
rule R-a). That gap is the empirical slot the fI/fU candidates aim at: keep the
double-tie's warm strength (0.66 vs the feature rows' 0.46–0.51) while landing
in the native-feature band on cold/tail columns. The baseline table becomes
truly informative only once the candidate rows land on it — by design (charter
C6, computed once, frozen).

Documentation: `Master/docs/eval_sampled_metrics_and_d3.md` (D3+D4 definitions,
literature verification — Rendle 2019 and Dallmann 2021 read in full, framing
rules for sampled metrics); interpretation rules R-a..R-g in the S0.7 dossier
section.

[[phase-4]] [[evaluation]] [[experiments]]
