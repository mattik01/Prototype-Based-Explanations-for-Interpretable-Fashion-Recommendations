---
date: 2026-08-17
time: "14:07"
phase: 4
tags: [thesis/data]
placed: 2026-08-20
---
# User-history gap quantified: ml-1m users have ~11× the history of hm_1_month users

Matteo's hypothesis during deep-dive 1.2: the healthy ml-1m prototype space vs the collapsed hm_1_month one (see [[2026-08-17_1128_shifted-cosine-baseline-asymmetry-popularity-channel]]) might be driven by **history length** — short histories make users barely localizable in taste space, so prototypization has nothing to hold onto. Measured on the train splits (login node, 2026-08-17):

| dataset | users | mean hist | median | p10 | p90 |
|---|---|---|---|---|---|
| hm_1_month | 73,418 | 6.5 | **5** | 3 | 12 |
| hm_3_month | 256,701 | 9.3 | **7** | 3 | 18 |
| hm_full (~2 yr) | 889,062 | 27.5 | **15** | 4 | 65 |
| ml-1m | 6,034 | 93.2 | **56** | 15 | 222 |

Median gap ml vs hm_1_month: **11×**; even the full 2-year H&M window (median 15) is far below ml-1m. Mechanism: a user with ~5 purchases contributes almost no taste signal — their embedding is shaped by a handful of mostly-head items, the user cloud loses geometry, and prototypes have no niches to anchor on (only categorically segregated segments — sport/socks/Mama/plus-size — survive even short histories).

**Verdict (via the hm_3_month probe, see [[2026-08-17_1407_coverage-regularizer-uncollapses-hm-prototype-space]]):** history length is a real *contributor* but **not the binding constraint** — the coverage regularizer held the space open at median-7 histories. Apportioning history vs regularizer would need a stratified retrain (hm_1_month restricted to users with ≥N purchases) — recorded as a designable future arm, not queued.

[[phase-4]] [[data]] [[explanations]] [[experiments]]
