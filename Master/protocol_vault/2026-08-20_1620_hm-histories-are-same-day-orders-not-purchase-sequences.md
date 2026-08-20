---
date: 2026-08-20
time: "16:20"
phase: 4
tags: [thesis/data]
---
# H&M "histories" are same-day orders, not purchase sequences: 94% of train purchases share their date

Measured at dc05's SC.8 (login node, 2026-08-20; F-DC05-09 working-basis
readout) on the canonical `hm_1_month` train split (476,394 rows, 73,418
users), directly from the split's own `t_dat` column:

- **94.27% of train purchases share their date with at least one other
  purchase by the same user** (same-day multi-item order);
- **99.07% of users** have at least one same-day multi-purchase;
- **mean distinct purchase days per user: 1.73** — the median user's entire
  train "history" is same-day orders (per-user same-day share median 1.00).

Reading for the data section: a V1 user history is typically **one or two
shopping trips**, not a sequence of independent purchase events. Items
inside one order are chosen together (co-purchase lockstep is strongest
within orders — the SC.1b P6 point, empirically confirmed in magnitude),
so any per-purchase presentation of history evidence (dc05's per-purchase
zoom) describes *order members*, not independent taste evidences. The
lineage deliberately drops per-event interaction context (F-DC05-09
wontfix; future direction per [[2026-07-12_0051_future-direction-interaction-attribute-aware]]),
and the presentation-level independence implication routes to the dc05
hidden-effects section.

Provenance caveat (standing rule): scrutiny-phase session measurement, no
committed generator (F-DC05-09 disposition explicitly declined one); the
thesis data section re-derives the number with its own committed script at
writing time — trivial (one pandas groupby over the split train file).

[[phase-4]] [[data]]
