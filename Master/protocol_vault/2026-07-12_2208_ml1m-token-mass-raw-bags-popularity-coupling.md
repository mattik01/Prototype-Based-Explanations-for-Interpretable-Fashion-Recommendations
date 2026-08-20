---
date: 2026-07-12
time: "22:08"
phase: 4
tags: [thesis/data, thesis/hidden-effects]
placed: 2026-08-20
---
# ml-1m token mass: raw LightFM bags ratified — popularity coupling is a must-mention hidden effect

At the S0.5-addendum gate the token-mass convention for ml-1m feature bags was
decided: **raw indicator bags, "like LightFM did it"** — every token (genre or
genome-tag @0.8) weighs 1, so a movie's vote mass in any sum equals its token
count; the per-movie-normalization alternative (every movie votes mass 1) stays
an evidence-triggered fallback under charter C5's deviation clause. Rationale:
B5-faithfulness (the whole ml-1m field set is anchored to LightFM's published
recipe — deviating on weighting would weaken that citation) and moderate
measured skew (per-user effective basket size shrinks only to ~77% of nominal;
the single heaviest movie carries ~6.5% of a user's mass on average).

**The fact we must keep mentioning (user directive):** tag-richness is not
random — token count correlates with train popularity at **Spearman ρ = 0.672**
(measured by `Master/temp/dc_checks/s0/ml1m_feature_audit.py`; scrutiny working
basis, no thesis quotes without dedicated re-measurement). Under raw bags,
popular movies therefore speak systematically louder inside fU's
history-composed user representation — an implicit popularity-weighting
channel of the same family as the interaction-weighted prototype-coverage
cloud (F-DC05-02), and the two compound: heavy users AND popular movies both
gain mass. Obligation: mention this wherever ml-1m fU results are presented;
routed to the dc05 hidden-effects section and the SC.8 instrument set
(per-user ESS and mass-share readouts; token-mass axis on the two-weightings
readout).

[[data]] [[decisions]] [[phase-4]]
