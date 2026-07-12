# dc05 SC.2 claims spec — black-box toy re-check (addendum round)

You receive ONLY this spec. Do not read any other project document, including
the original claims spec or its check script.

## Mechanism definition (self-contained)

Vocabulary: F categorical fields; field g has V_g values; total V = Σ V_g
"word" rows. Word embedding table E ∈ R^{V×d}, entries i.i.d. N(0,1), FROZEN
(no training anywhere in this check). Each item carries exactly one value per
field (F word tokens per item).

A user u has a basket H_u of items. Weights: w̄_{u,f} = n_{u,f}/|H_u| where
n_{u,f} = number of basket items carrying word f. Composed metadata vector:
m_u = Σ_f w̄_{u,f}·E[f]. Optionally a per-user ID row e_ID(u) ∈ R^d
(i.i.d. N(0,1), added with weight 1): q_u = m_u + e_ID(u).

Heterogeneous population model: per field g, a population distribution
p_pop^g over its V_g values (draw once per seed from a flat Dirichlet).
Per-user taste: p_u^g ~ Dirichlet(α·V_g·p_pop^g), independently per field —
the concentration α > 0 controls heterogeneity (small α = idiosyncratic
users far from the population profile; large α = users nearly identical to
it). A user's items are drawn i.i.d.: each item's value in field g ~ p_u^g.
User centroid: c_u = Σ_g Σ_{f∈g} p_{u,f}·E[f].

Suggested toy dims: F = 5 fields, V_g ≈ 15–30, d = 32, ≥ 300 users per
condition, several seeds, float64. Angles in degrees; "angle between users"
means the angle between their vectors (composed m_u or centroids c_u).

## Claims (each: confirmed / falsified / untestable-at-toy-scale, with numbers)

**C10′ — crowding vs noise under a HETEROGENEOUS population (trend claims,
no ID row, i.e. q_u = m_u).** For basket sizes |H| ∈ {3, 10, 30, 100, 300}
and concentrations α ∈ {0.2, 2, 20}:
(a) *Noise half:* mean angle(m_u, c_u) — each user's composed vector to
    their OWN centroid — decreases monotonically with |H|, for every α.
(b) *Individuation half:* the mean pairwise angle between distinct users'
    composed vectors converges, as |H| grows, toward the mean pairwise angle
    between their centroids (and NOT toward 0 when the centroids are
    separated) — report both curves per α.
(c) *Crowding half:* the asymptotic between-centroid mean pairwise angle
    DECREASES as α increases (population homogenizes) — crowding is governed
    by the taste-heterogeneity axis, not by |H| alone. Report the
    between-centroid mean pairwise angle per α.
(d) *Joint signal-to-noise:* SNR(|H|, α) = (mean pairwise angle between
    users' composed vectors) / (mean angle to own centroid) increases
    monotonically with |H| for every α. Report the SNR table.

**C11′ — norm handover (trend claims, fixed α = 2).** With the ID row
enabled:
(a) mean ‖m_u‖ decreases monotonically with |H| and asymptotes to the mean
    centroid norm ‖c_u‖ (positive — NOT toward 0). Report mean ‖m_u‖ per
    |H| ∈ {3, 10, 30, 100, 300} and mean ‖c_u‖.
(b) the ID row's share of the composition, measured as
    mean ‖e_ID(u)‖² / (‖m_u‖² + ‖e_ID(u)‖²) (and/or mean ‖m_u‖/‖e_ID(u)‖),
    increases (resp. decreases) monotonically with |H|. Report the curve.

## Deliverable

A toy script `check_claims_sc2.py` in `Master/temp/dc_checks/dc05/`
(numpy/torch, float64, toy dims, seconds of runtime, no training, no
datasets), plus a per-claim report: confirmed / falsified /
untestable-at-toy-scale, with the measured numbers (tables over |H| × α).
Use ≥ 3 seeds; note any non-monotonicity however small. Python env:
`conda activate protomf` (numpy/torch available, CPU fine).
