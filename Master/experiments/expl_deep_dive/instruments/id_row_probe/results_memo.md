# ID-row probe — cross-arm memo (2026-09-07)

**Tier: working evidence.** Single checkpoint per arm, post-hoc on the saved embedding
table; never thesis numbers (scrutiny-evidence rule). Script: `Master/scripts/id_row_probe.py`
(commit 6a0baed); per-arm `readout.json`/`readout.md` in the sibling folders; runs on the
LEO5 login-node A30s (54 s hm, ~200 s ml-1m).

Anchor check: all four arms reproduce the `composition_analysis_2026-08-24` numbers exactly
(hm fI: ID/meta ratio 0.4604, ID share 0.3096, eff. rank 1.2178; ml-1m fI 30-trial: 0.2471 /
0.1256 / 1.7754).

## Arms

| arm | checkpoint | tier |
|---|---|---|
| hm fI | `feature_item_proto_hm_1_month_s38210573` (sc01 4.1) | dev, 30 trials |
| ml fI n100 | `feature_item_proto_ml-1m_s38210573_n100` | paper budget, 100 trials |
| ml fI n30 | `feature_item_proto_ml-1m_s38210573` (sc01 4.2) | dev, 30 trials |
| ml fI′-ids | `feature_item_proto_c_ml-1m_s38210573` (dc06, weighted centring + gauge snap) | dev, 30 trials |

## A — pair encoding (U4 trigger i): COLD in substance on every arm

| arm | R² unary → +pairs (e_ID) | ΔR² | null p95 | rule |
|---|---|---:|---:|---|
| hm fI | −0.004 → −0.003 | +0.0006 | −0.0003 | "hot" on sign only |
| ml fI n100 | −0.008 → −0.018 | −0.011 | −0.003 | cold |
| ml fI n30 | −0.004 → −0.011 | −0.007 | −0.001 | cold |
| ml fI′-ids | −0.007 → −0.015 | −0.009 | −0.001 | cold |

Reading. The ID row is **not linearly predictable from the attribute indicators at all** —
held-out R² is zero on unary indicators and stays zero with up to 5000 pair columns, on
every arm. On hm the pre-registered rule fires technically (ΔR² +0.0006 clears a null p95 of
−0.0003; with 20 permutations the minimum p is 1/21 = 0.048), but the explained variance is
nil, so the pre-registered rule lacked a magnitude floor and "hot" here is a sign artefact.
Verdict per the reading rule's intent: **E3 (no conjunctions) stays a cited, not observed,
limitation; U4 does not earn a candidate cycle on this evidence.** Interpretation: whatever
e_ID carries is orthogonal to the attribute vocabulary's linear span — consistent with the
2026-07-11 "residual = what attributes cannot express" framing and with the popularity
reading in B. The secondary response (ID contribution to prototype activations) shows a
small consistent pair effect (ΔR² +0.004 … +0.012, all above null) — real but at R² ≈ 0.02,
not decision-relevant.

Top hm pairs by coefficient norm are sensible cross-field conjunctions (Sweater×Melange,
Dress×All-over-pattern, Divided×Black) but with coefficients ~0.003 on ‖e_ID‖ median 0.061.

## B — popularity-stratified ID share: the head/tail split is real, and dataset-shaped

Energy-level ID share `e_ID·q/‖q‖²` per train-popularity bucket:

| arm | 1–10 | 11–100 | 101+ | Spearman(pop, share) | D2 |
|---|---:|---:|---:|---:|---:|
| hm fI | 0.19 | 0.33 | **0.84** | +0.49 | 0.007 |
| ml fI n100 | 0.12 | 0.13 | 0.14 | +0.14 | 0.069 |
| ml fI n30 | 0.11 | 0.13 | 0.13 | +0.12 | 0.086 |
| ml fI′-ids | **0.52** | 0.39 | 0.19 | −0.48 | **0.494** |

- **hm: "the model mostly uses the ID" is TRUE for the head, FALSE for the tail.** Items
  with >100 train interactions carry 84 % of their energy in the ID row (norm ratio 1.9);
  tail items 19 %. The head's mean activation is 0.60 (< 1, i.e. anti-aligned with the
  prototype cloud) vs 1.51 for the tail, and the ID row's activation-level share is −1.2:
  the popular item's ID row points *against* the collapsed prototype direction. This is the
  self-built popularity channel (vault 2026-07-15_1112) located in the ID block, per item.
- **ml-1m fI (both tiers): flat.** ID share ≈ 0.13 in every bucket; direction flips
  (tail with the prototypes, head against) but magnitudes are small. On ml-1m the ID row is
  a mild corrective, not the carrier. Tier (30 vs 100 trials) changes nothing here.
- **ml-1m fI′-ids: centring REVERSED the profile.** ID share 0.52 on the tail, 0.19 on the
  head; overall 0.32 (vs 0.13 uncentred). And **D2 = 0.494**: half of the mean ID-row norm
  is a shared direction, cos 0.67 with the metadata common direction — the dc06 D2 risk
  ("common direction re-emerges in the ID block") is **realised** on the ids arm. The
  centred metadata sum shrinks for long-bag items (residuals cancel), so the ID row
  dominates exactly where the bag is short/tail — see C.

## C — bag-size channel (ml-1m only): active, but the direction story is inverted

| arm | bag↔pop | bag↔mean act. | bag↔ID share | bag↔‖q_meta‖ |
|---|---:|---:|---:|---:|
| ml fI n100 | 0.67 | −0.28 | −0.37 | 0.85 |
| ml fI n30 | 0.67 | −0.36 | −0.37 | 0.80 |
| ml fI′-ids | 0.67 | **−0.69** | **−0.72** | 0.87 |

- Tag count is a popularity proxy (0.67, identical to the S0.5 ρ = 0.672 on the split).
- The vault's layer-2 prediction ("many-tag movies drift to the centre and get elevated
  similarity to many prototypes") is **not** what the geometry shows: big bags are *less*
  activated on average, mildly on fI and strongly under centring. The channel is active,
  but as a *de-activation* of heavily-tagged (popular) items — the same sign as hm's head.
- The ID-share meter confound (vault bite #1) is confirmed at −0.37 / −0.72: bag size must
  be disclosed next to any per-item ID share on ml-1m.

## What this settles / opens

1. U4 (FM pair terms): no observed conjunction encoding in the ID row → stays parked.
2. Taxonomy row 13 measurement exists: hm head/tail split 0.84 / 0.19 is the number the
   2026-07-11 vault asked for; on ml-1m the ID row is not the carrier at any popularity.
3. dc06 D2 risk realised on fI′-ids (0.49 vs 0.07 uncentred). The α-control row (dc06 A1)
   and a 100-trial fI′-ids remain the designed follow-ups; the D2 read-out is now one
   command on any checkpoint.
4. The bag-size channel's *sign* needs the vault entry amended (de-activation, not
   elevation) — thesis hidden-effects section input.
5. Reading-rule lesson: pre-register a magnitude floor with any permutation test.
