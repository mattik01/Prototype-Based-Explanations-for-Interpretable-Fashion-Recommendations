# U2a/U3 composition instrument — results memo (2026-08-24)

Post-hoc, checkpoint-only (working-evidence tier). Arms: fI 4.1 / noid 5.1 hm_1_month
winners (local `expl_deep_dive` copies). Script: `analyze.py`, raw numbers:
`results.json`. Vocabulary: 426 word rows over 5 fields, 13,651 items; `Solid` = 58.1%
of catalog. **All committed anchors reproduced to 2–3 digits** (fI activation eff. rank
1.218 vs sc01's 1.22; noid 1.929 vs 1.93; ID/meta norm ratio median 0.460 vs 0.46; ID
share of ‖q‖² mean 0.310 vs 0.31; Solid linear share 0.315/0.058 vs 0.31/0.058) — same
definitions as the committed geometry instruments, numbers comparable.

## Headline findings

### H1. On the ids arm, the composed metadata vector is MOSTLY the shared component C
fI: ‖C_weighted‖ = 0.209 vs mean ‖q_meta‖ = 0.246 → **85% of the average metadata
composition is the catalog-constant C**; cos(C, mean q) = 1.000. Word norms are tiny
(median 0.063). noid is far less C-dominated (C share 0.45 of mean ‖q_meta‖).

### H2. The Solid omnipresence (M2) is essentially a C artifact, not learned leverage
Solid's raw norm is *small* — 0.102, rank 103/426 (it is NOT a fat vector). Its big
share (0.315 of ‖q‖²) comes from being the dominant chunk of its field's weighted mean,
i.e. of C. Under weighted centring: Solid centred norm 0.007 (≈ the field mean itself),
share 0.315 → **0.022**. noid: 0.058 → 0.008. **Reading for the chapter: the M2
omnipresence stat measures the shared component wearing Solid's name — an uncentred-
gauge artifact of the read-out, not "the model cares about Solid".**

### H3. The activation-cloud collapse splits into two separable phenomena
Post-hoc weighted centring (a gauge change of the geometry, NOT the deployed scorer)
moves fI's activation eff. rank **1.218 → 1.979** (unweighted: 2.005); item pairwise
cos mean 0.422 → 0.113. noid: 1.929 → 2.162, cos 0.121 → 0.006. So the bulk of the
*measured activation collinearity* on the ids arm is attributable to C. **But prototype
duplication is untouched**: mean prototype-profile correlation stays 0.2655 → 0.2647
(fI) — the literal duplicate prototypes (pairwise cos 0.274, p99 = 1.0) are a separate,
real degeneracy that centring does not repair. Two phenomena, previously conflated:
(a) C-driven activation collinearity — removable by a linear gauge/centring; (b)
prototype-table duplication — needs the sim_batch_weight/coverage route (isolating
rerun still the primary experiment).

### H4. The frequency–norm confound is weaker than feared, and field-dependent (U3)
Spearman(norm, log freq) overall fI: 0.348; per field: department +0.49,
product_type +0.06, section +0.23, colour **−0.28**, graphical **−0.17**. Top raw
norms are *seasonally discriminative*, not popular: Swimwear dept 0.696 (n=526),
Womens Swimwear/beachwear section 0.679 — July window signal. After weighted centring
the overall correlation drops to 0.193. **Reading: on this checkpoint, length-as-
leverage is not popularity-swamped at the value level (the popularity mass sits in C,
per H1); centred length is a fairly clean distinctiveness signal.** noid: overall 0.53
raw / 0.375 centred-weighted; top norms are niche sections (Campaigns, Mama,
Womens Tailoring).

### H5. Weighted vs unweighted mean: weighted is the effective flavour
On fI, weighted centring achieves item-cos 0.113 and Solid share 0.022 vs unweighted
0.309 / 0.085 (the unweighted C is bigger in norm but less aligned with the actual
item cloud). SIF/all-but-the-top precedent confirmed in-house: subtract the
*frequency-weighted* mean.

## Consequences for the plan (composition_upgrades_plan_2026-08-24.md)

- **U2b gate: PASSED on the instrument's terms** — C is load-bearing for the measured
  geometry and for the M2 read-out artifact. In-training centring (~15 lines) is now a
  justified designed arm; its open question is what training *with* the centred forward
  pass does to accuracy and to prototype duplication (H3b is not expected to move —
  that stays with the sim_batch_weight rerun).
- **U3 presentation input (thesis-time decision, deferred):** raw fI word norms are
  near-meaningless as "importance" (tiny, C-dominated shares); the candidate read-out
  is **centred-weighted length** with the frequency scatter alongside.
- **sc01 caveat feed:** M2's interpretation (H2) and the collapse split (H3) qualify
  two committed observations; route into the chapter caveats when writing (working
  evidence, not thesis quotes — dedicated experiments at writing time per policy).

## ml-1m reference (standing rule 2026-08-24: MovieLens always included as reference)

Arms 4.2 (fI, d=80, K=77) / 5.2 (noid, d=81, K=76); bags layout (genres+tags,
1,061 tokens, 3,125 items, mean bag size 12.3). On bags the centring shift is
item-dependent (std > 0), "shared in expectation" only.

- **H1/H2 replicate directionally but MUCH weaker:** shift is 33% (fI) / 42% (noid) of
  the mean metadata composition — vs hm-fI's 85%. The omnipresent token (`Drama`,
  39.6% of items, norm rank 291/1061 — again not a fat vector) drops share
  0.197 → 0.121 (fI) and 0.360 → 0.084 (noid) under weighted centring. The extreme
  C-dominance and the full Solid artifact are thus substantially an
  hm/fixed-layout phenomenon — the reference kept the claim honest.
- **H3 replicates:** eff. rank 1.775 → 1.974 (fI), 1.488 → 1.950 (noid); item cos
  0.191 → 0.016 / 0.270 → −0.001. Note the hm ordering *reverses*: on ml-1m the noid
  arm is the more collapsed one raw. And fI_ml's prototypes are NOT duplicated
  (profile corr −0.002 vs fI_hm's 0.266) — more evidence collapse is data×config,
  not candidate-caused.
- **Two ml-1m-specific channels quantified:**
  (a) **bag-size vote mass:** corr(bag size, ‖q_meta‖) = 0.80 (fI) / 0.58 (noid) —
  many-tag movies get big compositions (the S0.5-addendum must-mention channel, now
  measured);
  (b) **singleton tags act as covert per-item IDs** — top norms are count-1..35 tags
  (noid_ml top: "sad but good", n=1, norm 1.17): a tag carried by one item IS a
  private ID row. **noid_ml is not truly ID-free** — read ml-1m noid contrasts with
  this in mind (candidate relevance for the −0.0555 ml-1m regression reading).
- **Frequency–norm correlation is NEGATIVE on ml-1m** (overall −0.21 fI / −0.39 noid;
  tags −0.22/−0.38): rare tokens carry the big norms. The NISER worry inverts here —
  further support that the length read-out must always ship with the frequency
  scatter. fI ID stats for reference: ID/meta ratio median 0.247, ID share 0.126
  (both ≈ half the hm values).

Caveats: single checkpoint per arm (hyperopt winners), post-hoc gauge (the deployed
model still scores with raw cosine — ranking impact of C is NOT measured here),
working-evidence tier. hm + ml-1m both covered (standing ml-1m-reference rule).
