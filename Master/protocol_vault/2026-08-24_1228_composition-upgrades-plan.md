---
date: 2026-08-24
time: "12:28"
phase: 4
---
# Composition scrutiny: deep research, four upgrade decisions, first instrument results

Question (Matteo): is dc01's unweighted sum of attribute-word embeddings the right way
to compose an item, and what are the alternatives? Four-agent literature sweep
(synthesis: `Master/temp/composition_deep_research_2026-08-24.md`), then in-model
mathematical scrutiny + decisions (`Master/temp/composition_upgrades_plan_2026-08-24.md`,
PLANNED NOT QUEUED), then the U2a/U3 instrument run
(`Master/temp/composition_analysis_2026-08-24/`).

**Research verdict:** the sum is the canonical choice (LightFM — built at Lyst for
fashion — uses exactly this design and its Table 1 supports the ID+attributes sum warm
AND cold); the literature's critique targets the *uniform weighting*, not additivity
(AFM/FwFM/FiBiNET/DIN); frequency pathologies are the theory-backed risk (additivity
provably exact only under uniform value frequencies — Gittens 2017; common-direction
inflation — SIF/all-but-the-top; norm-popularity — NISER); everything input-dependent
(attention pooling) or deep forfeits exact attribution (attention-is-not-explanation
verdict; GA²M is the precedent that named pair terms stay faithful); gains beyond tuned
low-order structure are small under fair benchmarking (Zhu 2021, Rendle 2020).

**Decisions (Matteo, 2026-08-24):**
- **U1 ID-dropout training: ADOPT.** Randomly drop the ID summand during training so
  train matches the cold drop-the-ID rule. Cited (DropoutNet/Heater) + observed
  (F-DC01-04; cold-vs-all fI 0.218 vs noid 0.394; ID mass 0.31 of ‖q‖² / norm ratio
  0.46 vanishing at cold). ~10 lines + one knob ρ; inference and all read-outs
  byte-identical. Cosine bonus: no 1/(1−ρ) rescale needed.
- **U2 field-mean centering: ADOPT, instrument-first.** Key math: with one value per
  field, every item's composition contains an EXACTLY shared component C = Σ_f μ_f;
  it is pure baseline mass in the numerator but rewards prototypes for aligning with
  it (collapse seed), and field-mean subtraction is simultaneously the gauge fix for
  the per-field share ambiguity (sI fork's identifiability result). Instrument gate
  PASSED same day (below) → in-training centred variant is a justified designed arm.
- **U3 norm-as-weight read-out: no model intervention.** Redundancy proof: under
  cosine with free norms (hyperopt selected no max_norm), learned scalar weights are
  exactly absorbable into embedding norms — the model already weights words via
  ‖e_v‖; the gap is that nothing renders it. Presentation/interpretation (raw
  "leverage" vs frequency-residual vs centred "distinctiveness") explicitly DEFERRED
  to thesis-writing time, decided with instrument outputs in hand.
- **U4 FM-style pair terms: PARKED, explicitly recorded.** Conjunctions are provably
  inexpressible (prototype response affine in word indicators — no AND semantics),
  but the only observed deficit is the unattributed ml-1m −0.0555 (main benchmark —
  correction Matteo: hm parity alone was an overstated basis). Revisit triggers
  recorded (ID-pair-encoding instrument; regression attribution; fUfI merge). If ever
  adopted: selected pairs, constant coefficients, Lengerich identifiability disclosure.

**U2a/U3 instrument results (working evidence; all 5 committed anchors reproduced —
eff. rank 1.218/1.929, ID ratio 0.460, ID share 0.310, Solid shares 0.315/0.058):**
- hm fI: C is **85% of the mean metadata composition**; the M2 "Solid omnipresence"
  (share 0.315) is essentially the shared component wearing Solid's name — Solid's own
  vector is small (rank 103/426); centred share 0.022. **Gauge artifact, not learned
  leverage** → feeds the sc01 caveats at writing time.
- The activation-cloud collapse SPLITS into two phenomena: C-driven collinearity
  (post-hoc centring lifts eff. rank 1.218 → 1.979) vs prototype-table duplication
  (profile correlation unchanged 0.266 → 0.265) — the latter stays with the
  sim_batch_weight isolating rerun. Weighted mean is the effective centring flavour.
- Frequency–norm coupling weaker than feared on hm (spearman 0.35; colour/graphical
  negative; top norms are seasonal Swimwear signal).
- **ml-1m reference (new standing rule: ALWAYS include MovieLens as reference):**
  effects replicate directionally but weaker (shift 33–42% of composition vs 85%;
  Drama share 0.20 → 0.12); collapse ordering reverses (noid more collapsed raw);
  fI_ml prototypes NOT duplicated (corr ≈ 0) — collapse is data×config, not
  candidate-caused. New: bag-size vote-mass channel corr 0.80 (own vault entry);
  **singleton tags act as covert per-item IDs → noid_ml is not truly ID-free**
  (relevant to reading the ml-1m regression); frequency–norm correlation NEGATIVE
  on ml-1m — the length read-out must always ship with the frequency scatter.

[[decisions]] [[experiments]] [[explanations]] [[phase-4]]
