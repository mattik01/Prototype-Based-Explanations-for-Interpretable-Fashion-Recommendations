# nl track — sub-problem evaluation: which algorithmic forms deserve a light design-candidate cycle

**Written:** 2026-09-09 (Claude, Fable 5.1), synthesising three literature evaluations written the same day, all citations verified by fetch (⚠️ where not):
`research/item_side_algorithms.md` (38 sources), `research/user_side_algorithms.md` (19), `research/explanation_regimes.md` (35).
**Ground rules applied throughout:** the prototype layer (shifted cosine) is frozen; linearity below it is lifted; the honest test bed is the arm without a free ID row (absorption theorem); every nonlinear arm gets a linear twin and a plain-dot twin (the 2×2); noise bar 2 × 0.0026 HR@10; any positive is re-run at 100 trials; every explanation carries a regime tag; results are read per |H| band with the paired SP0 script, never only overall.

---

## 0. Three structural facts that reorganise the whole menu

These were derived in the evaluations, not taken from papers, and **must be unit-tested before they are quoted** (pre-registered checks in §5).

**F1 — per-token nonlinearity is a re-parametrisation of the free row.** Inputs are categorical with a fixed vocabulary. Any learned map applied to a single token's embedding (MLP, gate, value projection, Deep-Sets φ) takes exactly one value per vocabulary entry, so it is again a free table. The only nonlinearity NOT already inside "sum of rows" is (i) a product/interaction *between* tokens or purchases, (ii) a score for one element that depends on the *other* elements (self-attention), or (iii) a map applied *after* the sum (ρ).

**F2 — element-only attention pooling is invisible to the cosine.** MIL/gated attention, single-seed PMA, ACF component attention without the candidate as query: each element gets a positive weight that depends only on itself, then a softmax. Over a fixed vocabulary the weight is a per-token constant β_v; the softmax denominator is a positive scalar the cosine ignores. So q = Σ β_v e_v = Σ e″_v with e″_v = β_v e_v, a row-norm rescaling the free table already has. Consequences: on the noid arm such attention can change accuracy only through optimisation dynamics, not function class; on the ids arm the normaliser *does* matter because it sets the history-vs-ID norm split (P3). Its explanation object (a printable learned per-token weight table, a learned IDF) is still attractive, and the attention-faithfulness debt collapses to "is the β table meaningful", a cheap test. The user-side analogue: content attention over purchases ≈ one free scalar per item — the same garment gets the same weight for every user.

**F3 — the bag-size / history-length "channel" is two channels.** Norm (‖Σ‖ grows with n): invisible to the cosine, visible only through the ID mix. Direction (the sum drifts toward the common component μ as n grows): visible to the cosine, and the mechanism behind "long histories look generic". Softmax/mean fixes only the first; centring (fI′/fU′ common-component removal) fixes the second; products (bi-interaction) feed both. Hence every nonlinear arm's linear twin must be the **centred** variant, or the arm gets credit for an affine fix any ρ with a bias learns for free.

What the literature does not supply, and our noid arms will: no verified paper reports an interaction module's gain **without** an ID field on categorical recommendation data (NFM, AFM, DCN, AutoInt, xDeepFM evaluations all carry user-ID + item-ID fields); Deep Sets / Set Transformer / MIL papers never sit under a cosine over a fixed vocabulary; no recommender paper runs the full attention-faithfulness battery on history attention.

---

## 1. fI side — item attributes

### 1.1 Sub-problem: within-item field interactions (hm, 5 fixed fields)

Forms evaluated: FM, NFM bi-interaction, AFM, PNN, DCN/DCN-v2, xDeepFM, AutoInt, FiBiNET, HOFM, FFM, explicit pair rows, GA²M-style additive pairwise selection.

Verdicts: DCN-v1 under a cosine degenerates to a learned scalar on the input (a collapse knob, not an interaction). AutoInt's attention explains a post-ReLU vector the cosine never sees as pairs. PNN / xDeepFM / DCN-v2 / FiBiNET are opaque, their published gains are ID-bred, and all are bounded by the 6,517-row tuple-ID table that exists as an arm. HOFM at order 5 *is* the tuple. FFM quadruples the vocabulary with no unary rows. AFM's own ablation (§5.3) shows a linear pair scorer gets most of its gain.

**Gate first:** tuple-ID − fI-noid on hm (HR@10 vs 0.4229, N@10 vs 0.2317). If < 0.005, no within-item interaction of any kind can help on hm; write the null. If ≥ 0.005, in order:

| rank | form | linear twin | dot twin | regime | falsifier |
|---|---|---|---|---|---|
| 1 | **explicit pair rows** for the top-2/3 field pairs (GA²M-style selection) | itself | `lightfm_tags` + same pair columns | exact-linear; units "department ∧ product type" | if pair rows recover < 50 % of the tuple-ID gap, degree-2 conjunctions are not where the gap lives → stop |
| 2 | **NFM-0 bi-interaction pooling** over the existing rows (0 params) | rank 1 | `lightfm_tags_bi` | exact pair summands (e_f ⊙ e_g), a new *unit type*; report unary-vs-pair mass per prototype | must beat rank 1 at equal cost and pair mass share ≥ 10 % of activation, else the units exist but the layer ignores them |
| 3 | **AFM scalar pair weights**, only if rank 2 gains | rank 2 (uniform) | AFM pooling + dot | exact-given-weights | frozen-uniform-weights retrain (AFM §5.3); gap < noise → drop |

One number to fetch on LEO5 before an hm cycle: observed pair values per field pair (bounds the pair-row vocabulary).

### 1.2 Sub-problem: within-bag token interactions / re-weighting (ml-1m, 3–72 tokens)

Forms evaluated: MIL/gated attention, PMA, Deep Sets over tokens, ACF component attention, SIF/IDF fixed re-weighting, bi-interaction over tokens, self-attention (SAB/ISAB), explicit token-pair rows.

Ceiling: bags are near-unique (3.8 % twins), so tuple-ID ≈ `item_proto`; every within-bag form is bounded by item_proto − fI-noid, and any wide encoder that closes it must be shown not to memorise the bag as an item ID.

**Linear first.** The plan's SP2 premise ("every token weighs 1 created the bag-size channel") is answered by a zero-parameter linear change before any nonlinearity:

| rank | form | linear twin | dot twin | regime | falsifier |
|---|---|---|---|---|---|
| 1 | **SIF composition**: IDF-type weights a/(a+p(t)), mean, common-component removal (= fI′ centring with a frequency prior; 3 zero-param sub-ablations) | itself (mean-only / IDF-only / centring-only) | `lightfm_tags_sif` | exact-linear on projected units | success = bag-size-channel correlation drops below 0.3 AND HR@10 ≥ 0.4960 − noise; accuracy loss while the channel closes ⇒ the channel carried signal (a finding) |
| 2 | **bi-interaction over tokens**, mean-of-products | rank 1 | `lightfm_tags_bi` | exact pair summands; pair mass share | **covert-ID test**: gain must appear in the dot twin too; leave-one-token-out at eval degrades a memoriser far more; compare to the ids-arm gain (0.5156 vs 0.4960) |
| 3 | **Deep Sets post-sum residual** q = Σ w_t e_t + ε·ρ(·), zero-init, ε ∈ {0.5, ∞} (SP8 shape) | ε = 0 | same + dot | exact for the linear part; residual = disclosed fidelity loss; Shapley on small bags | same covert-ID test; ε frontier in the dot twin |

Demoted: token-only attention pooling (F2) — keep only as the "learned β vs SIF β" diagnostic if rank 1 works. ISAB self-attention: 2×2 ceiling cell only. Token-pair rows: vocabulary explosion plus covert ID. Candidate-conditioned attention: SP9's business.

Note the tension the vault records: fI′ centring was *rejected* on ml-1m at 100 trials in the ids arm (−0.05) and gained +0.014 in the noid arm (vault 2026-09-08_1509). SIF rank 1 is centring plus IDF weights **in the noid arm only**; its ids-arm result is already known to be negative and is not re-run.

### 1.3 Sub-problem: item-side bottleneck / part prototypes — deferred (needs images; catalogue completeness only).

---

## 2. fU side — purchase history

**Prior the literature imposes:** set-level nonlinearity is not expected to help at |H| ≈ 5. Wagstaff's latent-size bound is trivially satisfied at M = 5, d = 64 (the mean's failure at |H| = 3 is not capacity); MIL attention only matches mean pooling on small bags; NAIS's target-attention gain shrinks to +0.2–1.0 HR points on sparse short-history data. Plan hm as the null and spend hm effort on the *explanation* side (which purchase counted, with a faithfulness number). On ml-1m the literature's biggest lever is **identity** (FISM/SVD++ item-ID history rows: 66.5–70.2 HR@10 with item identity vs our content mean at 61.2, same 1+99 protocol, no user row) — not nonlinearity; NAIS attention adds +1.6–3.2 on top.

### 2.1 Sub-problem: per-purchase weighting from content (SP3)
Forms: MIL/gated attention, ACF item-level attention, RETAIN, FISM (uniform ancestor), per-item scalar. Verdict: under the cosine content attention ≈ a per-item scalar (F2); the mandatory control is fU-noid + one free scalar per item, and the linear twin is fU′ (centred). Explanation regime: exact-given-weights, with the W&P battery and attention-vs-exact-Shapley agreement (instrument exists).

### 2.2 Sub-problem: purchase × purchase interactions (SP4)
Forms: Janossy k = 2, bi-interaction, SAB, co-occurrence models. Verdict: **skip on both datasets.** Janossy's own result: k = 2 helps only when ρ is linear; our frozen 1+cos is a fixed ρ, so pairs are an attribution device, not capacity. On hm pairs are same-day baskets (~10 per user) and bi-interaction ≈ μ ⊙ μ (feeds the collapse). SAB is O(|H|²) (2 M pairs at the ml-1m tail) and lost to max pooling in Set Transformer's own table.

### 2.3 Sub-problem: general set encoder (SP5)
Forms: Deep Sets, PMA. Verdict: **2×2 headroom instrument only**, never a candidate (fails taxonomy row 16 by construction). Deep Sets' universality proof is an injective (memorisation) construction; a deep net over near-unique keys matches an embedding table (DHE) — with 0.10 % basket twins, "noid" is not ID-free once the encoder is nonlinear. For ml-1m d < |H| for half the users, so theory promises no fix at the 223+ tail where both columns collapse. Controls: hashed-history-ID arm (memorisation ceiling), width ≤ 2d, train/val gap, and an H_cap-matched linear control.

### 2.4 Sub-problem: item-ID rows in the history (SP6)
Verdict: **first ml-1m run.** FISM/SVD++ shape, noid arm, FISM-style LOO (sum over R_u⁺ \ {i}), mean pooling; dot twin first: if `lightfm_hist` + identity rows − `lightfm_hist` ≥ 0.02, identity is the headroom and every nonlinear form is second-order. Regime: exact summands, but the rows are unnamed → name-coverage share per prototype mandatory. Skip on hm (the hm ID row is inert, identity carries nothing there).

### 2.5 Sub-problem: target-conditioned attention (SP9)
Forms: NAIS, DIN, ACF. Verdict: last, ml-1m only, against SP3 not against fU. NAIS's gain needed β < 1 smoothing of the softmax denominator, a magnitude trick the cosine cannot see; DIN's unnormalised "intensity" is the same trick. Cost: the per-user profile disappears (contract change under R2/R3, disclosed not averaged). Compute is fine only because eval is 1+99.

### 2.6 Cross-cutting: leave-one-out and norm
LOO everywhere by **mask-then-pool** (FISM/NAIS practice), asserting the positive's slot is absent in the forward, never the algebraic mean subtraction (correct only for the uniform mean). Norm: in the noid-proto cell mean/sum/FISM-α are indistinguishable (scale invariance); in the dot cells first-order; in the ids-proto cell they set the history-vs-ID split per |H| — Steck 2024's arbitrariness applies to that split, not the score. ‖mean‖ shrinks toward ‖μ‖ with |H|, so under mean pooling the ID row's weight grows with |H| (matches fU − noid = +0.01–0.02 in the ml-1m mid bands). **Zero-code read-out first:** ‖q̄_H‖ vs ‖e_ID‖ per |H| band on existing checkpoints.

### Ranked cycles, user side

| dataset | rank | form | twins / controls | first read |
|---|---|---|---|---|
| hm | 1 | MIL/gated attention over purchases, noid | fU-noid + per-item scalar; fU′; `lightfm_hist` + attention | bands 4–5 and 13–24; attention − scalar < bar ⇒ null, keep as explanation instrument |
| hm | 2 | norm read-out + FISM-α in the ids arm (instrument) | fU (α = 1); `lightfm_hist_ids` α sweep | zero-code first, then one α = 0.5 run |
| hm | 3 | Deep Sets zero-init residual — 2×2 cell only | fU′; `lightfm_hist_nl` | plan §3 go/no-go |
| ml-1m | 1 | item-ID history rows (SP6), noid, FISM LOO | fU-noid; `lightfm_hist` + rows (dot first) | ≥ 0.02 in the dot twin ⇒ identity dominates |
| ml-1m | 2 | MIL/gated attention with scalar control | fU-noid + scalar; fU′ | bands 26–56 and 122–222 |
| ml-1m | 3 | NAIS-β target attention (SP9), only if rank 2 moves | FISM-with-content; NAIS-with-content | compare to rank 2 |
| ml-1m | 4 | Deep Sets / PMA residual — 2×2 instrument | fU′ with H_cap-matched linear control | plan §3 go/no-go |

---

## 3. Explanation regimes — what each form can still say

Generalised Score Fidelity: named |φ| mass over total, with a **regime tag** ∈ {exact, exact-given-‖q‖, exact-given-weights, Shapley-with-error}, an error term ε_A, and a support/contradict split (SF⁺/SF⁻; ~40–48 % of purchases carry negative Shapley value in every arm).

| form | regime | mandatory tests | R2 status |
|---|---|---|---|
| linear sum (today) | exact (dot) / exact-given-‖q‖ (proto) | normaliser-disagreement read-out (done: L1 0.44–0.63, top-1 flips 26–28 %, Spearman 0.96 on hm) | intrinsic |
| attention-weighted sum (SP2/SP3) | exact-given-weights | 7-test battery: uniform baseline (= linear arm), seed variance, frozen-weight retrain, adversarial same-score weights, agreement with exact Shapley, polarity, erasure-to-flip | intrinsic decomposition; weights printed only after the battery |
| named bottleneck (SP7) | exact-given-weights downstream; leakage above | re-weighting divergence, leakage probe, cross-seed semantic stability | intrinsic **only with an anchoring loss** (CBM costs +2.4–6.8 error points on CUB without capacity leaks; dc04's lesson) |
| capped residual (SP8) | exact for the named part + one opaque unit of bounded mass | residual share vs accuracy at each cap (the frontier); Shapley of the residual unit | instrument (Part-II frontier tracer) |
| Deep Sets / PMA (SP5/SP6-nl) | Shapley-with-error | declared baseline, coverage, POS/INS/CDCG, IG cross-check, positive-leakage seam | instrument / control only |
| target-conditioned attention (SP9) | exact-given-weights per candidate | battery per (u, i); disclosure that the profile is per-candidate | intrinsic decomposition; the explanation object changes most |

Rules carried into every cycle: the plain-dot twin is the unit test for every attribution instrument (φ must equal linear shares there); baselines are declared, never defaulted (renorm for the shopper sentence, fixed/ID-only for comparability); bottlenecks come with an anchor and a leakage probe or are not named; Steck 2024 travels with every cosine row (exactness of the split says nothing about whether the cosine means similarity).

---

## 4. Consequences for the plan document

- SP2(a) (token attention) and SP3 (purchase attention) are **demoted from capacity candidates to explanation-object candidates** (F2); their controls become the per-item scalar and the centred linear twin.
- SP4 (purchase pairs) is **dropped** on both datasets; SP1(b) AFM becomes conditional on SP1(a).
- SP6 (identity rows) moves **ahead** of SP3/SP5 on ml-1m; it is the literature's expected headroom.
- A **linear ml-1m item-side form (SIF)** is inserted before any item-side nonlinearity.
- Two **zero-code instruments** go first: the norm read-out per |H| band, and the pair-value count per field pair on hm.
- The 2×2's nonlinear cells (Deep Sets) stay, as instruments that bound the headroom, with a hashed-ID memorisation ceiling arm beside them.

## 5. Pre-registered checks before anything is quoted
1. **F2 unit test:** an attention arm and a row-rescaled linear arm reach the same training loss up to optimisation noise (build the attention head, freeze its weights, refit the table; or show the equivalence on a toy).
2. **F3 unit test:** ‖mean‖ vs |H| curve and ID-share per band on existing fU checkpoints (zero code).
3. Tuple-ID gate on hm (2 SLURM runs, blocked on commit).
4. The four hm user-side controls re-run under leave-one-out before any user-side 2×2 row is quoted (decision pending).

## 6. Suggested order of the light design-candidate cycles
1. Zero-code instruments (norm read-out; pair-value counts) — this week, login node.
2. Tuple-ID gate (hm, ml-1m confirmatory).
3. ml-1m user side: identity rows (SP6), dot twin first.
4. ml-1m item side: SIF (linear), then bi-interaction if SIF leaves the bag-size channel open.
5. hm user side: purchase attention with scalar and centred controls — expected null, valuable as the explanation study.
6. The 2×2 Deep Sets cells with the memorisation ceiling arm, both sides.
7. SP9 only if attention moved on ml-1m.

## 7. Open verifications (⚠️ in the source docs)
SIF/Arora 2017 (OpenReview and Princeton pages blocked; local PDF F7 exists — read it), ACF and SVD++ quantitative details (ACM 403; metadata verified only), DIN's use of a user-ID embedding, Castro 2009 error bounds not re-read, GA²M per-dataset deltas not transcribed. AIRec ⚠️ unused.
