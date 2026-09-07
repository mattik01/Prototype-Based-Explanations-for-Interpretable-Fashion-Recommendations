# Composition upgrades — in-model scrutiny & plan (PLANNED, NOT QUEUED)

> **DESTINATION (Matteo, 2026-08-24): this plan feeds a full `/design-candidate` cycle
> for a candidate provisionally named fI′ (fresh session)** — **fI′ = U2b field-mean
> centring ONLY** (single-variable candidate, knobs on `feature_item_proto`, new
> default if it wins). U1 ID-dropout DEFERRED (mention/future; ids-vs-noid endpoints
> suffice); U3 read-outs DEFERRED alongside (both dropped from fI′ scope, Matteo
> 2026-08-24 — the U2a instrument outputs remain available for the thesis-time
> presentation decision); U4 parked. Goal: reduce fI's
> deficit vs the ProtoMF host and LightFM, with ml-1m as
> the reference benchmark (standing rule). Thesis-grade ml-1m benchmarks are being
> queued on the cluster in parallel (separate sessions, /leo5-submit discipline).
> Inputs for the cycle: this plan, `composition_deep_research_2026-08-24.md`,
> `composition_analysis_2026-08-24/results_memo.md`, vault entries 2026-08-24_1228/1229.

> 2026-08-24. Follow-up to `composition_deep_research_2026-08-24.md`. Task (Matteo):
> scrutinize the four literature-flagged effects *inside our model* (mathematically where
> needed), state what we already do, then decide a small subset of additions — for each:
> precise change + size, justification (cited AND observed-in-own-experiments where
> possible), which issue it targets, and why explanations stay intrinsically intact.
> **Status: plan only. Nothing implemented, nothing queued.** Any adoption is a
> post-SC.11 variant wave on a closed candidate (same footing as the designed-not-queued
> popularity wave), not an edit to the scrutinized dc01 arms.

Model under scrutiny (dc01 / `feature_item_proto`, `FeatureEmbedding` fixed layout):

    t_i = Σ_f e(x_if) + e_ID(i)          (one value per field, F fields, all weight 1.0)
    t*_k(i) = 1 + cos(t_i, p_k),  s(u,i) = Σ_k u_k t*_k(i)

Relevant code facts: `nn.Embedding(..., max_norm=None)` — the dc01 hm winner's hyperopt
had max_norm available and **selected none** (sc01 §4.4), so row norms are fully free.
Bag layout already supports per-token weights (buffer `feature_weights`).

---

## 1. Scrutiny of the four effects inside OUR model

### E1. Uniform weighting — the critique only PARTIALLY transfers (redundancy proof)

Claimed issue (AFM/FwFM/FiBiNET/DIN): unweighted sums let useless features pollute the
composition; learned weights fix it.

**In our model, per-value constant weights are capacity-redundant.** A weighted sum
`Σ_f w_{x_if}·e(x_if)` is exactly reparameterized by `e'_v := w_v·e_v` — and since norms
are unconstrained (no max_norm), the optimizer already owns this degree of freedom:
**the summand's norm IS its learned weight** (under cosine, only the direction of t_i
matters, and each summand's vote in that direction is proportional to its norm). A
per-field weight w_f is likewise absorbable (scale all rows of field f). So adding
learned scalar weights changes the *prior/optimization path*, not the function class.
The same argument covers the ID row: its weight-1.0 is not "silently fixed" — the model
tunes the ID/metadata balance through ‖e_ID‖, and we have *measured* it doing so
(ID/metadata norm ratio median 0.46; ID linear share of ‖q‖² mean 0.31, sc01 §4.3).

**What actually transfers:** (i) the weighting is learned but **unread** — no read-out
currently renders ‖e_v‖ as "the model's learned importance of word v"; (ii) nothing
regularizes or structures the norms, so frequency drives them (→ E2); (iii) the
observed M2 omnipresence — `Solid` (58% of catalog) mean linear share **0.31** on fI vs
**0.058** on noid — is the AFM pathology realized *through* the free norms, not despite
them. Verdict: the fix is a **read-out + instrument**, not new parameters. (A
non-redundant weights variant exists — unit-normalize word rows via max_norm=1 and add
explicit scalars, separating direction from magnitude — kept as an optional geometry
probe, not a default.)

### E2. Frequency / common-direction pathologies — transfers EXACTLY, with a twist

Decompose each word row against its field mean: `e_v = μ_f(v) + δ_v`,
`μ_f := mean over the catalog's field-f values (frequency-weighted)`. Then, because the
fixed layout gives every item exactly one value per field:

    t_i = C + Σ_f δ_{x_if} + e_ID(i),        C := Σ_f μ_f   — **identical for every item**

The shared component C is not an approximate average (SIF's c₀) but an *exactly* shared
vector. Its effects, precisely:
- **Numerator:** `t_i·p_k = C·p_k + (variable)·p_k`. The `C·p_k` term is
  item-independent per prototype — pure baseline mass, useless for ranking — yet it leaks
  item dependence through the item-varying denominator ‖t_i‖, and it rewards prototypes
  for aligning with C: activation mass is cheapest along the common direction. That is a
  mechanism-level seed for the observed near-collinear activation cloud (eff. rank 1.22
  fI / 1.93 noid / 1.14 f0-host, sc01 §4.4) — though NOT the sole cause: the host arm
  (f0, no word sums) collapses too, and the sim_batch_weight probe (vault
  2026-08-17_1407) shows regularizer-configuration dependence. Honest claim: composition
  adds a *structural* common component on top of a collapse pressure that already exists.
- **Norm growth:** e_v receives gradient from every item carrying v; with free norms,
  expected update mass scales with value frequency (NISER/Gao mechanism). Popular-value
  rows grow and dominate sum directions — `Solid` 0.31 is the measured footprint.
- **Gauge tie-in (own result):** the sI fork's identifiability toy showed per-field
  offsets are exactly the unidentified directions of attribute profiles. Field-mean
  centering is the canonical gauge fix — the same operation resolves the share-rendering
  gauge AND removes C. Two open questions, one linear operation.

**What we already do:** the shifted cosine normalizes the *composed* vector (item-level
scale invariance) but nothing normalizes or centers the *summands*; R_batch
(`sim_batch_weight`) is the existing anti-collapse knob (effective, per the probe) but it
fights the symptom in activation space, not the seed C in embedding space. Complementary,
not redundant.

### E3. No conjunctions — real in the math, UNOBSERVED as a deficit

Pre-normalization, prototype response is affine in the word indicators:
`t_i·p_k = Σ_v x_iv (e_v·p_k) + e_ID·p_k`. Hence no prototype can respond
super-additively to a conjunction — "fires for dark∧denim but not for either alone" is
inexpressible (the denominator adds only weak, unstructured curvature). Conjunction
taste can only live in u (spread across word-aligned prototypes) or in e_ID (per-item,
unnamed). Same conclusion the wI fork reached independently ("conjunction moves up").

**Observed evidence of harm: one candidate, unattributed.** On ml-1m — the main
host-benchmarking dataset (Matteo, plan discussion 2026-08-24) — fI sits **−0.0555 below
the host**, cause unattributed; hm shows warm parity. So "no observed harm" is too
strong: there IS an observed accuracy deficit on the reference benchmark, and the
conjunction gap is a *candidate* explanation for it (not an attribution). The
literature's +7.3% (NFM over FM) is a different scorer and task. By the "cited AND observed" standard this
issue currently has citation only — and the fix (selected pair-value vocabulary,
GA²M-style) grows the explanation object, needs the Lengerich (AISTATS 2020)
main-vs-interaction identifiability disclosure, and overlaps the lattice-node/gI
territory of the grounding pass. → Defer with trigger conditions (§2, U4).

### E4. Cold-path train/inference mismatch — transfers exactly, STRONGLY observed

Training always composes with e_ID; the cold rule deletes it. The deleted summand
carries median **0.46** of the metadata mass (norm ratio) and mean **0.31** of ‖q‖²
(linear share) — so a cold item is evaluated at an operating point the training
distribution never contained (the named finding **F-DC01-04**, "operating-point
mismatch"). The two existing arms are the endpoints of exactly this axis: ids (never
drop, ρ=0) → cold-vs-all **0.2179**; noid (always dropped, ρ=1) → **0.3935**, with noid
additionally keeping warm parity on hm. ID-dropout (drop e_ID with probability ρ per
item per batch during training) interpolates: the model must learn compositions that
survive the drop, while keeping ID capacity for warm individuation (ids' twin q-cos
0.58 vs noid's structural 1.0). A cosine-specific bonus: **no inverse-scaling
compensation is needed** (dropout's usual 1/(1−ρ) rescale is irrelevant — the remaining
sum's scale cancels in the cosine), so the mechanism is cleaner here than in
dot-product models. This is DropoutNet's training principle applied at the single
summand our cold rule deletes.

---

## 2. Decision: the adopted subset

Chosen: **U1 (adopt), U2 (adopt, instrument-first), U3 (adopt, read-out only),
U4 (defer)** — i.e., one training change, one instrument-gated variant, one pure
read-out, one recorded deferral.

Discussion outcomes (Matteo, 2026-08-24): **U1 confirmed yes. U2 confirmed yes —
U2a analysis to run immediately after the plan discussion (checkpoints turned out to be
local → run locally; pure table algebra, no GPU benefit).
U3 resolved: no model intervention; the presentation/interpretation question (raw
length = "leverage" vs frequency-corrected residual vs centred length = within-field
distinctiveness) is DEFERRED to thesis-writing time, decided with the U2a/U3 instrument
outputs in hand. Centering and the frequency correction are two routes to the same
cleanup (geometry-side vs display-side); centering precedent: Mu & Viswanath 2018,
SIF 2017, Gao et al. 2019. **U4 confirmed parked, explicitly recorded**; the E3 harm wording
corrected — ml-1m (main benchmark) deficit −0.0555 stands as an unattributed candidate
observation.

### U1 — ID-dropout training (targets E4). ~~ADOPT~~ → **DEFERRED (Matteo, 2026-08-24):
mention / possibly test in the future — NOT part of fI′.** Rationale: the ids-vs-noid
arms already bracket the axis (ρ=0 / ρ=1 endpoints); the interpolating training trick
is not needed for the current contrast and would drag the matched-control obligation
(LightFM+dropout arm) with it. Thesis treatment: mention as a designed, literature-
backed repair (DropoutNet/Heater) for the F-DC01-04 operating-point mismatch, left as
future work. The fair-comparison analysis below (matched control, architecture-
dependent mismatch reversal) is RETAINED as the design spec for if/when it is tested.
Original adopt-case, kept for that future decision:
- **Change, precisely:** in `FeatureEmbedding.forward`, training-mode only: Bernoulli
  mask (prob ρ) zeroing the ID summand per item per batch — in the fixed layout, mask
  the ID column's contribution before `.sum(dim=-2)`; in the bag layout, set the
  appended ID weight to 0 with prob ρ. One new constructor arg `id_dropout` threaded
  through `feature_extractor_factories.py`; one new hyperopt dimension in
  `confs/hyper_params.py` (ρ ∈ {0, 0.1, …, 0.5} or continuous). **~10 lines of model
  code + config plumbing.** Not a new `ft_type` — a knob on `feature_item_proto`
  (ablation arms ρ=0 and ρ=1 already exist as ids/noid). Modifications-log entry due
  (`feature_extractors.py` is upstream-derived).
- **Justification:** cited — DropoutNet (Volkovs et al., NeurIPS 2017: train for cold
  by dropping the preference/ID input), Heater's randomized training (SIGIR 2020);
  observed — F-DC01-04, the 0.218-vs-0.394 cold-vs-all gap, ID mass 0.31/0.46 vanishing
  at the drop. Strongest cited+observed pairing of the four.
- **Issue targeted:** E4 — close (part of) the fI→noid cold gap without giving up warm
  ID individuation.
- **Fair-comparison requirement (Matteo, 2026-08-24):** `lightfm_tags_ids` has the
  SAME train/cold operating-point mismatch (trains with ID, drops at cold) and no
  alignment mechanism; it shares the `FeatureEmbedding` class, so the ρ knob is free
  for it. The fI′ cycle must include a **`lightfm_tags_ids` + ID-dropout matched
  control arm** — fairness (both sides of the cold headline comparison get the same
  training technique) + attribution (equal lift ⇒ generic trick; disproportionate fI′
  lift ⇒ property of the prototype route; the observed reversal — LightFM ids HELP
  cold 0.508 vs 0.470, fI ids HURT cold 0.218 vs 0.394 — hints the mismatch is
  architecture-dependent). Host `item_proto`: not applicable (no content route);
  `lightfm_tags`/noid: structurally immune (ρ=1 endpoints). Frozen S0 rows untouched —
  these are new arms under equal treatment. Also decided: fI′ = knobs on
  `feature_item_proto` (new default if it wins), NOT a new ft_type.
- **Explanations intact:** inference forward pass is byte-identical; every read-out
  (profiles, shares, ID-share meter) unchanged. If anything the explanation *improves*:
  training pressure pushes score-relevant structure from the unnamed e_ID into named
  word rows, which the existing ID-share meter will show — the meter doubles as the
  effect's own instrument. ρ is Rashomon-tunable (requirements §4 discipline).

### U2 — Field-mean centering (targets E2). ADOPT AS INSTRUMENT; variant gated on it.
- **U2a instrument (no training, first step):** post-hoc analysis on existing
  checkpoints (~30 lines in `utilities/explanations_utils.py`-adjacent script): compute
  μ_f from the checkpoint's word table, re-render prototype word-profiles, share
  decompositions, and geometry stats (eff. rank, pairwise prototype cos, `Solid` share)
  under `e'_v = e_v − μ_f(v)` vs raw; plus the E1/U3 norm-vs-frequency scatter.
  **Both mean flavours** per field (frequency-weighted and unweighted — any per-field
  constant is a valid gauge since every item has one value per field; the weighted mean
  is the SIF/all-but-the-top precedent and targets the popularity direction, e.g. it
  nearly erases `Solid`; the unweighted mean measures distance from the field's
  value-typical word). One extra column each. Note:
  centred rendering is a *declared gauge choice* of the read-out — label it as such.
  Login-node class. Standing directive applies: scrutinize the graphics.
- **U2a RUN 2026-08-24 — gate PASSED** (see `composition_analysis_2026-08-24/
  results_memo.md`): C is 85% of the mean metadata composition on fI; post-hoc centring
  lifts activation eff. rank 1.218 → 1.979 and exposes the Solid/M2 omnipresence as a
  C artifact (share 0.315 → 0.022); prototype duplication untouched (separate
  phenomenon, stays with the sim_batch_weight route). Weighted mean is the effective
  flavour.
- **U2b variant (gate passed; now a justified designed arm):** subtract a
  per-field-mean buffer inside the forward pass —
  `t_i = Σ_f (e(x_if) − μ_f) + e_ID`, μ_f recomputed differentiably per step (one
  V×d-scale reduction, negligible; epoch-refreshed buffer as fallback). **~15 lines**,
  no new hyperparameters (or one: center on/off as an ablation switch).
- **Justification:** cited — Mu & Viswanath ICLR 2018 (all-but-the-top), Arora et al.
  ICLR 2017 (SIF common-component removal), Gao et al. ICLR 2019 (cone collapse);
  observed — eff. rank 1.14–1.93 collapse, `Solid` 0.31 omnipresence, plus the exact-C
  structure derived in §1.E2 and the sI gauge result. Honest scope: collapse is also
  regularizer-dependent (the queued-but-not-run sim_batch_weight isolating rerun stays
  the primary collapse experiment; centering is complementary — seed vs symptom).
- **Issue targeted:** E2 — remove the structurally shared component that subsidizes
  collapse and pollutes cosine geometry; fix the per-field share gauge at the same time.
- **Explanations intact:** U2a is read-out-only (a disclosed gauge). U2b keeps every
  rendered number a forward-pass quantity — shares decompose over centred word vectors
  exactly as before (`(t_i)·p_k = Σ_f δ'·p_k + e_ID·p_k`, one named term per word), with
  one new disclosed line ("shared catalog component removed: C"). Fully linear — the
  attribution contract is untouched.

### U3 — Norm-as-weight read-out (targets E1). ~~ADOPT~~ → **DEFERRED (Matteo,
2026-08-24): dropped from fI′ scope along with U1 — fI′ is centring-only.** The
one-off instrument numbers already exist (`composition_analysis_2026-08-24/`) and
feed the thesis-time presentation decision; a standing read-out on the explanation
cards waits. Original case, kept:
- **Change, precisely:** ~20-line instrument + one line on the explanation cards:
  render ‖e_v‖ (per word, grouped per field, vs value frequency) as "the model's
  learned word weights", and ‖e_ID(i)‖/Σ‖e‖ next to the existing ID-share meter.
  No parameters, no training, no new knob.
- **Justification:** cited — the AFM/FwFM/FiBiNET line says weighting matters;
  §1.E1's redundancy proof says our model already *has* the weights (as norms) —
  we're just not reading them. Observed — Solid 0.31/0.058 and ID-ratio 0.46 were
  produced by exactly this channel. The optional non-redundant variant
  (max_norm=1 + explicit scalars) is recorded but NOT adopted — it constrains geometry
  for no demonstrated benefit, and the ids-arm twin individuation explicitly relied on
  hyperopt *rejecting* max_norm.
- **Issue targeted:** E1 — turn the implicit weighting into a first-class, disclosed
  explanation object.
- **Explanations intact:** trivially — pure read-out of forward-pass parameters.

### U4 — FM-style pair terms (targets E3). DEFER — recorded, not adopted.
- **Why deferred (decision Matteo 2026-08-24: parked, definitely recorded):** the
  weakest cited+observed pairing of the four — the ml-1m −0.0555 host deficit (main
  benchmarking dataset) is real but unattributed to composition expressivity, and no
  instrument has yet tied it to the conjunction gap. Cost side is real: a selected
  pair-value vocabulary (explanation-object growth), the Lengerich main-vs-interaction
  identifiability disclosure, and overlap with the grounding pass's lattice/gI
  territory — conjunction expressivity may be better served there or at the fUfI merge.
- **Trigger conditions to revisit:** (i) an instrument shows conjunction deficit is
  real — e.g. residual analysis: does the ID-share/e_ID systematically encode
  specific word *pairs* (regress ID contributions on pair indicators; cheap, designable);
  or (ii) the ml-1m regression gets attributed to composition expressivity; or
  (iii) the fUfI merge design wants shared conjunction structure anyway.
- **If ever adopted, the shape is fixed here:** *selected* pairs only (GA²M/FAST-style
  support-based selection), **constant** coefficients (never attention — AFM is the
  cautionary contrast), rendered as named pair terms with the identifiability
  disclosure. Implementation sketch: pair values as derived vocabulary rows appended to
  `feature_ids` at build time (~build-side change + vocab growth; model forward
  untouched — it's still a sum, which is the point).

---

## 3. Suggested sequencing (when this wave is opened — NOT now)

1. **U3 + U2a first** (pure measurement, login-node class, zero risk): norms-vs-
   frequency, centred-vs-raw geometry and profiles. These *inform* whether U2b is worth
   an arm and give the thesis's hidden-effects section two ready-made instruments.
2. **U1** as the first training change (one knob; its evaluation rides the existing
   cold-vs-all machinery; compared rows ⇒ SLURM, one /leo5-submit per run, C8 manifest).
3. **U2b** only on a positive U2a verdict, as a designed ablation arm.
4. U4 stays parked behind its triggers.

Process note: dc01 is closed through SC.11 — this wave would enter as new designed
variants under the protocol's staging rules (R6 staging note in the requirements doc),
alongside the popularity wave. Two-tier numbers policy applies: everything here is
working-evidence tier.
