# Nonlinearity below the prototype layer — research programme & handoff

**Written:** 2026-09-08 (session with Claude, Fable 5.1). **Status:** research plan; no code.
**Track:** parallel exploratory track, starts immediately, dev tier (30 trials / 60 epochs /
patience 7, seed 38210573), one `/leo5-submit` per run. **Frozen:** the prototype layer
(shifted cosine, score Σ_l t_{i,l}(1+cos(q,p_l)), both hosts and the double tie); the frozen S0
reference table; input scope = current splits only (hm: 5 categorical fields, one value each,
V=426; ml-1m: genres + Tag-Genome tags as bags, ≤72 tokens; user = set of train purchases with
their attribute words and item IDs). **Not in scope here:** dc07 (similarity change; own
handoff `Master/temp/dc07_membership_similarity_requirements_2026-09-08.md`; a parallel session
has ALREADY BUILT the dc07 softmax/sigmoid variants on 2026-09-08 and holds them uncommitted —
this track must not touch `PrototypeEmbedding` and should pull before building to avoid a merge
conflict on `feature_extractors.py`), candidate prime (user attributes), event context (needs
data the splits do not carry).

Read first: `Master/docs/feature_aware_solution_requirements.md` (R1–R8, S1, S5);
`Master/protocol_vault/2026-07-12_1200_linearity-interpretability-additive-attribution.md`
(the constraint and its caveat (b)); `2026-07-14_1504_fidelity-term-disambiguation-relocated-not-escaped.md`
(the sentence this programme invalidates: "contribution-level fidelity = 1 by construction");
`Master/docs/interpretability_knobs_metrics_taxonomy.md` (rows 8, 13, 16 are the ones this track
moves); the six 2026-09-08 vault entries; the fU vocabulary research memo
`Master/temp/user_vocabulary_deep_research_2026-09-07.md`.

---

## 0. The reframing in one page

**What linearity bought.** score = Σ_l t_l · a_l, a_l = 1 + cos(q, p_l), q = Σ_r c_r e_r. Because q is
a sum of named rows and cos is linear in q up to a shared scalar, every activation splits exactly
into per-row shares; every prototype is named by cos(e_f, p_l); the ID row is the disclosed
residual; Score Fidelity (share of score mass on named rows) is exact. Mechanism fidelity = 1 by
construction.

**What it cost, as measured.** P1 collapse; P2 shared input direction; P3 norm handover;
P4 evidence mass; P5 ID absorption; P6 layer bottleneck (dot beats prototypes by 0.05–0.09);
P7 count bottleneck; P8 +1 popularity channel; P9 coverage regulariser off; P10 naming
completeness. P1, P2, P4, P6, P8 belong to the LAYER (frozen here; dc07's business). P5 is the
free-row absorption theorem and is untouched by any nonlinearity that is a fixed function of the
user's own row. **The one problem this programme can attack is P7** — and, through it, whatever
part of P6 is not the layer's fault: the composition's inability to represent interactions.

**The key fact that scopes everything.** Inputs are categorical. For a single categorical field a
free embedding row is already the most general function of that field. Nonlinearity can add
nothing per field. It can only add INTERACTIONS: (i) field × field inside an item (black∧dress),
(ii) token × token inside a bag (ml-1m tags), (iii) purchase × purchase inside a history
(co-occurrence, identity, count), (iv) history × candidate (target-conditioning — the only escape
from P5 in the ids arm). Every model class below is a way of representing one of these four
interactions, and every interpretability hook is a way of attributing an interaction.

**The three attribution regimes the programme moves between.**
- *Exact-linear* (today): additive rows; shares exact; no approximation.
- *Exact-with-weights* (RETAIN form): nonlinear WEIGHTS on linear rows, q = Σ_r w_r(x)·c_r e_r;
  shares are exact given the weights; the weights need a faithfulness story (Jain & Wallace /
  Wiegreffe & Pinter tests). Mechanism fidelity stays 1 for the decomposition, not for the "why"
  of the weights.
- *Post-hoc* (nonlinear rows): q = ρ(Σ φ(x)) or an MLP; attribution imposed from outside
  (Shapley, IG, deletion). At hm scale exact Shapley is ENUMERABLE: 5 fields → 2^5 = 32
  coalitions; 5 purchases → 32; up to |H| = 12 → 4,096 forward passes per user, trivially cheap on
  a GPU for a sample. This is the one place where "post-hoc" does not mean "approximate".
Rule for the whole track (caveat (b) of the linearity entry): push nonlinearity upstream, keep
the last mile into the score additive where possible, and DISCLOSE what the nonlinear step drops
— now as a measured number (Score Fidelity generalised, §7) rather than a structural guarantee.

**Absorption holds on BOTH sides.** An item's attribute tuple is fixed per item exactly as a
user's history is fixed per user, so with the item ID row any nonlinear function of the item's
fields is absorbable in function class (vault 2026-07-12_1739 generalised). Consequence for every
subproblem: nonlinear composition can only change (a) inductive bias / generalisation in the ids
arms, (b) capacity in the noid arms, (c) cold behaviour. Headroom is REAL only in noid arms, in
cold evaluation, or under target-conditioning (SP9). A large ids-arm gain is a budget or
optimisation artefact until proven otherwise (the fI ml-1m 30-vs-100-trial flip is the precedent).

**Two facts that bound the whole programme before any model is built.** (1) On hm,
`lightfm_hist` (linear mean, plain dot) at 0.6511 is within 0.008 of the double-tie headline
0.6587 — the user-side composition is not the weak part; the layer is. (2) The item-side ceiling
for any conjunction model is the "tuple-ID" arm: fI-noid with ONE synthetic field = the 5-field
tuple (F=1). Whatever field×field nonlinearity could add on the training objective is bounded by
tuple-ID minus fI-noid. On ml-1m bags are near-unique so tuple-ID ≈ item_proto. Both are
zero-model-code measurements (SP0).

## 1. Subproblem catalogue (smallest first)

Format: what nonlinearity could add · model class · interpretable hook · instrument · expected
headroom · cost. "Arm" = which host branch it modifies. All configs are dev tier unless noted.

### SP0 — Zero-code bounds and instruments (week 1, no model code)
- hm item 5-tuple twin rate (share of items sharing an identical tuple; 67.5 % share a signature
  with ≥1 other item per the dc01 SC.8 record — re-measure on the split); user basket-signature
  twin rate (0.61 % known); history-length distribution and H_max on both splits (ml-1m p90 = 222,
  max unmeasured).
- HR@10 of fU vs lightfm_hist STRATIFIED by |H| (existing checkpoints, login node): does the
  0.079 gap live in long histories (composition) or everywhere (layer)?
- Tuple-ID item arm (fI-noid, F=1 synthetic tuple field) on hm and ml-1m: 2 runs. Bounds SP1/SP2.
- Gate G1 (end of week 1): if the fU-vs-dot gap is flat across |H| AND tuple-ID ≈ fI-noid on hm,
  set-level and field-level nonlinearity are both weak levers; the 2×2 (§3) still runs, as the
  documented closing experiment.

### SP1 — Within-item field interactions (arm: fI item side, hm)
- Adds: conjunction information (black∧dress vs black∧skirt) lost by Σ_f e_f. Item-side probe
  (2026-09-07 ID-row probe, section A) found the ID row is NOT predictable from up to 5,000 pair
  columns (R² ≈ 0): the model did not need conjunctions when it had a free row. Headroom on the
  ids arm ≈ 0; on the noid arm unknown and the honest test.
- Model classes, cheapest first: (a) **NFM bi-interaction pooling**, parameter-free given the
  field embeddings: q_i = Σ_f e_f + ½[(Σ_f e_f)² − Σ_f e_f²] (elementwise) — adds exactly the 10
  field-pair products; (b) **AFM**: attention over the 10 pairs, a_{fg} = softmax(hᵀ ReLU(W(e_f⊙e_g)+b))
  → learned pair weights, ~d·h params; (c) explicit pair rows e_{f∧g} (LINEAR — the dc-menu M5;
  the control for (a)/(b)); (d) AutoInt/DCN as the unconstrained upper bound.
- Hook: (a) pair units are exact summands (bi-interaction is a sum over pairs) → per-pair shares,
  naming by cos(e_f⊙e_g, p_l); (b) AFM weights disclosed as weights; (c) exact-linear.
- Instrument: fI-noid hm HR@10 vs 0.4229/0.2317; twin-signature separation is NOT affected
  (twins share all fields; this is not the twin problem); name-coverage share per prototype
  (fraction of activation mass on unary vs pair units).
- Headroom: small on hm (5 fields; the survey memo's item probe); the ml-1m analogue is SP2.
- Cost: (a) ~10 lines in `FeatureEmbedding.forward`; one run pair (hm noid, ids). 1 day + 2 runs.

### SP2 — Within-bag token interactions (arm: fI item side, ml-1m)
- Adds: tag co-occurrence inside a movie's bag (up to C(72,2) pairs) and, more importantly,
  RE-WEIGHTING of tokens: today every token weighs 1 (raw indicator bags), which created the
  bag-size popularity channel (corr 0.67–0.80 with ‖q‖).
- Model classes: (a) **attention pooling over tokens** (MIL attention, Ilse et al. 2018):
  w_t = softmax(wᵀ tanh(V e_t)), q_i = Σ_t w_t e_t — nonlinear weights on linear rows (RETAIN form);
  learns which tags define the movie; (b) Set Transformer / self-attention over tokens (upper
  bound, opaque); (c) bi-interaction over tokens (quadratic in 72 → fine).
- Hook: (a) per-token weights + exact shares given weights; the attention "why" is the
  faithfulness question, testable with W&P's uniform baseline (= today's fI) and frozen-weight
  diagnostics.
- Instrument: fI ml-1m dev rows 0.5156/0.2836 (ids), 0.4960/0.2735 (noid); bag-size-channel
  correlation (id_row_probe section C); Score Fidelity unchanged in form (rows still linear).
- Headroom: plausible on ml-1m (the bag-size channel is a known distortion); nil on hm (one token
  per field, attention over 5 fixed slots ≈ learned field weights = linear reparam).
- Cost: ~30 lines (a mask-aware attention pooling in `FeatureEmbedding`); 2 runs.

### SP3 — Per-purchase weighting from the purchase's own content (arm: fU, both datasets)
- Adds: a purchase's weight in the user mean depends on WHAT it is (an outlier purchase, a
  gift, a basic) — nonlinear weights, linear rows. Today w_j = 1/|H| for all j.
- Model class: **MIL attention over purchases**: w_j = softmax_j(wᵀ tanh(V c_j)), c_j the
  purchase's composed word vector; q_u = Σ_j w_j c_j (+ e_ID). ~d·h params. Variant: gated
  attention (Ilse) for sharper weights.
- Hook: exact per-purchase attribution given the weights ("your cardigan counted 2.3× a plain
  purchase"); faithfulness tests: uniform-weights baseline = today's fU (W&P), frozen-weight
  retrain, adversarial-weights (alternative w with the same score, cheap in a linear-rows model),
  seed variance of w.
- Instrument: fU-noid rows (hm 0.5833/0.3507; ml 0.5424/0.3016, pre-LOO); the fU 100-trial/LOO
  wave rows when they land; attention-entropy distribution; agreement between attention weights
  and exact Shapley of purchases (SP-X2) — a small novel study.
- Headroom: hm — at 5 purchases there is little to weight (the F-critic's objection to M3); but the
  ids-arm absorption argument does NOT fully apply: w_j depends on c_j (shared rows), so the same
  garment gets the same weight across users — an inductive bias e_ID cannot copy for free.
  ml-1m — real (56 films of which some are noise).
- Pitfall: leave-one-out pooling (commit 89a1818) subtracts 1/|H|·c_pos; with learned weights the
  subtraction becomes "recompute the softmax without the positive" — the attention module must
  receive the positive's index in training (the pos_i_idxs seam exists).
- Cost: ~40 lines subclassing `HistoryFeatureEmbedding` (weights over the D_max padded slots need
  a per-PURCHASE grouping, which the current per-TOKEN buffers do not carry → builder emits a
  purchase-level layout: `(n_users, H_max, D_item)` ids or a purchase-index map). 2–3 days; 4 runs.

### SP4 — Purchase × purchase interactions (arm: fU)
- Adds: the "these two purchases co-occur" content — the sentence the whole fU research asked
  for and the linear sum cannot represent (Janossy k=1). With nonlinearity the pair unit IS
  representable: pairs of purchases as summands.
- Model classes: (a) **Janossy k=2 with a small MLP** f(c_j, c_k) summed over the C(|H|,2) pairs
  (10 at hm median; 1,540 at ml median → subsample pairs per step, Janossy's π-SGD); (b)
  bi-interaction over purchases (parameter-free: ½[(Σc)² − Σc²]) — the cheapest, exact per pair,
  but the survey's F3 shows for WORD pairs this is ≈ the constant squared; over PURCHASE vectors
  it still is (c_j all contain μ) → centre first or accept it feeds P2; (c) self-attention over
  purchases (Set Transformer SAB) → pairwise weights, opaque values.
- Hook: (a),(b) pair units are exact summands → "because you bought X and Y together"; the
  strength is the model's, disclosed as a learned pair score (two-part honesty rule from the
  survey: presence is yours, strength is learned across users).
- Instrument: fU-noid rows; a within-history pair-signal test FIRST (survey instrument §4.3:
  lift of item pairs vs chance, same-day vs cross-day) — if pair signal ≈ chance on hm, skip hm.
- Headroom: hm low (94 % same-day → pairs = basket complements, few); ml-1m real (genre/tag
  combinations across films).
- Cost: builder emits purchase-level layout (shared with SP3); model 60–100 lines; 4 runs.

### SP5 — Nonlinear set encoder over purchases, general (arm: fU)
- Adds: any permutation-invariant function of the history: q_u = ρ(Σ_j φ(c_j)) with ρ a 2-layer
  MLP (Deep Sets), or PMA pooling (Set Transformer) with k seeds.
- Build form (chosen so SP3/SP5/SP7 are ONE class with three heads): q_u = linear mean backbone
  + ρ(pool_j φ(c_j)) with the residual's last layer ZERO-INITIALISED → bitwise fU / lightfm_hist
  at init (a keystone test like dc01's F=0 reduction); optional norm cap ε on the residual; pool
  = mean or gated attention (head = SP3 alone when ρ is off). The residual share per user and per
  prototype activation is the disclosed instrument (taxonomy row 8) — meaningful only in noid
  arms or with the ID row capped, because residual and ID row are interchangeable in function
  class (pathology P7: an optimizer-chosen split).
- Hook: none intrinsic for the residual → **exact Shapley over purchases** by enumeration (|H| ≤ 12
  → ≤ 4,096 forwards; sample users above), counterfactual leave-one-purchase-out (5 forwards),
  deletion curves. This arm defines the POST-HOC end of the S1 spectrum and is the accuracy upper
  bound for user-side composition under the frozen layer.
- Pitfall specific to nonlinear set encoders: **they can act as hashed ID rows.** hm basket
  signatures are almost unique (0.61 % twins) and ml-1m singleton tags are covert item IDs, so a
  wide ρ can memorise per user through the history key — "noid" is not ID-free once the encoder is
  nonlinear. Control: a hashed-history-ID arm (free embedding per distinct basket signature) as the
  memorisation ceiling; keep hidden widths ≤ 2d; watch the train/val gap.
- Instrument: the 2×2 headroom design (§3) — this is the nonlinear×proto cell.
- Headroom: whatever P7 is worth under the frozen layer. Prediction: small on hm, moderate on
  ml-1m, and in both cases smaller than the layer's own loss.
- Cost: 40 lines; the Shapley instrument 100 lines; 4 runs (+ 4 for the dot-product twin in §3).

### SP6 — Item-ID rows inside a nonlinear history encoder (arm: fU, SVD++ shape)
- Adds: item identity per purchase (P7's identity half): φ(purchase) = c_j + y_j with y_j a free
  history row, pooled by SP3/SP5.
- Hook: per-purchase shares still exact under SP3; the y_j part is unnamed → name-coverage
  share per prototype MANDATORY (E-4 of the survey); noid twin mandatory.
- Pitfall: leave-one-out is non-negotiable (with y_pos in q_u the model memorises the positive).
- Headroom: lightfm_hist-style lift expected (the survey's honest expectation), i.e. a gain that
  bypasses the vocabulary. Run only after SP3 so the weights and the rows are separable.

### SP7 — Named-bottleneck user encoder (hybrid; arm: fU)
- Adds: nonlinear reasoning (pairs, conjunctions, weighting) FORCED to express itself as a
  NAMED PROFILE: a set encoder outputs a V-dimensional non-negative profile π_u over the attribute
  vocabulary (softplus), and q_u = Σ_f π_{u,f} e_f (+ e_ID) — linear from the bottleneck on.
  Concept-bottleneck form (Koh 2020, E1 local) with the vocabulary as the concept set; trained
  end-to-end on the rec loss, optionally with a reconstruction/regression target toward the
  observed word counts (so the profile stays "your history, re-weighted" and not a free code).
- Hook: exact-linear attribution from π_u onward ("the model summarises your history as this
  profile"); the encoder's nonlinearity is quarantined behind a named layer; Score Fidelity
  exact again; the gap between π_u and the raw counts is the DISCLOSED nonlinear step (a
  per-word "re-weighting factor" the shopper can read).
- Risk: with V=426 and d≈64, the bottleneck is wider than the embedding — the profile is
  non-identifiable in the same way dc04's was (a ≥ V−d−1 dimensional null space per q); an
  anchoring loss toward the counts is what makes it identifiable. This is dc04's lesson applied
  on the user side; state it up front.
- Support decision: with π restricted to the user's OWN words this collapses to SP3 (re-weighting);
  with OPEN support the encoder can add words the user never bought — accuracy up, the "your
  history says" reading broken (the label-hallucination lesson, vault 2026-07-12_1659). Run open
  support only as an ablation and report the hallucination rate (π mass on words not in the
  history) as its cost.
- Headroom: bounded above by SP5 (same encoder capacity, plus a bottleneck).
- Cost: 80 lines; 4 runs. The most thesis-shaped hybrid: it keeps R2/R4 and quantifies the cost.

### SP8 — Linear backbone + capped, disclosed nonlinear residual (hybrid; both arms)
- Adds: q = q_linear + r(x), r an MLP over the same inputs, with ‖r‖ ≤ α‖q_linear‖ (cap) and
  the residual share reported per recommendation — the ID-row idea generalised: the residual is
  the "everything the vocabulary misses" channel, now a function of the inputs rather than a
  free row, so it works for cold objects too.
- Hook: exact-linear for the backbone; the residual is one disclosed lump ("+12 % unexplained
  content term") attributed post-hoc if wanted (Shapley on its inputs).
- Headroom: by construction between linear and SP5; α sweeps trace the fidelity–accuracy
  frontier DIRECTLY — this is the Rashomon-device experiment Ch 9/11 needs (accuracy vs Score
  Fidelity curve), on one knob.
- Cost: 40 lines; α ∈ {0.1, 0.3, 1.0} × 2 datasets × 1 arm = 6 runs (fixed-config retrains of the
  winner are acceptable here because α is not tuned, it is swept).

### SP9 — Target-conditioned attention over purchases (arm: fU; NAIS/DIN)
- Adds: the only nonlinearity that escapes P5 in the ids arm (depends on the candidate).
- Hook: per-candidate weights; the user profile stops being a per-user object (survey verdict);
  RETAIN exactness constraint keeps per-(purchase, word) attribution exact given the weights.
- Headroom: ml-1m (NAIS > FISM precedent); hm nil (5 items to weight).
- Cost: new ft_type (the prototype layer must receive the item index on the user side; the
  double tie complicates it). LAST in the programme; run only if SP3 shows attention helps at all.

### SP10 — Order-agnostic sequence encoder as a control (arm: fU)
- A GRU over purchases in a random order, no timestamps (we have none). Purpose: falsify
  "sequence models help" claims on this data by construction — if a random-order GRU beats Deep
  Sets, the gain is capacity, not order. A control, not a candidate. 2 runs, ml-1m only.

### SP11 — Item-side bottleneck / part-based prototypes (arm: fI; deferred)
- With categorical-only inputs the item-side bottleneck is trivial (the attributes ARE the
  concepts). Part-based prototypes (scratchpad L69) need images → out of scope here; recorded so
  the catalogue is complete.

## 2. What each subproblem does to the problem list

| | P1 collapse | P2 shared dir. | P3 handover | P4 evidence | P5 absorption | P6 bottleneck | P7 count | P8 +1 | P10 naming |
|---|---|---|---|---|---|---|---|---|---|
| SP1 pairs in item | – | feeds (pairs of μ) unless centred | – | – | – | small | conjunctions ✔ | – | pair names ✔ |
| SP2 tag attention | – | can DOWN-weight the constant tags → partial ✔ | – | – | – | small | – | – | ✔ |
| SP3 purchase attention | – | can down-weight μ-heavy purchases → partial ✔ | partially (weights can shrink outliers, not ‖q‖ vs ‖e_ID‖) | – | weights are shared functions → partial escape | small | – | – | ✔ |
| SP4 purchase pairs | – | feeds unless centred | – | – | – | small | co-occurrence ✔ | – | pair units ✔ |
| SP5 Deep Sets | – | ρ can re-centre internally | ρ can rescale | ρ can express |H| | – (fixed fn of own row) | the composition's share of P6 | ✔ all | – | ✘ (post-hoc only) |
| SP6 item-ID rows | feeds (popular rows) | feeds | second handover | – | – | lift via bypass | identity ✔ | – | ✘ unnamed rows |
| SP7 bottleneck | – | π can re-weight | – | – | – | ≤ SP5 | ✔ inside encoder | – | ✔ exact-linear |
| SP8 residual | – | – | cap fixes handover for the residual | – | – | frontier traced | ✔ in residual | – | disclosed lump |
| SP9 target attention | – | – | – | – | ✔ escapes | ml-1m | ✔ | – | per-candidate |
None of them touches P1, P4, P8 or P9 — those are the layer's (dc07) and the regulariser's.

## 3. The first experiment: nonlinear headroom, cleanly separated from the layer

A 2×2 per dataset: {linear, nonlinear composition} × {plain dot, frozen prototype layer}.

| | plain dot (no prototypes) | frozen prototype layer |
|---|---|---|
| linear composition | EXISTS: lightfm_tags/_ids (item side), lightfm_hist/_ids (user side) | EXISTS: fI, fU (+ noid) |
| nonlinear composition (SP5-type Deep Sets on users; SP1(d)/SP2(b) on items) | BUILD: `lightfm_hist_nl` / `lightfm_tags_nl` — same encoder, dot score | BUILD: `feature_user_proto_nl` / `feature_item_proto_nl` |

Reading, as three hypotheses the design separates:
- **H_comp**: nonlinearity adds information the mean loses → the gain appears in BOTH columns at
  similar size (the layer passes it through).
- **H_geo**: nonlinearity helps the prototype layer READ its input (learned de-collapse, learned
  norm handling) → the gain appears mainly in the proto column, i.e. the interaction term is
  positive. This is the only outcome that justifies nonlinearity as a fix for "the loss is at the
  prototype layer".
- **H_null**: neither column moves beyond noise (yardstick 0.0026 HR@10, single draw; use 2× as
  the bar).
(nl,dot) − (lin,dot) = composition headroom the layer cannot eat; (nl,proto) − (lin,proto) = what
the frozen layer lets through; their difference = how much of the nonlinear gain the layer
destroys. This isolates P7 from P6 in one design. Run noid arms first (P5 makes the ids arms
uninformative about composition); ids arms as the absorption control. User side first (its gap is
the large one); item side only if SP0's tuple-ID ceiling leaves room. 4 runs per side per arm
type; ~14 h per block of four.

Build notes for the missing cells (scoping, no code here): (1) the history buffer is a padded
per-user WORD layout (`hist_value_ids`/`hist_weights`), not a per-purchase item list — the builder
(`feature_ids.py`, `build_user_history_weights`) must additionally emit `hist_item_ids (n_users,
H_max)` + mask from its existing `pairs` frame via `groupby('user_id').cumcount()`, behind a flag,
with an `H_cap` for ml-1m (truncate with a fixed seed) — and because the linear cells trained on
FULL histories, a linear-mean-with-H_cap control is required or the comparison is confounded;
(2) new class `SetHistoryEncoder(HistoryFeatureEmbedding)`: materialise the composed item table
once per forward from the LOO payload (`item_token_ids`/`item_token_w`) and the shared word table,
gather `(B, H_cap, d)`, apply LOO by MASKING the slot whose item id equals `pos_i_idxs` (the
algebraic subtraction at the linear path only works for the mean; attention softmax must be
computed AFTER masking), linear part = masked mean (must equal the parent bitwise in the no-cap
case — test), then the head; `supports_loo` stays true and the `RecSys.forward` pass-through
(`pos_i_idxs`) is reused unchanged; (3) factory branches copied from `feature_user_proto` and
`lightfm_hist`, swapping the class; item side mirrors `feature_item_proto` / `lightfm` with a
`SetItemEncoder(FeatureEmbedding)` overriding only the reduction; (4) hyperopt: reuse the fU /
lightfm_hist search spaces exactly plus at most TWO encoder knobs (hidden width ∈ {d, 2d}; cap
ε ∈ {∞, 0.5}); same 30 trials, same ASHA — check the stopping-epoch distribution, since
zero-init residuals look identical to fU at early rungs and may need more epochs; (5) any positive
result is re-run at 100 trials before it is believed. Existing rows at the correct tier:
lightfm_hist hm 0.6511/ml 0.6107 (ids 0.6464/0.6374); lightfm_tags hm 0.4582/ml 0.5565
(ids 0.5087/0.5612); fU-noid hm 0.5833/ml 0.5424; fI-noid hm 0.4229/ml 0.4960
(`sc05_chapter_draft.md:144-166`, `sc01_chapter_draft.md:129-148`; the hm fU/lightfm_hist rows
pre-date leave-one-out — use the LOO wave rows when they land, or re-run the two hm controls).
Go/no-go: if (nl,dot) − (lin,dot) < 0.01 on both datasets, user-side nonlinearity has no
headroom under ANY layer and the programme collapses to SP1/SP2 (item side) plus the hybrids as
fidelity instruments. If ≥ 0.02 on ml-1m, proceed down §1 in order.

## 4. Hybrids that keep exact attribution — ranked

1. **Nonlinear weights × linear rows (SP2, SP3)** — the RETAIN form. Exact shares given weights;
   the smallest models in the catalogue; the attention debate is the known cost and the
   faithfulness tests are cheap in a linear-rows model. Most promising: least change to the
   explanation object, direct attack on the bag-size channel (ml-1m) and on outlier purchases.
2. **Named bottleneck (SP7)** — keeps R2/R4 intact after the bottleneck; the disclosed nonlinear
   step is a per-word re-weighting; identifiability must be solved by an anchoring loss (dc04's
   lesson). Most thesis-shaped.
3. **Capped disclosed residual (SP8)** — one knob, traces the fidelity–accuracy frontier the Part
   II chapters need; least interpretable per se, most useful as an instrument.
4. **Explicit pair units (SP1(a), SP4(b))** — exact by construction; they extend the vocabulary
   rather than the function class; legibility cost (units multiply).

## 5. Post-hoc explanation toolkit for the nonlinear arms (and its own evaluation)

- **Exact Shapley by enumeration** over fields (32 coalitions) and purchases (≤ 4,096 for |H| ≤
  12; sample beyond). Baseline for absent players: the empty coalition = zero row (fU) / the ID
  row alone (ids arms). Deliverable: per-purchase Shapley shares comparable to the linear shares
  — on the LINEAR model they must coincide exactly (unit test).
- **Counterfactual leave-one-out**: rank change of the recommended item when purchase j is
  removed (5 forwards at hm median). The shopper-facing sentence "without your X this would not
  have been recommended" — sufficiency/necessity in one number.
- **Attention faithfulness battery** (W&P 2019, E4b local): uniform baseline (= linear model),
  frozen-weight retrain, adversarial weights with equal score, seed variance; plus agreement
  (Spearman) between attention weights and Shapley shares — the study "are learned purchase
  weights faithful importances in a linear-rows recommender" is itself a small contribution.
- **Deletion/insertion curves** (ERASER-style sufficiency/comprehensiveness) over purchases and
  fields, comparing linear shares, attention, Shapley, and random.
- **Integrated gradients** only as a cross-check on SP5/SP8 (G1 SPINRec is the local IG-family
  reference for recsys fidelity).
- **Prototype naming** stays intrinsic wherever rows stay linear (cos(e_f,p_l)); for SP5/SP6 it
  degrades to the post-hoc top-k route the lineage started from — say so.

## 6. Pitfalls the plan must carry

- **P5 absorption**: any nonlinearity that is a fixed function of the user's own history is
  absorbable by e_ID in function class. Headroom is only REAL in noid arms or with SP9. Every
  ids-arm claim needs the ID-share read-out with a magnitude floor (dc06 lesson).
- **Positive leakage**: a history encoder that sees the training positive learns to score it;
  the leave-one-out seam (`RecSys.forward` → `pos_i_idxs`) must be honoured by every new encoder
  (attention: recompute without the positive; Deep Sets: drop φ(pos) before ρ; pairs: drop pairs
  containing pos).
- **hm depth**: 5 purchases → set-level nonlinearity nearly void; 94 % same-day → pairs =
  baskets. Expect ml-1m to carry every user-side result; report hm as the null.
- **ml-1m bags**: 72 tokens → attention and pairs are real; quadratic pair terms need
  subsampling; the bag-size channel must be read out before/after.
- **Hyperopt asymmetry**: frozen linear rows vs freshly tuned nonlinear arms with extra
  hyperparameters (h, dropout); declare it; use the noise yardstick 0.0026 HR@10 (hm, one draw).
- **Evaluation protocol**: unchanged (LOO ranking, sampled negatives); no context, no cold-user
  split needed for this track.
- **Compute**: every arm is a gather + small MLP; per-batch cost ≈ 1.1–1.5× fU; hm RAM peaks
  ~27 GB at concurrency 5 → `--mem 40G`. Never materialise `(B, H, 72, d)` per purchase on ml-1m;
  compose the item table once per forward and gather.
- **Sampled-negative evaluation** keeps SP9's per-candidate user vectors affordable; a full-ranking
  protocol would not (verify the eval negative count before building SP9).
- **Explanation objects change** under attention (shares not comparable across users without the
  weights) and under SP9 (per-candidate profile); disclose, do not average away.
- **R8 parsimony**: each SP is one idea; hybrids combine at most two; no stacking.
- **Unverified at writing** (check in W1): eval negative count; H_max on both splits; hm item
  count and tuple twin rate on the split; whether ASHA's grace period truncates slower nonlinear
  trials.

## 7. Metrics this track adds to the taxonomy

- **Score Fidelity, generalised**: fraction of score mass attributable to named units, where
  attribution is exact-linear (regime 1), exact-given-weights (regime 2), or Shapley (regime 3),
  with the regime and the approximation error reported. Taxonomy row 13 gets a regime tag.
- **Attribution agreement**: Spearman/Top-k overlap between attention weights and Shapley shares
  (regime 2 vs 3) — a faithfulness number for the weights.
- **Residual share** (SP8) and **bottleneck re-weighting divergence** (SP7): the disclosed size
  of the nonlinear step.
- **Interaction share**: for pair models, mass on pair units vs unary units.
- Existing instruments unchanged: sc8 geometry read-outs, id_row_probe B (ID share), naming
  route where rows are linear.

## 8. Eight-week immediate track (dev tier; gates in bold)

Skeptical prior, stated up front so the gates are honest: expect H_null on hm for every user-side
form except SP9 (which is unnatural there), a small H_comp on ml-1m for SP3/SP5, and the only
clearly positive cell to be SP9 under a plain dot. If that is the outcome, the thesis gains a
clean quantitative sentence: below a frozen cosine layer, nonlinearity over the same categorical
inputs buys little; the layer, not the composition, is the bottleneck.

- **W1 — bounds and instruments, no model code (SP0).** Twin rates, H_max / history-length
  distributions, HR stratified by |H| on existing checkpoints, tuple-ID item arm (2 runs). Write
  the exact-Shapley / counterfactual instrument against the LINEAR model (its Shapley shares must
  coincide with the linear shares exactly — the unit test). **Gate G1** as in SP0.
- **W2 — build.** Purchase-level buffer + mask behind a flag; `SetHistoryEncoder` with the three
  heads (attention / residual / bottleneck), masked LOO, zero-init residual; keystone tests
  (residual off ≡ fU and ≡ lightfm_hist bitwise; single-purchase LOO → zero metadata mass, mirroring
  dc05 t14); factory rows in both registries; login-node smoke. Commit + push (ask first).
- **W3 — user-side missing cells, noid, both datasets (4 runs).** **Gate G2**: classify H_null /
  H_comp / H_geo at the 2× noise bar. H_null → write the negative; effort moves to the layer
  (dc07). H_comp only → the layer still swallows it; layer work first, this track continues as
  fidelity instrumentation only. H_geo → continue.
- **W4 — ids-arm controls (4 runs) + geometry instruments on the nonlinear cells** (sc8 effective
  rank and clone detector, residual share, norm handover) + the hashed-history-ID memorisation
  ceiling arm. **Gate G3**: ids-arm gain ≈ 0 and noid gain > 0 confirms absorption → any candidate
  must be noid or ID-capped (better cold story, worse warm story).
- **W5 — SP3 attention head + faithfulness battery** on the winning dataset; exact Shapley on hm
  (32 coalitions per user). **Gate G4**: if attention weights do not track LOO deltas / Shapley,
  drop the RETAIN-form candidate and keep SP8 as the disclosed-residual instrument only.
- **W6 — SP9 target-conditioned attention on ml-1m** (noid + ids, 2–4 runs, the heaviest) OR, if
  G2 was H_null, the SP8 ε-sweep (6 retrains) to draw the accuracy-vs-residual-share curve.
  **Gate G5**: if SP9 under the layer beats fU by more than SP5 did, the only non-absorbable form
  is the one carrying headroom — record that the price is the per-candidate profile.
- **W7 — item side only if G1 left room:** SP1(a)/(c) on hm noid, SP2 tag attention on ml-1m
  (+ `_nl` dot twins); else finish the frontier plot (accuracy vs Score Fidelity across linear /
  weights / bottleneck / residual-ε / Deep Sets) and the attention-vs-Shapley agreement study.
- **W8 — 100-trial confirmation of any positive; consolidation memo; thesis placement** (S1
  spectrum realised as a section; Ch 11 movement 2 / Ch 9 accuracy-cost numbers); future-work
  paragraph for SP4/SP6/SP10/SP11; protocol entries; decision whether one form enters a
  `/design-candidate` cycle.
Compute total ≈ 24–36 dev-tier runs plus ≤ 4 at 100 trials; runs alongside the fU LOO wave and
dc07 (no shared code path except the history builder's new layout, which is additive).

## 9. Literature: local vs to fetch (paper-search MCP was down 2026-09-08 — verify)

Local (`Master/literature/papers/`): AFM B8; NFM B9a; DeepFM B9b; Wide&Deep B9c; LightFM B5;
ACF A2; ProtoPNet C1; ProtoTree C5a; CBM E1, CBM-without-concepts E3, SENN E2; Jain & Wallace
E4a; Wiegreffe & Pinter E4b; Rudin S1/C3; Zhang & Chen S2; PPM challenges S5; attribute-aware
survey S7; SPINRec G1 (IG-family fidelity); FacT factorization trees F4; MacridVAE F1.
To fetch: Deep Sets (Zaheer 2017); Set Transformer (Lee 2019); Janossy pooling (Murphy 2019);
attention-based MIL (Ilse, Tomczak, Welling 2018); RETAIN (Choi 2016); NAIS (He 2018) / DIN
(Zhou 2018); AutoInt (Song 2019); SHAP (Lundberg & Lee 2017) and exact-Shapley enumeration;
Integrated Gradients (Sundararajan 2017); ERASER (DeYoung 2020); Serrano & Smith 2019 ("Is
attention interpretable?"); Neural Additive Models (Agarwal 2021) and NA²M / GA²M — the
canonical "nonlinearity upstream, additive last mile" family and the right citation for §0's key
fact and for SP1(c)'s pair vocabulary; Balog 2019 (already cited in the thesis without a copy);
Steck 2024 (cosine reliability). Kaggle H&M write-ups (`Master/docs/kaggle_sources/`) for the
industry view of attribute interactions.

## 10. What the thesis gets from this even if every gate fails

A controlled answer to the reviewer's question "isn't the whole power of ML nonlinearity?": the
2×2 shows how much nonlinear composition is worth on these data under a plain dot and under the
prototype layer; the hybrids show what exactness costs in accuracy and what nonlinearity costs in
fidelity, on one frontier plot; and the Shapley-vs-attention study says whether the cheap
explanation (weights) agrees with the expensive one (exact Shapley) on a recommender. That is
the S1 "deliberately decoupled variant" the requirements asked for, and the empirical movement
Ch 11 is missing.

## 11. Deliverables checklist for the receiving session
- [ ] Builder: purchase-level history layout behind a flag (additive; existing tables untouched).
- [ ] Encoders: SP5 Deep Sets (user), SP2 tag attention (item), SP1(a) bi-interaction (item),
      SP3 purchase attention (user), SP8 residual (both), with LOO honoured; `_nl` dot twins.
- [ ] Instruments: exact Shapley + counterfactual LOO + faithfulness battery + generalised Score
      Fidelity; dc_checks for each (linear-model coincidence tests).
- [ ] 2×2 headroom results table per dataset; Gate 1 memo; protocol entries at each gate.
- [ ] Registries in both `start.py` and `run_combo.py`; modifications log; `--skip-explanations`
      for arms whose rows are nonlinear (breakdown refuses non-affine paths by design).
- [ ] Do not touch: the frozen reference table; the cosine host; the dc07 files.
