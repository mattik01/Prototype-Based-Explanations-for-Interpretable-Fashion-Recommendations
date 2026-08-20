# fU-ProtoMF (dc05) — Results Memo

> SC.9 artifact of the dc05 scrutiny routine (protocol v1.3; drafted
> 2026-08-20). **Status: dev-profile provisional throughout** — every number
> is a scrutiny-phase working number under the two-tier policy (single seed
> 38210573, dev search budget 30/60/7/4), never a thesis number; thesis-grade
> claims get dedicated experiments at writing time.
> **Backfill notice (re-sequencing decision, SC.8 gate follow-up):** the S1
> decoupled-control rows (`lightfm_hist`/`lightfm_hist_ids`, 4 warm hyperopts
> + 2 hm cold retrains) were ratified and QUEUED at drafting time (jobs
> 7548787–7548790, PENDING) — every table that owes them carries a
> **[CONTROL PENDING]** marker; a dated backfill edit completes this memo
> when they land, and SC.10's cold read runs only on the backfilled draft.
> Basis: dc05 dossier + design doc + direct vault sweep (SC.9 step 0).

## 1. Mechanism

fU-ProtoMF replaces exactly one object in U-ProtoMF: the free per-user
embedding. Each user is instead **composed from their own purchase history's
attribute words**:

> q_u = Σ_{f∈V} w̄_{u,f}·e_f  (+ e_ID(u)),   w̄_{u,f} = n_{u,f} / |H_u|,

where H_u is the user's train-window purchase set (deduplicated, exactly as
the interaction matrix sees it), n_{u,f} counts how many of those purchases
carry attribute value f, and the e_f are trainable word embeddings over the
item-side vocabulary (hm: canonical 5 fields, V=426, one value per field;
ml-1m: genres+tags@0.8 indicator bags, V=1,061). The optional per-user ID row
e_ID(u) is the residual channel (everything behavior says that the basket
words cannot). q_u feeds the **unchanged** user-prototype layer: u*_l =
1 + cos(q_u, p^u_l), score S(u,t) = Σ_l t_l·u*_l against the host's free
item vector t ∈ R^{K_u}. Prototypes, inclusion regularizers, loss, negative
sampling: byte-identical to `user_proto` (pinned; the keystone reduction
fU(V=0,+ID) ≡ `user_proto` is bit-identical under weight copy).

**The score identity that shapes everything on this host:**

> S(u,t) = **B(t)** + Σ_l t_l·cos_l,   B(t) = 1ᵀt.

Unlike on the item-side host, the +1-shift baseline **does not cancel when
ranking items**: B(t) is a learned, user-independent, per-item score channel
— a de-facto item intercept living inside the "bias-free" fleet. It is
host-inherited (published U-ProtoMF carries it identically), disclosed on
every rendered artifact, and measured (§4.5). Consequently the exactness
claim is: **100% of the *personalized* score (S − B(t)) decomposes exactly**
over (prototype, history-row) pairs — as per-word shares AND, by regrouping
the same sum, as **per-purchase shares** ("you activate *dark-denim
menswear* because of these 12 purchases, +0.31 of 0.42").

**Zero-interaction users (exact behavior, stated prominently by standing
directive):** (1) ids arm — an interaction-free user's q_u would be their ID
row alone, which is untrained noise, so cold-user scoring *drops* it (the
mirror of fI's cold-item ID drop, built and pinned); (2) after the drop (or
natively in the noid arm) q_u = 0, the eps-guarded cosine gives u* ≡ 1, and
the score collapses to S = B(t): the model degrades to the non-personalized
item baseline — a popularity-shaped ranking, verified literally at toy
scale, not a hand-waved fallback; (3) one purchase later, fold-in applies
(recompute w̄_u, no retraining): the user converts to warm-thin in a single
transaction. In the current split this user cannot occur (5-core floor: ≥3
train rows) — a verified *property*, not a measured result.

## 2. Motivation and relation to the host

**Facet A (intrinsic interpretability).** S0.2 identified the user surface
as the lineage's post-hoc remainder: U-ProtoMF interprets user prototypes
through synthetic maximally-activating users, and every rendered explanation
carries CF-only coefficients. fU closes exactly this surface: taste
communities become **readable from parameters alone** (cos(e_f, p^u_l) word
profiles), and every community activation decomposes exactly over the user's
own purchases — recommendation evidence stated in the user's own behavior.
The closest transparent ancestor is Balog et al. (SIGIR 2019: user = weighted
tag set over rated items, no embeddings); fU is the *embedded* version of
that aggregation, and inherits a latent **scrutability affordance** from
linearity ("remove this purchase" = subtract a row, effect immediate, no
retraining) — enabled, not built; the ID row is the non-scrutable residual.

**Facet B (sparsity).** The central bet: a thin-history user's free embedding
is underdetermined, while fU's word rows are trained by *every* user sharing
them — statistical strength moves to where the data is thinnest. The bet's
home regime is hm_1_month (median |H|=5; 25.1% of users at the 3-row floor);
ml-1m (median 56) is the deliberate counter-regime probing mean-regression /
omnipresence / B(t).

**Information ordering (lineage narrative):** info(U-ProtoMF) ⊂ info(fU) ⊆
info(fI). fU adds to the host exactly one source — the item-attribute
catalog, reached through baskets — and uses at most fI's information (it
only ever reads attributes of train-purchased items). **The lineage holds
the information constant and moves where it lives**: items built from their
own attributes (fI) vs users built from their purchases' attributes (fU).
The history itself is *not* new information (the loss already consumes every
interaction) — what it contributes is structure: a stated aggregation
replacing free compression.

**Theoretical frame for the arms.** With the ID row kept, the variant's
parameter space contains the host exactly (all word rows zero), and at that
point every objective term takes the host's value — so under training
adequacy the ids arm concedes nothing on the training objective
(generalization stays an open empirical question; that is why the host gap
is measured, never assumed). The noid arm is a genuinely restricted class
(user matrix confined to span of 426 word rows; basket-signature twins
forced identical — 0.61% of hm users, 0% of ml-1m users). On sparse data the
restriction can *flip sign* into a better-conditioned estimator — the
sparsity-flip hypothesis the D4 strata test.

## 3. Experimental setup

Per the S0.6 charter (SC.6 plan cites compliance clause-by-clause; SC.8
audited the landed runs against it — clean): datasets **hm_1_month
(+ hm_1_month_cold) primary, ml-1m second testbed** (pinned at SC.5/SC.6);
uniform 1+99 negatives; `use_bias=0` explicit everywhere; dev profile
(30 trials / 60 epochs / patience 7 / ASHA), seed 38210573, selection on val
HR@10; single code lineage `feat/scrutiny`; baselines from the frozen S0.7
table (hm) and the fleet's lazily-added ml-1m rows — nothing re-run.

**Disclosure — winning-config coincidence:** all five hyperopts (fU, noid ×
both testbeds, plus the host's ml-1m row) selected the *identical* config
(d=81, K_u=76, batch 256, adagrad lr 0.099, wd 3.7e-4, sim_proto 1.798,
**sim_batch 0.0018**, neg_train 39): same seed ⇒ same 30 sampled candidates
per search. Every comparison below is therefore architecture-matched; the
flip side (the search never adapted per-arm) is a dev-tier property. The
near-zero coverage weight (sim_batch) matters for §4.5.

**Controls.** The isolating ablation is the ids/noid pair (both testbeds).
The S1 decoupled control — `lightfm_hist(_ids)`: the same composed user,
plain dot product, **no prototypes** — was ratified at the SC.8 gate (full
suite, naming and treatment mirroring the item-side lightfm rows) and is
**[CONTROL PENDING]** (jobs 7548787–90 queued at drafting). Until it lands,
"composition helps" vs "the prototype layer's contribution" is not
separated; this memo flags every claim that awaits it. The fleet noise
yardstick is dc01's R3 draw, |Δ| ≈ 0.003 HR@10 (declared I-host limitation).

## 4. Results

### 4.1 Warm accuracy (R6 stage bar: vs `user_proto`) — both testbeds

| hm_1_month (primary) | HR@10 | N@10 | vs host |
|---|---:|---:|---|
| fU (ids) | 0.5832 | 0.3496 | +0.0107 / +0.0174 |
| fU-noid | 0.5833 | 0.3507 | +0.0108 / +0.0185 |
| `user_proto` (frozen bar) | 0.5725 | 0.3322 | — |
| `lightfm_hist(_ids)` | **[CONTROL PENDING]** | | |
| context: `lightfm_tags` / `_ids` | 0.4582 / 0.5087 | | |
| context: `user_item_proto` | 0.6587 | | |

| ml-1m (second testbed) | HR@10 | N@10 | vs host |
|---|---:|---:|---|
| fU (ids) | 0.5491 | 0.3090 | −0.0177 / −0.0140 |
| fU-noid | 0.5424 | 0.3016 | −0.0244 / −0.0214 |
| `user_proto` (fleet) | 0.5668 | 0.3230 | — |
| `lightfm_hist(_ids)` | **[CONTROL PENDING]** | | |
| context: `item_proto` / `mf` | 0.5711 / 0.5022 | 0.3187 / 0.2793 | |
| context: `lightfm_tags` / `_ids` | 0.5565 / 0.5612 | 0.3197 / 0.3261 | |
| context: `user_item_proto` | 0.6246 | 0.3573 | |

**Read:** on the thin-history primary testbed both fU arms sit ABOVE the
stage bar on both metrics (gap ≈ 3.5× the noise yardstick); on the
heavy-history testbed fU pays a tax. The paper-headline bar (UI-ProtoMF) is
**unassessed — deferred to the merge stage** (R6 staging note); the UI rows
above are context, not a comparison this stage claims.
**Committed secondary regimes, quoted here:** D3 (99 *popularity-sampled*
negatives) preserves both orderings — hm: fU 0.2652 / noid 0.2694 / host
0.2320; ml-1m: fU 0.2365 / noid 0.2310 / host 0.2607 — the warm readings are
not uniform-negative artifacts.

### 4.2 The ids-vs-noid pair (§3.6 fU interpretation rule)

Warm: the hm gap is ≈ 0 (0.0001 HR / 0.0011 N — inside the yardstick, noid
nominally ahead); the ml-1m gap is +0.0067 HR for ids (~2× yardstick). The
geometry instruments explain both: on hm the trained **ID row is nearly
inert** (0.9–5.0% of user-vector energy across history bands) — the
composition carries essentially the whole representation, so removing the ID
row costs nothing; on ml-1m the ID share reaches 7% in the heavy band, where
the division-of-labor handover (words carry the taste class, ID carries
idiosyncrasy, handover grows with |H|) has something to hand over. Per the
§3.6 rule, popularity blindness is NOT a gap component on this host
(popularity lives in B(t), untouched by the user-side arm choice); the raw
gap is never quoted as "the cost of interpretability."

### 4.3 History-length strata (D4 — the central-bet readout)

hm_1_month, HR@10 / N@10 gap (fU − host) by train-history quartile:

| stratum | rows | fU − host | noid − host |
|---|---:|---|---|
| Q1 [3] | 18,442 | +0.0101 / +0.0179 | +0.0128 / +0.0212 |
| Q2 [4–5] | 23,124 | +0.0124 / +0.0187 | +0.0143 / +0.0206 |
| Q3 [6–8] | 16,658 | +0.0068 / +0.0158 | +0.0112 / +0.0178 |
| Q4 [9–89] | 15,194 | +0.0076 / +0.0163 | +0.0102 / +0.0155 |

ml-1m (fU − host): Q1 [3–25] −0.0290 · Q2 −0.0389 · Q3 −0.0094 ·
Q4 [122+] −0.0046.

**Read:** on hm the gain is broad-based with the largest HR gains in the
thin strata (3–7× yardstick) — the central bet is **directionally
supported in its home regime**. On ml-1m the tax concentrates in the
*relatively* thin band and vanishes toward the heavy end — note ml-1m's
"thin" (≤25) already exceeds hm's maximum quartile start; the two testbeds
bracket the regime axis cleanly. Whether the hm gain is attributable to the
composition alone or to its interaction with the prototype layer awaits
**[CONTROL PENDING]**.

### 4.4 Cold-start (R5; hm only — explicit scope statement)

Cold rows exist **only on hm**: the ml-1m cold block is a standing
commitment deliberately deferred until the bags-aware kNN/tie machinery
lands (charter scope decision 2026-08-15) — its absence here is scope, not
omission. All numbers carry their F-S0-14 raw-logit brackets.

| hm_1_month_cold | fU | fU-noid | host | popularity |
|---|---|---|---|---|
| attr-kNN cold-vs-cold | 0.3265 [.3259,.3274] | 0.3418 [.3412,.3427] | 0.2999 [.2993,.3007] | 0.1202 |
| attr-kNN cold-vs-all (committed secondary) | 0.2829 | 0.2998 | 0.2217 | 0.0000 |
| native cold-vs-cold | chance floor (bracket 0.0955; point 0.00 = sigmoid tie artifact, F-S0-15a) | 0.1133 | 0.1081 | — |

**Read:** fU's cold-item story is *host behavior preserved* — the item side
is untouched, so native cold rows sit at the chance floor exactly as the
host's do (the mirror claim, verified rather than asserted). Under the
attr-kNN item patch **both fU arms clear the patched host on BOTH cold
regimes with non-overlapping brackets**, and both sit clearly below the
LightFM cold bar (0.4705/0.5080) — the honest ordering. noid > ids
cold (Δ 0.0153, non-overlapping) has no resolved mechanism (the
operating-point-mismatch mirror does not apply — cold *items*, user side
intact); recorded open, small, not claim-carrying. Disclosed honestly: on
the variant *warm* test the host sits above both fU arms (0.6346 vs
0.6249/0.6222) — the host gains more from cold-item removal; not a claim
surface, stated so the reader never discovers it elsewhere. The cold-USER
case is the §1 zero-interaction chain: NOT rescued (honest NO), fallback =
B(t), fold-in converts after one purchase; no cold-user testbed exists this
cycle.

### 4.5 Geometry (what the numbers stand on)

- **B(t) channel (M6′, comparative rule):** B(t) carries **70–81% of
  cross-item score variance** at corr(B, log popularity) 0.93–0.97 — on
  BOTH fU arms AND the host, with fU at or slightly below the host on both
  testbeds. Verdict by the ratified comparative rule: **host-class
  property**, priced to the U-host class, not to the candidate. The
  "largely popularity" annotation of the baseline line is now licensed by
  measurement. (The pause-period 2×2 established that this channel is the
  *honest* outlet: removing the +1 shift relocates popularity into the
  embedding geometry where no additive instrument can see it.)
- **Prototype collapse (M1′/M4′ materialized; host-class in kind):**
  activation-cloud effective rank hm: fU 1.79 / noid 2.20 / host 1.82;
  ml-1m: fU 2.63 / noid 2.46 / host 4.44. Profiles are diffuse (softmax
  entropy ≈ uniform; top1−top2 margin 0.007) with exact duplicates (p99
  pairwise overlap = 1.0): the hm cards show ~5–6 distinct taste-community
  profiles among 76 prototypes. Two contextualizations keep this honest:
  (i) the shared winning config has the **coverage regularizer effectively
  off** (sim_batch 0.0018; the pause-period hm_3_month probe showed 56/58
  distinct names at sim_batch 5.7 — collapse is *data pressure ×
  regularizer configuration*, not fate; the coverage knob is the identified
  lever, Rashomon-stage work); (ii) on hm the collapse is host-level, but
  on ml-1m fU is measurably MORE collapsed than its host — the one geometry
  axis where the candidate aggravates, co-located with its accuracy tax.
- **M5′ (mean-regression) verdict:** the norm-handover half is CONFIRMED on
  real checkpoints (metadata norm falls, ID-energy share rises with |H|, both
  testbeds); the angle-to-mean half holds moderately on hm (83°→74°
  thin→heavy) and **INVERTS on ml-1m** (73°→78°; host likewise) — heavy
  raters are *more* directionally distinctive, i.e. real taste heterogeneity
  correlates with |H|, which the homogeneous toys could not exhibit
  (empirical-not-scholastic, vindicated). The F-DC05-02 interaction-weighted
  coverage effect measured SMALL everywhere (top-eig-share delta ≤ 0.03).

## 5. Explanation showcase

**The axis: behavioral-evidence attribution.** (i) Taste-community meaning
read from parameters alone — intrinsic word profiles replacing the host's
synthetic-user procedure; (ii) the genuinely new half (not even fI has it):
every community activation decomposes exactly over the user's own purchases.

**Side-by-side, same user + same recommended item (shared renderer, real
checkpoints).** Thin user (|H|=3), item "Sweater · Black · Knitwear": the fU
panel shows S = B(t) 3.71 + personalized 1.17, top community +0.227 with
named profile and a per-purchase zoom (+0.184 trousers, +0.158 blouse,
−0.110 light-blue trousers, ID row −0.005 as its own hatched line),
feature-explained 98.8%. The host's panel on the *identical* prediction:
top-6 communities all +0.033 with identical duplicated names, remainder
+0.847 spread over 70 unnamed lines, zoom = post-hoc nearest-user lift
(declared). The comparison instrument works exactly as designed — the fU
explanation carries sharply more attributable structure — with one honest
symmetric caveat: **both** models' top lines carry duplicate community
names (the §4.5 collapse mars both sides of the showcase).

**The honesty devices earn their keep on the hard case.** For the heaviest
user (|H|=89) the fU footer reads *"feature-explained −0.47 vs user-ID
+0.77 of the personalized score +0.30 (features: −159.0%)"* — the M5′
division-of-labor handover, rendered rather than hidden: for heavy users the
ID row carries the personalization and the arithmetic says so. The ml-1m
first render (tag-vocabulary communities, movie titles in the zoom, features
101%) confirms the surface works on the second testbed's bags layout.

**Profile validity (committed spot-check, both post-hoc arms):** the
dual-route naming comparison validates the dominant community cleanly
(intrinsic top-3 == post-hoc lift top-3) and the distinct minority only
partially; the member-purchase-lift arm finds 2.4/5 top intrinsic
descriptors in the top-10 lift on hm, with the ml-1m readings dominated by
a collapse signature (near-identical correlations across all prototypes ⇒
shared member sets). **Verdict: partially validated — cleanly on the
dominant cluster, weakly on the tail; per-prototype validity claims are not
supportable at dev-profile, and the question is entangled with collapse.**

**Presentation-level context (working basis):** 94.3% of hm train purchases
share their date with another purchase by the same user (median user: all
purchases in same-day orders; 1.73 distinct purchase days on average). The
per-purchase zoom therefore describes *order members*, not independent
taste evidences — per-event context is the lineage's declared drop; the
epistemics live in the hidden-effects section, the figures stay
arithmetic-honest.

## 6. Requirements scorecard (evidence-backed, SC.8.5)

| Req | Rating (scope in the grade) | One-line evidence |
|---|---|---|
| R1 | strong (operative core; exemplar deferred to merge) | composition IS the scored representation; B(t) measured host-class |
| R2 | strong on derivability; profile *informativeness* collapse-limited at dev tier | exact shares verified in rendered artifacts |
| R3 | strong | prototype-shaped throughout |
| R4 | partial (attribution maximal; profile sharpness/distinctness collapsed, host-class) | entropy ≈ uniform, p99 overlap 1.0, ~5–6/76 clusters |
| R5 | partial (warm-thin delivered; absent-signal honest NO) | thin-strata gains 3–7× yardstick; cold rows above patched host; zero-interaction chain exact |
| R6 | split: stage bar MET on hm, MISSED on ml-1m; primary (UI) bar unassessed-deferred | §4.1 |
| R7 | partial | per-purchase zoom legible; taxonomy names partly R7-hostile |
| R8 | strong | one idea; host recovered exactly at V=0+ID |
| S1 | partial → **[CONTROL PENDING]** | decoupled control queued (F-DC05-21) |
| S2 | strong (re-scoped: history vocabulary; demographic variant deferred) | delivered |
| S3 | partial (open, not natural) | unchanged |
| S4 | strong (duplicate profiles would name identically — collapse caveat) | intrinsic profiles = the naming input |
| S5 | strong, run-confirmed | 415 MB/trial; 2h51m fleet walltime; no full-catalog pass |

## 7. Limitations and honest caveats

1. **Dev-profile provisionality.** Single seed, dev budget, one winning
   config shared by all five searches; nothing here is a thesis number.
2. **Prototype collapse** is the dominant qualitative limitation of the
   explanation surface at this tier — host-class in kind, config-mediated
   (coverage regularizer ≈ off in the shared winner), aggravated on ml-1m
   on the candidate's side. The identified lever (sim_batch) is
   deliberately NOT tuned here (Rashomon-stage scoping).
3. **B(t) dominance:** ~3/4 of cross-item score variance is non-personalized
   and popularity-shaped — host-class by measurement, disclosed on every
   artifact; the effective-bias share is a *lower bound* on popularity mass
   (geometry can conceal more), so it is always read with the geometry
   indicators.
4. **Attribution limit [CONTROL PENDING]:** until the lightfm_hist rows
   land, the hm warm win is "fU beats its host," not "the composition alone
   beats the host" — the prototype layer's share is unseparated (S1).
5. **Steck cosine caveat:** the trained pairing (q_u vs p^u_l) is protected
   by training-through-the-cosine; the profile read-out cos(e_f, p^u_l) is
   a different pairing and carries the residual risk class — mitigated by
   the dual-route validation (§5), which is itself only partially passed.
6. **Identifiability epistemics** (share gauge, lockstep cliques,
   canonical-representative t_l) are deliberately OFF the figures and live
   in the thesis hidden-effects section (ratified split; dc05 adds its own
   fU-emergent effects there: interaction-weighted coverage, norm handover,
   same-day order structure).
7. **ml-1m mass heterogeneity:** raw bags couple vote mass to popularity
   (ρ=0.672; baskets carry ~2× the catalog's token mass) — the user-side
   face of a standing must-mention; measured by the same instruments.
8. **Cold scope:** cold-item rows are hm-only (ml-1m cold deferred with the
   bags-kNN enablement); no cold-user testbed exists; the zero-interaction
   behavior is a verified property, not a measurement.
9. **Open items at drafting:** control-fleet backfill (warm ×4 + hm cold
   retrains ×2 + evals); noid>ids cold mechanism unresolved (small, open);
   no open findings in the ledger otherwise.

## Provenance

Dossier: `Master/docs/scrutiny/sc05_history_composed_user_factors.md`
(SC.1a–SC.8). Design doc: `Master/docs/design_candidates/
dc05_history_composed_user_factors.md` (all dated amendments). Instruments +
renders: `Master/experiments/sc8_dc05/` (mirror; generators committed —
`sc8_fu_geometry_readout.py`, `stratified_readout.py`, `popneg_readout.py`;
cluster re-run from committed scripts verified bit-identical). Frozen
references: `s0_foundation.md` §S0.7. Vault entries consumed at step 0:
information-ordering, ids-dominance theorem, sparsity-flip,
zero-interaction directive, Balog/scrutability, collapse trilogy
(self-built popularity bias / coverage-regularizer probe / per-item
mobility), base-settings 2×2 decision, ml-1m token-mass coupling, same-day
orders.
