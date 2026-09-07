# Prototype grounding, item side — the design space beyond "constrain the item embeddings"

> Light design-candidate pass, 2026-08-22 (not a protocol cycle; nothing written to
> `design_candidates/`). Question (Matteo): dc01 grounds the prototypes *by consequence* —
> item embeddings are no longer free vectors, so the space is grounded and the prototypes
> inherit it. Is there something that touches the prototypes more directly, without
> constraining the model so aggressively — even a tag-on — while staying intrinsic and
> inside ProtoMF's philosophy? Linear tail: nice, not sacred (user, mid-task).
> Four independent forks explored one locus each; full write-ups in
> `grounding_candidates_2026-08-22/cand_L2..L5.md`. This memo is the synthesis.

## 1. The axis behind the question: WHERE the vocabulary attaches

Invariant in every design: `s(u,i) = Σ_k u_k · t*_k(i)` on the I-ProtoMF host; designs differ
only in how `t*_k` is produced and how `p_k` is named.

| locus | what is written in the vocabulary | items | existing / new |
|---|---|---|---|
| **L1 entities** | item representation itself | composed (not free) | dc01 (emergent), dc02 (bottleneck) |
| **L2 prototype parameterization** | the prototype *is* a named point | free | dc04 (dead: preimage) → **wI** (word-anchored, §2.1) |
| **L3 item→prototype interface** | a supervised decode of the item | free | **dI** (decoded-attribute bottleneck, §2.2) |
| **L4 objective only** | a profile bound to activations by a loss | free | **sI** (surrogate-profile loss, §2.3) — the pure tag-on |
| **L5 the similarity itself** | a declared facet spec that gates the similarity | free | **gI** (facet-gated prototypes, §2.4) |

Second axis (already in the thesis narrative): enforcement strength — emergent (dc01) →
loss-enforced (sI) → by-construction (dc02, wI, dI, gI).

**No-free-lunch finding (holds across all four forks).** A prototype cannot be *intrinsically*
grounded unless something in the forward pass is written in the vocabulary — otherwise the
reading is post-hoc by definition. With items free, the vocabulary must be re-anchored to
items by one of exactly four means, each with its own signature risk:
(a) **data statistics** (centroids) → identifiability: continuous profiles over V anchors in
d<V dims are arbitrary preimages (dc04's death); only *discrete* profiles (prototype = one
word / one signature node) survive; (b) **supervision** (attribute decoder) → concept
leakage (confidently-wrong attribute statements); (c) **observed attributes gating** the CF
similarity → capacity flight (gates → open) and coverage; (d) **a loss only** → fidelity is
empirical (R²), never structural. "Aggressiveness" does not disappear — it *relocates*:
dc01 constrains the item representation; wI the coordinate system; dI the interface; gI
what each prototype may see; sI nothing structurally (and pays in Rudin-strength).

## 2. The four candidates (one per locus), condensed

### 2.1 wI-ProtoMF — word-anchored prototypes ("the vocabulary IS the prototype set") — L2
Delete the prototype table. Prototype *w* := centred centroid of the items carrying
user-perceivable word w (K = |𝒲| ≈ 60–150, R7 fields only), computed **live and
differentiably** from the free item table each step (one sparse K×M·M×d product):
`p_w = c_w − μ`, `t*_iw = 1 + cos(t_i − μ, p_w)`, `s = Σ_w u_w t*_iw`. No prototype params,
no profile, no new loss. Live centroids = the captured-dc04 REGULARIZE route *with* a
read-out (gradient pulls same-word items toward shared prototype profiles). Cold item =
μ + mean of its words' anchors. Read-out: prototype meaning definitional; per-prototype
contribution exact; **no per-feature share by design** (nothing below one cosine) — rendered
as `tagged ✓/✗ | behaves-like 0.61` (mis-tag probe native). Toy: centring is load-bearing
(uncentred centroids all ≈ global mean → cos ≈ 0.66 for every word). Price: prototype =
one word, **not a taste group** — conjunction moves up into the user vector (u = taste over
words, fU's goal for free, but additive only). R1/R2/R5/R7/R8 strong, R3/R4 partial, R6 open.
Rejected siblings: signature-push (ProtoPNet push onto nearest signature centroid — keeps
groups, ≈ post-hoc nearest-class naming), lattice nodes, facet-softmax, dc04 with V≤d, ST-Gumbel.

### 2.2 dI-ProtoMF — decoded-attribute bottleneck ("ground the interface, not the item") — L3
Items free. Decoder `ĉ_i = σ_fields(W_D t_i + b_D)` (per-field softmax) predicts the item's
attribute row, supervised by `λ_c·CE(ĉ_i, x_i)` on batch items; item prototypes `a_k ∈ R^V`
live in attribute space (dc02's object); `t*_k = 1 + cos(ĉ_i, a_k)`; score unchanged. Cold
items: `ĉ := x` — same space, zero extra rule. ≈61k new params, one matmul/step. A *joint*
CBM (Koh et al., E1, verified) with the CF embedding as concept-predictor input. **Leakage is
the dial and it is readable:** `r_i = ĉ_i − x_i` is dc01's ID-residual expressed in vocabulary
("catalogue: Basic; model treats as 0.3 Premium"); λ_c→∞ = dc02 (twins collapse, 68%
ceiling), λ_c→0 = labelled black box; per-attribute shares exact under cos and split into
catalogue part + leak part (toy-verified); identifiable (named coordinates, no preimage).
Tied variant `W_D = Eᵀ` (items decoded by word directions = dc01 in reverse; a shared E at
the merge). Dot instead of cos = linear down to attributes (ablation rung). Risks: wrong
statements about the item on screen (R7), calibration, rare values, d<V reachability.
R2/R3/R5/R8 strong, R4 strong-conditional, R1/R6/R7 partial.

### 2.3 sI-ProtoMF — surrogate-profile loss (the pure tag-on, loss-enforced rung) — L4
Host byte-identical. Per prototype a profile `a_k ∈ R^V` + intercept; **one loss**
`mean_{i,k}((t*_k−1) − a_kᵀx_i − b_k)²` with weight λ_g on the extractor accumulator.
Two-way gradient: fits the profile to activations AND pulls activations toward
attribute-predictability (REGULARIZE route on the prototype-similarity profile). λ_g is a
continuous enforcement knob (0 = host + post-hoc probe; ∞ = dc02-like). Cold items:
`t*_k = 1 + clip(a_kᵀx_i + b_k)` — there the profile IS the forward pass. Read-out: per-field
profile **with an R² fidelity stamp per prototype** (gated on activation variance), exact
shares for the surrogate, rendered per-item residual r_ik = measured latent mass
(dc01's ID-share analogue, measured not parameterised). Identifiable up to F−1 per-field
offsets (toy). R2 verdict: "regularisation-enforced intrinsic" — defensible by ProtoMF's
own standard (its inclusion regularisers are what make its nearest-item reading
meaningful; A1 verified), EMF (S2 §3.2 verified), SENN (E2 verified); Rudin-weak for warm
items; the dc04-S1 objection transfers → the **λ_g=0 control must be run** (R²/latent-mass
distributions host vs candidate at matched accuracy, Rashomon-tuned). Brutal take:
mechanism = jointly trained linear probe + regulariser; real assets = cold route,
continuous spectrum knob on ONE host, fidelity stamp. R3/R5/R6/R8 strong, R1/R2/R4/R7
partial. Sibling: anchor-inclusive regularisation (most ProtoMF-native, ~10 lines, but
no per-feature shares and re-imports centroid pathologies).

### 2.4 gI-ProtoMF — facet-gated prototypes ("rule ∧ direction") — L5
Items free, prototypes keep a free CF direction p_k **and** gain a declared attribute facet
spec: field gates γ_kf and acceptance weights w_kfv on observed one-hots;
`g_k(i) = Σ_f γ_kf·w_kf[x_if] / Σ_f γ_kf` (fraction of declared facets satisfied, mean not
product), `t*_k = 1 + g_k(i)·cos(t_i, p_k)`. Train relaxed, harden to a rule (ProtoTree
soft→hard, C5a verified): "P7 = colour∈{Dark Blue, Black} ∧ garment∈{Trousers}" + a free
within-rule direction (latency cornered and *scoped* per prototype; γ≡0 prototypes are
declared ungrounded = honesty meter). Spec is load-bearing (multiplies the score →
Rudin-intrinsic) and identifiable (weights act on observed values, not centroids). Cold:
g exact, cos → running mean ρ_k. Attribute-group richness **survives and is learned**.
Toy: hard AND of ≥2 single values leaves most items unmatched (K=50, m=2 → 48% coverage)
→ value *sets*, mean gate, and an R_batch-style coverage regulariser are part of the
design. Risks: capacity flight (γ→0 = host) needs a facet budget; acceptance sparsity;
discrete-search fragility; accuracy ≤ host by construction. R2/R3/R4 strong, R7 a design
switch, R5 partial, R6 open, R8 partial (two guard knobs).

## 3. Cross-cutting observations (synthesizer's own)

1. **"Touches the prototypes directly" has three honest realizations, not one:** the
   prototype *is* a word (wI), the prototype *declares* a spec (gI), the prototype *carries* a
   fitted profile (sI). dI is different in kind: it grounds the *interface* and keeps dc02's
   prototype object. Only sI is a true tag-on (host untouched); gI is near (one multiply on
   the sim tensor); dI adds a decoder+loss; wI is a substitution.
2. **ProtoMF-philosophy check (learned taste-group prototypes + additive per-prototype
   explanation):** gI keeps it fully (learned groups, now with names that are rules); sI
   keeps it fully; dI keeps the prototype layer's form; wI keeps the scorer but gives up
   *learned* prototypes — it is the literal form of the chapter's musing "grounding the
   space may render prototypes obsolete" (prototypes = vocabulary; grouping moves to u).
3. **Fidelity becomes measurable exactly where the constraint is relaxed:** sI prints R²,
   dI prints r_i, gI prints the fraction of ungrounded prototypes, wI prints tagged-vs-behaves.
   dc01/dc02 cannot print a fidelity number because their grounding is by construction —
   that is a narrative asset for the "corner latency and quantify it" thesis line.
4. **Thesis §5.1 consequence.** "Grounding = constraining the latent factors to a chosen
   vocabulary" covers L1 only. If any L2–L5 candidate enters, widen to: *constraining the
   meaning-bearing components of the score — the item representation, the coordinate
   system the items are measured in, the item→prototype interface, or what a prototype may
   compare on — to a chosen vocabulary*. "Its meaning is its composition" generalizes to
   "its meaning is its definition / its spec / its profile".
5. **Two designed controls fall out for free:** the λ_g=0 post-hoc probe (sI) and the
   λ_c→∞ hard bottleneck (dI = dc02) — both are S1 rungs the requirements doc already asks for.
6. **Merge-stage hooks:** wI's u is already a taste-over-words vector (fU by another route);
   dI's tied variant `W_D = Eᵀ` would share the word table with fI at the fUfI merge.

## 4. What I think (recommendation, not a protocol ranking)

- If the goal is *minimal change, ProtoMF-native, and the thesis's spectrum narrative*:
  **sI** is the missing middle rung and costs almost nothing to build — but it is only
  worth claiming if the λ_g=0 control shows a real fidelity delta; otherwise demote it to a
  diagnostic. Build it as an instrument first, candidate second.
- If the goal is a *genuinely new intrinsic free-item candidate that keeps learned
  prototypes*: **gI (facet-gated)** is the most ProtoMF-faithful answer to the question as
  asked — prototypes stay learned taste groups, items stay free, the grounding is a
  load-bearing, identifiable rule. Its risk is the discrete optimisation and capacity flight.
- **dI** is the most thesis-coherent with dc02 (same prototype object, lifts the twin
  ceiling by a disclosed amount, makes the ID residual readable) — effectively "dc02 with a
  soft, measured bottleneck". If dc02 is revived, it should be revived as dI.
- **wI** is the cleanest and most identifiable, but it answers a different question
  (prototypes as vocabulary) and sacrifices learned prototypes; keep it as the end-point of
  the re-coordinatization argument and as a strong, cheap baseline/contrast.

Suggested next move: a scratchpad capture for each (four `### [captured light-pass
2026-08-22]` entries), then pick at most one for a full `/design-candidate` cycle.
Candidate files: `Master/temp/grounding_candidates_2026-08-22/`.
