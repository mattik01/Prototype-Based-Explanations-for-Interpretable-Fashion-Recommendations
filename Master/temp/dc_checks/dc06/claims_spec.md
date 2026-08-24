# dc06 claims spec (black-box toy check)

## Mechanism definition (self-contained)

Entities ("items") i = 1..n. There are F categorical fields; field f has a vocabulary
of values; each item carries exactly one value per field: x_if ("fixed layout").
Each vocabulary value v has a learned embedding row e_v ∈ R^d; each item additionally
has a free ID row e_ID(i) ∈ R^d. A catalog assigns values to items, inducing
per-field value frequencies p_f(v) = (#items with x_if = v)/n.

Per-field means over the vocabulary of field f:
- weighted flavour:   μ_f = Σ_{v∈f} p_f(v) · e_v
- unweighted flavour: μ_f = (1/|vocab_f|) Σ_{v∈f} e_v

Raw composition:     t_i,raw = Σ_f e_{x_if} + e_ID(i)
Centred composition: t_i     = Σ_f (e_{x_if} − μ_f) + e_ID(i)

Scoring host ("cosine host"): K prototype vectors p_k ∈ R^d; activation
t*_k(i) = 1 + cos(t_i, p_k); user affinity vector u ∈ R^K; score s(u,i) = Σ_k u_k·t*_k(i).

Alternative scorer ("dot host"): s(u,i) = q_uᵀ t_i with per-user vector q_u ∈ R^d.

Cold path: for a "cold" item the ID row is absent: t_cold = Σ_f (e_{x_if} − μ_f).

μ can be computed differentiably from the embedding table each forward pass
("live mean"), or from a detached copy ("detached mean").

Optional loss terms (not part of the base mechanism):
- gauge pin: L_pin = Σ_f ‖μ_f‖²  (weighted flavour μ)
- ID-mean pin over a batch B of items: L_idpin = ‖(1/|B|) Σ_{i∈B} e_ID(i)‖²

"Bag layout" variant (used by one dataset): an item carries a variable-length set of
tokens with per-token weights w_iv ≥ 0 (not one per field); centring subtracts each
token's field mean per token: t_i = Σ_v w_iv (e_v − μ_{f(v)}) + e_ID(i).

## Claims to check (per claim: confirmed / falsified / untestable-at-toy-scale)

- **C1 (exact shared vector, fixed layout):** t_i = t_i,raw − C with
  C = Σ_f μ_f identical for every item — for BOTH μ flavours.
- **C2 (gauge invariance):** adding an arbitrary constant vector c to EVERY embedding
  row of one field's vocabulary leaves every centred t_i exactly unchanged (both
  flavours), and leaves gradients of any loss on centred t_i w.r.t. all OTHER
  parameters unchanged; the raw composition t_i,raw does change.
- **C3 (exact share decomposition):** t_iᵀp_k = Σ_f (e_{x_if} − μ_f)ᵀp_k
  + e_ID(i)ᵀp_k exactly (numerator shares sum to the total with no residual).
- **C4 (cold path):** t_cold is generically nonzero and differs between items with
  different field values; two cold items sharing ALL field values have identical
  t_cold (before and after centring alike).
- **C5 (rank-inertness split):** on the dot host with fixed parameters, replacing all
  t_i by t_i − C (shared C) leaves every user's item RANKING unchanged. On the cosine
  host, the same replacement changes some user's item ranking in general (exhibit a
  concrete counterexample).
- **C6 (live vs detached mean gradient):** with live μ, the gradient of a loss
  depending only on item i's centred t_i is NONZERO on embedding rows of same-field
  values that item i does not carry; with detached μ it is exactly zero there.
- **C7 (gauge pin):** L_pin ≥ 0, equals 0 iff every weighted field mean is zero; its
  gradient reaches every metadata row of every field; at L_pin = 0, raw and centred
  compositions coincide.
- **C8 (ID-mean pin):** L_idpin = 0 iff the batch ID rows have zero mean; its
  gradient on each batch ID row is identical across rows (so all pairwise
  differences e_ID(i) − e_ID(j) receive exactly zero net gradient).
- **C9 (bag layout is weaker):** in the bag layout with variable token counts, the
  total subtracted mass Σ_v w_iv μ_{f(v)} is item-dependent in general (exhibit a
  counterexample to any shared-C claim); it reduces to the shared C when every item
  has exactly one token per field with weight 1.
- **C10 (shared component is not activation-inert on the cosine host):** adding a
  shared vector C to ALL compositions changes the activations t*_k in an
  item-dependent way (through the norms), i.e. there exist items i, j and a prototype
  p with t*_p(i)−t*_p(j) changed by the addition.

Write and run a toy script (numpy/torch, toy dims, seconds of runtime, no training
pipeline) in `Master/temp/dc_checks/dc06/`. Report per-claim verdicts with the
numbers/counterexamples that ground them.
