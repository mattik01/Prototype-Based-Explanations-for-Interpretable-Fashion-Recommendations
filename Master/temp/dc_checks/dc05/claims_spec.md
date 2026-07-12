# dc05 claims spec — black-box toy check (protocol Step 4b)

You receive ONLY this spec. Do not read any other project document.

## Mechanism definition

Entities: N users, M items, K prototypes, embedding dim d, vocabulary of V
"word" rows plus (optionally) N per-user ID rows, all embeddings in R^d.

Parameters:
- word/ID embedding table E ∈ R^{(V+N)×d} (rows 0..V−1 = words, row V+u = user
  u's ID row; when the configuration is "no-ID", only the V word rows exist),
- user prototypes P ∈ R^{K×d},
- item table T ∈ R^{M×K} (one free vector t_i ∈ R^K per item).

Per user u: a fixed weight vector w_u ∈ R^V, w_{u,f} ≥ 0, derived from a
"basket" — a multiset H_u of items, each item carrying exactly one value per
each of F categorical fields (so |f_i| = F word tokens per item); n_{u,f} =
number of items in H_u carrying word f; the default normalization is
w_{u,f} = n_{u,f}/|H_u|.

Forward:
- composed user embedding q_u = Σ_f w_{u,f}·E[f] (+ E[V+u] if the ID row is
  enabled — added with weight 1),
- shifted cosine sim(a,b) = 1 + a·b/(‖a‖·‖b‖); use torch's
  nn.CosineSimilarity semantics (eps-guarded at zero vectors) where the edge
  case matters,
- user activation vector u* = [sim(q_u, P[l])]_{l=1..K} ∈ R^K,
- score S(u,i) = Σ_l t_{i,l} · u*_l.

Reference model ("host"): identical in every respect except q_u = a free
per-user embedding row (a plain N×d table).

## Claims to check (each: confirmed / falsified / untestable-at-toy-scale)

**C1 (exact per-row decomposition).** For every prototype l:
u*_l − 1 = Σ_{rows r of u} ω_r·E[r]·P[l] / (‖q_u‖·‖P[l]‖), where the rows of
u are its basket words with ω_f = w_{u,f} plus (if enabled) the ID row with
ω = 1. Exact to float64 precision, for random shapes/scales including F ∈
{1,2,5}, K ∈ {2,7}, and negative embedding values.

**C2 (per-purchase regrouping).** Define the purchase share of basket item i:
c_{i,l} = Σ_{f∈f_i} (1/|H_u|)·E[f]·P[l]/(‖q_u‖·‖P[l]‖). Then
Σ_{i∈H_u} c_{i,l} equals the metadata part of C1's sum (all word terms)
exactly — i.e. grouping the same summands by word or by purchase gives the
same total, both equal to u*_l − 1 minus the ID term.

**C3 (degenerate inputs).** (a) A user with EMPTY basket and no ID row has
q_u = 0; under torch CosineSimilarity eps-guarding, u* is the all-ones vector,
hence S(u,i) = Σ_l t_{i,l} for every item — verify this literal behavior with
torch. (b) Two users with identical normalized basket weight vectors and no ID
rows have bit-identical u*. (c) Users with different baskets generically have
different u* (report the min over seeds of max |Δu*|).

**C4 (score decomposition + baseline NON-inertness).** (a) S(u,i) =
B(t_i) + Σ_l t_{i,l}·(u*_l − 1) with B(t_i) = Σ_l t_{i,l}, exactly. (b) The
baseline B(t_i) is NOT rank-inert for ranking items: construct two items with
equal personalized parts Σ_l t_{i,l}·(u*_l −1) but different B(t) and confirm
their score order is decided by B(t); also report the empirical variance of
B(t) across randomly initialized items (nonzero).

**C5 (keystone reduction).** With V = 0 word rows (empty baskets) and the ID
row ENABLED, the model reduces exactly to the host: copying user table rows
U_host[u] into the ID rows E[V+u] makes u*, and S bit-identical to the host's
for all users/items.

**C6 (joint-normalization coupling).** Scaling one word row E[f0] by
α ∈ {10, 0.1, −3}: the pairwise RATIOS of all other rows' C1 shares are
unchanged, and their absolute values rescale by the common factor
‖q_old‖/‖q_new‖ — the coupling acts only through ‖q_u‖.

**C7 (item-vector gauge).** Let Ŝ = span({1_K} ∪ {rows of P̂ q̂}) ⊂ R^K more
precisely: the set of realizable activation vectors u* lies in the affine
subspace 1 + {P̂ x : x ∈ R^d} where P̂ has rows P[l]/‖P[l]‖. For any w ∈ R^K
with 1ᵀw = 0 and (P̂)ᵀ… — concretely: choose w orthogonal to the all-ones
vector AND to every column of the K×d matrix whose l-th row is P[l]/‖P[l]‖
(such w exists whenever K > d+1). Check: (a) replacing t_i by t_i + w changes
NO score S(u,i) for any user, exactly; (b) the gradient of any loss that
depends on the t_i only through scores is orthogonal to w (verify: autograd
gradient of a sampled-softmax-style loss w.r.t. t_i has zero component along
w, machine precision); (c) under plain SGD on such a loss, the component of
t_i along w stays bitwise at its initialization across steps; (d) adding L2
regularization λ‖t_i‖² makes that component decay by exactly (1−2ηλ) per step.

**C8 (normalization scale invariance / where normalization bites).** (a) With
no ID row, scaling a user's whole weight vector w_u by any α > 0 leaves u*
unchanged (cosine scale-invariance) — sum-vs-mean normalization is a no-op in
the no-ID configuration. (b) With the ID row enabled, sum-weights vs
mean-weights produce genuinely different u* (the choice acts only through the
metadata-vs-ID balance) — exhibit a case.

**C9 (mean-regression of heavy baskets — statistical demonstration, trend
only).** Simulate a population word distribution p over V words; for basket
sizes |H| ∈ {3, 10, 30, 100, 300}, draw many users' baskets i.i.d. from p,
compose q_u with mean normalization and no ID row, and measure the angle
between q_u and the population mean direction Σ_f p_f E[f]. Confirm the mean
angle decreases monotonically with |H| (report the numbers). This is a
statistical trend claim, not an exact identity.

## Deliverable

A toy script (numpy/torch, float64 where precision matters, toy dims, seconds
of runtime, no training pipeline, no datasets) in
`Master/temp/dc_checks/dc05/` (create it), e.g. `check_claims.py`, plus a
per-claim report: confirmed / falsified / untestable-at-toy-scale, with the
measured numbers. Use several seeds and adversarial scales where cheap. Python
env: `conda activate protomf` (torch available, CPU fine).
