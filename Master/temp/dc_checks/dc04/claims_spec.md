# dc04 claims spec — black-box toy verification

You are verifying mathematical claims about a model mechanism. This document is
your ONLY input. Design and run a toy script (numpy/torch, toy dims, seconds of
runtime, float64, fixed seeds, no training pipeline / no datasets) in this
directory. Report per claim: **confirmed / falsified / untestable-at-toy-scale**,
with the concrete numbers that justify each verdict. Where a claim is ambiguous,
state the reading you tested.

## Mechanism definition

Entities: N_u users, M items, d-dim embeddings. Each item i has F categorical
attribute fields, exactly one value per field; its attribute-value id set is
x_i ⊂ {1..V} with |x_i| = F. Encode as an (M, F) integer matrix `feature_ids`.

Parameters:
- User table U (N_u, d); user prototypes P_u (L_u, d); W_u (d, K); all free.
- Item table T (M, d); W_t (d, L_u); both free.
- Profile logits Atil (K, V); profiles A = softmax(Atil, dim=-1), rows on the
  simplex.

Buffers (NOT parameters, no gradient):
- Anchors C (V, d) with c_v = mean of rows T[i] over all WARM items i whose
  x_i contains v (warm = a given boolean mask over items); counts n_v.
  C is recomputed by assignment from the current T ("refresh"), not learned.

Forward (batch B users, N items per user):
- u = U[u_idxs] (B, d); u_star = 1 + cos(u, P_u) rowwise → (B, L_u);
  u_hat = u @ W_u → (B, K).
- P_t = A @ C → (K, d)  [item prototypes: convex combinations of anchors]
- t = T[i_idxs] (B, N, d); t_star = 1 + cos(t, P_t) → (B, N, K);
  t_hat = t @ W_t → (B, N, L_u).
- score = Σ_l u_star[:,l]·t_hat[:,:,l] + Σ_k u_hat[:,k]·t_star[:,:,k] → (B, N).

cos(a, b) is the standard cosine similarity of vectors; "1 + cos" shifts it to
[0, 2].

Cold-item rule: an item with an untrained embedding row is represented by
t_cold = (1/F) Σ_{v ∈ x_cold} C[v] wherever T[i] would be used (both t_star and
t_hat paths).

Optional knobs (each a separate claim below, none part of the base forward):
- K-ENT: L_ent = (1/K) Σ_k H(a_k), H = Shannon entropy of the profile row.
- K-SEP: L_sep = mean over pairs k≠j of Σ_v min(a_{k,v}, a_{j,v}).
- K-CENTER: replace C by C' with c'_v = c_v − c̄, c̄ = mean of T's warm rows,
  in the P_t construction only.
- K-TEMP: A = softmax(Atil / τ), τ > 0.

## Claims

**CL1 (shapes).** With B=8, N=5, d=16, L_u=4, K=6, V=15, F=3, M=40, N_u=30:
the forward produces u_star (8,4), u_hat (8,6), t_star (8,5,6), t_hat (8,5,4),
score (8,5).

**CL2 (exact per-anchor decomposition).** For every item i and prototype k:
t_star_k − 1 = Σ_v A[k,v] · (t_iᵀ C[v]) / (‖t_i‖ ‖P_t[k]‖), exactly (float64
tolerance).

**CL3 (exact score decomposition).** score(u, i) equals the sum of L_u + K
scalar addends {u_star_l · t_hat_l} ∪ {u_hat_k · t_star_k} with no residual.

**CL4 (gradient routing).** Backprop from Σ score²: nonzero gradients reach
Atil, T, U, P_u, W_u, W_t; C receives none (buffer, requires_grad=False), and
if C were made a leaf tensor with requires_grad=True while still refreshed by
assignment-from-T-under-no_grad, the mathematical gradient ∂score/∂C is nonzero
— test the implemented (detached) behavior: C.grad is None / unused.

**CL5 (hull property & temperature limit).** Every P_t[k] lies in the convex
hull of the rows of C (verify: solving/checking barycentric coordinates =
A[k,:] ≥ 0 summing to 1 reproduces P_t[k]); with K-TEMP as τ → 0 (e.g. τ =
1e-6), each P_t[k] converges to the single anchor row argmax_v Atil[k,v]
(exact match at machine precision).

**CL6 (cold rule).** (a) Two cold items with identical attribute rows receive
bitwise-identical representations and hence identical scores for every user.
(b) A cold item's t_star is finite and non-constant across k (generic random
setup). (c) Two cold items differing in at least one attribute value receive
different t_star generically (report the min over ≥10 seeds of the max
difference).

**CL7 (spread vs table structure — two-sided property check).**
Construct two toy item tables of equal size (e.g. M=200, d=16, F=3, V=15):
(ISO) rows i.i.d. isotropic Gaussian, attributes assigned randomly;
(STRUCT) rows drawn from per-attribute-value cluster centers (embeddings
correlate with attribute values). Refresh C from each table. Compare, for the
same random profiles A: spread(t_star) = std over k of t_star, averaged over
items — against the same quantity for FREE prototypes drawn i.i.d. Gaussian
(K, d).
- CL7a: under ISO, spread with anchored prototypes is at least ~5× smaller
  than with free prototypes (the anchoring collapses similarity spread when
  embeddings carry no attribute structure).
- CL7b: under STRUCT, anchored spread is within the same order of magnitude as
  free-prototype spread (ratio ≥ ~0.3).
Report the actual ratios over ≥5 seeds.

**CL8 (weight-decay uniformity drift).** Applying logit decay Atil ← (1−ηλ)
Atil (η λ ∈ (0,1)) strictly increases the entropy of every non-uniform profile
row (report min entropy increase over random rows), i.e. plain L2 weight decay
on Atil drives profiles toward uniform.

**CL9 (K-ENT mechanics).** L_ent ≈ 0 at one-hot-like profiles (logits with one
large entry), maximal (= ln V) at uniform; gradient reaches Atil and a GD step
from a mildly peaked profile strictly decreases L_ent.

**CL10 (K-SEP mechanics).** L_sep = 1 for two identical profile rows; = 0 for
disjoint supports (report using near-one-hot profiles on different values);
for a near-duplicate pair (cos of profiles > 0.99 but not equal), one GD step
on L_sep strictly decreases their overlap. Also report the behavior at an
EXACTLY duplicated pair (is the gradient zero or nonzero? min is only
subdifferentiable at ties — report what autograd does).

**CL11 (K-CENTER effect in the degenerate regime).** Under CL7's ISO table
with an added common mean vector (all rows shifted by a constant vector μ with
‖μ‖ comparable to row norms — an anisotropic table), centered anchors (K-CENTER)
yield strictly larger spread(t_star) than uncentered anchors (report ratio).

**CL12 (refresh correctness).** The vectorized refresh (index_add over
feature_ids of warm rows, divided by counts) equals the naive per-value loop
mean over warm items exactly; counts equal the per-value warm class sizes;
items outside the warm mask do not influence C.

## Environment

Run with the repo's conda env: `conda activate protomf` (torch 1.9.1 CPU).
Write the script as `check_claims.py` in this directory
(`Master/temp/dc_checks/dc04/`). Total runtime target: seconds.
