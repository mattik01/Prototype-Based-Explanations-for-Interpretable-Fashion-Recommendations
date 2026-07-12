# Claims spec — dc01 toy math check

You are given ONLY this spec. Do not read anything else in the repository.

## Mechanism definition

Entities: N users, M items. Each item i has a fixed set of F categorical feature
values plus its own ID, encoded as F+1 integer ids into a shared embedding table.

Parameters (all real-valued, any sensible random init):
- Feature/ID embedding table `E ∈ R^{(V+M)×d}` where V = total number of distinct
  categorical feature values. Item i's composed embedding: `q_i = Σ_{r ∈ rows(i)} E[r]`
  over its F+1 rows (F feature-value rows + 1 ID row).
- User embedding table `U ∈ R^{N×d}`; user u's embedding `u = U[u]`.
- User prototypes `P_u ∈ R^{K_u×d}`; item prototypes `P_t ∈ R^{K_t×d}`.
- Projection matrices `W_u ∈ R^{K_t×d}` and `W_t ∈ R^{K_u×d}`.

Forward pass for a (user u, item i) pair:
- `ustar_k = 1 + cos(u, P_u[k])` for k = 1..K_u   (shifted cosine; cos = a·b/(‖a‖‖b‖))
- `tstar_k = 1 + cos(q_i, P_t[k])` for k = 1..K_t
- `uhat = W_u @ u ∈ R^{K_t}` ; `that = W_t @ q_i ∈ R^{K_u}`
- Score: `S(u,i) = ustar · that + uhat · tstar`

The same `q_i` (one shared computation) feeds both `tstar` and `that`.

## Claims to check (report per claim: confirmed / falsified / untestable-at-toy-scale)

**C1 (exact per-feature decomposition).** For every prototype k:
`tstar_k − 1 = Σ_{r ∈ rows(i)} c_{r,k}` with `c_{r,k} = (E[r]·P_t[k]) / (‖q_i‖·‖P_t[k]‖)`,
exactly (up to float tolerance), for arbitrary random parameters and arbitrary
feature-set sizes.

**C2 (cold-item non-degeneracy).** Construct two "cold" items whose ID rows are
freshly initialized (e.g., zero or tiny random) but whose feature rows are trained/
arbitrary fixed vectors. (a) If the two items have different feature sets, their
activation vectors `tstar` differ. (b) If they have identical feature sets and the
ID rows are zero, their `tstar` vectors are identical. (c) `tstar` is non-constant
across k in general (not all prototypes get the same activation).

**C3 (gradient flows through the tie, from both score halves).** With the score
S as defined: the gradient ∂S/∂E[r] for a feature row r of the scored item is
non-zero in general, and it receives non-zero contributions from BOTH terms of the
score (`ustar·that` and `uhat·tstar`) — check by backpropagating each term separately.

**C4 (per-prototype additive decomposition of the score).** S(u,i) equals
`Σ_k ustar_k·that_k + Σ_k uhat_k·tstar_k` with each summand computable independently
(i.e., the score is a sum of K_u + K_t scalar contributions), exactly.

**C5 (reduction to the ID-only model).** If F = 0 (each item has only its ID row),
the model is functionally identical to one where items have a plain per-item
embedding table: for matched weights, forward outputs and score are equal.

**C6 (norm coupling of the decomposition).** In C1's decomposition, scaling ONE
feature row E[r0] (e.g., ×10) changes the contributions c_{r,k} of the OTHER rows
r ≠ r0 only through the shared scalar ‖q_i‖ — i.e., the ratios c_{r,k}/c_{r',k'}
between other rows' contributions (r, r' ≠ r0) remain unchanged, while their
absolute values shrink/grow by a common factor.

## Additional claims (SC.2 amendment re-check, 2026-07-12)

**C7a (exact linear-half decomposition).** With `that = W_t @ q_i` as defined above:
`ustar · that = Σ_{r ∈ rows(i)} ustar · (W_t @ E[r])`, exactly (up to float
tolerance), for arbitrary random parameters and arbitrary feature-set sizes.

**C7b (counterfactual property of the linear half — and its absence in the cosine
half).** Remove one row r0 from the composition: `q' = q_i − E[r0]`. Claim: the
`ustar · that` half changes by EXACTLY `−ustar · (W_t @ E[r0])` (the r0 summand of
C7a). By contrast, in the cosine half the shares `c_{r,k}` (as in C1) of the
REMAINING rows r ≠ r0 change under the removal (through ‖q_i‖) — i.e. the linear
half's per-row terms are removal-exact while the cosine half's are not.

**C8 (lockstep gauge freedom — trained, not just algebraic).** Construct a toy
dataset in which two feature values a and b ALWAYS co-occur (every item carrying a
also carries b, and vice versa; other values vary freely). Train the full model
(any differentiable loss through the scores S; plain SGD, NO weight decay, a few
hundred steps suffice — convergence not required). Claims:
(a) at every training step, `grad E[a] == grad E[b]` exactly (they enter the loss
    only through their sum), hence `E[a] − E[b]` remains at its INITIAL value
    throughout training while `E[a] + E[b]` moves;
(b) with L2 weight decay added, `E[a] − E[b]` decays toward zero instead;
(c) across different random initializations (same data, same training), the SPLIT
    of the pair's combined cosine share (c_a,k vs c_b,k, per C1) varies across
    runs, while their SUM (c_a,k + c_b,k) is a trained quantity (varies far less —
    compare relative dispersions across ≥5 seeds).

## Procedure

- Write and run a small numpy/torch script (toy dims, e.g. d=8, K_u=3, K_t=4, F=3,
  a handful of users/items; seconds of runtime; no training, no datasets — EXCEPT
  C8, which requires the small training loop described there, still seconds).
- Save the script(s) in `Master/temp/dc_checks/dc01/` and report per-claim verdicts
  with the numerical evidence (max abs error etc.).
- SC.2 re-check scope: C7a, C7b, C8 (C1–C6 were confirmed 2026-06-11 and are not
  re-run unless a C7/C8 result contradicts them).

## fI re-check (re-host amendment, 2026-07-12)

**Mechanism delta — fI model (supersedes the forward pass above FOR THE CLAIMS OF
THIS SECTION ONLY):**
- Item side unchanged: item i's composed embedding `q_i = Σ_{r ∈ rows(i)} E[r]`
  over its F+1 rows (F feature-value rows + 1 ID row); item prototypes
  `P_t ∈ R^{K_t×d}`; `tstar_k = 1 + cos(q_i, P_t[k])` for k = 1..K_t.
- User side: a free vector `u = U[u] ∈ R^{K_t}` (NO user prototypes, NO projection
  matrices — W_u, W_t, P_u, ustar, uhat, that do not exist in this model).
- Score: `S(u,i) = u · tstar = Σ_k u_k · tstar_k`.

**Claims to check (same reporting format):**

**C3′ (gradient through the single path).** ∂S/∂E[r] is non-zero in general for a
feature row r of the scored item, with S = u·tstar(q_i) as the sole score path.

**C4′ (exact full-score decomposition + rank-inert baseline).**
(a) `S = Σ_k u_k·tstar_k` exactly, with each of the K_t summands independently
    computable.
(b) `S = Σ_k u_k + Σ_k Σ_{r ∈ rows(i)} u_k·c_{r,k}` exactly, with
    `c_{r,k} = (E[r]·P_t[k])/(‖q_i‖·‖P_t[k]‖)` (the C1 shares).
(c) The baseline term `Σ_k u_k` is item-independent: for any candidate set of
    items scored by the same user, the ranking (argsort) of `S(u,·)` is identical
    to the ranking of the feature part `S(u,·) − Σ_k u_k`, and pairwise score
    differences S(u,i) − S(u,j) equal the differences of the feature parts alone,
    exactly.

**C5′ (host reduction).** If F = 0 (each item has only its ID row), the model is
functionally identical to I-ProtoMF with a plain per-item embedding table: for
matched weights (Q[i] = E[V+i], same P_t, same U), tstar and S are bit-identical.

**Procedure (fI scope):** script(s) in `Master/temp/dc_checks/dc01/` (e.g.
`check_claims_fi.py`); adversarial sweep in the style above (multiple seeds,
F ∈ {0..several}, parameter scales, shapes incl. K_t=1); no training needed.
Scope: C3′/C4′/C5′ only — C1/C2/C6/C8 were verified previously and are
composition-level (mechanism identical); C7a/C7b are parked with the projection
branch (do not run them against this model — their objects do not exist here).

## C9 — user-side gauge (SC.1b-delta re-check, 2026-07-12)

fI model as defined in the "fI re-check" section above. Let P̂ ∈ R^{K_t×d} be the
row-normalized item-prototype matrix (row k = P_t[k]/‖P_t[k]‖). Define the gauge
space G(P̂) = {w ∈ R^{K_t} : P̂ᵀw = 0 and 1ᵀw = 0}. Note dim G ≥ K_t − d − 1;
construct toys with K_t > d + 1 (e.g. K_t = 8, d = 3) so G is nontrivial.

**C9(a) (exact score invariance).** For any u, any item i, and any w ∈ G(P̂):
S(u + w, i) = S(u, i) exactly (float tolerance), for arbitrary random parameters
and multiple seeds/scales. (Compute an orthonormal basis of G numerically, e.g.
from the SVD null space of the (d+1)×K_t matrix stacking P̂ᵀ and 1ᵀ.)

**C9(b) (gauge component frozen under training).** Train the fI toy with plain
SGD (any differentiable loss through the scores; a few hundred steps; NO weight
decay), in two regimes:
(b1) prototypes and item table FROZEN, only U (and optionally E) trainable:
     the projection of each trained user vector onto G(P̂) stays at its INITIAL
     value throughout training (up to update-arithmetic rounding), while the
     in-span component moves.
(b2) everything trainable: at EVERY step, the loss gradient wrt each user vector
     is orthogonal to G(P̂_current) (max |gᵀw| over a basis of the CURRENT gauge
     space ≈ 0 at float tolerance; the space itself may rotate as P_t moves —
     compute it per step).

**C9(c) (L2 canonicalization).** As (b1) but with SGD weight_decay = wd > 0: the
gauge component of u decays by exactly (1 − lr·wd)^T after T steps (the loss
gradient contributes nothing along G; only decay acts), driving u toward the
minimum-norm (in-span) representative.

**Procedure (C9 scope):** script `check_claims_c9.py` in
`Master/temp/dc_checks/dc01/`; report per-claim verdicts with numerical evidence.
C9 only — do not re-run earlier claims.
