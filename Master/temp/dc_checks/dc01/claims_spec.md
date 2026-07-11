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
