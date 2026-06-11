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

## Procedure

- Write and run a small numpy/torch script (toy dims, e.g. d=8, K_u=3, K_t=4, F=3,
  a handful of users/items; seconds of runtime; no training, no datasets).
- Save the script(s) in `Master/temp/dc_checks/dc01/` and report per-claim verdicts
  with the numerical evidence (max abs error etc.).
