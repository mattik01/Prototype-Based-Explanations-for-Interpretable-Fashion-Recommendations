# Claims spec — mechanism M (black-box toy check)

You receive ONLY this spec. Implement the mechanism exactly as defined, design
toy tests for the claims, and report per claim: **confirmed / falsified /
untestable-at-toy-scale** (with one-line reasoning each).

## Mechanism definition

Dimensions (toy scale suggestion): N_u users, M items, F categorical fields with
per-field vocabulary sizes V_1..V_F, V = Σ V_f. Latent dim d, user prototypes
L_u, item prototypes K. Batch B, n = 1 + n_neg sampled items per user.

Fixed data: every item i has exactly ONE value per field → a binary multi-hot
x_i ∈ {0,1}^V with exactly F ones (one inside each field's block). X ∈ {0,1}^{M×V}
is a fixed constant (NOT a parameter).

Learned parameters:
- U ∈ R^{N_u × d} (user embedding table)
- P ∈ R^{L_u × d} (user prototypes)
- W_u ∈ R^{d × K} (user projection, no bias)
- A ∈ R^{K × V} (item prototypes in attribute space)
- W_t ∈ R^{V × L_u} (item projection, no bias)
- NO per-item parameters of any kind; NO bias terms.

Shifted cosine: sc(a, b) = 1 + a·b / (‖a‖₂ ‖b‖₂).

Forward pass, given user indices u_idxs (B,) and item indices i_idxs (B, n):
- u = U[u_idxs]                            (B, d)
- u*_l = sc(u, P_l)                        (B, L_u)
- û = u @ W_u                              (B, K)
- x = X[i_idxs]                            (B, n, V)
- t*_k = sc(x, A_k)                        (B, n, K)
- t̂ = x @ W_t                             (B, n, L_u)
- score = Σ_l u*_l · t̂_[:,:,l] + Σ_k û_k · t*_[:,:,k]   (B, n)
  (i.e. score = concat(u*, û) · concat(t̂, t*) along the last dim, with u-side
  broadcast over n)

Regularizers, computed on S = t* reshaped to (B·n, K):
- R_batch = − mean over rows of (max over columns of S)
- R_proto = − mean over columns of (max over rows of S)

Optional constraint modules (each defined independently; none is part of the
base forward):
- (C1) Nonnegativity reparameterization: A = softplus(Ã), Ã ∈ R^{K×V} the
  actual parameter.
- (C2) Field crispness: L_crisp = (1/(K·F)) Σ_k Σ_f H(softmax(A[k, block_f]))
  where block_f indexes field f's V_f columns and H is Shannon entropy (nats).
- (C3) Separation: L_sep = mean over unordered pairs k≠j of
  max(0, cos(A_k, A_j) − m), plain cosine (not shifted), margin m ∈ [0,1).
- (C4) Data pull: L_push = (1/K) Σ_k min_{i ∈ S_items} (1 − cos(A_k, x_i)) over
  a fixed item subset S_items.

## Claims to check

1. **Shapes.** The forward pass runs at (B, n) granularity and yields score of
   shape (B, n) with the intermediate shapes exactly as annotated above.

2. **Cold-item non-degeneracy.** Construct a "cold" item index whose row of X is
   set but which is excluded from all training batches during a short toy
   optimization of all parameters (any loss that decreases). After training, its
   t* is (a) finite, (b) non-constant across k (unless X row is degenerate),
   (c) EXACTLY equal to the t* of a "warm" item that has an identical X row.

3. **Exact per-attribute decomposition.** For every item and prototype:
   t*_k − 1 = Σ_{v: x_i[v]=1} A[k, v] / (√F · ‖A_k‖₂), to float tolerance.

4. **Exact score decomposition.** score(u, i) equals the sum of the L_u + K
   per-prototype addends u*_l·t̂_l and û_k·t*_k, to float tolerance (no hidden
   interaction term).

5. **Gradient flow.** One backward pass from an arbitrary scalar loss on score
   reaches ALL of U, P, W_u, A, W_t with generically nonzero gradients; and X
   receives no gradient (it is constant).

6. **Same-signature ties.** Two distinct item indices whose X rows are identical
   receive identical scores for EVERY user (exact tie), under this mechanism
   (no biases). Additionally: adding a per-item bias b_i breaks the tie —
   i.e. the tie property is destroyed by that single addition.

7. **Regularizer direction.** (a) R_batch strictly decreases when A is changed
   so that some item's best-matching prototype gets strictly more aligned with
   its x (others held fixed). (b) R_proto strictly decreases when a previously
   worst-claimed prototype is rotated toward some batch item's x. Directional
   checks at a random configuration suffice (finite-difference or autograd
   sign check).

8. **C1 mechanics.** Under the softplus reparameterization: all entries of A are
   > 0; gradients reach Ã; and with A > 0 and binary x, every t* lies in [1, 2]
   (the repulsion half [0,1) becomes unreachable).

9. **C2 mechanics.** L_crisp is minimized (→0) when each field block of each
   prototype row is a one-hot-like concentration (entropy → 0 as one coordinate
   dominates) and is maximal at uniform blocks; gradient reaches A (or Ã).

10. **C3 mechanics.** L_sep = 0 when all prototype pairs have cosine ≤ m;
    positive when two prototypes are (near-)duplicates; gradient reaches A.

11. **C4 mechanics.** L_push = 0 iff every prototype is a positive scalar
    multiple of some x_i in S_items; positive otherwise; gradient reaches A.

## Operational constraints

- numpy/torch only, toy dims (e.g. N_u=30, M=40, F=3, V_f∈{4,5,6}, d=8, L_u=4,
  K=5, B=8, n=4), runtime seconds, NO real data, NO training pipeline.
- Write the script(s) to `Master/temp/dc_checks/dc02/` (repo root =
  /home/mattik01/Desktop/githubs/ProtoMF). Use `conda activate protomf`
  (torch 1.9.1, CPU).
- Report strictly per-claim verdicts with one-line evidence each.
