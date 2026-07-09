# dc03 claims spec — black-box toy check

You are given ONLY this spec. Design and run a toy numpy/torch script (toy dims,
seconds of runtime, no training pipeline) that tests each claim below. Report
per-claim: **confirmed / falsified / untestable-at-toy-scale**, with one line of
evidence (numbers) each.

## Mechanism definition

Dimensions (toy-scale suggestion: M=40 items, F=12, d=8, K=5 prototypes,
L_u=4 user prototypes, N_users=20, batch B=6, n=3 items per row).

- Frozen item content matrix `V ∈ R^{M×F}` (random, rows L2-normalized).
- Trainable projection `E ∈ R^{F×d}`; item representation `q_i = V[i] @ E`.
- Item prototypes `P ∈ R^{K×d}` (trainable).
- Shifted cosine: `sim(a,b) = 1 + cos(a,b)` (elementwise over prototype rows;
  use an eps-guarded cosine like PyTorch `CosineSimilarity(dim=-1, eps=1e-8)`).
- Item branch outputs, for item indices `i_idxs` of shape (B,n):
  `t_star[b,j,k] = sim(q_{i_idxs[b,j]}, P[k])` → shape (B,n,K), and
  `t_hat = q @ W_t` with trainable `W_t ∈ R^{d×L_u}` → shape (B,n,L_u).
- User branch: free user embeddings `U ∈ R^{N_users×d}`, user prototypes
  `P_u ∈ R^{L_u×d}`, `u_star[b,l] = sim(U[u_b], P_u[l])` → (B,L_u), and
  `u_hat = U[u_b] @ W_u` with `W_u ∈ R^{d×K}` → (B,K).
- Score for row b, item j:
  `score[b,j] = u_star[b] · t_hat[b,j] + u_hat[b] · t_star[b,j]`.
- **Push step** (no grad): for each prototype k,
  `i*(k) = argmax_i cos(q_i, P[k])` over all M items, then `P[k] ← q_{i*(k)}`.
- Host regularizer (item side): over a batch-vs-prototype similarity matrix
  `S[j,k] = sim(q_j, P[k])` (j over the flattened batch items),
  `R_proto = −mean_k max_j S[j,k]` and `R_batch = −mean_j max_k S[j,k]`.
- Proposed diversity loss: `L_div = (2/(K(K−1))) Σ_{k<k'} max(0, cos(P_k,P_k') − τ)`.
- Proposed orthogonality penalty: `L_orth = ‖EᵀE − I_d‖²_F`.

## Claims

- **CL1 (decomposition):** with `c[b,j,k] = u_hat[b,k] * t_star[b,j,k]`,
  `Σ_k c[b,j,k]` equals the second score term exactly (float tolerance), for all
  b,j — i.e. the score's prototype half decomposes into per-prototype
  contributions.
- **CL2 (tie gradient):** `E` receives nonzero gradient through BOTH item-branch
  paths: backprop a scalar built only from `t_star` → grad(E) ≠ 0; separately,
  a scalar built only from `t_hat` → grad(E) ≠ 0.
- **CL3 (cold item):** an item index never used in any training-like update,
  with a fresh content row v_new (L2-normalized random), gets an activation
  vector `sim(v_new @ E, P)` that is non-constant across prototypes (variance
  bounded away from 0) for a generic random E and P — i.e. content alone yields
  a discriminative activation pattern with no per-item parameters.
- **CL4 (zero-image neutrality):** `v_i = 0` yields activation exactly 1 for
  every prototype (shifted cosine with eps guard), i.e. defined and uniform.
- **CL5a (push idempotence):** running the push twice in a row with no parameter
  updates in between leaves P unchanged after the second push.
- **CL5b (push perturbation is monotone in pre-push proximity):** the change in
  the score matrix caused by a push is small when prototypes are already near
  (high-cosine to) their argmax items, and larger when prototypes are random —
  quantitatively: mean |Δscore| under P initialized AT item representations plus
  small noise (σ=0.01) is at least 10× smaller than under random P.
- **CL5c (exact grounding):** after a push, every prototype row equals some
  row of the item-representation matrix Q exactly (row-level match).
- **CL6 (R_proto pulls prototypes to items):** one gradient-descent step on
  `R_proto` w.r.t. P (batch fixed) strictly increases, for at least the
  arg-max-matched pairs, the cosine between each prototype and its current
  best-matching batch item (check: mean of max_j cos over k increases).
- **CL7 (L_div behavior):** L_div = 0 when all pairwise prototype cosines ≤ τ;
  when two prototypes are identical (cos = 1 > τ), one gradient step on L_div
  strictly decreases their pairwise cosine; L_div's gradient w.r.t. prototypes
  whose pairwise cosines are all ≤ τ is exactly 0.
- **CL8 (L_orth counteracts collapse):** if E is rank-collapsed (e.g. all
  columns equal), a few gradient steps on L_orth alone strictly increase a rank
  proxy (e.g. smallest singular value, or effective rank via entropy of
  normalized singular values).
- **CL9 (cosine-argmax vs L2-argmin):** with UNnormalized item representations
  q_i, there exist configurations where argmax_i cos(q_i, p) and
  argmin_i ‖q_i − p‖₂ select different items (exhibit one); when all q_i have
  equal L2 norm, the two selections coincide for all prototypes (check on
  random equal-norm data).
- **CL10 (greedy unique push):** a greedy uniqueness-constrained push (process
  prototypes in order of their best cosine, assign each to its best not-yet-
  taken item) yields K distinct exemplar indices whenever M ≥ K.

## Execution notes

- Write the script(s) into `Master/temp/dc_checks/dc03/` (this directory).
- Python env: `conda activate protomf` (torch 1.9.1 CPU + numpy available);
  plain `python` from that env.
- Seconds of runtime; fixed seeds; print a per-claim verdict table at the end.
