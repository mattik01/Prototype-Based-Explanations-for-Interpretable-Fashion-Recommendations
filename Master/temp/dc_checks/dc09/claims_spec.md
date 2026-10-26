# dc09 claims spec — black-box toy check (mechanism definition + claims only)

You receive ONLY this file. Implement the mechanism below from scratch in numpy/torch
(CPU, float64 unless a claim says float32), at toy dimensions, seconds of runtime, no
training pipeline. Report each claim as CONFIRMED / FALSIFIED / UNTESTABLE-AT-TOY-SCALE
with the measured numbers (max abs/rel errors, counts of cases, seeds). Design the tests
yourself; use several random seeds and shapes; include adversarial scales where sensible.

## Mechanism (the "merged model")

Notation: V words (shared vocabulary index), M items, N users, embedding dim d, K_u user
prototypes, K_t item prototypes.

Parameters:
- Item table E_I ∈ R^{(V+M)×d}: rows 0..V−1 are word rows, rows V..V+M−1 are item-ID rows.
- User table E_U ∈ R^{(V+N)×d}: rows 0..V−1 are word rows (same vocabulary index, SEPARATE
  parameters from E_I), rows V..V+N−1 are user-ID rows.
- Prototypes P_u ∈ R^{K_u×d}, P_t ∈ R^{K_t×d}.
- Projections W_u ∈ R^{K_t×d}, W_t ∈ R^{K_u×d} (free matrices, no bias).

Data: each item i has a word set f_i ⊂ {0..V−1} (fixed size F, or variable sizes); each
user u has a train basket H_u ⊂ items (a set, |H_u| ≥ 1 unless a claim says otherwise).

Compositions (vectors in R^d):
- q_i = Σ_{f∈f_i} E_I[f] + E_I[V+i]            ("ids" arm; the "noid" arm omits the ID row)
- q_u = (1/|H_u|) Σ_{j∈H_u} Σ_{f∈f_j} E_U[f] + E_U[V+u]   ("ids" arm; "noid" omits the ID row)
- Training-time leave-one-out (LOO): when user u is scored against a positive item i ∈ H_u,
  the basket used for q_u is H_u∖{i} with mean 1/(|H_u|−1); if |H_u| = 1 the word part is 0.

Similarity: sim(a, b) = 1 + aᵀb / (‖a‖‖b‖), computed with torch.nn.functional.cosine_similarity
(eps guard 1e-8) or an equivalent guarded cosine; values in [0, 2].

Two roles per side:
- u* = [sim(q_u, P_u[l])]_l ∈ R^{K_u}      (user cosine unit)
- û  = W_u q_u ∈ R^{K_t}                   (user linear unit)
- t* = [sim(q_i, P_t[k])]_k ∈ R^{K_t}      (item cosine unit)
- t̂  = W_t q_i ∈ R^{K_u}                   (item linear unit)

Score: S(u, i) = u*ᵀ t̂ + ûᵀ t*.

Host ("UI-ProtoMF"): the same with free per-entity tables — q_i = T[i] (T ∈ R^{M×d}),
q_u = U[u] (U ∈ R^{N×d}) — implemented the host way: the base vector is looked up
TWICE per step (once for the cosine unit, once for the linear unit; both lookups read the
same table). The merged model computes q once per side and feeds both units from it.

Loss for gradient claims: any differentiable loss of the scores, e.g. sampled softmax
cross-entropy over one positive and n_neg sampled negatives per user; unless a claim
names a loss, use that one.

Normalisation control ("normalised model"): q ← q / ‖q‖ (guarded) before BOTH units on a
side, everything else unchanged. "One-sided block normalisation": q ← q/‖q‖ before the
linear unit only, the cosine unit receives the raw q.

Definitions used by claims: 1 = all-ones vector of the matching length; P̂ = P with rows
L2-normalised; q̂ = q/‖q‖; "rank-inert for item ranking" = for a fixed user, adding the
term changes no pairwise order between items.

## Claims

C1 (four-component decomposition; intercept kinds). Exactly: S = 1ᵀt̂ + Σ_l (u*_l − 1)·t̂_l
+ 1ᵀû + Σ_k û_k·(t*_k − 1). For a fixed user, the term 1ᵀû is identical across items and
is rank-inert for item ranking; the term 1ᵀt̂ varies across items and in general changes
item rankings when removed (i.e. ranking by S and ranking by S − 1ᵀt̂ differ for some
user on random instances).

C2 (exact linear-half attribution, both sides). (a) t̂ = Σ_{f∈f_i} W_t E_I[f] + W_t E_I[V+i]
exactly, and removing word row f from q_i (holding q_u fixed) changes the score by exactly
−u*ᵀ W_t E_I[f] — the per-word share equals the removal effect on the u*ᵀt̂ half.
(b) û = (1/|H_u|) Σ_{j∈H_u} Σ_{f∈f_j} W_u E_U[f] + W_u E_U[V+u] exactly, so per-purchase
shares s_j = (1/|H_u|) W_u Σ_{f∈f_j} E_U[f] sum to the word part of û; the counterfactual
effect on û of REMOVING purchase j (re-meaning over |H_u|−1) is NOT in general equal to
−s_j; it equals −s_j exactly only when Σ_{f∈f_j} E_U[f] equals the basket's mean word sum.

C3 (cosine-unit exact shares survive inside the merged score). With joint normalisation
by ‖q‖: t*_k − 1 = Σ_{r∈rows(i)} E_I[r]·P_t[k] / (‖q_i‖‖P_t[k]‖) exactly (rows = words + ID
row), and u*_l − 1 = Σ_r w_r E_U[r]·P_u[l] / (‖q_u‖‖P_u[l]‖) exactly with w_r the basket
weights (count/|H_u| for words, 1 for the ID row); the per-purchase regrouping of the
user shares (sum word shares within each purchase, weight 1/|H_u|) equals the per-word
grouping exactly.

C4 (keystone reduction to the host). With V = 0 words and ID rows on both sides, setting
E_I[V+i] = T[i] and E_U[V+u] = U[u] (same P_u, P_t, W_u, W_t): (a) the merged model's
scores are bit-identical to the host's (float32 and float64) for all (u, i), with the
merged model computing q once and the host looking it up twice; (b) the gradients of a
sampled-softmax loss with respect to every parameter agree to machine precision between
the two implementations (float64), including the entity tables that the host touches
through two lookups.

C5 (single-side reductions). With only the ITEM side reduced (V_items = 0, item ID on),
the merged model equals a model whose item side is a free table T = E_I[V..] and whose
user side is the composed q_u (scores bit-identical); and symmetrically for the user side.

C6 (gauge freedom lives in W). Let A_u = [1ᵀ; P̂_u] ∈ R^{(d+1)×K_u}. For any Δ ∈ R^{K_u×d}
whose columns lie in the null space of A_u (A_u Δ = 0): (a) W_t → W_t + Δ leaves S(u, i)
unchanged for every (u, i), exactly; (b) the gradient of any loss with respect to W_t is
orthogonal to every such Δ (Frobenius inner product ⟨∇W_t, Δ⟩ = 0 to machine precision);
(c) under plain SGD (no weight decay) the component of W_t in that null space stays at its
initial value to machine precision over T steps while the in-row-space component moves;
(d) with L2 weight decay implemented as grad += wd·W, that component multiplies by exactly
(1 − lr·wd) per step. (e) The null space has dimension K_u − d − 1 whenever K_u > d + 1
(generic P_u). Mirror statements hold for W_u with A_t = [1ᵀ; P̂_t].

C7 (the live intercept is attributable and not gauge). 1ᵀt̂ = Σ_{f∈f_i} (W_tᵀ1)·E_I[f] +
(W_tᵀ1)·E_I[V+i] exactly, and 1ᵀt̂ is invariant under every gauge shift Δ of C6.

C8 (leave-one-out reaches both roles). In the merged model with LOO active, u* and û are
computed from the SAME vector q_u^{LOO}; an implementation that applies LOO to the cosine
unit only yields û' = W_u q_u^{full} with û' − û = W_u (q_u^{full} − q_u^{LOO}) ≠ 0
generically (nonzero whenever the positive's word sum differs from the basket mean).

C9 (normalisation controls). (a) In the normalised merged model, u*, t* are unchanged
(to float tolerance) relative to the unnormalised model with the same parameters, while
û, t̂ and S change. (b) In a model WITHOUT linear units (score u*ᵀt with a free t, or
uᵀt* with a free u), normalising q before the cosine unit leaves the scores AND the
gradients with respect to all parameters unchanged to machine precision (exact no-op).
(c) One-sided block normalisation (linear unit only) leaves u*, t* unchanged and makes
the linear unit's output scale-free in ‖q‖ (û(c·q) = û(q) for any c > 0).

C10 (twin separation: cosine squashed, linear not). For two items with identical word
sets (shared word mass m = Σ_f E_I[f]) and distinct ID rows e_i, e_j: as ‖m‖ grows with
e_i, e_j fixed, the cosine-unit difference ‖t*(i) − t*(j)‖ tends to 0 (decreasing in ‖m‖
over a sweep), whereas the linear-unit difference t̂_i − t̂_j = W_t(e_i − e_j) is exactly
independent of m.

C11 (degenerate paths). (a) Cold user: empty basket and no user-ID row → q_u = 0 → under
the guarded cosine u* ≡ 1 (all entries exactly 1), û = 0, and S(u, i) = 1ᵀt̂_i exactly for
every item. (b) Cold item: item-ID row dropped, words present → q_i = Σ_f E_I[f] is
generically nonzero, t* is generically non-constant across prototypes, t̂ generically
nonzero — both units carry item-dependent signal; two cold items with identical word
sets get identical t* and t̂.

C12 (bag-size channel; statistical trend). With word rows of equal norm and random
directions, the expected norm of the basket mean (1/n) Σ_{j≤n} Σ_{f∈f_j} E_U[f] decreases
with n (basket size) — demonstrate over a sweep of n with many random baskets — while
u* is invariant to any positive rescaling of q_u and ‖û‖ scales linearly with ‖q_u‖ at
fixed direction. (Trend demonstration, not an identity.)

C13 (collapsed function class). Exactly: S = (W_tᵀ1)·q_i + q̂_uᵀ (P̂_uᵀ W_t) q_i +
(W_uᵀ1)·q_u + q̂_iᵀ (P̂_tᵀ W_u) q_u, with P̂ row-normalised prototypes and q̂ unit vectors.

C14 (constraint candidate K10, Procrustes alignment penalty). L_align = ‖E_U^w − R E_I^w‖²_F
over the V word rows, with R the orthogonal Procrustes minimiser computed from the
current tables and treated as a constant: (a) L_align = 0 iff E_U^w = R E_I^w for some
orthogonal R; (b) its gradient is nonzero for both E_U^w and E_I^w whenever L_align > 0;
(c) scaling one table by c ≠ 1 makes L_align > 0 even when the two tables are otherwise
identical (the penalty is not scale-invariant).
