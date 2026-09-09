# dc08 — claims spec for an independent black-box check

You are given ONLY this specification. Do not look for other project documents;
do not ask what the "intended" answer is. Design and run small numpy/torch toy
scripts (toy dimensions, seconds of runtime, no training pipeline) that test each
claim, and report per claim: **confirmed / falsified / untestable-at-toy-scale**,
with the numbers you measured.

Write your scripts into `Master/temp/dc_checks/dc08/` (repo root is the working
directory). Use `conda activate protomf` if you need the project env; plain
numpy/torch in any available python is equally fine.

## Mechanism definition

Entities: N users, M items, a vocabulary of V "word" rows, embedding dim d,
K prototypes.

One embedding table `E` with V + M + N rows of width d:
- rows `0 .. V-1`      : word rows `e_f`
- rows `V .. V+M-1`    : item identity rows `y_j`  (row `V + j` belongs to item j)
- rows `V+M .. V+M+N-1`: user identity rows `e_ID(u)` (row `V+M+u`)

Each user u has a train purchase set `H_u ⊆ {0..M-1}`, `|H_u| ≥ 1`. Each item j
has an attribute word bag `f_j ⊆ {0..V-1}` (a multiset is allowed; treat weights
as counts).

**User composition:**

    q_u = (1/|H_u|) * Σ_{j∈H_u} ( Σ_{f∈f_j} e_f  +  λ_y * y_j )   +  [use_id] * e_ID(u)

with λ_y a scalar ≥ 0 and `use_id` a boolean.

**Implementation form to test (this is how it is actually computed):** a padded
bag per user, `ids (N, D)` and `weights (N, D)`, where for user u the real slots
are `(f, n_{u,f}/|H_u|)` for every word f occurring n_{u,f} times across H_u, and
`(V+j, λ_y/|H_u|)` for every j ∈ H_u; padding slots are `(id=0, weight=0.0)`.
Then `q_u = Σ_s w_s · E[ids_s]` plus the optional user row.

**Leave-one-out (training only).** When scoring user u against a positive item
i ∈ H_u, the composition must exclude item i entirely (its words AND its identity
row). The implementation does this algebraically, WITHOUT rebuilding the bag:

    q_meta ← ( |H_u| * q_meta  −  e_pos ) / (|H_u| − 1),   e_pos = Σ_{f∈f_i} e_f + λ_y * y_i

applied to the pooled metadata part BEFORE the user identity row is added, and
defined as exactly zero when |H_u| = 1.

**Scoring head (unchanged host).** With K prototypes `P ∈ R^{K×d}` and a free
per-item vector `t_i ∈ R^K`:

    u*_l = 1 + cos(q_u, P_l),   l = 1..K
    S(u,i) = Σ_l t_{i,l} · u*_l

## Claims to check

**C1 — Bag form equals the mathematical form.** The padded-bag computation of
q_u equals the definition `q_u = (1/|H_u|) Σ_{j∈H_u}(Σ_{f∈f_j} e_f + λ_y y_j)
+ [use_id] e_ID(u)` to floating-point tolerance, for random layouts including
users with repeated words across items and users with |H_u| = 1.

**C2 — Exact reduction to the no-identity model.** At λ_y = 0 the computed q_u
is bit-identical (or ≤1e-7) to the same computation with the identity slots
removed from the bag entirely. State whether identical or merely close.

**C3 — The algebraic leave-one-out is exact.** For random users with |H_u| ≥ 2
and a random positive i ∈ H_u, the algebraic update above yields exactly (to
tolerance) the metadata part recomputed from scratch over `H_u \ {i}`, i.e.
`(1/(|H_u|−1)) Σ_{j∈H_u\{i}} (Σ_{f∈f_j} e_f + λ_y y_j)`. Test at λ_y ∈ {0, 1}
and report whether the identity rows are removed correctly as well as the words.
Also check the |H_u| = 1 case gives exactly zero metadata mass.

**C4 — Exact additive decomposition of the activation.** Define
`cos_l = cos(q_u, P_l)`. Claim: for every user and prototype,

    cos_l  =  Σ_s w_s ⟨E[ids_s], P_l⟩ / (‖q_u‖ ‖P_l‖)  +  [use_id] ⟨e_ID(u), P_l⟩/(‖q_u‖‖P_l‖)

i.e. the per-slot shares (words and identity rows alike) sum exactly to the
activation. Report max absolute error over a random battery.

**C5 — Two exact regroupings of one sum.** The same per-slot shares can be
grouped (a) by word / identity-row / user-row, and (b) by purchased item j
(each item contributing its words' shares plus its own identity-row share).
Claim: both groupings sum to the same `cos_l`, and grouping (b)'s per-item
totals sum to `cos_l` minus the user-row share. Report max error.

**C6 — Score decomposition.** `S(u,i) = Σ_l t_{i,l} + Σ_l t_{i,l} cos_l`, i.e.
the "baseline" term `B(t_i) = Σ_l t_{i,l}` is exactly separable and does not
depend on the user. Report max error and confirm B does not change when q_u
changes.

**C7 — Scale invariance of the composition through the cosine.** Multiplying the
whole composed vector q_u by any positive scalar leaves every `u*_l` and hence
S(u,i) unchanged. Consequence to check separately: the *pooling exponent* is
therefore inert **only if it scales q_u uniformly** — verify that replacing the
mean 1/|H_u| by 1/|H_u|^α changes nothing when `use_id = False`, but DOES change
q_u's direction when `use_id = True` (because the user row is added after
pooling and is not rescaled). Report both.

**C8 — Gradient flow to identity rows.** With autograd, a loss on S(u,i)
produces non-zero gradient on `y_j` for every j ∈ H_u (excluding a
leave-one-out-removed positive, whose gradient must be exactly zero when LOO is
applied). Report which rows receive gradient.

**C9 — Identity/user-row non-identifiability (degenerate case).** Consider a
user u whose purchase set is disjoint from every other user's (items unique to
them). Claim: there is an exact one-parameter family of parameter changes that
leaves ALL users' compositions unchanged — specifically, adding a vector δ to
e_ID(u) while subtracting `(|H_u|/λ_y)·δ / |H_u|` appropriately from that user's
identity rows (work out the exact form yourself and state it) keeps q_u fixed.
Construct it numerically and report the residual. If the family does not exist
as described, say so and characterise what does hold.

**C10 — Shared-component sensitivity of the identity block.** If a constant
vector c is added to EVERY identity row y_j, the composed q_u changes by exactly
`λ_y · c` for every user (independent of |H_u|). Verify, and report whether this
changes the ranking of items for a fixed user (i.e. whether it is rank-inert).

**C11 — Degenerate empty-history case.** A user with |H_u| = 0 (no train
purchases): with `use_id = True`, q_u = e_ID(u); with `use_id = False`, q_u = 0
and the claim is that `u*_l = 1` exactly for all l (torch's cosine of a zero
vector), so `S(u,i) = Σ_l t_{i,l}` exactly. Verify the literal numerical
behaviour of whatever cosine implementation you use, and report it.
