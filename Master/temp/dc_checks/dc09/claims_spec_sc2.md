# dc09 claims spec, SC.2 round — black-box toy check (mechanism definition + claims only)

You receive ONLY this file. Implement the mechanism below from scratch in numpy/torch
(CPU, float64 unless a claim says float32), at toy dimensions, seconds of runtime, no
training pipeline. Report each claim as CONFIRMED / FALSIFIED / UNTESTABLE-AT-TOY-SCALE
with the measured numbers (max abs/rel errors, counts of cases, seeds). Design the tests
yourself; use several random seeds and shapes; include adversarial scales where sensible
(parameter scales 1e-3 … 1e3, negative entries). Write your script to
`check_claims_sc2.py` and the report to `report_sc2.md`, both next to this file.
Do not read any other file in this repository.

## Mechanism (the "merged model", two table layouts)

Notation: V words, M items, N users, embedding dim d, K_u user prototypes, K_t item
prototypes. Words are partitioned into **fields**: F fields φ = 1..F, each with its own
vocabulary subset V_φ (disjoint, ∪V_φ = all words). Two **data layouts**:

- **Fixed layout** ("hm-like"): every item has exactly one word per field (|f_i| = F,
  one word from each V_φ).
- **Bag layout** ("ml-1m-like"): fields are bags — an item has k_{i,φ} ≥ 0 words from
  field φ, with k_{i,φ} varying across items (e.g. 0–6).

Two **table layouts** (the arms):

- **`sep`** (separate tables): item table E_I ∈ R^{(V+M)×d} (rows 0..V−1 word rows,
  rows V..V+M−1 item-ID rows); user table E_U ∈ R^{(V+N)×d} (rows 0..V−1 word rows,
  SEPARATE parameters from E_I; rows V..V+N−1 user-ID rows).
- **`shared`** (one word table — the default): ONE word block E_w ∈ R^{V×d} used by both
  sides, plus an item-ID block E_{I,id} ∈ R^{M×d} and a user-ID block E_{U,id} ∈ R^{N×d}.
  Equivalent description: the `sep` model with the constraint E_I[0:V] = E_U[0:V] = E_w.

Prototypes P_u ∈ R^{K_u×d}, P_t ∈ R^{K_t×d}; projections W_u ∈ R^{K_t×d}, W_t ∈ R^{K_u×d}
(free matrices, no bias). "ids" arm = ID rows present; "noid" arm = the ID row term is
omitted on both sides.

Data: each item i has a word multiset f_i; each user u has a train basket H_u ⊂ items
(a set; |H_u| ≥ 1 unless a claim says otherwise).

Compositions (vectors in R^d; write e_f for the word row of word f on the relevant side):
- q_i = Σ_{f∈f_i} e^I_f + e_ID(i)                  (ids; noid omits the ID row)
- q_u = (1/|H_u|) Σ_{j∈H_u} Σ_{f∈f_j} e^U_f + e_ID(u)   (ids; noid omits the ID row)
- Training-time leave-one-out (LOO): when user u is scored against a positive item
  i ∈ H_u, the basket used for q_u is H_u∖{i} with mean 1/(|H_u|−1); if |H_u| = 1 the
  word part is 0. The SAME q_u^{LOO} feeds both user roles below.

Similarity: sim(a, b) = 1 + aᵀb/(‖a‖‖b‖), computed with a guarded cosine (eps 1e-8);
values in [0, 2].

Two roles per side:
- u* = [sim(q_u, P_u[l])]_l ∈ R^{K_u}      (user cosine unit)
- û  = W_u q_u ∈ R^{K_t}                   (user linear unit)
- t* = [sim(q_i, P_t[k])]_k ∈ R^{K_t}      (item cosine unit)
- t̂  = W_t q_i ∈ R^{K_u}                   (item linear unit)

Score: S(u, i) = u*ᵀt̂ + ûᵀt*. Four components: B_item(i) = 1ᵀt̂, U-half = (u*−1)ᵀt̂,
B_user(u) = 1ᵀû, I-half = ûᵀ(t*−1); S = B_item + U-half + B_user + I-half.

Inclusion regularisers (for invariance claims only — any function of the batch cosine
matrix will do): R_proto = Σ_k max_i cos(q_i, P_t[k]) and R_batch = Σ_i max_k cos(q_i,
P_t[k]) on the item side; mirror on the user side with P_u and the batch of q_u.

Losses for gradient claims: (a) sampled softmax cross-entropy over one positive and
n_neg sampled negatives per user; (b) **BCE**: −log σ(S(u,i⁺)) − Σ_neg log(1 − σ(S(u,i⁻)));
(c) **BPR**: −Σ_neg log σ(S(u,i⁺) − S(u,i⁻)). Unless a claim names a loss, use (a).

Definitions: 1 = all-ones vector of the matching length; P̂ = P with rows L2-normalised;
q̂ = q/‖q‖; mean_φ = (1/|V_φ|) Σ_{f∈V_φ} e_f (the field's vocabulary average of its word
rows); "rank-inert for item ranking" = for a fixed user, adding the term changes no
pairwise order between items.

**Canonical per-field-centred gauge** (a read-out transform, not a training change): for
every word f in field φ, ẽ_f = e_f − mean_φ; for every item ID row ẽ_ID(i) = e_ID(i) +
Σ_φ mean_φ; for every user ID row ẽ_ID(u) = e_ID(u) + Σ_φ mean_φ. Per-row read-outs are
functions of the rows, e.g. the per-word share of t̂ for word f is W_t e_f; the per-word
share of B_item is (W_tᵀ1)·e_f; the naming profile of word f against prototype k is
cos(e_f, P_t[k]).

**Cold conventions** (inference-time replacements of an entity's composition): (A) "drop":
cold item q_i = Σ_{f∈f_i} e^I_f (no ID row); (B) "mean-ID imputation": cold item
q_i = Σ_{f∈f_i} e^I_f + ē_ID where ē_ID = (1/M) Σ_m e_ID(m) over the warm item-ID rows;
user mirrors: cold user with a non-empty basket, q_u = basket mean + ē^U_ID (ē^U_ID the
mean over warm user-ID rows); empty-basket user: q_u = 0 under both conventions (the
claims test what happens if ē^U_ID is imputed there instead).

## Claims

D1 (sharing keystone — chain rule). Take a `sep` model with E_I[0:V] := E_U[0:V] := E_w
(numerically tied copies) and the `shared` model with the same E_w, ID blocks, P, W.
(a) All scores S(u, i), all four components, and both inclusion regularisers agree
bit-identically (float64 and float32). (b) For each of the three losses, the gradient of
the `shared` model with respect to E_w equals ∇_{E_I[0:V]} L + ∇_{E_U[0:V]} L of the `sep`
model at the tied point, to machine precision, row by row; the gradients with respect
to the ID blocks, P and W agree to machine precision. (c) At a generic (untied) point of
the `sep` model there is no such identity (sanity: the two word-row gradients differ).

D2 (LOO under sharing — gradient routing). In the `shared` ids model with LOO active, for
a training row (u, i⁺): (a) u* and û are computed from the same q_u^{LOO}, so an
implementation applying LOO only to the cosine role gives û' − û = W_u(q_u^{full} −
q_u^{LOO}) ≠ 0 generically; (b) a word f that occurs ONLY in the positive i⁺ (in no other
item of H_u and in no sampled negative) receives gradient through q_{i⁺} (both item
roles) and ZERO through u's own q_u^{LOO}: the row's gradient equals the `sep` model's
∇_{E_I[f]} at the tied point, and ∇_{E_U[f]} there is exactly 0 — construct such a case;
(c) if the same word also occurs in another basket item of u, the user-path gradient is
nonzero and the shared row's gradient is the sum of both paths (D1b).

D3 (centroid identity). In the `shared` model: (a) q_u^{words} = (1/|H_u|) Σ_{j∈H_u}
(q_j − e_ID(j)) exactly (ids arm; q_j the item compositions); (b) in the noid arm
q_u = (1/|H_u|) Σ_{j∈H_u} q_j exactly — the user is the mean (a convex combination) of
the purchased item vectors; (c) the LOO form: q_u^{LOO} = mean over H_u∖{i} of the same
summands. (d) In the `sep` model (a)–(c) fail generically.

D4 (gauge collapse under sharing). (a) `sep` model: the per-side rescaling (E_I, W_t) →
(c·E_I, W_t/c) with everything else fixed leaves S and all four components unchanged
exactly, for any c > 0 (and likewise (E_U, W_u)). (b) `shared` model: rescaling E_w and
E_{I,id} by c with W_t/c (W_u, E_{U,id} fixed) CHANGES S generically (û changes). (c)
`shared` model: the JOINT rescaling — E_w, E_{I,id}, E_{U,id} all by c, both W by 1/c —
leaves S and all four components unchanged exactly. (d) Under (c), the ratio
‖W_t‖_F/‖W_u‖_F is invariant, while under the `sep` per-side gauge it is not (so it is an
identified quantity under sharing and a gauge quantity under `sep`).

D5 (translation gauge of the ids arms — fixed layout). Fixed layout, ids arm, either
table layout. For arbitrary vectors v_1..v_F ∈ R^d apply: e_f ← e_f + v_{φ(f)} for every
word f (on the shared block, or on BOTH tables' word rows in `sep`), e_ID(i) ← e_ID(i) −
Σ_φ v_φ for every item, e_ID(u) ← e_ID(u) − Σ_φ v_φ for every user. (a) Every q_i and every
q_u (full basket AND every LOO basket with |H_u∖{i}| ≥ 1) is exactly unchanged; hence
every S(u,i), all four components, both inclusion regularisers and all three losses are
unchanged to machine precision. (b) The map has F·d independent directions (verify that
F independent shifts v_φ each leave everything invariant and are linearly independent as
parameter displacements). (c) Breakage: for a user with |H_u| = 1 the LOO basket is empty
(word part 0) and q_u^{LOO} = e_ID(u) − Σ_φ v_φ ≠ e_ID(u) — the LOO score changes. (d) Bag
layout: with k_{i,φ} varying across items the same map changes q_i for items with k_{i,φ}
≠ 1; an item-specific ID shift e_ID(i) ← e_ID(i) − Σ_φ k_{i,φ} v_φ restores every q_i exactly
(inference-exact with the bag-count correction), but no single per-user ID shift restores
both q_u^{full} and every q_u^{LOO} when the basket's per-field bag mass varies across
the removed positive (construct a case; show the LOO scores change).

D6 (canonical gauge makes per-row read-outs invariant). Fixed layout, ids arm. Take a
parameter set θ and a gauge-shifted θ' (D5 map with random v_φ). (a) Uncentred per-row
read-outs differ between θ and θ': per-word shares of t̂ (W_t e_f), per-word shares of
B_item ((W_tᵀ1)·e_f), the ID-row share, the per-word cosine-unit shares e_f·P_t[k]/(‖q_i‖‖P_t[k]‖),
and the naming profiles cos(e_f, P_t[k]) — all change generically. (b) After the
canonical per-field centring, θ and θ' map to the SAME rows (ẽ_f, ẽ_ID identical to
machine precision), hence every centred per-row read-out is identical. (c) Totals are
unchanged by centring: Σ_{f∈f_i} ẽ_f + ẽ_ID(i) = q_i exactly, so the four components and
S are the same whether computed from centred or uncentred rows; the sum of the centred
per-word shares plus the centred ID share equals t̂ (resp. B_item) exactly. (d)
Within-field contrasts are invariant WITHOUT centring: e_{f'} − e_f for f, f' in the same
field, and the substitution counterfactual u*ᵀW_t(e_{f'} − e_f), are identical between θ
and θ'; the removal effect u*ᵀW_t e_f is not. (e) Centring on the frequency-weighted
field mean (weights = word counts over items) is also invariant under the D5 map but
yields different centred rows than (b) — i.e. the two conventions are both invariant and
not equal.

D7 (cold conventions under the gauge). Fixed layout, ids arm, θ and gauge-shifted θ'
(D5). (a) Drop convention: the cold composition Σ_f e_f differs between θ and θ' by exactly
Σ_φ v_φ; the cold item's t*, t̂ and S(u, cold) differ; show a case where the ranking of
two cold items for a fixed user differs between θ and θ' (through the cosine unit), and
a case where the ranking of a cold item against a warm item differs. (b) Mean-ID
imputation: ē_ID shifts by exactly −Σ_φ v_φ, so the imputed cold composition is
identical between θ and θ' to machine precision, and all cold scores agree. (c) User
mirror: for a cold user with a non-empty basket, imputing ē^U_ID is invariant and the
drop is not; for an EMPTY basket, q_u = 0 is invariant while imputing ē^U_ID is NOT
(the word part has nothing to cancel).

D8 (common-mode mass in the personalised halves — shape, trend). Build users as q_u = μ +
δ_u + e_ID(u) with a common μ and per-user δ_u of scale σ. (a) The I-half term
(W_u μ)ᵀ(t*(i) − 1) and the U-half term (sim(μ, P_u) − 1)ᵀt̂(i) are per-item numbers
identical across users and rank-active (ranking by S and by S minus these terms differs
for some user on random instances). (b) Two-way decomposition on a user×item grid of
each personalised half H(u,i): item main = column mean − grand mean, user main = row mean
− grand mean, interaction = remainder. As σ → 0 the user-main and interaction shares of
the variance of each personalised half tend to 0 while the item-main share tends to 1,
and the total variance of the half across (u,i) pairs does NOT tend to 0 (M8-style
variance share stays large with zero personalisation). (c) In the noid arm with
e_ID(u) = 0 the same holds. Trend demonstration over a σ sweep, not an identity.

D9 (overlap / self-affinity channel). Define the personalised part pers(u,i) = U-half +
I-half. (a) Exact identity: with ‖q_u‖ and ‖q_i‖ treated as the fixed numbers they are at
the current parameters, pers(u,i) = q_uᵀ M_{ui} q_i with M_{ui} = A/‖q_u‖ + Bᵀ/‖q_i‖, where
A = P̂_uᵀW_t and B = P̂_tᵀW_u (both d×d); expand q_u = Σ_f w_{u,f} e^U_f + e_ID(u) and q_i =
Σ_g e^I_g + e_ID(i) and verify the pair-sum Σ_{f,g} w_{u,f} e^U_fᵀ M_{ui} e^I_g + ID terms
reproduces pers(u,i) to machine precision. (b) In the `shared` model, for a word f in both
basket and candidate, the diagonal pair term is w_{u,f} e_fᵀ M_{ui} e_f — a quadratic form
in one row: scaling e_f by c > 0 with M_{ui} and all other rows held fixed scales that term
by c²; in the `sep` model scaling only e^I_f scales the corresponding bilinear term by c.
(c) The full pers(u,i) (norms not frozen) is not polynomial in c, but for a fixed-layout
case the sign of ∂pers/∂c at c = 1 can be made positive by choosing M_{ui}-aligned rows —
a one-case demonstration that the overlap can be raised by growing a shared row.

D10 (nesting of the user cloud). `shared` noid arm, full baskets: every q_u is a convex
combination of item compositions {q_j : j ∈ H_u} (coefficients 1/|H_u|), hence lies in the
convex hull of the item cloud; verify by reconstructing the coefficients (least squares
on the basket's items gives exactly 1/|H_u| each, residual ≈ 0). In the ids arm the
statement holds for q_u − e_ID(u) with the item compositions minus their ID rows.

D11 (B_user is rank-inert but loss-active under BCE only). For a fixed user u, add a
constant b to S(u, i) for all items i (equivalently shift B_user). (a) The sampled-softmax
loss and the BPR loss are exactly unchanged (shift-invariant within the user), and their
gradients with respect to the direction w_u = W_uᵀ1 in parameter space — i.e. ∂L/∂b at
b = 0 — are exactly 0 to machine precision. (b) The BCE loss changes with b and ∂L_BCE/∂b
≠ 0 generically. (c) Item ranking for the user is unchanged under any b for all three
losses' scores (rank-inert).
