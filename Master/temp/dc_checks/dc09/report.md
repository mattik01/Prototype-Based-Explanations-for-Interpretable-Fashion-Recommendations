# dc09 claims check -- independent black-box toy verification (C1-C14)

- Script: `/Users/glaes/Desktop/ProtoMF-fufi/Master/temp/dc_checks/dc09/check_claims.py` (mechanism implemented from `claims_spec.md` alone; no other repository file read)
- Run command (from this directory): `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate protomf && python check_claims.py`
- Environment: torch 2.5.1, numpy 2.4.2, CPU, float64 default (float32 where C4 names it)
- Toy sizes: 4 shape configs (V 5-8, M 8-10, N 6-8, d 3-5, K_u 4-12, K_t 5-9; K_u<=d+1 and K_t=d+1 included), 6 seeds, 5 parameter scale sets (1e-3 ... 1e3, mixed), basket sizes 1-6 with forced |H|=1 users, fixed and variable word-set sizes
- Runtime: 3.1 s

## Verdict summary

| Claim | Verdict |
|---|---|
| C1 | CONFIRMED |
| C2 | FALSIFIED (final clause of (b) only; all identities and the 'not in general equal' statement CONFIRMED) |
| C3 | CONFIRMED |
| C4 | CONFIRMED |
| C5 | CONFIRMED |
| C6 | CONFIRMED |
| C7 | CONFIRMED |
| C8 | CONFIRMED |
| C9 | CONFIRMED |
| C10 | CONFIRMED (tends to 0; linear-unit difference exactly independent of m); the parenthetical 'decreasing in ||m|| over a sweep' FALSIFIED as a global statement -- non-monotone at small ||m|| in 6/8 seeds, monotone once ||m|| >= ID-row scale |
| C11 | CONFIRMED (S = 1^T t_hat to machine precision, 1e-16 relative; not bitwise through the matmul path) |
| C12 | CONFIRMED |
| C13 | CONFIRMED |
| C14 | CONFIRMED for c > 0 (a, b, c); literal (c) wording FALSIFIED at c = -1 (sign flip is orthogonal) |

## Interpretations made (spec ambiguities resolved by the verifier)

1. Sampled-softmax training step: one positive per user (drawn from H_u), n_neg = 4 uniform negatives, the LOO user
   vector q_u^{LOO}(positive) scores the positive AND its negatives; loss = mean cross-entropy, target index 0.
2. C2(a) "changes the score by exactly -u*^T W_t E_I[f]" is read, per its own dash clause, as the change of the
   u*^T t_hat half (q_u fixed, u* fixed). The change of the full S is also reported (it differs, since t* moves).
3. C5 "V_items = 0": the item table has no word rows (E_I in R^{M x d}); the users still compose from the items'
   word sets through E_U's V word rows (and symmetrically for V_users = 0). The comparison model looks the free table
   up twice (host style) and composes the other side once.
4. C6 A_u = [1^T ; P_hat_u] in R^{(d+1) x K_u} is read as [1^T ; P_hat_u^T] (P_hat_u is K_u x d). For (c)/(d) the
   null space is the one of A_u at step 0, so P_u is frozen while W_t, W_u, P_t, E_I, E_U train (if P_u also trains
   the null space rotates; that drift is reported separately as informative).
5. C9(b) "free t" / "free u": a free per-item table in R^{M x K_u} / per-user table in R^{N x K_t}; both the
   user-cosine and the item-cosine variants are tested, with the sampled-softmax loss and LOO.
6. C10: twin items share word set {0,1,2} (F = 3) with m = c * m0 swept over c in [1e-3, 1e3]; e_i, e_j fixed.
7. C11(a): the cold user is user 0 with an emptied basket in the noid-user arm (q_u = 0 exactly by construction).
8. C14: R in O(d) (reflections allowed, as the SVD Procrustes minimiser U V^T yields) acting on the embedding
   dimension row-wise: L = sum_f ||E_U[f] - R E_I[f]||^2 = ||E_U - E_I R^T||_F^2.
9. "Machine precision" verdict threshold: max relative error < 1e-11 (float64); "float tolerance" for cosine
   invariances: < 1e-9 abs; bit identity: torch.equal. Scale sets mix 1e-3 and 1e3 across tables, randn gives negatives.
10. Invariance claims (C6a, C7) are measured against the rounding floor of the invariant quantity (the magnitude of the
    terms being summed, e.g. sum_k |u_hat_k||t*_k|), because with ||Delta|| = 1e-6 ||W|| the only observable change is
    S's own float64 rounding; the deviation relative to the would-be change ||Delta|| ||q|| is reported alongside.
11. Ranking comparisons (C1) ignore pairs tied to within 1e-9 max|S| (twin items with identical word sets in the noid
    arm have scores that differ only at the ULP level and can tie after a constant shift).
12. C12 uses a paired design (nested baskets of one random item order per trial, 20000 trials) so that the trend is
    not masked by Monte-Carlo noise; the claim is statistical, so sample means and their standard errors are reported.

## Per-claim measurements

## C1 -- four-component decomposition; intercept kinds
- instances: 360 (4 configs x 6 seeds x 5 scale sets x 3 id-arms)
- decomposition S = 1^T t_hat + sum_l (u*_l-1) t_hat_l + 1^T u_hat + sum_k u_hat_k (t*_k-1): max rel err 5.342e-16
- 1^T u_hat rank-inert: pairwise item-order mismatches between S and S - 1^T u_hat = 0 (over all users/instances; 745 exactly/near-tied pairs -- twin items with identical word sets in the noid arm, |dS| <= 1e-9 max|S| -- excluded from all order comparisons)
- 1^T t_hat varies across items: min over instances of (max_i - min_i) 1^T t_hat_i / max_i |1^T t_hat_i| = 6.838e-01
- removing 1^T t_hat changes item rankings: instances with >=1 user whose ranking changes = 216/360; users with changed ranking per instance min/max = 0/8; total flipped pairs = 17945
- **Verdict: CONFIRMED**

## C2 -- exact linear-half attribution, both sides
- (a) t_hat = sum_f W_t E_I[f] + W_t E_I[V+i]: max rel err 3.684e-16
- (a) removal of word f from q_i: change of the u*^T t_hat half vs -u*^T W_t E_I[f]: max rel err 1.256e-14 over 1960 removals x all users
- (a) side note: change of the FULL score S vs -u*^T W_t E_I[f]: relative deviation min/max = 1.351e-10/3.674e+08 (the cosine half also moves, so the per-word share is exact for the linear half only)
- (b) u_hat = (1/|H|) sum_j sum_f W_u E_U[f] + W_u E_U[V+u]: max rel err 5.031e-16
- (b) per-purchase shares s_j sum to the word part of u_hat: max rel err 5.691e-16
- (b) counterfactual removal of purchase j (re-mean over |H|-1) vs -s_j: ||delta + s_j||/||s_j|| min/max = 1.200e-01/4.843e+00 over 1960 (u,j) pairs -> NOT equal in general (confirmed)
- (b) derived closed form delta = (a_bar - a_j)/(|H|-1) with a_j = W_u sum_f E_U[f], a_bar = basket mean: max rel err 7.825e-16
- (b) equality condition test, basket {A,B,C} with single disjoint words, word sums (m-dl, m, m+dl) so a_B == basket mean word sum:
    ||delta_B|| = 6.206e-17   ||s_B|| = 4.023e-02   ||delta_B + s_B|| = 4.023e-02   | alt. construction (others cancel, a_B != mean by 3.136e-02): ||delta_B + s_B|| = 1.289e-16
    ||delta_B|| = 0.000e+00   ||s_B|| = 1.870e+00   ||delta_B + s_B|| = 1.870e+00   | alt. construction (others cancel, a_B != mean by 1.393e+00): ||delta_B + s_B|| = 1.579e-15
    ||delta_B|| = 0.000e+00   ||s_B|| = 1.725e+01   ||delta_B + s_B|| = 1.725e+01   | alt. construction (others cancel, a_B != mean by 1.407e+01): ||delta_B + s_B|| = 2.275e-15
    ||delta_B|| = 5.684e-14   ||s_B|| = 1.699e+02   ||delta_B + s_B|| = 1.699e+02   | alt. construction (others cancel, a_B != mean by 1.323e+02): ||delta_B + s_B|| = 2.562e-14
    -> when a_j equals the basket mean, removing j leaves the mean unchanged (delta = 0) and delta != -s_j; equality delta = -s_j holds instead iff W_u a_j = W_u * (basket TOTAL word sum), i.e. the other purchases cancel.
- **Verdict: FALSIFIED (final clause of (b) only; all identities and the 'not in general equal' statement CONFIRMED)**

## C3 -- cosine-unit exact shares inside the merged score
- t*_k - 1 = sum_r E_I[r].P_t[k]/(||q_i|| ||P_t[k]||) (rows = words + ID): max abs err 6.661e-16
- u*_l - 1 = sum_r w_r E_U[r].P_u[l]/(||q_u|| ||P_u[l]||), w = count/|H| (words), 1 (ID): max abs err 1.110e-15 (max word count within a basket = 4)
- per-purchase regrouping == per-word grouping: max rel err 3.357e-16
- **Verdict: CONFIRMED**

## C4 -- keystone reduction to the host (V = 0, ID rows both sides)
- (a) scores bit-identical (float64): torch.equal on all 120 instances = True; max abs diff = 0.000e+00
- (a) scores bit-identical (float32): torch.equal on all 120 instances = True; max abs diff = 0.000e+00
- (b) sampled-softmax loss values torch.equal (float64): True
- (b) grad max rel err E_I~T: 0.000e+00
- (b) grad max rel err E_U~U: 0.000e+00
- (b) grad max rel err P_u~P_u: 0.000e+00
- (b) grad max rel err P_t~P_t: 0.000e+00
- (b) grad max rel err W_u~W_u: 0.000e+00
- (b) grad max rel err W_t~W_t: 0.000e+00
- **Verdict: CONFIRMED**

## C5 -- single-side reductions
- item side reduced (V_items=0, ID on) vs free item table looked up twice + composed users: torch.equal on all 120 instances = True; max abs diff 0.000e+00
- user side reduced (V_users=0, ID on) vs free user table looked up twice + composed items: torch.equal on all 120 instances = True; max abs diff 0.000e+00
- **Verdict: CONFIRMED**

## C6 -- gauge freedom lives in W
- (a) S invariant under W_t -> W_t + Delta (A_u Delta = 0), Delta scales 1e-3/1/1e3 (incl. ||Delta|| = 1e-6 ||W_t|| and 1e6 ||W_t||): max |dS| / rounding floor of S (max_(u,i) sum_l |u*_l||t_hat_l| + sum_k |u_hat_k||t*_k|) = 4.716e-16; max |dS| / (||Delta||_F max||q_i|| sqrt(K_u) 2) = 1.420e-13 (this second ratio is only informative when ||Delta|| >> ||W||; for ||Delta|| << ||W|| the floor is S's own rounding)  [540 gauge shifts incl. mirror]
- (b) <grad_W_t L, Delta> / (||grad|| ||Delta||): max 2.076e-16
- (a,b) mirror W_u with A_t: max |dS| / floor 3.986e-16 (vs would-be change 1.586e-10); |<grad_W_u, Delta>| normalised max 2.865e-16
- (e) nullity of A = [1^T; Phat^T] measured vs K-d-1 (0 if K<=d+1), all seeds/scales agree = True; (K, d, measured, expected): [(4, 4, 0, 0), (5, 4, 0, 0), (6, 3, 2, 2), (7, 3, 3, 3), (8, 3, 4, 4), (9, 5, 3, 3), (10, 3, 6, 6), (12, 5, 6, 6)]
- (c) plain SGD, 40 steps, lr=0.05, P_u frozen (A_u fixed), all other params trained, 2 configs x 4 seeds: null-space coords of W_t max rel drift 6.329e-16; row-space coords min rel movement 2.769e-01
- (c) informative: with P_u ALSO trained the step-0 null space rotates; drift of step-0 null coords = 4.827e-02 (claim presumes a fixed A_u)
- (d) weight decay wd=0.1 (grad += wd*W): per-step null coords vs (1-lr*wd) x previous: max rel err 6.530e-16; after 40 steps vs (1-lr*wd)^T x initial: max rel err 7.059e-16
- **Verdict: CONFIRMED** (interpretation for (c)/(d): A_u fixed, i.e. P_u frozen while W_t and all other tables train)

## C7 -- live intercept attributable and not gauge
- 1^T t_hat = sum_f (W_t^T 1).E_I[f] + (W_t^T 1).E_I[V+i]: max rel err 1.388e-15
- 1^T t_hat invariant under gauge shifts Delta (A_u Delta=0), 270 shifts (Delta scales 1e-3/1/1e3): max |change| / rounding floor (max_i sum_l |t_hat_l|) = 2.764e-16; max |change| / (||Delta|| max||q_i|| sqrt(K_u)) = 1.977e-13
- **Verdict: CONFIRMED**

## C8 -- leave-one-out reaches both roles
- u_hat' - u_hat = W_u (q_full - q_loo): max rel err 5.507e-16 over 2155 (u, positive) pairs
- ||u_hat' - u_hat|| / ||u_hat||, |H|>=2 and positive's word sum != basket mean: min 2.622e-02, max 3.357e+00 (generically nonzero)
- pairs where the positive's word sum == basket mean (baskets of twin items): 20, max ||u_hat' - u_hat||/||u_hat|| there = 0.000e+00 (zero as predicted)
- same for |H|=1 (LOO word part = 0): min 5.189e-01
- control: u* also differs between q_loo and q_full (min over pairs of max abs diff 0.000e+00) -- both roles see the same q_loo in the merged model
- constructed: positive B with word sum == basket mean -> ||u_hat' - u_hat|| = 0.000e+00; positive A (!= mean) -> 1.226e+00
- **Verdict: CONFIRMED**

## C9 -- normalisation controls
- (a) normalised model: u*, t* max abs change 4.441e-16; u_hat/t_hat min rel change 4.571e-01; S min rel change 4.571e-01
- (b) no-linear-unit models (u*^T t free t; u^T t* free u), q normalised before the cosine unit: scores max rel err 4.160e-16; grads (E tables, P, free table) max rel err 1.029e-15
- (c) one-sided block normalisation: u*, t* max abs change 0.000e+00 (bit-identical path); u_hat(c q) vs u_hat(q), c in {1e-3,0.37,1,2,1e3}: max rel err 4.608e-16
- **Verdict: CONFIRMED**

## C10 -- twin separation: cosine squashed, linear not
- sweep of word-mass scale c (m = c*m0, ||m|| = c||m0||): ['1e-03', '3e-03', '1e-02', '3e-02', '1e-01', '3e-01', '1e+00', '3e+00', '1e+01', '3e+01', '1e+02', '3e+02', '1e+03']
  seed 0: ||t*(i)-t*(j)|| = 1.76e+00 1.77e+00 1.79e+00 1.84e+00 1.97e+00 2.13e+00 9.89e-01 2.12e-01 6.06e-02 1.86e-02 5.84e-03 1.84e-03 5.82e-04  | increases: whole sweep 5, c>=1 only 0 | last/first = 3.30e-04 | log-log slope over c in [10,1000] = -1.009 | ||e_i||/||m0|| = 0.75
  seed 1: ||t*(i)-t*(j)|| = 2.13e+00 2.13e+00 2.13e+00 2.11e+00 2.04e+00 1.73e+00 1.01e+00 3.89e-01 1.28e-01 4.10e-02 1.30e-02 4.11e-03 1.30e-03  | increases: whole sweep 0, c>=1 only 0 | last/first = 6.10e-04 | log-log slope over c in [10,1000] = -0.997 | ||e_i||/||m0|| = 0.81
  seed 2: ||t*(i)-t*(j)|| = 2.65e+00 2.65e+00 2.64e+00 2.58e+00 2.27e+00 1.22e+00 4.42e-01 1.52e-01 4.99e-02 1.60e-02 5.10e-03 1.62e-03 5.11e-04  | increases: whole sweep 0, c>=1 only 0 | last/first = 1.93e-04 | log-log slope over c in [10,1000] = -0.995 | ||e_i||/||m0|| = 0.69
  seed 3: ||t*(i)-t*(j)|| = 1.36e+00 1.36e+00 1.36e+00 1.37e+00 1.33e+00 1.02e+00 4.59e-01 1.53e-01 4.86e-02 1.53e-02 4.85e-03 1.53e-03 4.85e-04  | increases: whole sweep 3, c>=1 only 0 | last/first = 3.56e-04 | log-log slope over c in [10,1000] = -1.000 | ||e_i||/||m0|| = 0.39
  seed 4: ||t*(i)-t*(j)|| = 2.64e+00 2.64e+00 2.65e+00 2.66e+00 2.68e+00 2.57e+00 1.60e+00 5.65e-01 1.78e-01 5.58e-02 1.76e-02 5.57e-03 1.76e-03  | increases: whole sweep 4, c>=1 only 0 | last/first = 6.66e-04 | log-log slope over c in [10,1000] = -1.002 | ||e_i||/||m0|| = 0.68
  seed 5: ||t*(i)-t*(j)|| = 1.61e+00 1.61e+00 1.62e+00 1.63e+00 1.67e+00 1.70e+00 1.19e+00 4.19e-01 1.29e-01 4.01e-02 1.26e-02 3.99e-03 1.26e-03  | increases: whole sweep 5, c>=1 only 0 | last/first = 7.81e-04 | log-log slope over c in [10,1000] = -1.005 | ||e_i||/||m0|| = 0.80
  seed 6: ||t*(i)-t*(j)|| = 1.80e+00 1.80e+00 1.80e+00 1.81e+00 1.80e+00 1.59e+00 8.28e-01 2.81e-01 8.89e-02 2.81e-02 8.87e-03 2.80e-03 8.86e-04  | increases: whole sweep 3, c>=1 only 0 | last/first = 4.92e-04 | log-log slope over c in [10,1000] = -1.001 | ||e_i||/||m0|| = 0.34
  seed 7: ||t*(i)-t*(j)|| = 9.68e-01 9.69e-01 9.72e-01 9.84e-01 1.02e+00 1.12e+00 9.09e-01 3.09e-01 9.24e-02 2.85e-02 8.94e-03 2.82e-03 8.91e-04  | increases: whole sweep 5, c>=1 only 0 | last/first = 9.21e-04 | log-log slope over c in [10,1000] = -1.008 | ||e_i||/||m0|| = 0.57
- cosine-unit difference tends to 0 in all 8 seeds: True (tail slope ~ -1, i.e. O(1/||m||))
- strictly decreasing over the WHOLE sweep in all seeds: False (non-monotone in 6/8 seeds while ||m|| < ||e||: the difference first grows slightly before the word mass dominates); decreasing for all c >= 1 (||m|| >= ||m0||, i.e. word mass at or above the ID-row scale) in all seeds: True
- linear-unit difference t_hat_i - t_hat_j vs W_t(e_i - e_j) across the sweep: max abs err 2.076e-12, max rel err 2.106e-13 (float64 rounding at c=1e3)
- **Verdict: CONFIRMED (tends to 0; linear-unit difference exactly independent of m); the parenthetical 'decreasing in ||m|| over a sweep' FALSIFIED as a global statement -- non-monotone at small ||m|| in 6/8 seeds, monotone once ||m|| >= ID-row scale**

## C11 -- degenerate paths
- (a) cold user (empty basket, no ID row), 120 instances: q_u == 0 exactly: True; u* == 1 exactly (torch.equal): True; u_hat == 0 exactly: True
- (a) S(u,i) == 1^T t_hat_i: via the matmul score path torch.equal = False, max abs diff 7.276e-12, max diff / sum_l |t_hat_l| = 2.258e-16 (machine precision; bitwise identity fails only because the matmul reduces in a different order than .sum); via element-wise product + .sum torch.equal = True
- (b) cold items (no ID row, words present): min ||q_i||/mean||E_I word row|| = 3.057e-01; min over items of (max_k - min_k) t*_k = 2.794e-01; min ||t_hat||/(||W_t|| ||q_i||) = 2.112e-01
- (b) twin cold items with identical word sets: identical t* and t_hat (torch.equal) = True; max abs diff 0.000e+00
- **Verdict: CONFIRMED (S = 1^T t_hat to machine precision, 1e-16 relative; not bitwise through the matmul path)**

## C12 -- bag-size channel (statistical trend)
  V=8, d=3, F=3, 20000 paired baskets: E||basket mean|| for n=1..10 = 1.632 1.340 1.234 1.178 1.143 1.117 1.099 1.086 1.075 1.066 (s.e. <= 0.0042) | strictly decreasing: True
  V=50, d=5, F=3, 20000 paired baskets: E||basket mean|| for n=1..10 = 1.669 1.201 1.003 0.894 0.819 0.766 0.726 0.694 0.668 0.647 (s.e. <= 0.0033) | strictly decreasing: True
- u*(c q_u) vs u*(q_u), c in {1e-3,0.5,3,1e3}: max abs diff 4.441e-16; ||u_hat(c q)|| vs c ||u_hat(q)||: max rel err 3.787e-16
- **Verdict: CONFIRMED**

## C13 -- collapsed function class
- S = (W_t^T 1).q_i + qhat_u^T (Phat_u^T W_t) q_i + (W_u^T 1).q_u + qhat_i^T (Phat_t^T W_u) q_u: max rel err 8.207e-16 over 240 instances (ids and noid arms)
- **Verdict: CONFIRMED**

## C14 -- K10 Procrustes alignment penalty (interpretation: R in O(d) acts on the embedding dimension, row-wise)
- (a) E_U = R0 E_I (R0 random orthogonal, det +1 and -1), 6 seeds x 3 shapes x 3 scales: max L_align/||E_I||^2 = 1.195e-30 (zero to machine precision)
- (a) random unrelated tables: min L_align/||E_I||^2 = 3.316e-01 (> 0); the converse (L=0 => E_U = R E_I) is definitional with R the Procrustes minimiser
- (b) R held constant: min ||grad_E_U||/(2 sqrt L) = 1.000e+00, min ||grad_E_I||/(2 sqrt L) = 1.000e+00; both equal 2 sqrt(L) to rel err 6.661e-16 -> nonzero whenever L > 0
- (c) E_U = c * E_I (otherwise identical): L_align/||E_I||^2 per (seed, d, c), predicted (|c|-1)^2:
    seed 0 d=3 c=+0.001: L/||E_I||^2 = 9.980010e-01  predicted (|c|-1)^2 = 9.980010e-01
    seed 0 d=3 c=+0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
    seed 0 d=3 c=+0.999: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
    seed 0 d=3 c=+1.001: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
    seed 0 d=3 c=+2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
    seed 0 d=3 c=+1000: L/||E_I||^2 = 9.980010e+05  predicted (|c|-1)^2 = 9.980010e+05
    seed 0 d=3 c=-0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
    seed 0 d=3 c=-1: L/||E_I||^2 = 5.020856e-32  predicted (|c|-1)^2 = 0.000000e+00
    seed 0 d=3 c=-2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
    seed 1 d=4 c=+0.001: L/||E_I||^2 = 9.980010e-01  predicted (|c|-1)^2 = 9.980010e-01
    seed 1 d=4 c=+0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
    seed 1 d=4 c=+0.999: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
    seed 1 d=4 c=+1.001: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
    seed 1 d=4 c=+2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
    seed 1 d=4 c=+1000: L/||E_I||^2 = 9.980010e+05  predicted (|c|-1)^2 = 9.980010e+05
    seed 1 d=4 c=-0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
    seed 1 d=4 c=-1: L/||E_I||^2 = 3.467819e-31  predicted (|c|-1)^2 = 0.000000e+00
    seed 1 d=4 c=-2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
    seed 2 d=5 c=+0.001: L/||E_I||^2 = 9.980010e-01  predicted (|c|-1)^2 = 9.980010e-01
    seed 2 d=5 c=+0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
    seed 2 d=5 c=+0.999: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
    seed 2 d=5 c=+1.001: L/||E_I||^2 = 1.000000e-06  predicted (|c|-1)^2 = 1.000000e-06
    seed 2 d=5 c=+2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
    seed 2 d=5 c=+1000: L/||E_I||^2 = 9.980010e+05  predicted (|c|-1)^2 = 9.980010e+05
    seed 2 d=5 c=-0.5: L/||E_I||^2 = 2.500000e-01  predicted (|c|-1)^2 = 2.500000e-01
    seed 2 d=5 c=-1: L/||E_I||^2 = 5.353706e-31  predicted (|c|-1)^2 = 0.000000e+00
    seed 2 d=5 c=-2: L/||E_I||^2 = 1.000000e+00  predicted (|c|-1)^2 = 1.000000e+00
- (c) L > 0 for every positive c != 1 tested: True; c = -1 gives L = 0 (R = -I is orthogonal): True -> the literal 'c != 1' wording fails at c = -1; not scale-invariant for c > 0 confirmed
- **Verdict: CONFIRMED for c > 0 (a, b, c); literal (c) wording FALSIFIED at c = -1 (sign flip is orthogonal)**

