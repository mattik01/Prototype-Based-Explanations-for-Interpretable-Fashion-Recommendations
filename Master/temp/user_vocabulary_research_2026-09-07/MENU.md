# Candidate-form menu from angles A–D (input to angles E and F)

Read angle_A.md … angle_D.md for the details behind each entry.

## Score-changing forms
M1  Words + item-ID history rows (SVD++/NSVD shape): q_u = (1/|H|)Σ_j (c_j + y_j) [+ e_ID]   (A1/A2)
M2  Leave-one-out pooling in training: exclude the scored positive's own words from q_u (FISM \{i}) (A3)
M3  Target-conditioned attention over purchases (NAIS-style smoothed softmax)                   (A4)
M4  Prototype score + additive FISM/SVD++ item-item term beside it (integrated model)          (A5)
M5  Conjunction vocabulary: within-item cross-field tokens (black∧dress), support-pruned, shared with fI; fU inherits via the mean  (B4 = D1)
M6  Attribute-level association-rule tokens ("bought A and B → C-type", confidence as weight)  (B5)
M7  Co-purchase-subgraph re-weighting of the user's own purchases (fixed population edges)      (B6)
M8  (purchase, target) complement pair term with attribute-composed complement for cold items  (B7; fUfI-merge material)
M9  Population-mean centring of the word mean (fU′), optionally with count-dependent shrink toward zero deviation (C2 + D4)
M10 Dirichlet shrinkage toward the crowd vector (only non-redundant in _noid)                   (C5)
M11 Per-user-per-field concentration gate ("consistent about colour → colour words weigh more") (C8)
M12 Top-k sparsified profile (Balog coverage×significance selection, 0/1 gate)                  (D3)
M13 Saturating / sublinear per-(user,word) counts (EFM)                                         (D5)
M14 Two-vocabulary concat: mean of item-ID rows ‖ mean of words, block-structured prototypes     (D8)
M15 EFM-style direct attribute-match score term beside the prototype score                      (D9)
M16 Disclosed ID-share cap: ‖e_ID‖ ≤ α‖q_words‖                                                 (D10)

## Explanation-layer-only (score untouched)
X1  Item examples rendered under each word / prototype                                          (D6)
X2  Deviation-from-crowd display of the user's words with standard errors                       (C10)

## Already rejected by A–D (state why only if you disagree)
Basket-restricted FM/NFM pair term (B1), AFM attention over word pairs (B2), explicit cross-item pair-row table (B3), Janossy k=2 MLP (B8), IDF/BM25/SIF constant weights (C1), PPMI weights (C4), NL/LLM profile bottleneck (D11), post-hoc Amazon-style lookups.
