# L4 — the tag-on: loss-grounded item prototypes on a byte-identical host

Locus: nothing in the forward pass of the host changes (free item table, free prototypes, `t*_k = 1 + cos(t_i, p_k)`, score `uᵀt*`). Grounding enters through **one added loss + one added read-out**. Chosen candidate = the **surrogate-profile loss** (the `[captured dc02]` loss-enforced entry, sharpened); anchor-inclusion is the runner-up/sibling (below).

## Candidate: surrogate-grounded item prototypes (working name "sI-ProtoMF", loss-enforced rung)

**New parameters.** Per prototype k a profile `a_k ∈ R^V` (+ intercept `b_k`), V = attribute-value vocabulary (field-offset multi-hot `x_i ∈ {0,1}^V`, F ones per item — the same `feature_ids` buffer dc01 uses). K×(V+1) ≈ 50×535 ≈ 27k params.

**The one loss (per batch, all scored items i, all k):**

    L_ground = mean_{i,k} ( (t*_k(i) − 1) − (a_kᵀ x_i + b_k) )²        (+ optional ridge/lasso on a_k)

added to the extractor's `get_and_reset_loss()` accumulator with weight λ_g (sits exactly beside `sim_proto_weight`/`sim_batch_weight`; S5 unchanged: one sparse (B·N×V)·(V×K) matmul per step).

**Two-way gradient (this is the mechanism, not a diagnostic):** ∂L_ground/∂a_k fits the profile to the activations (a_k = running least-squares of "prototype k's response" on attributes); ∂L_ground/∂t_i and ∂/∂p_k pull activations toward attribute-predictability — attribute-identical items are dragged to equal activation profiles, i.e. the S7 §4.1.3 REGULARIZE route realised on the *prototype-similarity profile* rather than on raw vectors. λ_g dials the split: λ_g→0 = host + post-hoc linear probe; λ_g→∞ = activations become linear functions of attributes = a dc02-like bottleneck in disguise (item capacity collapses to the ~68% signature ceiling). The thesis's enforcement spectrum becomes a *continuous knob*.

**Cold-item rule (what makes the profile load-bearing).** An item with no trained row gets `t*_k = 1 + clip(a_kᵀx_i + b_k, −1, 1)` — the surrogate IS the forward pass for cold items (R5 native; K numbers, zero new params). Variant for warm-thin items: convex blend by interaction count (one more knob; not stacked by default).

**Read-out.**
1. Prototype meaning = `a_k` rendered per field ("P7: dept Denim Trousers +0.31, colour Dark Blue +0.12, …, index Menswear +0.09"), **with its fidelity stamp**: in-sample R²_k of the surrogate (and activation variance — see degeneration) printed on every prototype card. A prototype with R² 0.85 is "attribute-explained"; one with R² 0.2 is honestly "mostly latent".
2. Per-prototype contribution `u_k·(t*_k−1)` unchanged (host semantics).
3. Per-feature shares of an activation: `a_{k,v}·x_{i,v}` for the item's own F values — exact for the surrogate, approximate for the model by the per-item residual `r_{ik} = (t*_k−1) − a_kᵀx_i − b_k`, which is rendered as an explicit "item-specific" line (the analogue of dc01's ID-share, but measured, not parameterised).
4. Item-level: Σ_k u_k r_{ik} = "how much of this score the vocabulary cannot explain" — per-recommendation latent mass.

**Identifiability (checked, toy).** With one value per field, the design matrix has rank V−(F−1): only the F−1 per-field offsets are unidentified (shift all of a field's coefficients by c, absorb in b_k). Within-field *contrasts* are identified (toy: N=2000, F=3, V=12 → contrast error ≤0.012 vs raw-coefficient error 0.6). Render per field with the field mean subtracted; cross-field correlated cliques (department↔garment group) inherit F-DC01-06's clique-level reading; ridge/lasso only as a preimage *selector*, declared as such.

**Degeneration modes.** (i) *Flat activations*: if t*_k is near-constant the surrogate "fits" with R²≈anything — toy shows residual and signal variance both ~1e-12; gate R² on activation variance (report var(t*_k) next to R²; the host's R_proto/R_batch already fight flat sims). (ii) *Gravity over-pull* at high λ_g: activation profiles of twins merge (accuracy drop, measurable ids-vs-noid-style). (iii) *Trivial surrogate*: a_k≈0, b_k fits the mean → R²≈0 — this is honest failure, visible, not silent. (iv) Nothing forces distinct profiles across prototypes — duplicates remain the host's M3/M4 problem, untouched.

**Rashomon-tunability.** λ_g is the textbook case of requirements §4's note: hyperopt on hit_ratio@10 is expected to be flat over a wide λ_g range; within that range pick the λ_g maximising mean R²_k (interpretability-tuned within the Rashomon set). This is also the *measurable delta over the post-hoc rung* (next point).

**Is it intrinsic (R2)? Both sides, honestly.**
- *Against (Rudin standard, S1/C3):* for warm items the score never touches a_k; a_k is a jointly trained linear surrogate — a "post-hoc explanation trained during training". Fidelity is empirical (R²), not structural. The devil's-advocate S1 objection transfers verbatim: the same cards can be produced post-hoc on the trained host by regressing its activations on attributes (λ_g=0).
- *For (ProtoMF's own standard, A1 §3 — verified in PDF):* ProtoMF's interpretability claim rests on *regularisers* (the two inclusion terms "forcing each user/prototype to get matched", inspired by Li et al.) that make a post-hoc reading (nearest items, "similar to Alvarez-Melis & Jaakkola") meaningful — the host itself is loss-enforced interpretability. Precedents: EMF (Abdollahi & Nasraoui 2016/17, verified in S2 survey §3.2: "explainability regularizer" pulling user/item vectors together when neighbours purchased the item) and SENN (E2, verified: "interpretability built-in architecturally and *enforced through regularization*"; faithfulness+stability as regularisers). The honest label: **regularisation-enforced intrinsic** — one rung below by-construction, one above emergent; and for *cold items* it is fully intrinsic (the surrogate is the forward pass).
- *What the loss buys over λ_g=0 (the measurable delta):* distributions of R²_k and of per-item latent mass, host vs candidate, at matched accuracy. If the delta is ~0, the candidate is a diagnostic dressed as a method and should be demoted to the S1 rung — that experiment is cheap (one hyperopt dimension, login-node-class analysis) and must be run before claiming anything.

**Hidden effects.** No new score channel, baseline mass unchanged (Σ_k u_k still user-keyed, rank-inert). The gravity term is item-side only; popularity still has no item-keyed home (F-DC01-09 holds). Interaction with R_proto/R_batch: both act on the same sim matrix — the three weights form a 3-D interpretability knob set; document, don't stack defaults.

**Contribution or diagnostic?** Brutal version: the *mechanism* is a jointly-trained linear probe plus a regulariser; its defensible contributions are (1) the cold-item route through the profile (load-bearing), (2) the continuous enforcement knob that turns the spectrum narrative into an ablation ladder on ONE host, (3) a per-prototype fidelity stamp (R²) — something dc01/dc02 cannot print because their grounding is by construction. Weak point: for warm items it never beats "post-hoc" structurally, only empirically. It is the **missing middle rung** (emergent dc01 → loss-enforced → by-construction dc02) — a thesis asset even if accuracy/fidelity deltas are small.

**Item-side aggressiveness vs dc01:** none by construction (free table retained, twins separable, popularity/length channels untouched); aggressiveness is λ_g-continuous. Attribute-group richness of prototypes survives fully (profiles over all V).

**R1–R8.** R1 partial — features enter the objective and the cold path, not the warm forward pass. R2 partial — regularisation-enforced (ProtoMF/EMF/SENN standard), cold-item strong. R3 strong — prototype-shaped, unchanged scorer. R4 partial — explicit learned association, identified up to field offsets, fidelity measured not guaranteed. R5 strong (mechanism) / unmeasured (needs the cold split). R6 strong by construction at small λ_g (host capacity intact), trades off as λ_g grows. R7 partial — vocabulary can be restricted to user-perceivable fields at will (V_small) without touching accuracy-bearing capacity (the §4 "accuracy-bearing vs read-out vocabulary" resolution realised literally). R8 strong — one loss, one read-out.

## Rejected / sibling seeds
- **Anchor-inclusive regularisation** (generalise `_inclusiveness_constraint` / max-reg to a prototype×anchor sim matrix, anchors = epoch-refreshed centroids of attribute values or full signatures): most ProtoMF-native tag-on, ~10 lines; but read-out = "nearest anchor + distance" with no per-feature shares (value anchors) or no "which field matters" (signature anchors), and the anchors re-import dc04's centroid pathologies (near-mean collapse, popularity-weighted means, coupled refresh dynamics). Keep as the *cheapest* variant if a profile is not needed.
- **Post-hoc linear probe only** (λ_g=0): the control, not a candidate — it is the S1 rung and must be run.
- **Surrogate in the score at train time** (t* = blend of CF and surrogate for warm items): becomes a dual-channel activation = dc02 ⊕ host, two mechanisms (R8), belongs to L3's territory.
- **Per-prototype attribute *masks* on dc01** (part-based subsets): changes dc01, not a tag-on on the host.

Literature status: A1 regulariser/inclusion + naming passages, S2 §3.2 EMF, E2 abstract/§1 verified in local PDFs; S7 §4.1.3 REGULARIZE via the local summary/vault; Li et al. (ProtoMF's [35]) and Liu et al. 2019 not checked locally — [from memory — verify].
