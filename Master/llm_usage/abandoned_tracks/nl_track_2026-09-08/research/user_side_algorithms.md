# nl track — user-side composition: literature-grounded evaluation of algorithmic forms

**Written:** 2026-09-09 (Claude, Fable 5.1), for SP3–SP10 of `Master/temp/nonlinear_below_prototypes_research_plan_2026-09-08.md`.
**Inputs:** plan §0/§1/§3/§4/§6; SP0 memo `sp0_w1_memo_2026-09-08.md` §2.1, §2.4, §3.1; requirements R1–R8/S1/S5; user-vocabulary research §0.
**Status:** working evidence for a light `/design-candidate` cycle — not thesis text. Every citation in §10 was fetched this session (arXiv PDF/abs or Semantic Scholar record); items marked ⚠️ could not be fetched and are stated from memory.

## 0. Setting, restated as constraints on the menu

- Composition today: q_u = (1/|H|) Σ_{j∈H} c_j (+ e_ID), c_j = Σ_{f∈item j} e_f. Score = Σ_l t_l (1 + cos(q_u, p_l)), the layer frozen.
- Measured (SP0 §2.4): the fU-vs-dot deficit is −0.050 at |H|=3 and −0.104 at |H|=13–24 on hm (CIs disjoint); on ml-1m it peaks at 26–56 films (−0.099) and is ≈0 at 223–492 where both models collapse. fU-noid ≡ fU on hm (ID row inert); on ml-1m the ID row is worth 0.01–0.02 mid-band.
- Two facts the literature must be read against: **(a) the shifted cosine is scale-invariant in q**, so any per-user positive scalar on the history part (mean vs sum vs |H|^{-α}, softmax vs unnormalised attention mass) is invisible in the noid arm and only matters relative to a concatenated/added ID row; **(b) P5 absorption** — any fixed function of the user's own history is in the ID row's function class, so nonlinear composition can only change capacity in noid arms, generalisation/inductive bias in ids arms, cold behaviour, or — via target conditioning — escape P5 altogether.
- Sizes: hm |H| median 5 / p90 12 / max 89; ml-1m median 56 / p90 223 / max 1,415; d ≈ 64–100.

## 1. Per-purchase weighting from content (SP3): MIL attention, gated attention, ACF item-level, RETAIN, FISM, per-item scalar

### 1.1 What each form represents

| form | weight w_j | what is nonlinear | new interaction represented |
|---|---|---|---|
| FISM (Kabbur et al. 2013) | uniform, ×|H|^{-α}, target excluded | nothing (α is a length normaliser) | none — the linear ancestor; item-ID rows p_j/q_i, no user row |
| per-item scalar (linear control) | w_j = exp(s_{item j}), one free scalar per item | none (row rescaling) | none — purchase importance as an item property |
| MIL attention (Ilse, Tomczak & Welling 2018, eq. 8) | softmax_j(wᵀ tanh(V h_j)), h_j = c_j | tanh MLP on the purchase content + softmax coupling | "this purchase counts more *because of what it is*"; weights shared across users |
| gated attention (Ilse et al. 2018, eq. 9) | softmax_j(wᵀ[tanh(V h_j) ⊙ sigm(U h_j)]) | gating removes tanh's near-linearity in [−1,1] | same, sharper |
| ACF item-level (Chen et al. SIGIR 2017) | softmax over history items conditioned on the **user vector** and item content | MLP on (u, item) | user-specific weights → an ids-arm form (u is a free row) |
| RETAIN (Choi et al. NeurIPS 2016) | scalar α_j (softmax) × vector β_j (tanh) from reverse-order RNNs | RNNs over visits | order-aware weights; contribution ω = α_j W(β_j ⊙ W_emb[:,k]) x_{j,k} exact given (α, β) (their eq. 5) |

Ilse et al. frame all of these as Zaheer's sum-decomposition with a *trainable* pooling operator: "replace ... max and mean by a trainable ... two-layered neural network" (their §2.3); attention weights are non-negative and sum to one, so the pooled vector stays inside the convex hull of the purchases — the only difference from the mean is *which* corner of the hull.

### 1.2 Critical reading

1. **The linear control is stronger than it looks.** Content attention w_j = g(c_j) is a per-*item* scalar (same garment → same pre-softmax logit for every user). Its only advantages over a free per-item scalar s_i are (i) generalisation to items never weighted in training (irrelevant for warm eval; relevant for cold), and (ii) the softmax coupling, which is a per-user normaliser and hence invisible under the cosine in the noid arm. So on warm hm/ml-1m, MIL attention ≈ per-item scalar + a normaliser the layer cannot see. Any attention gain must first beat the per-item-scalar arm before it is credited to "attention".
2. **Reported gains over mean pooling come from needle-in-haystack bags, not from small bags.** Ilse et al. Table 2/3 (histopathology, 5 seeds): breast cancer AUC Embedding+mean 0.796 vs Attention 0.775 vs Gated 0.799; colon cancer 0.940 vs 0.968 / 0.968. On the five classical MIL datasets (Table 1) attention is "comparable with the best performing classical MIL methods" — not better than MI-Net. The MIL premise is that one key instance triggers the bag label; with 5 purchases and a target that depends on the whole taste, the premise is weak on hm, plausible on ml-1m (56 films, some noise).
3. **Softmax over a variable-size set is itself a length-handling decision, and it failed in recsys.** NAIS §3.3 (He et al. 2018): the "standard solution of attention ... underperforms FISM significantly" because softmax L1-normalisation punishes active users; the fix is the β-smoothed denominator (their eq. 9), i.e. a controlled departure from mean pooling toward sum pooling. Under the frozen cosine (noid) this whole dimension collapses; under the ids arm it decides the history/ID norm handover (see §7).
4. **RETAIN's value for us is the attribution algebra, not the RNN.** Their Table 2: RNN+α_R (scalar attention only) AUC 0.8717 vs RNN+α_M 0.8624; RETAIN itself is reported "comparable". The point to import: with nonlinear weights on linear rows, the contribution of purchase j to prototype l is exactly w_j · cos-share — "exact given the weights". The weights then need Wiegreffe & Pinter's four tests (uniform baseline, seed variance, frozen-weight diagnostic, adversarial weights); Jain & Wallace's finding that alternative attention distributions yield equivalent predictions is the risk. In a *linear-rows* model adversarial-weight search is cheap (a small LP per user).
5. **ACF is an ids-arm form by construction** (user vector inside the attention), so its reported gains do not speak to composition capacity; and it was built for multimedia item content, not tag bags. Treat as precedent for "user-conditioned weights", nothing more. ⚠️ its SVD++-vs-ACF table numbers were not re-fetched (ACM 403; only the metadata record was reachable).

### 1.3 Parameter / compute / LOO

- Params: MIL attention d·h + h (h ≤ d): ≤ 10 k; gated 2d·h; per-item scalar |I| (13.7 k hm / 3.1 k ml). ACF adds user rows (already present in ids arm).
- Compute under sampled softmax: weights computed once per user per step (not per candidate); a `(B, H_cap, d)` gather + one `(B, H_cap, h)` MLP; ≈ 1.1–1.3× fU. S5 profile preserved.
- Leave-one-out: mask the slot whose item id equals `pos_i_idxs` *before* the softmax (the algebraic 1/|H| subtraction of commit 89a1818 only works for the mean); FISM/NAIS write this as Σ_{j∈R_u^+ \ {i}} (NAIS eq. 3/9; FISM slides). Cost: zero extra forwards.

## 2. Purchase × purchase interactions (SP4): Janossy k=2, bi-interaction, SAB, co-occurrence models

| form | representation | params | pairs at hm median 5 / ml median 56 / ml max 1,415 | attribution |
|---|---|---|---|---|
| Janossy k=2 (Murphy et al. ICLR 2019, Def. 2.2) | Σ_{j<k} f(c_j, c_k), f an MLP | ~2d·h | 10 / 1,540 / 1.0 M | pair units are exact summands (given f) |
| bi-interaction (He & Chua 2017 NFM) | ½[(Σ c_j)² − Σ c_j²] element-wise | 0 | same pairs, closed form O(|H|·d) | exact per pair, but see F3 of the vocabulary survey |
| SAB (Lee et al. ICML 2019, eq. 8) | MAB(X, X): multihead self-attention + rFF, O(|H|²) | ~4d² per block | as pairs; ISAB O(|H|·m) | pairwise *weights* readable, values opaque |
| co-occurrence linear models: FISM/SVD++ item-item, EASE (Steck WWW 2019) | score_ui = Σ_{j∈H} S_{ji} — history × **candidate** pairs, no history × history pairs | |I|·k or |I|² | linear in |H| | exact summands |

Critical reading:
1. **Expressivity says pairs buy nothing once ρ is nonlinear.** Murphy et al. Remark 2.1 / Corollary 2.1: Deep Sets is 1-ary Janossy pooling and "pushes the modeling of k-ary relationships to ρ". Their Table 1: with a *linear* ρ, k=2 lifts unique-count accuracy 0.36→0.74 and variance RMSE 119→4.4; with an MLP ρ, k=1 already reaches ≈1.00 on every task. For us ρ is the **frozen** 1+cos layer — a fixed, non-universal ρ. So explicit pairs are the *only* way to put pair content into q_u without a learned ρ — but that is an attribution argument (pair units as summands) more than a capacity one, because a learned ρ (SP5) would subsume it.
2. **The pair signal on hm is a basket signal.** 94.3 % of purchases share their date with another purchase of the same user; pairs = same-day complements, 10 per median user. The vocabulary research (§0 pt 2) already showed cross-item *word* pairs are a deterministic quadratic of the count vector and bi-interaction ≈ ½ μ⊙μ — it feeds the shared-direction collapse (P2) unless centred first. Over purchase vectors the same holds (every c_j contains μ).
3. **Set attention is not uniformly better than pooling.** Set Transformer Table 1 (max regression): rFF+max 0.1355 MAE vs SAB+PMA 0.2085 — the plain operator wins when it matches the task; Table 2 (unique-character counting): rFF+Pooling 0.438, rFF+PMA 0.457, SAB+PMA 0.604 — SAB wins when the task is genuinely about *distinctness among elements*. Our task has no known distinctness structure; SAB's cost is O(|H|²) per user (ml-1m tail: 2 M pair scores per user per step).
4. **Recsys "co-occurrence" models are history×candidate, not history×history.** FISM, SVD++, EASE score a candidate by its similarity to each history item — pairs of the SP9 kind. EASE's abstract claim that a closed-form linear item-item model "outperforms deep non-linear models" on most benchmarks is the standing warning that nonlinearity over interaction sets rarely beats well-normalised linear co-occurrence. None of these models a history×history term; nothing in this literature reports a gain *from* such a term.

LOO: drop every pair containing the positive (pairs recomputed after masking) — trivial for Janossy/bi-interaction, awkward for SAB (mask key/value rows).

## 3. General set encoders (SP5): Deep Sets, PMA, and expressivity limits

- Deep Sets (Zaheer et al. NeurIPS 2017), Theorem 2: for a countable universe any permutation-invariant f(X) = ρ(Σ_x φ(x)); the uncountable case is proved only for fixed set size M (Wagstaff et al. 2019 restate this as sum-decomposition via R^{M+1}). **The proof of Theorem 2 constructs an injective φ (a 4^{-c(x)}-style code) — i.e. the universality guarantee is literally a memorisation construction.** See §4.
- Wagstaff, Fuchs, Engelcke, Posner & Osborne (ICML 2019), Theorem 4.1 and summary items 1–2: a latent dimensionality of **at least M is necessary** (and M sufficient) for a *continuous* sum-decomposition to represent all continuous permutation-invariant functions on sets of size ≤ M; "a similar statement is true for ... max-decomposition". Toy (their Fig. 3): RMSE on set-median prediction falls with latent width until a "critical point" near the set size, then plateaus. Wagstaff et al. 2022 (JMLR, arXiv 2107.01959) sharpen this: below the threshold Deep Sets cannot approximate certain piecewise-affine functions better than a constant, and the analysis extends to Janossy pooling.
- PMA (Lee et al. 2019, eq. 11): PMA_k(Z) = MAB(S, rFF(Z)) — k learned seed queries attend over the set; k=1 is single-query attention pooling (≈ MIL attention with a learned query instead of a fixed vector; the paper says rFF+Dotprod "roughly corresponds" to Ilse et al.). Their universality proof goes through the same latent-dimension route.

Critical reading for d ≈ 64–100:
1. **hm: capacity is not the constraint.** p99 |H| = 24, max 89; d ≥ 64 satisfies Wagstaff's bound for > 99 % of users. If the mean fails at |H| = 5 it is not for lack of a latent dimension — consistent with SP0: the deficit is already −0.050 at |H| = 3, where a 3-element set function has essentially nothing to be nonlinear about.
2. **ml-1m: the bound is violated for ~half the users** (median 56 ≈ d; p90 223 ≫ d), so a Deep Sets encoder with latent d cannot be universal there — but neither is the mean, and the plan's H_comp hypothesis is not "universal", it is "represents *some* information the mean loses". The honest expectation from the theory is: for |H| ≫ d every sum-pooled encoder degrades toward a low-dimensional summary; the measured collapse of *both* columns above 223 films (SP0 §2.4) is what that looks like. Nothing in this literature promises a fix at |H| = 1,415 with d = 64 short of raising the latent width (Wagstaff) or capping H (which discards data).
3. **The one cheap thing ρ can do is affine.** Mean-centring q ← q̄ − μ is an affine map; the LLN collapse of the mean toward μ as |H| grows is exactly the P2 mechanism the plan attributes to the layer. Any ρ with a bias term learns this for free, so a nonlinear-vs-linear 2×2 whose linear twin is *uncentred* fU will credit centring to nonlinearity. The linear twin must be fU′ (centred) or the reading is confounded.

Params: φ, ρ two-layer MLPs at width ≤ 2d: ≈ 4d² + 4d² ≈ 33–80 k. Compute: one (B, H_cap, d) gather + two MLPs; PMA adds a (B, H_cap) attention. LOO: drop φ(c_pos) before ρ (Deep Sets); mask before PMA softmax.

## 4. Item-ID rows in the history (SP6) and the memorisation hazard

- SVD++ (Koren KDD 2008; metadata verified, formula ⚠️ from memory): p_u + |N(u)|^{-1/2} Σ_{j∈N(u)} y_j — a free user row plus a sum of free *history* item rows with inverse-square-root length normalisation. This is fU-with-ID with content rows replaced by ID rows; the ProtoMF ids arm is its direct descendant.
- FISM (Kabbur, Ning & Karypis KDD 2013; DOI 10.1145/2487575.2487589): ŷ_ui = b_i + |R_u^+ \ {i}|^{-α} Σ_{j∈R_u^+\{i}} p_jᵀ q_i — **no user row at all**; two item factor tables; the abstract states gains "increase as the data gets sparser". NAIS (He et al. TKDE 2018, arXiv 1809.07053) keeps this shape and adds target-conditioned weights.
- Precedent on *our* dataset: NAIS Table 5 (ml-1m, leave-one-out, HR@10 over 1 positive + 99 sampled negatives — the same candidate count as ours; k = 16): FISM 66.47 HR / 39.49 NDCG; MF-BPR 66.64; at k = 64 (Table 6) FISM 70.17 / 42.82. Our content-only history mean under a plain dot (`lightfm_hist`) scores 0.612 HR@10 on ml-1m; fU 0.544. Protocol differences (their held-out item is the user's *latest* interaction, ours is the canonical split's; their MovieLens is the NCF preprocessing, ours the S0 split; negative-sampling seeds differ) forbid a subtraction, but the order of magnitude is the point: **a history made of item-ID rows is worth ~5–9 HR points more than a history made of content rows on ml-1m, with no user row and no nonlinearity.** That is the SP6 headroom, and it dwarfs every nonlinear-composition gain in this file.

The hazard, stated precisely:
1. A user's history is a near-unique key: hm item-set twin rate 0.10 %, ml-1m 0 % (SP0 §2.2). Any encoder that is (nearly) injective on histories is a hashed user-ID table. Deep Sets' Theorem 2 proof *is* such an encoder; DHE (Kang et al. KDD 2021, arXiv 2010.10784) shows empirically that a deep network over a hashed identifier reaches "comparable AUC against the standard one-hot full embedding" — a nonlinear map of a unique key is an embedding table by another name.
2. With ID rows in the history (SP6) the hazard is sharper: y_pos inside q_u lets the model score the positive by *its own row* — the FISM `\{i}` exclusion is non-negotiable, and it must also hold for sampled negatives that happen to be in the history (they are then a *negative* row inside q_u — harmless but worth asserting).
3. Controls the literature implies: (a) a hashed-basket-ID arm (one free row per distinct item set) as the memorisation ceiling — on hm that is 73,377 rows ≈ a user table, i.e. the ceiling is simply the ids arm; (b) train/val gap; (c) name-coverage share per prototype (how much activation mass sits on unnamed y_j rows).

Attribution: per-purchase shares stay exact summands (y_j is a row like c_j); the y_j part is unnamed and must be disclosed as such (survey E-4).

## 5. Target-conditioned attention (SP9): NAIS, DIN, ACF

| model | weight a_{ij} | normaliser | user row? | reported gain over uniform/sum pooling |
|---|---|---|---|---|
| NAIS-prod / -concat (He et al. 2018, eq. 9) | exp f(p_i, q_j) with f an MLP on p_i ⊙ q_j or [p_i; q_j] | softmax denominator raised to β ∈ [0,1]; β = 1 (plain softmax) "worse than FISM" (Fig. 6) | **none** | Table 5 (k=16, 1+99 sampled protocol): ml-1m HR 66.47→69.69 (+3.2 pts), NDCG 39.49→41.94; Pinterest HR 87.40→88.44 (+1.0). Table 6 (k=64): ml-1m 70.17→71.82 (+1.65); Pinterest 88.62→88.84 (+0.2) |
| DIN (Zhou et al. KDD 2018, eq. 3) | v_U(A) = Σ_j a(e_j, v_A) e_j, a(·) an MLP over the behaviour embedding, the candidate embedding and their product | **none** — Σ w = 1 "relaxed ... to reserve the intensity of user interests" | user-profile features on Alibaba; ⚠️ not verified whether a user-ID embedding is used | Table 3: MovieLens AUC 0.7300→0.7337 (+0.0037), Amazon 0.8624→0.8818 (+0.019); Alibaba +0.0052 abs (Table 5) |
| ACF (Chen et al. SIGIR 2017) | softmax over history items conditioned on user vector + item content (NAIS: "the same as the history modeling part of the Attentive CF model") | softmax | yes (free user row) | ⚠️ numbers not re-fetched |

Critical reading:
1. **NAIS is the cleanest precedent in this file**: no user row, item-based, evaluated on ml-1m under a 1+100 sampled protocol close to ours, gain over its own linear ancestor (FISM) measured in the same code. The gain is +1.6 to +3.2 HR points on ml-1m and shrinks to +0.2–1.0 on the sparser, shorter-history Pinterest — the *direction* the plan predicts for hm (5 items to weight → nil).
2. **The gain depends on a norm trick the cosine cannot see.** Plain softmax (β = 1) lost to FISM; the win needs β < 1, which lets the pooled vector's *magnitude* grow with |H| for active users. Under our frozen cosine in the noid arm, β is irrelevant — the per-candidate profile enters as a direction only. Whether NAIS's gain survives that is an open question the literature does not answer; the dot twin will.
3. **DIN's "intensity" argument is the same trick from the other side** (abandon normalisation so Σw carries interest intensity); its gains over sum pooling are small on MovieLens (+0.0037 AUC) and large only on Amazon with "rich user behaviors". Sum pooling in DIN's base model also means their baseline already had the un-normalised norm channel.
4. **Cost shape.** NAIS: O(a·k·|R_u|) per (user, candidate) prediction with a reuse trick for the per-pair MLP. Under our sampled softmax: 40 candidates × |H| × d per user per step (hm: 40·5·64 = 13 k MACs — trivial; ml-1m at H_cap 128: 330 k — fine); eval: 100 candidates × |H|. The plan's caveat holds — affordable *because* the eval is 1+99, not full ranking.
5. **Explanation object changes**: the user profile becomes per-candidate; RETAIN-style exactness given the weights survives (score = Σ_j a_{ij} · share_j), but "your profile" is no longer one object per user (survey verdict). This is a contract change, not a repair.
6. **P5 escape is real but partial.** Target conditioning is the only form here that is not a fixed function of the user's row; but with 5 purchases the attention has 5 logits per candidate, and in the ids arm the free row still absorbs the *average* over candidates.

## 6. Target leakage / leave-one-out practice

| paper | how the target is kept out of the history at training time |
|---|---|
| FISM (2013) | explicit: Σ over R_u^+ \ {i} (verified in slides; NAIS eq. 3 reproduces it, "to avoid the modeling of self-similarity of the target item") |
| NAIS (2018) | same set R_u^+ \ {i} inside the softmax (eq. 8–9), i.e. weights renormalised without the target |
| DIN (2018) | by construction: "predict the (k+1)-th reviewed goods by making use of the first k"; every prefix k = 1 … n−2 is a training sample (Amazon protocol) |
| RETAIN (2016) | sequential prediction; the label at step i is a future event, never inside x_1…x_i |
| ACF (2017) | ⚠️ not verified |
| Deep Sets / Set Transformer / Janossy / MIL | n/a (bag label is not an element of the bag) |

Our situation: the vocabulary research found the positive's words inside q_u during training (20 % of the metadata mass at |H| = 5); commit 89a1818 introduced the mean-subtraction LOO. Every form in §1–5 needs the *masking* variant (mask slot, then weight/pool) — the FISM/NAIS practice, not the algebraic shortcut. Note the asymmetry the literature ignores: at training the history is |H|−1 long, at test |H| long — for the mean this is a norm shift the cosine ignores; for softmax attention it is a temperature shift that β-style tricks are meant to absorb.

## 7. Norm handling before a cosine layer

- Steck, Ekanadham & Kallus (2024, arXiv 2403.05440): under regularised linear MF, cosine similarities "can yield arbitrary results" and for one common regulariser "are not even unique", because the factorisation is invariant to an arbitrary diagonal rescaling D that cancels in the dot product but not after normalisation (their eq. 4 and §2.2). Remedies: train with the cosine in the loss, or normalise/scale deliberately before training. Our layer *is* trained through the cosine — so Steck's arbitrariness does not apply to the *score*. It applies to the **split** of q_u into history mass and ID mass: the cosine gives no gradient on ‖q_u‖, so the ratio ‖mean‖ : ‖e_ID‖ is set by weight decay and initialisation, not by data. That is P3 (norm handover) in Steck's language.
- Length normalisation choices in the field: FISM |H|^{-α}, α ∈ [0,1] tuned (α = 1 mean, α = 0 sum); SVD++ |N(u)|^{-1/2}; NAIS β on the softmax denominator; DIN no normalisation. All of them exist to control how the pooled magnitude scales with |H| *for a dot-product score*.
- Consequence for the 2×2: in the noid-proto cell mean/sum/α are indistinguishable (scale invariance); in the dot cells they are a first-order knob; in the ids-proto cell they set how much the ID row weighs at each |H|. Concretely: ‖mean‖ shrinks toward ‖μ‖ as |H| grows (non-aligned rows average out), so under **mean** pooling the ID row's relative weight *grows* with |H|; under **sum** pooling it *shrinks*. SP0 §2.4 shows fU−noid = +0.01–0.02 in the ml-1m mid bands and −0.005 at hm |H| ≤ 5: a length-dependent ID share is already visible. Read out ‖q̄_H‖ and ‖e_ID‖ per |H| band before any composition experiment — it is a zero-code instrument and it decides whether α (FISM) is a free win in the ids arm.
- Sentence-embedding practice (SIF, Arora, Liang & Ma ICLR 2017 ⚠️ unverified this session — OpenReview/Princeton pages blocked): frequency down-weighting + removal of the common component — the fU′ centring precedent the vocabulary research already cites.

## 8. Per-form summary table

| form | interaction | params (d=64) | attribution regime | literature gain vs mean | ID row present in that evidence? | compute / step | LOO |
|---|---|---|---|---|---|---|---|
| per-item scalar | none (row rescale) | |I| | exact-linear | — (control) | — | ≈ fU | mask slot |
| MIL / gated attention | content → weight | ≤ 10 k | exact-given-weights | ≈ mean on small bags; +0.03 AUC on colon histology (Ilse T3) | no | 1.1–1.3× fU | mask before softmax |
| RETAIN form (α scalar × β vector) | content/order → weight | 2 RNNs | exact-given-weights (eq. 5) | scalar-α RNN 0.8717 vs no-attention RNN 0.8624 AUC (T2) | n/a | RNN over H | sequential by design |
| Janossy k=2 | purchase pairs | ~2d·h | exact pair summands given f | only with linear ρ (T1) | n/a | O(|H|²) pairs, subsample | drop pairs with pos |
| bi-interaction | purchase pairs | 0 | exact per pair | NFM +7.3 % over FM (features, not histories) | n/a | O(|H|·d) | recompute after mask |
| SAB | all pairs, learned | ~4d² | post-hoc (weights readable) | +0.15 acc on unique-count; worse than max on max-regression (ST T1/T2) | n/a | O(|H|²·d) | mask rows |
| Deep Sets ρ(Σφ) | arbitrary (latent ≥ |H|) | ~8d² | post-hoc (exact Shapley enumerable ≤ 12) | universal in theory; no recsys mean-pooling comparison | n/a | 1.2–1.5× fU | drop φ(pos) |
| PMA_k | learned-query attention | ~4d² + k·d | exact-given-weights for k=1 | +0.02 acc over pooling on unique-count (ST T2) | n/a | 1.2× fU | mask before softmax |
| item-ID history rows (SVD++/FISM) | identity | |I|·d | exact summands, unnamed | FISM 66.5–70.2 HR on ml-1m vs our content mean 61.2 (different protocol) | FISM: no; SVD++: yes | ≈ fU | Σ over R_u^+ \ {i} |
| NAIS target attention | history × candidate | ~a·d + |I|·d | exact-given-weights, per-candidate | +1.6–3.2 HR pts ml-1m over FISM; +0.2–1.0 Pinterest | no | 40× attention per user | Σ over R_u^+ \ {i} |
| DIN activation unit | history × candidate | MLP | exact-given-weights, unnormalised | +0.004 AUC MovieLens, +0.019 Amazon | ⚠️ | 40× | prefix construction |

## 9. Recommendation

### 9.1 Does the literature expect set nonlinearity to help at |H| ≈ 5? — No.
Three independent lines say no: (i) Wagstaff's bound is satisfied trivially at M = 5 with d = 64, so the mean's failure there is not a capacity failure — and SP0 shows the deficit is already −0.050 at |H| = 3; (ii) MIL attention matches, and does not beat, mean-based pooling on small classical bags (Ilse T1), winning only where one rare instance carries the label; (iii) NAIS's target-attention gain over uniform pooling shrinks to +0.2–1.0 HR points on the sparse, short-history dataset. The hm result should be planned as the null and the effort spent on the *explanation* side (which purchase counted, with a faithfulness number).

### 9.2 hm_1_month — ranked forms worth a light cycle

| rank | form | linear twin | dot twin | explanation regime | specific risk here | cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | **MIL/gated attention over purchases** (SP3), noid arm, softmax after mask | fU-noid **+ per-item scalar** (the real control) and fU′ (centred) | `lightfm_hist` + same attention | exact-given-weights; W&P battery; attention-vs-Shapley agreement (SP0 instrument exists) | 5 logits per user; softmax invisible under cosine → gain ≈ per-item scalar; hm ID row already inert so P5 is not the issue, capacity is | 3 SLURM runs (attention / scalar / fU′) read in the 4–5 and 13–24 bands; bar 2 × 0.0026 HR@10. If attention − scalar < bar → null, keep only as explanation instrument |
| 2 | **Length read-out + FISM α in the ids arm** (§7) — not a model, an instrument | fU (α=1) | `lightfm_hist_ids` with α sweep | exact-linear | none | zero-code: ‖q̄‖ vs ‖e_ID‖ per band on existing checkpoints; then one α ∈ {0.5} run |
| 3 | **Deep Sets zero-init residual** (SP5) as the 2×2 headroom cell only | fU′ | `lightfm_hist_nl` | post-hoc; exact Shapley ≤ 12 covers 91 % of users | hashed-basket memorisation (0.10 % twins); ASHA grace 4 may kill zero-init trials early | 2 runs (proto, dot), noid; go/no-go rule of plan §3 |
| — | skip: Janossy/bi-interaction/SAB (10 same-day pairs, feeds P2), NAIS/DIN (nil expected; contract change), item-ID rows (hm ID row inert ⇒ identity carries nothing here; test on ml-1m first) | | | | | |

### 9.3 ml-1m — ranked forms

| rank | form | linear twin | dot twin | explanation regime | specific risk | cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | **Item-ID history rows** (SP6, SVD++/FISM shape), noid arm, FISM `\{i}` LOO, mean pooling | fU-noid | `lightfm_hist` + y_j rows (a FISM with content) | exact summands; y_j unnamed → name-coverage per prototype mandatory | memorisation of the positive if LOO leaks; ids-arm redundancy with e_ID | 1 dot run first: if `lightfm_hist`+y_j − `lightfm_hist` ≥ 0.02 (NAIS precedent says +5–9 pts) identity is the headroom and every nonlinear form below is second-order |
| 2 | **MIL/gated attention** (SP3) with per-item scalar control | fU-noid + scalar; fU′ | `lightfm_hist` + attention | exact-given-weights | 56 films: real noise to down-weight; softmax length variance (NAIS §3.3) irrelevant under cosine, relevant in dot twin; H_cap confound if capped | 3 runs; read the 26–56 band (largest deficit) and 122–222 |
| 3 | **NAIS-style target attention with β** (SP9) — last, only if #2 moves | FISM-with-content (uniform, α) | NAIS-with-content rows | exact-given-weights, per-candidate profile | β trick invisible under cosine; contract change (no per-user profile); double-tie plumbing | 2 runs (dot, proto) at β = 0.5; compare to #2, not to fU |
| 4 | Deep Sets / PMA residual (SP5) as 2×2 instrument | fU′ (centred; H_cap-matched linear mean control) | `lightfm_hist_nl` | post-hoc; Shapley by permutation sampling (24 % of users skipped at cap 128) | latent d < |H| for half the users (Wagstaff) → cannot represent much beyond a summary; H_cap discards data the linear cells used | 2 runs; go/no-go per plan §3 |
| — | skip: Janossy k=2 (1,540 pairs/user; no expressivity over ρ; attribution-only), bi-interaction (≈ μ⊙μ), SAB (O(|H|²), 2 M pairs at the tail) | | | | | |

### 9.4 Cross-cutting rules the literature imposes
1. Every nonlinear arm's linear twin is **fU′ (centred)**, or the arm gets credit for an affine fix.
2. Every attention arm's control is the **per-item scalar**, or it gets credit for a row rescaling.
3. Every gain quoted in an ids arm is a budget artefact until the ID-share read-out and the noid twin agree (P5; the fI 30-vs-100-trial flip).
4. Mask-then-pool LOO everywhere (FISM/NAIS practice); assert the positive's slot is absent in the forward, not merely subtracted.
5. Report per-|H|-band, paired, with the SP0 script — the effects the literature predicts are band-specific (hm 13–24, ml-1m 26–56), and an overall row will average them away.

## 10. Citations (all fetched this session unless ⚠️)

| ref | verified via |
|---|---|
| Ilse, Tomczak & Welling 2018, *Attention-based Deep Multiple Instance Learning*, ICML 2018, arXiv 1802.04712 | arXiv PDF (eqs. 8–9, Tables 1–3) |
| Chen, Zhang, He, Nie, Liu & Chua 2017, *Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-level Attention*, SIGIR 2017, DOI 10.1145/3077136.3080797 | Semantic Scholar record (metadata only; abstract elided; numbers ⚠️) |
| Choi, Bahadori, Kulas, Schuetz, Stewart & Sun 2016, *RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism*, NeurIPS 2016, arXiv 1608.05745 | arXiv PDF (Steps 2–4, eq. 5, Table 2) |
| Kabbur, Ning & Karypis 2013, *FISM: Factored Item Similarity Models for Top-N Recommender Systems*, KDD 2013, DOI 10.1145/2487575.2487589 | Experts@Minnesota record (abstract) + SZU lecture slides (formula) |
| Murphy, Srinivasan, Rao & Ribeiro 2019, *Janossy Pooling: Learning Deep Permutation-Invariant Functions for Variable-Size Inputs*, ICLR 2019, arXiv 1811.01900 | arXiv PDF (Def. 2.2, Remark 2.1, Cor. 2.1, Table 1) |
| Lee, Lee, Kim, Kosiorek, Choi & Teh 2019, *Set Transformer*, ICML 2019, arXiv 1810.00825 | arXiv PDF (eqs. 6–16, Tables 1–2) |
| Zaheer, Kottur, Ravanbhakhsh, Póczos, Salakhutdinov & Smola 2017, *Deep Sets*, NeurIPS 2017, arXiv 1703.06114 | arXiv PDF (Theorem 2, App. A) |
| Wagstaff, Fuchs, Engelcke, Posner & Osborne 2019, *On the Limitations of Representing Functions on Sets*, ICML 2019, arXiv 1901.09006 | arXiv PDF (Thm 4.1, Fig. 3) |
| Wagstaff, Fuchs, Engelcke, Osborne & Posner 2022, *Universal Approximation of Functions on Sets*, JMLR, arXiv 2107.01959 | arXiv abstract |
| Koren 2008, *Factorization meets the neighborhood: a multifaceted collaborative filtering model*, KDD 2008, DOI 10.1145/1401890.1401944 | Semantic Scholar record (metadata; SVD++ formula ⚠️ from memory; Rochester mirror 410) |
| He, He, Song, Liu, Jiang & Chua 2018, *NAIS: Neural Attentive Item Similarity Model for Recommendation*, IEEE TKDE 2018, arXiv 1809.07053 | arXiv PDF (eqs. 3, 8, 9; §3.3; Tables 5–6; Fig. 6) |
| Zhou et al. 2018, *Deep Interest Network for Click-Through Rate Prediction*, KDD 2018, arXiv 1706.06978 | arXiv PDF (eq. 3, §4.3, §6.1, Tables 3, 5) |
| Steck, Ekanadham & Kallus 2024, *Is Cosine-Similarity of Embeddings Really About Similarity?*, arXiv 2403.05440 (cs.IR, Mar 2024; venue not on the preprint) | arXiv PDF (§2.1–2.2, eq. 4) |
| He & Chua 2017, *Neural Factorization Machines for Sparse Predictive Analytics*, SIGIR 2017, arXiv 1708.05027 | arXiv abstract |
| Steck 2019, *Embarrassingly Shallow Autoencoders for Sparse Data*, WWW 2019, arXiv 1905.03375 | arXiv abstract |
| Jain & Wallace 2019, *Attention is not Explanation*, NAACL 2019, arXiv 1902.10186 | arXiv abstract |
| Wiegreffe & Pinter 2019, *Attention is not not Explanation*, EMNLP 2019, arXiv 1908.04626 | arXiv abstract |
| Kang, Cheng, Yao, Yi, Chen, Hong & Chi 2021, *Learning to Embed Categorical Features without Embedding Tables for Recommendation* (DHE), KDD 2021, arXiv 2010.10784 | arXiv abstract |
| ⚠️ Arora, Liang & Ma 2017, *A Simple but Tough-to-Beat Baseline for Sentence Embeddings*, ICLR 2017 | not fetchable this session (OpenReview challenge page, Princeton redirect, S2 429) |
