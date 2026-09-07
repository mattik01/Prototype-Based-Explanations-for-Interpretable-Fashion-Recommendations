# L3 — Decoded-attribute bottleneck prototypes ("ground the interface, not the item")

Working name: **dI-ProtoMF** (decoded-interface). Locus: item→prototype interface. Items stay free CF vectors; prototypes live in attribute space by construction; a supervised decoder bridges the two. A *joint* concept bottleneck (Koh et al. 2020, E1 §: independent / sequential / joint bottlenecks, λ trades concept vs task loss — verified in local PDF) whose concept predictor input is the CF embedding rather than an image.

## 1. Mechanism (I-ProtoMF host, score s = uᵀt* unchanged)

- Item table T ∈ R^{M×d}, **free** (host-verbatim; +ID capacity intact).
- Decoder D: R^d → R^V. Base form: linear + per-field softmax (one value per field on H&M) or sigmoid for multi-valued fields:
  `ĉ_i = σ_fields(W_D t_i + b_D)`  ∈ [0,1]^V — the item's *predicted attribute vector* (its "catalogue row as the model believes it").
- Item prototypes A ∈ R^{K×V}, a_k = attribute profile (dc02 object; rows non-negative via softplus or unconstrained — recommend non-negative + per-field structure so a profile reads as "0.6 Denim, 0.3 Dark Blue").
- Activation in attribute space: `t*_k = 1 + cos(ĉ_i, a_k)` (host shifted cosine kept; see §3 for the dot alternative).
- Losses: `L = L_rec(uᵀt*) + λ_c · L_concept(ĉ_i, x_i)` with L_concept = per-field cross-entropy (softmax fields) / BCE (multi-valued), on the *batch items* only (pos + sampled negs — free, they are already looked up). Host regularizers R_proto/R_batch act on the (B·N)×K sim matrix unchanged.
- Cold items: `ĉ_cold := x_cold` (one-hot per field; same space, calibrated by construction since ĉ is a probability vector per field) → full explanation, zero parameters. Native R5 with no rule beyond "use the catalogue row".
- Learned: T, W_D, b_D, A, u. New params: d·V + V + K·V (hm_3_month: 64·534+534+50·534 ≈ 61k — negligible).
- S5: one (B·N)×d×V matmul per step + field softmax; categorical lookup of x_i for the concept loss. Cheap; no precompute.

## 2. What leakage is here and why it is the dial

Joint CBMs let "the model's version of the concepts be refined to improve predictive performance" (E1, verified). In recsys terms: gradient from L_rec flows into ĉ through W_D and T, so ĉ can drift from x_i wherever that helps ranking — e.g. a white basic tee that behaves "premium" gets ĉ_premium = 0.3. That drift IS the ID-residual, but expressed **in the vocabulary** (dc01's e_ID is a latent vector; here the residual r_i = ĉ_i − x_i is a per-attribute number). λ_c sets the regime:
- λ_c → ∞ : ĉ ≈ x → **dc02** (hard bottleneck; twins indistinguishable; 68% signature ceiling).
- λ_c → 0 : ĉ is a free V-dim code with attribute *names* but no attribute *meaning* → black box wearing labels (the CBM-leakage failure; Mahinpei et al. 2021, Margeloiu et al. 2021 — [from memory — verify]).
- Middle: fidelity is **measured**, not assumed: decoder accuracy per field, mean |r_i|, and the fraction of activation mass carried by r (below). Rashomon-tunable: pick the largest λ_c within accuracy tolerance.
Anti-leakage structure worth building in (still one idea): (i) concepts are *bounded* (softmax per field / sigmoid) so no magnitude channel; (ii) per-field softmax means leakage must be spent as "plausible mis-tags" rather than free real numbers — which is exactly the interpretable form of leakage; (iii) optional `use_bias=0` as dc02 argued (the free item bias is the classic x→y side channel); (iv) no ID row — the item's only route to the score is through named attribute coordinates.

## 3. Read-out objects (intrinsic: all forward-pass quantities)

1. Prototype meaning = a_k, by construction (dc02 inheritance).
2. Item = ĉ_i rendered as its believed catalogue row + residual r_i = ĉ_i − x_i ("catalogue says Basic; model treats as 0.3 Premium") — the ID-residual diagnostic made intrinsic and vocabulary-valued. Warm items with |r_i| ≈ 0 are "explained entirely by their tags".
3. Per-prototype contribution u_k·(t*_k − 1) (host semantics).
4. Per-attribute shares of an activation — **exact under cos**: t*_k − 1 = Σ_v a_{kv} ĉ_{iv} /(‖a_k‖‖ĉ_i‖) (toy: share sum equals activation to machine precision). Shares split further into *catalogue part* a_{kv}x_{iv} and *leak part* a_{kv}r_{iv} (same denominator) — so every activation carries a disclosed "feature-explained fraction" vs "residual fraction", dc01's headline meter, now per attribute. Identifiability: a_k and ĉ_i are both named coordinates (no dictionary fit, no dc04 preimage problem — V coordinates, each anchored by its own supervised target). Caveat: correlated fields (department↔garment group) still give correlated *shares* — read at clique level as dc01 F-DC01-06.
   Dot vs cos: dot (`t*_k = a_kᵀĉ_i`) makes shares exact without the shared denominator and keeps "linear tail" all the way down to attributes (s = Σ_k u_k Σ_v a_{kv}ĉ_{iv} = ĉ_iᵀ(Aᵀu) — the user is effectively a V-vector of attribute affinities through K prototype lenses). But dot loses the host's bounded, scale-free activation (C2 robustness evidence cited by dc02) and lets ‖ĉ‖ become a magnitude channel (with per-field softmax ‖ĉ‖ is bounded in [√F/… , √F], so mild). Recommend cos as host-verbatim default, dot as the ablation rung.
5. Toy (numpy, `toy_l3.py`): two signature twins with different free t get different activations under soft concepts (max |Δt*| ≈ 0.14 at decoder-noise level), identical under the hard bottleneck — the dc02 ceiling is lifted exactly by leakage, and the lift is visible in r_i.

## 4. Positioning

- vs **dc02**: same prototype object and same vocabulary; adds per-item free parameters routed *through* the attribute layer; twins separable; cold path identical; the new risk is leakage, the new read-out is r_i.
- vs **dc01**: items are not composed of words — the item is a free vector decoded *into* words. Grounding sits on the interface; the item side keeps full CF capacity (popularity, twins, latent structure) but must express it as attribute coordinates to reach the score. dc01 corners latency into e_ID (latent, unreadable); dI corners it into r_i (readable, but risks confidently-wrong statements about the item on screen — R7 risk stated below).
- Tie story (R1): D can be weight-tied to a word table: W_D = Eᵀ (V×d word vectors), i.e. `ĉ_{iv} ∝ σ(e_vᵀ t_i)` — an item is "red" to the degree its CF vector aligns with the red word direction; the word table then plays the dc01 role in reverse (TCAV-style concept directions — [from memory — verify]). This makes a symmetric lineage story (fI: items = Σ words; dI: words = readings of items) and, at the fUfI merge, the same E could serve both sides. Keep as the tied variant, not the base; note it is also exactly a supervised version of dc04's centroid anchors (learned discriminative directions instead of class means — no preimage problem because the profile a_k is the parameter, not a preimage).
- Linear tail: preserved at the prototype level (s = Σ_k u_k t*_k); with dot, preserved down to attributes. The user said the tail is not sacred — note that this candidate doesn't need to spend it.

## 5. Hidden effects / risks

- **Leakage = confidently-wrong attribute statements** for the item on screen ("treated as 0.3 Premium"); mitigated only by rendering r_i honestly next to x_i and by bounding λ_c. This is the R7 cost.
- **Calibration**: softmax per field is well-calibrated only if L_concept is weighted enough; λ_c low ⇒ ĉ becomes peaky/uncalibrated; monitor ECE per field.
- **Rare values**: decoder never learns rare values ⇒ ĉ_v ≈ 0 for them, prototypes cannot use them; cold items *with* rare values get x-coordinates the warm world never produced (distribution shift between ĉ and x — test cold items with a decoder-calibrated x, e.g. temperature-smoothed one-hot).
- **V vs d**: decoder is d→V with d≈64 < V≈534: ĉ lies on a ≤d-dim manifold in attribute space, so not all attribute combinations are reachable — a structural ceiling on what leakage can express (good for fidelity, limits capacity).
- **Baseline mass**: Σ_k u_k·1 user-keyed, rank-inert (I host, as dc01). No live item intercept unless use_bias=1.
- **Dynamics**: T gets gradient from two losses; early training ĉ is garbage ⇒ prototypes chase noise; consider sequential warm start (train D on x first, E1's "sequential" rung) as an init policy, not a mechanism.

## 6. Requirements (one line each, no cross-candidate ranking)

- R1 tied: **partial** — features are first-class in the one representation that drives the score (no reranker); the "exemplar tie" only in the W_D=Eᵀ variant / merge stage.
- R2 intrinsic: **strong** — every rendered number (a_k, ĉ_i, r_i, shares) is a forward-pass quantity; nothing reconstructed.
- R3 prototype-shaped: **strong** — host scorer and prototype layer unchanged in form.
- R4 feature-grounded: **strong/partial** — profile by construction, item coordinates supervised; fidelity conditional on λ_c (measured, disclosed).
- R5 cold: **strong** — ĉ_cold = x_cold, same space, no extra rule.
- R6 dense accuracy: **partial (bet)** — free item vector restores capacity above dc02; ceiling from d→V manifold and bounded concepts; vs I-ProtoMF host unknown.
- R7 user vocabulary: **partial** — vocabulary = catalogue fields (choose user-perceivable subset); leakage can contradict the visible item.
- R8 parsimony: **strong** — one idea: "joint concept bottleneck on a CF embedding"; two losses but one mechanism (λ_c is the CBM's own knob, not a stacked trick).
- S1: the sequential/independent CBM rungs are the natural decoupled ablations (λ_c→∞ = dc02; decoder frozen = sequential). S5: cheap.

## 7. Rejected siblings

- **Decoder-only tag-on** (host untouched, D trained post-hoc to read CF item vectors; prototypes named via D(p_k)): post-hoc probe, R2 fails; keep as the S1 "probe rung" baseline for dI.
- **Two-channel activation** t*_k = cos(t_i,p_k) + a_kᵀx_i: additive CF+attribute activation, exact shares, but two prototype objects per k (latent p_k + profile a_k) — stacking (R8), and the latent channel is unnamed.
- **Concept-embedding (CEM-style) bottleneck** with per-concept vectors + activation scalars ([from memory — verify] Zarlenga et al. 2022): deliberately adds leakage capacity; wrong direction for a fidelity-first thesis.
