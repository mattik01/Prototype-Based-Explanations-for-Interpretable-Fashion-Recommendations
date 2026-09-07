# L5 — grounding via the SIMILARITY itself: facet-gated prototypes ("rule ∧ direction")

Lens: item embeddings free; prototypes grounded *directly and load-bearingly* by changing what "activation" means; intrinsic explainability carried by a non-additive vehicle (a declared attribute facet spec that gates the CF similarity). Scorer form kept only as far as it costs nothing.

## 0. Seeds weighed (rejected siblings, one line each)

- **(a) facet masks over a field-structured item embedding** — with *free* items there is no field decomposition to mask; a "colour block" of a free vector is not about colour unless tied to words (→ dc01 refinement) or supervised (→ L3). Not standalone. *Absorbed*: the mask idea survives below, applied to the observed attributes x_i, where it needs no anchoring.
- **(b) interpretable head over prototype activations (sparse logistic / ProtoTree-style routing, C5a verified: soft tree trained, converted to hard "without loss of accuracy")** — an Axis-B (scoring) change (scratchpad caution 2026-06-16); per-user trees are impossible (73k "classes"); and it grounds nothing — prototypes still need L1–L4. Rejected as a grounding candidate. *Salvage*: a tree whose nodes are attribute tests and whose leaves are conjunctions = "prototypes as learned attribute rules" — that is the rule half of the candidate below, without its CF half (then twins collapse, dc02's 68% ceiling again).
- **(c) exemplar push onto real items (C1 verified: "push" each prototype onto the nearest latent training patch, "conceptually equate each prototype with a training image patch")** — buys exactness + identifiability over ProtoMF's post-hoc top-1 (activation literally = cos to that garment) at ~zero cost, items fully free; loses the attribute *group* (one garment ≠ a group), and the exemplar's attribute row is description, not the reason (Rudin GC#4 whole-to-whole; Kim et al. 2014 part-based as the fix — mention verified in S1_SUMMARY only, paper not local [from memory — verify]). Kept as the *cheapest fallback* and as an optional initializer for the candidate's CF half.
- **(d) dual-channel additive activation** `t*_k = a_kᵀx_i + cos(t_i,p_k)` — honest "feature-explained share per activation", but two mechanisms summed (R8 stacking) and the CF channel can silently carry everything; it is dc02 ⊕ host. Rejected; the candidate below is its *multiplicative* cousin with one object.
- **(e) attention over attributes** — E4a verified ("learned attention weights are frequently uncorrelated with gradient-based measures … very different attention distributions yield equivalent predictions"): weights are not load-bearing in a way that survives scrutiny; rejected as a vehicle.

## 1. The candidate — facet-gated prototypes

**Idea (one sentence).** A prototype is a *declared attribute facet spec* **and** a free CF direction; an item activates the prototype by its CF similarity to the direction **only to the extent it matches the spec** — the spec is the grounding, and it is load-bearing because it multiplies the score. (Literature shape: Kim et al. 2014 "prototype + subspace" [from memory — verify]; the gate plays the subspace.)

**Host.** I-ProtoMF: `s(u,i) = Σ_k u_k · t*_k(i)`, `t*_k = 1 + cos(t_i, p_k)`, free t_i, free p_k.

**New parameters (per prototype k, over F fields with value sets 𝒱_f).**
- field gate `γ_kf ∈ [0,1]` (does prototype k care about field f),
- acceptance weights `w_kfv ∈ [0,1]` for v ∈ 𝒱_f (which values it accepts; default-off init).

**Activation.**
```
match_kf(i) = w_kf[x_if]                         # acceptance of the item's actual value on field f
g_k(i)     = Σ_f γ_kf · match_kf(i) / Σ_f γ_kf   # fraction of declared facets the item satisfies  (L5.1)
t*_k(i)    = 1 + g_k(i) · cos(t_i, p_k)          # CF similarity, gated                              (L5.2)
```
Mean (not product) over facets is deliberate — see the toy. If Σ_f γ_kf = 0 the prototype is declared **ungrounded** (g≡1, pure host prototype) and rendered as such.

**Hardening (ProtoTree move, C5a).** Train relaxed (sigmoids), then threshold γ, w to {0,1} and fine-tune u, t, p only; report accuracy soft vs hard. A hard spec reads as a rule: *"P7 = {colour ∈ {Dark Blue, Black}} ∧ {garment ∈ {Trousers}}"*.

**What is learned.** u, t_i, p_k (host, unchanged) + γ, w. Nothing is a buffer; no centroids; no refresh loop.

**Cold items (R5).** x_i is known → g_k(i) exact; cos undefined → replace by the prototype's running mean cos over matching warm items, ρ_k (a buffer): `t*_k = 1 + g_k(i)·ρ_k`. The cold item is placed by its attributes through exactly the objects the warm explanation uses.

## 2. Explanation vehicle — why intrinsic

- **Prototype meaning = its spec** (γ, w): discrete after hardening, directly readable, *and the only way the prototype can score an item*. No preimage problem: w acts on observed one-hot values, so each weight is identified by the items carrying that value (contrast dc04's centroid profiles, dc04 §5b pt.1).
- **Per-recommendation**: `u_k · g_k(i)·cos(t_i,p_k)` per prototype (the outer sum is still additive, for free); inside, `g_k` decomposes exactly over declared facets (L5.1 is a mean) → "P7 fired because the item is Dark Blue (✓) and Trousers (✓); CF closeness 0.6; your affinity 0.8".
- **Rudin standard**: the explanation IS the computation — the rendered rule multiplies the score; deleting a facet changes the number. Not a read-out.
- **Residual latency, disclosed and cornered**: the direction p_k is still latent ("taste *within* dark trousers"), but it is now *scoped* by a named rule; per-prototype the latent part is one cosine, and the fraction of ungrounded prototypes (γ≡0) is an honesty meter.

## 3. Aggressiveness vs dc01; what survives

- Item side: **fully free** (ID table intact, popularity/twin signal lives in t_i; none of dc01's angular tax). Capacity is only restricted through *which prototypes may see an item*.
- Prototype side: grounded by construction and load-bearing (stronger than dc01's emergent cos(e_f,p_k) reading).
- Attribute-group richness **survives and is learned**: specs are conjunctions of value sets — exactly "attribute groups", chosen by the recommendation loss.

## 4. Toy (numpy, synthetic H&M-like fields: 9 fields, sizes 50/250/21/5/130/56/30/10/20, Zipf values, independent fields — pessimistic)

Hard **product** (AND) rules of m single values, K prototypes seeded from real items: item coverage (matched by ≥1 rule) = K=50: m=1 0.99, m=2 **0.48**, m=3 0.07; K=100: m=2 0.70. → Hard AND gating with ≥2 facets leaves most of the catalogue invisible (an item matching no prototype scores the rank-inert baseline Σu_k). This is why (L5.1) is a **mean over facets with value *sets***, and why a ProtoMF-native coverage regularizer (R_batch analogue: "every item in the batch must reach g>0 on some prototype") is part of the design, not an add-on. Toy, not a result.

## 5. Hidden effects / degeneration

- **Capacity flight**: cheapest optimum is γ→0 on all prototypes (pure host). Needs grounding pressure: fixed facet budget (top-m straight-through per prototype, no loss weight) or a penalty on Σ(1−γ) — Rashomon-tunable (req. §4). Conversely w→1 everywhere = "accepts all" ungrounded; a sparsity budget on accepted values per declared field (or entropy) closes it. Two knobs → honest R8 concern (see §7).
- **Baseline mass**: non-matching items get activation 1 → Σu_k, rank-inert on the I host; at the fUfI merge the live-intercept rule (vault 2026-07-12_1815) applies.
- **Scope/coverage trade-off** (toy above); the coverage regularizer and value-set acceptance are the designed answers.
- **Hardening gap**: soft→hard may cost accuracy; report both (C5a reports none for trees; no guarantee here).
- **Correlated fields** (department ↔ garment group): a spec may name either — the rendered rule is still exact (it is what multiplies), but "why this field and not its twin" is the usual clique-level caveat (F-DC01-06).
- **Cold ρ_k** is a mean over matching warm items — popularity-weighted by nothing; fine, but disclosed.

## 6. S5 cost

Per batch item: F lookups (γ, w rows) + one mean + one multiply on the existing (B×N×K) sim tensor. Parameters: K·(F + V) ≈ 50·(9+534) ≈ 27k. No full-catalogue pass, no encoder, no refresh. Coverage regularizer = one reduction on the sim tensor (like R_batch). Essentially free.

## 7. Requirements (one line each, no ranking)

- **R1 tied**: strong-partial — the spec lives in the very activation that drives the score (no reranker); no weight-sharing object on the single-branch host (R1 clarification note applies).
- **R2 intrinsic**: strong — the rule multiplies the score; explanation = the computation.
- **R3 prototype paradigm**: strong — still "which prototypes, what they mean"; activation remains a similarity, now scoped.
- **R4 feature-grounded**: strong — explicit, learned, identifiable association (discrete spec on observed values).
- **R5 sparsity**: partial — spec-side exact for cold items, CF side via ρ_k surrogate.
- **R6 dense accuracy**: open/risky — gating can only remove capacity from the host; the bet is that learned scoping costs little (Rashomon) — must be measured soft and hard.
- **R7 vocabulary**: strong-partial — specs can be *restricted* to user-perceivable fields at zero cost (γ fixed 0 on taxonomy codes), making R7 a design switch rather than a hope.
- **R8 parsimony**: partial — one object (prototype = spec × direction), but two guard knobs (facet budget, acceptance sparsity) + coverage regularizer; defensible as "one idea with ProtoMF-style regularizers", not watertight.

## 8. Risks (honest)

Discrete search (Gumbel/ST) is fragile and init-sensitive; capacity flight is the default attractor; coverage vs specificity is a real tension (toy); hardening may regress; accuracy ceiling ≤ host by construction (only scoping is added) — the contribution is interpretability at a measured accuracy price, not a lift. Fallback if it fails: sibling (c) (exemplar push) gives free-item, identifiable prototypes at no accuracy risk but without attribute groups.

*Verified locally: C1 (push), C5a (soft→hard ProtoTree), E4a (attention not explanation), S1_SUMMARY (case-based = GC#4; Kim 2014 mention). Kim et al. 2014 BCM details: [from memory — verify].*
