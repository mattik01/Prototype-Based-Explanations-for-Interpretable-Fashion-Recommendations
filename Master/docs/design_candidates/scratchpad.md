# Design-Candidate Scratchpad

> Free-form idea inbox for the `/design-candidate` protocol (v1.2). Jot anything
> while reading — seed ideas, extensions, constraints, doubts, "what if X" — as
> quickly and roughly as you like; one `###` block per entry, date optional but
> helpful.
>
> What the protocol does with it: **Step 0** reads it, **Step 1** gives every
> open entry a one-line disposition (seed / shapes the candidate / not this
> cycle / answered inline) — nothing is silently ignored — and **Step 6**
> appends an indented `↪ dcNN` marker under consumed entries. Your text is
> never edited or deleted. Entries without a marker are still open.
>
> The protocol may also **add** entries here (v1.3, Ground rule 9): adjacent
> ideas or variants that surface mid-cycle but can't be pursued now (R8) are
> captured as `### [captured dcNN] <title>` blocks instead of being discarded.
> They are swept in Step 1 like any open entry; they're distinguishable from your
> own entries by the `[captured dcNN]` tag.

---

## Entries

### Attach feature profiles to existing CF prototypes (2026-06-11)
Instead of building separate "feature prototypes," attach feature information directly to the existing CF prototypes — one prototype = CF vector + feature profile (keeps the prototype set unified, features make existing prototypes nameable). Whether this or separate feature-prototype spaces is preferable is to be decided later. Caveat to remember from Rudin GC#4: this design reintroduces whole-to-whole comparison, so the part-based prototypes idea (compare on feature subsets, Kim et al. 2014 for tabular data) becomes the relevant refinement there — usable as future work. With separate feature-prototype spaces this concern mostly disappears, since similarity-to-all-prototypes already breaks an entity into facets.
  - ↪ dc04 2026-07-07: shaped Step 1 (declared convergence) — independent ideation landed in the same region (feature meaning at the prototype level, CF untouched); dc04 realizes it by construction (prototype = convex combination of attribute-value centroids) rather than by attaching a profile + binding loss; unified-set question answered: one set. The profile+loss realization stays open as `[captured dc02]`.

### Feature-injection fork: prediction/representation vs prototype-space prior (2026-06-16)
Design fork for how features enter feature-aware ProtoMF, mirroring the two slots in S7 Eq 25 (Chen et al. 2020, §4.1.1). (A) Inject into the prediction/representation — features feed directly into the entity representation or score (the obvious route, analogous to the additive likelihood terms α/β/γ). (B) Inject as a prior/regularizer on the latent space — features shape the prototype space without touching the score directly: pull feature-similar entities toward similar prototype-similarity profiles, or regularize prototypes to align with feature structure (analogous to p(W|X)p(H|Y)). ProtoMF already has structured-latent-space regularizers (R_proto, R_batch), so (B) is a natural fit. S7 flags (B) as an unexplored gap in the *linear* category specifically — combining additive feature terms with attribute-aware priors is "not yet observed" — so it's a citable novelty angle. Not mutually exclusive; a candidate could do both. To decide in a design cycle.
  - ↪ dc02 2026-07-07: shaped Step 1 — route A (representation) taken, in its strongest form (representation = attributes); route B (prior/regularizer) remains open for a future cycle.

### Three ways attributes can enter a CF model — add / interact / regularize (2026-06-16)
A design-shaping fork for feature-aware ProtoMF, from S7 §4.1 (the three DMF sub-types). Attributes/features can enter a model in three distinct places; future candidate designs should pick deliberately among these (or combine). Refines the earlier prediction-vs-prior injection fork into three concrete mechanisms.

**1. ADD — attributes add corrections to the score (linear, §4.1.1).** Prediction = taste-match + a learned correction per attribute. Win: simple, robust; great cold-start; degrades gracefully when only one side (user OR item) has attributes. Caveat: purely additive — cannot capture "feature A interacts with feature B"; each attribute contributes independently.

**2. INTERACT — attributes multiply/interact in the score (bilinear, §4.1.2).** A grid/table pairs user-features with item-features ("who likes what"). Win: captures cross-side user×item interactions the additive model structurally can't; the interaction is itself a bilinear form, same shape as ProtoMF's user·item dot product (prototypes = interpretable low-rank stand-in for the dense interaction matrix A). Caveat: needs BOTH user and item attributes — the term collapses if one side is missing. Directly relevant to H&M asymmetry (rich item/article features, thin customer features) → argues against a symmetric user×item feature interaction; steer asymmetric.

**3. REGULARIZE — attributes as gravity on the latent space, not in the score (similarity, §4.1.3).** Score stays plain MF; attributes pull attribute-similar entities toward similar latent vectors (a prior/regularizer). This is the literature realization of the "features as a prior on the prototype space" route (route B) — e.g. a feature-similarity pull could regularize prototype-similarity profiles rather than raw embeddings. Win: great cold-start — a user with no ratings still gets dragged toward attribute-similar users, so their vector isn't blank; accepts any attribute you can define a similarity over (even graphs/networks via kernels), without forcing them onto the rating scale. Caveat 1: you must choose the similarity function — hand-designed, not learned, so a bad choice bakes in a bad assumption. Caveat 2 (H&M): a full user×user table is Nu×Nu; ~1.3M customers makes the literal matrix astronomically large → any similarity-style idea on H&M needs a sparse/approximate or prototype-mediated version.
  - ↪ dc02 2026-07-07: shaped Step 1 — dc02 deliberately sits outside the add/interact/regularize score-level trio (features enter at representation level); REGULARIZE route stays open (rejected as standalone seed: no intrinsic read-out, weak R5).

### Attribute-source prioritization is interpretability-driven; intrinsic > post-hoc (2026-06-16)
Refines the prioritization thinking. **Item attributes = primary integration target** (rich on H&M; item prototypes ground directly in feature space → intrinsic prototype meaning). **User attributes = candidate-dependent**: don't rule out integrating them into the recommendation algorithm; when a candidate uses them, prefer intrinsic integration over post-hoc so they shape explanations by-design (thin H&M user features may cap how rich this gets). **Context = drop for now** (per-event, doesn't map to entity prototypes) unless cleanly convertible (S7 §3.2.3/3.2.4 context↔side-info conversion is the escape hatch). Key reframing from reading `utilities/explanations_utils.py`: ProtoMF's prototype *grounding* (what a prototype means) is currently **post-hoc** — `get_top_k_items` assigns meaning after training via nearest items; user prototypes are doubly indirect (read via the items a maximally-close user would be recommended) — while per-recommendation `weight_visualization` is **intrinsic** (decomposes the model's real score). The extension's core interpretability win = moving prototype grounding post-hoc → intrinsic via feature-space prototypes. Hence design principle: avoid post-hoc-only; prefer by-design integration. Logged to protocol (positioning node + design-principle node).
  - ↪ dc02 2026-07-07: shaped the candidate — its core reframing (intrinsic grounding via feature-space prototypes) is dc02's headline; item-attributes-primary followed.

### Candidate-selection sequencing: item-only → unified → separate sets (2026-06-16)
Proposed ordering for candidate rounds: (1) start with **item-attribute-only** candidates; (2) extend them to treat **item + user attributes as one undifferentiated feature set** (source-agnostic, leaning on S7 §3.2's claim that attribute sources are interconvertible/unifiable) to see if/which still work; (3) candidates that leverage user and item attributes as **separate feature sets** — mirroring ProtoMF's own `user_proto` / `item_proto` / `user_item_proto` progression. Scope caveat: this is a master's thesis, not a dissertation or a series of papers — the full three-stage progression may not be feasible; treat stage 1 as the must-have and stages 2–3 as stretch.
  - ↪ dc02 2026-07-07: shaped scoping — stage-1 item-attribute-only; user branch untouched (S2 mirror remains open).

### Design caution: two axes (grounding vs scoring) + Rudin axis (2026-06-16)
Keep two orthogonal axes straight when extending ProtoMF. **Axis A = representation/grounding**: how features get INTO / become prototypes — the thesis core. **Axis B = scoring/readout**: how prototype-similarities combine into a score (diagonal dot-product vs richer interaction). Don't let Axis-B cleverness (fancier interaction heads) masquerade as Axis-A feature work — they're independent; a more expressive scorer does not ground prototypes. **Rudin axis**: a change that is more accurate but less simple/intuitive is a COST on interpretability, not a win, unless it pays for itself in accuracy. Gut-check every candidate: does it advance Axis A (grounding / intrinsic interpretability), or just Axis B (expressiveness)?
  - ↪ dc02 2026-07-07: gut-check applied in Step 1 — pure Axis A (scorer untouched); Rudin axis: trades expressiveness for simplicity.

### Hybrid desideratum: bridge between added module and prototype grounding (2026-06-16)
For any hybrid (a separate feature module alongside ProtoMF), a **bridge** between the added module and the grounding of the prototypes is desirable. A bolt-on feature module that doesn't feed back into grounding the prototypes just adds features without advancing intrinsic interpretability (the actual goal).
  - ↪ dc03 2026-07-07: shaped Step 1 — the added content channel (frozen image encoder) satisfies the bridge structurally: its output IS the prototype space, grounded by push; no bolt-on without grounding feedback.

### [captured dc02] Loss-enforced grounding: learned attribute profile per CF prototype + consistency loss (2026-07-07)
Concrete mechanism refinement of the user's 2026-06-11 "attach feature profiles to existing CF prototypes" entry, surfaced while shortlisting dc02 seeds but not pursued (dc02 chose by-construction grounding instead). Keep prototypes and item embeddings in CF space untouched (R6-safe); attach to each prototype k a *learned* attribute profile a_k (vector over feature values); add ONE consistency loss binding the CF activation sim(item i, prototype k) to the profile prediction a_k·x_i (x_i = item's multi-hot attributes). Bidirectional effect: profiles become faithful read-outs AND features regularize the CF prototype space (touches the S7 §4.1.3 REGULARIZE route). Cold items activate via the surrogate path a_k·x_i (R5). Open fidelity question: the score does not flow through a_k, so faithfulness holds only as far as the loss binds — a devil's-advocate target. One idea at user_item_proto granularity; strong seed for a future cycle.

### [captured dc02] Grounding-enforcement spectrum as a thesis narrative (2026-07-07)
The candidate set is starting to span a clean 3-point axis of *how strongly prototype feature-grounding is enforced*: unenforced/emergent (dc01 — shared geometry, read-out only) → loss-enforced (the captured entry above) → by-construction (dc02 — prototypes live in attribute space). If all three mature, this is a ready-made thesis narrative and ablation ladder for the fidelity-vs-accuracy tension (requirements §4): same host, same read-out shape, increasing grounding strength. Cross-candidate framing device — for the user's ranking/synthesis, not a candidate itself.
  - ↪ dc03 2026-07-07: answered inline (Step 1b) — dc03 sits OFF this spectrum: exemplar/projection grounding is a different KIND of grounding (grounding vocabulary: visual exemplar vs attribute profile), not a different strength; the spectrum narrative stays clean and dc03 adds an orthogonal contrast axis to it.

### [captured dc03] Part/patch-level visual prototypes (garment parts) (2026-07-07)
Refinement of dc03 deferred under R8: ProtoPNet's actual granularity is *patches* — prototypes as garment PARTS (collar, sleeve, print) over spatial feature maps, pushed onto real image patches, giving "this collar looks like that collar" explanations. Directly answers the Rudin-GC#4 whole-to-whole caveat (user entry 2026-06-11) that whole-garment exemplars reintroduce. Cost: spatial feature maps ((H×W×D) per item, patch-argmax push, heavier buffers) — real machinery, its own candidate. Prerequisite: dc03's whole-item version working.

### [captured dc03] Multimodal frozen content space (image ⊕ attributes) for exemplar prototypes (2026-07-07)
Surfaced while weighing dc03's encoder options: the frozen content vector v_i need not be image-only — concatenating the image embedding with the attribute multi-hot (or an attribute-grade encoder output, H2-style) gives exemplar prototypes grounded in what the user sees AND what the catalog says, with the same push mechanism and read-out. Bridges dc02's and dc03's entry points; ensemble-risk under R8 must be argued (one representation, one mechanism — arguably still one idea). Future seed, competes in the next sweep.

### [captured dc04] Differentiable (live) centroids = the REGULARIZE route with a read-out attached (2026-07-07)
Variant of dc04 surfaced during ideation and not pursued (dc04 uses detached, periodically-refreshed centroid buffers for S5 cleanliness). If the attribute-value centroids c_v are instead computed *differentiably* from the live item embeddings (EMA or in-batch means kept in the graph), the score gradient flows through prototypes → centroids → **into the item embeddings of same-attribute items**, pulling attribute-similar items toward similar prototype-similarity profiles. That is precisely the S7 §4.1.3 REGULARIZE route (features as gravity on the latent space) — twice rejected as a standalone candidate for lacking an intrinsic read-out — except here the read-out exists (dc04's profiles/shares), so the old objection dissolves. Cost: per-step centroid work + trickier dynamics (moving anchors under the prototypes). One idea; a future cycle could take "dc04 with live centroids" as a deliberate REGULARIZE-entry candidate and measure the gravity effect against detached-buffer dc04.

<!-- Template (copy, or just write — format is not enforced):

### <short title> (2026-MM-DD)
<the idea / consideration / question, any length, any roughness>

-->
