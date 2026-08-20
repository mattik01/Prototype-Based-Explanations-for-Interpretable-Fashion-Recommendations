---
date: 2026-08-15
time: "16:39"
phase: 4
---
# Porting F5's neighbourhood purity (P_m): addressability ≠ metric, and what "ground-truth category" even is for us

*(Verbatim-capture entry from the F5 reading session (`F5_loveland_2026_preserving_item_semantics`, arXiv:2608.07816 — reading list §F). Two questions Matteo raised while reading were worked through interactively and are to be remembered when lessons are taken from the paper. The read is still in progress — nothing here is an adoption decision beyond the port-plan scoping at the end.)*

## Background: what P_m is and why we care

F5 defines **neighbourhood purity P_m**: for each embedding (their level-0 SID tokens), take its m nearest neighbours and compute the fraction sharing its ground-truth category; average. High P_m = the space's local geometry is organized by semantics. Their headline use: after standard random init + training, P₁₀ sits at 0.24–0.37 — barely above the random-neighbour floor and far below the source semantic space (0.58–0.93) — i.e. training never recovers the discarded semantics, even after a dedicated re-grounding stage (CPT). P_m is the most directly portable of F5's three instruments (P_m, R²_q(T), σ_geo) and a candidate row for the Ch 10 fidelity family / SC.8 geometry checks.

## Question 1 (Matteo): isn't P_m near-vacuous for us — our prototypes LIVE in ground-truth category space, so shouldn't neighbours share categories by construction?

Answer worked through: **no — composition guarantees *addressability*, not the *metric*.** Composition fixes who is built from what (every fI item embedding is a sum of attribute rows, so we can always ask cos(e_f, p_k) and get an answer). But the rows are **learned vectors**, and which items end up geometrically near which is decided entirely by what training does to the rows' norms and directions. Four concrete ways the "guarantee" fails:

1. **Training can demote a field** — shrink the product-type rows, inflate the colour rows → nearest neighbours are colour-mates, not category-mates. P_m(category) drops, legitimately: the learned metric stopped caring.
2. **Different values can converge** — nothing stops e_denim ≈ e_jersey; then neighbourhoods mix even though every embedding "contains" its correct row.
3. **The ID row can dominate** (ids arm) — if free per-item rows carry most of the norm, proximity decouples from attributes entirely (the ID-residual / feature-explained-fraction concern in geometric form).
4. **The F5 scenario proper** — training organizes the dominant axis around popularity, routed through ID rows or the attribute rows themselves; neighbours become "similarly popular items."

The only structural guarantee is degenerate: identical signatures coincide (twins, noid arm). One shared value buys one shared *summand*; whether that summand matters for distance is precisely what training decides.

**Consequence — what P_m measures shifts.** For F5 the semantics were discarded at init and P_m measures *recovery*. For us the semantics are structurally present at t=0 (composition induces an overlap-kernel geometry even with random rows) and P_m measures **retention vs demotion**: did training keep attribute structure dominant in the metric or reorganize it around something else? Neat symmetry with their Figure 2: their green (source-space purity) vs blue (post-training) bars = our "composed geometry at init" vs "after training."

**Concession Matteo's intuition earns:** composition does give a **structural floor** — our absolute P_m is inflated relative to a free-embedding model and must never be quoted like F5's raw numbers. The instrument is only meaningful in *relative* arms: trained vs init, fI vs host, ids vs noid — plus a shuffled-label control for the floor. For **prototypes** proper, nothing is guaranteed at all (they are free vectors; composition constrains items, not prototypes — empirically the S0.7 bundle parked on the unpopularity pole with a category-mixed neighbourhood). Prototype-neighbourhood purity is therefore the least trivial variant: it measures whether the things we name are geometrically coherent enough to deserve names. Note: F5 uses L2 neighbours; we would use cosine (the only geometry our model reads).

## Question 2 (Matteo): what is "ground-truth category" supposed to BE for H&M? All five canonical columns contributed; none is privileged.

Findings worked through:

- **F5 had the same problem and dodged it.** Their embedding input is multi-feature (title, brand, category path, price — their Fig. 6), but the purity/semantic target uses *only* the category path, without argument. Amazon/MovieLens/Steam happen to have one conventionally privileged taxonomy column; H&M doesn't. The discomfort is not a porting artifact — it is a real weakness of the metric that our setting exposes.
- **Reframe: ground truth is the claim being tested, not a property of the data.** Our explanations make per-field claims (names drawn from department, product_type, section, colour_group, graphical_appearance). The fully honest port is a **per-field purity profile** — five numbers per model — where each tests "the learned metric respects this field well enough that names from it are geometrically meaningful." The ambiguity then flips from bug to readout: which fields organize the metric = which fields explanations may legitimately lean on (e.g. high P_m(colour_group) + low P_m(product_type) ⇒ product-type words in prototype names are decoration).
- **Cardinality floors are load-bearing.** The five fields differ wildly in cardinality; purity floors rise as cardinality falls. Every per-field number must be reported as lift over a shuffled-label / random-neighbour floor (F5's gray bars, done per field), or coarse fields win for free.
- **External (non-circular) targets are thin for H&M — and for a principled reason:** the excluded columns were excluded *because* they are near-duplicates (garment_group ≈ department, FD 0.999) or IDs in disguise (product_code) or leakage-prone (price_band) — so the "held-out column as external semantics" trick mostly evaporates. Genuinely external options are heavier: a text embedding of `detail_desc`, or the **frozen image embeddings as an evaluation instrument** — a small clean dc03 salvage (images never enter the model, sidestepping everything dc03 was abandoned for). Both optional depth. One free non-circular contrast: the **host** model's free embeddings never saw any field, so every field is external there — host-vs-fI purity profiles cleanly show what composition buys.
- **ml-1m: genres are natural but popularity-contaminated** (token-mass coupling ρ=0.672, vault 2026-07-12_2208): low genre purity could mean "semantics lost" OR "popularity dominates and popularity is genre-correlated" — P_m alone cannot tell which. F5's **R²_q two-target decomposition** (T_pop and T_sem against the *same* leading PCs) is the adjudicating instrument; P_m is the legible one. On ml-1m both are needed, and the standing popularity-coupling disclosure line applies to any genre-purity number.

## Port-plan decision (Matteo, 2026-08-15)

When P_m is ported: **start with ONE ground-truth column, following the paper** (which column = open, chosen at design time); the **5-field per-field profile is an option, not the default**; and the **single-column arbitrariness criticism is itself to be levied and investigated** — i.e. treated as a methodological point in its own right (F5 chose a column silently; whether/how much the choice matters is checkable and possibly reportable). Instruments remain candidates only until the F5 read completes; scrutiny-evidence-not-thesis applies to anything measured.

Cross-refs: [[2026-07-15_1112_prototype-collapse-is-a-self-built-popularity-bias]] (the finding F5 corroborates); [[2026-07-11_1643_canonical-feature-set-price-productcode]] (the five fields and why the excluded ones can't serve as external targets); roadmap parallel thread "Evaluate F5" in `Master/temp/NEXT_STEPS_roadmap_2026-07-12.md`.

[[explanations]] [[evaluation]] [[decisions]] [[paper-understanding]] [[phase-4]]
