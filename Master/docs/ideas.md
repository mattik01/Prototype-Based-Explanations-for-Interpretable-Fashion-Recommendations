# Ideas File
Promising ideas to explore in future phases. Each entry is recorded thoroughly so we can evaluate it when the time comes.

---

## 1. Cold-Start Evaluation for Feature-Aware Prototypes

**Phase relevance:** Phase 4 (after feature-aware architecture is implemented)

**Core insight:** Feature-aware prototype models can assign meaningful prototype activations to *new items with no interaction history*, purely from their metadata (category, color, department, etc.). Standard CF-based ProtoMF cannot do this — an unseen item has no learned embedding.

**Why this matters for the thesis:**
- Cold-start is one of the most-cited weaknesses of CF methods. If our feature-aware extension addresses it, that's a strong, concrete contribution beyond "better accuracy."
- The H&M dataset is ideal for this: fashion has high item turnover (new products constantly), so cold-start is a real production problem.
- It directly motivates *why* features matter — not just "features help accuracy" but "features enable recommendations where CF literally cannot."

**Proposed evaluation approach:**
1. **Held-out new items:** After training, select items that appeared in test but had very few interactions in training (e.g., < 5 or < 10). Evaluate HR@10 / NDCG@10 on these items separately.
2. **Simulated cold-start:** Remove a subset of items from training entirely (but keep them in test). CF models will produce random predictions for these; feature-aware models should still produce meaningful ones.
3. **Compare:** CF-only ProtoMF vs. feature-aware ProtoMF on the cold-start subset specifically.

**Explanation angle:** For cold-start items, the explanation is *purely* feature-driven: "Recommended because this item's attributes (Trousers, Dark, Menswear) match prototypes you've engaged with." This is a clean, interpretable story.

**Effort estimate:** Low-to-moderate. Evaluation code is a filtering step on top of existing eval. No new model training needed — just re-evaluate existing checkpoints on the cold-start subset.

---

## 2. Explanation Quality Evaluation — Approaches to Research

**Phase relevance:** Phase 4.5 (explanation analysis)

**Problem:** The plan says "compare CF-only vs. feature-aware explanation quality" but doesn't define how to measure it. This is a known open problem in explainable AI.

**Approaches to investigate (from lightweight to heavyweight):**

1. **Qualitative case studies** — Pick 5-10 (user, item) pairs, show prototype activations, interpret manually. Easiest, acceptable for a master's thesis.

2. **Prototype coherence / purity metrics** — For each prototype, look at its top-K most activated items. Measure how homogeneous they are in feature space (e.g., do items closest to prototype #3 all share a category?). Higher coherence = more interpretable prototypes.

3. **Fidelity metrics** — Do the explanations actually reflect the model's reasoning? Measure correlation between prototype activation strength and recommendation score.

4. **User study** — Show explanations to real users, ask if they find them helpful/accurate. Gold standard but expensive. Probably out of scope unless supervisor requests it.

**Decision:** Defer to Phase 4. Research existing explainable RecSys evaluation literature at that point. At minimum, do (1) and (2).

---

## 3. User-Side Feature-Aware Prototypes

**Phase relevance:** Future work / Phase 4 stretch goal

**Core insight:** ProtoMF's architecture is symmetric — user prototypes and item prototypes. The current plan only adds features on the item side. But H&M's `customers.csv` provides user metadata (age, club membership, fashion news frequency, postal code) that could feed into user-side feature prototypes.

**What it would look like:**
- Item prototypes operate on item features (color, category, department) — "what kind of product is this?"
- User prototypes operate on user features (age, membership, location) — "what kind of shopper is this?"
- Explanation: "Recommended because you're similar to young club-member shoppers (user prototype) who like dark-colored trousers (item prototype)."

**Why defer:**
- Roughly doubles Phase 4 implementation (separate `FeatureEncoder` for users, new `ft_type` branches for every user-side combination).
- H&M user features are weaker than item features: age has missing values, postal code is high-cardinality/noisy, club membership is just binary. Item features are richer and more obviously useful for fashion.
- Thesis story is cleaner with one focused contribution (item features).

**Decision:** Keep scope to item-features-only for the core thesis. Design the Phase 4 architecture so user features could be added later without refactoring. Mention in thesis as "future work" with a concrete sketch. If item-side pipeline turns out easy and time allows, revisit.

---

## Open Questions (to resolve when approaching the relevant phase)

These came up during plan review but are too early to answer now. Revisit when the phase arrives.

- **Thesis narrative framing:** Is the primary contribution accuracy improvement or explanation quality? Leaning toward explainability-first framing. Decide during Phase 4 when results are in.
- **External baselines:** Should we compare against content-aware CF methods beyond the paper's 5 models (e.g., LightFM, DeepFM)? Decide during Phase 3/4.
- **H&M product images:** Not out of the picture. Could be used in the model (CNN feature extractor) or purely for explanation visualization in the thesis. Decide during Phase 3 exploration.
- **Literature review scope and timing:** Thesis needs Related Work chapter covering explainable RecSys, content-aware CF, prototype networks in other domains. Need a reading list / literature search step — decide timing during Phase 2.
- **H&M preprocessing specifics:** K-core threshold, temporal window, deduplication strategy for repeat purchases — decide during Phase 3 exploration after seeing the data.
- **Explanation quality evaluation method:** Qualitative case studies, prototype coherence metrics, fidelity metrics, or user study? Research and decide during Phase 4.

---
