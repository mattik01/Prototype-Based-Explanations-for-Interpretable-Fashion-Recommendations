---
date: 2026-07-13
time: "20:33"
phase: 4
---
# What remains latent after grounding — still MF, and latency becomes a measured quantity

*(Verbatim-capture entry — thesis-framing explanation worked through during the thesis-outline session, user requested protocol. Destined for Background (§2.1/2.2) and echoing into the results story. Belongs beside [[2026-07-11_2201_latent-directions-meaning-prototype-recoordinatization]].)*

## The question (Matteo, near-literal)

"We are not doing matrix factorization anymore, are we? We are getting closer to CF/CBF if we eliminate so many latent factors. What is still latent — affinity vectors? And those, if shared via a double tie like ProtoMF — are they even latent, are they not just *learned*?"

## The answer, as built up

**1. Is it still matrix factorization? Yes, in the sense that matters.** MF's defining property: the score is an inner product of a user vector and an item vector — the interaction matrix is approximated by a low-rank product. That stays true through the whole lineage (ProtoMF, fI, fU, the merge). What the lineage changes is not the factorization but the **parameterization of the factors**: classic MF gives each entity a *free* vector; fI makes the item factor a *recipe over learned ingredient vectors* — q_i = Σ_f e_f (+ e_ID) — where the recipe (which attributes the item has) is **observed** and only the ingredients are fitted. That is the SVDFeature/LightFM move, and that literature never stopped calling itself matrix factorization. Honest description: **attribute-structured MF**. (Consequence: "prototype-based matrix factorization" in the working thesis title stays legitimate.)

**2. Latent vs learned vs anchored — three different things.** "Latent" classically = unobserved *and semantically unassigned*. "Learned" = fitted by optimization. Everything in our models is learned; not everything is latent anymore. After grounding, the embedding e_red for attribute "red" is learned, but it is **anchored to an observed symbol** — its *coordinates* are latent, its *identity* is not. The four parameter kinds:

- **Attribute/feature embeddings** — learned, anchored. Semantically no longer latent.
- **Prototypes** — learned, unanchored... *until* the grounded read-out names them. This is precisely the naming gap closing.
- **ID rows** (item-ID in fI, user-ID in fU) — learned AND fully latent. The honest residuum.
- **The free side** (u* in plain fI) — fully latent, disclosed as such in the four-component inventory.

**3. The punchline (thesis-quotable):** In classic MF, "what is latent?" has the answer "*everything*". In our models it becomes **an empirical, per-model number** — the feature-explained fraction and score-share disclosure literally measure the remaining latent mass. The lineage does not eliminate latency; it **corners it into the ID rows and quantifies it**.

**4. The double tie changes none of this.** Weight sharing constrains parameters (fewer independent ones, compromised double duty — [[2026-03-18_1234_double-tie-explainability-tradeoff]]), but tied-and-learned is orthogonal to latent-vs-anchored. A tied vector is just a latent vector with two jobs.

**5. Drift toward content-based filtering? Toward a hybrid, deliberately — but the collaborative core is untouched.** What makes a method collaborative is *what signal trains it*, and every parameter here is still fitted purely to interaction data. The noid arms are the extreme case: content-**parameterized**, collaboratively **trained**. This is also exactly why LightFM is the closest baseline and not decoration.

[[phase-4]] [[explanations]] [[thesis-writing]]
