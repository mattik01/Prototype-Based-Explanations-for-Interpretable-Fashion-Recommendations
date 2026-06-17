# B5 — Kula 2015, "Metadata Embeddings for User and Item Cold-start Recommendations" (LightFM)

Mini-summary, thesis-relevant only. CBRecSys 2015 workshop paper (single author, Lyst — a fashion company). PDF + reading-guide highlights (`-annotated.pdf`) in this folder. No user PDF annotations; built from the full read + protocol vault.

## One line
Represent every user and item as the **sum of its content features' latent factors**, score by `sigmoid(q_u·p_i + b_u + b_i)`, train on (binarized) interactions — a hybrid that degrades to pure-CB on sparse data and to plain MF on dense data.

## Why it matters to us
- It is the **canonical instantiation of dc01's seed** (`feature-composed item factors`): the entity factor *is* a linear combination of feature embeddings. Reading it pins down what dc01 borrows (the mechanism) and what it adds (prototype grounding, which LightFM has no notion of).
- Closer to our **item-ranking regime** than most B-papers: implicit-feedback-flavored (binarized ≥4★, negative sampling), evaluated by **AUC** (ranking), not RMSE.
- Same domains we use: **fashion** (Lyst) + **MovieLens**.

## Most citable for (★ = strongest)
- ★ **Feature-composed factors / cold-start mechanism:** "users and items are represented as latent vectors … entirely defined by … linear combinations of embeddings of the content features." New items are scorable immediately — no retraining — as the sum of their feature vectors (§2.1–2.2).
- ★ **CB↔MF spectrum:** indicator-only features → reduces to standard MF; metadata features → extends MF; "contains both the pure CB model at the sparse end and the MF model at the dense end" (§2.3). The clean theoretical framing of hybrid-as-spectrum.
- **LightFM = a restricted Factorization Machine:** "a special case of Factorisation Machines … LightFM further restricts the interaction structure by only estimating the interactions between user and item features" (§3). The FM→LightFM lineage in one line; pair with **B4 (Rendle FM)**.
- **Additive/transparency interpretability (NOT prototype grounding):** feature embeddings carry word2vec-like semantic structure (§6.3, Table 2); recommendations justified "by the two features' similarity … the distance between their latent factors" (§6.3.3). Useful as the *weak* interpretability foil.
- Result anchor: metadata helps even in dense warm-start, and collaborative info is crucial for good feature embeddings (LightFM ≫ LSI baselines, §6.1, Table 1).

## Coverage map (relevance-flagged)
§1 intro (sum-of-features + fashion motivation) · **§2.2 model** · **§2.3 CB↔MF spectrum** · §3 related/**FM connection** · §4 datasets · §5 setup (AUC) · **§6.1 results** · §6.2 small-d · **§6.3 embeddings + justification** · §7 production ANN (skip) · §8 future (visual/CNN)

## Our insights from reading it (protocol links)
- FM/LightFM family is the right **feature-aware baseline**, not the prototype mechanism — [[2026-06-16_1824_feature-aware-baselines]]
- A hybrid feature module is only worthwhile with a **bridge back to prototype grounding**; bolt-on features (LightFM-style) don't advance intrinsic interpretability — [[2026-06-16_1838_fm-gmf-suitability-hybrid-bridge]]
- FM×prototype richer interaction direction — [[2026-06-16_1837_fmxprototype-richer-interaction]]
- Feature-motivation storyline for the intro — [[2026-06-16_1858_thesis-intro-storyline-feature-motivation]]

## Caveats when citing
- **Metric mismatch:** AUC (pairwise rank, no cutoff), not our `hit_ratio@10` / NDCG@K. Don't transfer absolute numbers.
- **"Interpretability" here is additive feature transparency**, not prototype grounding — strictly the shallow end of the Rudin axis. It is an *ancestor/foil*, not a target.
- Additive (sum-of-features) composition only — no learned interaction among features (that's the FM/NFM step beyond it).
- Workshop paper, single dataset pair, 2015 — cite for the *mechanism and framing*, not as a SOTA result.
