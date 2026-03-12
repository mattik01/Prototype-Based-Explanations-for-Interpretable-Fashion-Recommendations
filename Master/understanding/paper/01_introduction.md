# Section 1: Introduction & Motivation

Paper pages 246-247 (Section 1)

---

## Conceptual Questions

### Q1: What problem does ProtoMF solve that standard MF doesn't?
Standard MF gives you user/item embeddings and a dot product score. What's missing that motivates this work?

**Your answer:**
A way to make the recommendation explainable beyond just based on behaviour of similar users (based on overall interaction patterns)

**Evaluation:**
Core idea (explainability gap) is correct but incomplete in two ways:

1. Standard MF isn't even explainable at the "similar users" level. The embedding dimensions are completely opaque latent factors — dimension 7 of a user vector has no human-interpretable meaning. You can't point to *anything* in an MF prediction and say "this is why." The dot product mixes all dimensions into a single number with no decomposition.

2. ProtoMF doesn't just add explainability — the paper claims it also *improves accuracy* over MF. This matters because there's usually a trade-off (interpretable models sacrifice performance). ProtoMF claims both. That's the "effective AND explainable" in the title.

**More complete answer:** Standard MF produces opaque predictions with no way to explain *why* an item is recommended. ProtoMF makes predictions decomposable into interpretable prototype contributions, while also improving accuracy.

---

### Q2: What does "prototype" mean in this context?
The paper defines it as a learned vector representing general consumption characteristics. How is this different from (a) a cluster centroid, and (b) a regular embedding vector?

**Your answer:**
cluster centroid is geometrical, while a prototype is just some representitive, and can be an element of the cluster or some other artificat.

**Evaluation:**
Partially right but misses key distinctions:

**(a) vs. cluster centroid:** Correct that a centroid is the geometric mean. But the bigger difference: clustering does hard or soft *assignment* — each user belongs to one (or few) clusters. ProtoMF prototypes don't assign users to groups. Every user gets a similarity score to *every* prototype, producing a full similarity vector. A user can be 0.8 similar to prototype 3 and 0.7 similar to prototype 12 simultaneously. No assignment step.

**(b) vs. regular embedding:** Not addressed. A regular embedding (as in MF) is a unique vector per entity — user 42 gets their own `u_42`. A prototype is a *shared* vector that multiple users are compared against. There are L^u prototypes for N users, where L^u << N. Prototypes are shared reference points; embeddings are individual representations.

---

### Q3: Why is the linear decomposition of the score so important?
The paper repeatedly emphasizes that the affinity score is a *linear combination* of prototype contributions. Why does this matter for explainability — and what would break if they used a non-linear combination?

**Your answer:**
FOr non linear combination, the contribution of the prototypes towards the affinity scores becomes unclear and unproportional. with it being linear we can decompose it.

**Evaluation:**
Direction is right — linearity enables decomposition. More precisely:

With a linear score like `U-score(u,t) = sum of s_l^user`, each `s_l^user` is one prototype's contribution. You can literally say "prototype 3 contributed 0.6, prototype 7 contributed 0.3." The contributions *add up* to the final score. You can rank them, show top contributors, and explain the recommendation.

If you passed prototype similarities through a neural network, you'd lose this. A non-linear function `f(s_1, s_2, ..., s_L)` creates interactions between prototypes — prototype 3's contribution would *depend on* prototype 7's value. You could no longer isolate individual contributions.

This is why the paper also uses a simple **sum** (not weighted) for UI-ProtoMF's final score (Eq. 12): `UI-score = U-score + I-score`. They deliberately avoid any non-linear combinatorial function.

---

## Framing Questions

### Q7: The paper mentions EU regulations (AI Act, Digital Services Act). Why include this?
This is a framing choice. What does it signal about the intended audience and impact beyond pure accuracy?

**Your answer:**


---

### Q8: The three contributions are: (a) the model, (b) experiments, (c) gender bias analysis. Why is (c) a core contribution?
Why include bias analysis as a core contribution rather than a side analysis? What does this tell you about the research direction?

**Your answer:**


---

*When done, tell Claude and we'll move to Section 2: Related Work.*
