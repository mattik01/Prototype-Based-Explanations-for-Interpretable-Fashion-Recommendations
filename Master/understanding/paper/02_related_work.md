# Section 2: Related Work

Paper pages 247-248 (Sections 2.1, 2.2, 2.3)

---

## 2.1 Prototypes in Recommender Systems

### Q1: ProtoMF vs. ACF (Anchor-based Collaborative Filtering)
Both ACF and ProtoMF use "representative vectors" (anchors/prototypes). The paper claims two key differences. What are they, and why do they matter?

*Hint: Think about (a) whether prototypes are shared or separate per side, and (b) what you can trace back.*

**Your answer:**
in ProtoMF both items and users are transformed into their own non shared similarity verctors to their own prototypes. can be broken down in a non mangled way to a similarity to prototypes.

**Evaluation:**
Got one of the two differences — ProtoMF has **separate** prototype sets for users and items, while ACF uses the **same** anchor set for both sides. Correct.

Second difference was vague. The paper states it explicitly: ProtoMF allows you to **trace back the contributions to the prototype vectors**, providing a direct explanation. ACF uses anchors to represent users and items but doesn't structure the score so you can isolate each anchor's contribution. In ProtoMF, the score literally decomposes into `s_1^user + s_2^user + ...`, each traceable to one prototype.

---

### Q2: ProtoMF vs. RBMF (Representative-based Matrix Factorization)
RBMF aligns latent factors with specific real users as "representatives." ProtoMF learns prototypes from scratch instead. What's the advantage of learning prototypes freely rather than tying them to actual users?

**Your answer:**
This onew I do not know please tell me it is just a more general capturing of consumption patters, the prototype in itself means something, is not just typical.

**Evaluation:**
RBMF picks **actual real users** to serve as representatives, then aligns latent factors to these specific people. Problems:

1. **Constrained expressiveness:** Prototypes can only represent patterns embodied by a single real user. If no user perfectly represents "Action+SciFi fans," you can't capture that. Learned prototypes can represent *any* consumption pattern — including combinations no real user exhibits.
2. **Sensitivity:** If the representative user's data changes, alignment shifts. Learned prototypes are stable optimized parameters.
3. **Generality:** RBMF only has user representatives. ProtoMF generalizes to both user AND item prototypes.

Think: RBMF says "how similar are you to User 42?" ProtoMF says "how similar are you to this learned consumption archetype?" The archetype is richer and more targeted.

---

### Q3: ProtoMF vs. clustering approaches
The paper mentions clustering-based and group discovery methods. It claims ProtoMF differs in two specific ways. What are they?

*Hint: One is about how users/items relate to prototypes, the other is about how the final score is computed.*

**Your answer:**
users /items are not assigned to single clusters, they just have a similarity to every "cluster" represented by its prototype. score is not just limited to in  cluster but the recommendation score uses all similarities in a linear combination, contributions are clear

**Evaluation:**
Solid. Both differences correctly identified:

1. **Soft similarity instead of hard assignment** — every user has a continuous similarity to every prototype, not discrete cluster membership. Correct.
2. **Linear aggregation into the score** — recommendation score is a linear combination of prototype similarities, enabling decomposition. Correct.

Small addition: clustering methods recommend based on **in-group/neighboring entities** (what did others in your cluster like?). ProtoMF doesn't use cluster membership — the similarities *are* the representation, and the dot product with item embeddings produces the score.

---

## 2.2 Explainability

### Q4: ProtoMF vs. post-hoc explanations
The paper contrasts with methods that use association rules, influence functions, or linear models *after* training. What's the fundamental advantage of building explainability *into* the model architecture rather than applying it after?

**Your answer:**
any external information can be used to interpret the prototypes, instead of leveraging fixed external features (not quite sure what this mean though). id need an example quite frankly.

**Evaluation:**
Mixed up with Q7. The actual distinction:

**Post-hoc example:** Train a standard MF model. It recommends Movie X to Alice with no explanation. So *after the fact*, you run a separate analysis — fit a linear model approximating MF's predictions using interpretable features (genre, director), or use LIME/SHAP to see which training interactions influenced this prediction. The explanation is **separate from the model** and may not faithfully reflect what the model actually did internally.

**ProtoMF's intrinsic approach:** The explanation *is* the model's actual computation. When ProtoMF recommends Movie X, you look at s^user — those are the real prototype contributions that produced the score. No separate analysis needed. The explanation **is** the prediction, decomposed.

**Why intrinsic is better:** Post-hoc explanations can be unfaithful — they approximate what the model did but might miss the true reasoning. ProtoMF's explanations are the actual mechanism.

---

### Q5: Why prototypes instead of attention?
Attention-based models (e.g., Barkan et al.'s persona model) can also explain recommendations by showing which inputs the model "attended to." What does the prototype approach offer that attention doesn't?

*Hint: Think about what a prototype represents vs. what an attention weight represents.*

**Your answer:**
I need help here.

**Evaluation:**
Attention says: "the model paid 40% attention to your interaction with Movie A, 30% to Movie B..."
Prototypes say: "this is driven by your similarity to the Action/Thriller archetype (0.8) and Comedy/Romance archetype (0.3)."

Key differences:

1. **Abstraction level:** Attention points to specific input items — instance-level. Prototypes represent **general consumption patterns** — concept-level. "You like Action movies" is more useful than "you watched Die Hard, which is similar to this movie."
2. **Stability and reusability:** Attention weights change per prediction. Prototype 3 always means the same thing — interpret it once, reuse across all recommendations. It's a **global** explanation component.
3. **Decomposability:** Attention explains "which of your past items mattered." Prototypes explain "which consumption archetypes drive this." More compact and arguably more informative.

---

### Q6: The paper distinguishes two ways to interpret prototypes (from classification literature). What are they?
One comes from Li et al. [35], the other from Alvarez-Melis and Jaakkola [3]. The paper says ProtoMF uses both. Describe each approach.

**Your answer:**
either interpret them by k neirest neighbours, and determine what qualities this set has, or look at the output the model produces, with artificial input, that would maximally activate a prototype

**Evaluation:**
Correct. Both methods identified accurately:

1. **Nearest neighbors (Alvarez-Melis & Jaakkola):** Find real items/users closest to the prototype in embedding space. Interpret by what neighbors share. E.g., "Prototype 6's nearest items are all Friday the 13th sequels → Horror prototype."
2. **Synthetic maximal activation (Li et al.):** Create artificial input where only one prototype dimension is maximal. See what comes out. E.g., "Synthetic user maximally similar to prototype 71 gets recommended Fugitive, Seven, Die Hard → Action/Thriller prototype."

Table 3 shows both: user prototypes via synthetic activation (left), item prototypes via nearest neighbors (right).
**especially items lend themselves well to neares neighbour** leverage common rich metadata about the items to interpret

---

### Q7: What does ProtoMF NOT use that many explainable RS methods do?
The paper explicitly states a difference from aspect-based and feature-based explanation methods (Zhang et al., Cheng et al.). What external information does ProtoMF avoid relying on, and why is this both a strength and a limitation?

**Your answer:**
external features, again I dont quite know what that means. but we need to interpret or portotypes, that is something we ened to do.

**Evaluation:**
"External features" made concrete — many explainable RS methods use **side information** (item attributes):
- Movie genre, director, cast
- Product category, brand, price
- Song tempo, key, artist

E.g., Zhang et al. might explain: "Recommended because you like **Action** movies by **Spielberg**." Requires having genre and director data.

ProtoMF uses **none of this**. Purely interaction matrix — who clicked/bought/watched what.

**Strength:** Works on any dataset without metadata. No feature engineering.
**Limitation:** Prototypes must be interpreted *after the fact* by examining neighbors. The prototype doesn't inherently "know" it represents Horror — a human must look at neighbors and conclude that.
**this is super impoRTANT STRENGTH AND DISTINCTION**

**Directly relevant to your thesis:** The paper's conclusion literally says: "we believe that including external features... might further benefit the interpretability of the prototypes." That's your Phase 4.

---

## 2.3 Fairness and Bias in Recommender Systems

### Q8: Why does the paper include a bias analysis at all?
This is a paper about prototypes and explainability. Why dedicate a whole subsection (and experiments) to gender bias? What capability of ProtoMF makes bias analysis natural here, whereas it would be harder with standard MF?

**Your answer:**
because we do not have to compare just outputs, we can compare prototypes and wheither they capture unwanted sensitive information about a user.

**Evaluation:**
Good core idea. Sharpened: In standard MF, if you suspect gender bias, you can only measure at the output level (do men and women get different quality recommendations?). You see *that* bias exists but not *where* it lives in the model.

With ProtoMF, you look *inside*: which specific prototypes are associated with which gender? You identify "Prototype 14 is male-leaning (Action/SciFi)" and "Prototype 40 is female-leaning (Romance/Comedy)." Bias is visible **in the model's internal structure**, not just outputs. And crucially, you can *intervene* — dampen specific prototype contributions (the lambda experiment in Table 5b).

---

### Q9: What's the connection between explainability and bias detection?
The paper frames bias detection as a *consequence* of having interpretable prototypes. Explain the logical chain: how do you go from "learned prototypes" to "we found gender bias in the system"?

**Your answer:**
If the prototypes are not distributed evenly across sensitive attributes like gender, then they encode the ssensitive attribute in some way, which might be unwanted.

**Evaluation:**
Correct direction. The precise chain the paper follows:

1. Prototypes are interpretable → examine what each one represents
2. Each prototype has users most similar to it (its "representatives")
3. If representatives are disproportionately one gender → prototype encodes gendered behavior
4. Tested statistically with Fisher exact tests (not just eyeballing)
5. Result: ML-1M has 36 male-associated vs. 9 female-associated prototypes
6. The system allocates more "representational capacity" to male patterns → reflects/amplifies the dataset's gender imbalance (72% male)

Key insight: **uneven prototype distribution = uneven representational capacity = one group gets more nuanced recommendations.** A concrete, measurable fairness problem visible only because prototypes are interpretable.

---

*When done, tell Claude and we'll move to Section 3: Methodology.*
