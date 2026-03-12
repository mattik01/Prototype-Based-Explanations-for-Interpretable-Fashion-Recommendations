# Section 3.1: U-ProtoMF

Paper pages 248-249 (Section 3.1, including Section 3 preamble)

---

## Preamble & Setup

### Q1: What is the input to all ProtoMF models?
The paper defines the data available. What exactly does the model have access to, and what does it explicitly NOT have access to? Why does this matter for your thesis extension?

**Your answer:**
Just implicit interactions no rating, no stars no item characteristics, just consumption essentially right? This matters for my thesis, because in the h&M we also jjust have interactions like, that meaning the buy signal. but we also would like to intergrate item features, the eaxct point and method of integrating it remains to be determined.

**Evaluation:**
Correct. The model only has implicit interaction data I = {(u_i, t_j)} — binary "user interacted with item." No ratings, no metadata, no features. Thesis connection is spot on — ProtoMF's conclusion explicitly calls out feature integration as future work, which is Phase 4.

One small precision: the paper formally defines it as a set of (user, item) pairs. No explicit negatives either — negatives are sampled during training.

---

### Q2: Where does ProtoMF sit relative to standard MF?
The paper says ProtoMF "builds on top of" MF. What does it keep from MF, and what does it change?

**Your answer:**
embeddings for users and items, for proto-U it keeps item embeddings, but user embeddings become a vector of prototype similarities.

**Evaluation:**
Correct. It keeps the embedding lookup from MF (both users and items still get d-dimensional embeddings), but for users, the raw embedding **u** gets transformed into **u*** via prototype similarities before computing the score. The item side stays as a plain embedding in U-ProtoMF.

Key nuance: **u** still exists as a learned embedding. It doesn't disappear — it's the input to the prototype similarity computation. The model still learns a unique vector per user, it just doesn't use it directly for scoring.

---

## The Prototype Transformation

### Q3: Walk through the transformation from user embedding to user prototype representation.
Given a user u with embedding **u** ∈ ℝ^d and L^u prototypes, describe step by step what happens. What is u* and what are its dimensions?

**Your answer:**
similarities computed for the user to each prototype, through shifted cosine similarity. dimnesions is a vector of #of prototypes

**Evaluation:**
Correct direction but could be more precise. The full step-by-step:
1. Look up user embedding **u** ∈ ℝ^d
2. Compute sim(**u**, **p**_l^u) for each of the L^u prototypes (each **p**_l^u ∈ ℝ^d)
3. Stack these into **u*** = [sim(**u**, **p**_1^u), ..., sim(**u**, **p**_{L^u}^u)] ∈ ℝ^{L^u}

**Important dimensional shift:** **u** lives in ℝ^d, but **u*** lives in ℝ^{L^u}. The representation changed dimensionality. And critically, **the item embedding t must also be in ℝ^{L^u}** for the dot product to work (Eq. 2). So in U-ProtoMF the item embeddings are L^u-dimensional, not d-dimensional.

---

### Q4: Why shifted cosine similarity?
The similarity function is sim(a,b) = 1 + (a^T b)/(||a||·||b||). Standard cosine ranges from [-1, 1]. Why shift it to [0, 2]? What would go wrong if they used standard cosine?

**Your answer:**
more naturally interpretable as positive values (similarity to prototype), also used with weights later, where negative values would be bad.

**Evaluation:**
Good intuition, incomplete. The interpretability argument is correct. The more concrete reason: **u*** is element-wise multiplied with **t** to produce s^user, and then summed into U-score. If similarities could be negative, a negative similarity × negative item weight = positive contribution. That means "user is dissimilar to this prototype" could *increase* the score — which breaks the interpretability story. You couldn't say "prototype 3 contributed positively" anymore because the sign interactions become confusing.

With all values in [0, 2], every s_l^user ≥ 0 and higher always means "more contribution." Clean, monotonic, interpretable.

---

## The Score Computation

### Q5: Write out U-score(u, t) and explain what the ⊙ operator does.
The paper gives two equivalent formulations of U-score. What are they, and why does the paper present both?

**Your answer:**
its element wise multiplication, its really just formulated in this way with the operator, to keep the contributions seperated of the prototype, and not colelct them into one value directly

**Evaluation:**
Got what ⊙ does (element-wise multiplication) and why it matters (keeps contributions separate). But the two formulations weren't written out:

1. **Decomposed form:** U-score(u,t) = Σ s_l^user, where **s**^user = **u*** ⊙ **t** — each element is one prototype's contribution
2. **Compact form:** U-score(u,t) = **u***^⊤ **t** — just a dot product

They're mathematically identical (dot product = sum of element-wise products). The paper presents both because: the compact form shows it's just a dot product (efficient, familiar), while the decomposed form shows how each prototype contributes separately (explainable). Same computation, two perspectives — efficiency vs. interpretability.

---

### Q6: What is s^user and why is it the key to explainability?
The paper defines s^user = u* ⊙ t. What does each element of this vector represent, and how does this enable explanations that standard MF cannot provide?

**Your answer:**
prototype contributions. MF is completely opaque, the latent factors could be anything, while prototypes are interpretable and their contributions measurable.

**Evaluation:**
Correct direction, could go deeper. More precisely: s_l^user = u*_l · t_l = sim(**u**, **p**_l^u) · t_l. This is "how similar user is to prototype l" × "how much item t weighs on characteristic l." You can rank these, show the top contributors, and say: "Recommended because you strongly match the Action/Thriller archetype (prototype 3) and this movie scores high on that characteristic."

In MF, the score is also a dot product (**u**^⊤ **t**), but each dimension is an opaque latent factor — dimension 7 has no meaning you can point to.

---

## Interpretability Properties

### Q7: The paper claims three interpretability properties of U-ProtoMF. What are they?
The text after Eq. 2 describes how the model enables interpretation. Identify all three properties and explain each briefly.

*Hint: one is about user representation, one is about item embeddings, one is about the score itself.*

**Your answer:**
1. the vlaues of u* as consumptions characteristics share tied to a user prototype.
2. item embeddings as weights of corresponding characteristics tied to user-prototypes.
3. score as a linear combination, that can be decomposed into contributions building on 1 and 2

**Evaluation:**
All three correctly identified. Sharpened:

1. **User representation is interpretable:** Each value u*_l corresponds to a specific consumption characteristic (the l-th prototype). A user is now described by positive, meaningful similarities.
2. **Item embeddings become weights:** Since U-score = **u***^⊤ **t**, the item embedding dimensions now serve as weights for each prototype characteristic. A Heavy Metal song will have high weight on the "Metal fans" prototype dimension. The item embedding dimensions have acquired meaning via the prototypes.
3. **Score is a linear function:** The sum of weighted prototype similarities is decomposable. You can trace any recommendation back to which prototypes contributed most.

---

*When done, tell me and we'll evaluate together, then move to Section 3.2 (I-ProtoMF).*
