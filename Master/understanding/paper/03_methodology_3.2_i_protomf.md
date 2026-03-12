# Section 3.2: I-ProtoMF

Paper page 249 (Section 3.2)

---

### Q1: What is the core assumption of I-ProtoMF?
U-ProtoMF assumes prototypical *users* exist. What does I-ProtoMF assume instead, and what kind of patterns are item prototypes meant to capture? The paper gives an example.

**Your answer:**
prototipical items like the action movie

**Evaluation:**
Correct. Item prototypes capture co-consumption patterns — items that tend to be consumed together or share behavioral similarity. Paper's examples: items within a specific musical genre or product category.

---

### Q2: Write out the item prototype transformation and the I-score.
By analogy to U-ProtoMF: what is t*, what are its dimensions, and what does I-score(u, t) look like? What role does **u** play here compared to **t** in U-ProtoMF?

**Your answer:**
I dont wnat to repeat this its the same

**Evaluation:**
Fair — structurally identical, just mirrored. For the record: t* ∈ ℝ^{L^t}, I-score(u,t) = t*^⊤ u, and **u** ∈ ℝ^{L^t} plays the "plain embedding" role that **t** played in U-ProtoMF.

---

### Q3: What is s^item and what does it explain?
In U-ProtoMF, s^user explained the recommendation from the user's perspective. What perspective does s^item provide? What kind of statement can you make with it?

**Your answer:**
how much an item contains of the different prototypes, based on user interactions.
I wonder how this will work out though for item sets in datasets that are more strictly categorized anyway, for example in fashion items. i hope the prototypes will capture more something like "winter war" instead of something like pants

**Evaluation:**
Close but the perspective is flipped. s^item doesn't describe how much an item "contains" of prototypes — that's t* (the item's similarity to each item prototype). s^item = t* ⊙ **u**, so each element is "item's similarity to prototype l" × "user's weight on that characteristic." It explains the recommendation **from the item's perspective**: "this item was recommended because it's strongly a Thriller (prototype 5) and you have high affinity for Thrillers."

Fashion intuition is good — whether prototypes capture "warm winter outfit" vs. just "pants" depends on what interaction patterns exist in the data. This is exactly where your feature-aware extension could help: features could guide prototypes toward semantically meaningful groupings rather than purely behavioral ones.

---

### Q4: What is symmetric about U-ProtoMF and I-ProtoMF, and what is NOT symmetric?
The paper says I-ProtoMF "follows the same structure." Identify what is structurally identical and what differs between the two models.

**Your answer:**
This please you tell me

**Evaluation:**
**Symmetric:** The architecture template — embedding → prototype similarities → new representation → dot product with other side → score. Same shifted cosine. Same inclusion criteria structure. Same loss structure.

**NOT symmetric:**
- Different prototype sets: P^u (L^u prototypes) vs P^t (L^t prototypes), can have different counts
- Different embedding spaces: in U-ProtoMF, **t** ∈ ℝ^{L^u}; in I-ProtoMF, **u** ∈ ℝ^{L^t}. The "plain" side's embedding dimension is dictated by the *other* side's prototype count
- Different explanation perspective: s^user explains via user archetypes, s^item explains via item archetypes
- They are **separate models** — you pick one or the other (or combine in UI-ProtoMF)

---

### Q5: The inclusion criteria for I-ProtoMF (Eq. 9) mirror those of U-ProtoMF (Eq. 4-5). What do they enforce, and what changes?
Don't just say "same thing but for items." Be precise about what R_{P^t → T} and R_{T → P^t} each ensure.

**Your answer:**
Each item prototype matches on to at least one item, and evey item is associated strongly with one prototypew, the degree of this can be controlled via the lamdas. for users I will say its analog sorry.

**Evaluation:**
Correct. The inclusion criteria are a mutual coverage constraint:
- R_{P^t → T}: each prototype must "claim" at least one item (no dead prototypes)
- R_{T → P^t}: each item must be strongly associated with at least one prototype (no orphan items)
Lambdas (λ3, λ4) control the strength. Analogous to user side with λ1, λ2.

---

*When done, tell me and we'll evaluate together, then move to Section 3.3 (UI-ProtoMF).*
