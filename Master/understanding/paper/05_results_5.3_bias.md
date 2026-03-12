# Section 5.3: Showcasing Societal Biases

Paper pages 253-254 (Section 5.3, Figure 4, Table 5)

---

### Q1: How do they determine which prototypes are "male-related" or "female-related"?
Describe the methodology step by step. What threshold defines a prototype's "representatives," and what statistical test determines gender association?

**Your answer:**
set of 95th percentile users in similarity represent the prototype. then fisher tests.

**Evaluation:**
Correct. Full pipeline: (1) For each user prototype, select all users whose similarity is above the 95th percentile — the "representatives." (2) Count male vs female among representatives. (3) Compare this gender ratio to overall dataset ratio using Fisher exact tests at α=1%, with Bonferroni correction for multiple comparisons. Significantly skewed → labeled male/female. Not significant → neutral (gray bars in Figure 4).

---

### Q2: What do the numbers tell us?
ML-1M: 36 male vs 9 female prototypes (out of 93). The dataset is 72% male / 28% female. Is the prototype imbalance just reflecting the data imbalance, or is something worse happening?

**Your answer:**
The ratio is more severe than the gender imbalance. This could be a magnifying effect, or reflect some true world property of women being less diversified in movie preferences, regardless it might be undesirable

**Evaluation:**
Good thinking. Dataset is 72/28, prototypes are 36/9 ≈ 80/20. The model allocates more "representational capacity" to male patterns. Whether this is amplification or reflects real preference diversity is debatable — but the key point is that ProtoMF makes this **visible and measurable**. With MF you'd never know this was happening inside the model.

---

### Q3: The bias mitigation method (Table 5b) — how does it work?
They multiply female-related prototype values in u* by a factor λ. Walk through what λ=1.0, λ=0.33, and λ=0.0 mean, and what happens to the recommendations in Table 5b.

**Your answer:**
essentially they are reducing the prototypes impact that are associated with "female preferences", balancing out or even flipping the models gender bias in recommendation, if that is a factor that is not desired.

**Evaluation:**
Correct idea. Precise λ values:
- λ=1.0: no change — original recommendations (Romance/Musical heavy for this female user)
- λ=0.33: female-related prototype values dampened to 1/3 — recommendations shift toward more genre-diverse (Action appears alongside Romance)
- λ=0.0: female-related prototypes completely zeroed out — only male and neutral prototypes contribute, recommendations flip to Action/Adventure/Sci-Fi

Key point: **controllable steering at inference time** — no retraining needed. A system designer can choose the degree of intervention.

---

### Q4: Why is this bias mitigation possible with ProtoMF but NOT with standard MF?
This is the key structural argument. What property of ProtoMF's architecture enables targeted intervention on specific prototypes?

**Your answer:**
transparency of latent factors I guess, but could analysis on latent factors not also be connected to gender info statistically and be modified, please help me answer this?

**Evaluation:**
Good question. In MF, dimensions are **entangled**. Dimension 7 might encode a mix of genre preference, activity level, and age. You *could* run post-hoc correlation analysis (correlate each dimension with gender), but:

1. Correlations would be weak and mixed — no clean separation
2. Even if "dimension 7 correlates with female," modifying it has **unpredictable side effects** because it also encodes other things
3. You'd modify the embedding itself, changing the user's representation permanently

With ProtoMF, each prototype is **independently interpretable** and contributes **additively** to the score. You can identify "prototype 40 = female-associated" cleanly (via Fisher test), then scale its contribution without touching anything else. Linear decomposition means modifying one prototype's weight has a **predictable, isolated effect** on the final score. No side effects, no retraining, adjustable at inference time per-user.

The structural advantage: **interpretable components + linear aggregation = targeted intervention**.

---
