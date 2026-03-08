
### Introduction Context

**Why Prototypes?**

- Prototype-based models provide a novel paradigm for learning latent factors
- Prediction score = linear combination of similarities between a data point and prototypes
- This linear combination enables clear separation of each prototype's contribution → explainability

**ProtoMF Key Claims:**

- Builds on Matrix Factorization (MF) but adds user/item prototypes
- User prototype example: embodies preference for Drama/Romance movies
- Item prototype example: represents specific musical genres or product categories
- Achieves both effectiveness (accuracy metrics) AND explainability

**Regulatory Alignment:**

- EU Regulatory Framework for AI
- EU Digital Service Act
- These encourage transparent RS models that can explain predictions and correct misconducts

---

### Related Work

**2.1 Prototypes in Recommender Systems**

|Approach|Description|Limitation vs ProtoMF|
|---|---|---|
|**RBMF** (Liu et al.)|Aligns MF latent factors with specific users as representatives; used for cold-start|ProtoMF learns _general_ prototypical patterns, not tied to specific users|
|**ACF** (Barkan et al.)|Uses same anchor vectors for both users and items|ProtoMF has _separate_ prototypes for users and items; enables tracing contributions|
|**Clustering-based**|Assigns users/items to subgroups|ProtoMF redefines users/items via _similarity vectors_ to prototypes (not hard assignments) + linear aggregation enables easier interpretation|

**2.2 Explainability Approaches**

- **Aspect-based** (Zhang et al., Cheng et al.): Use external info like reviews to create interpretable representations
- **Persona-based** (Barkan et al.): Model user via attentive mixture of personas
- **Post-hoc methods**: Association rules, influence functions, linear models applied after training

**ProtoMF differs by:**

1. Not relying on fixed external features
2. Providing novel explanation via user/item prototype contributions

**2.3 Fairness and Bias**

- RS algorithms show different performance across user groups (gender, age, personality)
- Example: Female users often receive worse accuracy (Ekstrand et al., Melchiorre et al.)
- ProtoMF can expose whether learned prototypes encode gender/sensitive information

---
# Section 3: Methodology Summary

## Setup and Notation

| Symbol | Description |
|--------|-------------|
| U = {u₁, ..., uₙ} | Set of N users |
| T = {t₁, ..., tₘ} | Set of M items |
| I = {(uᵢ, tⱼ)} | Implicit feedback (binary interactions) |
| d | Embedding dimension |
| Lᵘ | Number of user prototypes (Lᵘ ≪ N) |
| Lᵗ | Number of item prototypes (Lᵗ ≪ M) |

---

## Three Model Variants

### 3.1 U-ProtoMF (User Prototype MF)

**Core Idea:** Represent users by their similarity to *prototypical users* (e.g., "action movie lover", "romance fan").

**Components:**
- **User prototypes:** P^u = {**p**₁ᵘ, ..., **p**ᵘₗᵤ}, each **p**ᵘ ∈ ℝᵈ
- **Transformed user:** **u*** ∈ ℝᴸᵘ (vector of similarities to all prototypes)
- **Item embeddings:** **t** ∈ ℝᴸᵘ (weights for each prototype)

**Similarity Function (Shifted Cosine):**
```
sim(a, b) = 1 + (aᵀb)/(‖a‖·‖b‖) ∈ [0, 2]
```
Shifted to ensure positive values for interpretability.

**Score:** `U-score(u,t) = u*ᵀt = Σₗ sₗᵘˢᵉʳ`

Each sₗᵘˢᵉʳ shows the contribution of user prototype l.

---

### 3.2 I-ProtoMF (Item Prototype MF)

**Core Idea:** Represent items by their similarity to *prototypical items* (e.g., "horror movies", "romantic comedies").

**Components:**
- **Item prototypes:** P^t = {**p**₁ᵗ, ..., **p**ᵗₗₜ}, each **p**ᵗ ∈ ℝᵈ
- **Transformed item:** **t*** ∈ ℝᴸᵗ (vector of similarities to all prototypes)
- **User embeddings:** **u** ∈ ℝᴸᵗ (affinities for each item prototype)

**Score:** `I-score(u,t) = uᵀt* = Σₗ sₗⁱᵗᵉᵐ`

---

### 3.3 UI-ProtoMF (Combined)

**Core Idea:** Combine both models for richer explanations from user AND item perspectives.

**Challenge:** Dimensional mismatch between U-ProtoMF (ℝᴸᵘ) and I-ProtoMF (ℝᴸᵗ).

**Solution:** Linear transformations to share parameters:
```
û = Wᵘu ∈ ℝᴸᵗ    (project user to item-prototype space)
t̂ = Wᵗt ∈ ℝᴸᵘ    (project item to user-prototype space)
```

**Final Score:**
```
UI-score(u,t) = u*ᵀt̂ + ûᵀt*
                ↑        ↑
           U-ProtoMF  I-ProtoMF
```

Addition preserves linearity → clean separation of contributions for explainability.

---

## Loss Function

**Base Loss (Cross-Entropy):**
```
ℒᵣₑ꜀ = -Σ ln p(t|u) + λ_L2‖Θ‖

p(t|u) = exp(score(u,t)) / Σⱼ exp(score(u,tⱼ))
```

**Inclusion Regularization:**  (??? Is this required/good? experiment with this )

Forces prototypes to be meaningful and used:

| Term | Purpose |
|------|---------|
| R{P→U} | Each prototype has ≥1 similar user |
| R{U→P} | Each user is close to ≥1 prototype |

```
R{Pᵘ→U} = -(1/Lᵘ) Σₗ maxᵢ sim(uᵢ, pₗᵘ)
R{U→Pᵘ} = -(1/N) Σᵢ maxₗ sim(uᵢ, pₗᵘ)
```

Analogous terms exist for item prototypes.

**Final Losses:**
```
ℒ_U-Proto = ℒᵣₑ꜀ + λ₁R{Pᵘ→U} + λ₂R{U→Pᵘ}
ℒ_I-Proto = ℒᵣₑ꜀ + λ₃R{Pᵗ→T} + λ₄R{T→Pᵗ}
ℒ_UI-Proto = ℒ_U-Proto + ℒ_I-Proto  (ℒᵣₑ꜀ included once)
```

---

## Why This Enables Explainability

| Component | Interpretation |
|-----------|----------------|
| **u*** | Which consumption patterns describe this user |
| **t*** | Which item categories this item belongs to |
| **sᵘˢᵉʳ** | User prototype contributions to recommendation |
| **sⁱᵗᵉᵐ** | Item prototype contributions to recommendation |

**Key Property:** Linear structure means `Total Score = Sum of Interpretable Parts`

---

## Parameter Complexity

| Model | Extra Parameters vs MF |
|-------|------------------------|
| ACF | K × d |
| UI-ProtoMF | 4K × d |

Since N, M ≫ K, d in typical scenarios, overhead is minimal relative to interpretability gains.
