# ProtoMF: Prototype-based Matrix Factorization

## Core Concept: Prototype

Actual items or artefacts acting as a prototype representing a category or group of items or representing a user.

- Represent users and items in terms of similarities to corresponding prototypes
- User-item affinity score as a linear combination of user/item prototype similarities and corresponding item/user embeddings
- Extension of matrix factorization

## Core Concept: Explainability

- Recommendation can be decomposed into linear contributions of prototypes
- BUT requires interpretation of prototypes, 2 approaches:
  - Activate a prototype maximally through synthetic inputs and observe the model's output (for users in ProtoMF)
  - Interpret through a set of maximally close entities (for items in ProtoMF)

### Transparency
By Arrieta et al. [4]: "understanding the process followed by the model to produce any given output from its input data"

---

## From Matrix Factorization to UI-ProtoMF

#### Standard Matrix Factorization (Baseline)

Each user and item gets a learned embedding vector of dimension d:

$$\mathbf{u} \in \mathbb{R}^d, \quad \mathbf{t} \in \mathbb{R}^d$$

Score is the dot product:

$$\text{MF-score}(u,t) = \mathbf{u}^\top \mathbf{t} = \sum_{k=1}^{d} u_k \cdot t_k$$

**Limitation:** ❌ No interpretability - the d dimensions are latent features with no inherent meaning.

---

### Step 1: U-ProtoMF (Add User Prototypes)

**Intuition:** Represent each user by how similar they are to prototypical users (e.g., "Action enthusiast", "Romance lover", "Sci-Fi fan").

**Introduce** Lᵘ user prototypes (Lᵘ ≪ N):

$$\mathcal{P}^u = \{\mathbf{p}^u_1, ..., \mathbf{p}^u_{L^u}\}, \quad \mathbf{p}^u_l \in \mathbb{R}^d$$

**Transform** user u into similarity vector **u***:

$$\mathbf{u}^* = \begin{bmatrix} \text{sim}(\mathbf{u}, \mathbf{p}^u_1) \\ \vdots \\ \text{sim}(\mathbf{u}, \mathbf{p}^u_{L^u}) \end{bmatrix} \in \mathbb{R}^{L^u}$$

**Similarity function** (shifted cosine, ensures positive values ∈ [0,2]):

$$\text{sim}(\mathbf{a}, \mathbf{b}) = 1 + \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

**Item embeddings** now in prototype space: **t** ∈ ℝᴸᵘ (each dimension = weight for corresponding prototype)

**Score:**

$$\text{U-score}(u,t) = \mathbf{u}^{*\top} \mathbf{t} = \sum_{l=1}^{L^u} \underbrace{u^*_l \cdot t_l}_{s^{\text{user}}_l}$$

✅ Each $s^{\text{user}}_l$ shows how much prototype l contributed to the recommendation.

---

### Step 2: I-ProtoMF (Add Item Prototypes)

**Intuition:** Represent each item by how similar it is to prototypical items (e.g., "Horror movies", "Romantic comedies").

**Introduce** Lᵗ item prototypes (Lᵗ ≪ M):

$$\mathcal{P}^t = \{\mathbf{p}^t_1, ..., \mathbf{p}^t_{L^t}\}, \quad \mathbf{p}^t_l \in \mathbb{R}^d$$

**Transform** item t into similarity vector **t***:

$$\mathbf{t}^* = \begin{bmatrix} \text{sim}(\mathbf{t}, \mathbf{p}^t_1) \\ \vdots \\ \text{sim}(\mathbf{t}, \mathbf{p}^t_{L^t}) \end{bmatrix} \in \mathbb{R}^{L^t}$$

**User embeddings** now in item-prototype space: **u** ∈ ℝᴸᵗ

**Score:**

$$\text{I-score}(u,t) = \mathbf{u}^\top \mathbf{t}^* = \sum_{l=1}^{L^t} \underbrace{u_l \cdot t^*_l}_{s^{\text{item}}_l}$$

✅ Each $s^{\text{item}}_l$ shows how much item prototype l contributed.

---

### Step 3: UI-ProtoMF (Combine Both)

**Intuition:** Explainability from both perspectives - "We recommended this Sci-Fi movie because you're a Sci-Fi fan."

**Problem:** Dimensional mismatch (U-ProtoMF uses ℝᴸᵘ, I-ProtoMF uses ℝᴸᵗ)

**Solution:** Linear transformations to share parameters:

$$\hat{\mathbf{u}} = \mathbf{W}^u \mathbf{u} \in \mathbb{R}^{L^t}, \quad \hat{\mathbf{t}} = \mathbf{W}^t \mathbf{t} \in \mathbb{R}^{L^u}$$

**Final Score:**

$$\boxed{\text{UI-score}(u,t) = \underbrace{\mathbf{u}^{*\top} \hat{\mathbf{t}}}_{\text{U-ProtoMF}} + \underbrace{\hat{\mathbf{u}}^\top \mathbf{t}^*}_{\text{I-ProtoMF}}}$$
