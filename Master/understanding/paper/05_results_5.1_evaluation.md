# Section 5.1: Evaluation Results

Paper page 251 (Section 5.1, including t-SNE discussion)

---

### Q1: Read Table 2. What is the overall narrative?
Look at the results across all models and datasets. Which model wins overall? Are there any surprises or exceptions to the trend?

**Your answer:**
ProtoMF clearly wins out, surprise is that the most "explainable model" does not come at a tradeoff in terms of performance

**Evaluation:**
Correct. UI-ProtoMF wins across all datasets on both metrics (†‡ everywhere). The "no tradeoff" point is exactly what the paper emphasizes: effective AND explainable.

---

### Q2: Why does RBMF perform worse than MF on ML-1M?
RBMF is supposed to be an improvement over MF (it adds representative users). Yet on ML-1M it scores .282/.505 vs MF's .326/.571. What might explain this?

**Your answer:**
I dont know, for this one.

**Evaluation:**
Likely explanation: RBMF ties prototypes to *actual real users*, which constrains expressiveness. On ML-1M (6K users, relatively dense), MF's free embeddings have enough signal to learn good representations. RBMF's constraint of aligning to specific users actually hurts — it forces the model to represent patterns through the lens of individual users who may not be ideal representatives. On larger datasets (Amazon, LFM2b) where RBMF does better than MF, the additional structure from representatives might help regularize against sparsity. But RBMF still loses to ACF and ProtoMF everywhere — learned prototypes consistently beat fixed representatives.

---

### Q3: U-ProtoMF vs I-ProtoMF — which is better and why might that be?
Look at the pattern across datasets. Is one consistently better? The paper doesn't deeply analyze this, but think about what each model captures and which datasets might favor which.

**Your answer:**
amazon and music the i proto is better, but for movie lens the user interaction is better. I think its difficult to capture why that is, but surely there is a correlation also with item to user ratio.

**Evaluation:**
Good observation. The item-to-user ratio intuition is reasonable. Another angle: ML-1M has only 3,125 items — few enough that item embeddings can be well-learned directly, so prototyping users adds more value. Amazon/LFM2b have 14K-78K items — harder to learn good individual item embeddings, so item prototypes help more by sharing structure across items.

---

### Q4: The parameter complexity argument.
The paper compares parameter counts: MF has (N+M)×d, while ACF/UI-ProtoMF add 4K×d. Why does the paper make this argument, and why is "N and M are much larger than K" the key point?

**Your answer:**
it means that the apparant complexity does not really scale with dataset size in terms of parameters, but just with prototypes. but prototypes even with huge datasets, and of course depended on the domain, still will not be that many, otherwise defeating the purpose of a prototype.

**Evaluation:**
Excellent. The added parameters are 4K×d where K is the prototype count (typically 20-100), while N and M are thousands to millions. The overhead is negligible. The point about prototypes "defeating their purpose" if too numerous is sharp — that's a real design constraint.

---

### Q5: What do the t-SNE plots (Figure 2) show, and what do they confirm about the inclusion criteria?
Figure 2a shows user embeddings + user prototypes on ML-1M. Figure 2b shows item embeddings + item prototypes on LFM2b. What pattern do you see in where the prototypes land relative to the embeddings?

**Your answer:**
basically that all prototypes are close to some data points, but no point is completely removed from any prototype. some points are more in the middle of several prototypes, which makes sense

**Evaluation:**
Correct. Prototypes sit at the centers of user/item clusters (not isolated), confirming the inclusion criteria work — no orphan prototypes floating in empty space, and no data points completely unrepresented. The "middle of several prototypes" observation maps to users with complex tastes needing multiple prototypes to describe them.

---
