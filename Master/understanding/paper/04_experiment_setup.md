# Section 4: Experiment Setup

Paper pages 250-251

---

## Datasets

### Q1: What preprocessing does each dataset go through, and why?
The paper applies different filtering to each dataset. For ML-1M: what threshold makes a rating "positive"? What is "5-core filtering"? Why filter at all?

**Your answer:**
postive is 3.5 and above, 5 core filtering means both users and items need to have at least 5 interactions. FIlter to ensure qualitiy of data reduce noise.

**Evaluation:**
Correct. Rating threshold is 3.5 (paper says "above 3.5"). 5-core filtering is exactly as described. The "why" is right: sparse users/items with 1-2 interactions add noise and make evaluation unreliable. Note: for Amazon they also use ratings above 3.5 as positive then 5-core, and for LFM2b they do additional outlier removal (99th percentile listeners) and age filtering (10-95).

---

### Q2: How does H&M compare to these three datasets?
Looking at Table 1 (users, items, interactions), where would H&M (~31M transactions, ~105K articles, ~1.3M customers) sit relative to ML-1M, LFM2b-1mon, and AmazonVid? What challenges do you expect from the scale difference?

**Your answer:**
The scale is astronomically different. computational effort might be a big problem, specifically the tuning pipeline probably needs to be adjusted, or hardware upgraded.

**Evaluation:**
Correct concern. ML-1M has ~574K interactions, AmazonVid ~132K, LFM2b ~877K. H&M raw has ~31M — that's 35-230x larger. Even after filtering it'll be massive. Hyperopt pipeline (100 configs × epochs × all users) will be the bottleneck. The masterplan addresses this — ASHA scheduler, fewer samples, shorter temporal window. The number of items (105K vs 3K-15K) is also a big jump and affects evaluation time.

---

## Evaluation Protocol

### Q3: Explain the leave-one-out split strategy.
The paper describes ordering interactions by timestamp and splitting. What goes into train, validation, and test? Why is temporal ordering important here (as opposed to random splitting)?

**Your answer:**
the order establishes a sequence with val/test coming at the end. this mirrors actual recommendation use cases, where from a users interaction history known, a recommendation is made (unknown)

**Evaluation:**
Correct idea, missing the specific split. The specific split:
- **Test:** each user's **last** interaction (chronologically)
- **Validation:** each user's **second-to-last** interaction
- **Train:** everything else

This is "leave-one-out" — literally one interaction held out per user for test, one for validation. It's per-user, not global. Random splitting would leak future information — the model could train on an interaction from March and be tested on one from January, which is unrealistic.

---

### Q4: How does negative sampling work during evaluation vs. training?
The paper fixes x=99 negatives for evaluation but treats x as a hyperparameter for training. Why the difference? What does "rank the positive item among the sampled ones" mean concretely?

**Your answer:**
this to me please explain in detail i did not quite understand it in the paper.

**Evaluation:**
Concretely, for user Alice at evaluation time:
- Her test item is "The Matrix" (her last interaction)
- We sample 99 random items she has NOT interacted with (e.g., random movies)
- We compute the model's score for all 100 items (1 positive + 99 negatives)
- We rank them by score, highest first
- **Question:** where does "The Matrix" land in that ranking?

If it's in the top 10 → Hit@10 = 1. If it's ranked 3rd → it contributes to NDCG@10 with a higher discount than if ranked 9th.

**Why 99 for evaluation?** Fixed for fair comparison. Every paper using 99 negatives is directly comparable. Standard protocol from the "Neural Collaborative Filtering" paper.

**Why variable x for training?** During training, x controls the loss function difficulty. More negatives = harder problem = potentially better gradients, but also slower. The loss function (Eq. 3, sampled softmax) sums over these negatives in the denominator — more negatives means a tighter approximation of the full softmax over all M items.

**Why not rank against ALL items?** For ML-1M with 3,125 items you could. For LFM2b with 78K items, ranking every user against every item is expensive. Sampled evaluation is a practical compromise.

---

### Q5: What are the two evaluation metrics, and what does each capture?
Hit Ratio@10 and NDCG@10. One is binary, one is position-sensitive. Explain the difference and why you need both.

**Your answer:**
1 captures how many know to be good/correct ones were actually in the recommendations, one captures if the hit recommendations are more towards the top. cut off and evaluated just on the top 10 recommendations. ndcg to me makes more sense though for a true ranking of items with values, something like an enjoyment factor instead of just 0 or 1 for if an item is a good rec or not. tell me if i am misunderstanding though

**Evaluation:**
Partially right, one misunderstanding.

**Hit Ratio@10:** For each user, binary — did the one positive test item appear in the top-10? 1 or 0. Averaged over all users. "How often does the model get it right?"

**NDCG@10:** Also evaluates that one positive item, but cares about **position**. Ranked 1st → NDCG = 1. Ranked 5th → NDCG ≈ 0.43. Ranked 11th+ → NDCG = 0. "When the model gets it right, how high up is the correct item?"

**Misunderstanding corrected:** NDCG is NOT using graded relevance here. There's only one positive item with relevance 1, everything else is 0. NDCG here purely measures **rank position** of that single hit. It degrades gracefully — a hit at position 1 is better than position 10 — while HR@10 treats both the same.

**Why need both?** A model could have high HR@10 (often gets right item in top-10) but low NDCG (consistently ranks it 8th-10th). Another model might have slightly lower HR@10 but higher NDCG (when it hits, it ranks first). Different failure modes, both informative.

---

## Baselines and Statistical Testing

### Q6: What are the three baselines, and what does each represent?
MF, RBMF, ACF. Why these three specifically — what "story" does comparing against them tell?

**Your answer:**
MF is the base and popular approach that ProtoMF extends, the other two are related approaches that also work with a prototype like idea.

**Evaluation:**
Correct, can be sharper. The specific story each baseline tells:
- **MF:** "Does adding prototypes help at all vs. plain embeddings?"
- **RBMF:** "Are *learned* prototypes better than *real-user* representatives?"
- **ACF:** "Are *separate* user/item prototypes with linear decomposition better than *shared anchors*?"

Each baseline isolates one design decision of ProtoMF. If ProtoMF beats all three, it validates: prototypes > no prototypes, learned > fixed, separate > shared.

---

### Q7: The paper uses Mann-Whitney U test with Bonferroni correction, and aggregates seeds with Fisher's method. Why all this statistical machinery?
Many RecSys papers just report averages. Why go further here, and what does p < 0.01 after correction actually guarantee?

**Your answer:**
This one please explain in detail I do not know any of these methods

**Evaluation:**
**The problem:** You run 5 models × 3 datasets × 2 metrics = 30 comparisons. If you just look at averages ("0.657 > 0.597, ProtoMF wins"), you have no idea if that difference is real or just luck from random initialization/data sampling.

**Mann-Whitney U test:** A non-parametric statistical test. For each user, you have a metric value (HR or NDCG). Collect per-user values for Model A and Model B, test: "are these two distributions significantly different?" Doesn't assume normal distribution (unlike t-test), important because HR values are binary (0 or 1).

**Bonferroni correction:** When doing many comparisons, you inflate the chance of finding "significant" results by accident. Testing 30 things at p < 0.01 → expect ~0.3 false positives. Bonferroni divides the threshold by number of comparisons: p < 0.01/30 ≈ 0.00033. Much stricter — if it still passes, very likely real.

**Fisher's method:** They run 3 seeds (random initializations). Each seed gives a p-value. Fisher's method combines multiple p-values into one: "considering all three seeds together, is the improvement significant?" More powerful than requiring each seed to be independently significant.

**What p < 0.01 after correction guarantees:** Less than 1% chance that ANY of the reported significant improvements (†/‡ in Table 2) happened by chance across ALL comparisons.

**For your thesis:** You should use the same protocol — it's already implemented in the codebase and will make your H&M results credible.

---

## Hyperparameter Tuning

### Q8: Describe the hyperparameter tuning setup.
How many configurations are sampled? What is the early stopping criterion? What is HyperBand and why is it only used for LFM2b? How is the final model selected?

**Your answer:**
100 configurations of hyperparameters, early stopping if the hit10 ratio does not improve anymore (no wasted computation on overfitting) I dont know what tree structure parzen estimators are. hyperband I dont know either. the final model is selected based on highest hit ratio over the training.

**Evaluation:**
Mostly correct. 100 configurations — correct. Early stopping on HitRatio@10 with patience 10 (no improvement for 10 consecutive epochs) — correct. Final model selected by highest HitRatio@10 on **validation set** (not "over training") — small correction.

**Tree-structured Parzen Estimators (TPE):** An intelligent hyperparameter sampling strategy. Instead of random search (try 100 random configs), TPE builds a probabilistic model of "which hyperparameter values tend to give good results" and samples more configurations from promising regions. Like Bayesian optimization — earlier trials inform later ones. In the codebase, handled by Ray Tune.

**HyperBand:** An early-stopping scheduler for hyperparameter trials. Starts many configs with a small budget (few epochs), kills worst performers early, gives more budget to promising ones. Think tournament brackets — most configs eliminated quickly, only the best survive to full training. Used only for LFM2b because it's the largest dataset (877K interactions), so full 100-epoch training of 100 configs would be prohibitively slow.

---

*When done, tell me and we'll evaluate together, then move to Section 5 (Results).*
