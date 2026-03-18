# Session 1: Data Layer

**Files:** `rec_sys/protomf_dataset.py` (~193 lines), `data/ml-1m/movielens_splitter.py` (~97 lines)

Open these files side-by-side while answering. The splitter prepares data; the dataset class feeds it to the model during training.

---

## Part A: The Splitter (`movielens_splitter.py`)

### Q1: What does the 5-core filtering do, and why is it in a `while True` loop?
The splitter filters ratings ≥ 3.5, then applies 5-core filtering. What does "5-core" mean, and why can't you do it in a single pass?

**Your answer:**
K 5 filtering, means items and users all repspectively are only kept if they have at least 5 positive interactions, ("not dead users and items that add noise), repeat because iteratively removing items and users, may brake 5 cores of previous iterations. which used to be in tact.

**Evaluation:** ✅ Correct. Both the definition and the reasoning for the loop are right. Minor addition: it's not just about "noise" — users/items with < 5 interactions don't have enough signal for collaborative filtering to learn meaningful embeddings. The loop exits when `len(lhs) == start_number` (line 48), meaning no more removals occurred.

---

### Q2: How does the temporal leave-k-out split work?

**Your answer:**
Walk through what happens for a user with 10 interactions (sorted by timestamp) when k=1. Which interactions end up in train/val/test?
1-8 in train, 1 in val, 1 in test

**Evaluation:** ✅ Correct. Interactions 1–8 → train, 9 → val, 10 → test. That's exactly lines 75–77. One nuance: users with ≤ 2 interactions (≤ 2k) go entirely into train with no val/test entries (lines 71–73), since there aren't enough interactions to split.
This is impossible for k=1 because of 5 core filtering, but for higher ks, it might be an issue. 

---

### Q3: What's the purpose of the user_id/item_id remapping (lines 54–61)?
The original data has MovieLens IDs (e.g., user 6040, item 3952). Why create new integer IDs, and what would break downstream if you skipped this step?

**Your answer:**
this one I do not quite know

**Evaluation:** ❌ Gap identified — this is important for Phase 3 (H&M integration). The remapping creates **contiguous 0-based integer IDs**. This matters because:
1. **Sparse matrices** (`sp.csr_matrix`) use row/column indices as array positions. If user IDs go up to 6040 but only 6000 users exist, you'd allocate a 6040-row matrix with 40 empty rows — wasted memory.
2. **Embedding layers** (`nn.Embedding(n_users, dim)`) would allocate 6040 embedding vectors instead of 6000, with 40 untrained rows.
3. Without contiguous IDs, you risk index-out-of-bounds errors or silently oversized tensors. The H&M splitter you build will need the same remapping.

---

## Part B: The Dataset (`protomf_dataset.py`)

### Q4: Why two sparse matrices (COO and CSR)?
The class maintains both a COO matrix and a CSR matrix. What is each used for, and why not use just one format?

**Your answer:**
two representations of sparse matrixes (interaction) csr is used for negative samplix, coo for training iteration. train csr is always loaded. But I do requre a report of what amtrix for waht split contains exactly what. there seem to be caveats. please put a short report as an md file into the understanding codebase folder.

**Evaluation:** ✅ Correct direction. COO for iteration, CSR for negative sampling exclusion — that's the key insight. Report created at `01_sparse_matrix_report.md` with full details including the per-split content differences.

---

### Q5: For a validation dataset, what data is in `self.coo_matrix` vs `self.csr_matrix`?
Be specific — which interactions are in each, and why are they different?

**Your answer:**
include the answer of this question in the report above.

**Evaluation:** Covered in the report. Key answer: `coo_matrix` = val interactions only (defines what to iterate over), `csr_matrix` = val + train interactions (so negative sampling excludes both — you don't want a training item sampled as a "negative" during validation). See `01_sparse_matrix_report.md`.

---

### Q6: What does `__getitem__` return, and why is the positive item always at index 0?
Describe the exact shape and content of the three returned values for a single call. Why does the position of the positive item matter?

**Your answer:**
it matters, so that the caller knows which item is the correct one, its just a list otherwise how would one know, but i am suspecting it gets shuffled or something during val.

**Evaluation:** ⚠️ Partially correct. The convention part is right, but **no shuffling happens**. The returns are:
- `user_idx`: scalar int64
- `item_idxs`: array shape `(1 + n_neg,)` — positive at index 0, then negatives
- `labels`: array shape `(1 + n_neg,)` — `[1, 0, 0, ..., 0]`

The labels vector already encodes which is positive. The fixed position matters because loss functions (BCE, BPR, sampled softmax) rely on this ordering — BPR specifically needs to compute `score(pos) - score(neg)`.

---

### Q7: Compare the two negative sampling strategies.
The code has `_neg_sample_uniform` and `_neg_sample_popular`. What's the practical difference, and what does the 0.75 exponent do in the popular strategy?

**Your answer:**
it takes into account fequency of items during sampling, making it a more realistic "candidate set", 0.75 seems to be an apha but please explain it

**Evaluation:** ✅ Good intuition. The 0.75 exponent is a **smoothing factor** borrowed from Word2Vec's negative sampling. It flattens the popularity distribution so that:
- Popular items still appear more as negatives, but not as dominantly
- Rare items get a relatively higher chance than raw frequency would give
- Example: popularity 10000 vs 100 → raw ratio 100:1, with 0.75 exponent → 10000^0.75 / 100^0.75 ≈ 31.6:1
- Without smoothing, the model would mostly see top-100 items as negatives and never learn to discriminate among the long tail.

---

### Q8: Why is `pop_distribution` always computed from training data, even for val/test splits?
Line 81–82 computes popularity from `train_csr` regardless of `split_set`. What would go wrong if you computed it from the val or test set instead?

**Your answer:**
please give me a detailed explanation for this one

**Evaluation:** This is about **information leakage** — a fundamental ML evaluation principle:
1. **Data leakage:** Computing popularity from val/test would inject future knowledge into the sampling process. The popularity of items in val/test reflects interactions that haven't happened yet from the model's perspective.
2. **Deployment realism:** In production, you only have historical data. Popularity stats from the future aren't available, so using them would make evaluation unrealistically optimistic.
3. **Evaluation integrity:** Negative sampling during val/test should mirror the training distribution. Changing the distribution between splits confounds evaluation — you wouldn't know if metric differences come from the model or the shifted negative distribution.

The training set represents the "known world" — all statistics should derive from it.

---

## Synthesis

### Q9: Trace the data journey from raw `ratings.dat` to a single training batch.
Starting from the raw file, list the transformations in order until you get the `(user_idx, item_idxs, labels)` tuple that the model receives. Be specific about what format the data is in at each step.

**Your answer:**
ratings, the data is first trhough the spliter, filtered and then split into csv tables items/users, train val and split .
then the protoRecDataset interface is applied to those tables, and it transforms into 2 sparse matrix representation.
after init, and the load data. it has the operations required to provide a dataloader. with a sampling strategy as one of its parameters

**Evaluation:** ✅ Correct flow, but could be more precise. The full chain:
1. **Raw file** (`ratings.dat`): `user::item::rating::timestamp` text
2. **Filter**: Keep ratings ≥ 3.5 → implicit positive feedback
3. **5-core**: Iteratively remove users/items with < 5 positives
4. **Remap IDs**: Original IDs → contiguous 0-based integers; save `user_ids.csv`, `item_ids.csv`
5. **Temporal split**: Sort by timestamp, last → test, second-to-last → val, rest → train. Save as 3 CSVs.
6. **`ProtoRecDataset.load_data()`**: Read CSVs, build COO (iteration) and CSR (neg sampling exclusion)
7. **`__getitem__(i)`**: COO index → `(user, pos_item)` → sample `n_neg` negatives via CSR → return `(user_idx, [pos, neg1, ..., negN], [1, 0, ..., 0])`
8. **DataLoader**: Batches these tuples into tensors the model receives

---

## Session Summary

| Question | Rating | Notes |
|----------|--------|-------|
| Q1 5-core | ✅ | Solid |
| Q2 Leave-k-out | ✅ | Correct |
| Q3 ID remapping | ❌ | Key gap — important for H&M splitter |
| Q4 COO vs CSR | ✅ | Right direction |
| Q5 Val matrices | — | Deferred to report |
| Q6 `__getitem__` | ⚠️ | No shuffling happens; know the exact shapes |
| Q7 Neg sampling | ✅ | Good intuition |
| Q8 Train-only stats | — | Explained: information leakage |
| Q9 Data journey | ✅ | Right flow, could be more precise |

**Key takeaway for Phase 3:** The ID remapping (Q3) and train-only statistics (Q8) are patterns you'll need to replicate exactly in the H&M splitter.
