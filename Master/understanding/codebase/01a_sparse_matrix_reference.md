# 01a — Sparse Matrix Design in `ProtoRecDataset`

*Companion reference to [01_data_layer.md](01_data_layer.md) — covers the dual sparse matrix architecture in `rec_sys/protomf_dataset.py`.*

## Overview

`ProtoRecDataset` maintains two sparse matrices with **different purposes** and, for val/test splits, **different contents**.

## Matrix Roles

| Matrix | Format | Purpose | Access Pattern |
|--------|--------|---------|----------------|
| `coo_matrix` | COO (Coordinate) | **Iteration** — defines which (user, item) pairs exist in this split | `coo.row[index]`, `coo.col[index]` in `__getitem__` |
| `csr_matrix` | CSR (Compressed Sparse Row) | **Negative sampling exclusion** — fast lookup of all items a user has consumed | `csr.indices[csr.indptr[row]:csr.indptr[row+1]]` in `_neg_sample_*` |

## Contents Per Split

| Split | `coo_matrix` | `csr_matrix` |
|-------|-------------|-------------|
| **train** | Train interactions | Train interactions |
| **val** | Val interactions **only** | Val + Train interactions |
| **test** | Test interactions **only** | Test + Train interactions |

### Why they differ for val/test

- **COO** defines the sample set — during validation, you iterate over val interactions (each val pair becomes one `__getitem__` call).
- **CSR** is used to find which items to **exclude** from negative sampling. When sampling negatives for a val interaction, you must exclude both:
  - The user's val items (so you don't sample the positive as a negative)
  - The user's train items (the user already consumed these — they aren't truly negative)

  Hence `csr_matrix = val_csr + train_csr` (line 94).

## Why Two Formats?

**COO** stores three flat arrays: `row`, `col`, `data`. This makes random access by index trivial — `__getitem__(i)` just reads `row[i]` and `col[i]`. But finding "all items for user X" requires scanning the entire `row` array.

**CSR** stores `indptr` (row pointers) and `indices` (column indices). This makes "all items for user X" an O(1) slice operation: `indices[indptr[x]:indptr[x+1]]`. But iterating over individual entries by sequential index is not its strength.

## Code References

```
protomf_dataset.py:76-78   — train CSR construction
protomf_dataset.py:87-94   — val: separate COO, combined CSR
protomf_dataset.py:99-106  — test: separate COO, combined CSR
protomf_dataset.py:109-112 — train: same data for both
protomf_dataset.py:121     — CSR slice for consumed items (neg sampling)
protomf_dataset.py:164-165 — COO access in __getitem__
```

## Diagram

```
Train split:
  COO: [train pairs]          CSR: [train pairs]
  (iterate over these)         (exclude these from neg sampling)

Val split:
  COO: [val pairs]             CSR: [val + train pairs]
  (iterate over these)         (exclude these from neg sampling)

Test split:
  COO: [test pairs]            CSR: [test + train pairs]
  (iterate over these)         (exclude these from neg sampling)
```
