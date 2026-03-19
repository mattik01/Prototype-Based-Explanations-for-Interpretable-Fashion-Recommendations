# H&M Dataset Analysis Report

Generated: 2026-03-19

## articles.csv

- **Shape:** 105,542 × 25
- **No nulls** except `detail_desc` (416, 0.4%)
- **article_id:** 10-digit zero-padded string (e.g., `0108775015`)

### Feature columns (useful for Phase 4)

| Column | Cardinality | Notes |
|--------|-------------|-------|
| product_type_name | 131 | Fine-grained type (e.g., Trousers, Bra, Vest top) |
| product_group_name | 19 | Coarse type (Garment Upper/Lower/Full body, Accessories, etc.) |
| graphical_appearance_name | 30 | Solid (47%), All over pattern (16%), Melange, Stripe, Denim... |
| colour_group_name | 50 | Specific colour |
| perceived_colour_value_name | 8 | Dark (40%), Dusty Light (21%), Light, Medium Dusty... |
| perceived_colour_master_name | 20 | Black (21%), Blue (17%), White (12%), Pink, Grey... |
| index_name | 10 | Ladieswear, Divided, Menswear, Children, Baby, Sport |
| index_group_name | 5 | Ladieswear, Baby/Children, Divided, Menswear, Sport |
| section_name | 56 | Mid-level department grouping |
| garment_group_name | 21 | Jersey Fancy, Accessories, Jersey Basic, Knitwear... |
| department_name | 250 | Detailed department |

### Recommended feature columns for `item_features.csv`
- `product_group_name` (19) — coarse garment type
- `product_type_name` (131) — fine garment type
- `graphical_appearance_name` (30) — pattern/appearance
- `colour_group_name` (50) — colour
- `perceived_colour_master_name` (20) — colour (coarser)
- `index_group_name` (5) — demographic target
- `garment_group_name` (21) — garment category
- `section_name` (56) — section
- `department_name` (250) — department

## customers.csv

- **Shape:** 1,371,980 × 7
- `customer_id`: 64-char SHA-256 hex hash
- `FN`, `Active`: mostly null (~65%), only value is 1.0 — not useful
- `age`: 84 unique values, 15,861 nulls (1.2%)
- `club_member_status`: ACTIVE (93%), PRE-CREATE (7%), LEFT CLUB (<0.1%)
- `postal_code`: 352,899 unique — too high cardinality for features

## transactions_train.csv

- **Shape:** 31,788,324 × 5
- **Date range:** 2018-09-20 to 2020-09-22 (2 years)
- **No nulls anywhere**
- Columns: `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id` (1=store, 2=online)

### Interaction distributions

| Stat | Per customer | Per article |
|------|-------------|-------------|
| Mean | 23.3 | 304.1 |
| Median | 9 | 65 |
| P25 | 3 | 14 |
| P75 | 27 | 286 |
| P95 | 91 | 1,318 |
| P99 | 187 | 3,185 |
| Max | 1,895 | 50,287 |

### Duplicate pairs
- 4,481,885 duplicate (customer, article) pairs (14.1%)
- Must deduplicate, keeping first interaction

### Temporal windows

| Window | Rows | Users | Items |
|--------|------|-------|-------|
| 3 months | 4,056,792 | 535,417 | 42,611 |
| **6 months** | **8,185,912** | **748,053** | **51,478** |
| 9 months | 11,324,412 | 875,256 | 63,002 |
| 12 months | 14,932,373 | 995,078 | 70,976 |

### After 6-month window + dedup + 10-core filtering

| Metric | Value |
|--------|-------|
| Interactions | 4,982,296 |
| Users | 237,830 |
| Items | 28,218 |
| Density | 0.074% |
| Avg interactions/user | 20.9 |
| Avg interactions/item | 176.6 |
| Convergence | 5 iterations |

This is comparable to other datasets used in the paper (ml-1m: 575K interactions, lfm2b-1mon: ~2M).
Slightly larger but very manageable for training.

## Decision: 6-month window + 10-core

- **6 months**: Balances recency vs. volume. Fashion is seasonal, so older data may hurt.
- **10-core**: Standard filtering threshold used in lfm2b splitter. Ensures every user and item has enough interactions for meaningful leave-1-out split.
- These are the defaults in `hm_splitter.py` but configurable via CLI flags.
