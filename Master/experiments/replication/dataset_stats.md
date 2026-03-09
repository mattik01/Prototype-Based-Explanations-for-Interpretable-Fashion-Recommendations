# Dataset Statistics (Phase 2.1)

## Preprocessing Summary

All datasets use: positive interactions only (rating ≥ 3.5), k-core filtering, temporal leave-1-out split.

## MovieLens-1M

| Stage | Entries | Users | Items |
|-------|---------|-------|-------|
| Original | 1,000,209 | 6,040 | 3,706 |
| Positive (≥ 3.5) | 575,281 | 6,038 | 3,533 |
| 5-core filtered | 574,376 | 6,034 | 3,125 |
| **Train** | **562,308** | **6,034** | **3,125** |
| Val | 6,034 | 6,034 | 1,792 |
| Test | 6,034 | 6,034 | 1,806 |

## Amazon2014 Video Games

| Stage | Entries | Users | Items |
|-------|---------|-------|-------|
| Original | 1,324,753 | 50,210 | 826,767 |
| Positive (≥ 3.5) | 970,030 | 44,666 | 632,869 |
| Deduplicated | 970,030 | 44,666 | 632,869 |
| 5-core filtered | 132,209 | 6,950 | 14,494 |
| **Train** | **118,309** | **6,950** | **14,394** |
| Val | 6,950 | 6,950 | 4,441 |
| Test | 6,950 | 6,950 | 3,640 |

## LFM-2b 1-Month (2020 Subset)

**Not available** — dataset removed from official source (cp.jku.at) due to license issues. Skipped for now.
