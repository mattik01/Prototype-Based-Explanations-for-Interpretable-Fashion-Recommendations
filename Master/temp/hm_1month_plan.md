# Plan: H&M dataset strategy — two versions — NOT YET EXECUTED

Status: **plan only, do not execute.** Decision converged 2026-06-07.

## Strategic framing (the "why")
Thesis focus = beat the **ProtoMF paper CF-only baseline** with feature-aware prototypes +
strong fashion **explanations**. Kaggle/MAP@12 is a side note (single-stage CF vs two-stage
retrieve→rerank = non-comparable).

**Why H&M at all** (older datasets like ml-1m/amazon have *latent* features too — genres,
category/brand — so H&M is not the "only" featured option). Ranked reasons:
1. **Rigorous reason — sparse + feature-rich.** H&M is the sparsest by density (0.033–0.062% at
   5-core, vs ml-1m 3.05% and amazon 0.131%), the regime where pure CF leaves value on the table,
   so feature-aware extensions have headroom to *demonstrably* help.
   **Do NOT oversimplify as "dense paper datasets vs sparse H&M":** only **ml-1m is dense** (the
   control where features should add little); **amazon is also sparse and item-cold** (~9.1 int/item,
   colder items than H&M's 46–289). Honest framing = a dense control (ml-1m) + two flavors of
   sparsity — item-cold amazon vs user-thin, large-catalog, feature-rich H&M — a richer "where do
   features help" ablation than a binary.
2. **Showcase reason — interpretable fashion domain.** Style archetypes + visual, human-relatable
   explanations. Qualitative pillar, not the evidence pillar.

> Optional future axis: run feature-prototypes on ml-1m/amazon (their latent features) to sharpen
> a "**where** does it help" story (little on dense, lots on sparse). Scope cost: codebase needs
> feature ingestion for those datasets. Flagged, not committed.

**Critical design constraint:** any shrinking lever must **preserve H&M's defining sparsity +
feature richness**. Levers ranked by how well they preserve that:
window length (best, biases on recency — competition-justified) > random user subsample (good) >
~~active-user sampling / high k-core~~ (rejected — they *raise density* and bias toward power-
shoppers, sanding off the exact trait that justifies the dataset).

## Version 1 — `hm_1_month` (dev / judgment testbed) ← BUILD THIS
Small, efficient, but preserves sparsity + features. Source of all **quantitative** claims.

- **Window:** last 1 month (`t_dat >= 2020-08-22`; data ends 2020-09-22). Justified by competition
  (top solutions used 4–5 week windows; recency carries the signal) and mirrors the paper's
  lfm2b-1mon being a 1-month set. **This is the only real deviation from the paper, and it's the
  best-justified knob.**
- **K-core:** **5-core**, iterative (item→user to convergence) — identical to ml-1m/amazon/hm_3_month
  and our splitter. Paper-faithful; keeps ~29% of users (not the ~5% that 10-core keeps); preserves
  sparsity (0.062% vs 10-core's 0.335%).
- **Dedup:** keep-first (unchanged).
- **Random user subsample:** **FALLBACK ONLY.** Do not apply day-one. Add a random (not most-active)
  user subsample to hit size *only if* runtimes exceed the soft limit after the epoch-cap lever.
- `item_features.csv`: same 9 categorical columns as hm_3_month.

### Size (read-only probe `hm_1month_sizing_probe.py`)
1-month window: raw 1,190,911 → dedup 1,051,730 (256,355 users / 29,548 items).

| k-core | interactions | users | items | density | int/user | int/item |
|:------:|-------------:|------:|------:|--------:|---------:|---------:|
| 3  | 867,731 | 139,968 | 19,187 | 0.032% | 6.2  | 45.2 |
| **5 (chosen)** | **623,230** | **73,418** | **13,651** | **0.062%** | **8.5** | **45.7** |
| 10 (rejected) | 171,164 | 12,337 | 4,136 | 0.335% | 13.9 | 41.4 |

Refs: ml-1m 574,376 / 6,034 / 3,125 (5-core). hm_3_month 2,898,804 / 256,701 / 26,963 (5-core).

### Feasibility
Cost ≈ ml-1m on interactions (623K ≈ 574K) + a ~4.4× item-count surcharge (eval + proto reg).
Heaviest models (item_proto/user_item_proto) at full 100-sample/100-epoch ≈ ~12–18h → brushes the
1-day soft limit.
VRAM is a non-issue on the cluster (13.7K items ≈ half of hm_3_month's 27K; smoke-tested <6 GB).

**Feasibility lever: TBD.** The epoch-cap reduction (100→50) idea is **dropped for now** — the user
has a different approach in mind (to be specified). Do not assume epoch-cap or hyperopt-sample
reduction as the mitigation until the new approach is recorded here.

## Version 2 — `hm_3_month` (qualitative showcase, later)
Already built (5-core). Repurposed as the **explanation/visualization showcase**, used *after* the
approach is developed. May be slightly adjusted (e.g. focus a coherent segment) to demonstrate the
method's strengths and find a compelling use-case.

> **Integrity guardrail:** V2 is for *showing, not proving*. A curated/segment-focused showcase for
> **qualitative** explanation demos is legitimate *if clearly labeled illustrative*. V2 metrics must
> NOT become headline quantitative claims (that would be cherry-picking). All quantitative
> comparisons rest on V1.

## Implementation (when approved — DO NOT run yet)
- `hm_splitter.py` supports it: `--months 1 --core 5`. Random subsample would need a small addition
  (only if the fallback is triggered).
- Output `data/hm_1_month/` (mirror hm_3_month layout: 5 split files + item_features.csv).
- Register `hm_1_month` in `start.py` dataset choices.
- Verify item_features.csv on the subset; sanity-check split sizes; benchmark first run for timing.
