# Sampled metrics & the D3/D4 secondary readouts — documentation

> Written 2026-07-15 (S0.7 closing-gate discussion). Documents what the D3 and D4
> readouts are, why they exist, what the literature licenses, and what it forbids.
> Papers verified by actually reading them (GR6 status per citation at the bottom).

## 1. What D3 is

One extra **test-only pass** over each frozen canonical checkpoint in which the 99
eval negatives per row are drawn **proportional to item train-popularity^0.75**
(the codebase's own `_neg_sample_popular` sampler — an inherited word2vec-style
heuristic, disclosed, not a principled choice) instead of uniformly. Same
checkpoints, same rows, same metrics; only the candidate pool changes. Runner:
`Master/scripts/popneg_readout.py` → `<results_dir>/popneg_readout/`. Draws are
identical across model rows (num_workers=0 after `reproducible(seed)`, the
t05-pinned convention), so the column is internally comparable.

## 2. Why it exists

S0.1 kept the headline eval at **uniform-99** (the ProtoMF paper's protocol — D2,
replication comparability) and declared its known weakness in the "gaps we will
NOT fix" list: uniform negatives are mostly obscure items, so a model can score
well by simply preferring popular items — the headline cannot distinguish
popularity-following from taste modeling. **D3 is the ratified mitigation**: the
per-model **drop** uniform→popular is read as "how much of the headline rests on
popularity preference." It is a *contrast diagnostic*, never a headline.

S0.7 measured it: CF fleet drops 0.18–0.34 HR@10; lightfm feature rows drop only
0.04–0.08 (see `replication_report.md` §S0.7 Reference Fleet). Converges with
the prototype-collapse finding (item_proto rebuilds a popularity-bias term from
prototype capacity — vault `2026-07-15_1112`).

## 2b. What D4 is (stratified readout)

Not a new metric — the **same warm sampled-HR@10/NDCG@10, sliced into subgroups**
from one test pass over a frozen checkpoint (zero training; runner
`Master/scripts/stratified_readout.py` → `<results_dir>/stratified_readout/`):

- **By user train-history quartile** (facet-B warm-thin lens): does accuracy
  degrade for thin-history users?
- **By the positive item's train-popularity bucket** (1-5 / 6-20 / 21-100 / >100;
  bucket 0 = the F-S0-04 train-unseen slice): can the model rank rare items when
  they are the right answer?

Its job is turning "average accuracy" into "*whose* accuracy". S0.7 findings:
history quartiles flat for all seven models; popularity buckets split the fleet
(CF near-zero on the tail — with the R-b tie-floor caveat: exact zeros are
artifact-clipped — feature rows nearly flat).

**Scope limits (recorded at the walkthrough, 2026-07-15):**
1. **Window-dependence of the flatness claim:** on `hm_1_month` the quartiles
   span a compressed range (Q1 = exactly 3 purchases; Q4 = 9–89, median ~11) —
   one month + 5-core yields no deep histories. The honest claim is "history
   strata don't differentiate *on the one-month window*", NOT "history length
   doesn't matter"; on `hm_3_month`/`hm_full` the strata genuinely separate and
   the result could differ. (D4 is zero-cost to re-run on other checkpoints if
   ever wanted; old hm_3_month runs carry pre-scrutiny provenance.)
2. **Dataset scope:** all D3/D4 numbers exist for `hm_1_month` ONLY. ml-1m fleet
   rows are charter-lazy (arrive at dc01/dc05 SC.6); ml-1m is the deliberate
   opposite regime (heavy histories × coarse vocabulary), so neither the
   flatness nor the popularity findings should be assumed to transfer — running
   the same readouts there comes free with those rows.
3. Both readouts inherit the sampled-metrics caveats of §3–4 (the sliced
   quantity is the sampled, AUC-leaning HR — slices are protocol-relative too).

## 3. What the literature says (verified 2026-07-15)

**Rendle 2019 (arXiv:1912.02263, read in full)** — the core analysis, extended by
Krichene & Rendle KDD 2020:
- Sampled top-k metrics (HR/Recall@k, NDCG@k, AP@k) are **inconsistent** with
  their exact full-ranking versions: the relative order of two algorithms is not
  preserved, *not even in expectation*. Only AUC is consistent under uniform
  sampling (unbiased estimator).
- **Sampled metrics change character:** as the sample shrinks, every metric
  converges toward AUC behavior — linear in rank, not top-heavy. A "sampled
  HR@10" at m=99 negatives measures a smoother, AUC-leaning quantity, NOT the
  exact top-10 semantics its name implies.

**Dallmann, Zoller & Hotho, RecSys 2021 (arXiv:2107.13045, read in full)** —
exactly our contrast, studied as first-class conditions:
- Compares full ranking vs uniform-sampled vs **popularity-sampled** target sets
  (η=100) over 4 neural sequential models × 5 datasets.
- **Both** sampling strategies produce model rankings inconsistent with the full
  ranking (and mostly with each other); popularity sampling does **not** improve
  consistency despite the intuition that it approximates the item distribution.
  No safe sample size η exists; they recommend full ranking for SOTA claims.
- Precedent note: popularity-sampled eval negatives are established practice in
  famous prior work (BERT4Rec — Sun et al., CIKM 2019 — reported its results
  under popularity-sampled 100-negatives), which is what Dallmann critiques.

## 4. What this licenses and what it forbids (the framing rules)

**Licensed (what we do):**
- Comparing models **under one declared sampled protocol** (our uniform-99
  headline = the ProtoMF paper's own protocol) as a *protocol-relative*
  comparison — the basis for replication and for like-for-like fleet deltas.
- Reading the **uniform→popular delta per model** (D3) as a popularity-dependence
  diagnostic. Two disclosures ride along: the popularity distribution is the
  inherited pop^0.75 sampler, and the delta is distribution-relative — not a
  calibrated "amount of bias."

**Forbidden (what the literature refutes):**
- Claiming any sampled number (uniform OR popular) estimates full-ranking
  performance, or that the fleet's sampled ordering equals the full-ranking
  ordering (Rendle 2019 Def. 1; Dallmann Table 3 — Kendall's τ as low as −0.67).
- Reading "@10" as exact top-10 behavior — the sampled quantity is smoother and
  AUC-leaning (Rendle 2019 §4.4–4.5).
- Quoting D3 as a bias *measurement* rather than a contrast diagnostic.

**Thesis-writing consequences:**
- Accuracy claims are always phrased "under the paper's sampled protocol."
- Open option for the thesis-grade tier: hm_1_month has only 13,651 items — an
  **exact full-ranking eval pass is computationally trivial** for frozen
  checkpoints and would immunize the headline claims against the whole critique.
  Decide at model freeze (two-tier policy); not a scrutiny-phase change.

## 5. Citation status (GR6 ledger)

| Work | Status |
|---|---|
| Rendle, "Evaluation Metrics for Item Recommendation under Sampling", arXiv:1912.02263 (2019) | **Read in full** 2026-07-15. Safe to cite for: inconsistency of sampled top-k metrics, AUC exception, metric-character drift. |
| Krichene & Rendle, "On Sampled Metrics for Item Recommendation", KDD 2020, DOI 10.1145/3394486.3403226 (also CACM 65(7), 2022) | Existence/venue/DOI verified; **full text NOT read** (ACM paywall) — content corroborated via Rendle 2019 (its precursor) and Dallmann's replication. Adds corrected estimators. **Obtain via university access before citing it specifically** (cite Rendle 2019 for the core result meanwhile). |
| Dallmann, Zoller & Hotho, "A Case Study on Sampling Strategies…", RecSys 2021, DOI 10.1145/3460231.3475943 | **Read in full** 2026-07-15 (arXiv:2107.13045). Safe to cite for: popularity-sampled negatives also inconsistent, both strategies mutually inconsistent, no safe η, BERT4Rec precedent. |
| Cañamares & Castells, "On Target Item Sampling in Offline Recommender System Evaluation", RecSys 2020; Li et al., "On Sampling Top-K Recommendation Evaluation", KDD 2020 | **Not read** — known only through Dallmann's related-work section; pointers for the thesis's related work if needed. |

Paper texts saved under the session scratchpad; PDFs re-fetchable via arXiv ids.
