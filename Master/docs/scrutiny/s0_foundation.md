# S0 — Foundation Scrutiny Dossier

> Scrutinized under **Scrutiny Protocol v1.0** (`Master/docs/scrutiny_protocol.md`,
> 2026-07-11). Branch: `feat/scrutiny`. One appended section per step
> (S0.1 → … → S0.7, plus S0-build). Findings live in
> `findings_ledger.md` (F-S0-NN); this dossier owns the step narrative.

---

## S0.1 Eval protocol audit (2026-07-11)

**Scope:** what the live evaluation actually measures, verified against code
(`utilities/eval.py`, `utilities/consts.py`, `rec_sys/protomf_dataset.py`,
`rec_sys/trainer.py`, `rec_sys/tester.py`, `experiment_helper.py`,
`data/hm/hm_splitter.py`) and against the ProtoMF paper protocol as recorded in
`replication_report.md`.

### 1. Current state vs. paper protocol

| Aspect | Live implementation | Paper protocol | Verdict |
|---|---|---|---|
| Split | Temporal leave-one-out per user: last interaction → test, second-to-last → val, rest → train (`hm_splitter.py` step 6; ml-1m/amazon splitters equivalent, replication-validated) | Same | ✅ match |
| Eval candidates | 1 positive + `NEG_VAL=99` sampled negatives per eval row (`experiment_helper.load_data`), `eval_neg_strategy='uniform'` (`base_param`) | 1 + 99 uniform | ✅ match |
| Negative exclusion | Val negatives exclude train+val items of the user; test negatives exclude train+test (NOT the val item) (`ProtoRecDataset` docstring + `csr_matrix` construction) | Same (original code inherited) | ✅ match (quirk noted, not-fix #2) |
| Metrics | HR@K, NDCG@K for K ∈ {1,3,5,10,50} (`K_VALUES`); NDCG single-positive normalization correct (IDCG=1); top-k via `bn.argpartition` correct | Same | ✅ match |
| Model selection | `OPTIMIZING_METRIC='hit_ratio@10'` on val, best checkpoint saved | Same | ✅ match |
| Score monotonicity | Sigmoid applied before eval — monotonic, rank-invariant | n/a | ✅ no effect |
| K-core | 5-core on all in-scope datasets (ml-1m, amazon2014, hm_1_month) | 5-core (10 only for out-of-scope lfm2b) | ✅ match |
| Dedup | H&M: `drop_duplicates(['customer_id','article_id'], keep='first')` — **all repeat purchases removed**, discovery-only framing | Paper datasets have no repeats by construction | ✅ consistent; doc drift fixed (F-S0-01) |
| Early stopping | patience (10 prod / 7 dev) + `min_delta=1e-4` (`MIN_DELTA_DEFAULTS`) | patience only, any improvement | ⚠️ declared deviation (not-fix #6) |
| Hyperopt regime | dev profile 30 trials/60 epochs + ASHA (grace 4) for scrutiny runs; TPE seeded `HyperOptSearch(random_state_seed=seed)` | 100/100, 3 seeds | ⚠️ declared deviation — budget matter, goes into the S0.6 charter |
| Seeds | `SINGLE_SEED=38210573` used everywhere so far; `SEED_LIST` (3 seeds) exists unused | 3-seed means | ⚠️ decision D1 below |

### 2. Empirical verifications (run 2026-07-11, laptop)

- **One-eval-row-per-user invariant** (the `Evaluator` divides metric sums by
  `n_users` — correct only if every user contributes exactly one val and one
  test row): verified exactly for ml-1m (6,034/6,034), amazon2014
  (6,950/6,950), hm_3_month (256,701/256,701). **hm_1_month could not be
  verified locally** — see F-S0-05. → F-S0-03 proposes a permanent assert.
- **Train-unseen items in standard eval** (items whose 5-core-surviving
  interactions all fell into val/test rows — they carry random-init, never-
  trained ID embeddings): ml-1m **0**; amazon2014 **101 test items → 490/6,950
  test rows (7.1%)** (+140 val rows); hm_3_month 8 items → 23/256,701 rows
  (~0.01%). Paper-faithful accident, but it means amazon's replication numbers
  include a 7% naturally-cold slice scored on noise. → F-S0-04; direct input
  to the S0.3 cold-item definition. hm_1_month count to be measured when the
  data is reachable (S0.5/S0.7).
- **Metric implementations** read line-by-line — no defects found
  (`Hit_Ratio_at_k_batch` / `NDCG_at_k_batch` in `utilities/eval.py`).

### 3. Eval-negative RNG regime (audit)

Negatives are drawn per `__getitem__` via the **global numpy RNG**. With
`num_workers>0` (Linux default 2 — laptop and LEO5), DataLoader workers fork
from the parent each epoch and **numpy is never reseeded per worker** (only
torch is), so: (a) both workers run identical numpy streams (draws are
correlated across workers, though per-user exclusion masks make actual
negative sets differ), and (b) the parent's numpy state never advances, so
**val negatives are bit-identical across epochs** — an accidental but
beneficial stability property for early stopping. With `num_workers=0`
(Windows GPU machine), negatives are re-drawn from an advancing RNG every
epoch — a *different eval regime*. All scrutiny-phase runs are LEO5-only
(compute envelope), so the regime is consistent where it matters. → F-S0-02
(recommended disposition: document, wontfix).

### 4. Findings (details + proposals in `findings_ledger.md`)

| ID | Severity | One-liner | Recommended disposition |
|---|---|---|---|
| F-S0-01 | trivial | `hm_feature_analysis.md` claimed dedup key `(customer, article, date)`; code dedups `(customer, article)` keep-first | **fixed now** (trivial policy) |
| F-S0-02 | minor | numpy RNG duplicated across DataLoader workers; eval-negative regime differs `num_workers` 0 vs >0 | document, wontfix |
| F-S0-03 | minor | `Evaluator` silently assumes exactly one eval row per user (divide-by-`n_users`) | S0-build: assert rows==n_users in `val()`/`test()`; S0.3 must not blindly reuse the divisor |
| F-S0-04 | minor | Standard test sets already contain train-unseen items on random-init embeddings (amazon 7.1% of test rows) | wontfix (paper-faithful); S0.3 defines cold sets deliberately; disclose for amazon |
| F-S0-05 | minor | `hm_1_month` (V1, the primary scrutiny dataset) absent from the laptop; LEO5 unreachable today (no VPN) | scp splits+`item_features.csv` from LEO5 next VPN session — prefer copy over rebuild (splitter's non-stable sort makes a byte-identical rebuild non-guaranteed). Needed by S0.5. |

### 5. Decisions (proposed — for ratification at this gate)

- **D1 — Seeds policy:** all scrutiny-phase runs use the **single seed
  `SINGLE_SEED=38210573`** (dev profile, replication precedent). `SEED_LIST`
  3-seed averaging is reserved for post-selection full-profile runs. The
  known HyperOptSearch fixed-seed startup-trial sharing across combos
  (replication report ‡-note) is accepted with it.
- **D2 — Headline negatives stay uniform-99** — replication comparability
  with the paper and with our own frozen numbers (protocol strong default).
- **D3 — Popularity-sampled secondary readout: adopt.** `eval_neg_strategy=
  'popular'` is already implemented (`_neg_sample_popular`, pop^0.75); a
  secondary test-only pass on frozen checkpoints costs minutes per model.
  Run it once for the S0.7 reference fleet and report as a secondary table —
  addresses the popularity-blindness caveat of sampled uniform negatives at
  ~zero cost. (Sampled-metric distortion caveat: literature-known; exact
  citation to be verified before thesis use — GR6, no from-memory citations.)
- **D4 — User-activity-stratified test readout: adopt as charter option.**
  `Tester.get_test_logits(aggregated=False)` already returns per-row metrics
  in dataset order; a small script can bucket HR@10/NDCG@10 by train-history
  length. Nearly free and directly serves facet B (sparsity) claims. To be
  specified in S0.3 alongside the cold eval (shared machinery).
- **D5 — Metric set unchanged** (HR/NDCG@{1,3,5,10,50}). Sufficiency per
  facet: facet A (interpretability) is *not* measured by accuracy metrics by
  design — its instrument is the explanation pipeline + SC.5/SC.8 checks;
  facet B (cold/sparse) is currently **unmeasured** — that is exactly S0.3's
  deliverable, not a metric-set change.

### 6. Gaps we will NOT fix (mandatory declaration)

1. **Uniform-99 sampled eval is popularity-blind** — kept for paper/replication
   comparability; mitigated by the cheap D3 secondary readout.
2. **Test negatives may include the user's val item** — paper-faithful; ≤1
   item in ~13k–27k candidates, negligible.
3. **Repeat purchases structurally invisible** (dedup + set-based LOO) —
   deliberate protocol choice; metrics measure *discovery* recommendation
   only; honest thesis paragraph already planned (feature analysis Part E).
4. **H&M timestamps are date-only** — within-day ordering in the temporal LOO
   is arbitrary-but-frozen in the split artifacts; disclose in the Dataset
   chapter, no fix possible from the data.
5. **Train-unseen items in standard test sets** (F-S0-04) — paper-faithful;
   formalized deliberately in S0.3 instead of patched here.
6. **`min_delta=1e-4` + ASHA + dev profile deviations from the paper's
   training regime** — uniform across all compared rows → internally fair;
   already declared in `replication_report.md`; absolute paper-comparability
   caveat stands.
7. **HR@50/NDCG@50 weakly informative** with 100 candidates (near-saturated) —
   kept because it is the paper's metric set; just don't lean on @50 in prose.
8. **Beyond-accuracy metrics (coverage/diversity/novelty)** — not effectively
   free on a 100-candidate sampled eval (would need full-catalog scoring);
   out of scope per protocol default.
9. **MAP@12 contextual readout** (masterplan 3.5.1/3.5.2) — still unbuilt;
   positioning-only side note, irrelevant to candidate comparison; stays a
   masterplan item outside this protocol.

### Gate

**Status: awaiting user review.** Decisions D1–D5 and dispositions for
F-S0-02…F-S0-05 are the gate questions. F-S0-01 applied under the trivial-fix
policy. Approved non-trivial fixes (F-S0-03 assert) land in S0-build.
