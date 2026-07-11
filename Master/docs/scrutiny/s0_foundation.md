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

**Addendum (2026-07-11, later same day — VPN restored, F-S0-05 resolved):**
hm_1_month copied from LEO5 (`/scratch/c7031336/protomf_data/hm_1_month/`,
byte-identical sizes) and verified locally:
- Invariant holds exactly: val = test = **73,418** = n_users, one row per
  user; totals match the canonical stats (623,230 interactions, 13,651 items);
  `item_features.csv` covers `item_id` 0…13,650 exactly.
- Train-unseen items: **3 items → 9/73,418 test rows (≈0.012%)** — negligible
  on the primary dataset (amazon2014 remains the only materially affected set).
- Bonus S0.6 input: LEO5 quota check — **home is capped at 5 GB** (1.6 GB
  used); scratch (GPFS) has hundreds of TB free. The dc03 ~25 GB image
  re-download MUST target scratch, confirming the protocol's S0.6 note.

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

**Status: CLOSED — ratified by user 2026-07-11.** All decisions accepted as
recommended: D1 (single seed 38210573 for scrutiny phase), D2 (uniform-99
headline), D3 (popularity-negatives secondary readout at S0.7 — adopted),
D4 (user-activity-stratified readout, spec'd in S0.3 — adopted), D5 (metric
set unchanged). Finding dispositions: F-S0-02 wontfix (documented), F-S0-03
assert lands in S0-build (open until resolving commit), F-S0-04 wontfix
(routed to S0.3 + amazon thesis disclosure), F-S0-05 resolved same day
(V1 scp'd from LEO5). → Next step: S0.2.

---

## S0.2 Side reasoning (2026-07-11)

**Scope:** the written argument for *where* the four item-side-only design
candidates can honestly be expected to improve on the CF-only host, per
thesis facet — (A) intrinsic prototype interpretability via feature
grounding, (B) sparsity/cold-start robustness — and the formal disposition
of user-side features. Grounded in the double-tie wiring as implemented
(`feature_extraction/feature_extractor_factories.py`, `prototypes_double_tie`
branch, verified 2026-07-11), not in an idealized architecture.

### 1. Setting: one bilinear score, two blocks, one side touched

In the `user_item_proto` host the score is a dot product of concatenated
representations:

> s(u, i) = ⟨sim(e_u, P^u) ; e_u·W_u⟩ · ⟨e_i·W_i ; sim(e_i, P^t)⟩
> = **[Block U]** sim(e_u, P^u) · (e_i·W_i)  +  **[Block I]** (e_u·W_u) · sim(e_i, P^t)

Block I scores the item by its similarity profile over **item prototypes**
P^t, weighted by the user's learned affinity to each item prototype. Block U
scores the user's similarity profile over **user prototypes** P^u, weighted
by the item's learned projection onto user-prototype space. All four design
candidates (dc01–dc04) re-parameterize the **item branch only** — what feeds
sim(·, P^t) and/or P^t itself; the user branch (e_u, P^u, W_u) remains free
collaborative filtering everywhere. Two consequences follow immediately:
every grounding claim lives in Block I, and any user-side benefit must
arrive *indirectly*, through training dynamics, since no user-side
information is added.

### 2. Facet A — interpretability: what becomes grounded, what stays post-hoc

**Grounded (Block I, the candidates' contribution).** The item-prototype
*vocabulary* becomes feature-grounded: a prototype is readable as an
attribute profile (dc02, dc04: by construction; dc01: via shared geometry
between prototypes and feature embeddings; dc03: as a real exemplar
garment). Consequently the per-recommendation read-out û_k · t*_k — "this
item is recommended because it activates prototype k, and you have high
affinity for k" — is stated in a vocabulary the model itself computes at
inference time: intrinsic in the R2 sense, prototype-shaped in the R3 sense,
feature-grounded in the R4 sense. This is the honest form of the thesis's
interpretability claim.

**Deliberately CF (and staying so).** The user's affinity weights û_k =
(e_u·W_u)_k are learned collaborative quantities. The candidates ground
*what the prototypes mean*, not *why the user likes them*. This is not a
defect but the R7 balance point: recommendation quality is carried by CF
where CF is strong; transparency is achieved by expressing the decision in
user-perceivable item vocabulary. Thesis claims must not slide from
"explanations are stated in intrinsic, feature-grounded vocabulary" to "the
model's reasoning is fully feature-transparent" — the latter is false for
every candidate.

**Post-hoc (Block U, untouched).** User prototypes P^u remain CF clouds,
interpretable only through the existing post-hoc machinery (top-k user
profiling), and the item projection e_i·W_i (dc01: feature-composed but not
prototype-shaped; dc02/dc03/dc04: per design) carries score mass with no
prototype semantics at all. **Methodological consequence (binding for SC.5/
SC.8):** every per-recommendation read-out must report the *score share* of
Block I vs Block U. An "intrinsic explanation" that narrates 20% of the
score while 80% flows through the ungrounded block would be overclaiming;
the share must be measured on real checkpoints, not assumed. This check is
hereby part of the explanation-scrutiny standard.

**Scoping of thesis claims (quotable):** *the feature-aware candidates make
the item axis of ProtoMF's explanation intrinsically feature-grounded; the
user axis remains collaborative and post-hoc by design. In the fashion
domain this is the right asymmetry: items carry rich, human-legible
attributes (garment kind, colour, department), while users are described
only by thin demographics — the semantics worth grounding live on the item
side.*

### 3. Facet B — cold-start: addressable vs structurally out of reach

**Cold items — addressable, the facet-B claim surface.** Every candidate
can represent an item with zero training interactions from its features
alone: dc01 composes the item factor from feature embeddings (with
`use_id_feature` the ID row is additive, without it purely compositional);
dc02 has *no per-item parameters* — any item with an attribute signature is
scoreable; dc03 embeds the item's image through a frozen encoder; dc04
scores cold items via its anchor-average rule. The CF-only host, by
contrast, scores unseen items with random-init embeddings — i.e. noise
(S0.1/F-S0-04 measured exactly this accident). Per-candidate caveats the
S0.3 eval must be able to *detect*, not paper over: dc01's ID-keyed bias
channel (`dc01_feature_composed_bias_gap.md`), dc02's signature-collision
tie-cap (identical signatures ⇒ identical scores; number recomputed in
S0.5), dc03's dependence on image availability for the cold item, dc04's
anchor-average rule being an untrained-at-that-point convention.

**Cold users — structurally unaddressable, and unmeasurable here.** An
item-side design adds no user-side information: a user without training
history has only a random e_u, and no item representation can compensate in
a bilinear score. Three independent reasons close this door for the
protocol: (1) architectural — above; (2) data — H&M user features are thin
demographics (age, postal code, club status, news frequency), which the
Kaggle field used only for cold-user *popularity* fallbacks, i.e.
non-personalized heuristics, not representation learning; (3) protocol —
under the k-core + leave-one-out split every evaluated user has ≥3 train
interactions by construction (verified S0.1), so a cold-user metric does
not even exist in this eval design. Cold-user claims are therefore out of
scope for the thesis's quantitative story.

**Warm-but-thin users — indirect, plausible, measurable.** Feature-grounded
item representations smooth the item space (similar-attribute items get
similar representations), which should help most where user preference
estimates rest on the fewest interactions. This is an *indirect* expectation
— it must be phrased as such — and it is exactly what the D4
activity-stratified readout (spec in S0.3) measures: HR/NDCG by
train-history-length bucket, feature-aware vs CF-only. If the candidates
help, the gain should concentrate in the thin buckets.

### 4. Expected-impact map (facet × side)

| | Item side (dc01–dc04 touch this) | User side (untouched CF) |
|---|---|---|
| **A — intrinsic interpretability** | **Direct.** Item prototypes gain feature-grounded meaning; per-recommendation read-outs in item-attribute vocabulary (Block I). Score-share reporting mandatory. | **None.** User prototypes stay post-hoc CF clouds (Block U); existing top-k profiling remains their only interpretation. |
| **B — cold items** | **Direct.** Feature-only representation for unseen items; per-candidate caveats (bias channel, tie-cap, image availability, anchor rule) must be detectable by the S0.3 eval. | n/a (cold items are an item-branch matter). |
| **B — cold users** | **Structurally none.** No user information added; bilinear score cannot compensate. | **Not addressed** (out of scope; thin features; unmeasurable under the current split). |
| **B — warm-but-thin users** | **Indirect, plausible.** Feature-smoothed item space should help short-history users most; measured via D4 strata. | None beyond training dynamics. |
| **R6 — dense-regime accuracy** | **Risk axis, not a gain axis.** Grounding constrains the item space (fidelity-vs-accuracy tension); measured on the standard eval vs S0.7 references. | Unchanged by construction. |

### 5. User-side features: disposition

**Out of scope for this protocol (default upheld).** Promotion of user-side
features would be a new design-candidate cycle (requirements doc S2), not
scrutiny work. Reasons, in force order: (1) the semantics worth grounding
are item-side in this domain (facet A argument above); (2) H&M user features
are thin and demographically flavoured — low explanation payoff, privacy-
adjacent vocabulary ("recommended because you are 34 and live in …" is not
the thesis's explanation story); (3) the current eval cannot measure the
cold-user scenario user features would most plausibly serve; (4) protocol
budget — four candidates × twelve gates is the committed scope. This
confirms the standing decision in the protocol vault (2026-06-16_1807:
item-first, user candidate-dependent, context dropped). **Non-preclusion
(S2 kept alive):** the factory pattern keeps the user branch modular — a
later user-side extractor is a factory-level swap, and nothing in dc01–dc04
hard-codes against it.

### Findings

None. This step produced reasoning and one binding methodological
requirement (Block-I/Block-U score-share reporting, folded into SC.5/SC.8
practice), but no defects in code or documents.

### Gate

**Status: awaiting user review.** Review points: (a) the Block-I/Block-U
decomposition and its score-share reporting requirement, (b) the claim
scoping for facet A (vocabulary grounded, personalization CF), (c) the
cold-user out-of-scope argument, (d) the user-side disposition.
