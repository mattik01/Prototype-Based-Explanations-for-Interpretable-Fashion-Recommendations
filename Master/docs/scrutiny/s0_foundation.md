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

**The cold-start 2×2 (added at gate review, 2026-07-11).** The bilinear
score makes the scoping exact: each side of ⟨u, i⟩ can draw its meaning
from interactions *or* features, and the product is meaningful iff both
sides are. Warm-user × cold-item: item features suffice — the user's warm
history already encodes their taste *in item-prototype terms*, and the cold
item is placed into that same space from attributes (learned-taste-to-
attribute matching through the shared prototype space, not attribute-to-
attribute). Cold-user × warm-item: needs user-side features; item-side
grounding cannot rescue a dot product with a noise vector. Cold × cold:
needs both sides featurized — the pure "feature matching mode." The
candidates address exactly the first cell; the domain cooperates (fashion
catalogs churn items weekly — H&M is item-cold — while cold users were
0.7% in the Kaggle field and are absent from our split entirely).
Follow-ups captured to the design scratchpad (`[captured scrutiny s0]`):
a dual-side attribute-aware candidate (mirror vs symmetric-at-the-tie
routes) and a realistic 3-scenario cold testbed.

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

**Status: CLOSED — reviewed by user 2026-07-11.** Facet A judged
"super important" and logged to the protocol vault
(`2026-07-11_1526_facet-a-claim-boundary-coldstart-2x2.md`). Facet B
discussion sharpened the argument into the cold-start 2×2 (amended into §3
above). User directives captured to the design scratchpad: dual-side
attribute-aware candidate (two routes, both open) + realistic 3-scenario
cold testbed — both explicitly *eventual*, not scrutiny-scope. User-side
disposition upheld. → Next step: S0.3.

---

## S0.3 Cold-start eval spec (2026-07-11)

**Scope:** the implementable specification of the cold-item evaluation every
model runs (warm-user × cold-item cell of the S0.2 2×2), plus the D4
activity-stratified readout it shares machinery with. Prior art honored:
dc01 design doc §3.6 (2026-06-11 amendment: held-out item split and/or tail
stratification), dc02 design doc §3.6 (attribute-kNN fallback baseline #7 +
full-catalog tie diagnostic, 2026-07-07 amendments), F-S0-03/F-S0-04
constraints, `dc01_feature_composed_bias_gap.md`.

> **Revision 1 (2026-07-11, gate review — user directive: anchor to the
> LightFM paper's published cold-start protocol).** B5 (Kula 2015) §5 was
> read in full before this revision (GR6). Its protocol: *"all interactions
> pertaining to 20% of items are removed from the training set and added to
> the test set"*; evaluation ranks **within the cold pool** (per-user mean
> ROC AUC over the test set — the "pool of items for which no collaborative
> information has been gathered"); 10 random split repetitions; d fixed.
> Changes to this spec: **(1)** fraction 5% → **20%** (paper-faithful;
> collateral re-measured); **(2)** primary cold ranking flipped to
> **cold-vs-cold** — the original draft rejected it as "unrealistic," but
> the seed paper makes it the headline and it is *cleaner* (removes the
> cold-warm score-calibration confound; it is Lyst's new-arrivals scenario);
> cold-vs-all demoted to secondary diagnostic; **(3)** ~~sampled AUC added
> as a companion metric~~ *(proposed in rev. 1, dropped at the same gate by
> user decision: B5's absolute AUC doesn't transfer to a 100-slot sampled
> eval, sampled AUC is a linear function of mean rank (redundant with the
> HR/NDCG family), and all cold comparisons are internal — the B5 anchoring
> lives in the split design and ranking pool, not the metric; HR@10 chance
> level on cold-vs-cold is a clean 0.10 noise floor)*. Declared *non*-adoptions, argued:
> test-set early stopping (B5 stops on test performance — leakage by modern
> standards; we stop on warm val), atemporal random splits (we derive from
> the host protocol's temporal artifact — ProtoMF fidelity outranks B5
> here), and 10 split repetitions (10× retraining vs the D1 single-seed
> budget; mitigated by per-item bootstrap CIs, with 3 repetitions optional
> at full-profile stage). Edits below marked *(rev. 1)*.

### 1. Design overview — two strata, two costs

- **Stratum TC (true-cold, the headline R5 instrument):** a derived dataset
  variant `hm_1_month_cold` in which a seeded, popularity-stratified **20%**
  of items *(rev. 1 — LightFM §5 fraction)* have ALL their interactions
  removed from train/val; those removed interactions become the cold test
  rows. Requires one cheap retrain per model (single config, no search —
  §6). This measures the real thing: scoring items the model has *never
  seen interact*.
- **Stratum TW (tail-warm) + D4 user strata (free readouts):** the standard
  test rows of the *canonical* runs, sliced by the positive item's train
  popularity (item buckets: 1–5, 6–20, 21–100, >100) and by the user's
  train-history length (user buckets: quartiles). Zero training cost — a
  per-row readout on existing checkpoints (ProtoCF-style stratification;
  covers "popularity tail" without new runs).

Rationale for held-out rather than temporal cold definition: on V1 the
1-month window is an alive-catalog by construction (98.9% of last-week
transactions are on items already active the prior week — 4.0 analysis), so
"items first appearing after the train cutoff" yields a tiny,
popularity-skewed cold set. The held-out design gives controlled size and
popularity coverage; the temporal-realistic 3-scenario testbed is captured
in the scratchpad as future work. Declared deviation, not hidden.

### 2. Cold set + variant construction (deterministic, from the frozen artifact)

Derived by a committed generator script from the canonical `hm_1_month`
split — **never re-split from raw, never re-k-cored** (the frozen artifact
is the single source; re-coring would silently reshape the warm regime and
delete cold candidates — leakage checklist item 3).

1. Popularity deciles over items with ≥1 train interaction; sample **20%**
   per decile with `numpy default_rng(38210573)` → cold set **C** *(rev. 1;
   measured 2026-07-11: **2,730 items**; stratified sampling is a
   variance-reduction refinement of B5's uniform random draw — identical
   marginal inclusion probability, and it enables popularity-sliced cold
   reporting)*.
2. `train/val := canonical minus rows with item ∈ C`. Removed rows
   *(rev. 1, measured: 93,805 train (19.7%) + val + test = **122,808**)*
   become `cold_test.csv` (user, item, original-split tag).
3. Cold rows of users left with **0** train interactions are dropped
   *(rev. 1, measured: 167 users of 73,418 ≈ 0.2%; a further 11,905 users
   fall below 3 remaining train rows — accepted, uniform across all models,
   reported by the generator; B5 accepts the same distortion silently)*.
4. **ID universe unchanged:** `user_ids.csv`, `item_ids.csv`,
   `item_features.csv` copied byte-identically from canonical — same
   n_items, same integer ids, same feature tensors, table shapes identical.
   Warm test := canonical test minus cold-positive rows.
5. Artifacts: `data/hm_1_month_cold/` (gitignored like all splits) +
   `cold_items.csv`; the generator script + seed are committed, so the
   variant is exactly reproducible; scp to LEO5 alongside.

> **Edit note (2026-07-11, S0-build):** the rev. 1 "measured" row-level
> numbers in §2/§3 came from an ad-hoc in-session computation whose script
> was not committed; the committed generator (`data/hm/make_cold_variant.py`,
> commit 7f7f24a) reproduces the ratified **2,730-item / 13,648-eligible**
> cold set size exactly but draws a different C, so the incidental
> collateral shifts: removed train **92,817 (19.48%)**, cold rows
> **121,024**, zero-train collateral **143 users / 515 rows**, variant val
> **59,066**, warm test **59,048**, users below 3 train rows **11,714**.
> The committed, reproducible draw is canonical from here on; details in
> the S0-build section (deviation 1), decision at the S0-build gate.

### 3. Eval semantics on the variant

- **Primary — cold-vs-cold** *(rev. 1, LightFM-faithful)*: cold row = 1 cold
  positive + 99 negatives drawn **from C** (the cold pool; 2,730 items make
  99 draws unproblematic), excluding the user's canonically-consumed items.
  This is B5's headline setting ("recommendations from a pool of items for
  which no collaborative information has been gathered" — the new-arrivals
  scenario) and the cleaner isolation: every candidate slot is feature-only,
  so cold-warm score-calibration differences cannot masquerade as cold
  skill. CF-only baselines degrade to chance here — exactly B5's own
  finding ("MF performs no better than random in the cold-start case").
- **Secondary — cold-vs-all** *(rev. 1, demoted)*: same positives, 99
  negatives from the FULL catalog (warm + cold), same exclusion rule. The
  deployment-shaped diagnostic ("can a new item surface against the whole
  catalog"); explicitly confounded by score calibration across the
  cold/warm boundary (e.g. dc01's dead bias channel would surface *here*,
  not in cold-vs-cold) — that confound is the point of keeping it.
- **Negative exclusion (both):** the user's train+val+test consumption from
  the *canonical* histories, so no user-consumed item can appear as a
  negative — leakage item 5.
- **Metrics:** the standard HR/NDCG@{1,3,5,10,50} machinery on the 100-slot
  rows, and nothing else (gate decision — see the rev. 1 banner note);
  headline cold numbers HR@10/NDCG@10 over cold rows. Chance level on
  cold-vs-cold: HR@K = K/100 — the CF noise floor is directly readable.
- **Divisor (F-S0-03 resolution design):** the `Evaluator` divides by the
  **counted number of evaluated rows** and asserts it equals the
  dataset-declared expectation. On canonical datasets rows == n_users, so
  all existing numbers are unchanged; on the variant *(rev. 1: val 59,052
  rows, warm test 58,781 rows, cold test 122,808 rows)* it is correct
  automatically.
- **Reporting:** cold slice (overall + by original-popularity decile of the
  cold item), warm-test of the variant (sanity: removal must not distort
  the warm regime — compare against canonical warm numbers), row counts
  always printed.

### 4. Leakage checklist (mandatory walk)

1. **Attribute availability at scoring time:** item attributes come from the
   static `articles.csv` catalog → available for cold items by nature;
   `item_features.csv` is copied unchanged (cold items keep their rows). ✅
2. **Transaction-derived features train-only:** `price_band` (built,
   unwired) would have to be computed from the *variant's reduced train*
   transactions only — under which cold items have **no price at all** and
   need an explicit "unknown band" policy. → recorded as an input to S0.5's
   price_band decision; no transaction-derived feature may enter before that
   policy exists. ⚠️ handed to S0.5.
3. **K-core must not silently delete the cold set:** no re-coring (§2);
   collateral measured and reported (2,911 users drop below 3 train rows —
   accepted, uniform across all models; 2 users dropped). ✅
   *(Edit note 2026-07-11, S0.7: the parenthetical figures are stale rev-0
   (5%-draft) text the rev. 1 pass missed; the committed draw's collateral is
   11,714 users below 3 train rows / 143 zero-train users dropped — see the
   §2 edit note. Caught by the S0.7 black-box audit.)*
4. **No future information in features:** attributes are timeless catalog
   properties; no temporal fields enter. ✅
5. **Eval negatives for cold rankings:** exclusion by canonical consumption
   (§3); verified impossible for a user's own future purchase to appear as
   its negative. ✅
6. **Training negatives (new item, found in this walk):** cold items remain
   in the catalog at training time and would be drawn as *training
   negatives* — the model would learn "cold items are disliked" (CF rows
   pushed negative; feature embeddings polluted). The variant's training
   therefore **excludes C from the negative-sampling distribution**
   (`p[C]=0` — mimics "not yet in catalog during the training period").
   Eval negatives keep the full catalog (item just launched, competes with
   everything). ✅ by construction once implemented.
7. **Model selection blind to cold:** val contains no cold positives (§2.2)
   and early stopping stays on warm `hit_ratio@10` — configs cannot be
   selected *for* cold performance. ✅

### 5. How every model scores cold items (conventions — all rows defined)

| Model row | Cold representation | Notes |
|---|---|---|
| `mf`, `acf`, `user_proto`, `item_proto`, `user_item_proto` (CF-only) | **(i) Noise floor:** the untrained random-init embedding, as the architecture actually behaves. **(ii) Attr-kNN fallback** (dc02 §3.6 baseline #7): cold item's representation := mean of its top-n (n=20) attribute-cosine warm neighbors' *trained* item representations, patched in at eval time. | Both reported; (i) is the honest architectural answer, (ii) the strongest cheap competitor. Zero-vector convention rejected (degenerate under cosine). |
| Popularity reference | train popularity score; cold items share popularity 0 → bottom ties | floor reference row |
| CBF baseline (S0.4) | native — features only | defined in S0.4 |
| dc01 | native: feature composition; ID column dropped at cold inference (design doc §3.4/§3.7.6) | **Bias-channel detection note:** `use_bias=0` in `base_param` — channel currently inert in every scored run. The runner must *record* `use_bias` per run and refuse the cold eval (loud error) if a config enables it without a declared cold-bias policy (`dc01_feature_composed_bias_gap.md`: per-ID bias of a cold item is an untrained zero — a silent depressant). |
| dc02 | native: bottleneck over attributes (no per-item params) | **Tie-block diagnostic** (dc02 §3.6 amendment): report signature-class statistics of full-catalog top-10 (sampled ≥5k users) — implemented as a model-agnostic diagnostic module, run for every model on the variant. |
| dc03 / dc04 | native per design (frozen image embedding / anchor-average rule) | dc03 requires the cold item's *image* — availability asserted at eval time (ties into the S0.6 image precondition). |

> *(Edit note 2026-07-11, S0.7: the CF attr-kNN convention is implemented as
> S0-build ratified deviation 3 — the patch target is the base ID-embedding
> rows, with the model's representation function then applied to the patched
> base. For nonlinear branches (acf, prototype models) this is deliberately
> NOT identical to averaging final representations; the base-patch reading is
> the ratified one. Noted here so future audits don't re-flag it.)*

### 6. Training protocol on the variant (budget-honest)

No second hyperopt. Per model: take the **best config from the canonical
hm_1_month search** (existing S0.7 references / the candidate's SC.6 run),
retrain **once** on `hm_1_month_cold` (same seed 38210573, dev-profile
epochs, early stopping on the variant's warm val), then run warm-test +
cold-test + diagnostics. Defensible: configs are selected on warm data
(blind to cold, §4.7), and the cold eval measures how the *selected* model
generalizes. Cost: ≤~1 h/model on LEO5 (single trial) — trivially inside
the envelope. The retrained-variant checkpoints are frozen alongside the
S0.7 references. *(rev. 1)* Early stopping runs on the variant's **warm
val** — deliberately NOT B5's test-based stopping (leakage by modern
standards; declared improvement over the seed protocol). Split repetitions:
**one seeded split** (D1 budget) vs B5's 10 — variance reported instead via
**per-item bootstrap CIs** over the 2,730 cold items (free); 3 split
repetitions remain an option for post-selection full-profile runs.

### 7. D4 stratified readout (shared machinery)

One script, canonical runs, no retraining: per-row metrics via
`Tester.get_test_logits(aggregated=False)` joined with per-user
train-history length and per-item train popularity → HR@10/NDCG@10 by user
quartile (facet-B warm-thin claim) and item bucket (stratum TW). Output: one
table per model, assembled into the comparison at SC.8.
*(Edit note 2026-07-11, S0.7: implemented via the cold_eval `evaluate_rows`
path over a fresh seeded test dataset rather than the method named above —
`Tester.get_test_logits` has no `aggregated` kwarg. Functionally equivalent
per-row metrics; negatives are re-drawn (seeded), so late-decimal differences
vs a run's `test_metrics.json` are expected. Caught by the S0.7 audit.)*

### 8. Module touch-points (S0-build map)

1. `data/hm/make_cold_variant.py` (new) — generator per §2, prints the
   collateral report, writes `cold_items.csv`.
2. `rec_sys/protomf_dataset.py` — (a) optional excluded-items mask in
   `_neg_sample_uniform/_neg_sample_popular`; (b) a cold-test mode reading
   `cold_test.csv` with canonical-consumption exclusion sets.
3. `utilities/eval.py` — divisor = counted rows + expected-rows assert
   (resolves F-S0-03).
4. `utilities/cold_eval.py` (new) — runner: load checkpoint → warm test,
   cold test, attr-kNN patch for CF rows, tie-block diagnostic, popularity-
   decile slicing; refuses `use_bias≠0` without policy.
5. `Master/scripts/stratified_readout.py` (new) — §7, works on any results
   dir.
6. `start.py`/`run_combo.py` — register `hm_1_month_cold`; a
   `--retrain-config` path for the single-config variant runs.
7. Tests in `Master/temp/dc_checks/s0/` — generator determinism +
   collateral counts; divisor/assert semantics; negative-mask correctness
   (no cold item sampled in training negatives; no consumed item in eval
   negatives); attr-kNN patch shape/identity checks; F=0-style sanity that
   canonical-dataset numbers are bit-identical pre/post Evaluator change.

### 9. Declared alternatives not taken

- **Temporal cold definition** — underpowered on V1 (§1); future testbed
  captured in scratchpad.
- ~~**Cold-vs-cold ranking** — unrealistic headline; derivable later.~~
  *(rev. 1: judgment REVERSED at gate review — cold-vs-cold is B5's
  published headline and the cleaner isolation; now primary (§3).
  Cold-vs-all demoted to secondary. The reversal is recorded, not hidden.)*
- **Full re-split with cold-aware k-core** — would reshape the warm regime
  and break the frozen-artifact lineage.
- **Hyperopt on the variant** — pure budget burn; selection must stay blind
  to cold anyway.
- *(rev. 1)* **B5's test-based early stopping and atemporal random splits**
  — not adopted; argued in the Revision-1 banner.

### Gate

**Status: CLOSED — ratified by user 2026-07-11 (rev. 1 + AUC-drop applied
at gate).** Ratified: (a) held-out **20%** popularity-stratified cold set,
B5-faithful (2,730 items / 122,808 cold rows / 167 users collateral);
(b) **cold-vs-cold primary** + cold-vs-all secondary, canonical-consumption
exclusion; (c) training-negative exclusion of C; (d) single-config retrain
convention; (e) Evaluator divisor change + assert (F-S0-03 resolution);
(f) CF cold conventions = noise floor + attr-kNN fallback (n=20);
(g) price_band cold policy → S0.5; (h) metric set HR/NDCG only.
**Scrutiny-phase budget principle (user directive at this gate, feeds the
S0.6 charter):** dev hyperopt profile, single seed, no split repetitions —
ALL repetition/multi-seed work deferred until final or near-final candidate
versions stand. → Next step: S0.4.

### S0.3 addendum — ml-1m cold variant (2026-07-12, re-run mode `step0.3 s0`)

**Scope:** the cold-item eval spec extended to the second testbed (charter
C2/C5/C6 amendments 2026-07-12): `ml-1m_cold` variant definition, the
generalization plan for `data/hm/make_cold_variant.py`, the ml-1m leakage
walk including the **written Tag-Genome provenance disposition**, and the
model-conventions deltas. Anchoring unchanged: B5 §5's published protocol
(20% of items, interactions removed, cold-vs-cold primary ranking) — the
same source the H&M spec anchored to at rev. 1; every rev.-1 gate decision
(fraction, ranking pools, no test-based stopping, single seeded split,
warm-val early stopping, single-config retrain) carries over verbatim.

**A1 — Variant definition (`data/ml-1m_cold/`).** Identical machinery, no
new concepts: derived from the frozen canonical ml-1m split (the
replication artifact — never re-split, never re-5-cored), popularity-
stratified 20% draw per decile with `default_rng(38210573)`, removed
train/val rows + canonical cold test rows → `cold_test.csv`, zero-train
collateral users dropped and reported, ID universe copied byte-identically.
Expected magnitudes (**illustrative only** — the committed generator's
printed report is canonical when it runs at the S0-build extension, per the
S0-build learnings rule): eligible = 3,125 (F-S0-04 measured 0 train-unseen
items on ml-1m) → **C ≈ 625 items**; expected removed train ≈ 20% of
562,308 ≈ 112k rows; val/test removals ≈ 20% of 6,034 rows each.

**A2 — Generator generalization plan.** Verified against the code this
session: `make_cold_variant.py` is **already dataset-agnostic in its core**
— generic split filenames, `user_id`/`item_id` columns (present in the
ml-1m files, checked), `--canonical`/`--out` args; the H&M coupling is only
its defaults and its `data/hm/` location. Plan (all landing at the S0-build
extension, gated there):
1. **No draw-logic change.** Invocation:
   `python data/hm/make_cold_variant.py --canonical data/ml-1m --out
   data/ml-1m_cold`. The script stays in `data/hm/` (moving it would churn
   provenance for zero behavior gain); a one-line docstring note declares
   dataset-generality.
2. **Precondition, sequencing-critical:** the model-grade ml-1m
   `item_features.csv` (bag layout, genres+tags@0.8) must exist BEFORE
   generation — the generator copies it byte-identically into the variant
   (S0.3 §2.4). Order inside the S0-build extension: builder (d) → cold
   generator run (e). The current display-grade file must NOT be the one
   copied; the generator gains a loud error naming the precondition if the
   file is absent.
3. **New assert — cold-pool sufficiency (ml-1m-specific hazard):** with
   |C| ≈ 625 and heavy histories (max |H_u| = 1,415 of 3,125 items), a
   user's canonical consumption shrinks their cold-vs-cold negative pool.
   Worst case ≈ 625 − 20%·1,417 ≈ 342 ≥ 100 — safe, but by arithmetic, not
   by construction. Spec: `ColdTestDataset` asserts per-row available pool
   ≥ NEG_VAL+1 (fails loud at eval time); the generator report prints
   min-pool-over-cold-rows at generation time. Covered by a dc_checks/s0
   test. (Not an issue on H&M: |C| = 2,730, median |H| = 5.)
4. **Registration:** `ml-1m_cold` in start.py/run_combo.py (mirrors
   `hm_1_month_cold`); scp variant to LEO5 alongside the canonical ml-1m
   copy before any cluster retrain cites it.

**A3 — Eval semantics deltas.** None conceptual: cold-vs-cold primary
(chance floor HR@K = K/100 unchanged), cold-vs-all secondary,
canonical-consumption exclusion, training-negative exclusion of C via the
`cold_items.csv` marker (dataset-generic, verified in S0-build), warm-val
early stopping, metrics HR/NDCG only. The tie-block diagnostic runs
unchanged and inherits the S0.5-addendum signature numbers (3.8% catalog
twins — mostly the genres-only tail — vs H&M's 67.5%; the within-C sampled
tie rate is reported by the diagnostic itself, expected negligible).
Retrain convention unchanged (best canonical config, one retrain, dev
profile, seed 38210573) — its inputs are the lazily-submitted ml-1m fleet
hyperopts (C6 amendment), so ml-1m cold retrains follow whenever a run plan
first cites those rows.

**A4 — Leakage checklist, walked for ml-1m.**
1. **Attribute availability at scoring time ✅ with a written provenance
   disposition (the roadmap's mandate).** Genres: static catalog properties
   (`movies.dat`) — available for cold items by nature. **Tag-Genome:** the
   staged 2014 dataset (README read this session) is a fixed per-(movie,
   tag) relevance table computed by GroupLens "using a machine learning
   algorithm on user-contributed content including tags, ratings, and
   textual reviews" (Vig, Sen & Riedl 2012, TiiS — the dataset's own
   citation). Disposition, three parts: **(i) No split leakage:** the table
   is a static item-side artifact, invariant to our train/val/test
   partition — it cannot encode which of OUR rows went where; identical
   rows are served for cold and warm items (byte-identical copy, §2.4).
   **(ii) Temporal provenance, disclosed:** the genome (2012/2014)
   post-dates ml-1m's 2000–2003 interactions — a system deployed in 2003
   could not have had it. Declared deviation with precedent: B5 §4.1 used
   the genome on MovieLens the same atemporal way; our features are treated
   as timeless catalog descriptors on both testbeds, and all compared
   models see the same features. **(iii) Community-preference content,
   disclosed:** the genome is computed partly from ratings/tagging
   activity, so a movie's tag-bag richness encodes its historical
   popularity (measured: token count vs train popularity ρ = 0.672, S0.5
   addendum — the standing must-mention channel). Consequence for the cold
   claim's wording: ml-1m cold items are **catalog-established movies with
   mature, popularity-correlated metadata, artificially interaction-hidden**
   — NOT a new-release simulation; a genuinely new movie would enter with a
   thin tag bag. Same held-out (not temporal) design as H&M, but on H&M the
   attributes are birth-time catalog properties while here metadata
   richness is itself an outcome of exposure. This sentence accompanies
   every ml-1m cold-result presentation (routes with the hidden-effects
   material).
2. **Transaction-derived features:** none exist on ml-1m (no price
   analogue; genres/tags static) — the canonical set stays leakage-clean by
   construction. ✅
3. **No re-coring:** generator filters rows of the frozen artifact only. ✅
4. **No future information in features:** subsumed by 1(ii) — the one
   temporal nuance is disclosed there. ✅ (as disclosed deviation)
5. **Eval negatives for cold rankings:** canonical-consumption exclusion +
   the NEW pool-sufficiency assert (A2.3). ✅ once implemented
6. **Training negatives:** C excluded from the training distribution via
   the marker — dataset-generic, already built and tested. ✅
7. **Model selection blind to cold:** variant val contains no cold
   positives; early stopping on warm `hit_ratio@10`. ✅ by construction

**A5 — How every model scores cold items (deltas only).** The S0.3 §5
conventions table transfers row-for-row; ml-1m rows are the C6-amended
fleet (`mf`, `user_proto`, `item_proto`, popularity, `lightfm_tags`,
`lightfm_tags_ids`) plus dc05's arms (and dc01's at its resume). Machinery
verified dataset-generic: the `use_bias` refusal, `config_expects_id_drop`
+ ID-row drop, popularity reference row. **One named gap:** the attr-kNN
cold fallback for CF rows builds item signatures via `build_attr_multi_hot`,
which assumes one value per field — it needs the bag-layout extension
(multi-hot over the genres+tags vocab) before ml-1m cold evals of CF rows
can run; scoped to the S0-build extension component (a/d), keystone: on
H&M inputs the extended builder is bit-identical to the current one.

**Findings:** none new — no code touched this step; the two
implementation-bearing items (pool assert A2.3, bag multi-hot A5) are
scoped into the already-planned S0-build extension components rather than
filed as ledger findings (they are new-scope work, not defects).

### Gate (addendum)

**Status: CLOSED — ratified by user 2026-07-12, all four items.**
(a) `ml-1m_cold` via the identical B5-anchored machinery (magnitudes
illustrative until the committed generator runs); (b) generalization plan
incl. the item_features precondition ordering and the pool-sufficiency
assert; (c) the leakage walk — the three-part Tag-Genome provenance
disposition and the **"interaction-hidden, not new-release" cold-claim
wording** (accompanies every ml-1m cold-result presentation); (d) attr-kNN
bag gap scoped to the S0-build extension. → Next: S0-build extension
(bag layout, builders, field plumbing, model-grade item_features, cold
generator run — one gate per component).

---

## S0.4 CBF baseline spec (2026-07-11)

**Scope:** the pure-content baseline that separates "features help" from
"prototypes help" in the results table. Prior art: dc01 design doc §3.6
("LightFM-style MF: the `mf` model with the same `FeatureEmbedding` item
branch but *no* prototype layer"), dc02 §3.6 (LightFM as external
feature-aware baseline), B5 §5 model roster (read for S0.3 rev. 1), vault
note 2026-06-16_1824 (FM/NFM/DeepFM/NCF+ shortlist).

> **Revision 1 (2026-07-11, gate review — user directive: baseline suite =
> published rosters).** Suite framing fixed to "all models from the two
> source papers, run under OUR protocol," for reviewer-facing credibility:
> the ProtoMF five (already the reference fleet) + **two models from B5**,
> chosen per user directive as *one strong feature-aware baseline* —
> **LightFM(tags+ids)**, B5's best MovieLens variant — and *one
> feature-only baseline* — **LightFM(tags)** (item = attributes only,
> cold-capable by construction). LightFM(tags) is preferred over LSI-LR for
> the feature-only slot on isolation grounds: the candidates are
> attribute-vocabulary models trained collaboratively, so LightFM(tags)
> differs from them in exactly ONE thing (no prototype layer), while LSI-LR
> differs in two (no prototypes AND content-only estimation). Exclusions,
> each one sentence: **LightFM(tags+about)** — requires user metadata text;
> H&M has none and the user side is closed (S0.2). **LSI-LR / LSI-UP** —
> B5's own results show them dominated in every scenario; their published
> defeat is cited, not re-run (and LSI-LR would be the costliest alien
> pipeline in the fleet for a strawman row). **Naming:** the working name
> `feature_mf` is retired — the rows are named what they are:
> **`lightfm_tags`**, **`lightfm_tags_ids`** (ft_type `lightfm`), so code,
> results dirs, W&B runs, and thesis tables all read "LightFM". Bias-free
> deviation from B5's published form stays declared (fleet parity; vault
> entry 2026-07-11_1556). Edits below marked *(rev. 1)*.

### 1. The isolation argument decides the count: ONE trainable model

The results table needs one missing row-type: **features without
prototypes**. With it, the ladder isolates every effect:

| row | item branch | isolates (by delta) |
|---|---|---|
| `mf` | ID embedding | CF floor |
| **`lightfm_tags`** *(rev. 1 name)* | features only | "features alone" vs CF floor — the feature-only baseline |
| **`lightfm_tags_ids`** *(rev. 1 name)* | features + ID | ID/feature complementarity (B5 Table 1's contrast) — the strong feature-aware baseline |
| `item_proto` / `user_item_proto` | ID + prototype layer | "prototype layer alone" |
| candidates dc01–dc04 | features + prototype layer | "the thesis mechanism" vs all of the above |

One model, two configs (`use_id_feature` flag), no other machinery
*(rev. 1: the two configs ARE the two B5 roster picks)*:
- **attribute-kNN is NOT built as a second standalone recommender** — it
  already exists as the S0.3 eval-time cold-fallback convention on CF rows
  (dc02 baseline #7). Warm-regime isolation is `feature_mf`'s job; the
  cheap cold competitor is the kNN rows. Count justified: 1.
- **FM / NFM / DeepFM / NCF+** (vault 2026-06-16_1824) are *thesis-level
  external baselines* — expressiveness foils, not isolation instruments;
  explicitly deferred out of scrutiny scope (budget; no isolation value the
  ladder above doesn't already provide).

### 2. Model definition — `lightfm` ft_type *(rev. 1 name; was `feature_mf`)*

- **User branch:** plain `Embedding` (free CF — consistent with S0.2's
  side disposition; B5's users are indicator-only in its tags/tags+ids
  variants, so this matches the seed).
- **Item branch:** the dc01-tested `FeatureEmbedding` (sum of feature-value
  embeddings over the **S0.5 canonical field set** — dependency declared;
  `q_i = Σ_f e_f`, optional per-item ID row via `use_id_feature`).
- **Score:** plain dot product (the `mf` path). **No prototype machinery, no
  biases** — `use_bias=0` fleet parity. Declared deviation from B5's exact
  form `σ(q_u·p_i + b_u + b_i)`: LightFM's feature-composed biases are part
  of ITS cold mechanism; our whole fleet runs bias-free, so parity outranks
  form-fidelity (and avoids importing the dc01 bias-gap asymmetry into the
  baseline — `dc01_feature_composed_bias_gap.md`).
- **Configs** *(rev. 1 roles per user directive)*: **`lightfm_tags`**
  (`use_id_feature=False`) — the feature-only baseline, natively
  cold-capable; **`lightfm_tags_ids`** (`use_id_feature=True`) — the strong
  feature-aware baseline (B5's best variant), cold inference via dc01's
  ID-column-drop convention.

### 3. Implementation slot (concrete)

1. `feature_extraction/feature_extractor_factories.py`: new `ft_type ==
   'lightfm'` branch *(rev. 1 name)* — `detached`-style: user `Embedding`, item
   `FeatureEmbedding(n_items, feature_ids, n_features, d, use_id_feature)`.
   Pure reuse of existing, dc01-tested classes; no new module.
2. `feature_extraction/feature_ids.py` `inject_feature_ids`: add
   `'lightfm'` to the injection guard (same non-mutating discipline —
   tensor never serialized).
3. `confs/hyper_params.py`: `lightfm_tags_ids_hyper_params` mirroring
   `mf_hyper_params`' search space exactly (same emb-dim range, loss,
   optimizer, negatives) + `feature_fields` (canonical set) +
   `use_id_feature=True`; `lightfm_tags_hyper_params` sibling via
   `copy.deepcopy` with the flag off (the dc01 ablation pattern).
4. `start.py` / `run_combo.py`: register model names `lightfm_tags`,
   `lightfm_tags_ids`; add `lightfm` to CLAUDE.md's reserved `ft_type`
   list at S0-build.
5. Tests `Master/temp/dc_checks/s0/`: **keystone — `lightfm` ft_type with
   `use_id_feature=True` and zero fields reduces bit-identically to `mf`**
   (the FeatureEmbedding F=0 equivalence, already proven for dc01 — re-run
   in this wiring); shape checks; cold-path check (tags-only scores an
   unseen item finitely and non-constantly); injection-seam no-mutation
   check.

### 4. Parity contract

Same splits (canonical + `hm_1_month_cold`), same NEG_VAL=99 uniform
negatives, same HR/NDCG metric set, same dev hyperopt profile and single
seed (S0.3 gate budget principle), cold eval per S0.3 (both rankings, D4
readouts). Runs in S0.7's reference freeze alongside the CF fleet and
popularity row.

### 5. Cost estimate

Two dev-profile hyperopts on hm_1_month (search space ≈ `mf`'s → ~2–3 h
each on LEO5, the lightest model class in the fleet) + two single-config
cold-variant retrains (≤1 h). Comfortably inside the envelope.

### Gate

**Status: CLOSED — ratified by user 2026-07-11 (rev. 1 applied at gate).**
All items (a)–(f) accepted: published-roster suite (ProtoMF five + B5 two,
exclusions documented), lightfm_tags / lightfm_tags_ids picks, bias-free
parity deviation, S0.5 field-set dependency, `lightfm` naming, attr-kNN as
eval-time convention with FM/NFM/DeepFM/NCF+ deferred. Decision logged to
the protocol vault (2026-07-11_1629_baseline-suite-published-rosters).
Bias-channel discussion at this gate logged separately
(2026-07-11_1556 — fleet stays bias-free; evidence-triggered ablation).
→ Next step: S0.5.

---

## S0.5 Feature audit (2026-07-11)

**Scope:** entry-correctness audit of the feature plumbing, the canonical
field-set decision, written dispositions for `price_band` / `product_code`,
and the collision-rate recompute dc02's SC.1a must inherit. All numbers
measured on V1 (`hm_1_month`, 13,651 items) this session.

### 1. Entry correctness (`feature_extraction/feature_ids.py`)

Audited line-by-line (full read at session start):

- **Offsets / vocab construction:** per-field lexicographically-sorted local
  vocab + running global offset — deterministic, disjoint blocks. ✅
- **`item_id` coverage guard:** both builders enforce exact 0…n−1 coverage
  (missing/extra/duplicate all caught) and re-sort rows to id order. ✅
  V1 verified: `item_features.csv` covers 0…13,650 exactly (S0.1 addendum).
- **Serialization boundary:** `inject_feature_ids` returns shallow copies;
  tensor never enters the caller's dict → never reaches `config.json`
  (P6 discipline; trainer/tester call sites verified in S0.1). ✅
- **Cold-variant consistency (new check):** vocab derives from the split's
  own `item_features.csv`; the S0.3 variant copies that file byte-identically
  → identical vocab, offsets, and tensors across canonical and cold runs. ✅
- **⚠️ NaN-policy asymmetry (→ F-S0-06, minor):** `build_attr_multi_hot`
  raises on NaN cells; `build_feature_ids` does `astype(str)` first, so a
  NaN would be **silently encoded as a legitimate `'nan'` vocab value**.
  Currently inert — V1 has 0 NaN cells across all 9 columns (measured
  today; consistent with the 4.0 no-missingness finding) — but it is a
  latent divergence between the two builders. Proposal: symmetric NaN
  guard in `build_feature_ids` (S0-build; no behavior change on current
  data, covered by a new dc_checks/s0 test).

### 2. Canonical field set — decision

**Canonical = the dc01 5-field set** for every feature-aware row
(`lightfm_tags`, `lightfm_tags_ids`, dc01, dc02, dc04's anchors):

`department_name, product_type_name, section_name, colour_group_name,
graphical_appearance_name`

Empirical basis (V1, measured today):

| Set | Vocab V | Distinct signatures | Items sharing a signature | Max class |
|---|---:|---:|---:|---:|
| 5-field (dc01) | 426 | 6,517 | 9,211 (**67.5%**) | 97 |
| 9-field (dc02) | 483 | 6,786 | 8,871 (65.0%) | 97 |

The 4 extra fields (`product_group`, `perceived_colour_master`,
`index_group`, `garment_group`) add **+269 signatures (+4%) and −2.5 pp
collisions** — near-zero discrimination, exactly as the 4.0 redundancy
analysis predicts (FDs ≈ 1.0: product_type→product_group,
department→garment_group, section→index_group; colour↔colour_master NMI
0.81). What they *would* add is double-counting (dc01-style sums count the
same fact twice) and redundant coordinates in dc02's attribute space
(prototype profiles asserting "department X" and its determined
garment-group Y as if independent). The 5-field set covers all four
quasi-orthogonal groups of the 4.0 analysis — product-kind (department +
product_type), context (section), colour (colour_group), pattern
(graphical_appearance) — and keeps the R7-legible colour/pattern fields.

**Per-candidate deviation clause:** deviation from the canonical set
requires a written justification in the candidate's dossier + a declared
comparability caveat. **dc02's current 9-field config is now such a
deviation** — its SC.1a inherits the numbers above and the proposal to
amend to the canonical 5 (its V drops 483→426; its profiles lose the
redundant coordinates). dc01's config already matches. dc03 is image-based
(attributes only in diagnostics); dc04's anchor vocabulary = the canonical
set.

### 3. `price_band` and `product_code` — written dispositions

- **`price_band` — OUT of the canonical set for the scrutiny phase.**
  It is transaction-derived: under the S0.3 cold variant it must be
  computed from reduced-train only, and cold items have *no price at all*
  — an imputation policy for a mid-strength signal (lift 2.36). The
  canonical set stays **static-catalog-only, leakage-clean by
  construction**; the S0.3-handed cold-policy question thereby dissolves.
  `data/hm/price_band.py` remains available as a post-scrutiny,
  per-candidate optional extension (dc01 §3.5's F=6 stays deferred).
  **User nuance recorded at gate:** for *model comparison* price is
  irrelevant (all rows see the same features either way), but for
  *absolute* recommendation performance it may well matter — revisit as a
  performance upgrade once the candidate comparison is settled (protocol
  vault 2026-07-11_1643).
- **`product_code` — stays excluded from model features.** 7,174 values
  over 13,651 items (~1.9 items/value) make it a per-item ID in disguise:
  admitting it would reintroduce identity capacity through the feature
  door and hollow out every "feature-grounded" claim (R4) while trivially
  inflating warm accuracy. Its legitimate roles — variant-aware
  explanation/eval machinery ("another colour of what you bought") —
  remain a non-blocking splitter follow-up (4.0 doc, open list).

### 4. Collision recompute for dc02's SC.1a (the inherited numbers)

Under the canonical 5-field set on V1:

- **Full catalog:** 6,517 signature classes; **67.5%** of items share their
  signature with ≥1 other item (max class 97). This is the ceiling for any
  no-ID attribute-only representation (dc02, `lightfm_tags`): signature
  twins are architecturally indistinguishable. → feeds the full-catalog
  tie-block diagnostic (S0.3).
- **Cold-vs-cold sampled eval (new, decisive):** within the S0.3 cold set
  (2,730 items, deterministic regeneration verified), 43.0% of cold items
  share a signature with some other cold item — but the mean probability
  that a *sampled* negative shares the positive's signature is **0.0006 ≈
  0.06 expected same-signature negatives per 99-negative row**: ~94% of
  cold rows contain no tie competitor at all. **The "68% tie-cap" is a
  full-catalog phenomenon, largely invisible in the 1+99 protocol** —
  dc02's cold numbers will not be tie-throttled, and the honest place to
  show the ceiling is the diagnostic, not the headline metric. (The stale
  hm_3_month/9-field "68%" figure is superseded by these numbers.)

### Gate

**Status: CLOSED — ratified by user 2026-07-11, all five items.**
(a) canonical 5-field set + deviation clause (dc02 → SC.1a); (b) price_band
out, with the user's comparison-vs-absolute-performance caveat recorded
above; (c) product_code excluded — reasoning flagged by user as a future
thesis section; (d) F-S0-06 NaN-guard accepted for S0-build; (e) collision
numbers adopted as dc02's current inheritance (old 68% figure superseded).
Decisions logged to protocol vault
(2026-07-11_1643_canonical-feature-set-price-productcode).
→ Next step: S0.6.

### S0.5 addendum — ml-1m field set (2026-07-12, re-run mode `step0.5 s0`)

**Scope:** the S0.5 data-facts pass for the second quantitative testbed
ratified by the 2026-07-12 charter amendments (C2/C5/C6): entry facts for
the ml-1m canonical field set (**genres + Tag-Genome tags @ relevance ≥ 0.8,
indicator bags** — B5's published MovieLens recipe, C5 as amended), the
missingness policy, the signature-collision numbers (fI twin counterpart),
train-basket stats (fU side), and the **token-mass gate decision**. All
numbers regenerate from `Master/temp/dc_checks/s0/ml1m_feature_audit.py`
(committed with this addendum — the S0-build learnings rule; scrutiny
working basis only, no thesis quotes per the F-DC01-06 restriction).
Companion reasoning: `Master/temp/ml1m_expansion_scope.md`.

**1. Data staging + join integrity.** `data/ml-1m/movies.dat` + the 2014
Tag-Genome files staged locally (gitignored). Join facts: all 3,125 split
movieIds match `movies.dat` (0 missing — the scope doc's "4 items had no
match" display-build observation is superseded); genome list covers
**3,093/3,125 = 99.0%**; 62 covered movies carry zero tags @0.8 →
**97.0% full rows / 3.0% (94 items) genres-only rows**; genres-only items
receive only **0.54%** of train interactions. Vocab: **18 genres + 1,043
tags = V 1,061** (B5 reported 1,030 tags on ml-10M — same order, recipe
validates). Bag sizes: genres 1–6/item (0 empty); total tokens/item mean
12.3, median 10, p10 3, p90 24, min 1, max 72.

**2. Entry correctness — current state, deferred audit.** No model-grade
ml-1m feature entry exists yet: `data/ml-1m/build_item_features.py` is the
display-grade builder (title/year/genres for explanation CSVs only), and
`build_feature_ids` assumes fixed one-value-per-field width — variable-size
bags cannot enter today. The entry-correctness audit of the **bag layout**
(embedding-bag variant + model-grade `item_features.csv` builder + fU
multi-value builder support) therefore transfers to the S0-build extension,
with the keystones already named (bag-vs-padded bit-identity on H&M inputs;
F=0 reductions). Missingness discipline note: genres-only rows are *short
bags*, not NaN cells — the F-S0-06 guards keep firing on true NaN; no new
machinery needed (and min tokens = 1, so no empty bag exists on this data).

**3. Missingness policy (the disclosure sentence, for ratification).**
*The 3.0% of split movies (94 items, 0.54% of train interactions) not
covered by Tag-Genome @0.8 enter with genres-only indicator bags; declared
here once, no imputation machinery.*

**4. Signature collisions — fI twin counterpart.** Under genres+tags@0.8
bags: **3,027 distinct signatures / 3,125 items; 118 items (3.8%) share a
signature; max class 32**. Of the 118, **78 sit inside the 94 genres-only
items** (the max-32 class is a genres-only bag) — i.e. among full rows the
twin population is ~40 items (~1.3%). Contrast H&M's 67.5%: the ml-1m twin
ceiling is negligible outside the genres-only tail, and the tail is exactly
where the missingness disclosure already points. dc01's ml-1m extension
inherits THESE numbers.

**5. Train baskets — fU side.** |H_u|: 6,034 users, mean 93.2, median 56,
p10 15, p90 223, min 3, max 1,415 — the heavy-history × coarse-vocabulary
regime the charter amendment wanted (H&M V1: mean 6.49/median 5). D4 band
edges for ml-1m stay TBD at SC.6 per the roadmap (≥20-rating floor
population; p10 = 15 says a genuinely thin stratum barely exists).

**6. GATE DECISION — token mass in the bags.** Two candidate conventions
for how a movie's variable-size bag enters the sums (fU basket sum
`q_u = Σ w·e_token`; fI/lightfm item sum `q_i = Σ e_token`):

- **(A) Raw indicator bags — B5-faithful default.** Every token weighs 1
  (fU: 1/|H_u| per token, the H&M rule verbatim); a movie's vote mass = its
  token count. This is LightFM's native input form (C5 amendment, B5 §2.2)
  — the published recipe the user directive anchored to.
- **(B) Per-movie normalization.** Tokens weigh 1/|bag|; every movie votes
  mass 1. Deviates from the published recipe → needs written justification
  + comparability caveat under C5's deviation clause.

Measured effect size of (A) on this data: basket ESS/|H| mean **0.773**
(raw weighting shrinks the effective basket to ~77% of its nominal size;
p10 0.710), single-heaviest-movie mass share mean **6.5%** (median 4.4%) —
moderate concentration, no single-movie domination. The load-bearing new
fact: **token count correlates with train popularity at Spearman ρ =
0.672** — under (A), popular movies are systematically up-weighted inside
fU's basket sum, an *implicit popularity weighting channel* of the same
family as F-DC05-02's interaction-weighted cloud (and it compounds with
it: heavy users AND popular movies both gain mass). Under the
empirical-not-scholastic principle this is a hypothesis with an
instrument, not a veto.

**Recommendation: adopt (A)**, on B5-faithfulness (the anchor the user
directive set; every deviation weakens the "features the way LightFM did"
citation) and on the moderate measured concentration — and record the
popularity-coupling channel as a named hidden-effect with its instrument:
the two-weightings readout family (F-DC05-02) gains a token-mass axis, and
the SC.8 working-basis set gains the per-user ESS + mass-share readouts
(regenerable from this script's logic; no new committed script — F-DC05-04
disposition respected). (B) remains the evidence-triggered fallback if
SC.8 instruments show the channel dominating.

**Findings:** none new — no code was touched; the one latent-divergence
class this pass surfaced (bag layout absent) was already scoped as the
S0-build extension, and missingness needs no machinery (item 2).

### Gate (addendum)

**Status: CLOSED — ratified by user 2026-07-12.** (a)–(c) accepted as
presented; (d) decided **(A) raw indicator bags — "like LightFM did it"**
(user decision). **Standing mention obligation (user directive at this
gate):** the popularity-coupling fact (token count vs train popularity
Spearman ρ = 0.672 → raw bags implicitly up-weight popular movies in fU's
basket sum, compounding with F-DC05-02's interaction weighting) is kept on
the record and MUST be mentioned wherever ml-1m fU results are presented —
routed to the dc05 hidden-effects section + the SC.8 instrument set
(ESS/mass-share readouts, token-mass axis on the two-weightings readout);
logged to the protocol vault. → Next: S0.3 addendum (`step0.3 s0`).

---

## S0.6 Charter (2026-07-11)

**The fixed contract.** Every candidate's SC.6 run plan cites compliance
with these clauses one by one; SC.8 audits the actual runs against them.
Changing a clause requires a gated charter amendment (dated, with a
retro-invalidation note on affected runs). All clauses were individually
ratified at the S0.1–S0.5 gates of 2026-07-11; this section only fixes
them as one citable contract.

**C1 — Single code lineage.** All scrutiny work, fixes, and runs live on
branch `feat/scrutiny` (Ground rule 4). Every run manifest records the
commit hash it ran from. Merge to `dev` only at protocol completion.

**C2 — Datasets.** `hm_1_month` (canonical, frozen artifact — provenance
verified against LEO5 copies) is the primary quantitative testbed, plus its
derived `hm_1_month_cold` variant (S0.3 §2, deterministic, seeded).
Any other dataset (ml-1m, amazon2014, hm_3_month) appears only with a
written reason (replication context, qualitative showcase).
*(Charter amendment 2026-07-12, user decision — vault `2026-07-12_2114`:
**ml-1m joins as a second quantitative testbed** for the feature-aware
lineage — the complementary regime (heavy histories × coarse vocabulary vs
H&M's thin × rich); per-candidate dataset lists stay pinned at SC.5/SC.6.
**amazon2014 is discarded from the feature-aware scope** — honestly
justified: the dataset as used by ProtoMF is ratings-only, no item features
exist, so a feature-aware comparison has no object; it remains replication
context. hm_3_month unchanged (written-reason clause).)*

**C3 — Eval protocol.** Temporal leave-one-out; 1 positive + 99 uniform
negatives (`NEG_VAL=99`); HR/NDCG@{1,3,5,10,50}; model selection on val
`hit_ratio@10`; `use_bias=0` fleet-wide (bias-on only as the
evidence-triggered ablation, vault 2026-07-11_1556); Evaluator
divisor-by-counted-rows + expected-rows assert (F-S0-03). Cold eval per
the S0.3 spec: cold-vs-cold primary, cold-vs-all secondary, single-config
retrain on the variant, training-negative exclusion of C, warm-val early
stopping. Secondary readouts: popularity-sampled negatives test pass (D3,
computed once at S0.7) and the activity/popularity-stratified readout (D4).

**C4 — Budget & seeds.** Dev profile: **30 trials / 60 epochs / patience 7 /
ASHA grace 4**, deliberately NOT the paper's 100/100. Single seed
**38210573**. **No multi-seeding, no split repetitions** — deferred until
final or near-final candidate versions stand (S0.3 gate directive). One
run = one job (no fragmenting); compute envelope: >1 day soft flag,
>10 days infeasible.

**C5 — Features.** Canonical field set = the S0.5 five
(`department_name, product_type_name, section_name, colour_group_name,
graphical_appearance_name`); static-catalog-only (no transaction-derived
features this phase — price_band deferred with the recorded
performance-upgrade caveat); `product_code` never a model feature.
Per-candidate deviation only with written justification + declared
comparability caveat (dc02's 9-field config resolves at its SC.1a).
*(Charter amendment 2026-07-12 — the canonical field set becomes
**per-dataset**: H&M keeps the S0.5 five unchanged; **ml-1m canonical set =
B5's own published MovieLens recipe (user directive "features the way
LightFM did", superseding the same-evening genres+year interim proposal in
vault `2026-07-12_2114`): `genres` + Tag-Genome tags at relevance ≥ 0.8
(B5 §4.1's published threshold — theirs, not ours), both as indicator
bags** (variable-size bags are LightFM's native input, B5 §2.2; implemented
via the variable-length index+weight bag layout). `year` excluded (LightFM
used none); `title` excluded as identity-like (the `product_code`
analogue). Disclosed nuance: B5 used MovieLens-10M; we port the feature
recipe onto ProtoMF's own ml-1m artifact (movieIds stable across releases);
genome coverage verified at the S0.5 addendum — uncovered movies get
genres-only rows (declared missingness policy, F-S0-06 discipline).
Preconditions: `movies.dat` + Tag-Genome download; model-grade
`item_features.csv` build; bag-layout builder support (S0-build
extension). Full magnitude scoping: `Master/temp/ml1m_expansion_scope.md`.)*

**C6 — Baseline set (the frozen reference fleet).** ProtoMF five (`mf`,
`acf`, `user_proto`, `item_proto`, `user_item_proto`) + the B5 two
(`lightfm_tags` feature-only, `lightfm_tags_ids` strong feature-aware) +
the popularity reference row; CF rows additionally carry the attr-kNN
cold-fallback columns (S0.3 §5). Computed **once** at S0.7 on the
integration branch with the cold eval active; no SC.6 re-runs baselines
without a charter amendment. Exclusions documented in S0.4 rev. 1
(tags+about inapplicable; LSI-LR/LSI-UP cited, not re-run).
*(Charter amendment 2026-07-12 — **ml-1m reference-fleet rows, added
lazily**: the fleet definition extends to ml-1m (at minimum the stage-bar
hosts `user_proto`/`item_proto`, `mf`, popularity; lightfm rows once the
bag-layout support exists), but rows are submitted only when a run plan
first cites them — each its own /leo5-submit, dev profile, cold variant per
the S0.3 machinery. The H&M fleet rows and the compute-once rule are
untouched.)*

**C7 — Ablation discipline.** Per candidate: its design doc's committed
isolating ablations only (e.g. dc01 `use_id_feature` fork + F=0 keystone;
dc02 knobs-off base) — no exploratory sweeps inside the scrutiny phase.

**C8 — Run manifests.** Every submitted run records: model/config, feature
fields, `use_id_feature`/knob states, negatives, seed, profile, dataset
(+ variant), branch+commit, `use_bias`. SC.8's compliance audit walks this
list clause by clause.

### Preconditions & pause-filler work (named, non-blocking)

- **dc03 go/no-go precondition — H&M image re-download (~25 GB).**
  Target: **LEO5 scratch** (`/scratch/c7031336/`) — measured today: scratch
  has hundreds of TB free; **home is capped at 5 GB and must not be the
  target**; the laptop is excluded (no space, images only needed where
  pre-extraction runs). Steps: kaggle CLI + API token on LEO5 → download
  the competition images → verify subdirectory range `000`–`0NN` and
  ~105,100 images against `data/hm/raw/README.md` (the local set has only
  86 subdirs — known-partial, CLAUDE.md caveat). To be kicked off in the
  background during the S0/dc01 window; **must not block dc01/dc02**;
  dc03's SC.3 STOPS if unmet.
- **dc04 literature-support search** (model-knowledge provenance for its
  ingredient relatives) — designated pause-filler for cluster-run waits.

### Gate

**Status: CLOSED — ratified by user 2026-07-11.** No clause disputed (all
were individually ratified at earlier gates this session). The dc03
precondition's download was kicked off at this gate: kaggle CLI + token
deployed to LEO5, 28.7 GB competition zip downloading to
`/scratch/c7031336/hm_raw_kaggle/` (detached, ~15 min ETA); unzip +
subdirectory/count verification due next session; token to be expired
after verification. → Next: S0-build (fresh session).

---

## S0-build (2026-07-11)

> **Execution mode note:** built in a single autonomous session under the
> user's explicit away-authorization ("get as much done as you can, I'll
> check in"). All four components were implemented, tested, and committed;
> the protocol's per-component gates could not be held live and are
> presented below as **one batched gate review** — nothing has been merged
> or pushed; every component is a separate revertable commit on
> `feat/scrutiny`.

### Component list (one commit each)

| # | Component | Spec | Commit | Tests (dc_checks/s0) |
|---|---|---|---|---|
| A | Evaluator counted-rows divisor + expected-rows assert | F-S0-03 / S0.3 §3 | `543e2d5` | t01 (11 checks) |
| B | NaN guard in `build_feature_ids`, symmetric with dc02 builder | F-S0-06 | `a46d4e8` | t02 (7) |
| C | `lightfm` CBF baseline (`lightfm_tags`, `lightfm_tags_ids`) | S0.4 | `9779597` | t03 (19) + i01 (toy end-to-end train/test, both configs) |
| D | Cold-start eval machinery (generator, dataset semantics, runner, D4 readout, registration) | S0.3 | `7f7f24a` | t04 (16) + t05 (16) + i02 (toy end-to-end: variant train → full cold_eval run, 17) |

All suites green; regressions green: dc01 t01/t04/t05/t06/i02(keystone)/i04,
dc02 t04/t05/t06/i02/i04/i05/i06. Ledger updated: F-S0-03 →
fixed(543e2d5), F-S0-06 → fixed(a46d4e8). Modifications log updated per
component (Ground rule 11).

### What landed (by component)

**A — Evaluator (F-S0-03).** `Evaluator` now divides aggregated metrics by
the **counted** rows and asserts the count against a declared expectation
(constructor arg, now optional; None = caller owns its divisor).
Implemented **inside `get_results`** rather than in `Trainer.val()` /
`Tester.test()` as the ledger proposal had sketched — the assert thereby
covers both call paths plus every future caller. Bit-identity with the
historical divide-by-n_users while rows == n_users is pinned by t01.

**B — NaN guard (F-S0-06).** `build_feature_ids` now raises on NaN cells
exactly like `build_attr_multi_hot`. Inert on V1 (0 NaN cells re-verified;
canonical 5-field vocab V = 426 re-confirmed). t02 also pinned a subtlety:
a literal `'nan'` CSV cell is parsed as NaN by pandas defaults, so BOTH
builders reject it — that is precisely the phantom-vocab-value scenario
the guard exists for.

**C — lightfm baseline (S0.4).** New factory branch `ft_type='lightfm'`:
user = plain CF `Embedding`, item = dc01-tested `FeatureEmbedding` over
the canonical 5-field set, plain dot product, bias-free. Pure reuse — no
new module. `inject_feature_ids` guard extended; configs mirror
`mf_hyper_params`' search space exactly; models registered in start.py /
run_combo.py; `lightfm` added to CLAUDE.md's reserved ft_type list.
**Keystone green:** `use_id_feature=True` + zero fields is bit-identical
to `mf` (identical state_dict keys AND tensors, identical logits). Cold
path pinned: tags-only scores unseen items finitely, signature twins tie
exactly, no per-item rows in the table. Not added to EXPLAINABLE_MODELS
(no prototypes — nothing for the pipeline to explain; consistent with the
S0.4 "no prototype machinery" definition).

**D — Cold-start machinery (S0.3).** Per the §8 touch-point map:
1. `data/hm/make_cold_variant.py` — deterministic generator (seeded
   popularity-stratified 20% draw, ascending-popularity deciles via
   `np.array_split`, `default_rng(38210573)`); never re-splits/re-cores;
   ID universe copied byte-identically; collateral report printed and
   persisted (`cold_variant_report.json`).
2. `ProtoRecDataset` — training-split negative sampling excludes C when
   `cold_items.csv` marks the data dir (leakage item 6); eval splits keep
   the full catalog; **the one-row-per-user invariant moved into the
   dataset** (fatal on canonical splits, relaxed-with-notice on the marked
   variant). New `ColdTestDataset`: cold rows, cold-pool (primary) /
   full-catalog (secondary) negatives, canonical-consumption exclusion
   (leakage item 5), dedicated seeded RNG — identical draws across model
   rows under the runner's num_workers=0.
3. `Trainer.val` / `Tester.test/get_test_logits` — Evaluator expectation =
   the dataset's declared row count (`coo.nnz`): identical on canonical
   splits, correct automatically on the variant (this is what makes
   variant *training* runnable at all — warm val has fewer rows than
   users).
4. `utilities/cold_eval.py` — the runner: warm test + cold-vs-cold
   (chance line HR@K = K/100 declared) + cold-vs-all; per-original-
   popularity-decile slices; per-item **cluster** bootstrap 95% CIs;
   attr-kNN fallback for CF rows (n=20, patches the item ID embedding
   table — one patch covers both tied double-tie branches — then restores
   weights); model-agnostic tie-block diagnostic (full-catalog top-10
   signature stats, ≥5k sampled warm users, seeded); popularity reference
   row; **refuses `use_bias≠0`** without a declared cold-bias policy.
5. `Master/scripts/stratified_readout.py` — D4: HR@10/NDCG@10 by
   user-history quartile and item-popularity bucket (bucket 0 = the
   F-S0-04 train-unseen slice, reported separately) on any results dir.
6. `start.py`/`run_combo.py` — `hm_1_month_cold` registered;
   `--retrain-config <config.json>` implements the §6 single-config
   retrain convention (fixed config, num_samples=1, no search, device
   re-detected). `.gitignore` covers the variant artifacts.
7. Tests as listed above; `data/hm_1_month_cold/` generated locally by the
   committed generator.

### Deviations & decisions for this gate (numbered for review)

1. **Generator draw vs the S0.3 rev. 1 illustrative numbers.** The spec
   session measured its numbers with an ad-hoc, uncommitted computation;
   its exact RNG stream is unreproducible. The committed generator
   reproduces the ratified **2,730 cold items over 13,648 eligible**
   exactly, but draws a different C, so the row-level collateral differs:
   **removed train 92,817 (19.48%)** vs "93,805 (19.7%)"; **cold rows
   121,024** (after dropping **515 rows of 143 zero-train users**) vs
   "122,808 / 167"; **variant val 59,066 / warm test 59,048** vs
   "59,052 / 58,781"; **11,714** users below 3 train rows vs "11,905".
   Same order of magnitude everywhere; the committed, reproducible draw is
   now the canonical one (dated edit note added to S0.3 §2). The S0.5
   cold-tie companion numbers (43.0% within-pool twins, 0.0006 sampled-tie
   probability) were computed on the session's C and shift trivially —
   marked stale-in-detail, direction unaffected; recomputed in-session on
   the committed C (2,730 items, 1,968 signature classes under the
   canonical 5): within-pool twin share **40.8%** (was 43.0%), mean
   same-signature probability per sampled negative **0.00058 ≈ 0.06
   expected tie competitors per 99-negative row** (row-weighted 0.00066)
   — the S0.5 conclusion ("tie-cap largely invisible in the 1+99
   protocol") stands unchanged; dc02's SC.1a inherits THESE numbers.
2. **Invariant placement (F-S0-03).** The dataset (not the trainer)
   enforces one-row-per-user, and the Evaluator expectation is the
   declared row count. Rationale: the invariant is a *split* property; the
   variant legitimately violates it, and only the dataset knows which case
   it is in (the `cold_items.csv` marker).
3. **attr-kNN patch target** = the item branch's base ID-embedding rows
   (mean of top-20 attribute-cosine warm neighbors' rows). For prototype
   models the item representation is a deterministic function of the base
   embedding, so patching the base is the architecture-consistent reading
   of dc02 §3.6 #7; for the double-tie the tied Parameter means one patch
   covers both halves.
4. **`--retrain-config` untested locally** — experiment_helper targets the
   Ray 2.x API (LEO5), the laptop has Ray 1.6 (known, pre-existing);
   its config-whitelist logic is deterministic and the path will be
   exercised in the S0.7 smoke on LEO5 before any reference run depends
   on it.
5. **CLAUDE.md image caveat corrected** (trivial doc fix, logged): see
   precondition note below.

### S0.6 precondition progress (dc03 images — non-blocking)

- LEO5 zip download **complete and verified**: 30,810,293,747 bytes;
  central directory holds **105,100 jpgs in 86 subdirectories, 010–095
  contiguous** — the authoritative inventory.
- **Finding: the local `data/hm/raw/images/` set is COMPLETE** — exactly
  105,100 jpgs in 010–095. The "likely partial" caveat (CLAUDE.md,
  2026-06-07) rested on an assumed 000–0NN range; article ids simply never
  start below 010. CLAUDE.md corrected with the evidence.
- Extraction to `/scratch/c7031336/hm_raw_kaggle/images/` **complete and
  verified in-session: 105,100 jpgs on disk, subdirs 010–095** — the dc03
  go/no-go precondition is MET. The S0.3 cold variant was also scp'd to
  `leo5:/scratch/c7031336/protomf_data/hm_1_month_cold/` (S0.7 prep).
- **Kaggle token expiry: user action at this gate** (verification done).

### Gate

**Status: CLOSED — ratified by user 2026-07-11 (batched review after
walkthrough).** Review mode: user requested a plain-language component
walkthrough plus in-depth explanations of the three technical calls
before deciding. Outcomes: components A–D approved as landed (D with the
standing note that S0.7 opens with the independent spec-vs-code audit of
exactly this machinery); **deviation 1 ratified — the committed
generator's draw is canonical** (S0.3 §2 edit note stands; dc02 inherits
the recomputed tie numbers); technical calls (2)–(4) accepted; CLAUDE.md
image-caveat correction (5) accepted implicitly with the precondition
review. Kaggle token deleted from LEO5 in-session (verified gone).
→ Next step: S0.7 (foundation verification + reference runs on LEO5).

### S0-build extension (2026-07-12, late evening — ml-1m support, per the S0.3/S0.5 addenda)

**Scope:** the five components the 2026-07-12 addenda scoped for the second testbed
(charter C2/C5/C6 amendments): bag layout, fU multi-value support, per-dataset field
plumbing, model-grade ml-1m `item_features.csv`, generalized cold generator +
`ml-1m_cold`. Supervised build, one live user gate per component (all five ratified
in-session 2026-07-12), plus this closing section as the combined-artifact gate.

**Component list (one commit each; tests in `Master/temp/dc_checks/s0/`):**

| # | Component | Commit | Tests (green) |
|---|---|---|---|
| a | Bag layout: `build_feature_bags` + `FeatureEmbedding` optional `feature_weights` (weighted-bag mode) | `3295c47` | t07 (25) |
| b | fU builder multi-value: `build_user_history_weights(layout='bags')` | `d660f66` | t08 (15) |
| c | Per-dataset `feature_fields`: `CANONICAL_FEATURE_FIELDS` registry + `'canonical'` sentinel resolved in `start_hyper` (C8 gets literals) + injection layout dispatch + factory pass-through | `e8f519c` | t09 (23); dc01 t05/t09 + dc05 t04 pins updated (dated notes) |
| d | Model-grade ml-1m `item_features.csv`: builder extended with Tag-Genome tags @0.8 (names, pipe-joined, sorted) | `74906f6` | t10 (13) |
| e | Generalized cold generator (docstring generality, loud `item_features` precondition, A2.3 min-pool report) + `ml-1m_cold` generated + registration | `41fb655` | t11 (17) |

**Keystones (all green):** (a) bag-vs-fixed **bit-identity on real V1** — builder tensors equal
(V=426, no padding) and `FeatureEmbedding` outputs bitwise equal (maxΔ=0.0); F=0 bag mode ≡
plain `Embedding` bitwise. (b) `layout='bags'` **bit-identical to the shipped fixed path on
real V1** (73,418×112 tensors, maxΔ=0.0). (c) factory-level lightfm fixed-vs-bags bitwise
equality. Every existing H&M path is therefore provably untouched; regressions green across
s0 t01–t06 + i01/i02, dc01 t01/t04/t05/t06/t09/i02(keystone)/i04/t11, dc05 t01–t05(keystone)/
t06/t07/t08.

**Design decisions taken in-build (each ratified at its component gate):**
1. ONE class, two layouts — `FeatureEmbedding` gained an optional `feature_weights` buffer
   instead of a sibling class, so the dc01/lightfm factory branches and `cold_eval`'s ID-drop
   primitive work on bag models with zero changes; `None` preserves the original path
   byte-identically. Padding uses vocab row 0 at weight 0.0 (`HistoryFeatureEmbedding`
   precedent).
2. Missingness mechanics: '' cell = legal short bag (the ratified 3% genres-only tail);
   literal `'nan'` token, duplicate in-cell tokens, and all-fields-empty items refuse loudly
   (F-S0-06 discipline; empty composition degenerate under cosine).
3. Sentinel-only resolution: `'canonical'` resolves per dataset family (hm_* → S0.5 five,
   fixed; ml-1m* → genres+tags@0.8, bags; else loud refusal); literal lists (F=0 `[]`,
   `--retrain-config` saved configs) pass through untouched; absent `feature_layout`
   defaults 'fixed' (legacy-config compat).
4. `ml-1m_cold` facts (the committed generator's draw is canonical, per the S0-build
   deviation-1 precedent): **C = 625/3,125 eligible; removed train 110,558 (19.66%); cold
   rows 112,979; ZERO zero-train collateral; 4 users below 3 train rows; min cold-vs-cold
   pool 341** (addendum arithmetic predicted ≈342), no user below 100. `hm_1_month_cold`
   regenerated post-edit and verified **byte-identical** (draw untouched; its min pool:
   2,708). LEO5 scp deferred to the first run plan citing ml-1m rows (A2.4).

**Notable in-build catches (no ledger findings — all caught pre-landing inside their
component):** an empty-width tensor slice (`w[..., :1]` at D_max=0) would have silently
zeroed the ID row via 0-size broadcasting — caught by the F=0 keystone, fixed, pinned
(learnings entry added); and the t10 vote-mass check surfaced the **popularity-coupling
channel live**: basket-weighted mean tokens/movie = 24.3 vs catalog 12.3 (~2× — the ρ=0.672
channel's first magnitude measurement on real model input; protocol vault
`2026-07-12_2313`, joins the dc05 hidden-effects record; scrutiny working basis only).

**Entry-correctness audit transfer (S0.5 addendum item 2 closed):** the deferred
entry-correctness audit of the bag layout is discharged by t07/t08/t09/t10 — offsets/vocab
conventions shared with the audited builders, item_id coverage + NaN policy guards active,
serialization boundary preserved (weights are non-persistent buffers, never in config.json;
t09 pins the payload keys).

### Gate (extension, combined artifact)

**Status: CLOSED — ratified by user 2026-07-12** (all five component gates live in sequence —
(d) with the protocol-entry directive, executed — then the combined-artifact review).
Roadmap step 3 (NEXT_STEPS_roadmap_2026-07-12.md Block 1) is complete → next: dc05 SC.3
(`/scrutiny dc05`, fresh session), which consumes these builders; S0.7 part 2 (H&M fleet →
frozen table) remains the independent cluster-side thread; ml-1m fleet rows stay lazily
submitted (C6); ml-1m_cold scp to LEO5 deferred to the first run plan citing ml-1m rows.

---

## S0.7 Verification + reference runs (2026-07-11 — part 1: verification; gate OPEN)

**Scope:** scrutiny of S0-build itself (the instrument), then the frozen
reference fleet. Part 1 (this section as first written): black-box
spec-vs-code audits, first-party leakage walk, smoke runs, findings, and the
reference-run plan as a gate proposal. Part 2 (the frozen table) is appended
after the gate and the LEO5 runs. HEAD at audit time: `93b1edd`.

### 1. Smoke runs

All seven `dc_checks/s0` suites re-run green on current HEAD this session:
t01 (Evaluator divisor, 11), t02 (NaN guard, 7), t03 (lightfm, 19), t04
(generator, 16), t05 (cold dataset semantics, 16), i01 (lightfm toy
end-to-end, both configs), i02 (cold-eval end-to-end incl. kNN patch +
`use_bias` refusal, 17). The `--retrain-config` LEO5 smoke (S0-build
deviation 4) is scheduled as the first post-gate submission.

### 2. Black-box audit A — cold machinery vs S0.3 spec

Independent subagent, inputs per Ground rule 10 (spec extract + code only —
deliberately blind to the S0-build narrative, so ratified deviations get
independently re-derived or re-flagged). **Verbatim report:**
`s0_7_audit_cold_machinery.md`. Headline: **all seven leakage-checklist
items verified correct in the actual code paths — no leakage divergence
found**; the auditor additionally proved the training-negative mask cannot
fail silently (a missing `cold_items.csv` trips the one-row-per-user assert
before training starts). Point-by-point response:

| Audit item | Response | Route |
|---|---|---|
| #1 dc01 ID-column drop missing (major) | **Concede.** Dormant (no dc01 cold run exists), but the runner must refuse rather than silently mis-score. | **F-S0-07** (guard now; drop itself at dc01 SC.3) |
| #2 popularity row degenerate on cold-vs-cold (major for that row) | **Concede** — found independently first-party the same session (all-tie probe: HR@10 reads 0.00, not chance 0.10). | **F-S0-08** (rank-based scorer + seeded tie-break) |
| #3 attr-kNN patches base embeddings, not representations (minor) | **Rebut: ratified.** This is S0-build deviation 3, gated 2026-07-11 — the base-patch is the architecture-consistent reading. Spec now carries an edit note so future audits don't re-flag. | trivial doc fix (applied) |
| #4 no canonical-warm juxtaposition (minor) | **Concede.** | **F-S0-11** |
| #5 retrain profile footgun (minor/operational) | **Concede.** | **F-S0-10** |
| #6 D4 mechanism named wrongly in spec (cosmetic) | **Concede** — spec edit note applied. | trivial doc fix (applied) |
| #7 chance line only @10 (cosmetic) | **Concede** — folded into F-S0-08's fix. | F-S0-08 |
| B2 sigmoid saturation ties | **Concede as observation**; pre-existing host-wide behavior. | **F-S0-12** (wontfix proposed) |
| B3 warm negatives can contain removed cold purchases | **Concede** — spec-silent but directionally biased against cold-generalizing models. | **F-S0-09** |
| B4 cross-row determinism is convention-dependent | Noted; pinned by t05, runner complies. | learnings |
| §4.3 stale rev-0 collateral numbers in spec | **Concede** — edit note applied. | trivial doc fix (applied) |
| §5 "noise floor" nuance: cold rows are weight-decay-shrunk after retraining, not literal random-init | Accepted wording sharpening; the convention is unchanged ("as the architecture actually behaves"). | dossier note only |

### 3. Black-box audit B — lightfm baseline vs S0.4 spec

Independent subagent, same isolation. **Verbatim report:**
`s0_7_audit_cbf_baseline.md`. Verdict: **FULL CONFORMANCE** — every §2/§3
commitment implemented as specified (search-space mirror verified by
programmatic deep-diff: the only differences vs `mf_hyper_params` are the
spec-mandated ones; the two configs differ in exactly `use_id_feature`).
Three latent observations (no current effect): `use_bias` defaults True when
absent (`rec_sys.py:36`), `use_id_feature` defaults True when absent
(factory), shared sampler instances across configs (harmless). Routed to the
learnings ledger as config-default watch-outs; C8 manifests already record
`use_bias` per run.

### 4. First-party leakage walk (independent of audit A, convergent)

All seven S0.3 §4 items walked against actual code paths this session before
the audit returned: (1) static attributes, byte-identical copies (t04);
(2) canonical field set transaction-free (C5); (3) no re-coring — the
generator only filters rows of the frozen artifact; (4) no temporal fields;
(5) `ColdTestDataset` excludes canonical train+val+test consumption;
(6) `ProtoRecDataset` zeroes cold items from the training-negative
distribution (train split only, marker-gated); (7) variant val contains no
cold positives, early stopping on warm `hit_ratio@10`, Evaluator asserts
declared row counts. **7/7 verified — two independent walks agree.**

### 5. Findings (proposals in `findings_ledger.md`, decision at this gate)

| ID | Severity | One-liner | Recommended disposition |
|---|---|---|---|
| F-S0-07 | major (dormant) | dc01 ID-column drop at cold inference unimplemented | guard in runner now; drop at dc01 SC.3 |
| F-S0-08 | minor | popularity reference reads 0.00 on cold-vs-cold (argpartition tie artifact) | fix: rank-based scorer + seeded tie-break |
| F-S0-09 | minor | variant warm-eval negatives can include the user's removed cold purchases | fix: fold cold_test rows into variant val/test exclusion |
| F-S0-10 | minor | retrain path silently defaults to production profile | fix: `--retrain-config` implies dev profile unless overridden |
| F-S0-11 | minor | runner omits the mandated canonical-warm comparison | fix: optional `--canonical-results-dir` delta row |
| F-S0-12 | minor | fp32 sigmoid saturation collapses order among extreme logits (host-wide, pre-existing) | document-only wontfix |

Trivial fixes applied this session (Ground rule 3): three dated edit notes in
the S0.3 spec section (§4.3 stale collateral numbers; §5 attr-kNN ratified
patch target; §7 D4 mechanism naming).

### 6. Reference-run plan (part 2 — proposal for this gate)

Fleet per charter C6, all on `feat/scrutiny` with C8 manifests:

1. **Canonical `hm_1_month` hyperopts (dev profile, seed 38210573):**
   the ProtoMF five (`mf`, `acf`, `user_proto`, `item_proto`,
   `user_item_proto`) + the B5 two (`lightfm_tags`, `lightfm_tags_ids`).
   The lightfm two must run fresh (never trained). **Open question Q1** for
   the five: complete 30-trial runs from 2026-06-08 exist
   (`Master/experiments/results/*_hm_1_month_s38210573`), but their metadata
   records **no commit hash** — adopting them violates the letter of C1/C8.
   **Recommendation: re-run all five on the integration branch** (clean
   provenance; June numbers kept as a cross-check — large disagreement would
   itself be a finding). Cost ~3–5 h/model, parallel on LEO5 → ~1 day wall.
2. **Cold-variant retrains:** best canonical config per model,
   `--retrain-config`, dev profile, seed 38210573 → 7 single-config runs
   (≤1 h each). The first doubles as the LEO5 `--retrain-config` smoke.
3. **Cold evals:** `utilities/cold_eval.py` per model row (cold-vs-cold
   primary + cold-vs-all + kNN fallback for CF rows + tie diagnostic +
   popularity reference), post-fix versions of F-S0-08/09/11.
4. **D3 secondary readout:** popularity-sampled negatives test pass on the
   frozen canonical checkpoints (minutes per model).
5. **D4 stratified readouts** on the canonical runs (local, free).

Ordering: gated fixes land first (they touch the instrument), then
submissions. Envelope: comfortably inside C4 (>1 day soft flag not tripped
by any single run).

### Gate (part 1)

**Status: CLOSED — ratified by user 2026-07-11.** F-S0-07/08/09/11/12
approved as proposed; **F-S0-10 amended by the user to wontfix** — profile
selection is a queue-time decision for every run type (all scrutiny
hyperopts equally take `--profile dev` at submission; nothing is hardwired
per-model), so a retrain-only default would be the fleet's sole special
case; the run plan carries explicit `--profile dev` on every submission
line instead. Q1: re-run approved with all other proposals.

**Gate-execution addendum (same day): F-S0-13.** While wiring the F-S0-07
guard it emerged that S0.4 §2 commits `lightfm_tags_ids` to the same
ID-column-drop convention at cold inference — and that row is in the
reference fleet NOW, so the drop could not wait for dc01's SC.3. Both
black-box audits missed it (it sat in the seam between their scopes:
audit A had only the S0.3 spec, audit B did not receive `cold_eval.py`).
Fixed within the ratified scope: generic `drop_cold_id_rows` in
`cold_eval.py` (zeros cold items' ID rows in a `FeatureEmbedding` branch;
applied automatically by the runner; dc01's nested case stays refused
until SC.3). Ledger: F-S0-13. Learnings: audit-scope-seam watch-out.

**Fixes landed (all suites green post-fix, t01–t06 + i01–i02):**
F-S0-08 rank-based `PopularityScorer` (+ all-K chance lines), F-S0-09
variant eval-negative exclusion (`protomf_dataset.py`, modifications-log
cross-ref), F-S0-11 `--canonical-results-dir` delta row, F-S0-07 guard,
F-S0-13 drop — pinned by the new `dc_checks/s0/t06` (16 checks); i02's
chance-line check updated to the new all-K schema. → Part 2: LEO5
reference runs per §6 (re-run the five + lightfm two, explicit
`--profile dev` everywhere).

### Run-submission record (part 2, hyperopt wave — 2026-07-11)

Charter C8 manifest, one row per submitted job. Common to all seven:
dataset `hm_1_month` (canonical, LEO5 scratch copy), seed 38210573, profile
`dev` (30/60/7/4, explicit `--profile dev` per the F-S0-10 ruling), ASHA on,
uniform-99 eval negatives, `use_bias=0` (config-inherited), canonical
5-field set (feature rows), num_workers=1 / concurrency 5 / 10 CPUs / 40G /
A30 (fleet mode — pinned per the leo5-submit fleet rules established this
session), W&B tags `s07-ref` + auto `dev`, branch `feat/scrutiny`, cluster
HEAD `385ab0e` (verified in the same shell as each submission). Each job
was submitted via its own `/leo5-submit` invocation (user hard gate).

| # | Model | Job ID | Time limit | Time basis |
|---|---|---|---|---|
| 1 | mf | 6988922 | 8:00:00 | exact benchmark 3h05m × 2 |
| 2 | acf | 6988960 | 8:00:00 | exact benchmark 2h56m × 2 |
| 3 | user_proto | 6988961 | 8:00:00 | exact benchmark 3h35m × 2 |
| 4 | item_proto | 6988962 | 12:00:00 | A100 bench 2h54m × 1.67 (A30) × 2 |
| 5 | user_item_proto | 6988963 | 1-00:00:00 | A100 bench 3h45m × 1.67 × 2 |
| 6 | lightfm_tags | 6988964 | 8:00:00 | mf bench × 1.1 (5-field gather) × 2 |
| 7 | lightfm_tags_ids | 6988966 | 8:00:00 | mf bench × 1.1 × 2 |

Still to run after this wave completes: per-model cold-variant retrains
(`--retrain-config`, first one doubles as the LEO5 retrain-path smoke),
cold evals (`--canonical-results-dir` wired), D3 popularity-negatives test
pass, D4 stratified readouts, popularity reference row — then the frozen
reference table closes this section.

### Hyperopt-wave verification + Q1 cross-check (2026-07-15)

All seven wave jobs completed 2026-07-14 (11:51–15:21 CEST), every one exit 0
with `RUN_OK: 30/30 trials`; full artifact sets persisted under
`/scratch/c7031336/protomf_results/<model>_hm_1_month_s38210573/`
(checkpoint, config, val/test metrics, hardware logs; explanations for the
three proto models); W&B synced. **Q1 cross-check vs the June 2026-06-08
runs (gate obligation): PASSED, no finding** — `mf`/`acf`/`user_proto`
test metrics identical to 4 decimals, `item_proto` +0.0001 (both metrics),
`user_item_proto` −0.0010 NDCG@10 / −0.0019 HR@10. The re-run five are the
canonical rows (clean provenance); June numbers retire to cross-check
status.

### Run-submission record (part 2, cold-retrain wave — 2026-07-15)

Precondition discovered & fixed at submission: `slurm/run_combo.slurm` had
no `--retrain-config` passthrough (the retrain path had only ever run
locally) — added (commit `57d963a`), pushed, cluster pulled forward to
`5dd99df` (= local HEAD). Retrains therefore run at `5dd99df` vs the
canonical wave's `385ab0e`; comparability unaffected (interim commits are
dc05/docs work, regression-clean on the H&M scored path per the dc05
SC.3/SC.4 records; C8 manifests record per-run commits).

Charter C8 manifest. Common to all seven: dataset `hm_1_month_cold`
(S0.3 variant, verified present on scratch), seed 38210573, explicit
`--profile dev` (F-S0-10 ruling), `--retrain-config
<canonical-results-dir>/config.json` (single fixed config, num_samples=1
forced python-side), uniform-99 eval negatives with the F-S0-09
cold-exclusion fold-in active (marker file present), `use_bias=0`
(config-inherited), canonical 5-field set (feature rows),
num_workers=1 / concurrency 1 (single trial) / 4 CPUs / 20G / A30,
W&B tags `s07-ref` + auto `dev`, branch `feat/scrutiny`, cluster HEAD
`5dd99df`. Each job submitted via its own `/leo5-submit` invocation (user
hard gate); job 1 (mf) doubled as the LEO5 retrain-path smoke — verified
live before jobs 2–7 were queued: single trial `7ab12_00000`, params ==
canonical best config except `data_path` (cold variant), dev profile
resolved (60 epochs), epoch 1 trained with warm-val HR@10 = 0.099,
41 s/epoch (worst-case 60 epochs ≈ 45 min, inside the 2 h limit).

| # | Model | Job ID | Time limit |
|---|---|---|---|
| 1 | mf | 7026366 | 2:00:00 |
| 2 | acf | 7026871 | 2:00:00 |
| 3 | user_proto | 7026894 | 4:00:00 |
| 4 | item_proto | 7026936 | 4:00:00 |
| 5 | user_item_proto | 7026968 | 4:00:00 |
| 6 | lightfm_tags | 7026969 | 2:00:00 |
| 7 | lightfm_tags_ids | 7027006 | 2:00:00 |

(Proto models got 4 h: their epochs run slower than mf's measured 41 s;
uniform CPUs/mem footprint per the fleet rules.)

*Provenance footnote (same day): committing this manifest moved the cluster
checkout to `572678f` while jobs 5–7 were still pending — those start from
`572678f`, a docs-only delta over `5dd99df` (byte-identical code paths).*

Still to run after the retrains complete: per-row cold evals
(`utilities/cold_eval.py` with `--canonical-results-dir`, incl. kNN
fallback for CF rows, tie diagnostic, popularity reference row), D3
popularity-negatives test pass on the canonical checkpoints, D4 stratified
readouts — then the frozen reference table closes this section.

### D3 + D4 secondary readouts (2026-07-15, login-node A30 per the small-workload policy)

D3 required a small runner (`Master/scripts/popneg_readout.py`, commit
`00877b0` — the ratified S0.1 D3 design had no script yet): one test-only
pass with `eval_neg_strategy='popular'` on a saved checkpoint,
num_workers=0 after `reproducible(seed)` → identical negative draws across
model rows (the t05 convention). Locally smoke-tested on the June mf
checkpoint before cluster use. D4 used the existing
`stratified_readout.py`. Both ran over all seven canonical results dirs,
zero failures; artifacts in `<results_dir>/popneg_readout/` and
`<results_dir>/stratified_readout/`. Full tables now live in
`replication_report.md` §"S0.7 Reference Fleet". Headline observations:

- **D3:** CF rows drop 0.18–0.34 HR@10 under popularity-sampled negatives;
  the two lightfm feature rows drop only 0.04/0.08 and land at
  0.418/0.432 — above every CF row except ~tied with `user_item_proto`
  (0.401). The uniform-99 headline's popularity-blindness is now
  quantified, per the S0.1 "gaps we will not fix" mitigation.
- **D4:** user-history quartiles are FLAT for all seven models (Q1≈Q4);
  the item-popularity tail is not — `user_proto` reads 0.0000 HR@10 on
  1–5-interaction items, mf/acf ≤0.02, while lightfm rows hold 0.34+
  across all buckets. Disclosure: D4's re-derived overall values differ
  from the stored test metrics by ≤0.002 (fresh negative draws in the
  num_workers=0 pass) — sampling noise, within one binomial SE.
- Both readouts are dev-tier working numbers (two-tier policy) and
  **secondary** — headline comparisons stay uniform-99 (D2).

### Explanation-artifact eyeball + prototype-collapse observation (2026-07-15)

The three proto models' auto-explanation artifacts (both scorings) verified
present and loadable; mirrored to
`Master/experiments/s07_reference/<model>_.../explanations/`. t-SNEs and
naming CSVs are legible (prototypes sit inside item clusters, names
sensible). **Working observation (scrutiny-phase basis, NOT thesis
material — dedicated experiments at writing time per the standing rule):**
severe prototype collapse in the single-sided hosts at their best dev
configs — `user_proto`: 76 prototypes but only 11 distinct top-k item
sets / 5 distinct names; `item_proto`: 76 → 21 sets / 7 names, many
prototypes at shifted-cosine 2.0 (cos = 1) on identical item sets;
`user_item_proto` much healthier (user side 76 → 64 sets / 23 names; item
side 11 → 8). Both June and S0.7 runs show the same pattern (systematic,
not run noise). Not an instrument defect — the naming pass faithfully
reports duplicate directions. Consequences routed: (a) SC.8
prototype-geometry instruments (pairwise prototype cosine, activation
spread) will quantify this on candidate/host pairs; (b) context for the
host-vs-variant interpretability comparison (the baseline's effective
prototype vocabulary is far smaller than K); (c) feeds the thesis
knobs/metrics (Rashomon) chapter as a concrete degeneracy exhibit.
Proxy caveat: top-k-set identity is coarse; dev-profile config selection
may differ from paper-budget selection.

*Mechanism + direct geometry (same day, user question at the observation):*
the winning single-sided config (both models byte-identical — fixed-seed
TPE startup sharing) has `sim_proto_weight 1.798` vs `sim_batch_weight
0.0018`: the prototype→data anchoring force (which tolerates/encourages
co-location; `reg_proto_func` rewards each prototype's best-item
similarity independently, `feature_extractors.py:495`) runs ~1000× the
coverage force that separates indirectly (F-DC01-05's account, here
empirically confirmed by its absence). Selection on val HR@10 is
collapse-blind. Direct checkpoint measurement (June checkpoints, local):
`item_proto` median pairwise prototype cosine **1.000**, 69.5% of pairs
>0.999, ~2 direction bundles in 76 slots; `user_proto` 5 direction
clusters (cos>0.999 within, spread between) — matching its 5 unique names
exactly. Prototype norms are near-uniform and tiny (gauge axis: cosine
read-out makes norm loss-invisible; weight decay canonicalizes it —
F-DC01-08's mechanism on the norm axis). The ordered top-k-set counts
above overstated distinctness (ordering noise between near-identical
directions); unique-name counts track direction clusters faithfully.
Instrument note for SC.8 figures: the t-SNE uses `metric='cosine'`
(correct geometry) but stacks duplicate directions on one point —
prototype multiplicity needs an explicit encoding (dot size / count
annotation) before t-SNEs can honestly show degeneracy.

*Popularity-bias identity of the collapsed bundle (same day, user
hypothesis, empirically confirmed with inverted mechanics):* the
`item_proto` dominant bundle (62/76 prototypes) does NOT point at popular
items — it points at the **rare pole** (Spearman(cos-to-bundle, train
popularity) = −0.76 over all 13,651 items; top-aligned items median pop 6
vs catalog median 13; anti-aligned = the head). Users weight the bundle
coordinates **negatively** (89.6% of the 73,418 users have a negative
bundle-coordinate sum; mean −0.73), so the bundle acts as a **self-built
global item-popularity bias term** — the explicit channel the fleet's
`use_bias=0` (C3) removed, reconstructed from prototype capacity. The
62-fold duplication is coefficient amplification under weight-decay-bounded
per-coordinate magnitudes (mean |coord| identical inside/outside the
bundle, 0.0127). Coheres with D4 (`item_proto` most popularity-skewed:
0.914 head / ~0 tail) and with what popular-negative sampled softmax
rewards. **Flag: candidate evidence for the C3 bias-on ablation trigger
(vault 2026-07-11_1556) — user decision, not queued.** Thesis routing:
hidden-effects exhibit (selection-blindness + gauge + missing bias channel
⇒ prototype capacity silently repurposed as a popularity knob). All
single-seed dev-config working evidence (standing rule).

### Retrain-wave completion + cold evals + THE FROZEN REFERENCE TABLE (2026-07-15)

**Retrain wave:** 7/7 COMPLETED exit 0, `RUN_OK 1/1` each (elapsed mf 1:10,
acf 1:50 — 9 min under its 2:00 limit, noted for future retrain sizing —,
user_proto 2:04, item_proto 1:37, user_item_proto 1:34, lightfm_tags 1:33,
lightfm_tags_ids 0:45). **Cold evals:** login-node A30 per the
small-workload policy (post-hoc measurement on trained checkpoints; the
SLURM hard gate covers submitted training runs — flagged to the user
before execution, no objection). First sweep attempt failed 7/7 on a
module-path detail (`python utilities/cold_eval.py` → needs `-m
utilities.cold_eval`); fixed, relaunched, row 1 verified end-to-end before
the rest ran (user directive, now standing memory). All 7 reports
complete: `<retrain_dir>/cold_eval/report.{json,md}`.

**Frozen reference table — hm_1_month, seed 38210573, dev profile.**
Warm = canonical test (uniform-99). Cold columns from the S0.3 machinery
(cold-vs-cold primary: 121,024 rows, pool 2,730 cold items, chance
HR@10 = 0.10; kNN = attr-kNN fallback patch, CF rows only). Popularity
reference (rank scorer, cold-vs-cold): **0.1202** HR@10, all rows.

| Model | Warm HR@10 | Warm N@10 | Cold-cold HR@10 | +kNN patch | Cold-all HR@50 | Warm-delta HR@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mf | 0.3652 | 0.2233 | 0.0010 | 0.2412 | 0.334 | +0.051 |
| acf | 0.5764 | 0.3307 | 0.0718 | 0.3094 | 0.996 | +0.054 |
| user_proto | 0.5725 | 0.3322 | 0.0000 | 0.2999 | 0.817 | +0.062 |
| item_proto | 0.4938 | 0.2631 | 0.0000 | 0.2358 | 0.880 | +0.044 |
| user_item_proto | 0.6587 | 0.4164 | 0.0000 | 0.4155 | 0.413 | +0.035 |
| lightfm_tags | 0.4582 | 0.2709 | **0.4705** | — | 0.810 | −0.019 |
| lightfm_tags_ids | 0.5087 | 0.3063 | **0.5080** | — | 0.855 | −0.021 |

Readings (working evidence, two-tier policy):
1. **Native feature models nearly close the warm-cold gap** (0.47/0.51
   cold vs 0.46/0.51 warm) — features fully carry cold items; the ID-drop
   convention (F-S0-13) costs `lightfm_tags_ids` nothing measurable.
2. **Raw CF rows sit at/below the disclosed noise floor** (proto rows
   exactly 0.0000: decayed cold rows → constant prototype similarities →
   all-tie → introselect exclusion, the F-S0-08-documented convention;
   acf reads 0.0718 — its cold repr degrades to a near-uniform anchor mix,
   a generic but non-zero score). All far below the 0.10 chance line and
   the 0.1202 popularity reference.
3. **The attr-kNN patch lifts CF rows to 0.24–0.42 — clearly below the
   native feature rows**: post-hoc attribute patching < built-in feature
   composition, the cleanest isolation the fleet provides for "features
   must enter the model, not just the fallback".
4. Tie-block diagnostic: lightfm rows show ~0.99 signature-twin share in
   cold top-k (cold scoring is feature-only ⇒ signature twins tie —
   dc02's tie-cap concern, now measured on the baseline); CF+kNN rows
   0.37–0.54; item_proto 0.0002.
5. Warm deltas: CF rows +0.035..+0.062 (variant warm test excludes the
   cold-test users — a population shift, uniform in sign across CF rows),
   lightfm rows −0.02 (opposite sign; small). Disclosed, no repair —
   cross-model comparisons stay within-column.
6. Oddity noted, no action: acf's cold-vs-all HR@50 = 0.996 (its generic
   anchor-mix score ranks cold items high among all items at K=50) —
   diagnostic column only.

**S0.7 artifact set is complete** — canonical fleet + retrains + cold
evals + D3/D4 + explanation artifacts + collapse/bias observations —
pending the closing gate. Result-dir paths: canonical
`/scratch/c7031336/protomf_results/<model>_hm_1_month_s38210573/`, cold
`..._hm_1_month_cold_s38210573/` (cold_eval/ inside), explanations
mirrored to `Master/experiments/s07_reference/`.
