# Roadmap — ml-1m expansion + SC(dc05) continuation

> Written 2026-07-12 end-of-session, at the user's request, so a fresh
> context can follow the agreed plan. Companion (full reasoning):
> `Master/temp/ml1m_expansion_scope.md`. Authority for procedure:
> `Master/docs/scrutiny_protocol.md` (v1.2) — always re-read the step
> section before executing; this file is only the itinerary.

## State snapshot (2026-07-12 evening)

- Branch `feat/scrutiny`, clean. Today's commits: `9e571e4` (SC.1a),
  `40aec33` (SC.1b), `a609148` (SC.2), `df512ac` + `ac640ae` (charter
  amendments: ml-1m in / amazon2014 out / strict-LightFM fields / scoping).
- **dc05 scrutiny:** SC.1a, SC.1b, SC.2 CLOSED (dossier
  `Master/docs/scrutiny/sc05_history_composed_user_factors.md`; findings
  F-DC05-01..10 all dispositioned). **Next dc05 step: SC.3.**
- **Charter amended** (S0.6 C2/C5/C6, dated notes): ml-1m = second
  quantitative testbed; ml-1m canonical fields = **genres + Tag-Genome tags
  @ relevance ≥ 0.8, indicator bags, no year** (B5's published MovieLens
  recipe, user directive); amazon2014 discarded (no features — vault
  `2026-07-12_2114`); ml-1m fleet rows defined, submitted lazily.
- **Data staged (laptop):** `data/ml-1m/movies.dat` + `data/ml-1m/tag-genome/`
  (gitignored; .gitignore line added). Audit script saved:
  `Master/temp/dc_checks/s0/ml1m_feature_audit.py` (commit it WITH the S0.5
  addendum). Measured facts: genome covers 99.0% of the 3,125 split movies;
  97.0% full rows / 3.0% genres-only; only 0.54% of train interactions touch
  genres-only items; vocab@0.8 = 1,043 tags (B5 reported 1,030 — recipe
  validates); token counts p10=3 / median 10 / p90=24 / max 72.
- **Still pending on the cluster (independent):** S0.7 part 2 — H&M fleet
  results → frozen reference table → replication_report. Gates every SC.6.

## The agreed sequence (each step ends at a user gate)

### Block 1 — S0 extension for ml-1m (instrument first; no cluster time)

> Progress note (2026-07-12, late evening): steps 1 and 2 are **DONE and
> gate-CLOSED** (commits `3796566`/`aa0eab6` = S0.5 addendum, raw bags
> ratified, popularity-coupling ρ=0.672 must-mention + vault entry
> `2026-07-12_2208`; `af929e4` + closure = S0.3 addendum, genome
> provenance disposition, "interaction-hidden, not new-release" wording,
> pool assert).
>
> Progress note (2026-07-12, night): step 3, the **S0-build extension, is
> DONE and gate-CLOSED** — all five components (a)–(e) landed, each with a
> live component gate + a closing combined-artifact gate (commits `3295c47`
> bag layout, `d660f66` fU bags, `e8f519c` field plumbing, `74906f6` ml-1m
> item_features + vault `2026-07-12_2313` popularity-coupling live magnitude
> (~2×), `41fb655` cold generator + ml-1m_cold [C=625, min pool 341, zero
> collateral; hm_1_month_cold byte-identical]). Dossier: s0_foundation.md
> "S0-build extension" section. **Block 1 complete → NEXT: step 4, dc05 SC.3
> (`/scrutiny dc05`) — fresh session**; ml-1m_cold scp to LEO5 deferred to
> the first run plan citing ml-1m rows.

1. **S0.5 addendum** — invoke `/scrutiny step0.5 s0`.
   Content: ratify ml-1m field set (charter C5 amendment already written);
   missingness policy = genres-only rows for the 3% tail (one disclosure
   sentence, no machinery); **GATE DECISION — token mass:** strict-LightFM
   raw bags (tag-rich movies weigh more; default, follows B5) vs per-movie
   normalization (every movie votes mass 1). Measure ml-1m item-signature
   collisions (fI twin counterpart) + basket stats; commit
   `ml1m_feature_audit.py` with the addendum.
2. **S0.3 addendum** — invoke `/scrutiny step0.3 s0`.
   Content: ml-1m cold-item variant spec anchored to B5 §5's published cold
   protocol (20% of items removed) exactly as the H&M spec anchored;
   generalization plan for `data/hm/make_cold_variant.py` (H&M-coupled);
   leakage checklist walk for ml-1m incl. the WRITTEN genome-provenance
   disposition (genome = static item descriptors computed from post-2003
   community tagging; not derived from OUR training matrix).
3. **S0-build extension** (one supervised code block, gated per component;
   dc_checks/s0 tests; modifications log for upstream files):
   (a) item-side **bag layout** — embedding-bag variant of `FeatureEmbedding`
   (+ keystone: bag-vs-padded bit-identity on H&M inputs; F=0 reductions);
   (b) fU builder (`build_user_history_weights`) multi-value field support;
   (c) per-dataset `feature_fields` plumbing (currently hardcoded H&M-5 in
   `confs/hyper_params.py`; C8 manifests must record resolved fields);
   (d) model-grade ml-1m `item_features.csv` builder (genres + tags@0.8;
   extends `data/ml-1m/build_item_features.py`, today display-only);
   (e) generalized cold generator → `ml-1m_cold` artifact.

### Block 2 — dc05 scrutiny resumes

4. **SC.3** — invoke `/scrutiny dc05` (dossier shows SC.2 closed → SC.3).
   Verification-heavy (SC.2 amendments were concept-level, no scored-path
   change): built `feature_user_proto` vs amended concept; carried checklist
   (NaN guard in the new builder — F-S0-06 symmetry; explicit config keys
   incl. `use_bias`/`use_id_feature`); fU-on-ml-1m via the new builders;
   ml-1m basket stats; dated dc05 design-doc note (§3.1/§4: "exactly 5
   tokens per purchase" is H&M-specific). Extend dc_checks/dc05.
5. **SC.4** — black-box spec-vs-code subagent (amended design doc + code
   ONLY) + first-party audit (host conventions, optimization parity, test
   adequacy, feature entry end-to-end BOTH datasets).
6. **SC.5** — comparability matrix fU-vs-host; attribution gates (GR7);
   **dual naming route for fU = F-DC05-03(b) obligation** (post-hoc pass
   beside intrinsic, F-DC01-13 shape); dataset list pinned
   {hm_1_month, ml-1m}; renderer/cards verified on BOTH datasets (movie
   titles as purchase labels — apply the layout-tolerance learnings entry);
   gating-consistency check. Index → hardened.
7. **SC.6** — run plan under the amended charter, clause by clause.
   PRECONDITION: S0.7 part-2 frozen H&M table landed (re-check S0 per the
   dossier header declaration). First citation of ml-1m fleet rows triggers
   the **FLEET MODE campaign** (leo5-submit skill has an explicit FLEET MODE
   section: benchmark-shaped uniform geometry, concurrency pinned 5,
   num_workers=1 uniform, Lean/num_workers=0 BANNED for compared rows) —
   ~6 ml-1m reference rows + cold retrains + dc05 rows (ids/noid × both
   datasets + cold retrains). **Every run = its own /leo5-submit invocation
   (hard gate), never batch.**
8. **SC.7** — flush learnings + protocol retrospective → **PAUSE** for runs.

### Block 3 — post-run (after cluster results)

9. **SC.8** — invoke `/scrutiny resume dc05`. Charter-compliance audit;
   committed instruments fire: comparative B(t) rule (F-DC05-06 — fU-vs-host
   variance share), angle-to-mean per history-length band diagram (M5′
   precondition figure), cloud spread in BOTH weightings (user-uniform +
   interaction-weighted, F-DC05-02), metadata-norm/ID-share vs |H| strata
   (F-DC05-01), gauge/twin instruments, profile-validity spot-check via the
   dual naming route, D4 bands (3/4–5/6–12/13+ on H&M; ml-1m bands TBD at
   SC.6 — ≥20-rating floor). Rendered side-by-side breakdowns spot-check.
10. **SC.9** chapter draft → **SC.11** via **SC.10** cold read.

### Parallel / independent threads

- **S0.7 part 2** on cluster progress: fleet results → frozen table →
  `replication_report.md`.
- **dc01 resume** (user's timing): SC.3/SC.4 *extension* for fI-on-ml-1m
  (consumes Block-1 bag layout; new checks only), SC.5 *partial* re-run
  (ml-1m rendering; protocol's stale-matrix mode), then SC.6. NO concept
  steps re-run anywhere (mechanism-level, dataset-agnostic).

## Standing rules that bit today (do not relearn)

- Every measured number needs its committed generator script (learnings
  ledger; `ml1m_feature_audit.py` awaits its commit at step 1).
- Work empirically, not scholastically: hypotheses get designed tests;
  attribution (host-class vs candidate pathology) is decided by comparative
  instruments on both arms, never by argument.
- Scrutiny-phase measurements = working evidence only; the thesis gets
  dedicated experiments designed at writing time.
