# ml-1m Expansion — Magnitude Scoping (pre-dc05-SC.3)

> 2026-07-12, written at the user's request before proceeding with dc05 SC.3.
> Decision context: charter amendment `2026-07-12` (C2/C5/C6), vault
> `2026-07-12_2114`, user directives at the SC.2 gate follow-up: ml-1m joins
> as second quantitative testbed; amazon2014 discarded (no features);
> **features "the way LightFM did"** — genres + Tag-Genome tags @ relevance
> ≥ 0.8 (B5 §4.1 published recipe; supersedes the interim genres+year
> proposal); cold variant, cluster-run plan, and explanation-system parity
> for every prototype model on ml-1m are all in scope.
> Temp-file rule: this is a planning artifact; the durable commitments live
> in the charter, the dossiers, and the S0 addenda it plans.

## 0. The user's list, correctness-checked (all four items CORRECT)

1. **"New cold version of the dataset" — correct.** `ml-1m_cold` via the
   S0.3 machinery. Verified: the generator `data/hm/make_cold_variant.py`
   is H&M-located/-coupled — it needs generalization (or an ml-1m sibling),
   plus an S0.3 spec addendum (removal protocol anchored to the same
   published B5 source the H&M spec anchored to; NEW leakage item: Tag-Genome
   provenance disposition — genome relevance scores are computed from
   MovieLens community tagging that post-dates ml-1m's 2000–2003
   interactions; they are static item descriptors, not derived from OUR
   training matrix, but the S0.3 checklist walk must say so in writing).
2. **"Follow LightFM closely while following ProtoMF" — correct and
   realizable, one nuance.** Dataset = ProtoMF's own ml-1m artifact (host
   comparability + replication continuity; ProtoMF's paper ran ml-1m).
   Features = LightFM's published MovieLens methodology (genres + Tag
   Genome @ 0.8, indicator bags). Nuance disclosed: LightFM used
   MovieLens-**10M**; we port their feature *recipe* onto ProtoMF's ml-1m
   (movieIds are stable across MovieLens releases → genome joins cleanly;
   coverage % to verify — fallback for uncovered movies: genres-only rows,
   declared missingness policy).
3. **"Plan new cluster runs" — correct.** See §3. ml-1m is the cheap
   dataset; the full addition is ~8–12 dev-profile runs, each its own
   `/leo5-submit` (hard gate), submitted lazily (C6 amendment).
4. **"Explanation system must work and run for each prototype model on
   ml-1m" — correct, and partially ready.** Verified ready: the naming
   layer already has ml-1m multi-value genre support
   (`naming/config.py: multi_value_sep="|"`); tsne/top-k/weight-viz ran on
   ml-1m in replication; `build_item_features.py` exists (display-grade).
   Missing: intrinsic naming vocab per dataset, renderer/cards slots
   verified on ml-1m (movie-title purchase labels — the fU layout-tolerance
   lesson applies), tag-name display (genome-tags.csv), per-dataset
   `feature_fields` plumbing into accessors, D4-band analogues for ml-1m.

## 1. "Maybe more?" — yes. The items the list didn't name

5. **Bag layout in the model code (the single biggest code item).**
   - fU side (`build_user_history_weights` + `HistoryFeatureEmbedding`):
     layout already variable-length — moderate builder extension (multi-token
     items; proposal: every token of every purchase gets weight 1/|H_u|,
     identical rule to H&M; the per-purchase regrouping C2′ stays exact since
     tokens partition by movie).
   - fI/lightfm item side (`FeatureEmbedding`): fixed one-value-per-field
     width breaks → embedding-bag variant, WITH keystone tests
     (bag-vs-padded bit-identity on H&M inputs; F=0 reductions). Upstream
     files → modifications log. Needed by the lightfm fleet rows and by fI
     when dc01 resumes — therefore **S0-build-extension scoped** (shared
     infra), not dc01-scoped.
6. **Per-dataset config/field plumbing.** `feature_fields` is hardcoded
   (H&M five) inside each config (verified `confs/hyper_params.py`);
   feature models on ml-1m need dataset-conditional field resolution +
   C8-manifest recording.
7. **Mechanism semantics under bags — dc05 design-doc note.** "Each purchase
   contributes exactly one value per field; metadata mass = F = 5" is
   H&M-specific; on ml-1m per-purchase token counts vary (1–6 genres +
   0–k tags). Mirror-geometry arguments and all §4 numbers are V1-scoped;
   ml-1m gets its own basket stats (D_max, weights distribution) at
   SC-stage. Dated amendment to dc05 §3.1/§4 at SC.3.
8. **Missingness goes LIVE.** H&M had 0% missing; ml-1m WILL have movies
   without genome tags (and 4 items had no movies.dat match in the display
   build). The F-S0-06 NaN/missing guards stop being defensive and start
   being exercised — missing-policy must be an explicit S0.5-addendum
   decision (proposal: genres-only rows for uncovered movies, disclosed).
9. **Hidden-effects instruments get per-dataset readings.** V changes
   (~18 genres + genome-tag vocab@0.8 — size to measure); omnipresence
   ("Drama" ≈ half the catalog) and lockstep (genre↔tag correlations!) need
   ml-1m decoupling/collision numbers; C10′'s heterogeneity-axis question
   and the D4 bands re-pose on a ≥20-rating floor population (thin-user
   stratum barely exists on ml-1m — the two datasets probe opposite ends,
   which is the point of including it).
10. **Data logistics.** Downloads: `movies.dat` (tiny, same GroupLens zip),
    Tag Genome (genome-scores/genome-tags, sub-GB) — laptop + LEO5 copy;
    `ml-1m_cold` variant dir generation; drive-sync unaffected (data
    gitignored).
11. **Unchanged (verified, no work):** Evaluator/eval protocol
    (dataset-generic; F-S0-03 assert covers all paths); F-S0-04 already
    measured ml-1m (0 train-unseen test items); RecSys/trainer/dataset
    classes (extractor interface absorbs everything).

## 2. Scrutiny-step re-run map (S0 + dc01 + dc05)

| Step | Verdict | Content |
|---|---|---|
| S0.1 eval audit | **no re-run** | ml-1m was inside the original audit (F-S0-03/04 measured it); protocol unchanged. Dated note only. |
| S0.3 cold spec | **re-run mode addendum** (`step0.3 s0`) | ml-1m cold-variant spec: same published-protocol anchoring, generator generalization named, leakage walk incl. the genome-provenance disposition (item 1 above). |
| S0.4 CBF spec | **small addendum** | lightfm rows on ml-1m; bag-layout reference; tags+ids/tags arms unchanged conceptually. |
| S0.5 feature audit | **re-run mode addendum** (`step0.5 s0`) | THE data-facts step: genome coverage %, vocab size, missing policy, ml-1m signature-collision numbers (fI twins over genre+tag signatures), entry-correctness of new builders. Needs the downloads; cheap, local. |
| S0.6 charter | **done** | C2/C5/C6 amended (C5 refined this evening to strict-LightFM fields per user directive). |
| S0-build | **extension block, gated per component** | bag layout + builders + per-dataset field plumbing + ml-1m `item_features.csv` (model-grade) + generalized cold generator; dc_checks/s0 additions (keystones incl. bag-vs-padded bit-identity). |
| S0.7 | **lazy extension** | ml-1m fleet rows (§3) + black-box spec-vs-code on the NEW bag code + smoke; frozen-table gains an ml-1m panel. |
| dc01 SC.1a/1b/SC.2 | **no re-run** | mechanism-level, dataset-agnostic; V1 numbers stay V1-scoped working evidence. |
| dc01 SC.3/SC.4 | **extension at resume** | fI consumes the S0-build bag layout on ml-1m; new checks (t-suite additions), no re-audit of the H&M-verified work. |
| dc01 SC.5 | **partial re-run at resume** (protocol's stale-matrix mode) | comparability matrix re-check + ml-1m rendering verification (cards/breakdowns over genre+tag vocab). |
| dc01 SC.6+ | not yet run | simply includes ml-1m rows when it happens. |
| dc05 SC.1a/1b/SC.2 | **no re-run** | same reasoning as dc01; the SC.2 toy results are dataset-free geometry. |
| dc05 SC.3 (next) | **scope grows** | original SC.3 verification + fU-side multi-value builder support + dc05 §3.1/§4 dataset note. Can consume the S0-build extension OR carry the fU-side part itself (sequencing decision below). |
| dc05 SC.5/SC.6 | as planned | SC.5 pins the dataset list {hm_1_month, ml-1m} + renders BOTH datasets for BOTH models; SC.6 plans both + cites the amended charter. |

## 3. Cluster-run additions (all dev-profile, single seed, lazy, one /leo5-submit each)

**Fleet (S0.7 extension), ml-1m + ml-1m_cold retrains:**
`user_proto` (dc05's stage bar), `item_proto` (dc01's), `mf`, popularity
reference (cheap/local-ish), `lightfm_tags`, `lightfm_tags_ids` (both need
the bag layout first) — ≈ 6 hyperopts + their single-config cold retrains.
ml-1m at dev profile is the cheap end (replication ran full paper profile in
~3–4 h/model; dev 30/60 well under that). **Estimate: no envelope flags.**

**dc05:** `feature_user_proto` + `_noid` on ml-1m + cold retrain(s) — joins
the SC.6 manifest alongside the H&M rows.

**dc01 (at resume):** fI ids/noid on ml-1m + cold retrain — its SC.6.

## 4. Open decisions for the user (the gate questions)

1. **Strict-LightFM fields ratified?** genres + Tag-Genome@0.8 (published
   threshold, citable), NO year (LightFM used none), title excluded.
   Fallback if genome coverage on ml-1m proves poor: genres-only, disclosed.
2. **Sequencing:** recommended order = S0.5-addendum (data facts, needs the
   two downloads) → S0.3-addendum (cold spec) → S0-build extension (bag
   layout + builders + cold generator, gated per component) → **then** dc05
   SC.3 against the settled spec. Alternative (faster to SC.3, riskier):
   dc05 SC.3 first on H&M-only verification, ml-1m block after — but then
   SC.5 cannot pin ml-1m rendering and the routine pauses twice.
3. Genome download timing (laptop now vs with the S0.5 addendum sitting).
