# Master Plan — Prototype-Based Explanations for Interpretable Fashion Recommendations
### Master's Thesis · ProtoMF + H&M · working document (updated 2026-07-12)

## What this is
Extend ProtoMF (RecSys 2022) with the H&M fashion dataset and **feature-grounded prototypes**. Two contributions:
1. A **staged feature-grounding lineage** — extend ProtoMF variant-by-variant (item → user → merged), each grounded variant compared against its own like-for-like host.
2. **Quantified interpretability** of the resulting fashion explanations, strictly after the lineage.

Kaggle/MAP@12 comparison is **cut** (single-stage CF scorer ≠ two-stage retrieve→rerank; qualitative note only).

---

## ▶ Where we are → where we're going

**Now: the Scrutiny Protocol.** All four design candidates (dc01–dc04) exist; new implementation is paused to **audit and repair** the eval foundation + each candidate first. Runs via `/scrutiny` on branch `feat/scrutiny`; source of truth `Master/docs/scrutiny_protocol.md` + dossiers under `Master/docs/scrutiny/`.
- **S0 foundation** — mostly closed; ◐ 7-job reference fleet in flight on LEO5 (`hm_1_month`). On landing → benchmark rows in `replication_report.md` + frozen dossier table → per-model cold retrains/evals.
- **SC(dc01)** — SC.1a/1b/2 ✅ closed; SC.3 paused by host redirection. **Next concrete step:** the dc01 host-redirection amendment cycle (re-host UI→I → **fI-ProtoMF**; delta re-scrutiny + toy re-check + SC.3 re-run).

**The lineage (thesis part one).** Not "compare four candidates" — a lineage mirroring ProtoMF's own I / U / UI build order. Each grounded variant is a whole, explainable score with no opaque passenger part, compared to its host from the reference fleet. Each new stage passes the design-candidate + scrutiny machinery before it counts.

| # | Stage | Host / compare-to | Idea | Status |
|---|---|---|---|---|
| 1 | **fI-ProtoMF** (= dc01 re-hosted) | I-ProtoMF (`item_proto`) | item factors composed from feature words | ◐ scrutiny (host-redirect next) |
| 2 | Route-1 | — | user-proto → item-proto space via tie | ▢ presented, not implemented |
| 3 | **fU-ProtoMF** | U-ProtoMF (`user_proto`) | user = aggregate of purchased items' feature words (per-purchase attribution) | ▢ new DC cycle |
| 4 | **afU-ProtoMF** (working name) | U-ProtoMF / fU | fU + user attributes, entering last & weakest (cold users) | ▢ new DC cycle |
| 5 | **fUfI-ProtoMF** | UI-ProtoMF (`user_item_proto`) | both grounded sides under the double tie | ▢ new DC cycle |
| 6 | Speculative | — | e.g. CNN pulling images in under the rec loss | ▢ later |

Then, **strictly after**: an **Interpretability Quantified** section (facet-A: profile entropy/sharpness, feature-explained fraction, prototype purity, Rashomon-set tuning).

**Queued dataset variant (decided 2026-08-17): `hm_kcore`** — H&M 3–6-month window + drastically higher user k-core, tuned so total size stays ≈ `hm_1_month`. Goal: push mean/median user history up as far as possible **without** the problems of long windows in this dataset (seasonality/catalog drift diluting co-purchase structure). Rationale: `hm_1_month`'s short histories (median 5 purchases) let the collaborative space collapse into a popularity ranker (2026-08-17 vault entries: popularity channel, coverage regularizer, history-gap measurements) — it stays as the deliberately-kept pathological exhibit — but the candidates' feature/cold-item story needs a fashion testbed where prototypes can actually develop; with a collapsed warm space, feature embeddings can only encode "has popular features". Build/scope decision at the fUfI base-settings gate; needs a splitter variant + a bounded mini-foundation of baseline rows.

**Guiding principle:** an interaction is `taste·content`; item content is recorded, user taste is latent and revealed through behavior → ground explanations in item vocabulary first.
**Fallback/parallel:** dc02–dc04 stay as fallback / parallel idea-sources; literature reading is always-on, never a discrete phase.
**Reasoning trail:** 2026-07-11/12 protocol-vault entries + memory `project_scrutiny_protocol` / `project_c1_lineage_commitment`.

---

## Fixed constraints
- **Datasets:** `hm_1_month` (V1, 623K int, all *quantitative* claims) · `hm_3_month` (V2, *qualitative* showcase only) · `hm_full` (optional final confirmation). Shrink levers must preserve H&M sparsity. **Planned addition (2026-08-17): `hm_kcore`** — 3–6-month window + high user k-core at ≈V1 size, the long-history fashion testbed for the candidates' feature/cold story (see queued-variant note above; decision at the fUfI base-settings gate).
- **Compute:** LEO5 cluster · **one run = one job** (no fragmenting) · dev runs aim ≪1 day · hard wall 10 days.
- **Eval:** HR@10 / NDCG@10 leave-one-out is the sole metric + model-selection criterion.
- **Branches:** `main` untouched reference · `feat/scrutiny` integration branch. **Temp files → `Master/temp/`.**

---

## Progress

**Phase 1 — Setup ✅** environment (CUDA stack, RTX 4070 Ti SUPER + LEO5), WoL/SSH tooling, W&B, conventions.

**Phase 2 — Replication & understanding ✅** 10/10 replication runs (5 models × ml-1m/amazon2014; lfm2b unavailable) → `replication_report.md`; explanations generated; deep paper + codebase walkthroughs in `Master/understanding/`.

**Phase 3 — H&M integration ✅** splitter with k-core/leave-1-out; `hm_1/3_month` + `hm_full` built & registered; Kaggle solutions studied (`hm_dataset_notes.md`); all 5 CF baselines run on `hm_1_month` (dev profile) → `replication_report.md`. Open: whether prod-profile re-runs are needed before headline numbers.

**Phase 4 — Feature-grounding lineage ◐ (current)** — the pivot: design-candidate protocol produced dc01–dc04 (not a bake-off); the "implement-all & compare" plan and dual-mode competition eval are **cut**. Now in Scrutiny Protocol → then the lineage above.
- dc01 `feature_item_proto` → **fI-ProtoMF** ✅ implemented, in scrutiny · dc02 `attr_item_proto` ✅ implemented+validated · dc03/dc04 ✅ drafted.
- H&M feature analysis is **not** a standalone deliverable — it feeds the feature-lockstep / id-twin "braid" discussion inside each candidate's concept scrutiny.
- Explanations: names read from parameters (not post-hoc lift); per-purchase attribution for user-history stages; prototype-space projections *illustrative only* — all quantitative claims computed in original embedding space.

**Phase 5 — Writing (fragment-first, starting soon)** — small pieces written as their reasoning finalizes (much already exists as protocol/scrutiny material), glued progressively. Part one = the lineage in order (grounding argument + re-coordinatization figure → fI → fU/afU/fUfI); Interpretability Quantified after; Related Work / Future Work fed by ongoing reading.
