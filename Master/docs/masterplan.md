# Master Plan: Prototype-Based Explanations for Interpretable Fashion Recommendations
## Master's Thesis — ProtoMF + H&M Dataset

---

## Context

This plan aims to cover the road to extending the ProtoMF (RecSys 2022) codebase to:
1. Reproduce the original paper's results across all datasets and models
2. Integrate the H&M Fashion dataset (Kaggle, ~31M transactions, ~105K articles with rich metadata)
3. Extend the prototype architecture to leverage item features (not just collaborative filtering)
4. Extract and analyze fashion-specific prototype-based explanations
5. Write the thesis in LaTeX within this workspace
It is iteratively refined and then simplified again and changed and such, it is a working document.

**Key decisions recorded:**
- **Thesis focus (realigned 2026-06-07):** The contribution is **(1) how much a feature-aware prototype extension beats the ProtoMF paper's CF-only baseline, and (2) the quality of fashion explanations/visualization.** Comparison to the Kaggle H&M competition (MAP@12) is a **side note only** — ProtoMF is a single-stage CF scorer and is structurally non-comparable to the competition's two-stage retrieve→rerank pipelines.
- **H&M dataset strategy (2026-06-07):** Two-version design. **V1 = `hm_1_month`** (1-month window, 5-core; ~623K int / 73K users / 13.7K items) — small efficient dev/judgment testbed, source of all *quantitative* claims. **V2 = `hm_3_month`** — repurposed as the later *qualitative* explanation showcase (showing, not proving; its metrics must not become headline claims). Shrink levers must preserve H&M's defining **sparsity**: window length and random user subsampling OK; high k-core / most-active-user sampling rejected (they raise density + bias to power-shoppers). `hm_full` retained only for an optional final confirmation run. Raw data shared in `data/hm/raw/`. *(Plan detail in `Master/temp/hm_1month_plan.md`; V1 built 2026-06-08 from full raw — 623,230 int / 73,418 users / 13,651 items, density 0.0622%, matching the sizing probe; registered in `start.py` and `run_combo.py`.)*
- **Compute feasibility envelope:** All runs on the **LEO5 cluster**, **no run fragmenting** (one run = one job). **Hard limit:** anything > 10 days is infeasible (LEO5 `std`-partition wall limit). **Soft limit (dev/testing):** anything > ~1 day is infeasible — aim well below.
- Environment: Upgraded from PyTorch 1.9.1 / CUDA 10.2 to CUDA 12.7 compatible stack — RTX 4070 Ti SUPER fully working
- Thesis: `Master/thesis/` (English, template TBD from supervisor)
- Replication: Full (all 5 models × 2 datasets — LFM-2b unavailable) ✅ complete
- W&B: Keep it, configure at start of Phase 2 (not Phase 1)
- Temp files: Always go to `Master/temp/` per CLAUDE.md rule

---

## ▶ Current Position & Direction (updated 2026-07-12)

*This banner is the "you are here" marker; it supersedes the phase text below where they disagree. The phase sections are kept as the historical plan-of-record.*

**Where we are — the Scrutiny Protocol phase (Phase 4, evolved).** All four feature-aware design candidates were drafted/implemented (dc01–dc04), then the user halted new implementation (2026-07-11) to **audit and repair the evaluation foundation and every candidate before building further.** This runs via `/scrutiny` on the `feat/scrutiny` branch; source of truth is `Master/docs/scrutiny_protocol.md` + the dossiers/ledgers under `Master/docs/scrutiny/`.
- **S0 foundation:** S0.1–S0.6, S0-build, and S0.7 part 1 (black-box audits) **closed**; gate fixes landed. S0.7 part 2 = a 7-job reference fleet **in flight on LEO5** (`hm_1_month`). When it lands: results → `replication_report.md` benchmark rows + dossier frozen table, then per-model cold retrains, cold evals, popularity-negative + stratified-readout passes.
- **dc01 = Candidate 1:** SC.1a / SC.1b / SC.2 **closed** (findings F-DC01-01..07 applied); SC.3 executed 2026-07-12 (F-S0-07(b) cold ID-drop verified+landed, decoupling-mass script committed) — its gate was then **paused by the host redirection below**. **Next concrete step: the dc01 host-redirection amendment cycle** (re-host the concept from UI-ProtoMF to I-ProtoMF → fI-ProtoMF; delta re-scrutiny + delta toy re-check + SC.3 re-run; see the sc01 dossier redirection note).

**Where we are going — the committed lineage (thesis part one), restructured 2026-07-12 to mirror ProtoMF's own U / I / UI architecture.** The thesis is no longer "compare four candidates," and no longer "one candidate carrying an opaque user half." It is a **staged lineage that grounds each ProtoMF variant separately against its own host** — clean like-for-like comparisons, no unexplainable passenger parts. Motivating principle unchanged: *an interaction is `taste·content`; item content is recorded, user taste is latent and revealed through behavior — so ground explanations in item vocabulary first.* Stages (each stage passes design-candidate protocol + scrutiny before it counts; later stages get their own dcNN numbers):
1. **fI-ProtoMF** (= dc01 re-hosted): feature-composed item factors on the **I-ProtoMF** host. The whole score is u·t* — every scored point flows through the grounded item-prototype surface; no projection half, no user-prototype opacity inside the candidate. Compare to **I-ProtoMF** (`item_proto`, already in the S0.7 reference fleet).
2. **Route-1 presented, not implemented:** pushing user prototypes through the tie into named item-prototype space — kept as a presented idea/bridge (its caveats included), implementation deliberately skipped.
3. **fU-ProtoMF** (new candidate, own DC cycle): user-side grounding on the **U-ProtoMF** host via interaction-history composition (route-2: user = aggregate of purchased items' feature words; per-purchase attribution). Compare to **U-ProtoMF** (`user_proto`, in the fleet).
4. **afU-ProtoMF** ("all-feature-User" — working name, to be settled with the supervisor; new candidate): fU + user attributes — attributes enter last, weakest, where history is thin (cold users). Compare to U-ProtoMF/fU.
5. **fUfI-ProtoMF** (new candidate): the merge — both grounded sides under the double tie (this is where the shared-instance tie, the exact linear-half attribution read-out, and the "both explanation surfaces at once" argument return). Compare to **UI-ProtoMF** (`user_item_proto`, in the fleet).
6. Speculative extensions (e.g. CNN bringing images in under the rec loss).

Then, **strictly after** the lineage: an **"Interpretability Quantified"** section (facet-A instrumentation — profile entropy/sharpness, feature-explained fraction, purity, determinacy annotations, Rashomon-set tuning).

**Fallback / parallel.** dc02–dc04 remain scrutinized/implemented and serve as fallback or parallel idea-sources (possibly their own lineages); the C1 lineage comes first. **Literature reading is an always-ongoing parallel process** feeding fresh ideas into all of the above — never a discrete phase that "finishes."

**Future Work earmark (not in scope):** interaction-attribute-aware prototypes — grounding on the *event's* context (season, channel, price) the current design deliberately drops; conditional on item-attribute grounding paying off first.

**Thesis writing:** first-generation text begins soon — **fragments first**, progressively glued into a coherent whole (see Phase 5, updated). The C1 lineage above is the part-one skeleton those fragments hang on.

Full reasoning trail for all of the above: the 2026-07-11/12 protocol entries in `Master/protocol_vault/` (thesis-outline-c1-lineage, taste-revealed, id-twins-popularity-braid, share-identifiability, future-direction) and the memory files `project_scrutiny_protocol` / `project_c1_lineage_commitment`.

---

## Folder Structure

```
ProtoMF/
├── CLAUDE.md                          # Updated with all project conventions
├── README.md, LICENSE, protomf.yml
├── start.py, experiment_helper.py
├── confs/hyper_params.py
├── data/
│   ├── amazon2014/, ml-1m/, lfm2b-1mon/   # Paper datasets (lfm2b unavailable)
│   ├── hm/                                 # H&M splitter + shared raw data
│   │   ├── hm_splitter.py
│   │   └── raw/                             # Raw Kaggle download (gitignored)
│   ├── hm_full/                             # Full H&M (~24 mo, 5-core, 26.2M interactions) — optional final-run only
│   ├── hm_3_month/                          # 3-month H&M window (5-core, 2.9M) — V2 qualitative showcase
│   └── hm_1_month/                          # 1-month H&M window (5-core, ~623K) — V1 dev/judgment testbed (not yet built)
├── feature_extraction/
├── rec_sys/
├── utilities/
├── pdfs_and_images/
└── Master/
    ├── Protomf-paper.pdf
    ├── docs/                           # Thesis notes, design docs, replication report
    ├── experiments/                    # All experiment results
    ├── notebooks/                      # Jupyter notebooks for analysis
    ├── literature/
    ├── scripts/                        # GPU monitor, combo runner, sampler
    ├── sensitive/                      # Gitignored — API keys, network config, SSH details
    ├── understanding/                  # Deep-dive learning sessions
    │   ├── paper/                     # Section-by-section paper Q&A (Phase 2.6)
    │   └── codebase/                  # Module-by-module code walkthroughs (Phase 2.7)
    ├── thesis/                         # Phase 5 — LaTeX thesis
    └── temp/                           # Temporary files (per CLAUDE.md rule)
```

---

## Phase 1: Setup and Tooling ✅

**Goal:** Working development environment, project conventions locked in, remote GPU accessible.

### 1.1 Folder Structure ✅
Created all directories, added `.gitkeep` for empty dirs, `.gitignore` in `data/hm/raw/`.

### 1.2 CLAUDE.md ✅
Established branch strategy, naming conventions, experiment naming, `ft_type` reservations, results storage paths, modification map, H&M data conventions.

### 1.3 Remote GPU Machine Setup ✅

**Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER — 16 GB VRAM, CUDA 12.7. University HPC cluster available as fallback possibly (NEEDS TO BE REQUESTED).

**Completed:**
- SSH access via Tailscale (`ssh gpu`) with key-based auth
- GPU confirmed: RTX 4070 Ti SUPER, CUDA 12.7
- Wake-on-LAN fully working from anywhere — phone relay (Termux + Tailscale) solves router ARP limitation
- Scripts with ping+SSH verification: `Master/sensitive/wake_gpu.sh`, `Master/sensitive/shutdown_gpu.sh`
- Claude Code skills: `/gpupc-start`, `/gpupc-stop`
- Reference docs: `Master/sensitive/gpu_machine.md`, `Master/sensitive/wol_relay.md`, `Master/sensitive/laptop.md`
- End-to-end tested: wake + SSH + shutdown from both home and external networks (2026-03-11)
- Repo cloned on remote machine
- ~~Upgrade PyTorch/Ray/wandb for CUDA ≥ 11~~ — conda env `protomf` created with CUDA-compatible stack
- ~~Create conda environment on remote, verify GPU in PyTorch~~ — CUDA confirmed working via bottom-up test (2026-03-18)
- ~~Test trivial training run remotely~~ — full stack validated: PyTorch+CUDA, Ray Tune, W&B all passing (2026-03-18)
- **Bug fix:** removed legacy `_metric/` prefix in `experiment_helper.py` — newer Ray Tune no longer auto-prefixes reported metrics, was causing silent hangs with W&B callback

**Workflow:** Direct VSCode SSH session optionally with Claude Code running locally on the GPU machine — no tmux/nohup job management needed.

**HPC cluster (if/when needed):**
- Request access, set up SLURM job scripts, test with small job

### 1.3.1 W&B Experiment Tracking Setup ✅
W&B fully integrated. Run naming follows `{model}_{dataset}_s{seed}_{trial_id}`, grouped by `{model}_{dataset}`, tagged with model/dataset/seed. `WandbLoggerCallback` logs config, per-epoch metrics (loss, HR@K, NDCG@K, timing, LR, patience), and system metrics. Dashboard used throughout replication for trial comparison and hyperopt analysis.

### 1.4 Local CPU Fallback ✅
Created local conda env (CPU-only, pip-installed with Bottleneck 1.3.4 build fix, protobuf 3.20.0 Ray compat). Verified `device='cpu'` path works. Added `debug_config` to `hyper_params.py` for fast local iteration.

### 1.5 Claude Code Environment ✅
Installed: context7 (docs), pyright-lsp (types), code-simplifier, claude-md-management, claude-code-setup. Custom commands: `/protocol`, `/commit`, `/gitcheck`, `/gpupc-start`, `/gpupc-stop`. GitHub token configured.

---

## Phase 2: Replication and Deep Understanding

**Goal:** Reproduce all paper results; fully understand the codebase; map code extension points.

**Execution order:** 2.6 → 2.7 → 1.3 (finish GPU env) → 2.3 → 2.5

### 2.1 Dataset Download and Preprocessing ✅ (2/3 datasets)
- **MovieLens-1M:** ✅ preprocessed
- **Amazon2014 Video Games:** ✅ preprocessed
- **LFM-2b 1-Month:** ⚠️ SKIPPED — dataset removed from official source (license issues). Replication proceeds with ML-1M and Amazon2014 only.

Each produces 5 files: `listening_history_{train,val,test}.csv`, `user_ids.csv`, `item_ids.csv`

### 2.2 W&B and DATA_PATH Configuration ✅
`DATA_PATH` auto-detected via relative path. W&B key loaded from `Master/sensitive/api_keys/wandb_key.txt`.

### 2.3 Full Replication ✅
10 experiments (5 models × 2 datasets), single seed — all complete. Test HR@10/NDCG@10 and best hyperparameters for all 10 in-scope runs (amazon2014 × 5, ml-1m × 5) recorded in [`Master/docs/replication_report.md`](replication_report.md). LEO5 job IDs traceable in the report.

Results → `Master/docs/replication_report.md` (canonical; the planned `results_table.csv` was never created)

> ⚠️ **Caveat (updated 2026-06-11):** Everything on LEO5 scratch has been mirrored to `Master/experiments/results/` via scp — 4/10 replication dirs are now local (`user_item_proto_ml-1m`, `mf_ml-1m`, `item_proto_ml-1m`, `item_proto_amazon2014`), plus all 5 `hm_1_month` baseline dirs and 3 `hm_3_month` dirs. The remaining **6 replication dirs** (`acf_ml-1m`, `user_proto_ml-1m`, and `mf`/`acf`/`user_proto`/`user_item_proto` on amazon2014) are **not on LEO5** — they live on the Windows GPU machine; retrieval deferred (user decision 2026-06-11, "we can get them later").

### 2.4 Modification Map ✅
Created `Master/docs/modification_map.md` documenting code extension points for adding datasets (Phase 3), item features (Phase 4), and explanation extraction (Phase 4).

### 2.5 Generate Replication Explanations ✅
- Loaded the best `user_item_proto` checkpoint on ml-1m (LEO5 job 6535752, Test HR@10 0.6246) via the modular pipeline (`utilities/explanations/loader.load_recsys_from_results_dir`)
- Ran all explainers (TSNE of user/item prototype spaces, top-K items per prototype, recommendation weight decomposition for user 42)
- Outputs in `Master/experiments/replication/explanations/ml-1m/` (per-dataset subfolder; amazon outputs sit alongside)
- Workflow documented in `Master/notebooks/03_ml-1m_explanations.ipynb` (sibling of `02_amazon_explanations.ipynb`)

### 2.6 Deep Paper Re-Read (Interactive Comprehension) ✅
**Goal:** Thorough understanding of the ProtoMF paper — section-by-section guided re-read with targeted questions on mechanisms, design choices, and mathematics.

**Notes:** All Q&A documents in `Master/understanding/paper/`. Key thesis-relevant insights captured inline (feature-aware prototype interpretation gap in 5.2, s^user contribution equivalence concern in 5.2).

**Sections:** Introduction & Motivation ✅ → Related Work ✅ → Methodology 3.1-3.3 ✅ → Experiment Setup ✅ → Results 5.1-5.3 ✅ → Conclusion (self-read)

### 2.7 Deep Codebase Understanding (Interactive Learning) ✅
**Goal:** Bottom-up mental model of the codebase for confident extension.

**Learning path:** Data Layer → Model → Prototype Architecture (most depth) → Training Pipeline → Evaluation & Explanations → Integration Exercise

**Notes:** All session documents in `Master/understanding/codebase/` (01–10 numbered modules plus reference deep-dives). Key topics: data layer & sparse matrices, config & hardware tuning, orchestration, core model, training & evaluation, feature extractors (base + prototypes), double-tie mechanism, factory assembly, explanations pipeline, integration exercise. Additional paper understanding sessions in `Master/understanding/paper/`.

**Completion criteria:** Can describe any component's role, trace data flow end-to-end, and articulate where/how to extend for Phases 3–4.

---

## ━━━ MILESTONE: Preparation Phase Complete ✅ ━━━
**Gate:** All of the following are done:
- [x] 2.6 Deep paper understanding complete
- [x] 2.7 Deep codebase understanding complete
- [x] 1.3 GPU environment working (CUDA PyTorch + test training run) — validated 2026-03-18
- [x] 2.3 Replication complete — all 10/10 experiments done (results + hyperparameters in [`replication_report.md`](replication_report.md)). Caveat: 4/10 run dirs local (all LEO5 had); remaining 6 on the GPU machine, retrieval deferred.
- [x] 2.5 Replication explanations generated — ml-1m `user_item_proto`, outputs in `Master/experiments/replication/explanations/ml-1m/`

**Meaning:** Foundation is solid — paper understood, code understood, environment ready, replication well underway. New work (H&M integration) has begun in parallel.

---

## Phase 3: H&M Dataset Integration

**Goal:** H&M dataset in the pipeline; CF-only baselines running.

### 3.1 Download and Explore H&M Dataset ✅
Raw Kaggle files in `data/hm/raw/` (gitignored). Exploration documented in `Master/temp/hm_data_analysis.md`.

### 3.2 Study Kaggle Competition Solutions ✅
Done — 10 per-source write-ups in `Master/docs/kaggle_sources/`, synthesized into [`Master/docs/hm_dataset_notes.md`](hm_dataset_notes.md) (approach taxonomy, feature consensus, candidate-generation playbook, pitfalls, metric landscape, citable quotes). Notable: **no source reports HR@K/NDCG@K** (MAP@12 only), and **no prototype/interpretable architectures appear anywhere** in the competition landscape — clean thesis gap. Original scope was:
- **Approaches:** top solution architectures, feature engineering strategies, candidate generation methods
- **Common issues & pitfalls:** data leakage, cold-start items, seasonal effects, dedup strategies
- **Insights:** which features matter most, temporal patterns, user segmentation findings
- **Reported metrics:** MAP@12 (competition), but especially any HR@K or NDCG@K results that overlap with our evaluation metrics (HR@{1,3,5,10,50}, NDCG@{1,3,5,10,50}) — these are critical for contextualizing our results

### 3.3 Design H&M Preprocessing ✅
Two dataset variants created with `data/hm/hm_splitter.py`:
- `hm_full` — all data, 5-core filtering → 889K users, 91K items, 26.2M interactions
- `hm_3_month` — 3-month window, 5-core filtering → 257K users, 27K items, 2.9M interactions
- Both include `item_features.csv` (9 categorical columns from `articles.csv`) for Phase 4

### 3.4 Write `data/hm/hm_splitter.py` ✅
CLI with `--months` and `--core` flags. Dedup, k-core filtering, ID remapping, leave-1-out split, item feature extraction.

### 3.5 Register H&M in Pipeline ✅
`hm_full` and `hm_3_month` added to `start.py` dataset choices.

### 3.5.1 Add MAP@12 Metric
Implement MAP@12 in `utilities/eval.py` alongside existing HR@K and NDCG@K. Enables direct comparison with Kaggle competition results.

### 3.5.2 Lightweight Contextual MAP@12 + Baselines
**Scope deliberately reduced** (2026-06-07): a full dual-mode evaluation pipeline (temporal-split competition mode + paper mode, switchable) is **not** worth the engineering cost. ProtoMF is a single-stage CF scorer; the Kaggle leaderboard is dominated by two-stage retrieve→rerank pipelines with repeat-purchase heuristics and heavy feature engineering, so any MAP@12 comparison is structurally apples-to-oranges and **purely contextual, not a benchmark we compete on**. HR@10 / NDCG@10 leave-one-out (paper protocol) stays the **primary** eval.

Minimal implementation instead:
- A lightweight temporal-week MAP@12 evaluation (hold out the last purchase week as ground truth, predict top-12, score over eligible users) — for *positioning only*, clearly labelled as non-comparable to the leaderboard.
- Cheap reference baselines so the MAP@12 number is interpretable: **popularity** and **repeat-purchase** (last-week-bought minus reorders). Without these, a bare MAP@12 means nothing to a reader.
- No switchable dual-mode rework, no production hyperopt under competition mode.

### 3.6 Run CF-Only Baselines on H&M ✅ (dev profile)
All 5 models run on **`hm_1_month`** on LEO5 (jobs 6610328–6610333, dev profile: 30 samples / 60 epochs, ~3–4 h/model). Test results, best hyperparameters, and hardware reference recorded in [`replication_report.md`](replication_report.md) (canonical — the planned `hm_baseline/results_table.csv` was not created). Best: `user_item_proto` HR@10 0.6606 / NDCG@10 0.4174. Open: decide whether prod-profile (100/100) re-runs are needed before these become headline Phase 3 numbers.

### 3.7 Compare to Competition Context
Document metric differences and run a popularity baseline for internal comparison.

---

## Phase 4: Feature-Aware Extensions

**Goal:** Incorporate item features into the prototype-based recommendation framework; compare approaches; generate fashion-specific explanations.

> **⟳ How this phase actually unfolded (2026-07-12):** 4.0–4.3 ran roughly as planned and produced **four design candidates** (dc01–dc04) via the Design Candidate Protocol (`/design-candidate`), not a single winner-take-all bake-off. 4.4's "implement & compare all proposals" then gave way to the **Scrutiny Protocol** — a deliberate audit-and-repair of the eval foundation + every candidate before advancing — and to the **committed C1 lineage** as the thesis spine. Read the **Current Position & Direction** banner at the top of this document for the live state; the subsections below remain as the original plan-of-record.

### 4.0 H&M Feature Analysis
Deep analysis of the 9 item feature columns in `item_features.csv`: distribution, cardinality, co-occurrence patterns, feature importance signals. Identify which features carry the most discriminative power for recommendations. Feeds directly into architecture decisions in 4.3. Informed by Kaggle findings from 3.2 (which features top solutions found most useful).

### 4.1 Literature Review — Feature Inclusion in CF
Research existing techniques for incorporating side information / item features into collaborative filtering and prototype-based models. Document findings in `Master/docs/feature_inclusion_literature.md`.

### 4.2 Requirements Document
Define clear requirements and evaluation criteria for any proposed feature-aware architecture. What must a valid solution satisfy? (e.g., compatibility with prototype explanations, no degradation of CF signal, scalability to H&M size, etc.)

### 4.3 Brainstorm & Propose Solutions
Based on 4.1 and 4.2, propose multiple candidate architectures. Document each with rationale, expected trade-offs, and implementation complexity.

### 4.4 Implement & Compare All Proposals
Every proposed solution gets implemented and tested on H&M. Head-to-head comparison on the same splits with the same hyperopt budget. Results → `Master/experiments/hm_features/results_table.csv`

### 4.5 Fashion-Specific Explanations
Extend `explanations_utils.py` for the winning approach(es). Produce explanation visualizations (TSNE, prototype interpretation, case studies, CF-only vs. feature-aware comparison).
- Dimension-reduction algorithm comparison: our prototype-space plots currently use t-SNE, but the DR method chooses which distance/neighborhood structure it preserves (Rudin et al. 2022, GC#7). Worth exploring whether UMAP, PaCMAP, or PCA gives more faithful / more readable prototype-space explanations — DR choice affects how trustworthy the visual explanation is, not just its looks. PaCMAP is the natural first pick (Wang et al. 2020b, same Rudin-group lineage, preserves local + global structure simultaneously); t-SNE notably distorts global geometry and inter-cluster distance, so avoid it for any figure making a *visual* claim about prototype-to-prototype distances. Scale note: the speed variants (BH t-SNE O(n log n), FIt-SNE O(n)) only matter if we project many points (e.g. all H&M items); for plots of prototypes + top-k representative items (small n) it is purely about structure preservation.
- **Methodological guardrail (applies regardless of DR choice):** the 2D projection is *illustrative only* — Rudin's own caption notes distances are not quantified on an abstract 2D space. All quantitative explanation claims (which items represent a prototype, which prototype a user is closest to) must be computed in the **original embedding space** via cosine similarity / lift (as the naming layer already does), never read off the projection. This means the DR algorithm affects readability/persuasiveness, not correctness — which de-risks the choice.

### 4.6 Ablation Studies
Results → `Master/experiments/hm_features/ablations/`
- Feature ablation, prototype count sweeps, regularization variants
- Temporal window: `hm_3_month` vs `hm_full`
- Regularizer-weight × explainability study: `sim_proto_weight`/`sim_batch_weight` are pure interpretability penalties but currently tuned against `hit_ratio@10` — sweep them, measure prototype explainability (coherence/nameability, dead-prototype share) alongside accuracy; consider selecting them on the explainability metric instead (see `notes/ideas.md` §5). Theoretical grounding: selecting on explainability subject to an accuracy tolerance ε = searching the Rashomon set for its most interpretable member (Rudin et al. 2022, GC#9)

---

## Phase 5: Thesis Writing

**Strategy:** Write progressively — don't wait until Phase 4 is done.

> **⟳ Approach update (2026-07-12):** first-generation text begins **soon and fragment-first** — small pieces (arguments, mechanism write-ups, figure captions) generated under user direction, then progressively glued into coherent chapters. Much of the raw material already exists as protocol-vault entries and scrutiny dossiers, so early writing is largely *consolidation* of captured reasoning, not first-draft-from-scratch. **Part one follows the committed C1 lineage** (see Current Position banner: honest-C1 → taste-argument → route-1 → route-2 → double-tie merge → candidate-prime → speculative extensions), with the **"Interpretability Quantified"** section placed strictly after it. Literature reading runs in parallel throughout and keeps feeding Related Work + Future Work.

### 5.1 LaTeX Setup (start of Phase 2)
Obtain template, set up `Master/thesis/` structure, verify compilation.

### 5.2 Progressive Writing Schedule

| During Phase | Write |
|---|---|
| Phase 2 | Ch. 3 Background (ProtoMF, MF, prototypes) + start Ch. 2 Related Work |
| Phase 3 | Ch. 4 Dataset (H&M, preprocessing, statistics) + update Ch. 2 with H&M literature |
| Phase 4 early | Ch. 5 Method (feature-aware architecture) |
| Phase 4 mid | Ch. 6 Experiments — replication section |
| Phase 4 late | Ch. 7 Explanations + complete Ch. 6 H&M results |
| Final weeks | Ch. 1 Introduction + Ch. 8 Conclusion + Abstract + polish |

### 5.3 Key Figures to Produce
Architecture diagrams, dataset statistics, TSNE plots, explanation bar charts, feature-prototype heatmaps, result tables. All sources in `Master/thesis/figures/`.

