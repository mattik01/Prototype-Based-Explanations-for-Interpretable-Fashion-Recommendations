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
- Environment: Upgraded from PyTorch 1.9.1 / CUDA 10.2 to CUDA 12.7 compatible stack — RTX 4070 Ti SUPER fully working
- Thesis: `Master/thesis/` (English, template TBD from supervisor)
- Replication: Full (all 5 models × 2 datasets — LFM-2b unavailable), pivot only if compute-infeasible
- W&B: Keep it, configure at start of Phase 2 (not Phase 1)
- H&M splits: Two variants — `hm_full` (all data, 5-core) and `hm_3_month` (3-month window, 5-core). Raw data shared in `data/hm/raw/`
- Temp files: Always go to `Master/temp/` per CLAUDE.md rule

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
│   ├── hm_full/                             # Full H&M (~24 mo, 5-core, 26.2M interactions)
│   └── hm_3_month/                          # 3-month H&M window (5-core, 2.9M interactions)
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

### 2.3 Full Replication (in progress)
10 experiments (5 models × 2 datasets), single seed. Detailed results, GPU benchmarks, and best hyperparameters tracked in [`Master/docs/replication_report.md`](replication_report.md).

Results → `Master/experiments/replication/results_table.csv`

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
- [x] 2.3 Replication in progress — 6/10 experiments complete, remaining 4 (item_proto, user_item_proto) running. See [`replication_report.md`](replication_report.md)
- [x] 2.5 Replication explanations generated — ml-1m `user_item_proto`, outputs in `Master/experiments/replication/explanations/ml-1m/`

**Meaning:** Foundation is solid — paper understood, code understood, environment ready, replication well underway. New work (H&M integration) has begun in parallel.

---

## Phase 3: H&M Dataset Integration

**Goal:** H&M dataset in the pipeline; CF-only baselines running.

### 3.1 Download and Explore H&M Dataset ✅
Raw Kaggle files in `data/hm/raw/` (gitignored). Exploration documented in `Master/temp/hm_data_analysis.md`.

### 3.2 Study Kaggle Competition Solutions
Extract everything useful from Kaggle discussions, submitted solutions, and public notebooks into [`Master/docs/hm_dataset_notes.md`](hm_dataset_notes.md):
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

### 3.5.2 Dual Evaluation Mode (Competition vs. Paper)
- Rework the evaluation pipeline to support switching between **competition mode** (MAP@12 as primary metric) and **paper mode** (HR@10 as primary metric / model selection criterion)
- Adjust splitters to serve both modes: competition mode needs a temporal split matching Kaggle's setup (last week of purchases as ground truth, predict next 12 items), while paper mode uses the existing leave-one-out protocol
- Pipeline should allow selecting the mode via a flag so experiments can be run and compared under both evaluation protocols

### 3.6 Run CF-Only Baselines on H&M
Run all 5 existing models on H&M. Results → `Master/experiments/hm_baseline/results_table.csv`

### 3.7 Compare to Competition Context
Document metric differences and run a popularity baseline for internal comparison.

---

## Phase 4: Feature-Aware Extensions

**Goal:** Incorporate item features into the prototype-based recommendation framework; compare approaches; generate fashion-specific explanations.

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

### 4.6 Ablation Studies
Results → `Master/experiments/hm_features/ablations/`
- Feature ablation, prototype count sweeps, regularization variants
- Temporal window: `hm_3_month` vs `hm_full`

---

## Phase 5: Thesis Writing

**Strategy:** Write progressively — don't wait until Phase 4 is done.

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

