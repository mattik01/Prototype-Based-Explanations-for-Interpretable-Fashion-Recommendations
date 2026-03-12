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
- Environment: Keep old versions (PyTorch 1.9.1) initially, but CUDA 10.2 will almost certainly be incompatible with the available RTX 30xx/40xx GPU — upgrade is planned at Phase 1.3
- Thesis: `Master/thesis/` (English, template TBD from supervisor)
- Replication: Full (all 5 models × 3 datasets), pivot only if compute-infeasible
- W&B: Keep it, configure at start of Phase 2 (not Phase 1)
- Temp files: Always go to `Master/temp/` per CLAUDE.md rule

---

## Proposed Folder Structure

```
ProtoMF/
├── CLAUDE.md                          # Updated with all project conventions
├── README.md, LICENSE, protomf.yml
├── start.py, experiment_helper.py
├── confs/hyper_params.py
├── data/
│   ├── amazon2014/, lfm2b-1mon/, ml-1m/   # Paper datasets
│   └── hm/                                 # Phase 3 — H&M dataset
│       ├── hm_splitter.py
│       ├── raw/                             # Raw Kaggle download (gitignored)
│       └── (generated split files)
├── feature_extraction/
├── rec_sys/
├── utilities/
├── pdfs_and_images/
└── Master/
    ├── Protomf-paper.pdf
    ├── docs/                           # Thesis notes and design documents
    ├── experiments/                    # All experiment results
    ├── notebooks/                      # Jupyter notebooks for analysis
    ├── literature/
    ├── scripts/                        # Standalone utility scripts
    ├── sensitive/                      # Gitignored — API keys, network config, SSH details
    ├── thesis/                         # Phase 5 — LaTeX thesis
    └── temp/                           # Temporary files (per CLAUDE.md rule)
```

---

## Phase 1: Setup and Tooling ✅ (mostly complete)

**Goal:** Working development environment, project conventions locked in, remote GPU accessible.

### 1.1 Folder Structure ✅
Created all directories, added `.gitkeep` for empty dirs, `.gitignore` in `data/hm/raw/`.

### 1.2 CLAUDE.md ✅
Established branch strategy, naming conventions, experiment naming, `ft_type` reservations, results storage paths, modification map, H&M data conventions.

### 1.3 Remote GPU Machine Setup (partially complete)

**Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER — 16 GB VRAM, CUDA 12.7. University HPC cluster available as fallback possibly (NEEDS TO BE REQUESTED).

**⚠️ Known issue:** PyTorch 1.9.1 ships with CUDA 10.2 — incompatible with RTX 40xx. Upgrade required.

**Completed:**
- SSH access via Tailscale (`ssh gpu`) with key-based auth
- GPU confirmed: RTX 4070 Ti SUPER, CUDA 12.7
- Wake-on-LAN fully working from anywhere — phone relay (Termux + Tailscale) solves router ARP limitation
- Scripts with ping+SSH verification: `Master/sensitive/wake_gpu.sh`, `Master/sensitive/shutdown_gpu.sh`
- Claude Code skills: `/gpupc-start`, `/gpupc-stop`
- Reference docs: `Master/sensitive/gpu_machine.md`, `Master/sensitive/wol_relay.md`, `Master/sensitive/laptop.md`
- End-to-end tested: wake + SSH + shutdown from both home and external networks (2026-03-11)
- Repo cloned on remote machine

**Remaining:**
1. Upgrade PyTorch/Ray/wandb for CUDA ≥ 11 (existing `protomf.yml` will fail)
2. Create conda environment on remote, verify GPU in PyTorch
3. Test trivial training run remotely

**Workflow:** Direct VSCode SSH session optionally with Claude Code running locally on the GPU machine — no tmux/nohup job management needed.

**HPC cluster (if/when needed):**
- Request access, set up SLURM job scripts, test with small job

**Verification:** GPU recognized by PyTorch, a training run starts without errors.

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

### 2.3 Full Replication
Run all 5 models × 2 available datasets = **10 experiments** (single seed first, then multi-seed if compute allows):

| | ml-1m | amazon2014 |
|---|---|---|
| `mf` | ✓ | ✓ |
| `acf` | ✓ | ✓ |
| `user_proto` | ✓ | ✓ |
| `item_proto` | ✓ | ✓ |
| `user_item_proto` | ✓ | ✓ |

Start with `mf` on `ml-1m` to validate the pipeline. Compare to paper's Table 1. If within ~3%, proceed.

Results → `Master/experiments/replication/results_table.csv`

### 2.4 Modification Map ✅
Created `Master/docs/modification_map.md` documenting code extension points for adding datasets (Phase 3), item features (Phase 4), and explanation extraction (Phase 4).

### 2.5 Generate Replication Explanations
- Load a trained `user_item_proto` checkpoint on ml-1m
- Run all existing `explanations_utils.py` functions: TSNE, top-K items per prototype, recommendation explanations
- Save outputs to `Master/experiments/replication/explanations/`
- Document workflow in `Master/notebooks/01_replication_explanations.ipynb`

### 2.6 Deep Paper Re-Read (Interactive Comprehension) — IN PROGRESS
**Goal:** Thorough understanding of the ProtoMF paper — section-by-section guided re-read with targeted questions on mechanisms, design choices, and mathematics.

**Progress:** Started in `Master/understanding/paper/` — completed Introduction (01) and Related Work (02).

**Sections:** Introduction & Motivation ✅ → Related Work ✅ → ProtoMF Architecture (most depth) → Regularization → Training & Loss Functions → Experiments & Results → Explanations & Qualitative Analysis

**Completion criteria:** Can explain every component from memory, justify design choices, and identify what to keep/modify/extend in Phase 4.

### 2.7 Deep Codebase Understanding (Interactive Learning)
**Goal:** Bottom-up mental model of the codebase for confident extension.

**Learning path:** Data Layer → Model → Prototype Architecture (most depth) → Training Pipeline → Evaluation & Explanations → Integration Exercise

**Completion criteria:** Can describe any component's role, trace data flow end-to-end, and articulate where/how to extend for Phases 3–4.

---

## ━━━ MILESTONE: Preparation Phase Complete ━━━
**Gate:** All of the following are done:
- [ ] 2.6 Deep paper understanding complete
- [ ] 2.7 Deep codebase understanding complete
- [ ] 1.3 GPU environment working (CUDA PyTorch + test training run)
- [ ] 2.3 Full replication (10 experiments)
- [ ] 2.5 Replication explanations generated

**Meaning:** Foundation is solid — paper understood, code understood, environment ready, results replicated. Everything from here is new work.

---

## Phase 3: H&M Dataset Integration

**Goal:** H&M dataset in the pipeline; CF-only baselines running.

### 3.1 Download and Explore H&M Dataset
- Download from Kaggle: transactions, articles, customers, (optionally images)
- Place raw files in `data/hm/raw/` (gitignored)
- Explore in `Master/notebooks/02_hm_exploration.ipynb`

### 3.2 Study Kaggle Competition Solutions
- Read top 3–5 public solutions, document insights in `Master/literature/notes/hm_kaggle_solutions.md`
- Note: competition metric was MAP@12; our pipeline uses HR@10/NDCG@10

### 3.3 Design H&M Preprocessing
Document design in `Master/docs/hm_preprocessing_design.md`:
- Implicit binary interactions, deduplication, temporal window (default 6 months), 10-core filtering
- Temporal leave-1-out split, output format matching `ProtoRecDataset`
- Extra output: `item_features.csv` for Phase 4

### 3.4 Write `data/hm/hm_splitter.py`
Follow `movielens_splitter.py` structure. Includes feature extraction from `articles.csv`.

### 3.5 Register H&M in Pipeline
Add `'hm'` to `start.py` dataset choices and ASHA scheduler branch in `experiment_helper.py`.

### 3.6 Run CF-Only Baselines on H&M
Run all 5 existing models on H&M. Results → `Master/experiments/hm_baseline/results_table.csv`

### 3.7 Compare to Competition Context
Document metric differences and run a popularity baseline for internal comparison.

---

## Phase 4: Feature-Aware Prototype Architecture

**Goal:** Extend ProtoMF to use item metadata; experiments and explanations on H&M.

### 4.1 Design Feature-Aware Architecture
Document in `Master/docs/feature_architecture_design.md`.

> **Architecture decision:** Aim for Option C (Dual Prototype Spaces) as primary contribution. Final decision confirmed once codebase is well understood.

**Option A — Feature-Only Embedding (ablation baseline):**
Replace item embedding with feature encoder MLP. Purely content-based prototypes.

**Option B — Hybrid CF + Features (intermediate step):**
Concatenate CF embedding + feature encoding before cosine similarity with prototypes.

**Option C — Dual Prototype Spaces (primary contribution):**
- Separate CF prototypes + feature prototypes in their own spaces
- `item_repr = concat(sim_cf_protos, sim_feat_protos)` — two explanation streams
- "Recommended because of interaction patterns (CF) AND because of attributes (features)"

**H&M feature encoding:**
- Categorical features → learned embeddings (dim 8–16) → concatenate → MLP → `feature_embedding`
- Feature lookup table stored as non-learnable tensor in extractor (no dataloader changes)

### 4.2 Data Pipeline Changes
Feature lookup table inside new feature extractor. Zero changes to dataloader/trainer/tester.

### 4.3 Implement New Feature Extractors
`FeatureEncoder` and `FeatureAwarePrototypeEmbedding` in `feature_extractors.py`. New `ft_type` branches in factory.

### 4.4 Experiments
| Model | H&M CF-only (Phase 3) | H&M Feature-Aware | Delta |
|---|---|---|---|
| `mf` | baseline | N/A | — |
| `item_proto` | ✓ | `feature_item_proto` | compare |
| `user_proto` | ✓ | (if applicable) | compare |
| `user_item_proto` | ✓ | `feature_user_item_proto` | compare |

Results → `Master/experiments/hm_features/results_table.csv`

### 4.5 Fashion-Specific Explanations
Extend `explanations_utils.py` with feature-prototype profiling, recommendation explanations, and representative item visualization.

Produce `Master/notebooks/03_hm_explanations.ipynb` with TSNE plots, prototype interpretation, case studies, CF-only vs. feature-aware comparison.

### 4.6 Ablation Studies
Results → `Master/experiments/hm_features/ablations/`
1. Feature ablation: remove one feature at a time
2. n_prototypes sweep: {10, 20, 50, 100}
3. Feature embedding dim sweep: {16, 32, 64}
4. Regularization type: max vs. soft on H&M
5. Temporal window: {3 months, 6 months, full dataset}

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

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| CUDA 10.2 incompatible with RTX 4070 Ti SUPER (needs CUDA ≥ 11.1) | Upgrade PyTorch at Phase 1.3; document Ray Tune API changes |
| Ray Tune 2.x API breaks are too painful | Switch to Optuna (simpler, well-documented, popular) |
| H&M too large for 100-sample hyperopt | ASHA scheduler + reduce NUM_SAMPLES to 50; use shorter temporal window |
| Feature-aware model doesn't improve accuracy | Still valid thesis: quantify trade-off, focus on explanation quality gains |
| Replication doesn't match paper numbers | Document differences (hardware, library versions); partial replication is acceptable |
| Cluster access needed | HPC cluster already identified as last resort; SLURM job scripts to be prepared |

---

## Critical Files Reference

| File | Role |
|---|---|
| `feature_extraction/feature_extractors.py` | Add `FeatureEncoder`, `FeatureAwarePrototypeEmbedding` in Phase 4 |
| `feature_extraction/feature_extractor_factories.py` | Add new `ft_type` branches in Phase 4 |
| `rec_sys/protomf_dataset.py` | Understand output format; no changes needed if feature lookup is inside extractor |
| `rec_sys/trainer.py` | Pass `data_path` to model builder in Phase 4 |
| `utilities/explanations_utils.py` | Extend with fashion-specific functions in Phase 4 |
| `data/ml-1m/movielens_splitter.py` | Template for `hm_splitter.py` |
| `utilities/consts.py` | `DATA_PATH` (auto-detected) and `WANDB_API_KEY` (from `Master/sensitive/`) |
| `confs/hyper_params.py` | Add debug config (Phase 1) + new model configs (Phase 4) |
| `Master/docs/modification_map.md` | Living document of all code extension points (Phase 2) |
