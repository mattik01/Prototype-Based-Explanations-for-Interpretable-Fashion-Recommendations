# Master Plan: Prototype-Based Explanations for Interpretable Fashion Recommendations
## Master's Thesis — ProtoMF + H&M Dataset

---

## Context

This plan covers the full lifecycle of a Master's thesis extending the ProtoMF (RecSys 2022) codebase to:
1. Reproduce the original paper's results across all datasets and models
2. Integrate the H&M Fashion dataset (Kaggle, ~31M transactions, ~105K articles with rich metadata)
3. Extend the prototype architecture to leverage item features (not just collaborative filtering)
4. Extract and analyze fashion-specific prototype-based explanations
5. Write the thesis in LaTeX within this workspace

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
    │   ├── Masters Thesis.md
    │   ├── ProtoMF Allesandro.md
    │   ├── Summary Full.md
    │   ├── modification_map.md         # Created in Phase 2 — code insertion points
    │   ├── hm_preprocessing_design.md  # Created in Phase 3
    │   └── feature_architecture_design.md  # Created in Phase 4
    ├── experiments/                    # All experiment results
    │   ├── replication/                # Phase 2 results
    │   │   ├── dataset_stats.md
    │   │   ├── results_table.csv
    │   │   └── explanations/
    │   ├── hm_baseline/               # Phase 3 CF-only H&M results
    │   │   ├── results_table.csv
    │   │   └── explanations/
    │   └── hm_features/              # Phase 4 feature-aware results
    │       ├── results_table.csv
    │       └── ablations/
    ├── notebooks/                      # Jupyter notebooks for analysis
    │   ├── 01_replication_explanations.ipynb
    │   ├── 02_hm_exploration.ipynb
    │   └── 03_hm_explanations.ipynb
    ├── literature/
    │   ├── papers/                    # PDFs of referenced papers
    │   └── notes/
    │       └── hm_kaggle_solutions.md
    ├── scripts/                       # Standalone utility scripts
    ├── thesis/                        # Phase 5 — LaTeX thesis
    │   ├── main.tex
    │   ├── bibliography.bib
    │   ├── chapters/
    │   │   ├── 01_introduction.tex
    │   │   ├── 02_related_work.tex
    │   │   ├── 03_background.tex
    │   │   ├── 04_dataset.tex
    │   │   ├── 05_method.tex
    │   │   ├── 06_experiments.tex
    │   │   ├── 07_explanations.tex
    │   │   └── 08_conclusion.tex
    │   └── figures/
    └── temp/                          # Temporary files (per CLAUDE.md rule)
```

---

## Phase 1: Setup and Tooling

**Goal:** Working development environment, project conventions locked in, remote GPU accessible.

### 1.1 Folder Structure Creation ✅
- ~~Create all directories from the structure above~~
- ~~Add `.gitkeep` where needed to track empty dirs~~
- Also added `.gitignore` in `data/hm/raw/` to exclude large Kaggle files

### 1.2 CLAUDE.md Update ✅
Added sections covering:
- ~~**Branch strategy:** `dev` for all work, `main` stays clean as reference~~
- ~~**Naming conventions:** snake_case files, `hm_` prefix for H&M-specific modules, `feature_` prefix for feature-aware model variants~~
- ~~**Experiment naming:** `{model}_{dataset}_{variant}_{seed}`~~
- ~~**New `ft_type` values to reserve:** `feature_item_proto` (Option B), `dual_item_proto` (Option C primary) (Phase 4 — finalize names once codebase is well understood)~~
- ~~**Results storage:** All experiment results as CSVs in `Master/experiments/` subfolders~~
- ~~**Modification map placeholder:** Link to `Master/docs/modification_map.md` (completed Phase 2)~~
- **Claude Code skills/plugins discovered:** Document useful slash commands and MCP servers found during setup *(deferred to 1.5)*
- ~~**H&M data note:** Raw files in `data/hm/raw/` are gitignored; describe expected structure~~

### 1.3 Remote GPU Machine Setup ⚠️ CRITICAL

**Hardware situation:**
- **Primary compute:** NVIDIA GeForce RTX 4070 Ti SUPER — 16 GB VRAM, CUDA 12.7, Driver 566.36. Confirmed 2026-03-09 via `nvidia-smi`. This will be the workhorse for all training and hyperparameter search.
- **Fallback / scale-up:** University HPC cluster access can be requested if the single GPU proves insufficient (e.g., for large-scale hyperopt on H&M or multi-seed replication). Cluster likely has much stronger hardware (multi-GPU nodes). This is a last resort — requires setup overhead (SLURM job scripts, queue times, etc.).
- **Local machine:** CPU-only development and debugging (see 1.4).

**⚠️ Known issue:** PyTorch 1.9.1 ships with CUDA 10.2. Modern consumer GPUs (RTX 30xx/40xx) require CUDA ≥ 11.1. The old environment **will not work**. Plan accordingly.

**Steps for primary remote GPU:**
1. ~~Set up SSH access (configure `~/.ssh/config` with host alias + key-based auth)~~ ✅ `ssh gpu` works
2. Clone repo on remote machine
3. ~~**Test CUDA compatibility first:** Run `nvidia-smi` to confirm GPU model, VRAM, and driver version~~ ✅ RTX 4070 Ti SUPER, CUDA 12.7
4. **If CUDA ≥ 11 (almost certain):** The existing `protomf.yml` will fail. Upgrade path:
   - PyTorch ≥ 1.12 with CUDA 11.x (e.g., PyTorch 1.13 or 2.0)
   - Ray Tune: upgrade to ≥ 2.0 (significant API changes — document them)
   - wandb: upgrade freely
   - Record all version changes in `Master/temp/env_upgrade_notes.md`
5. Create conda environment on remote, verify GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
6. Set up remote job workflow: `tmux`/`screen` + `nohup` for persistent runs
7. Test trivial training run remotely to confirm pipeline starts
8. Document the full remote setup procedure in `Master/docs/remote_setup.md`
9. **Wake-on-LAN from outside network:** ✅ Fully working.
   - BIOS/Windows WoL configured ✅
   - Router: UDP port 9 forwarded to `192.168.68.58`, DHCP reservation for MAC `D8:43:AE:B3:D7:3D` ✅
   - **WoL relay:** Samsung Galaxy S10 running Termux (always-on, plugged in on home WiFi) acts as LAN relay — needed because TP-Link Deco has no static ARP and drops WoL packets when PC is off
   - **Remote access to relay:** Tailscale on the S10 (IP: `100.71.183.60`) + SSH key auth via Termux sshd on port 8022
   - **One-command wake from anywhere:** `ssh termux-tailscale "python ~/wol_gpu.py"`
   - Full setup documented in `Master/temp/termux_ssh_setup.md`
   - **TODO:** Copy laptop (`mattik01`) SSH public key to phone's `~/.ssh/authorized_keys` so laptop can also trigger WoL directly
10. **WoL end-to-end verification from laptop (next step):**
    - On laptop (`mattik01`): copy SSH public key to phone → `ssh-copy-id -p 8022 u0_a317@100.71.183.60`
    - Add SSH config entries for `termux`, `termux-wifi`, `termux-tailscale` on laptop (see `Master/temp/termux_ssh_setup.md`)
    - **Test A (same network):** From laptop on home WiFi, run `ssh termux-tailscale "python ~/wol_gpu.py"` → verify WoL received on GPU machine listener
    - **Test B (external network):** Move laptop to a different network (e.g., phone hotspot), run same command → verify WoL still arrives
    - GPU machine will be running `Master/temp/wol_listener.py` on port 9 to confirm packet arrival
    - Once both tests pass, WoL setup is fully verified ✅

> **⚠️ Open issue — SSH unavailable after boot without quick login:**
> After powering on the remote PC, SSH stops working if no user logs in promptly. Likely cause: Windows sleep settings are per-user — when no session is active (e.g., sitting at the lock screen), the system falls back to default power plan which may include sleep. Sleep is disabled on the admin user, but that only takes effect once logged in.
>
> **TODO (deferred):**
> - Run `powercfg /change standby-timeout-ac 0` in an elevated PowerShell to set "never sleep" at the system/power-plan level regardless of login state
> - Verify OpenSSH server start type: `Get-Service sshd | Select-Object StartType` — should be `Automatic`, not `Automatic (Delayed Start)`

**Steps for HPC cluster (if/when needed):**
1. Request access, get credentials
2. Understand job scheduler (likely SLURM)
3. Prepare job submission scripts (`sbatch`) — template in `Master/scripts/`
4. Adapt environment setup for cluster module system (may differ from conda)
5. Test with a small job before submitting large runs

**Verification:** SSH works, GPU recognized by PyTorch, a training run starts without errors.

### 1.4 Local CPU Fallback ✅
- ~~Create local conda environment (CPU-only — no CUDA packages)~~
- ~~Verify the `device='cpu'` path works (code already handles this via `torch.cuda.is_available()`)~~
- ~~Add a `debug_config` to `confs/hyper_params.py` with tiny values (`embedding_dim=8`, `n_prototypes=3`, `num_samples=1`) for fast local iteration without GPU~~
- Accept that CUDA-specific code paths (e.g., heavy batch operations) may require the GPU machine

### 1.5 Claude Code Environment ✅
- ~~Discover available MCP servers and slash commands useful for this project~~
- ~~Candidates to investigate: file search, web fetch for papers, code execution tools~~
- ~~Document useful ones in CLAUDE.md~~
- Installed: context7 (docs), pyright-lsp (types), code-simplifier, claude-md-management, claude-code-setup
- Custom commands: `/protocol`, `/commit`, `/gitcheck`
- GitHub token configured for GitHub MCP

---

## Phase 2: Replication and Deep Understanding

**Goal:** Reproduce all paper results; fully understand the codebase; map all code extension points.

### 2.1 Dataset Download and Preprocessing ✅ (2/3 datasets)
Download and preprocess all three paper datasets:
- **MovieLens-1M:** `ratings.dat` → `data/ml-1m/` → run `movielens_splitter.py` ✅
- **Amazon2014 Video Games:** `ratings_Video_Games.csv` → `data/amazon2014/` → run `amazon2014_splitter.py` ✅
- **LFM-2b 1-Month:** `users.tsv` + `listening_events.tsv` → `data/lfm2b-1mon/` → run `lfm2b-2020_splitter.py` ⚠️ **SKIPPED** — dataset removed from official source (license issues, cp.jku.at). Replication proceeds with ML-1M and Amazon2014 only. Can revisit if authors share the data.

Each produces 5 files: `listening_history_{train,val,test}.csv`, `user_ids.csv`, `item_ids.csv`

Record dataset statistics (n_users, n_items, n_interactions) in `Master/experiments/replication/dataset_stats.md`.

### 2.2 W&B and DATA_PATH Configuration ✅
- ~~Update `utilities/consts.py`: set `DATA_PATH` and `WANDB_API_KEY`~~
- W&B key stored externally in `api_keys/wandb_key.txt` (gitignored via `api_keys/.gitignore`)
- ~~Create W&B account at wandb.ai if not done; if issues arise, evaluate local alternatives (TensorBoard/CSV)~~

### 2.3 Full Replication
Run all 5 models × 3 datasets = **15 experiments** (single seed first, then multi-seed `-mp` if compute allows):

| | ml-1m | amazon2014 | lfm2b-1mon |
|---|---|---|---|
| `mf` | ✓ | ✓ | ✓ |
| `acf` | ✓ | ✓ | ✓ |
| `user_proto` | ✓ | ✓ | ✓ |
| `item_proto` | ✓ | ✓ | ✓ |
| `user_item_proto` | ✓ | ✓ | ✓ |

Start with `mf` on `ml-1m` to validate the pipeline. Compare to paper's Table 1. If within ~3%, proceed.

**Compute note:** lfm2b-1mon is the largest dataset (ASHA scheduler already in code). Estimate 1–3 GPU-days for single-seed, 3–9 for multi-seed. Pivot strategy: reduce `NUM_SAMPLES` to 50 or drop lfm2b-1mon if time-critical.

Results → `Master/experiments/replication/results_table.csv`

### 2.4 Modification Map — Code Extension Points ✅
Created `Master/docs/modification_map.md` documenting exactly where code needs to change for each thesis goal:

**A. Adding a new dataset (Phase 3):**
- `data/hm/hm_splitter.py` — new file, follow `movielens_splitter.py` pattern
- `start.py` line ~18 — add `'hm'` to dataset choices
- `experiment_helper.py` lines ~92–95 — add ASHA scheduler for H&M (same as lfm2b)

**B. Adding item features (Phase 4) — key files:**
- `feature_extraction/feature_extractors.py` — add `FeatureEncoder` and `FeatureAwarePrototypeEmbedding` classes
- `feature_extraction/feature_extractor_factories.py` — add new `ft_type` branches
- `rec_sys/trainer.py` — pass `data_path` or feature tensor to model builder
- `confs/hyper_params.py` — add new hyperparameter configs
- `start.py` — add new model CLI choices
- Recommended: feature lookup table lives inside the new feature extractor (zero changes to dataloader/trainer unpacking)

**C. Explanation extraction (Phase 4):**
- `utilities/explanations_utils.py` — extend with fashion-specific functions
- Key internal tensor: `sim_mtx` computed in `PrototypeEmbedding.forward()` — this is the explanation signal

### 2.5 Generate Replication Explanations
- Load a trained `user_item_proto` checkpoint on ml-1m
- Run all existing `explanations_utils.py` functions: TSNE, top-K items per prototype, recommendation explanations
- Save outputs to `Master/experiments/replication/explanations/`
- Document workflow in `Master/notebooks/01_replication_explanations.ipynb`

### 2.6 Deep Paper Re-Read (Interactive Comprehension)
**Goal:** Thorough understanding of the ProtoMF paper — not just the gist, but the precise mechanisms, design choices, and mathematical details. Critical foundation for extending the architecture in Phase 4.

**Format:** Section-by-section guided re-read of `Master/Protomf-paper.pdf`. For each section, Claude generates a block of targeted questions (mix of conceptual, mathematical, and "why did they do it this way?" questions). Questions are generated only when we reach that section — no peeking ahead.

**Paper sections to cover:**

1. **Introduction & Motivation**
   - Why prototypes? What gap do they fill vs. standard MF and existing explainable methods?
   - *Question block generated on arrival*

2. **Related Work**
   - Positioning vs. attention-based, post-hoc, and prototype-based explanations in other domains
   - *Question block generated on arrival*

3. **ProtoMF Architecture (core contribution)**
   - Prototype embedding, cosine similarity, shifted cosine, weight matrix
   - Single-side vs. dual-side variants (U-ProtoMF, I-ProtoMF, UI-ProtoMF)
   - **Most depth here** — every equation, every design choice
   - *Question block generated on arrival*

4. **Regularization**
   - Diversity (batch-level) and coverage (prototype-level) regularization
   - Max vs. soft (entropy) variants, inclusiveness constraint
   - Why these specific regularizations? What collapses do they prevent?
   - *Question block generated on arrival*

5. **Training & Loss Functions**
   - BCE, BPR, sampled softmax — when does each shine?
   - Negative sampling strategies
   - *Question block generated on arrival*

6. **Experiments & Results**
   - Datasets, baselines, metrics (HR@k, NDCG@k)
   - Key findings: which variant wins, by how much, on which datasets?
   - Ablation insights
   - *Question block generated on arrival*

7. **Explanations & Qualitative Analysis**
   - How prototypes produce explanations
   - TSNE visualizations, prototype interpretation
   - Strengths and limitations of the explanation approach
   - *Question block generated on arrival*

**Completion criteria:** You can explain every component of ProtoMF from memory, justify the design choices, and identify exactly which parts you'll keep, modify, or extend in Phase 4.

### 2.7 Deep Codebase Understanding (Interactive Learning)
**Goal:** Build complete mental model of the codebase — bottom-up, component by component — so that extending it for H&M and feature-aware prototypes feels natural rather than guesswork.

**Format:** Guided walkthrough with reading assignments, explanations, and comprehension checks. More depth on thesis-critical components (prototypes, feature extractors, factory pattern).

**Learning path (bottom-up):**

1. **Data Layer** — How data enters the system
   - `data/ml-1m/movielens_splitter.py` — preprocessing pattern (template for `hm_splitter.py`)
   - `rec_sys/protomf_dataset.py` — `ProtoRecDataset`, negative sampling, batch structure
   - `utilities/consts.py` — paths, constants, metrics
   - *Comprehension check: What does a single training batch look like? What would you change to add H&M?*

2. **Model Core** — The recommendation engine
   - `feature_extraction/feature_extractors.py` — `FeatureExtractor` base class, `Embedding`, `EmbeddingW`
   - `rec_sys/rec_sys.py` — `RecSys` module: how user/item representations become predictions
   - *Comprehension check: How does RecSys combine user and item extractors? Where does the loss come from?*

3. **Prototype Architecture** — The thesis-critical component (most depth here)
   - `feature_extraction/feature_extractors.py` — `PrototypeEmbedding` in detail: forward pass, similarity matrix, regularization losses
   - `feature_extraction/feature_extractors.py` — `ConcatenateFeatureExtractors` for dual prototypes
   - `feature_extraction/feature_extractor_factories.py` — `FeatureExtractorFactory`: how configs become model instances, weight tying
   - *Comprehension check: Walk through a forward pass of `user_item_proto`. Where does `sim_mtx` live? How do regularization losses flow? Where would you inject feature information?*

4. **Training Pipeline** — How experiments run end-to-end
   - `start.py` — CLI entry point, dataset/model dispatch
   - `experiment_helper.py` — Ray Tune setup, ASHA scheduler, wandb integration
   - `rec_sys/trainer.py` — training loop, early stopping, checkpointing
   - `confs/hyper_params.py` — search spaces per model
   - *Comprehension check: Trace a full run from `python start.py -m item_proto -d ml-1m`. What gets called in what order?*

5. **Evaluation & Explanations** — Measuring and interpreting results
   - `utilities/eval.py` — `Evaluator`, NDCG/HR computation
   - `rec_sys/tester.py` — test-time evaluation flow
   - `utilities/explanations_utils.py` — prototype interpretation, TSNE, explanation generation
   - *Comprehension check: How would you extract fashion-specific explanations from feature-aware prototypes?*

6. **Integration Exercise** — Putting it all together
   - *Given everything you now know: sketch the minimal set of changes to add H&M as a dataset and run CF-only baselines (Phase 3). Then sketch how you'd add feature-aware prototypes (Phase 4).*

**Completion criteria:** You can confidently describe any component's role, trace data flow end-to-end, and articulate where/how to extend the code for Phases 3–4.

---

## Phase 3: H&M Dataset Integration

**Goal:** H&M dataset in the pipeline; CF-only baselines running.

### 3.1 Download and Explore H&M Dataset
- Download from Kaggle competition: transactions, articles, customers, (optionally images)
- Place raw files in `data/hm/raw/` (gitignored)
- Explore in `Master/notebooks/02_hm_exploration.ipynb`:
  - Transaction volume over time, temporal patterns
  - Interactions per user/item distributions
  - Item feature cardinalities and missing values
  - Price distribution, seasonality

### 3.2 Study Kaggle Competition Solutions
- Read top 3–5 public solutions from competition discussion
- Note preprocessing approaches, modeling strategies, baselines
- Document insights in `Master/literature/notes/hm_kaggle_solutions.md`
- Note: competition metric was MAP@12; our pipeline uses HR@10/NDCG@10 — document difference

### 3.3 Design H&M Preprocessing
Key decisions to document in `Master/docs/hm_preprocessing_design.md`:
- **Interactions:** Implicit binary (purchase = 1), deduplicate user-item pairs (keep first timestamp)
- **Temporal window:** Default last 6 months (configurable — study competition solutions for guidance)
- **K-core:** 10-core (matching lfm2b scale), iterative until convergence
- **Split:** Temporal leave-1-out (last purchase = test, second-to-last = val, rest = train)
- **Output format:** Must match `ProtoRecDataset` expectations (5 standard CSV files)
- **Extra output:** `item_features.csv` (item_id → categorical feature vector) saved alongside, ready for Phase 4

### 3.4 Write `data/hm/hm_splitter.py`
Follow `movielens_splitter.py` structure exactly:
- argparse for paths
- Load `transactions_train.csv`
- Filter, deduplicate, k-core, split, index, save
- Additionally join `articles.csv` to save `item_features.csv`
- Add `data/hm/` entry to `data/README.md`

Verification checklist:
- [ ] IDs are contiguous integers starting at 0
- [ ] Every val/test user appears in train
- [ ] Every val/test item appears in train
- [ ] Every user has exactly 1 row in val, 1 in test
- [ ] `item_features.csv` has same item_ids as `item_ids.csv`

### 3.5 Register H&M in Pipeline
- `start.py`: add `'hm'` to dataset choices
- `experiment_helper.py`: add ASHA scheduler branch for H&M (same as lfm2b)

### 3.6 Run CF-Only Baselines on H&M
Run all 5 existing models on H&M. Start with `mf` to verify pipeline, then all others.
Results → `Master/experiments/hm_baseline/results_table.csv`

Generate explanations using existing `explanations_utils.py` on best model.

### 3.7 Compare to Competition Context
- Note MAP@12 vs HR@10/NDCG@10 metric difference
- Run a simple popularity baseline on same splits for internal comparison
- Document in `Master/experiments/hm_baseline/comparison_notes.md`

---

## Phase 4: Feature-Aware Prototype Architecture

**Goal:** Extend ProtoMF to use item metadata; experiments and explanations on H&M.

### 4.1 Design Feature-Aware Architecture
Document design in `Master/docs/feature_architecture_design.md`.

> **Architecture decision:** Aim for Option C (Dual Prototype Spaces) as the primary contribution. The paper already proves dual user+item prototypes win consistently (UI-ProtoMF outperforms all single-side variants across all 3 datasets and both metrics). Extending this duality into the feature space is the natural, well-motivated next step. Final architecture decision will be confirmed once the codebase is well understood — don't over-engineer before that point.

**Option A — Feature-Only Embedding (ablation baseline):**
Replace item `Embedding(n_items, dim)` with a feature encoder MLP. No CF signal. Purely content-based prototypes. Run as ablation to isolate feature contribution.

**Option B — Hybrid CF + Features (fallback / intermediate step):**
- Keep item CF embedding: `Embedding(n_items, cf_dim)`
- Add feature encoder: `FeatureEncoder(features) → feature_embedding`
- Combine: `concat(cf_embed, feature_embed)` before cosine similarity with prototypes
- Implement first as a stepping stone toward Option C, and keep as an ablation point

**Option C — Dual Prototype Spaces (primary contribution):**
- Separate prototype sets in their own spaces: CF prototypes + feature prototypes
- CF prototypes: operate on collaborative embeddings (as in existing UI-ProtoMF)
- Feature prototypes: operate on item feature encodings independently
- `item_repr = concat(sim_cf_protos, sim_feat_protos)` — two explanation streams
- "Recommended because users like you interact with it (CF prototypes) AND because of its attributes (feature prototypes)"
- Follows the exact same design principle the paper uses to justify UI-ProtoMF
- Richer, more separable explanations: can show CF-driven vs. content-driven reasoning independently

**H&M feature encoding:**
- Features to use: `product_type_name`, `product_group_name`, `colour_group_name`, `department_name`, `index_group_name`, `section_name`, `garment_group_name`
- Each categorical → small learned embedding (dim 8–16)
- Concatenate all → MLP → `feature_embedding`
- Store feature lookup table as non-learnable tensor in extractor (no dataloader changes needed)

### 4.2 Data Pipeline Changes
**Recommended approach:** Feature lookup table inside the new feature extractor.
- Load `item_features.csv` at model construction time, store as `self.feature_table` tensor
- `forward(item_idxs)` looks up `self.feature_table[item_idxs]`
- **Zero changes** to dataloader, trainer, or tester unpacking

Files to modify:
- `feature_extraction/feature_extractors.py` — add new classes
- `feature_extraction/feature_extractor_factories.py` — new `ft_type` branches, accept `data_path`
- `rec_sys/trainer.py` — pass `data_path` to `create_models()`
- `confs/hyper_params.py` — new hyperparameter configs
- `start.py` — new model CLI choices

### 4.3 Implement New Feature Extractors
New classes in `feature_extraction/feature_extractors.py`:

```python
class FeatureEncoder(nn.Module):
    # Per-feature nn.Embedding layers + MLP
    # Input: item_idxs (looks up stored feature table)
    # Output: feature_embedding of shape [..., feature_embedding_dim]

class FeatureAwarePrototypeEmbedding(PrototypeEmbedding):
    # Combines CF embedding + FeatureEncoder output
    # Prototypes operate in combined embedding space
    # Inherits all regularization logic from PrototypeEmbedding
```

**Verification:** Unit test — instantiate model, forward pass with random data, check output shapes, check loss computation.

### 4.4 Experiments
| Model | H&M CF-only (Phase 3) | H&M Feature-Aware | Delta |
|---|---|---|---|
| `mf` | baseline | N/A | — |
| `item_proto` | ✓ | `feature_item_proto` | compare |
| `user_proto` | ✓ | (if applicable) | compare |
| `user_item_proto` | ✓ | `feature_user_item_proto` | compare |

Results → `Master/experiments/hm_features/results_table.csv`

### 4.5 Fashion-Specific Explanations
Extend `utilities/explanations_utils.py`:
- `get_prototype_feature_profile()`: Dominant feature values per prototype (e.g., "Prototype 3: Trousers, Dark colors, Menswear")
- `explain_recommendation()`: Show which prototypes contributed + what they represent in feature space
- `visualize_prototype_items()`: Show representative items per prototype with metadata (+ images optionally)

Produce `Master/notebooks/03_hm_explanations.ipynb`:
- TSNE plots colored by item category
- Prototype interpretation tables
- Case studies: 5–10 specific (user, item) recommendation explanations
- Compare CF-only vs. feature-aware explanation quality

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
- Obtain university/supervisor template or use a clean academic template
- Set up `Master/thesis/` structure with chapter files
- Verify compilation: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

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
- Architecture diagram: original ProtoMF vs. feature-aware extension
- H&M dataset statistics (transaction distribution, temporal patterns)
- TSNE plots of prototype spaces
- Prototype explanation bar charts (similarity contributions)
- Feature-prototype association heatmaps
- Result tables (replication + H&M + ablations)

All figure sources saved in `Master/thesis/figures/`.

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
| `utilities/consts.py` | Configure `DATA_PATH` and `WANDB_API_KEY` at start of Phase 2 |
| `confs/hyper_params.py` | Add debug config (Phase 1) + new model configs (Phase 4) |
| `Master/docs/modification_map.md` | Living document of all code extension points (Phase 2) |
