# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## RULES
- Any obviously temporary files, summaries, markdowns, scripts — put them into `Master/temp/` folder and inform the user. The user might move them appropriately later.
- **Protocol entries:** At meaningful thesis-relevant milestones (research decisions, architecture choices, experiment results, methodology reasoning), consider commits and last protocol entries when making the new one. Prompt the user to log with:
  `━━━ PROTOCOL ━━━ Consider logging: <brief topic>. Trigger with /protocol`
  Skip pure infrastructure/tooling steps. Do not suggest too frequently — only at natural breakpoints.
- **Modifications log:** When modifying any file that exists in the original upstream repo, silently update `Master/docs/modifications_log.md` with the filename and a one-sentence summary of what changed. Do not ask — just keep it updated.
- **Git checkpoints:** Before risky changes (new feature attempts, large refactors, dependency upgrades, testing divergent approaches), prompt the user with:
  `━━━ CHECKPOINT ━━━ Consider saving current state before <risky thing>. Trigger with /gitcheck`
  Also suggest across session boundaries when there's uncommitted work. Do not suggest too frequently.
- **Git commits:** Never auto-push to remote. Use conventional commit prefixes (`feat:`, `docs:`, `refactor:`, `fix:`, `chore:`). Split commits only when concerns are clearly distinct.
- **Environment awareness:** Two machines matter — the **main Linux laptop** (development, debugging, small CPU tests) and the **LEO5 HPC cluster** (all GPU training/hyperopt; primary compute, `ssh leo5`, SLURM). Verify with `hostname` at session start; details in `Master/sensitive/` (`infrastructure.md`, `leo5_cluster.md`). The Windows GPU desktop is a **legacy fallback**, effectively unused now that cluster access exists — ignore unless explicitly asked to use it.
- **Smoke tests & small workloads (policy, 2026-07-13):** whenever the VPN is up, run smoke tests on the **LEO5 login node** rather than local CPU — it carries **2× A30 (24 GB) deliberately provided for smoke tests** (not SLURM-managed; run directly on the node, no sbatch; details + etiquette in `Master/sensitive/leo5_cluster.md`). Orders of magnitude faster than laptop CPU, and it mirrors the environment of the actual runs. If the VPN is **not** up, ask the user to bring it up rather than silently falling back to local CPU (local CPU only if the user declines or the cluster is unreachable). The same placement applies to **small experimental workloads with low expected wall time** (e.g. the hidden-effects-section instruments — post-hoc measurements on trained checkpoints): run them directly on the login node (user decision 2026-07-13; if the cluster staff ever object, stop and move them into SLURM). **Boundary:** anything that produces a *compared row* (fleet, reference, candidate, ablation, cold retrains — everything with a C8 manifest) stays SLURM, one `/leo5-submit` per run — that hard gate is untouched.
- **Code sync:** git is the sync mechanism between laptop and cluster. Pull before starting; when leaving work for a cluster run, commit with a clear message describing what to run next.
- **Cluster sync protocol:** When working on the cluster via SSH and a code change is needed, never edit files directly on the cluster. Instead, ask the user for permission to commit and push locally, then `git pull` on the cluster to sync. The laptop is always the source of truth for code changes.
- **LLM-usage transparency archive:** For end-of-thesis disclosure of how LLMs assisted this work, maintain `Master/llm_usage/` (staging; copied into the thesis's `LLM-Usage/` appendix at finalization). It collects tangible indications of AI assistance — the Claude Code skill/command/MCP setup, versioned `CLAUDE.md` snapshots, AI memory/orchestration documents, and the Scrutiny-Protocol dossiers and similar AI-driven artifacts. See `Master/llm_usage/README.md`. Two standing behaviours:
  - **Version CLAUDE.md:** before any *substantive* edit to this file, copy the current version to `Master/llm_usage/claude_md_versions/CLAUDE_vN.md` (increment N; skip trivial typo/format fixes), then edit — and add a row to the version log in `Master/llm_usage/README.md`. Git history is the fine-grained backstop.
  - **Preserve at-risk orchestration artifacts:** the concern is *only* things that get routinely deleted or refactored away — chiefly the Scrutiny-Protocol dossiers/ledgers when a phase closes. Don't clean those up; keep them and route a copy into `Master/llm_usage/`. Stable, persistent docs that aren't regularly churned (protocol vault, memory files, design-candidate docs, skill list) need no special handling — they survive on their own and can simply be gathered at the end.

## Claude Code Tools
- **Plugins:** context7 (library docs), pyright-lsp (type checking), code-simplifier (`/simplify`)
- **MCP:** context7 for live PyTorch/Ray/wandb docs lookup
- **Built-in:** WebSearch/WebFetch for papers, Glob/Grep for code search, Bash for execution
- **Drive sync:** `/drive-sync up|down` mirrors gitignored artifacts (literature PDFs, `*.pth` checkpoints) to a private Google Drive — a third channel beside git (code) and scp (data/secrets). See the skill for setup, the secrets boundary, and the new-machine bootstrap.
- **Paper reading:** `/paper-assistant <id>` runs the full reading-companion mode for a literature paper — explain dense passages plainly → assess relevance vs the thesis lenses → route keepers to the right artifact → finalize. It is the umbrella that composes `/annotate` (margin notes, wide default), `/protocol` (decisions/insights), `/design-candidate` (+ scratchpad, design ideas), and `/paper-finalize` (closing `_SUMMARY.md`). Use it whenever working through a paper in `Master/literature/`.
- **Scrutiny:** `/scrutiny s0|dcNN|stepX dcNN|resume dcNN` runs the Scrutiny Protocol (`Master/docs/scrutiny_protocol.md`) — step-gated audit-and-repair of the eval foundation (S0) and each design candidate (concept → implementation → explanations → cluster-run plan → post-run chapter draft), on the `feat/scrutiny` integration branch. Artifacts under `Master/docs/scrutiny/`.

## Branch Strategy
- `main` — clean reference, stays untouched
- **Feature branches** (`feat/*`) — where work happens; current integration branch is `feat/scrutiny` (the Scrutiny Protocol). Merge to `main` only at clean milestones.

## Naming Conventions
- **Files:** snake_case (e.g., `hm_splitter.py`, `feature_encoder.py`)
- **H&M-specific modules:** `hm_` prefix (e.g., `hm_splitter.py`)
- **Feature-aware model variants:** `feature_` prefix (e.g., `feature_item_proto`)
- **Experiments:** `{model}_{dataset}_{variant}_{seed}` (e.g., `user_item_proto_ml-1m_baseline_42`)

## `ft_type` Values (Phase 4)
- `feature_item_proto` — feature-composed item prototypes (dc01 → fI-ProtoMF; implemented; ablation variants `_noid`, `_f0`)
- `attr_item_proto` — attribute-space item prototypes, concept-bottleneck-anchored (dc02; implemented)
- `lightfm` — LightFM-style CBF baseline, no prototypes (S0 baseline; models `lightfm_tags`, `lightfm_tags_ids`)
- `feature_user_proto` — history-composed user factors on the U-ProtoMF host (dc05 → fU-ProtoMF; implemented at the dc05 build; ablation variant `_noid`). **dc08 knobs (2026-09-09, not a new `ft_type`):** `history_id_rows` + `history_row_weight` (λ_y) add SVD++/FISM item-identity rows y_j to the composition; `freeze_history_rows` is the capacity control; `history_pooling='auto'` switches the composition to a flat `embedding_bag` when the bag is wide (required on ml-1m, no-op on hm). Arms `_noid_itemid` (reporting), `_itemid`, `_noid_itemid_frozen`
- `lightfm_hist` — history-composed user × free item, plain dot, no prototypes (dc05 S1 decoupled control, F-DC05-21; models `lightfm_hist`, `lightfm_hist_ids`; dc08 identity-row twins `lightfm_hist_itemid`, `lightfm_hist_ids_itemid` — the dot half of the paired 2×2)
- `dual_item_proto` — dual CF + feature prototype spaces (reserved, not yet built)

## Results Storage
- **Canonical results table:** `Master/docs/replication_report.md` (paper replication + H&M CF baselines; the once-planned per-phase `results_table.csv` files were never created).
- Run artifacts mirrored under `Master/experiments/` (e.g. `replication/`, H&M run dirs).

## Modification Map
- See `Master/docs/modification_map.md`
- Documents code extension points for adding datasets, models, and features
- **Not an absolute source of truth** — treat as a valuable reference when planning implementation, but always verify against the actual code before relying on it

## H&M Data
- Raw Kaggle files go in `data/hm/raw/` (gitignored, too large for version control)
- Expected raw files: `transactions_train.csv`, `articles.csv`, `customers.csv`
- Processed splits will be generated by `data/hm/hm_splitter.py` into `data/hm/`

### ✅ Resolved: image set is COMPLETE (2026-07-11, was "may be incomplete" since 2026-06-07)
The 2026-06-07 consolidation note suspected the local `data/hm/raw/images/` set
(86 subdirectories) was partial, based on an assumed full range of `000`–`0NN`.
Verified against the official Kaggle zip's central directory on 2026-07-11
(S0-build session): the competition dataset contains exactly **105,100 images in
86 subdirectories `010`–`095`** (contiguous; article ids simply never start
below 010). The local set has exactly 105,100 jpgs in `010`–`095` — **complete**.
A fresh copy of the zip (30,810,293,747 bytes) lives on LEO5 at
`/scratch/c7031336/hm_raw_kaggle/` with images extracted alongside (the dc03
training-side copy; S0.6 precondition).

## Project Overview

ProtoMF is a research codebase implementing "Prototype-based Matrix Factorization for Effective and Explainable Recommendations" (RecSys 2022). It trains recommender system models using PyTorch, with hyperparameter optimization via Ray Tune and experiment logging via Weights & Biases.

## Environment Setup

**At the start of every session**, activate the conda environment before running any Python:
```bash
conda activate protomf
```

Device detection is automatic (`torch.cuda.is_available()`), so the same code runs everywhere.

| Machine | Device | Role |
|---------|--------|------|
| Main laptop | CPU | Development, debugging, small tests |
| LEO5 HPC cluster | CUDA (A100/A40/A30) | All training & hyperopt — primary compute (`ssh leo5`, SLURM) |

Cluster and SSH configuration are in `Master/sensitive/leo5_cluster.md` / `infrastructure.md`.

The local CPU env was pip-installed (not from `protomf.yml`) with minor version adjustments:
Bottleneck 1.3.4 (build fix), protobuf 3.20.0 (Ray 1.6.0 compat).

## Sensitive Files

`Master/sensitive/` (gitignored) contains:
- `api_keys/` — W&B API key (loaded by `utilities/consts.py`)
- `infrastructure.md` — machines/network overview (laptop, cluster, legacy GPU desktop)
- `leo5_cluster.md` — LEO5 reference (SSH, SLURM, VPN, scratch paths)
- `home_desktop.md` — laptop reference (SSH key, conda env)
- `wake_gpu.sh` / `shutdown_gpu.sh` — legacy GPU-desktop power scripts (rarely used)

## Configuration (Required Before Running)

- `DATA_PATH` in `utilities/consts.py` is auto-detected relative to repo root
- `WANDB_API_KEY`: place key in `Master/sensitive/api_keys/wandb_key.txt` (loaded automatically)

## Running Experiments

```bash
# Single seed hyperparameter optimization
python start.py -m <model> -d <dataset>

# Multi-seed (3 seeds) hyperparameter optimization
python start.py -m <model> -d <dataset> -mp

# Custom seed
python start.py -m <model> -d <dataset> -s <seed>
```

**Models** (`-m`): baselines `mf`, `acf`, `user_proto`, `item_proto`, `user_item_proto`; feature-aware `feature_item_proto` (+ `_noid`/`_f0` ablations), `attr_item_proto`, `feature_user_proto` (+ `_noid` ablation, `_debug` smoke); dc08 identity-row arms `feature_user_proto_itemid`, `feature_user_proto_noid_itemid`, `feature_user_proto_noid_itemid_frozen`; CBF baseline `lightfm_tags`, `lightfm_tags_ids`; decoupled controls `lightfm_hist`, `lightfm_hist_ids`, `lightfm_hist_itemid`, `lightfm_hist_ids_itemid`

**Datasets** (`-d`): `ml-1m`, `amazon2014` (paper); `hm_1_month`, `hm_3_month`, `hm_full`, `hm_1_month_cold` (H&M). `lfm2b-1mon` unavailable.

## Data Preprocessing

Download datasets per `data/README.md`, place files in their respective `data/<dataset>/` folder, then:

```bash
cd data/<dataset_folder>
python <dataset_name>_splitter.py -lh ./
```

This produces 5 files: train/val/test listening histories + user/item id files.

## Architecture

### Data Flow
`start.py` → `experiment_helper.py` → Ray Tune trials → `Trainer` → `RecSys` → `FeatureExtractor`

### Key Components

**`rec_sys/rec_sys.py` — `RecSys`**: Core PyTorch module. Holds user and item `FeatureExtractor` instances, computes dot-product similarity between user/item embeddings, and aggregates recommendation loss + feature extractor regularization losses.

**`feature_extraction/feature_extractors.py`**: All model variants as `FeatureExtractor` subclasses (abstract base):
- `Embedding`: Standard MF embedding
- `EmbeddingW`: Embedding + linear projection (for weight sharing)
- `AnchorBasedCollaborativeFiltering`: ACF baseline (Barkan et al., CIKM 2021)
- `PrototypeEmbedding`: Core ProtoMF block — represents objects by cosine similarity to learned prototypes, with configurable regularization losses
- `ConcatenateFeatureExtractors`: Concatenates two extractors' outputs (used for `user_item_proto`)

**`feature_extraction/feature_extractor_factories.py` — `FeatureExtractorFactory`**: Factory that instantiates user/item feature extractor pairs from a config dict. Handles weight tying between user embedding and item prototype branches in the `prototypes_double_tie` variant.

**`confs/hyper_params.py`**: Ray Tune search spaces for each model. Hyperparameters include embedding dim, n_prototypes, regularization weights (`sim_proto_weight`, `sim_batch_weight`), loss function (`bce`/`bpr`/`sampled_softmax`), optimizer, and negative sampling strategy.

**`rec_sys/trainer.py` — `Trainer`**: Builds model and optimizer, runs training loop with early stopping (patience=10), saves best checkpoint via Ray Tune, reports metrics to Ray Tune/wandb.

**`rec_sys/tester.py` — `Tester`**: Loads a saved checkpoint and evaluates on the test set.

**`rec_sys/protomf_dataset.py`**: Dataset and DataLoader with negative sampling (uniform or popularity-based).

**`utilities/eval.py` — `Evaluator`**: Computes NDCG@K and Hit Ratio@K (K ∈ {1,3,5,10,50}). The validation metric used for model selection is `hit_ratio@10` (configurable in `consts.py`).

**`utilities/explanations_utils.py`**: Post-hoc analysis tools — TSNE plots of prototype spaces, prototype interpretation by representative items, and graphical recommendation explanations.

### ProtoMF Model Variants

| CLI name | `ft_type` | Description |
|---|---|---|
| `mf` | `detached` | Standard Matrix Factorization |
| `acf` | `acf` | Anchor-based CF baseline |
| `user_proto` | `prototypes` | User prototypes + item embeddings |
| `item_proto` | `prototypes` | Item prototypes + user embeddings |
| `user_item_proto` | `prototypes_double_tie` | Both user and item prototypes with tied embedding weights |

### `PrototypeEmbedding` Key Parameters
- `n_prototypes`: number of prototype vectors (learned `nn.Parameter`)
- `cosine_type`: `shifted` (default, 1+cos), `standard`, or `shifted_and_div`
- `reg_proto_type` / `reg_batch_type`: `max` or `soft` (entropy) regularization on the prototype similarity matrix
- `sim_proto_weight` / `sim_batch_weight`: regularization loss weights
