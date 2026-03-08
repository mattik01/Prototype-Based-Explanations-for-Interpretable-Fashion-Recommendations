# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## RULES




## Project Overview

ProtoMF is a research codebase implementing "Prototype-based Matrix Factorization for Effective and Explainable Recommendations" (RecSys 2022). It trains recommender system models using PyTorch, with hyperparameter optimization via Ray Tune and experiment logging via Weights & Biases.

## Environment Setup

```bash
conda env create -f protomf.yml
conda activate protomf
```

The environment uses Python 3.9, PyTorch 1.9.1 with CUDA 10.2, Ray 1.6.0, and wandb 0.12.3.

## Configuration (Required Before Running)

Edit `utilities/consts.py` to set:
- `DATA_PATH`: absolute path to the `./data` folder
- `WANDB_API_KEY`: Weights & Biases API key (all results are logged there)

## Running Experiments

```bash
# Single seed hyperparameter optimization
python start.py -m <model> -d <dataset>

# Multi-seed (3 seeds) hyperparameter optimization
python start.py -m <model> -d <dataset> -mp

# Custom seed
python start.py -m <model> -d <dataset> -s <seed>
```

**Models** (`-m`): `mf`, `acf`, `user_proto`, `item_proto`, `user_item_proto`

**Datasets** (`-d`): `amazon2014`, `ml-1m`, `lfm2b-1mon`

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
