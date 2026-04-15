import os

# --- Experiment Constants --- #
SINGLE_SEED = 38210573
SEED_LIST = [SINGLE_SEED, 9491758, 2931009]
NUM_SAMPLES = 100  # How many hyperparameters samples will be taken into account
DATA_PATH = os.environ.get('PROTOMF_DATA_PATH',
                           os.path.join(os.path.dirname(__file__), '..', 'data'))
EXPERIMENT_RESULTS_PATH = os.environ.get('PROTOMF_RESULTS_PATH',
                                         os.path.join(os.path.dirname(__file__), '..', 'Master', 'experiments', 'results'))
GPU_PER_TRIAL = 0.0625  # Ray Tune parameter — 16 concurrent trials
CPU_PER_TRIAL = 1  # Ray Tune parameter, how many cpus are allocated for a single trial experiment
# --- Training Constants --- #
MAX_PATIENCE = 10  # How many epochs without an improvement must pass before stopping the experiment
# Minimum improvement on the optimizing metric required to reset patience.
# Per-metric defaults — both hit_ratio@K and ndcg@K live in [0, 1], so 1e-4 (≈0.01%) is meaningful but not noisy.
MIN_DELTA_DEFAULTS = {
    'hit_ratio@1': 1e-4, 'hit_ratio@3': 1e-4, 'hit_ratio@5': 1e-4,
    'hit_ratio@10': 1e-4, 'hit_ratio@50': 1e-4,
    'ndcg@1': 1e-4, 'ndcg@3': 1e-4, 'ndcg@5': 1e-4,
    'ndcg@10': 1e-4, 'ndcg@50': 1e-4,
}
MIN_DELTA_FALLBACK = 1e-4  # Used when the optimizing metric isn't in MIN_DELTA_DEFAULTS
import platform
NUM_WORKERS = 0 if platform.system() == 'Windows' else 2
# --- Evaluation Constants --- #
K_VALUES = [1, 3, 5, 10, 50]  # K value for the evaluation metrics
NEG_VAL = 99  # How many negative samples are considered during negative sampling
OPTIMIZING_METRIC = 'hit_ratio@10'  # Which metric will be used to assess during validation.
# --- Logger Constants --- #
_wandb_key_path = os.path.join(os.path.dirname(__file__), '..', 'Master', 'sensitive', 'api_keys', 'wandb_key.txt')
try:
    with open(_wandb_key_path, 'r') as _f:
        WANDB_API_KEY = _f.read().strip()
except FileNotFoundError:
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY', '')

PROJECT_NAME = 'protomf'
