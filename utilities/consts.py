import os

# --- Experiment Constants --- #
SINGLE_SEED = 38210573
SEED_LIST = [SINGLE_SEED, 9491758, 2931009]
NUM_SAMPLES = 100  # How many hyperparameters samples will be taken into account
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')  # Path pointing at the data folder
GPU_PER_TRIAL = 0.0625  # Ray Tune parameter — 16 concurrent trials
CPU_PER_TRIAL = 1  # Ray Tune parameter, how many cpus are allocated for a single trial experiment
# --- Training Constants --- #
MAX_PATIENCE = 10  # How many epochs without an improvement must pass before stopping the experiment
import platform
NUM_WORKERS = 0 if platform.system() == 'Windows' else 2
# --- Evaluation Constants --- #
K_VALUES = [1, 3, 5, 10, 50]  # K value for the evaluation metrics
NEG_VAL = 99  # How many negative samples are considered during negative sampling
OPTIMIZING_METRIC = 'hit_ratio@10'  # Which metric will be used to assess during validation.
# --- Logger Constants --- #
_wandb_key_path = os.path.join(os.path.dirname(__file__), '..', 'Master', 'sensitive', 'api_keys', 'wandb_key.txt')
with open(_wandb_key_path, 'r') as _f:
    WANDB_API_KEY = _f.read().strip()

PROJECT_NAME = 'protomf'
