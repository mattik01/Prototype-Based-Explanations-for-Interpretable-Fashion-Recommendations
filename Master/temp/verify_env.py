import os
import sys

# --- core (explanations) stack ---
import torch
import numpy
import scipy
import pandas
import sklearn
import matplotlib

print('torch', torch.__version__, '| numpy', numpy.__version__,
      '| scipy', scipy.__version__, '| sklearn', sklearn.__version__,
      '| mpl', matplotlib.__version__)

# --- training stack (Ray/W&B) ---
import ray
import wandb
import hyperopt
import tensorboardX
import google.protobuf
print('ray', ray.__version__, '| wandb', wandb.__version__,
      '| hyperopt', hyperopt.__version__,
      '| protobuf', google.protobuf.__version__)

# --- codebase still loads the checkpoint (core not broken) ---
sys.path.insert(0, os.getcwd())
from utilities.explanations.loader import load_recsys_from_results_dir
from utilities.explanations.items_info import load_items_info

m, cfg, meta = load_recsys_from_results_dir(
    'Master/experiments/results/user_item_proto_ml-1m_s38210573')
info = load_items_info('ml-1m')
print('OK: checkpoint loads ->', meta['model'], meta['dataset'],
      '| items_info cols:', list(info.columns))
