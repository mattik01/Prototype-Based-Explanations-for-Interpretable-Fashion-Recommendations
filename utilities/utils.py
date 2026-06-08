import json
import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn


def safe_wandb_init(**kwargs):
    """W&B init that can never abort a run.

    Observability must never terminate an experiment. Returns the run on
    success, or None on any failure (e.g. online init timeout / CommError),
    in which case training proceeds without W&B logging for this run.
    """
    try:
        import wandb
        return wandb.init(**kwargs)
    except Exception as e:
        print(f"[wandb] init failed ({type(e).__name__}: {e}); "
              f"continuing WITHOUT W&B for this run.")
        return None


def safe_wandb_finish():
    """W&B finish that never raises."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"[wandb] finish failed ({type(e).__name__}: {e}); ignoring.")


def safe_wandb_login():
    """Best-effort W&B login that never raises (key is also read from env by init)."""
    try:
        import wandb
        key = os.environ.get('WANDB_API_KEY', '')
        if key:
            wandb.login(key=key)
    except Exception as e:
        print(f"[wandb] login failed ({type(e).__name__}: {e}); continuing.")


def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(x, file_path):
    return pickle.dump(x, open(file_path, 'wb'))


def json_load(file_path):
    return json.load(open(file_path, 'r'))


def general_weight_init(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        if m.weight.requires_grad:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                torch.nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Embedding:
        if m.weight.requires_grad:
            torch.nn.init.normal_(m.weight)

    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def generate_id(prefix=None, postfix=None):
    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)
    if not prefix is None:
        uid = prefix + "_" + uid
    if not postfix is None:
        uid = uid + "_" + postfix
    return uid


def reproducible(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_results(metrics):
    """
    Prints the results on the command line.

    :param metrics: dict containing the metrics to print.
    :return:
    """

    STR_RESULT = "{:10} : {:.3f}"

    for metric_name, metric_value in metrics.items():
        print(STR_RESULT.format(metric_name, metric_value))
