"""Load a trained RecSys from a structured combo results directory."""
import json
import os

import pandas as pd
import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from rec_sys.rec_sys import RecSys
from utilities.consts import DATA_PATH


def _count_rows(csv_path: str) -> int:
    return pd.read_csv(csv_path).shape[0]


def load_recsys_from_results_dir(results_dir: str):
    """
    Rebuild a RecSys from the artifacts saved by `_save_combo_results` in run_combo.py.

    :param results_dir: path to a combo folder containing best_model.pth, config.json, metadata.json
    :return: (model, config, metadata) with model.eval() already called
    """
    config_path = os.path.join(results_dir, "config.json")
    metadata_path = os.path.join(results_dir, "metadata.json")
    checkpoint_path = os.path.join(results_dir, "best_model.pth")

    for p in (config_path, metadata_path, checkpoint_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found in results dir: {p}")

    with open(config_path) as f:
        config = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)

    dataset = metadata["dataset"]
    dataset_dir = os.path.join(DATA_PATH, dataset)
    n_users = _count_rows(os.path.join(dataset_dir, "user_ids.csv"))
    n_items = _count_rows(os.path.join(dataset_dir, "item_ids.csv"))

    user_fe, item_fe = FeatureExtractorFactory.create_models(
        config["ft_ext_param"], n_users, n_items
    )
    model = RecSys(
        n_users, n_items,
        config["rec_sys_param"],
        user_fe, item_fe,
        config["loss_func_name"],
        config.get("loss_func_aggr", "mean"),
    )
    model.init_parameters()

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, config, metadata
