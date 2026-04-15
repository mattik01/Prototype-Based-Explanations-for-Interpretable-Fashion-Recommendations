"""Resolve an items_info DataFrame for a dataset.

Prefers rich `item_features.csv` (H&M); falls back to minimal `item_ids.csv`
(present for every dataset). Returned DataFrame always has an `item_id` column.
"""
import os

import pandas as pd

from utilities.consts import DATA_PATH


def load_items_info(dataset: str) -> pd.DataFrame:
    dataset_dir = os.path.join(DATA_PATH, dataset)
    rich = os.path.join(dataset_dir, "item_features.csv")
    minimal = os.path.join(dataset_dir, "item_ids.csv")

    path = rich if os.path.exists(rich) else minimal
    df = pd.read_csv(path)
    if "item_id" not in df.columns:
        raise ValueError(f"{path} missing required 'item_id' column")
    return df
