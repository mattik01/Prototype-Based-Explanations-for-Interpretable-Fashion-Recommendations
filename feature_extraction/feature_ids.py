"""dc01 feature-id plumbing for the ``feature_item_proto`` model.

Builds the dense ``(n_items, F)`` LongTensor of per-item feature-value ids (with a global
field-offset encoding) from ``item_features.csv``, and injects it into the feature-extractor
config just before the factory builds the model.

Design notes (see ``Master/temp/dc_checks/dc01/dc01_feature_item_proto_implementation_plan.md``):
- The TENSOR is built here, inside ``trainer._build_model`` / ``tester._build_model`` — it is
  NEVER serialized into the Ray/JSON config (which only carries the lightweight spec:
  ``feature_fields`` + ``use_id_feature``). A 13,651×F tensor in the per-trial config dump is
  unacceptable.
- ``feature_ids`` is registered as a non-persistent buffer in ``FeatureEmbedding``, so it is
  reconstructed deterministically at build time and never round-trips through the checkpoint.
- ID rows (the LightFM "tags + ids" term) are handled inside ``FeatureEmbedding`` via the
  ``n_features`` offset and ``n_objects`` extra rows — this builder produces metadata ids only.
"""
import os

import pandas as pd
import torch


def build_feature_ids(data_path: str, fields):
    """Build the per-item feature-id tensor from ``<data_path>/item_features.csv``.

    Encoding: for each field (in the given order), values are mapped to a deterministic,
    lexicographically-sorted local vocabulary; a running global offset is added so the field's
    codes occupy a disjoint contiguous block. The total ``n_features`` is the sum of the per-field
    vocabulary sizes (metadata only — ID rows are appended downstream by ``FeatureEmbedding``).

    :param data_path: directory containing ``item_features.csv`` (and ``item_ids.csv``).
    :param fields: ordered list of column names to encode as features.
    :return: ``(feature_ids: LongTensor (n_items, F), n_features: int)``.
    :raises ValueError: if ``item_id`` does not cover ``0..n_items-1`` exactly (missing/extra/dup),
        or a requested field is absent.
    """
    csv_path = os.path.join(data_path, 'item_features.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'item_features.csv not found at {csv_path}')

    df = pd.read_csv(csv_path)
    if 'item_id' not in df.columns:
        raise ValueError(f"item_features.csv at {csv_path} has no 'item_id' column")

    n_items = len(df)
    # Alignment guard: item_id must cover 0..n_items-1 exactly (no missing/extra/duplicate).
    item_id_sorted = sorted(int(x) for x in df['item_id'].tolist())
    if item_id_sorted != list(range(n_items)):
        raise ValueError(
            f"item_id must cover 0..{n_items - 1} exactly; got "
            f"min={item_id_sorted[0]}, max={item_id_sorted[-1]}, "
            f"n_unique={len(set(item_id_sorted))}, n_rows={n_items}")

    # Align rows to item_id order so feature_ids[i] is item i (rows may be shuffled on disk).
    df = df.sort_values('item_id').reset_index(drop=True)

    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(
            f"fields not in item_features.csv: {missing}; available: {list(df.columns)}")

    feature_ids = torch.zeros((n_items, len(fields)), dtype=torch.long)
    offset = 0
    for j, field in enumerate(fields):
        vals = df[field].astype(str)
        vocab = sorted(vals.unique().tolist())
        code = {v: i for i, v in enumerate(vocab)}
        feature_ids[:, j] = torch.tensor([code[v] + offset for v in vals], dtype=torch.long)
        offset += len(vocab)

    n_features = offset
    return feature_ids, n_features


def inject_feature_ids(ft_ext_param: dict, data_path: str):
    """Guarded injection seam for ``trainer._build_model`` / ``tester._build_model``.

    No-op unless ``ft_type == 'feature_item_proto'``. When it matches, builds the ``feature_ids``
    tensor from ``data_path`` and writes ``feature_ids`` + ``n_features`` into
    ``ft_ext_param['item_ft_ext_param']`` for the factory to consume. Idempotent.
    """
    if ft_ext_param.get('ft_type') != 'feature_item_proto':
        return
    item_spec = ft_ext_param['item_ft_ext_param']
    fields = item_spec['feature_fields']
    feature_ids, n_features = build_feature_ids(data_path, fields)
    item_spec['feature_ids'] = feature_ids
    item_spec['n_features'] = n_features
