"""Item-feature plumbing for the feature-aware models.

dc01 (``feature_item_proto``): builds the dense ``(n_items, F)`` LongTensor of per-item
feature-value ids (with a global field-offset encoding) from ``item_features.csv``.
dc02 (``attr_item_proto``): builds the multi-hot ``(n_items, V)`` FloatTensor over attribute-value
columns from the same CSV (``build_attr_multi_hot``).
Both are injected into the feature-extractor config just before the factory builds the model.

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
        a requested field is absent, or a field has NaN/missing cells (F-S0-06 symmetric guard).
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
        vals = df[field]
        # NaN guard symmetric with build_attr_multi_hot (F-S0-06): astype(str) below would
        # otherwise silently encode NaN as a legitimate 'nan' vocab value (a phantom feature).
        if vals.isna().any():
            raise ValueError(
                f"field '{field}' has {int(vals.isna().sum())} NaN/missing cells — every item "
                f"must set exactly one value per field (4.0 analysis)")
        vals = vals.astype(str)
        vocab = sorted(vals.unique().tolist())
        code = {v: i for i, v in enumerate(vocab)}
        feature_ids[:, j] = torch.tensor([code[v] + offset for v in vals], dtype=torch.long)
        offset += len(vocab)

    n_features = offset
    return feature_ids, n_features


def build_attr_multi_hot(data_path: str, fields):
    """Build the dc02 multi-hot item-attribute matrix X from ``<data_path>/item_features.csv``.

    Encoding: the same deterministic vocabulary convention as ``build_feature_ids`` — for each
    field (in the given order) values map to a lexicographically-sorted local vocabulary occupying
    a disjoint contiguous column block ``[field_offsets[f], field_offsets[f+1])``. Every item sets
    exactly one column per field block, so each row has exactly F ones (‖x_i‖₂ = √F — the property
    behind dc02's exact per-attribute decomposition; see the design doc §3.4).

    Kept self-contained (small guard duplication with ``build_feature_ids``) so the dc01-tested
    builder stays byte-identical.

    :param data_path: directory containing ``item_features.csv``.
    :param fields: ordered list of column names to encode as attribute-value columns.
    :return: ``(attr_multi_hot: FloatTensor (n_items, V), field_offsets: list[int] of length F+1,
        code_to_field_value: list[(field, value)] — the (field, value) pair behind each column)``.
    :raises ValueError: if ``item_id`` does not cover ``0..n_items-1`` exactly, a requested field
        is absent, or a field has NaN/missing cells.
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

    # Align rows to item_id order so attr_multi_hot[i] is item i (rows may be shuffled on disk).
    df = df.sort_values('item_id').reset_index(drop=True)

    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(
            f"fields not in item_features.csv: {missing}; available: {list(df.columns)}")

    field_offsets = [0]
    code_to_field_value = []
    per_field_codes = []
    for field in fields:
        vals = df[field]
        if vals.isna().any():
            raise ValueError(
                f"field '{field}' has {int(vals.isna().sum())} NaN/missing cells — dc02 assumes "
                f"every item sets exactly one value per field (4.0 analysis)")
        vals = vals.astype(str)
        vocab = sorted(vals.unique().tolist())
        code = {v: i for i, v in enumerate(vocab)}
        per_field_codes.append(
            torch.tensor([code[v] for v in vals], dtype=torch.long) + field_offsets[-1])
        field_offsets.append(field_offsets[-1] + len(vocab))
        code_to_field_value.extend((field, v) for v in vocab)

    n_attr_values = field_offsets[-1]
    attr_multi_hot = torch.zeros((n_items, n_attr_values), dtype=torch.float32)
    for col_codes in per_field_codes:
        attr_multi_hot[torch.arange(n_items), col_codes] = 1.0

    # Construction invariant: exactly one 1 inside each field block → row sums == F.
    if not bool((attr_multi_hot.sum(dim=1) == float(len(fields))).all()):
        raise ValueError('attr_multi_hot construction error: a row lacks exactly one value per field')

    return attr_multi_hot, field_offsets, code_to_field_value


def inject_feature_ids(ft_ext_param: dict, data_path: str) -> dict:
    """Guarded injection seam for ``trainer._build_model`` / ``tester._build_model``.

    Returns the ``ft_ext_param`` the factory should consume.

    For ``ft_type == 'feature_item_proto'`` it returns a SHALLOW copy whose ``item_ft_ext_param``
    (and ``user_ft_ext_param``) sub-dicts are themselves shallow-copied, with the freshly-built
    ``feature_ids`` tensor + ``n_features`` added to the item copy. The CALLER's dict is left
    untouched on purpose: only the lightweight spec (``feature_fields`` / ``use_id_feature``) — never
    the tensor — must persist in the Ray/JSON config. (Mutating the caller in place would leak the
    tensor into ``config.json`` via ``best_trial_config`` → ``run_combo._save_combo_results``'s
    ``json.dump(default=str)``, bloating it with a stringified ``(n_items, F)`` tensor.) The user
    sub-dict is also copied because the factory mutates its ``ft_type``/``out_dimension`` in place.

    For ``ft_type == 'attr_item_proto'`` (dc02) the same discipline applies with the multi-hot
    payload instead: the config carries only the lightweight ``attr_fields`` spec; the copy gains
    the freshly-built ``attr_multi_hot`` (n_items, V) FloatTensor + ``n_attr_values`` +
    ``field_offsets``.

    For every other ``ft_type`` the SAME object is returned unchanged, so the factory mutates it in
    place exactly as before (no behavioural change for existing models).
    """
    ft_type = ft_ext_param.get('ft_type')

    if ft_type == 'feature_item_proto':
        item_spec = dict(ft_ext_param['item_ft_ext_param'])
        fields = item_spec['feature_fields']
        feature_ids, n_features = build_feature_ids(data_path, fields)
        item_spec['feature_ids'] = feature_ids
        item_spec['n_features'] = n_features
    elif ft_type == 'attr_item_proto':
        item_spec = dict(ft_ext_param['item_ft_ext_param'])
        fields = item_spec['attr_fields']
        attr_multi_hot, field_offsets, _ = build_attr_multi_hot(data_path, fields)
        item_spec['attr_multi_hot'] = attr_multi_hot
        item_spec['n_attr_values'] = int(attr_multi_hot.shape[1])
        item_spec['field_offsets'] = field_offsets
    else:
        return ft_ext_param

    new_param = dict(ft_ext_param)
    new_param['item_ft_ext_param'] = item_spec
    new_param['user_ft_ext_param'] = dict(ft_ext_param['user_ft_ext_param'])
    return new_param
