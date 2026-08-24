"""Item-feature plumbing for the feature-aware models.

dc01 (``feature_item_proto``): builds the dense ``(n_items, F)`` LongTensor of per-item
feature-value ids (with a global field-offset encoding) from ``item_features.csv``.
dc02 (``attr_item_proto``): builds the multi-hot ``(n_items, V)`` FloatTensor over attribute-value
columns from the same CSV (``build_attr_multi_hot``).
dc05 (``feature_user_proto``): builds the padded per-USER history-weight tensors from the split's
own train file + the same item vocabulary (``build_user_history_weights``).
All are injected into the feature-extractor config just before the factory builds the model.

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

# --- Per-dataset canonical field sets (charter C5, as amended 2026-07-12) -------------------
# The canonical field set is PER-DATASET: configs carry the sentinel 'canonical' in
# `feature_fields`, resolved against this registry by `resolve_feature_fields_in_conf`
# (called in experiment_helper.start_hyper, where conf and dataset name meet BEFORE Ray
# serializes the conf) — so every trial config / config.json / C8 manifest records the
# RESOLVED literal list + layout, never the sentinel. Saved configs (tester, cold_eval,
# explanations loader) therefore always carry literals and need no re-resolution.
# Layouts: 'fixed' = one value per field (build_feature_ids); 'bags' = variable-size
# indicator bags (build_feature_bags; S0.5-addendum decision A — raw bags).
CANONICAL_FEATURE_FIELDS = {
    'hm': {  # the S0.5 five (V=426); static-catalog-only, price_band deferred, product_code never
        'fields': ['department_name', 'product_type_name', 'section_name',
                   'colour_group_name', 'graphical_appearance_name'],
        'layout': 'fixed',
    },
    'ml-1m': {  # B5 §4.1's published MovieLens recipe: genres + Tag-Genome tags @0.8, bags
        'fields': ['genres', 'tags'],
        'layout': 'bags',
    },
}


def resolve_canonical_fields(dataset: str):
    """Map a dataset name (incl. derived variants like ``hm_1_month_cold``/``ml-1m_cold``) to
    its canonical field set. Loud error for datasets with no canonical set (e.g. amazon2014 —
    discarded from the feature-aware scope, charter C2 amendment: no item features exist).

    :return: ``(fields: list[str], layout: str)``
    """
    if dataset.startswith('hm'):
        entry = CANONICAL_FEATURE_FIELDS['hm']
    elif dataset.startswith('ml-1m'):
        entry = CANONICAL_FEATURE_FIELDS['ml-1m']
    else:
        raise ValueError(
            f"no canonical feature-field set for dataset {dataset!r} (charter C5 covers the "
            f"hm_* and ml-1m* families only; amazon2014 has no item features — C2 amendment)")
    return list(entry['fields']), entry['layout']


def resolve_feature_fields_in_conf(conf: dict, dataset: str):
    """Replace the ``'canonical'`` sentinel in a run conf's feature specs with the dataset's
    resolved field list + ``feature_layout`` (in place, pre-Ray — the C8-manifest seam).

    Only the sentinel is touched: literal field lists (e.g. the F=0 ablation's ``[]``, or a
    saved config retrained via ``--retrain-config``) pass through unchanged — no magic
    overrides (absent-key/config-default hazards, learnings ledger).
    """
    ft_ext_param = conf.get('ft_ext_param')
    if not isinstance(ft_ext_param, dict):
        return
    for side in ('item_ft_ext_param', 'user_ft_ext_param'):
        spec = ft_ext_param.get(side)
        if isinstance(spec, dict) and spec.get('feature_fields') == 'canonical':
            fields, layout = resolve_canonical_fields(dataset)
            spec['feature_fields'] = fields
            spec['feature_layout'] = layout
            print(f"[feature-fields] {side}: 'canonical' resolved for {dataset!r} -> "
                  f"{fields} (layout={layout})")


def _fixed_field_vocab(df, field: str):
    """Per-field value series + deterministic vocabulary for the FIXED layout — the single
    recipe behind both ``build_feature_ids``'s codes and any code→label reconstruction
    (F-DC05-13: read-out labels must come from the builders' own vocab logic, never from an
    independent re-derivation).

    :return: ``(vals: pd.Series[str], vocab: list[str] — lexicographically sorted unique values)``
    :raises ValueError: on NaN/missing cells (F-S0-06 symmetric guard).
    """
    vals = df[field]
    # NaN guard symmetric with build_attr_multi_hot (F-S0-06): astype(str) below would
    # otherwise silently encode NaN as a legitimate 'nan' vocab value (a phantom feature).
    if vals.isna().any():
        raise ValueError(
            f"field '{field}' has {int(vals.isna().sum())} NaN/missing cells — every item "
            f"must set exactly one value per field (4.0 analysis)")
    vals = vals.astype(str)
    return vals, sorted(vals.unique().tolist())


def _bag_field_tokens(cells, field: str, multi_value_sep: str):
    """Per-field token lists + deterministic vocabulary for the BAGS layout — the single
    recipe behind both ``build_feature_bags``'s codes and any code→label reconstruction
    (F-DC05-13). Cell content is taken verbatim (split on the separator only, no stripping).

    :param cells: the field's cell strings, row order (from a ``keep_default_na=False`` read).
    :return: ``(per_item: list[list[str]], vocab: list[str] — lexicographically sorted tokens)``
    :raises ValueError: on a 'nan' token, an empty token, or duplicate tokens within a cell.
    """
    per_item, vocab = [], set()
    for i, cell in enumerate(cells):
        toks = cell.split(multi_value_sep) if cell != '' else []
        for t in toks:
            if t.lower() == 'nan':
                raise ValueError(
                    f"field '{field}' item_id {i} carries a 'nan' token — phantom NaN value "
                    f"(F-S0-06 guard; a missing value must be an empty cell, not text)")
            if t == '':
                raise ValueError(
                    f"field '{field}' item_id {i} has an empty token (stray '{multi_value_sep}' "
                    f"separator in cell {cell!r})")
        if len(set(toks)) != len(toks):
            raise ValueError(
                f"field '{field}' item_id {i} has duplicate tokens in cell {cell!r} — an item "
                f"carries a value at most once (indicator bags)")
        per_item.append(toks)
        vocab.update(toks)
    return per_item, sorted(vocab)


def build_code_to_field_value(data_path: str, fields, layout: str = 'fixed',
                              multi_value_sep: str = '|'):
    """The metadata word behind each global vocab id, in exactly the builders' order —
    the SINGLE source of truth for code→label reconstruction (F-DC05-13).

    Reads ``<data_path>/item_features.csv`` with the same read discipline as the respective
    builder (bags: ``keep_default_na=False, dtype=str`` so '' stays a legal short bag) and
    concatenates the per-field vocabularies via the shared recipe functions the builders
    themselves call — so labels CANNOT drift from the model's vocabulary on either layout.
    Consumers: the breakdown renderer's ``feature_code_labels`` and the naming layer's
    ``name_prototypes_from_score_matrix`` (replacing their former ad-hoc reconstructions,
    which silently assumed the fixed layout).

    :return: ``list[(field, value)]`` of length ``n_features``, index == global vocab code.
    """
    if layout not in ('fixed', 'bags'):
        raise ValueError(f"layout must be 'fixed' or 'bags'; got {layout!r}")
    csv_path = os.path.join(data_path, 'item_features.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'item_features.csv not found at {csv_path}')
    if layout == 'bags':
        df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    else:
        df = pd.read_csv(csv_path)
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(
            f"fields not in item_features.csv: {missing}; available: {list(df.columns)}")
    code_to_fv = []
    for field in fields:
        if layout == 'bags':
            _, vocab = _bag_field_tokens(df[field].tolist(), field, multi_value_sep)
        else:
            _, vocab = _fixed_field_vocab(df, field)
        code_to_fv.extend((field, v) for v in vocab)
    return code_to_fv


def build_field_index(data_path: str, fields, layout: str = 'fixed', multi_value_sep: str = '|'):
    """dc06 (fI′ field-mean centring): global vocab code → field position, as a LongTensor
    ``(n_features,)`` with entries in ``0..F-1`` (F = len(fields), in the given order).

    Built on top of ``build_code_to_field_value`` — the single source of truth for the
    builders' vocab order (F-DC05-13) — so the mapping CANNOT drift from the codes that
    ``build_feature_ids`` / ``build_feature_bags`` produce on either layout.
    """
    code_to_fv = build_code_to_field_value(data_path, fields, layout, multi_value_sep)
    field_pos = {f: j for j, f in enumerate(fields)}
    return torch.tensor([field_pos[f] for f, _ in code_to_fv], dtype=torch.long)


def build_item_interaction_counts(data_path: str):
    """dc06 (fI′ ``center_mode='interaction'``): per-item train-interaction counts as a
    FloatTensor ``(n_items,)`` from the split directory's OWN ``listening_history_train.csv``.

    Leakage rule (dc05 M6.1a discipline, binding): consumes exactly ``data_path``'s train
    file — cold/variant seams rebuild from the VARIANT train file. Counting convention
    matches ``build_user_history_weights``: distinct ``(user_id, item_id)`` pairs,
    dedup keep-first — exactly the pairs behind the binary training matrix. Items absent
    from the train file get count 0 (they then simply contribute nothing to the
    interaction-weighted field mean).
    """
    csv_path = os.path.join(data_path, 'item_features.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'item_features.csv not found at {csv_path}')
    n_items = len(pd.read_csv(csv_path, usecols=['item_id']))

    train_csv = os.path.join(data_path, 'listening_history_train.csv')
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f'listening_history_train.csv not found at {train_csv}')
    pairs = pd.read_csv(train_csv, usecols=['user_id', 'item_id'])
    pairs = pairs.drop_duplicates(subset=['user_id', 'item_id'], keep='first')

    item_idx = pairs['item_id'].to_numpy()
    if len(item_idx) and (item_idx.min() < 0 or item_idx.max() >= n_items):
        raise ValueError(
            f'listening_history_train.csv references item_id outside 0..{n_items - 1} '
            f'(min={item_idx.min()}, max={item_idx.max()})')
    counts = torch.zeros(n_items, dtype=torch.float32)
    vc = pairs['item_id'].value_counts()
    counts[torch.tensor(vc.index.to_numpy(), dtype=torch.long)] = \
        torch.tensor(vc.to_numpy(), dtype=torch.float32)
    return counts


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
        # Shared vocab recipe (F-DC05-13): the same function serves build_code_to_field_value,
        # so read-out labels can never drift from these codes.
        vals, vocab = _fixed_field_vocab(df, field)
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


def build_feature_bags(data_path: str, fields, multi_value_sep: str = '|'):
    """Build variable-size indicator-bag item-feature tensors from ``<data_path>/item_features.csv``
    (S0-build extension, charter C5 ml-1m amendment: genres + Tag-Genome tags @0.8 as bags).

    Bag semantics (S0.5-addendum gate decision A — RAW indicator bags, B5 §2.2, "the way LightFM
    did it"): each cell may hold zero or more values joined by ``multi_value_sep``; every token
    weighs exactly 1.0 in the composition q_i = Σ e_token, so an item's vote mass is its token
    count (the popularity-coupling channel this implies on ml-1m is a standing must-mention —
    S0.5 addendum §6). Vocabulary convention identical to ``build_feature_ids``: per-field
    lexicographically-sorted local vocab + running global offset; ``n_features`` = Σ per-field
    vocab sizes; ID rows are appended downstream by ``FeatureEmbedding``. Cell content is taken
    verbatim (split on the separator only, no stripping) so that on one-value-per-field data the
    vocab is bit-identical to ``build_feature_ids``'s.

    Padded layout (``HistoryFeatureEmbedding`` convention): width D_max = max total token count
    over items; padding slots hold id 0 / weight 0.0 and contribute exactly nothing to the sum.

    Missingness policy (S0.5 addendum §3, ratified): an EMPTY cell ('') is a legal SHORT BAG for
    that field (the 3.0% genres-only ml-1m tail); an item with zero tokens across ALL fields is
    rejected loudly — no such item exists on the in-scope data, and the all-zero composition is
    degenerate under cosine. F-S0-06 discipline: the CSV is read with ``keep_default_na=False``
    (so '' stays a string and a float NaN can never enter), and any token that lowercases to
    'nan' is rejected — the phantom-vocab-value scenario the F-S0-06 guards exist for. Duplicate
    tokens inside one cell are rejected (corruption signal — an item carries a value once).

    Fixed-width equivalence (keystone dc_checks/s0/t07): on a one-value-per-field CSV (H&M) this
    builder returns exactly ``build_feature_ids``'s tensor plus an all-ones weight matrix
    (D_max = F, no padding).

    :param data_path: directory containing ``item_features.csv``.
    :param fields: ordered list of column names to encode as bag fields. ``fields == []`` is the
        F=0 reduction: returns ``(n_items, 0)`` tensors (composition = ID row only).
    :param multi_value_sep: the in-cell separator (the explanations naming layer's ml-1m
        convention, ``naming/config.py`` ``multi_value_sep='|'``).
    :return: ``(bag_value_ids: LongTensor (n_items, D_max), bag_weights: FloatTensor
        (n_items, D_max), n_features: int)`` — padding slots hold id 0 / weight 0.0.
    :raises ValueError: on the ``build_feature_ids`` guards (item_id coverage, missing field),
        a 'nan' token, a duplicate token within a cell, or an all-fields-empty item.
    """
    csv_path = os.path.join(data_path, 'item_features.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'item_features.csv not found at {csv_path}')

    # keep_default_na=False + dtype=str: '' stays a string (legal short bag); float NaN cannot enter.
    df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    if 'item_id' not in df.columns:
        raise ValueError(f"item_features.csv at {csv_path} has no 'item_id' column")

    n_items = len(df)
    df['item_id'] = df['item_id'].astype(int)  # dtype=str read — sort/guard must be numeric
    # Alignment guard: item_id must cover 0..n_items-1 exactly (no missing/extra/duplicate).
    item_id_sorted = sorted(int(x) for x in df['item_id'].tolist())
    if item_id_sorted != list(range(n_items)):
        raise ValueError(
            f"item_id must cover 0..{n_items - 1} exactly; got "
            f"min={item_id_sorted[0]}, max={item_id_sorted[-1]}, "
            f"n_unique={len(set(item_id_sorted))}, n_rows={n_items}")

    # Align rows to item_id order so bag row i is item i (rows may be shuffled on disk).
    df = df.sort_values('item_id').reset_index(drop=True)

    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(
            f"fields not in item_features.csv: {missing}; available: {list(df.columns)}")

    token_lists = [[] for _ in range(n_items)]  # global-offset token ids per item, field order
    offset = 0
    for field in fields:
        # Shared vocab recipe (F-DC05-13): the same function serves build_code_to_field_value,
        # so read-out labels can never drift from these codes.
        per_item, vocab = _bag_field_tokens(df[field].tolist(), field, multi_value_sep)
        code = {v: k for k, v in enumerate(vocab)}
        for i, toks in enumerate(per_item):
            token_lists[i].extend(code[t] + offset for t in toks)
        offset += len(code)
    n_features = offset

    if fields:
        empty = [i for i, tl in enumerate(token_lists) if not tl]
        if empty:
            raise ValueError(
                f"{len(empty)} item(s) have ZERO tokens across all fields {list(fields)} "
                f"(first ids: {empty[:5]}) — the all-empty composition is degenerate under "
                f"cosine; no such item exists on the in-scope data (S0.5 addendum §2)")

    d_max = max((len(tl) for tl in token_lists), default=0) if fields else 0
    bag_value_ids = torch.zeros((n_items, d_max), dtype=torch.long)
    bag_weights = torch.zeros((n_items, d_max), dtype=torch.float32)
    for i, tl in enumerate(token_lists):
        if tl:
            bag_value_ids[i, :len(tl)] = torch.tensor(tl, dtype=torch.long)
            bag_weights[i, :len(tl)] = 1.0

    return bag_value_ids, bag_weights, n_features


def build_user_history_weights(data_path: str, fields, layout: str = 'fixed',
                               multi_value_sep: str = '|'):
    """Build the dc05 (``feature_user_proto``) padded per-user history-weight tensors from the
    split directory's OWN ``listening_history_train.csv`` + ``item_features.csv``.

    Leakage rule (design doc M6.1a, binding): this builder consumes exactly ``data_path``'s train
    file — on cold/variant datasets the weights are therefore rebuilt from the VARIANT train file
    and removed rows can never leak into w̄ (the injection seams in trainer/tester/cold_eval/loader
    all pass their own directory).

    Encoding: the metadata vocabulary is the item builders' verbatim (same fields, same
    deterministic field-offset encoding) — labels stay renderer-compatible and the word ids are
    shared with the item-side models. Aggregation (design doc §3.1): H_u = the user's train-window
    purchase set deduplicated on (user_id, item_id) keep-first — exactly the distinct pairs behind
    the binary training matrix; n_{u,f} = how many items of H_u carry attribute value f;
    w̄_{u,f} = n_{u,f} / |H_u| (mean over purchases). Users absent from the train file (possible
    on variants only) get an all-zero row.

    Two layouts (S0-build extension component b, charter C5 ml-1m amendment):
    - ``layout='fixed'`` (default, the original dc05/H&M path, unchanged): item vocabulary from
      ``build_feature_ids`` — exactly one value per field per item.
    - ``layout='bags'``: item vocabulary from ``build_feature_bags`` — items carry variable-size
      indicator bags; every TOKEN of every purchase enters the count (raw bags, S0.5-addendum
      gate decision A), so n_{u,f} counts the items of H_u carrying token f and the per-purchase
      vote mass is the purchase's token count (the ml-1m popularity-coupling must-mention). The
      aggregation rule is otherwise identical; on one-value-per-field data the two layouts are
      BIT-IDENTICAL (keystone dc_checks/s0/t08).

    :param data_path: split directory (train CSV + item_features.csv + user_ids.csv).
    :param fields: ordered list of item_features.csv column names (the canonical field set).
    :param layout: ``'fixed'`` or ``'bags'`` (see above).
    :param multi_value_sep: in-cell separator for ``layout='bags'``.
    :return: ``(hist_value_ids: LongTensor (n_users, D_max), hist_weights: FloatTensor
        (n_users, D_max), n_features: int)`` — padding slots hold id 0 / weight 0.0.
    :raises ValueError: on the item-builder guards, an unknown layout, or if the train file
        references a user_id outside 0..n_users-1 or an item_id outside 0..n_items-1.
    """
    if layout not in ('fixed', 'bags'):
        raise ValueError(f"layout must be 'fixed' or 'bags'; got {layout!r}")
    if layout == 'bags':
        item_bag_ids, item_bag_w, n_features = build_feature_bags(data_path, fields, multi_value_sep)
        n_items = item_bag_ids.shape[0]
    else:
        feature_ids, n_features = build_feature_ids(data_path, fields)
        n_items = feature_ids.shape[0]

    users_csv = os.path.join(data_path, 'user_ids.csv')
    if not os.path.exists(users_csv):
        raise FileNotFoundError(f'user_ids.csv not found at {users_csv}')
    n_users = len(pd.read_csv(users_csv))

    train_csv = os.path.join(data_path, 'listening_history_train.csv')
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f'listening_history_train.csv not found at {train_csv}')
    pairs = pd.read_csv(train_csv, usecols=['user_id', 'item_id'])
    # F-DC05-11 guard: a NaN user_id row would otherwise be SILENTLY dropped by the groupbys
    # below (pandas drops NaN keys) — the purchase vanishes from w̄ with no error; a NaN item_id
    # would raise only accidentally (opaque IndexError from the long cast). Fail loud instead.
    n_nan = int(pairs['user_id'].isna().sum() + pairs['item_id'].isna().sum())
    if n_nan:
        raise ValueError(
            f"train file {train_csv} has {n_nan} NaN/missing user_id/item_id cell(s) — "
            f"malformed rows must not silently vanish from the history weights (F-DC05-11)")
    # The splitter already dedups (customer, article) keep-first; dedup defensively on the same
    # key so w̄ counts exactly the distinct pairs the binary training matrix sees.
    pairs = pairs.drop_duplicates(subset=['user_id', 'item_id'], keep='first')

    if len(pairs) > 0:
        if pairs['user_id'].min() < 0 or pairs['user_id'].max() >= n_users:
            raise ValueError(
                f"train file references user_id outside 0..{n_users - 1} "
                f"(min={pairs['user_id'].min()}, max={pairs['user_id'].max()})")
        if pairs['item_id'].min() < 0 or pairs['item_id'].max() >= n_items:
            raise ValueError(
                f"train file references item_id outside 0..{n_items - 1} "
                f"(min={pairs['item_id'].min()}, max={pairs['item_id'].max()})")

    n_fields = len(fields)
    if len(pairs) == 0 or n_fields == 0:
        # Degenerate but well-defined: no metadata mass anywhere (all-zero weight rows).
        return (torch.zeros((n_users, 1), dtype=torch.long),
                torch.zeros((n_users, 1), dtype=torch.float32),
                n_features)

    # Long form: one row per (user, purchased item, token) → attribute-value id.
    item_idx = torch.as_tensor(pairs['item_id'].to_numpy(), dtype=torch.long)
    if layout == 'bags':
        # Variable token counts per purchase: flatten the padded item bags row-major and keep
        # only real slots (weight > 0); repeat each user by their purchase's token count.
        vals = item_bag_ids[item_idx]  # (P, D_max_item)
        real = item_bag_w[item_idx] > 0  # (P, D_max_item)
        long_df = pd.DataFrame({
            'u': pd.Series(pairs['user_id'].to_numpy()).repeat(
                real.sum(dim=1).numpy()).to_numpy(),
            'v': vals[real].numpy(),
        })
    else:
        item_vals = feature_ids[item_idx]  # (P, F)
        long_df = pd.DataFrame({
            'u': pd.Series(pairs['user_id'].to_numpy()).repeat(n_fields).to_numpy(),
            'v': item_vals.reshape(-1).numpy(),
        })
    counts = long_df.groupby(['u', 'v']).size().reset_index(name='n')  # n_{u,f}
    basket_sizes = pairs.groupby('user_id').size()  # |H_u| (distinct purchased articles)

    counts['col'] = counts.groupby('u').cumcount()
    d_max = int(counts['col'].max()) + 1

    hist_value_ids = torch.zeros((n_users, d_max), dtype=torch.long)
    hist_weights = torch.zeros((n_users, d_max), dtype=torch.float32)
    u_idx = torch.as_tensor(counts['u'].to_numpy(), dtype=torch.long)
    c_idx = torch.as_tensor(counts['col'].to_numpy(), dtype=torch.long)
    hist_value_ids[u_idx, c_idx] = torch.as_tensor(counts['v'].to_numpy(), dtype=torch.long)
    hist_weights[u_idx, c_idx] = torch.as_tensor(
        (counts['n'] / basket_sizes.loc[counts['u']].to_numpy()).to_numpy(), dtype=torch.float32)

    return hist_value_ids, hist_weights, n_features


def _checked_fields(spec: dict):
    """Injection-time guard: ``feature_fields`` must be a literal list here. The ``'canonical'``
    sentinel is resolved pre-Ray by ``resolve_feature_fields_in_conf`` (start_hyper); reaching
    injection unresolved means a hand-built config skipped resolution — fail loud."""
    fields = spec['feature_fields']
    if isinstance(fields, str):
        raise ValueError(
            f"feature_fields is the unresolved sentinel {fields!r} — resolution runs in "
            f"experiment_helper.start_hyper (resolve_feature_fields_in_conf); a hand-built "
            f"config must carry a literal field list")
    return fields


def _checked_layout(spec: dict):
    """Injection-time guard: ``feature_layout`` absent (legacy/saved configs) defaults to
    'fixed'; anything else must be a known layout."""
    layout = spec.get('feature_layout', 'fixed')
    if layout not in ('fixed', 'bags'):
        raise ValueError(f"feature_layout must be 'fixed' or 'bags'; got {layout!r}")
    return layout


def inject_feature_ids(ft_ext_param: dict, data_path: str) -> dict:
    """Guarded injection seam for ``trainer._build_model`` / ``tester._build_model``.

    Returns the ``ft_ext_param`` the factory should consume.

    For ``ft_type == 'feature_item_proto'`` (dc01) and ``'lightfm'`` (the S0.4 CBF baseline —
    same FeatureEmbedding payload) it returns a SHALLOW copy whose ``item_ft_ext_param``
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

    For ``ft_type == 'feature_user_proto'`` (dc05) and ``'lightfm_hist'`` (the dc05 S1
    decoupled control — same HistoryFeatureEmbedding payload) the payload lands on the USER
    side instead:
    the config carries only the lightweight spec (``feature_fields`` / ``use_id_feature``, dc01
    key names on purpose); the user sub-dict copy gains the freshly-built ``hist_value_ids`` +
    ``hist_weights`` (n_users, D_max) tensors + ``n_features``. Because every seam passes its own
    directory, cold/variant paths rebuild the weights from the VARIANT train file (leakage rule).

    For every other ``ft_type`` the SAME object is returned unchanged, so the factory mutates it in
    place exactly as before (no behavioural change for existing models).
    """
    ft_type = ft_ext_param.get('ft_type')
    user_spec = None

    if ft_type in ('feature_item_proto', 'lightfm'):
        item_spec = dict(ft_ext_param['item_ft_ext_param'])
        fields = _checked_fields(item_spec)
        layout = _checked_layout(item_spec)
        if layout == 'bags':
            feature_ids, feature_weights, n_features = build_feature_bags(data_path, fields)
            item_spec['feature_weights'] = feature_weights
        else:
            feature_ids, n_features = build_feature_ids(data_path, fields)
        item_spec['feature_ids'] = feature_ids
        item_spec['n_features'] = n_features
        # dc06 fI′ centring payloads (only when the config asks — 'lightfm' and plain fI
        # configs are untouched). Same leakage discipline: built from THIS seam's data_path.
        if item_spec.get('center_fields', False):
            item_spec['field_index'] = build_field_index(data_path, fields, layout)
            if item_spec.get('center_mode', 'weighted') == 'interaction':
                item_spec['item_weights'] = build_item_interaction_counts(data_path)
    elif ft_type == 'attr_item_proto':
        item_spec = dict(ft_ext_param['item_ft_ext_param'])
        fields = item_spec['attr_fields']
        attr_multi_hot, field_offsets, _ = build_attr_multi_hot(data_path, fields)
        item_spec['attr_multi_hot'] = attr_multi_hot
        item_spec['n_attr_values'] = int(attr_multi_hot.shape[1])
        item_spec['field_offsets'] = field_offsets
    elif ft_type in ('feature_user_proto', 'lightfm_hist'):
        # 'lightfm_hist' (dc05 S1 decoupled control, F-DC05-21) carries the same USER-side
        # payload — identical builder, identical leakage discipline.
        user_spec = dict(ft_ext_param['user_ft_ext_param'])
        fields = _checked_fields(user_spec)
        layout = _checked_layout(user_spec)
        hist_value_ids, hist_weights, n_features = build_user_history_weights(
            data_path, fields, layout=layout)
        user_spec['hist_value_ids'] = hist_value_ids
        user_spec['hist_weights'] = hist_weights
        user_spec['n_features'] = n_features
        item_spec = dict(ft_ext_param['item_ft_ext_param'])
    else:
        return ft_ext_param

    new_param = dict(ft_ext_param)
    new_param['item_ft_ext_param'] = item_spec
    new_param['user_ft_ext_param'] = user_spec if user_spec is not None \
        else dict(ft_ext_param['user_ft_ext_param'])
    return new_param
