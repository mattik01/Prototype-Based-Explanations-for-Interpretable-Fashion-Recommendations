"""Canonical per-field-centred read-out gauge (dc09 SC.5 — F-DC09-07, design doc 3b-xi).

A composition with an ID row, q = Σ_f w_f·e_f + e_ID, has a translation gauge: shifting every
word row of field φ by v_φ and each object's ID row by −Σ_φ (its field-φ weight)·v_φ leaves q
unchanged — and so every score, loss and inclusion regulariser (exact on the fixed
one-word-per-field layout; on bags exact at inference, broken in training only by the user
side's leave-one-out). Per-row read-outs are not invariant under it: per-word shares, the
intercept zoom, the ID line, the named coverage and the naming profiles cos(e_f, p) of the RAW
table report weight decay's tie-break, not the model.

The canonical gauge reads every word row as its deviation from its field's vocabulary mean,
e_f − m_φ (the unweighted average over the field's vocabulary rows — the stated convention),
and moves the removed mass into the ID line, e_ID + Σ_φ (field-φ weight)·m_φ, read "specific +
field baselines". The rows still sum to q exactly, and the centred numbers are gauge-invariant
functions of the trained model. A composition WITHOUT an ID row has no gauge (nothing absorbs a
shift) and is read raw. Read-out time only: nothing here touches training.
"""
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import torch


def field_of_codes(feature_fields: Sequence[str],
                   code_to_field_value: Optional[List[Tuple[str, str]]] = None,
                   items_info: Optional[pd.DataFrame] = None) -> torch.LongTensor:
    """Global word code → field position ``(V,)``, from the builder-owned
    ``code_to_field_value`` (F-DC05-13). Without it, the fixed-layout recipe is rebuilt from
    ``items_info`` (per field: sorted unique values — the builders' own order)."""
    pos = {f: j for j, f in enumerate(feature_fields)}
    if code_to_field_value is not None:
        return torch.tensor([pos[f] for f, _ in code_to_field_value], dtype=torch.long)
    if items_info is None:
        raise ValueError("field_of_codes needs code_to_field_value or items_info")
    out = []
    for f in feature_fields:
        out += [pos[f]] * int(items_info[f].astype(str).nunique())
    return torch.tensor(out, dtype=torch.long)


def field_means(word_rows: torch.Tensor, field_of_code: torch.LongTensor,
                n_fields: int) -> torch.Tensor:
    """``(n_fields, d)`` vocabulary mean of each field's word rows (unweighted)."""
    sums = torch.zeros(n_fields, word_rows.shape[1], dtype=word_rows.dtype)
    sums.index_add_(0, field_of_code, word_rows)
    counts = torch.bincount(field_of_code, minlength=n_fields).clamp_min(1).to(word_rows.dtype)
    return sums / counts.unsqueeze(1)


def centred_word_table(word_rows: torch.Tensor, field_of_code: torch.LongTensor,
                       n_fields: int) -> torch.Tensor:
    """The word table in the canonical gauge, e_f − m_φ(f) — the naming source of ids arms."""
    return word_rows - field_means(word_rows, field_of_code, n_fields)[field_of_code]


def code_field_mass(codes: Sequence[int], weights: torch.Tensor,
                    field_of_code: torch.LongTensor, n_fields: int) -> torch.Tensor:
    """``(R, n_fields)``: row r is one word slot (code, weight) — its weight in its own field."""
    mass = torch.zeros(len(codes), n_fields, dtype=weights.dtype)
    if len(codes):
        mass[torch.arange(len(codes)), field_of_code[torch.as_tensor(list(codes))]] = weights
    return mass


def centre_rows(word_rows: torch.Tensor, field_mass: torch.Tensor,
                means: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centre a composition's word rows; return ``(centred rows (R, d), ID-line shift (d,))``.

    :param word_rows: (R, d) the composition's word part, one row per reading unit (a word
        slot, or a purchase's word sum) — rows sum to q − e_ID.
    :param field_mass: (R, n_fields) per row, the summed token weight of each field inside it.
    :param means: (n_fields, d) this table's field means (:func:`field_means`).
    Centred rows + (e_ID + shift) sum to q exactly.
    """
    removed = field_mass @ means
    return word_rows - removed, removed.sum(dim=0)
