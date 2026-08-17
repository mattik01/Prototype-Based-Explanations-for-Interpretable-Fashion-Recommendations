from rec_sys.rec_sys import RecSys

from utilities.explanations.accessor.base import ProtoAccessor
from utilities.explanations.accessor.attr_item_proto import AttrItemProtoAccessor
from utilities.explanations.accessor.feature_item_proto import FeatureItemProtoAccessor
from utilities.explanations.accessor.feature_user_proto import FeatureUserProtoAccessor
from utilities.explanations.accessor.item_proto import ItemProtoAccessor
from utilities.explanations.accessor.user_item_proto import UserItemProtoAccessor
from utilities.explanations.accessor.user_proto import UserProtoAccessor

_ACCESSORS = {
    "item_proto": ItemProtoAccessor,
    "user_proto": UserProtoAccessor,
    "user_item_proto": UserItemProtoAccessor,
    "feature_item_proto": FeatureItemProtoAccessor,
    "feature_user_proto": FeatureUserProtoAccessor,
    "attr_item_proto": AttrItemProtoAccessor,
}

# Ablation arms share their headline model's architecture surface (noid: no ID row;
# f0: feature channel ablated) — normalize their keys to the base model so the
# explanations layer reaches them (dc05 SC.4 note, decided at the SC.8 gate).
# Special-experiment arms (shifted-cosine/bias 2x2, vault 2026-08-17_1128) normalize
# the same way; longest suffix first so "_cosstd_bias" is not half-stripped by "_bias".
_ABLATION_SUFFIXES = ("_cosstd_bias", "_cosstd", "_bias", "_noid", "_f0")


def base_model_type(model_name: str) -> str:
    """Strip a trailing arm suffix (``_noid``, ``_f0``, ``_bias``, ``_cosstd``,
    ``_cosstd_bias``) off a model name."""
    for suffix in _ABLATION_SUFFIXES:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


def get_accessor(model_type: str, model: RecSys) -> ProtoAccessor:
    key = base_model_type(model_type)
    if key not in _ACCESSORS:
        raise ValueError(
            f"No accessor registered for model_type={model_type!r}. "
            f"Registered: {sorted(_ACCESSORS.keys())}"
        )
    return _ACCESSORS[key](model)


__all__ = ["ProtoAccessor", "base_model_type", "get_accessor"]
