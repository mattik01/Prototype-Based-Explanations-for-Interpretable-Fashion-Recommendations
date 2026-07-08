from rec_sys.rec_sys import RecSys

from utilities.explanations.accessor.base import ProtoAccessor
from utilities.explanations.accessor.attr_item_proto import AttrItemProtoAccessor
from utilities.explanations.accessor.feature_item_proto import FeatureItemProtoAccessor
from utilities.explanations.accessor.item_proto import ItemProtoAccessor
from utilities.explanations.accessor.user_item_proto import UserItemProtoAccessor
from utilities.explanations.accessor.user_proto import UserProtoAccessor

_ACCESSORS = {
    "item_proto": ItemProtoAccessor,
    "user_proto": UserProtoAccessor,
    "user_item_proto": UserItemProtoAccessor,
    "feature_item_proto": FeatureItemProtoAccessor,
    "attr_item_proto": AttrItemProtoAccessor,
}


def get_accessor(model_type: str, model: RecSys) -> ProtoAccessor:
    if model_type not in _ACCESSORS:
        raise ValueError(
            f"No accessor registered for model_type={model_type!r}. "
            f"Registered: {sorted(_ACCESSORS.keys())}"
        )
    return _ACCESSORS[model_type](model)


__all__ = ["ProtoAccessor", "get_accessor"]
