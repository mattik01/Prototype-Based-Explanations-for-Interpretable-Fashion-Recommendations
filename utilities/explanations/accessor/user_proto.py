"""Accessor for user_proto: user side has PrototypeEmbedding, item side is plain Embedding."""
from typing import Optional

import numpy as np

from utilities.explanations.accessor.base import ProtoAccessor


class UserProtoAccessor(ProtoAccessor):
    model_type = "user_proto"
    has_user_prototypes = True
    has_item_prototypes = False
    has_projections = False

    def user_prototypes(self) -> np.ndarray:
        return self._as_numpy(self.model.user_feature_extractor.prototypes)

    def item_prototypes(self) -> Optional[np.ndarray]:
        return None

    def user_embeddings(self) -> np.ndarray:
        return self._as_numpy(self.model.user_feature_extractor.embedding_ext.embedding_layer.weight)

    def item_embeddings(self) -> np.ndarray:
        # item_feature_extractor is Embedding; its dim equals n_user_prototypes by construction,
        # so the raw embedding already lives in user-proto space.
        return self._as_numpy(self.model.item_feature_extractor.embedding_layer.weight)

    def items_in_user_proto_space(self) -> np.ndarray:
        return self.item_embeddings()
