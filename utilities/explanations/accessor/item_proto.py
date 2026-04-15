"""Accessor for item_proto: item side has PrototypeEmbedding, user side is plain Embedding."""
from typing import Optional

import numpy as np

from utilities.explanations.accessor.base import ProtoAccessor


class ItemProtoAccessor(ProtoAccessor):
    model_type = "item_proto"
    has_user_prototypes = False
    has_item_prototypes = True
    has_projections = False

    def user_prototypes(self) -> Optional[np.ndarray]:
        return None

    def item_prototypes(self) -> np.ndarray:
        return self._as_numpy(self.model.item_feature_extractor.prototypes)

    def user_embeddings(self) -> np.ndarray:
        # user_feature_extractor is Embedding
        return self._as_numpy(self.model.user_feature_extractor.embedding_layer.weight)

    def item_embeddings(self) -> np.ndarray:
        # item_feature_extractor is PrototypeEmbedding
        return self._as_numpy(self.model.item_feature_extractor.embedding_ext.embedding_layer.weight)
