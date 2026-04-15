"""Accessor for user_item_proto: weight-tied double-branch (ConcatenateFeatureExtractors)."""
import numpy as np
import torch

from utilities.explanations.accessor.base import ProtoAccessor


class UserItemProtoAccessor(ProtoAccessor):
    model_type = "user_item_proto"
    has_user_prototypes = True
    has_item_prototypes = True
    has_projections = True

    def __init__(self, model):
        super().__init__(model)
        # user_feature_extractor: ConcatenateFeatureExtractors(PrototypeEmbedding, EmbeddingW, invert=False)
        # item_feature_extractor: same but invert=True
        self._user_proto_fe = model.user_feature_extractor.model_1   # PrototypeEmbedding
        self._user_proj_fe = model.user_feature_extractor.model_2    # EmbeddingW
        self._item_proto_fe = model.item_feature_extractor.model_1   # PrototypeEmbedding
        self._item_proj_fe = model.item_feature_extractor.model_2    # EmbeddingW

    def user_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._user_proto_fe.prototypes)

    def item_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._item_proto_fe.prototypes)

    def user_embeddings(self) -> np.ndarray:
        return self._as_numpy(self._user_proto_fe.embedding_ext.embedding_layer.weight)

    def item_embeddings(self) -> np.ndarray:
        # Tied with user_proj_fe's embedding on the user side; but here we return the item-side
        # PrototypeEmbedding's underlying embedding, which is tied to item_proj_fe's embedding.
        return self._as_numpy(self._item_proto_fe.embedding_ext.embedding_layer.weight)

    @torch.no_grad()
    def user_proj_to_item_proto_space(self, user_ids: torch.Tensor) -> np.ndarray:
        """Per-user projection onto item-prototype space via EmbeddingW (n_item_protos,)."""
        return self._as_numpy(self._user_proj_fe(user_ids))

    @torch.no_grad()
    def item_proj_to_user_proto_space(self, item_ids: torch.Tensor) -> np.ndarray:
        """Per-item projection onto user-prototype space via EmbeddingW (n_user_protos,)."""
        return self._as_numpy(self._item_proj_fe(item_ids))

    @torch.no_grad()
    def items_in_user_proto_space(self) -> np.ndarray:
        all_ids = torch.arange(self.n_items)
        return self.item_proj_to_user_proto_space(all_ids)
