"""Accessor for feature_item_proto (dc01).

Same weight-tied double branch as user_item_proto, with ONE difference that matters to the
explainers: the item branch's base embedding is a FeatureEmbedding, so ``item_embeddings()``
returns the COMPOSED per-item factor q_i = Σ_f e_f (built by a forward over all items), not a
free per-item table. It additionally exposes the feature-value embedding table (the e_f rows),
which is what lets prototypes be grounded INTRINSICALLY (cos(e_f, p_k)) rather than post-hoc.
"""
import numpy as np
import torch

from utilities.explanations.accessor.base import ProtoAccessor


class FeatureItemProtoAccessor(ProtoAccessor):
    model_type = "feature_item_proto"
    has_user_prototypes = True
    has_item_prototypes = True
    has_projections = True
    has_intrinsic_item_grounding = True  # item prototypes share the feature-embedding space

    def __init__(self, model):
        super().__init__(model)
        # user side: ConcatenateFeatureExtractors(PrototypeEmbedding, EmbeddingW, invert=False)
        # item side: ConcatenateFeatureExtractors(PrototypeEmbedding(embedding_ext=FeatureEmbedding),
        #                                          FeatureEmbeddingW, invert=True)
        self._user_proto_fe = model.user_feature_extractor.model_1   # PrototypeEmbedding
        self._user_proj_fe = model.user_feature_extractor.model_2    # EmbeddingW
        self._item_proto_fe = model.item_feature_extractor.model_1   # PrototypeEmbedding
        self._item_proj_fe = model.item_feature_extractor.model_2    # FeatureEmbeddingW
        self._item_feat_embed = self._item_proto_fe.embedding_ext    # FeatureEmbedding (shared)

    def user_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._user_proto_fe.prototypes)

    def item_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._item_proto_fe.prototypes)

    def user_embeddings(self) -> np.ndarray:
        return self._as_numpy(self._user_proto_fe.embedding_ext.embedding_layer.weight)

    @torch.no_grad()
    def item_embeddings(self) -> np.ndarray:
        # Composed item factor q_i for every item (sum of feature-value embeddings, + ID row).
        return self._as_numpy(self._item_feat_embed(torch.arange(self.n_items)))

    @torch.no_grad()
    def user_proj_to_item_proto_space(self, user_ids: torch.Tensor) -> np.ndarray:
        return self._as_numpy(self._user_proj_fe(user_ids))

    @torch.no_grad()
    def item_proj_to_user_proto_space(self, item_ids: torch.Tensor) -> np.ndarray:
        return self._as_numpy(self._item_proj_fe(item_ids))

    @torch.no_grad()
    def items_in_user_proto_space(self) -> np.ndarray:
        return self.item_proj_to_user_proto_space(torch.arange(self.n_items))

    # --- intrinsic feature grounding (the dc01 addition) ---

    def feature_value_embeddings(self) -> np.ndarray:
        """The learned metadata feature-value embeddings e_f — shape (n_features, d).

        Excludes the per-item ID rows (those occupy indices n_features .. n_features+n_items-1).
        Row order matches feature_ids' field-offset code order, so it aligns with the
        deterministic per-field sorted vocabulary used by build_feature_ids.
        """
        nf = self._item_feat_embed.n_features
        return self._as_numpy(self._item_feat_embed.embedding_layer.weight[:nf])

    @property
    def n_feature_values(self) -> int:
        return int(self._item_feat_embed.n_features)
