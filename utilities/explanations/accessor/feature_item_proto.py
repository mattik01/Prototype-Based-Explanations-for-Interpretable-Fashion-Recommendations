"""Accessor for feature_item_proto (dc01, fI host — re-pointed at the SC.3 re-run 2026-07-12).

Same shape as item_proto (item side = PrototypeEmbedding, user side = plain Embedding in
R^{K_t}), with ONE difference that matters to the explainers: the item branch's base embedding
is a FeatureEmbedding, so ``item_embeddings()`` returns the COMPOSED per-item factor
q_i = Σ_f e_f (built by a forward over all items), not a free per-item table. It additionally
exposes the feature-value embedding table (the e_f rows), which is what lets item prototypes be
grounded INTRINSICALLY (cos(e_f, p_k)) rather than post-hoc.

There are no user prototypes and no projection branches on this host (the double-tie surfaces
return at the lineage's fUfI merge stage). The user side appears in explanations only as the
per-prototype affinity coefficients u_k — CF-learned, rendered under the F-DC01-08 disclosure
discipline (SC.5 work, not accessor work).
"""
import numpy as np
import torch

from utilities.explanations.accessor.base import ProtoAccessor


class FeatureItemProtoAccessor(ProtoAccessor):
    model_type = "feature_item_proto"
    has_user_prototypes = False
    has_item_prototypes = True
    has_projections = False
    has_intrinsic_item_grounding = True  # item prototypes share the feature-embedding space

    def __init__(self, model):
        super().__init__(model)
        # user side: Embedding (n_users, K_t); item side: PrototypeEmbedding(embedding_ext=FeatureEmbedding)
        self._item_proto_fe = model.item_feature_extractor           # PrototypeEmbedding
        self._item_feat_embed = self._item_proto_fe.embedding_ext    # FeatureEmbedding

    def user_prototypes(self):
        return None

    def item_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._item_proto_fe.prototypes)

    def user_embeddings(self) -> np.ndarray:
        # The free user vector u in item-prototype-similarity space — shape (n_users, K_t).
        return self._as_numpy(self.model.user_feature_extractor.embedding_layer.weight)

    @torch.no_grad()
    def item_embeddings(self) -> np.ndarray:
        # Composed item factor q_i for every item (sum of feature-value embeddings, + ID row).
        return self._as_numpy(self._item_feat_embed(torch.arange(self.n_items)))

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
