"""Accessor for attr_item_proto (dc02).

Same weight-tied double branch as user_item_proto on the USER side; the ITEM side has NO
per-item parameters: ``item_embeddings()`` returns the observed multi-hot X itself, and
``item_prototypes()`` returns the EFFECTIVE profile matrix A (softplus(Ã) when the K1
nonnegativity knob is on) — prototypes are grounded BY CONSTRUCTION (each column of A is a
named attribute value), which is what ``has_attr_space_prototypes`` signals to the pipeline.
The inherited ``item_to_item_proto_sim()`` (shifted cos of embeddings vs prototypes) therefore
computes exactly t* for every item — the explainers consume the true forward-pass quantity.
"""
import math
from typing import List

import numpy as np
import torch

from utilities.explanations.accessor.base import ProtoAccessor


class AttrItemProtoAccessor(ProtoAccessor):
    model_type = "attr_item_proto"
    has_user_prototypes = True
    has_item_prototypes = True
    has_projections = True
    has_intrinsic_item_grounding = True   # grounding needs no learned e_f — it holds by construction
    has_attr_space_prototypes = True      # pipeline routing switch (checked BEFORE the dc01 route)

    def __init__(self, model):
        super().__init__(model)
        # user side: ConcatenateFeatureExtractors(PrototypeEmbedding, EmbeddingW, invert=False)
        # item side: ConcatenateFeatureExtractors(AttributePrototypeEmbedding, AttributeProjection,
        #                                          invert=True) over ONE shared AttributeLookup
        self._user_proto_fe = model.user_feature_extractor.model_1   # PrototypeEmbedding
        self._user_proj_fe = model.user_feature_extractor.model_2    # EmbeddingW
        self._item_proto_fe = model.item_feature_extractor.model_1   # AttributePrototypeEmbedding
        self._item_proj_fe = model.item_feature_extractor.model_2    # AttributeProjection
        self._lookup = self._item_proto_fe.embedding_ext             # AttributeLookup (shared)

    # --- user side (identical wiring to user_item_proto / feature_item_proto) ---

    def user_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._user_proto_fe.prototypes)

    def user_embeddings(self) -> np.ndarray:
        return self._as_numpy(self._user_proto_fe.embedding_ext.embedding_layer.weight)

    @torch.no_grad()
    def user_proj_to_item_proto_space(self, user_ids: torch.Tensor) -> np.ndarray:
        return self._as_numpy(self._user_proj_fe(user_ids))          # û

    @torch.no_grad()
    def item_proj_to_user_proto_space(self, item_ids: torch.Tensor) -> np.ndarray:
        return self._as_numpy(self._item_proj_fe(item_ids))          # t̂

    @torch.no_grad()
    def items_in_user_proto_space(self) -> np.ndarray:
        return self.item_proj_to_user_proto_space(torch.arange(self.n_items))

    # --- item side (attribute space) ---

    def item_prototypes(self) -> np.ndarray:
        """EFFECTIVE A (K, V) — softplus(Ã) under the K1 knob. All read-outs consume this."""
        return self._as_numpy(self._item_proto_fe.effective_prototypes())

    def raw_item_prototypes(self) -> np.ndarray:
        """The stored parameter (Ã under K1, A otherwise). For diagnostics only."""
        return self._as_numpy(self._item_proto_fe.prototypes)

    def item_embeddings(self) -> np.ndarray:
        """The observed multi-hot X (n_items, V) — the item's ENTIRE representation."""
        return self._as_numpy(self._lookup.attr_multi_hot)

    # --- attribute-space grounding surface (the dc02 addition) ---

    @property
    def field_offsets(self) -> List[int]:
        return list(self._item_proto_fe.field_offsets)

    @property
    def n_attr_values(self) -> int:
        return int(self._lookup.n_attr_values)

    def user_prototype_attr_profiles(self) -> np.ndarray:
        """W_t rows (L_u, V): the learned attribute-response profile of each USER prototype
        (§3.4 pt. 4) — score-mediated/unenforced grounding, unlike A's by-construction one."""
        return self._as_numpy(self._item_proj_fe.linear_layer.weight)

    def attr_share_matrix(self) -> np.ndarray:
        """(V, K) EXACT-share score matrix: entry (v, k) = A[k, v] / (√F·‖a_k‖) — the additive
        share value v contributes to t*_k when an item has it (claim 3). This is the score
        matrix the dc02 naming route feeds to ``name_prototypes_from_score_matrix``."""
        prototypes = torch.as_tensor(self.item_prototypes())
        n_fields = len(self.field_offsets) - 1
        norms = prototypes.norm(dim=1).clamp_min(1e-8)               # (K,)
        shares = prototypes / (math.sqrt(n_fields) * norms.unsqueeze(1))  # (K, V)
        return shares.T.cpu().numpy()
