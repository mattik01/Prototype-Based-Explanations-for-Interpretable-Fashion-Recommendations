"""Accessor for feature_user_item_proto (dc09, fUfI-ProtoMF — the UI double tie with both base
tables replaced by the lineage's compositions).

Shape of user_item_proto (both prototype sets, both projection branches), with dc01's and
dc05's difference on BOTH sides: each side's base is a composition, read once and fed to its two
roles by a TwoRoleComposedExtractor, so ``user_embeddings()`` / ``item_embeddings()`` return the
COMPOSED factors (q_u = basket-mean words + ID row, q_i = item words + ID row), built by forwards
over all objects, and both prototype sets can be grounded INTRINSICALLY in the word rows
(cos(e_f, p^u_l), cos(e_f, p^t_k)). With the shared word table (the default) both sides read the
SAME word rows; under the ``_sep`` arms each side has its own, so the word view is per side.

In ids arms the per-row read-outs carry the translation gauge of F-DC09-07 (words +v_φ, ID rows
−Σ_φ v_φ). ``feature_value_embeddings`` returns the RAW rows; ``naming_word_rows`` returns what the
intrinsic naming reads — the canonical per-field-centred rows where that side has an ID row
(SC.5), the raw rows otherwise.
"""
import numpy as np
import torch

from utilities.explanations.accessor.base import ProtoAccessor
from utilities.explanations.gauge import centred_word_table

_BATCH = 8192


class FeatureUserItemProtoAccessor(ProtoAccessor):
    model_type = "feature_user_item_proto"
    has_user_prototypes = True
    has_item_prototypes = True
    has_projections = True
    has_intrinsic_item_grounding = True  # item prototypes share the word-embedding space
    has_intrinsic_user_grounding = True  # user prototypes share it too

    def __init__(self, model):
        super().__init__(model)
        # both sides: TwoRoleComposedExtractor(base = composition, proto = PrototypeEmbedding)
        self._user_fe = model.user_feature_extractor
        self._item_fe = model.item_feature_extractor
        # the prototype layers, under the UI-host accessor's names (weight_viz reads their affine)
        self._user_proto_fe = self._user_fe.proto
        self._item_proto_fe = self._item_fe.proto

    def user_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._user_fe.proto.prototypes)

    def item_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._item_fe.proto.prototypes)

    @torch.no_grad()
    def _batched(self, fn, n: int) -> np.ndarray:
        # batched so the (n, D_max, d) gathers of the compositions never materialize at once
        return self._as_numpy(torch.cat([fn(torch.arange(s, min(s + _BATCH, n)))
                                         for s in range(0, n, _BATCH)], dim=0))

    def user_embeddings(self) -> np.ndarray:
        # composed q_u for every user (full train basket — leave-one-out is training-only)
        return self._batched(self._user_fe.base, self.n_users)

    def item_embeddings(self) -> np.ndarray:
        return self._batched(self._item_fe.base, self.n_items)

    @torch.no_grad()
    def user_proj_to_item_proto_space(self, user_ids: torch.Tensor) -> np.ndarray:
        """û = W^u q_u per user — shape (len(user_ids), n_item_protos)."""
        return self._as_numpy(self._user_fe.linear_layer(self._user_fe.base(user_ids)))

    @torch.no_grad()
    def item_proj_to_user_proto_space(self, item_ids: torch.Tensor) -> np.ndarray:
        """t̂ = W^t q_i per item — shape (len(item_ids), n_user_protos)."""
        return self._as_numpy(self._item_fe.linear_layer(self._item_fe.base(item_ids)))

    def items_in_user_proto_space(self) -> np.ndarray:
        return self._batched(lambda idx: self._item_fe.linear_layer(self._item_fe.base(idx)), self.n_items)

    # --- intrinsic word grounding (both sides) ---

    @property
    def shares_word_table(self) -> bool:
        return self._item_fe.base.embedding_layer.weight is self._user_fe.base.embedding_layer.weight

    def feature_value_embeddings(self, side: str = "item") -> np.ndarray:
        """The learned word rows e_f of one side — shape (n_features, d), the field-offset code
        order shared by both builders. Identical for both sides under the shared table."""
        base = {"item": self._item_fe.base, "user": self._user_fe.base}[side]
        return self._as_numpy(base.embedding_layer.weight[:base.n_features])

    def user_feature_value_embeddings(self) -> np.ndarray:
        return self.feature_value_embeddings("user")

    def has_id_row(self, side: str) -> bool:
        return bool({"item": self._item_fe.base, "user": self._user_fe.base}[side].use_id_feature)

    def naming_word_rows(self, side: str, field_of_code: torch.LongTensor, n_fields: int) -> np.ndarray:
        """The word rows intrinsic naming reads, cos(e_f, p): in the canonical per-field gauge
        (e_f − field vocabulary mean) where this side's composition has an ID row — there the raw
        cosine is set by weight decay's tie-break, not the model (F-DC09-07); raw otherwise."""
        rows = torch.as_tensor(self.feature_value_embeddings(side))
        if self.has_id_row(side):
            rows = centred_word_table(rows, field_of_code, n_fields)
        return rows.numpy()

    @property
    def n_feature_values(self) -> int:
        return int(self._item_fe.base.n_features)
