"""Accessor for feature_user_proto (dc05, fU host — the U-ProtoMF mirror of dc01's fI).

Same shape as user_proto (user side = PrototypeEmbedding, item side = plain Embedding in
R^{K_u}), with ONE difference that matters to the explainers: the user branch's base embedding
is a HistoryFeatureEmbedding, so ``user_embeddings()`` returns the COMPOSED per-user factor
q_u = Σ_f w̄_f·e_f (+ ID row) (built by a forward over all users), not a free per-user table. It
additionally exposes the feature-value embedding table (the e_f word rows), which is what lets
USER prototypes be grounded INTRINSICALLY (cos(e_f, p^u_l) — taste-community profiles in item
vocabulary, design doc §3.4 read-out 4) rather than through the host's synthetic-user procedure.

There are no item prototypes and no projection branches on this host (the double-tie surfaces
return at the lineage's fUfI merge stage). The item side appears in explanations as the free
per-community coefficients t_l — CF-learned, rendered under the gauge-disclosure discipline
(3b-i; thesis hidden-effects section, not the figures).
"""
import numpy as np
import torch

from utilities.explanations.accessor.base import ProtoAccessor


class FeatureUserProtoAccessor(ProtoAccessor):
    model_type = "feature_user_proto"
    has_user_prototypes = True
    has_item_prototypes = False
    has_projections = False
    has_intrinsic_user_grounding = True  # user prototypes share the word-embedding space

    def __init__(self, model):
        super().__init__(model)
        # user side: PrototypeEmbedding(embedding_ext=HistoryFeatureEmbedding); item side:
        # Embedding (n_items, K_u)
        self._user_proto_fe = model.user_feature_extractor           # PrototypeEmbedding
        self._user_hist_embed = self._user_proto_fe.embedding_ext    # HistoryFeatureEmbedding

    def user_prototypes(self) -> np.ndarray:
        return self._as_numpy(self._user_proto_fe.prototypes)

    def item_prototypes(self):
        return None

    @torch.no_grad()
    def user_embeddings(self) -> np.ndarray:
        # Composed user factor q_u for every user (mean of basket word embeddings, + ID row).
        # Batched so the (n_users, D_max, d) gather never materializes at once.
        outs = []
        for start in range(0, self.n_users, 8192):
            idx = torch.arange(start, min(start + 8192, self.n_users))
            outs.append(self._user_hist_embed(idx))
        return self._as_numpy(torch.cat(outs, dim=0))

    def item_embeddings(self) -> np.ndarray:
        # The free item vector t in user-prototype-similarity space — shape (n_items, K_u).
        return self._as_numpy(self.model.item_feature_extractor.embedding_layer.weight)

    def items_in_user_proto_space(self) -> np.ndarray:
        return self.item_embeddings()

    # --- intrinsic word grounding (the dc05 addition, mirror of dc01's) ---

    def feature_value_embeddings(self) -> np.ndarray:
        """The learned metadata word embeddings e_f — shape (n_features, d).

        Excludes the per-user ID rows (those occupy indices n_features .. n_features+n_users-1).
        Row order matches the field-offset code order shared with build_feature_ids, so it
        aligns with the deterministic per-field sorted vocabulary.
        """
        nf = self._user_hist_embed.n_features
        return self._as_numpy(self._user_hist_embed.embedding_layer.weight[:nf])

    @property
    def n_feature_values(self) -> int:
        return int(self._user_hist_embed.n_features)
