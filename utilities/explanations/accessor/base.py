"""ProtoAccessor ABC: uniform view onto a trained RecSys for explainers.

Concrete subclasses (one per model variant) handle the per-variant wiring;
explainers stay variant-agnostic by asking the accessor capability questions.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from rec_sys.rec_sys import RecSys


class ProtoAccessor(ABC):
    model_type: str = ""
    has_user_prototypes: bool = False
    has_item_prototypes: bool = False
    has_projections: bool = False  # True only for user_item_proto (weight-tied double branch)

    def __init__(self, model: RecSys):
        self.model = model
        self.n_users = model.n_users
        self.n_items = model.n_items

    # --- prototypes / embeddings (numpy arrays or None) ---

    @abstractmethod
    def user_prototypes(self) -> Optional[np.ndarray]: ...

    @abstractmethod
    def item_prototypes(self) -> Optional[np.ndarray]: ...

    @abstractmethod
    def user_embeddings(self) -> np.ndarray: ...

    @abstractmethod
    def item_embeddings(self) -> np.ndarray: ...

    def items_in_user_proto_space(self) -> Optional[np.ndarray]:
        """Every item represented in user-prototype space — shape (n_items, n_user_protos).

        Used by the top-k explainer to find items most aligned with each user prototype
        (matches the ProtoMF paper's interpretation of user prototypes).
        Only applicable when the model has user prototypes.
        """
        return None

    # --- full object-to-prototype similarity matrices ---

    def user_to_user_proto_sim(self) -> Optional[np.ndarray]:
        if not self.has_user_prototypes:
            return None
        return self._shifted_cosine_sim(self.user_embeddings(), self.user_prototypes())

    def item_to_item_proto_sim(self) -> Optional[np.ndarray]:
        if not self.has_item_prototypes:
            return None
        return self._shifted_cosine_sim(self.item_embeddings(), self.item_prototypes())

    # --- projections (user_item_proto only) ---

    def user_proj_to_item_proto_space(self, user_ids: torch.Tensor) -> Optional[np.ndarray]:
        return None

    def item_proj_to_user_proto_space(self, item_ids: torch.Tensor) -> Optional[np.ndarray]:
        return None

    # --- shared helpers ---

    @staticmethod
    def _shifted_cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """(1 + cos(a, b)). Consumed only for RANKING purposes (post-hoc naming, top-k
        alignment), which are invariant under any of the model's cosine_types — all are
        increasing affine transforms of cos (cosine-offset generalization, cosbias2x2).
        Score-decomposing surfaces (breakdown) read the config's own affine instead.
        dc07 disclosure (2026-09-08): under the membership family (cosine_type softmax /
        sigmoid) the model ranks prototypes by the DOT product ⟨q, p_l⟩/τ, not by cos, so a
        prototype's top-k members here can differ from the model's own ordering when norms
        vary. Deliberately NOT changed — documented as a change of ranking unit."""
        a_t = torch.as_tensor(a, dtype=torch.float32)
        b_t = torch.as_tensor(b, dtype=torch.float32)
        a_n = F.normalize(a_t, dim=1)
        b_n = F.normalize(b_t, dim=1)
        return (1.0 + a_n @ b_n.T).cpu().numpy()

    @staticmethod
    def _as_numpy(param_or_tensor) -> np.ndarray:
        return param_or_tensor.detach().cpu().numpy()
