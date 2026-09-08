from abc import abstractmethod, ABC

import torch
import torch.nn as nn

from utilities.utils import general_weight_init


class FeatureExtractor(nn.Module, ABC):
    """
    Abstract class representing one of the possible FeatureExtractor models. See also FeatureExtractorFactory.
    """

    def __init__(self):
        super().__init__()
        self.cumulative_loss = 0.
        self.name = "FeatureExtractor"

    def init_parameters(self):
        """
        Initial the Feature Extractor parameters
        """
        pass

    def get_and_reset_loss(self) -> float:
        """
        Reset the loss of the feature extractor and returns the computed value
        :return: loss of the feature extractor
        """
        loss = self.cumulative_loss
        self.cumulative_loss = 0.
        return loss

    @abstractmethod
    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        Performs the feature extraction process of the object.
        """
        pass


class Embedding(FeatureExtractor):
    """
    FeatureExtractor that represents an object (item/user) only given by its embedding.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, only_positive: bool = False):
        """
        Standard Embedding Layer
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param max_norm: max norm of the l2 norm of the embeddings.
        :param only_positive: whether the embeddings can be only positive
        """
        super().__init__()
        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.only_positive = only_positive
        self.name = "Embedding"

        self.embedding_layer = nn.Embedding(self.n_objects, self.embedding_dim, max_norm=self.max_norm)
        print(f'Built Embedding model \n'
              f'- n_objects: {self.n_objects} \n'
              f'- embedding_dim: {self.embedding_dim} \n'
              f'- max_norm: {self.max_norm}\n'
              f'- only_positive: {self.only_positive}')

    def init_parameters(self):
        self.embedding_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        assert o_idxs is not None, f"Object Indexes not provided! ({self.name})"
        embeddings = self.embedding_layer(o_idxs)
        if self.only_positive:
            embeddings = torch.absolute(embeddings)
        return embeddings


class EmbeddingW(Embedding):
    """
    FeatureExtractor that places a linear projection after an embedding layer. Used for sharing weights.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, out_dimension: int = None,
                 use_bias: bool = False):
        """
        :param n_objects: see Embedding
        :param embedding_dim: see Embedding
        :param max_norm: see Embedding
        :param out_dimension: Out dimension of the linear layer. If none, set to embedding_dim.
        :param use_bias: whether to use the bias in the linear layer.
        """
        super().__init__(n_objects, embedding_dim, max_norm)
        self.out_dimension = out_dimension
        self.use_bias = use_bias

        if self.out_dimension is None:
            self.out_dimension = embedding_dim

        self.name = 'EmbeddingW'
        self.linear_layer = nn.Linear(self.embedding_dim, self.out_dimension, bias=self.use_bias)

        print(f'Built Embeddingw model \n'
              f'- out_dimension: {self.out_dimension} \n'
              f'- use_bias: {self.use_bias} \n')

    def init_parameters(self):
        super().init_parameters()
        self.linear_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        o_embed = super().forward(o_idxs)
        return self.linear_layer(o_embed)


class FeatureEmbedding(FeatureExtractor):
    """
    FeatureExtractor that represents an object (item/user) as the SUM of its feature-value
    embeddings (LightFM / SVDFeature style): q_i = Σ_{f ∈ f_i} e_f. Optionally appends a
    per-object ID feature row ("tags + ids", LightFM §2.3 / Table 1) so the object also keeps a
    free per-object embedding term. With ``use_id_feature=True`` and zero metadata fields (F=0)
    this reduces EXACTLY to a plain per-object ``Embedding`` table — the dc01 F=0 equivalence
    keystone (see ``Master/temp/dc_checks/dc01``).

    Two layouts (S0-build extension, charter C5 ml-1m amendment):
    - **Fixed one-value-per-field** (``feature_weights=None``, the original dc01/H&M path,
      byte-identical to the pre-extension class): ``feature_ids`` holds exactly one vocab id per
      field, all summed unweighted.
    - **Weighted indicator bags** (``feature_weights`` given, from ``build_feature_bags``):
      ``feature_ids`` is a padded (n_objects, D_max) bag; ``q = Σ w·e_token`` with weight 1.0 on
      real tokens and 0.0 on padding slots, which therefore contribute exactly nothing. On
      padding-free all-ones inputs this path is BIT-IDENTICAL to the fixed path (keystone
      dc_checks/s0/t07). Padding uses id 0 (a real vocab row) — safe because its weight is
      exactly 0.0; under ``max_norm`` row 0 may be renorm-clamped on padded accesses, an
      idempotent no-op semantically (the ``HistoryFeatureEmbedding`` precedent, accepted dc05).
    """

    def __init__(self, n_objects: int, feature_ids: torch.LongTensor, n_features: int,
                 embedding_dim: int, use_id_feature: bool = True, max_norm: float = None,
                 feature_weights: torch.Tensor = None, center_fields: bool = False,
                 center_mode: str = 'weighted', center_detach: bool = False,
                 metadata_scale: float = 1.0, field_index: torch.LongTensor = None,
                 item_weights: torch.Tensor = None):
        """
        :param n_objects: number of objects in the system (items).
        :param feature_ids: LongTensor of shape (n_objects, F) — per-object feature-value ids with
            a global field-offset encoding (vocab = concatenated per-field vocabs). Registered as a
            NON-persistent buffer: it is rebuilt deterministically from item_features.csv at model
            build time and is never written to the checkpoint, so only the learned embedding weights
            round-trip (no shape-mismatch on load).
        :param n_features: size of the metadata feature vocabulary (Σ per-field vocab sizes). ID rows
            (when used) are indexed at n_features + o_idx, after the metadata rows.
        :param embedding_dim: embedding dimension d.
        :param use_id_feature: if True, append a per-object ID feature to every object's feature set.
        :param max_norm: max norm of the l2 norm of the embedding rows.
        :param feature_weights: optional Tensor of the same shape as ``feature_ids`` — per-token
            weights for the bag layout (1.0 real / 0.0 padding, from ``build_feature_bags``).
            None (default) = the fixed one-value-per-field layout, behaviour unchanged.
        :param center_fields: dc06 fI′ — subtract the per-field mean μ_f from every metadata
            summand in the forward pass (``t = Σ (e_v − μ_{f(v)}) + e_ID``); the ID row is
            never centred. Off (default) = dc01 behaviour, byte-identical.
        :param center_mode: which per-field mean — 'weighted' (catalog-frequency-weighted,
            the fI′ default), 'unweighted' (uniform over the field's vocab — no
            zero-frequency blind spot), 'interaction' (train-interaction-weighted; needs
            ``item_weights``). Any per-field constant is a valid gauge (dc06 4b C1).
        :param center_detach: compute μ from a detached view (D4 fallback — no live-mean
            gradient coupling). Default False = differentiable μ, recomputed per forward.
        :param metadata_scale: dc06 α-control arm — constant scalar on the metadata summands
            (``t = α·Σ e_v + e_ID``); composes with (but is meant INSTEAD of) centring.
            Default 1.0 = no-op.
        :param field_index: LongTensor (n_features,) vocab code → field position, from
            ``build_field_index``. Required iff ``center_fields``.
        :param item_weights: FloatTensor (n_objects,) per-item weights for
            ``center_mode='interaction'`` (from ``build_item_interaction_counts``).
        """
        super().__init__()
        assert feature_ids.dim() == 2 and feature_ids.shape[0] == n_objects, \
            f'feature_ids must have shape (n_objects, F); got {tuple(feature_ids.shape)} for n_objects={n_objects}'

        self.n_objects = n_objects
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.use_id_feature = use_id_feature
        self.max_norm = max_norm
        self.name = 'FeatureEmbedding'

        # Non-persistent buffer: moves with .to(device) but is excluded from state_dict().
        self.register_buffer('feature_ids', feature_ids.long(), persistent=False)
        if feature_weights is not None:
            assert feature_weights.shape == feature_ids.shape, \
                f'feature_weights {tuple(feature_weights.shape)} must match feature_ids ' \
                f'{tuple(feature_ids.shape)}'
            self.register_buffer('feature_weights', feature_weights.float(), persistent=False)
        else:
            self.feature_weights = None

        # --- dc06 fI′ centring / α-control state -------------------------------------------
        self.center_fields = center_fields
        self.center_mode = center_mode
        self.center_detach = center_detach
        self.metadata_scale = float(metadata_scale)
        if center_fields:
            assert center_mode in ('weighted', 'unweighted', 'interaction'), \
                f"center_mode must be 'weighted'|'unweighted'|'interaction'; got {center_mode!r}"
            assert field_index is not None and tuple(field_index.shape) == (n_features,), \
                f'center_fields needs field_index of shape ({n_features},); got ' \
                f'{None if field_index is None else tuple(field_index.shape)}'
            assert n_features > 0, 'center_fields with zero metadata rows is meaningless (F=0)'
            self.register_buffer('field_index', field_index.long(), persistent=False)
            n_fields = int(field_index.max().item()) + 1

            # Row-stochastic per-field probability matrix P (n_fields, n_features):
            # μ = P @ E_meta. Counts come from the SAME feature_ids/feature_weights the
            # composition uses (bag padding carries weight 0.0 → contributes nothing).
            counts = torch.zeros(n_features, dtype=torch.float64)
            flat_ids = feature_ids.reshape(-1).long()
            if feature_weights is None:
                flat_w = torch.ones(flat_ids.shape[0], dtype=torch.float64)
            else:
                flat_w = feature_weights.reshape(-1).double()
            if center_mode == 'interaction':
                assert item_weights is not None and tuple(item_weights.shape) == (n_objects,), \
                    f"center_mode='interaction' needs item_weights of shape ({n_objects},)"
                per_item = item_weights.double().unsqueeze(1).expand(-1, feature_ids.shape[1])
                flat_w = flat_w * per_item.reshape(-1)
            if center_mode == 'unweighted':
                counts = torch.ones(n_features, dtype=torch.float64)
            else:
                counts.scatter_add_(0, flat_ids, flat_w)
            field_tot = torch.zeros(n_fields, dtype=torch.float64)
            field_tot.scatter_add_(0, field_index.long(), counts)
            assert bool((field_tot > 0).all()), \
                f'a field has zero total count under center_mode={center_mode!r} — no mean definable'
            probs = torch.zeros(n_fields, n_features, dtype=torch.float64)
            probs[field_index.long(), torch.arange(n_features)] = counts / field_tot[field_index.long()]
            self.register_buffer('field_probs', probs.float(), persistent=False)

            # Per-item per-field token-weight totals W (n_objects, n_fields): the centring
            # correction is applied at ITEM level, Σ_tokens w·μ_{f(v)} = W_i @ M — exactly
            # equal to per-token subtraction by linearity, without materializing the
            # (B, N, D_max, d) per-token correction tensor (the 7575510 smoke showed that
            # tensor thrashing the A30 allocator at ~1 GB/temporary on ml-1m bags).
            w_tot = torch.zeros(feature_ids.shape[0], n_fields, dtype=torch.float64)
            tok_fields = field_index.long()[feature_ids]  # (n_objects, T)
            if feature_weights is None:
                w_tot.scatter_add_(1, tok_fields, torch.ones_like(tok_fields, dtype=torch.float64))
            else:
                w_tot.scatter_add_(1, tok_fields, feature_weights.double())
            self.register_buffer('field_weight_totals', w_tot.float(), persistent=False)
        else:
            self.field_index = None
            self.field_probs = None

        n_rows = n_features + (n_objects if use_id_feature else 0)
        self.embedding_layer = nn.Embedding(n_rows, embedding_dim, max_norm=max_norm)

        print(f'Built FeatureEmbedding model \n'
              f'- n_objects: {self.n_objects} \n'
              f'- n_features: {self.n_features} \n'
              f'- F (fields per object): {feature_ids.shape[1]} \n'
              f'- layout: {"weighted bags (D_max=%d)" % feature_ids.shape[1] if feature_weights is not None else "fixed one-value-per-field"} \n'
              f'- embedding_dim: {self.embedding_dim} \n'
              f'- use_id_feature: {self.use_id_feature} \n'
              f'- n_embedding_rows: {n_rows} \n'
              f'- max_norm: {self.max_norm}')

    def init_parameters(self):
        self.embedding_layer.apply(general_weight_init)

    def _field_means(self):
        """dc06: μ matrix (n_fields, d) = ``field_probs @ metadata rows``. Reads
        ``embedding_layer.weight`` directly — under ``max_norm`` this sees rows before any
        renorm the current forward's lookup applies (committed fI′ configs never selected
        max_norm; semantics stay well-defined either way)."""
        m = self.field_probs @ self.embedding_layer.weight[:self.n_features]
        return m.detach() if self.center_detach else m

    def _transform_meta_sum(self, meta_sum: torch.Tensor, o_idxs: torch.Tensor) -> torch.Tensor:
        """dc06: apply centring and/or the α-control constant to the already-reduced metadata
        sum. Centring is applied at ITEM level — ``Σ_tokens w·(e − μ) = Σ w·e − W_i @ M`` with
        W = per-field token-weight totals (exact by linearity; equals per-token subtraction,
        verified t02 V2/V4) — so no (…, T, d) per-token correction tensor is ever built
        (the 7575510 smoke's allocator-thrash lesson). ``meta_sum``: (..., d)."""
        if self.center_fields:
            meta_sum = meta_sum - self.field_weight_totals[o_idxs] @ self._field_means()
        if self.metadata_scale != 1.0:
            meta_sum = meta_sum * self.metadata_scale
        return meta_sum

    @torch.no_grad()
    def gauge_snap(self):
        """dc06 A11 (post-step gauge projection): subtract μ_f from every metadata row so the
        RAW table itself sits on the μ=0 slice between steps. Function-inert (dc06 4b C2/C7 —
        per-field offsets carry no score content under centring; the forward's subtraction
        then removes ~0), and it keeps raw-table diagnostics and the existing share renderers
        gauge-clean (checkpoints land exactly field-centred). Called by the Trainer after
        every optimizer step whenever ``center_fields`` is on; no-op otherwise."""
        if not self.center_fields:
            return
        w = self.embedding_layer.weight
        m = self.field_probs @ w[:self.n_features]
        w[:self.n_features] -= m[self.field_index]

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        assert o_idxs is not None, f"Object Indexes not provided! ({self.name})"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        feat = self.feature_ids[o_idxs]  # (..., F)
        n_meta = feat.shape[-1]
        needs_meta_transform = self.center_fields or self.metadata_scale != 1.0
        if self.feature_weights is None:
            # Fixed layout — with both dc06 knobs off, byte-identical to the pre-extension path.
            lookup = feat
            if self.use_id_feature:
                id_col = (o_idxs + self.n_features).unsqueeze(-1)  # (..., 1)
                lookup = torch.cat([feat, id_col], dim=-1)  # (..., F+1)
            emb = self.embedding_layer(lookup)  # (..., F[+1], d)
            if needs_meta_transform:
                meta_sum = self._transform_meta_sum(emb[..., :n_meta, :].sum(dim=-2), o_idxs)
                return meta_sum + emb[..., n_meta:, :].sum(dim=-2)  # + ID part (empty ⇒ zero)
            return emb.sum(dim=-2)  # (..., d)

        # Bag layout: weighted sum; ID row (weight 1.0) is concatenated BEFORE the reduction so
        # the summation shape/order matches the fixed path exactly (bit-identity keystone t07).
        w = self.feature_weights[o_idxs]  # (..., D_max)
        lookup = feat
        if self.use_id_feature:
            id_col = (o_idxs + self.n_features).unsqueeze(-1)  # (..., 1)
            lookup = torch.cat([feat, id_col], dim=-1)  # (..., D_max+1)
            # ones column built from shape, NOT by slicing w — w[..., :1] is empty when D_max=0
            # (the F=0 reduction) and would silently zero the ID row via 0-size broadcasting.
            w = torch.cat([w, w.new_ones(w.shape[:-1] + (1,))], dim=-1)  # (..., D_max+1)
        emb = self.embedding_layer(lookup)  # (..., D_max[+1], d)
        if needs_meta_transform:
            # dc06 §3.1 bag form: t = Σ w·(e − μ_{f(v)}) + e_ID — computed as the raw weighted
            # sum minus the item-level total W_i @ M (exact by linearity, 4b C9 semantics
            # unchanged: the removed mass stays item-dependent on bags).
            meta_sum = (emb[..., :n_meta, :] * w[..., :n_meta].unsqueeze(-1)).sum(dim=-2)
            meta_sum = self._transform_meta_sum(meta_sum, o_idxs)
            id_sum = (emb[..., n_meta:, :] * w[..., n_meta:].unsqueeze(-1)).sum(dim=-2)
            return meta_sum + id_sum
        return (emb * w.unsqueeze(-1)).sum(dim=-2)  # (..., d)


class FeatureEmbeddingW(FeatureExtractor):
    """
    Projection branch mirroring ``EmbeddingW`` but wrapping a SHARED ``FeatureEmbedding`` instance
    (passed in, not constructed) followed by a bias-free linear projection. Sharing the SAME
    ``FeatureEmbedding`` instance between the item prototype branch and this projection branch is
    how dc01 realizes the double-tie (R1): one composed representation q_i feeds both halves of the
    UI-score. ``forward = linear(shared(o_idxs))``.
    """

    def __init__(self, shared: FeatureEmbedding, out_dimension: int = None, use_bias: bool = False):
        """
        :param shared: the shared FeatureEmbedding instance (also used by the item PrototypeEmbedding).
        :param out_dimension: out dimension of the linear layer. If None, set to the shared emb dim.
        :param use_bias: whether to use the bias in the linear layer.
        """
        super().__init__()
        self.shared = shared
        self.embedding_dim = shared.embedding_dim
        self.out_dimension = out_dimension if out_dimension is not None else shared.embedding_dim
        self.use_bias = use_bias
        self.name = 'FeatureEmbeddingW'

        self.linear_layer = nn.Linear(self.embedding_dim, self.out_dimension, bias=self.use_bias)

        print(f'Built FeatureEmbeddingW model \n'
              f'- out_dimension: {self.out_dimension} \n'
              f'- use_bias: {self.use_bias} \n')

    def init_parameters(self):
        # Single-owner init: the shared FeatureEmbedding is initialized exactly once by its owner
        # (the factory). This branch initializes ONLY its own linear layer — never the shared table.
        self.linear_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(self.shared(o_idxs))


class HistoryFeatureEmbedding(FeatureExtractor):
    """
    dc05 (``feature_user_proto``) user base: represents a USER as the weighted sum of the
    attribute-word embeddings of their train-window purchase history (history-composed user
    factors, design doc §3.1/§3.2): q_u = Σ_j w̄_{u,j}·e_j (+ e_ID(u)). Sibling of
    ``FeatureEmbedding``, which cannot serve here: that class assumes fixed-width
    one-value-per-field rows, while user baskets are variable-length and weighted.

    The padded per-user layout (``hist_value_ids``/``hist_weights``, width D_max = max distinct
    attribute values in any user's train basket) is built once from the split's OWN
    ``listening_history_train.csv`` + ``item_features.csv`` (``build_user_history_weights``) and
    registered as NON-persistent buffers — rebuilt deterministically at model build time, never
    checkpointed (``FeatureEmbedding`` convention). Padding slots carry word id 0 with weight 0.0
    and contribute exactly nothing. With ``use_id_feature=True`` and zero metadata words this
    reduces EXACTLY to a plain per-user ``Embedding`` table — the dc05 keystone reduction
    fU(F=0, +ID) ≡ ``user_proto`` (see ``Master/temp/dc_checks/dc05``). Corner on record
    (dc05 SC.4 audit, F-DC05-17): in that F=0 reduction ``n_features=0``, so padding id 0
    aliases USER 0's ID row — inert as a summand (weight 0.0), but under ``max_norm`` the
    padded accesses would renorm-clamp that row every forward, perturbing training dynamics
    and the keystone's bit-identity. Unreachable in every committed config (none sets
    ``max_norm``); any future max_norm config must re-verify the keystone max_norm-free.
    """

    def __init__(self, n_objects: int, hist_value_ids: torch.LongTensor, hist_weights: torch.Tensor,
                 n_features: int, embedding_dim: int, use_id_feature: bool = True,
                 max_norm: float = None, item_token_ids: torch.LongTensor = None,
                 item_token_w: torch.Tensor = None, hist_sizes: torch.Tensor = None,
                 loo_pooling: bool = True):
        """
        :param n_objects: number of objects in the system (users).
        :param hist_value_ids: LongTensor (n_objects, D_max) — per-user padded attribute-value ids
            with the global field-offset encoding shared with ``build_feature_ids`` (labels stay
            renderer-compatible). Padding slots hold id 0.
        :param hist_weights: Tensor (n_objects, D_max) — mean-normalized per-value weights
            (w̄ = n_{u,f}/|H_u|); exactly 0.0 on padding slots. A user absent from the train
            window has an all-zero row (q_u = e_ID(u), or 0 without the ID row — the verified
            degenerate path, 4b C3).
        :param n_features: size of the metadata word vocabulary (Σ per-field vocab sizes). ID rows
            (when used) are indexed at n_features + o_idx, after the metadata rows.
        :param embedding_dim: embedding dimension d.
        :param use_id_feature: if True, add the per-user ID row e_ID(u) to every composition.
        :param max_norm: max norm of the l2 norm of the embedding rows.
        :param item_token_ids: LongTensor (n_items, D_item) — per-item attribute-word ids (the fI
            item table, same vocabulary/offsets); padding id 0. Leave-one-out payload.
        :param item_token_w: Tensor (n_items, D_item) — 1.0 on real slots, 0.0 on padding.
        :param hist_sizes: Tensor (n_objects,) — |H_u|, distinct train purchases per user.
        :param loo_pooling: leave-one-out pooling in TRAINING mode (P5 hygiene, 2026-09-08). The
            history buffers are built from the full train file, and every training positive is a
            train pair, so without this the scored positive's own words sit inside q_u at train
            time and never at val/test (FISM set-minus-i mismatch; ≈15 % of the metadata mass per
            training pair on hm_1_month, ≈1 % on ml-1m). When True and the payload is present,
            ``forward(o_idxs, pos_i_idxs=...)`` composes the mean over the OTHER purchases:
            q_meta = (|H_u|·q̄_u − Σ_{t∈bag(i⁺)} e_t) / (|H_u| − 1), exactly zero metadata mass for
            a single-purchase user. Eval mode and calls without ``pos_i_idxs`` are unchanged.
        """
        super().__init__()
        assert hist_value_ids.dim() == 2 and hist_value_ids.shape[0] == n_objects, \
            f'hist_value_ids must have shape (n_objects, D_max); got {tuple(hist_value_ids.shape)} ' \
            f'for n_objects={n_objects}'
        assert hist_value_ids.shape == hist_weights.shape, \
            f'hist_value_ids {tuple(hist_value_ids.shape)} and hist_weights ' \
            f'{tuple(hist_weights.shape)} must have identical shapes'

        self.n_objects = n_objects
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.use_id_feature = use_id_feature
        self.max_norm = max_norm
        self.name = 'HistoryFeatureEmbedding'

        # Non-persistent buffers: move with .to(device) but are excluded from state_dict().
        self.register_buffer('hist_value_ids', hist_value_ids.long(), persistent=False)
        self.register_buffer('hist_weights', hist_weights.float(), persistent=False)

        # Leave-one-out payload (optional; all three or none).
        has_loo_payload = item_token_ids is not None or item_token_w is not None or hist_sizes is not None
        if has_loo_payload:
            assert item_token_ids is not None and item_token_w is not None and hist_sizes is not None, \
                'leave-one-out payload must be complete: item_token_ids, item_token_w, hist_sizes'
            assert item_token_ids.shape == item_token_w.shape, \
                f'item_token_ids {tuple(item_token_ids.shape)} and item_token_w ' \
                f'{tuple(item_token_w.shape)} must have identical shapes'
            assert hist_sizes.dim() == 1 and hist_sizes.shape[0] == n_objects, \
                f'hist_sizes must have shape (n_objects,); got {tuple(hist_sizes.shape)}'
            self.register_buffer('item_token_ids', item_token_ids.long(), persistent=False)
            self.register_buffer('item_token_w', item_token_w.float(), persistent=False)
            self.register_buffer('hist_sizes', hist_sizes.float(), persistent=False)
        self.loo_pooling = bool(loo_pooling) and has_loo_payload
        # Introspection for tests/smokes: did the LAST forward apply leave-one-out?
        self.last_forward_loo = False

        n_rows = n_features + (n_objects if use_id_feature else 0)
        self.embedding_layer = nn.Embedding(n_rows, embedding_dim, max_norm=max_norm)

        print(f'Built HistoryFeatureEmbedding model \n'
              f'- n_objects: {self.n_objects} \n'
              f'- n_features: {self.n_features} \n'
              f'- D_max (padded basket width): {hist_value_ids.shape[1]} \n'
              f'- embedding_dim: {self.embedding_dim} \n'
              f'- use_id_feature: {self.use_id_feature} \n'
              f'- n_embedding_rows: {n_rows} \n'
              f'- max_norm: {self.max_norm} \n'
              f'- loo_pooling (train-time leave-one-out): {self.loo_pooling}')

    @property
    def supports_loo(self) -> bool:
        """True when RecSys should hand the training positive to ``forward`` (``pos_i_idxs``)."""
        return self.loo_pooling

    def init_parameters(self):
        self.embedding_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor, pos_i_idxs: torch.Tensor = None) -> torch.Tensor:
        """
        :param o_idxs: user indexes, shape (B,) or (B, n).
        :param pos_i_idxs: OPTIONAL, training only — the positive item scored against each user,
            shape (B,), which MUST be one of that user's train purchases (the dataloader guarantees
            this: every training pair is a train-file pair). When given, in training mode, with
            ``loo_pooling`` on, the positive's words are removed from the composition
            (leave-one-out). Ignored in eval mode (the val/test target is never in the history).
        """
        assert o_idxs is not None, f"Object Indexes not provided! ({self.name})"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        ids = self.hist_value_ids[o_idxs]  # (..., D_max)
        w = self.hist_weights[o_idxs]  # (..., D_max)
        q = (self.embedding_layer(ids) * w.unsqueeze(-1)).sum(dim=-2)  # (..., d)

        apply_loo = pos_i_idxs is not None and self.training and self.loo_pooling
        self.last_forward_loo = bool(apply_loo)
        if apply_loo:
            assert o_idxs.dim() == 1 and pos_i_idxs.shape == o_idxs.shape, \
                f'leave-one-out expects (B,) users and (B,) positives; got {tuple(o_idxs.shape)} ' \
                f'/ {tuple(pos_i_idxs.shape)}'
            n_h = self.hist_sizes[o_idxs].unsqueeze(-1)  # (B, 1) = |H_u|
            p_ids = self.item_token_ids[pos_i_idxs]  # (B, D_item)
            p_w = self.item_token_w[pos_i_idxs]  # (B, D_item)
            e_pos = (self.embedding_layer(p_ids) * p_w.unsqueeze(-1)).sum(dim=-2)  # (B, d)
            # |H|·q̄ = Σ_f n_{u,f} e_f (raw counts); minus the positive's tokens (each counted
            # once, as in n_{u,f}); re-mean over the |H|−1 remaining purchases. A user whose
            # only purchase is the positive keeps exactly zero metadata mass.
            denom = (n_h - 1.0).clamp(min=1.0)
            q = torch.where(n_h > 1.0, (n_h * q - e_pos) / denom, torch.zeros_like(q))

        if self.use_id_feature:
            q = q + self.embedding_layer(o_idxs + self.n_features)  # (..., d)
        return q


class AnchorBasedCollaborativeFiltering(FeatureExtractor):
    """
    Anchor-based Collaborative Filtering by Barkan et al. (https://dl.acm.org/doi/10.1145/3459637.3482056) published at CIKM 2021.
    """

    def __init__(self, n_objects: int, embedding_dim: int, anchors: nn.Parameter, delta_exc: float = 0,
                 delta_inc: float = 0, max_norm: float = None):
        super().__init__()
        """
        NB. delta_inc and delta_exc should be passed only when instantiating this FeatureExtractor for Items.

        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param anchors: nn.Parameters with shape (n_anchors,embedding_dim)
        :param delta_exc: factor multiplied to the exclusiveness loss
        :param delta_inc: factor multiplied to the inclusiveness loss
        :param max_norm: max norm of the l2 norm of the embeddings. 
        """

        self.anchors = anchors
        self.n_anchors = anchors.shape[0]
        self.delta_exc = delta_exc
        self.delta_inc = delta_inc

        self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)

        self._acc_exc = 0.
        self._acc_inc = 0.
        self.name = "AnchorBasedCollaborativeFiltering"

        print(f'Built AnchorBasedCollaborativeFiltering module \n'
              f'- n_anchors: {self.n_anchors} \n'
              f'- delta_exc: {self.delta_exc} \n'
              f'- delta_inc: {self.delta_inc} \n')

    def init_parameters(self):
        torch.nn.init.normal_(self.anchors, 0, 1)
        torch.nn.init.normal_(self.embedding_ext.embedding_layer.weight, 0, 1)  # Overriding previous init

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        :return:
        """
        assert o_idxs is not None, "Object indexes not provided"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        o_embed = self.embedding_ext(o_idxs)  # [...,embedding_dim]

        o_dots = o_embed @ self.anchors.T  # [...,n_anchors]

        o_coeff = nn.Softmax(dim=-1)(o_dots)  # [...,n_anchors]

        o_vect = o_coeff @ self.anchors  # [...,embedding_dim]

        # Exclusiveness constraint (BCE)
        exc = - (o_coeff * torch.log(o_coeff)).sum()

        # Inclusiveness constraint
        q_k = o_coeff.reshape(-1, self.n_anchors).sum(axis=0).div(o_coeff.sum())  # [n_anchors]
        inc = - (q_k * torch.log(q_k)).sum()

        self._acc_exc += exc
        self._acc_inc += inc

        return o_vect

    def get_and_reset_loss(self) -> float:
        acc_inc, acc_exc = self._acc_inc, self._acc_exc
        self._acc_inc = self._acc_exc = 0
        return - self.delta_inc * acc_inc + self.delta_exc * acc_exc


class PrototypeEmbedding(FeatureExtractor):
    """
    ProtoMF building block. It represents an object (item/user) given the similarity with the prototypes.
    """

    def __init__(self, n_objects: int, embedding_dim: int, n_prototypes: int = None, use_weight_matrix: bool = False,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.,
                 reg_proto_type: str = 'soft', reg_batch_type: str = 'soft', cosine_type: str = 'shifted',
                 max_norm: float = None, embedding_ext: 'FeatureExtractor' = None):
        """
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param n_prototypes: number of prototypes to consider. If none, is set to be embedding_dim.
        :param use_weight_matrix: Whether to use a linear layer after the prototype layer.
        :param sim_proto_weight: factor multiplied to the regularization loss for prototypes
        :param sim_batch_weight: factor multiplied to the regularization loss for batch
        :param reg_proto_type: type of regularization applied batch-prototype similarity matrix on the prototypes. Possible values are ['max','soft','incl']
        :param reg_batch_type: type of regularization applied batch-prototype similarity matrix on the batch. Possible values are ['max','soft']
        :param cosine_type: type of cosine similarity to apply. Possible values ['shifted','standard','shifted_and_div']
        :param max_norm: max norm of the l2 norm of the embeddings.
        :param embedding_ext: optional pre-built FeatureExtractor used to produce the base object
            embedding `o_embed`. When None (default) the original behaviour is preserved exactly —
            a fresh `Embedding(n_objects, embedding_dim, max_norm)` is constructed here. When given,
            the passed instance is used as-is (e.g. a shared `FeatureEmbedding` for feature-composed
            item factors). This module never initializes a passed-in extractor (single-owner init).

        """

        super(PrototypeEmbedding, self).__init__()

        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.use_weight_matrix = use_weight_matrix
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight
        self.reg_proto_type = reg_proto_type
        self.reg_batch_type = reg_batch_type
        self.cosine_type = cosine_type

        # embedding_ext=None preserves the original behaviour bit-for-bit (same RNG draw order:
        # the base Embedding is constructed here, before the prototypes below). A passed-in
        # extractor was constructed elsewhere (its weights drawn earlier) and is used as-is.
        if embedding_ext is None:
            self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)
        else:
            self.embedding_ext = embedding_ext

        if self.n_prototypes is None:
            self.prototypes = nn.Parameter(torch.randn([self.embedding_dim, self.embedding_dim]))
            self.n_prototypes = self.embedding_dim
        else:
            self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]))

        if self.use_weight_matrix:
            self.weight_matrix = nn.Linear(self.n_prototypes, self.embedding_dim, bias=False)

        # Cosine Type
        if self.cosine_type == 'standard':
            self.cosine_sim_func = nn.CosineSimilarity(dim=-1)
        elif self.cosine_type == 'shifted':
            self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y))
        elif self.cosine_type == 'shifted_and_div':
            self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2
        else:
            raise ValueError(f'Cosine type {self.cosine_type} not implemented')

        # Regularization Batch
        if self.reg_batch_type == 'max':
            self.reg_batch_func = lambda x: - x.max(dim=1).values.mean()
        elif self.reg_batch_type == 'soft':
            self.reg_batch_func = lambda x: self._entropy_reg_loss(x, 1)
        else:
            raise ValueError(f'Regularization Type for Batch {self.reg_batch_func} not yet implemented')

        # Regularization Proto
        if self.reg_proto_type == 'max':
            self.reg_proto_func = lambda x: - x.max(dim=0).values.mean()
        elif self.reg_proto_type == 'soft':
            self.reg_proto_func = lambda x: self._entropy_reg_loss(x, 0)
        elif self.reg_proto_type == 'incl':
            self.reg_proto_func = lambda x: self._inclusiveness_constraint(x)
        else:
            raise ValueError(f'Regularization Type for Proto {self.reg_proto_type} not yet implemented')

        self._acc_r_proto = 0
        self._acc_r_batch = 0
        self.name = "PrototypeEmbedding"

        print(f'Built PrototypeEmbedding model \n'
              f'- n_prototypes: {self.n_prototypes} \n'
              f'- use_weight_matrix: {self.use_weight_matrix} \n'
              f'- sim_proto_weight: {self.sim_proto_weight} \n'
              f'- sim_batch_weight: {self.sim_batch_weight} \n'
              f'- reg_proto_type: {self.reg_proto_type} \n'
              f'- reg_batch_type: {self.reg_batch_type} \n'
              f'- cosine_type: {self.cosine_type} \n')

    @staticmethod
    def _entropy_reg_loss(sim_mtx, axis: int):
        o_coeff = nn.Softmax(dim=axis)(sim_mtx)
        entropy = - (o_coeff * torch.log(o_coeff)).sum(axis=axis).mean()
        return entropy

    @staticmethod
    def _inclusiveness_constraint(sim_mtx):
        '''
        NB. This method is applied only on a square matrix (batch_size,n_prototypes) and it return the negated
        inclusiveness constraints (its minimization brings more equal load sharing among the prototypes)
        '''
        o_coeff = nn.Softmax(dim=1)(sim_mtx)
        q_k = o_coeff.sum(axis=0).div(o_coeff.sum())  # [n_prototypes]
        entropy_q_k = - (q_k * torch.log(q_k)).sum()
        return - entropy_q_k

    def init_parameters(self):
        if self.use_weight_matrix:
            nn.init.xavier_normal_(self.weight_matrix.weight)

    @property
    def supports_loo(self) -> bool:
        """Delegates to the wrapped extractor (HistoryFeatureEmbedding leave-one-out)."""
        return bool(getattr(self.embedding_ext, 'supports_loo', False))

    def forward(self, o_idxs: torch.Tensor, **ext_kwargs) -> torch.Tensor:
        """
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        :param ext_kwargs: forwarded verbatim to ``embedding_ext`` (e.g. ``pos_i_idxs``).
        :return:
        """
        assert o_idxs is not None, "Object indexes not provided"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        # Pass-through of extractor kwargs (leave-one-out `pos_i_idxs`, P5 hygiene 2026-09-08);
        # the bare call is kept bit-identical for every embedding_ext that takes only o_idxs.
        o_embed = self.embedding_ext(o_idxs, **ext_kwargs) if ext_kwargs \
            else self.embedding_ext(o_idxs)  # [..., embedding_dim]

        # https://github.com/pytorch/pytorch/issues/48306
        sim_mtx = self.cosine_sim_func(o_embed.unsqueeze(-2), self.prototypes)  # [..., n_prototypes]

        if self.use_weight_matrix:
            w = self.weight_matrix(sim_mtx)  # [...,embedding_dim]
        else:
            w = sim_mtx  # [..., embedding_dim = n_prototypes]

        # Computing additional losses
        batch_proto = sim_mtx.reshape([-1, sim_mtx.shape[-1]])

        self._acc_r_batch += self.reg_batch_func(batch_proto)
        self._acc_r_proto += self.reg_proto_func(batch_proto)

        return w

    def get_and_reset_loss(self) -> float:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        return self.sim_proto_weight * acc_r_proto + self.sim_batch_weight * acc_r_batch


class ConcatenateFeatureExtractors(FeatureExtractor):

    def __init__(self, model_1: FeatureExtractor, model_2: FeatureExtractor, invert: bool = False):
        super().__init__()

        """
        Concatenates the latent dimension (considered in position -1) of two Feature Extractors models.
        :param model_1: a FeatureExtractor model
        :param model_2: a FeatureExtractor model
        :param invert: whether to place the latent representation from the second model on top.
        """

        self.model_1 = model_1
        self.model_2 = model_2
        self.invert = invert

        self.name = 'ConcatenateFeatureExtractors'

        print(f'Built ConcatenateFeatureExtractors model \n'
              f'- model_1: {self.model_1.name} \n'
              f'- model_2: {self.model_2.name} \n'
              f'- invert: {self.invert} \n')

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        o_repr_1 = self.model_1(o_idxs)
        o_repr_2 = self.model_2(o_idxs)

        if self.invert:
            return torch.cat([o_repr_2, o_repr_1], dim=-1)
        else:
            return torch.cat([o_repr_1, o_repr_2], dim=-1)

    def get_and_reset_loss(self) -> float:
        loss_1 = self.model_1.get_and_reset_loss()
        loss_2 = self.model_2.get_and_reset_loss()
        return loss_1 + loss_2

    def init_parameters(self):
        self.model_1.init_parameters()
        self.model_2.init_parameters()


class AttributeLookup(FeatureExtractor):
    """
    dc02 (``attr_item_proto``) parameter-free item base: represents an object ONLY by its observed
    multi-hot attribute vector x_i ∈ {0,1}^V (concept-bottleneck property — no per-object
    parameters of any kind). The (n_objects, V) matrix is registered as a NON-persistent buffer:
    it moves with ``.to(device)`` but is excluded from ``state_dict()`` and rebuilt
    deterministically from ``item_features.csv`` at model build time (cf. ``FeatureEmbedding``),
    so checkpoints hold only learned weights and the tester's strict ``load_state_dict`` works.
    """

    def __init__(self, attr_multi_hot: torch.Tensor):
        """
        :param attr_multi_hot: (n_objects, V) multi-hot attribute matrix (one column block per
            categorical field, exactly one 1 inside each block per row). Stored as float32 —
            required downstream by ``nn.CosineSimilarity`` / ``nn.Linear``.
        """
        super().__init__()
        assert attr_multi_hot.dim() == 2, \
            f'attr_multi_hot must have shape (n_objects, n_attr_values); got {tuple(attr_multi_hot.shape)}'

        self.n_objects = attr_multi_hot.shape[0]
        self.n_attr_values = attr_multi_hot.shape[1]
        self.name = 'AttributeLookup'

        self.register_buffer('attr_multi_hot', attr_multi_hot.float(), persistent=False)

        print(f'Built AttributeLookup model \n'
              f'- n_objects: {self.n_objects} \n'
              f'- n_attr_values (V): {self.n_attr_values}')

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        assert o_idxs is not None, f"Object Indexes not provided! ({self.name})"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'
        return self.attr_multi_hot[o_idxs]  # (..., V)


class AttributePrototypeEmbedding(PrototypeEmbedding):
    """
    dc02 (``attr_item_proto``) item prototype block: structurally the host ``PrototypeEmbedding``
    with the base embedding replaced by a parameter-free ``AttributeLookup`` and prototypes living
    in attribute space — ``self.prototypes`` IS the readable profile matrix A ∈ R^{K×V}
    (``embedding_dim = V``). The similarity/regularizer machinery is inherited from the host
    (same cosine lambdas, same accumulators, same ``sim_proto_weight``/``sim_batch_weight``);
    ``forward`` is a faithful mirror of the host's, reading ``effective_prototypes()`` in place of
    the raw parameter so the K1 reparameterization composes with everything else.

    The four dc02 §3.5 constraint knobs are config-gated and DEFAULT OFF:
    - K1 ``nonneg_prototypes``: A = softplus(Ã) — the stored parameter becomes Ã (state_dict key
      stays ``prototypes``; the config flag fixes the semantics, and tester/loader always rebuild
      from the checkpoint's own config.json).
    - K2 ``crisp_weight``: per-field crispness — mean Shannon entropy of softmax(A[k, block_f]).
    - K3 ``sep_weight``/``sep_margin``: separation hinge mean(max(0, cos(a_k, a_j) − m)) over
      unordered prototype pairs, PLAIN cosine.
    - K4 ``push_weight``/``push_n_items``/``push_seed``: data pull mean_k(1 − max_{i∈S} cos(a_k, x_i))
      over a per-step resampled item subset S, drawn from a DEDICATED torch.Generator so the
      global RNG stream (batches, negative sampling) is untouched.
    K2–K4 hook into ``get_and_reset_loss()`` — evaluated once per optimization step, in-graph.
    """

    def __init__(self, n_objects: int, attr_lookup: AttributeLookup, n_prototypes: int,
                 field_offsets, sim_proto_weight: float = 1., sim_batch_weight: float = 1.,
                 reg_proto_type: str = 'soft', reg_batch_type: str = 'soft',
                 cosine_type: str = 'shifted',
                 nonneg_prototypes: bool = False,
                 crisp_weight: float = 0.,
                 sep_weight: float = 0., sep_margin: float = 0.5,
                 push_weight: float = 0., push_n_items: int = 256, push_seed: int = 98765):
        """
        :param n_objects: number of objects (items).
        :param attr_lookup: the shared parameter-free multi-hot base (also used by the projection
            branch — instance sharing realizes the tie). Used as-is, never re-initialized.
        :param n_prototypes: K — number of attribute-space prototypes.
        :param field_offsets: cumulative field block bounds, length F+1 (block f =
            ``[field_offsets[f], field_offsets[f+1])``). Needed by K2; plain list, rebuilt at
            build time (not in state_dict).
        :param sim_proto_weight / sim_batch_weight / reg_proto_type / reg_batch_type / cosine_type:
            host ``PrototypeEmbedding`` regularizer knobs, inherited semantics.
        :param nonneg_prototypes / crisp_weight / sep_weight / sep_margin / push_weight /
            push_n_items / push_seed: the dc02 constraint knobs (see class docstring).
        """
        assert isinstance(attr_lookup, AttributeLookup), \
            f'attr_lookup must be an AttributeLookup; got {type(attr_lookup).__name__}'
        super().__init__(n_objects, embedding_dim=attr_lookup.n_attr_values,
                         n_prototypes=n_prototypes, use_weight_matrix=False,
                         sim_proto_weight=sim_proto_weight, sim_batch_weight=sim_batch_weight,
                         reg_proto_type=reg_proto_type, reg_batch_type=reg_batch_type,
                         cosine_type=cosine_type, max_norm=None, embedding_ext=attr_lookup)
        assert list(field_offsets)[0] == 0 and list(field_offsets)[-1] == attr_lookup.n_attr_values, \
            f'field_offsets must span [0, V]; got {list(field_offsets)} for V={attr_lookup.n_attr_values}'

        self.field_offsets = list(field_offsets)
        self.nonneg_prototypes = nonneg_prototypes
        self.crisp_weight = crisp_weight
        self.sep_weight = sep_weight
        self.sep_margin = sep_margin
        self.push_weight = push_weight
        self.push_n_items = push_n_items
        self.push_seed = push_seed
        # Dedicated RNG for the K4 subset — never the global stream, so a knobs-on run draws the
        # identical batch/negative-sampling sequence as knobs-off under reproducible(seed).
        # (Generator state is not checkpointed; a resumed run only changes the subset sequence.)
        self._push_gen = torch.Generator().manual_seed(push_seed)
        self.name = 'AttributePrototypeEmbedding'

        print(f'Built AttributePrototypeEmbedding model \n'
              f'- n_attr_values (V): {attr_lookup.n_attr_values} \n'
              f'- n_fields (F): {len(self.field_offsets) - 1} \n'
              f'- nonneg_prototypes: {self.nonneg_prototypes} \n'
              f'- crisp_weight: {self.crisp_weight} \n'
              f'- sep_weight: {self.sep_weight} (margin {self.sep_margin}) \n'
              f'- push_weight: {self.push_weight} (n_items {self.push_n_items})')

    def effective_prototypes(self) -> torch.Tensor:
        """The profile matrix A the model actually uses: softplus(Ã) under K1, else the raw
        parameter. All read-outs must consume THIS, not ``self.prototypes``."""
        if self.nonneg_prototypes:
            return nn.functional.softplus(self.prototypes)
        return self.prototypes

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        Faithful mirror of the host forward (``use_weight_matrix`` is always False here) with
        ``effective_prototypes()`` in place of the raw parameter. Same reg accumulation on the
        flattened sim matrix. Bitwise host equivalence in base mode is pinned by t02.
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        """
        assert o_idxs is not None, "Object indexes not provided"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        o_embed = self.embedding_ext(o_idxs)  # [..., V]

        sim_mtx = self.cosine_sim_func(o_embed.unsqueeze(-2), self.effective_prototypes())  # [..., K]

        batch_proto = sim_mtx.reshape([-1, sim_mtx.shape[-1]])

        self._acc_r_batch += self.reg_batch_func(batch_proto)
        self._acc_r_proto += self.reg_proto_func(batch_proto)

        return sim_mtx

    def get_and_reset_loss(self) -> float:
        loss = super().get_and_reset_loss()
        if self.crisp_weight > 0 or self.sep_weight > 0 or self.push_weight > 0:
            prototypes = self.effective_prototypes()
            if self.crisp_weight > 0:
                loss = loss + self.crisp_weight * self._field_crispness_loss(prototypes)
            if self.sep_weight > 0:
                loss = loss + self.sep_weight * self._separation_loss(prototypes)
            if self.push_weight > 0:
                loss = loss + self.push_weight * self._data_pull_loss(prototypes)
        return loss

    def _field_crispness_loss(self, prototypes: torch.Tensor) -> torch.Tensor:
        """K2: L_crisp = (1/(K·F)) Σ_k Σ_f H(softmax(A[k, block_f])) — Shannon entropy in nats."""
        entropies = []
        for f in range(len(self.field_offsets) - 1):
            block = prototypes[:, self.field_offsets[f]:self.field_offsets[f + 1]]  # (K, V_f)
            p = nn.Softmax(dim=1)(block)
            entropies.append(- (p * torch.log(p)).sum(dim=1))  # (K,)
        return torch.stack(entropies, dim=1).mean()

    def _separation_loss(self, prototypes: torch.Tensor) -> torch.Tensor:
        """K3: L_sep = mean over unordered pairs k≠j of max(0, cos(a_k, a_j) − m), PLAIN cosine."""
        if self.n_prototypes < 2:
            return prototypes.sum() * 0.
        normed = prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cos = normed @ normed.T  # (K, K)
        iu = torch.triu_indices(self.n_prototypes, self.n_prototypes, offset=1)
        return torch.relu(cos[iu[0], iu[1]] - self.sep_margin).mean()

    def _data_pull_loss(self, prototypes: torch.Tensor) -> torch.Tensor:
        """K4: L_push = (1/K) Σ_k min_{i∈S} (1 − cos(a_k, x_i)), S resampled every call.
        If ``push_n_items >= n_objects`` the full catalog is used deterministically."""
        attr_multi_hot = self.embedding_ext.attr_multi_hot
        if self.push_n_items >= self.n_objects:
            idxs = torch.arange(self.n_objects, device=attr_multi_hot.device)
        else:
            idxs = torch.randint(0, self.n_objects, (self.push_n_items,),
                                 generator=self._push_gen).to(attr_multi_hot.device)
        x_sub = attr_multi_hot[idxs]
        x_normed = x_sub / x_sub.norm(dim=1, keepdim=True).clamp_min(1e-8)
        a_normed = prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cos = a_normed @ x_normed.T  # (K, |S|)
        return (1. - cos.max(dim=1).values).mean()


class AttributeProjection(FeatureExtractor):
    """
    dc02 projection branch (t̂ = W_t·x_i): ``FeatureEmbeddingW``'s role over the SHARED
    parameter-free ``AttributeLookup`` base, followed by a bias-free linear V → out_dimension.
    Sharing the SAME lookup instance with the item prototype branch realizes the double-tie (R1):
    one observed representation x_i feeds both halves of the UI-score. There is no learnable
    table to single-owner-init here — only this branch's linear layer has parameters.
    """

    def __init__(self, lookup: AttributeLookup, out_dimension: int):
        """
        :param lookup: the shared AttributeLookup instance (also used by AttributePrototypeEmbedding).
        :param out_dimension: out dimension of the linear layer (= number of user prototypes L_u).
        """
        super().__init__()
        self.lookup = lookup
        self.out_dimension = out_dimension
        self.name = 'AttributeProjection'

        self.linear_layer = nn.Linear(lookup.n_attr_values, out_dimension, bias=False)

        print(f'Built AttributeProjection model \n'
              f'- in_dimension (V): {lookup.n_attr_values} \n'
              f'- out_dimension: {self.out_dimension}')

    def init_parameters(self):
        self.linear_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(self.lookup(o_idxs))
