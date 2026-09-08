from typing import Tuple

import torch
from torch import nn

from feature_extraction.feature_extractors import FeatureExtractor, Embedding, AnchorBasedCollaborativeFiltering, \
    PrototypeEmbedding, ConcatenateFeatureExtractors, EmbeddingW, FeatureEmbedding, \
    AttributeLookup, AttributePrototypeEmbedding, AttributeProjection, HistoryFeatureEmbedding



def _loo_kwargs(user_param: dict) -> dict:
    """Leave-one-out pooling kwargs for HistoryFeatureEmbedding (P5 hygiene, 2026-09-08).

    The payload (`hist_item_token_ids`, `hist_item_token_w`, `hist_sizes`) is injected by
    ``feature_ids.inject_feature_ids`` next to the history tables; hand-built params without it
    (older dc_checks) simply get the pre-fix full-mean composition. ``loo_pooling`` (config knob,
    default True) can switch the hygiene off to reproduce pre-2026-09-08 training."""
    if 'hist_item_token_ids' not in user_param:
        return {}
    return dict(item_token_ids=user_param['hist_item_token_ids'],
                item_token_w=user_param['hist_item_token_w'],
                hist_sizes=user_param['hist_sizes'],
                loo_pooling=user_param.get('loo_pooling', True))


class FeatureExtractorFactory:

    @staticmethod
    def create_models(ft_ext_param: dict, n_users: int, n_items: int) -> Tuple[FeatureExtractor, FeatureExtractor]:

        """
        Helper function to create both the user and item feature extractor. It either creates two detached
        FeatureExtractors or a single one shared by users and items.
        :param ft_ext_param: parameters for the user feature extractor model. ft_ext_param.ft_type is used for
            switching between models.
        :param n_users: number of users in the system.
        :param n_items: number of items in the system.
        :return: [user_feature_extractor, item_feature_extractor]
        """
        assert 'ft_type' in ft_ext_param, "Type has not been specified for FeatureExtractor! " \
                                          "FeatureExtractor model not created"
        ft_type = ft_ext_param['ft_type']
        embedding_dim = ft_ext_param['embedding_dim']

        if ft_type == 'detached':
            # Build the extractors independently (e.g. two embeddings branches, one for users and one for items)
            user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users,
                                                                          embedding_dim)
            item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items,
                                                                          embedding_dim)
            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'prototypes':
            # The feature extractors are related, e.g. one of them contains a prototype layer and the other an embedding
            if 'prototypes' in ft_ext_param['user_ft_ext_param']['ft_type'] and \
                    ft_ext_param['item_ft_ext_param']['ft_type'] == 'embedding':
                # User Proto

                user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']

                user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'],
                                                                              n_users,
                                                                              embedding_dim)
                item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'],
                                                                              n_items,
                                                                              user_n_prototypes)

            elif 'prototypes' in ft_ext_param['item_ft_ext_param']['ft_type'] and \
                    ft_ext_param['user_ft_ext_param']['ft_type'] == 'embedding':
                # Item Proto
                item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']

                user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'],
                                                                              n_users,
                                                                              item_n_prototypes)
                item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'],
                                                                              n_items,
                                                                              embedding_dim)

            else:
                raise ValueError('Combination of ft_type of user/item feature extractors not valid for prototypes')

            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'prototypes_double_tie':
            # User-Item Proto
            item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']
            user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']
            user_use_weight_matrix = ft_ext_param['user_ft_ext_param']['use_weight_matrix']
            item_use_weight_matrix = ft_ext_param['item_ft_ext_param']['use_weight_matrix']

            assert not user_use_weight_matrix and not item_use_weight_matrix, 'Use Weight Matrix should be turned off to tie the weights!'

            # Building User Proto branch
            ft_ext_param['user_ft_ext_param']['ft_type'] = 'prototypes'
            user_proto = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)
            ft_ext_param['user_ft_ext_param']['ft_type'] = 'embedding_w'
            ft_ext_param['user_ft_ext_param']['out_dimension'] = item_n_prototypes
            user_embed = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)

            # Building Item Proto branch
            ft_ext_param['item_ft_ext_param']['ft_type'] = 'prototypes'
            item_proto = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)
            ft_ext_param['item_ft_ext_param']['ft_type'] = 'embedding_w'
            ft_ext_param['item_ft_ext_param']['out_dimension'] = user_n_prototypes
            item_embed = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)

            # Tying the weights together
            user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight
            item_embed.embedding_layer.weight = item_proto.embedding_ext.embedding_layer.weight

            user_feature_extractor = ConcatenateFeatureExtractors(user_proto, user_embed, invert=False)
            item_feature_extractor = ConcatenateFeatureExtractors(item_proto, item_embed, invert=True)

            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'feature_item_proto':
            # dc01 fI-ProtoMF — feature-composed item factors on the I-ProtoMF host (re-hosted
            # 2026-07-12 from the UI double-tie; the double-tie build returns at the lineage's
            # fUfI merge stage — the UI-hosted branch lives in git history, FeatureEmbeddingW
            # stays for that stage). Mirrors the 'prototypes' Item-Proto shape exactly, with ONE
            # change: the free per-item embedding feeding the prototype layer is replaced by a
            # FeatureEmbedding (q_i = Σ_f e_f, optional per-item ID row). User side = I-ProtoMF's
            # free Embedding(n_users, K_t) living in item-prototype-similarity space.
            # `feature_ids` / `n_features` are injected into item_ft_ext_param by
            # feature_ids.inject_feature_ids (called in trainer/tester._build_model) — never
            # serialized into the Ray/JSON config.
            item_param = ft_ext_param['item_ft_ext_param']
            item_n_prototypes = item_param['n_prototypes']

            assert ft_ext_param['user_ft_ext_param']['ft_type'] == 'embedding', \
                "feature_item_proto (fI host) expects a plain 'embedding' user branch " \
                f"(got {ft_ext_param['user_ft_ext_param']['ft_type']!r})"
            assert not item_param['use_weight_matrix'], \
                'use_weight_matrix must be off: the user vector lives in R^{K_t} (host Item-Proto shape)'
            assert 'feature_ids' in item_param and 'n_features' in item_param, \
                "feature_ids/n_features not injected — call feature_ids.inject_feature_ids in _build_model first"

            # --- User branch: I-ProtoMF free user vector, dimension K_t (host 'prototypes' path) ---
            user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'],
                                                                          n_users, item_n_prototypes)

            # --- Item branch: feature-composed embedding feeding the prototype layer ---
            item_max_norm = item_param['max_norm'] if 'max_norm' in item_param else None
            use_id_feature = item_param['use_id_feature'] if 'use_id_feature' in item_param else True
            # bag layout (S0-build extension): weights injected alongside feature_ids when the
            # dataset's canonical layout is 'bags'; absent = the original fixed path.
            feature_weights = item_param['feature_weights'] if 'feature_weights' in item_param else None

            item_feat_embed = FeatureEmbedding(n_items, item_param['feature_ids'], item_param['n_features'],
                                               embedding_dim, use_id_feature=use_id_feature, max_norm=item_max_norm,
                                               feature_weights=feature_weights,
                                               # dc06 fI′ knobs (defaults = dc01 behaviour, byte-identical);
                                               # field_index/item_weights injected by inject_feature_ids
                                               # only when center_fields is on in the config.
                                               center_fields=item_param.get('center_fields', False),
                                               center_mode=item_param.get('center_mode', 'weighted'),
                                               center_detach=item_param.get('center_detach', False),
                                               metadata_scale=item_param.get('metadata_scale', 1.0),
                                               field_index=item_param.get('field_index'),
                                               item_weights=item_param.get('item_weights'))
            item_feature_extractor = PrototypeEmbedding(
                n_items, embedding_dim,
                n_prototypes=item_n_prototypes,
                use_weight_matrix=False,
                sim_proto_weight=item_param['sim_proto_weight'] if 'sim_proto_weight' in item_param else 1.,
                sim_batch_weight=item_param['sim_batch_weight'] if 'sim_batch_weight' in item_param else 1.,
                reg_proto_type=item_param['reg_proto_type'] if 'reg_proto_type' in item_param else 'soft',
                reg_batch_type=item_param['reg_batch_type'] if 'reg_batch_type' in item_param else 'soft',
                cosine_type=item_param['cosine_type'] if 'cosine_type' in item_param else 'shifted',
                max_norm=item_max_norm,
                embedding_ext=item_feat_embed)

            # Init ownership (pinned at the SC.3 re-run): the FACTORY initializes the feature
            # table, exactly once. PrototypeEmbedding never inits a passed-in ext (its contract),
            # so without this call no init_parameters() path would ever reach the nested table —
            # RecSys.init_parameters cascades only one level. Single consumer now, same owner.
            item_feat_embed.init_parameters()

            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'feature_user_proto':
            # dc05 fU-ProtoMF — history-composed user factors on the U-ProtoMF host. Mirrors the
            # 'prototypes' User-Proto shape exactly, with ONE change: the free per-user embedding
            # feeding the user prototype layer is replaced by a HistoryFeatureEmbedding
            # (q_u = Σ_f w̄_{u,f}·e_f over the train basket's attribute words, optional per-user
            # ID row). Item side = U-ProtoMF's free Embedding(n_items, K_u) living in
            # user-prototype-similarity space (the host's free item vector t ∈ R^{K_u}).
            # `hist_value_ids`/`hist_weights`/`n_features` are injected into user_ft_ext_param by
            # feature_ids.inject_feature_ids (called in trainer/tester._build_model and the
            # cold/explanations seams) — never serialized into the Ray/JSON config.
            user_param = ft_ext_param['user_ft_ext_param']
            user_n_prototypes = user_param['n_prototypes']

            assert ft_ext_param['item_ft_ext_param']['ft_type'] == 'embedding', \
                "feature_user_proto (fU host) expects a plain 'embedding' item branch " \
                f"(got {ft_ext_param['item_ft_ext_param']['ft_type']!r})"
            assert not user_param['use_weight_matrix'], \
                'use_weight_matrix must be off: the item vector lives in R^{K_u} (host User-Proto shape)'
            assert 'hist_value_ids' in user_param and 'hist_weights' in user_param \
                   and 'n_features' in user_param, \
                "hist_value_ids/hist_weights/n_features not injected — call " \
                "feature_ids.inject_feature_ids in _build_model first"

            # --- User branch: history-composed embedding feeding the prototype layer ---
            user_max_norm = user_param['max_norm'] if 'max_norm' in user_param else None
            use_id_feature = user_param['use_id_feature'] if 'use_id_feature' in user_param else True

            user_hist_embed = HistoryFeatureEmbedding(n_users, user_param['hist_value_ids'],
                                                      user_param['hist_weights'],
                                                      user_param['n_features'], embedding_dim,
                                                      use_id_feature=use_id_feature,
                                                      max_norm=user_max_norm,
                                                      **_loo_kwargs(user_param))
            user_feature_extractor = PrototypeEmbedding(
                n_users, embedding_dim,
                n_prototypes=user_n_prototypes,
                use_weight_matrix=False,
                sim_proto_weight=user_param['sim_proto_weight'] if 'sim_proto_weight' in user_param else 1.,
                sim_batch_weight=user_param['sim_batch_weight'] if 'sim_batch_weight' in user_param else 1.,
                reg_proto_type=user_param['reg_proto_type'] if 'reg_proto_type' in user_param else 'soft',
                reg_batch_type=user_param['reg_batch_type'] if 'reg_batch_type' in user_param else 'soft',
                cosine_type=user_param['cosine_type'] if 'cosine_type' in user_param else 'shifted',
                max_norm=user_max_norm,
                embedding_ext=user_hist_embed)

            # --- Item branch: U-ProtoMF free item vector, dimension K_u (host 'prototypes' path) ---
            item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'],
                                                                          n_items, user_n_prototypes)

            # Init ownership (dc01 F-DC01-10 pattern): the FACTORY initializes the history word
            # table, exactly once. PrototypeEmbedding never inits a passed-in ext (its contract),
            # and RecSys.init_parameters cascades only one level.
            user_hist_embed.init_parameters()

            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'attr_item_proto':
            # dc02 — Attribute-space item prototypes (concept-bottleneck-anchored prototype layer).
            # Cloned from 'prototypes_double_tie'. The USER side is byte-identical. On the ITEM side
            # there are NO per-item parameters at all: the item is its observed multi-hot x_i
            # (AttributeLookup, a non-persistent buffer) and prototypes live in attribute space
            # (AttributePrototypeEmbedding, A ∈ R^{K×V}); the projection is a bias-free Linear(V, L_u)
            # (AttributeProjection). The tie (R1) is realized by SHARING the one AttributeLookup
            # instance between both item branches — both halves of the UI-score consume the same x_i.
            # `attr_multi_hot` / `n_attr_values` / `field_offsets` are injected into item_ft_ext_param
            # by feature_ids.inject_feature_ids (called in trainer/tester._build_model) — never
            # serialized into the Ray/JSON config. NOTE: dc02 requires use_bias=0 in rec_sys_param
            # (a per-item bias would be a bottleneck side channel — design doc §3.2/§3.5(e)).
            item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']
            user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']
            user_use_weight_matrix = ft_ext_param['user_ft_ext_param']['use_weight_matrix']
            item_use_weight_matrix = ft_ext_param['item_ft_ext_param']['use_weight_matrix']

            assert not user_use_weight_matrix and not item_use_weight_matrix, 'Use Weight Matrix should be turned off to tie the weights!'

            # --- User branch: identical to prototypes_double_tie (pure CF, weight-tied) ---
            ft_ext_param['user_ft_ext_param']['ft_type'] = 'prototypes'
            user_proto = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)
            ft_ext_param['user_ft_ext_param']['ft_type'] = 'embedding_w'
            ft_ext_param['user_ft_ext_param']['out_dimension'] = item_n_prototypes
            user_embed = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)
            user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight

            # --- Item branch: attribute-space, ONE shared AttributeLookup instance ---
            item_param = ft_ext_param['item_ft_ext_param']
            assert 'attr_multi_hot' in item_param and 'field_offsets' in item_param, \
                "attr_multi_hot/field_offsets not injected — call feature_ids.inject_feature_ids in _build_model first"
            # Alignment guard (mirrors dc01's FeatureEmbedding assert): the multi-hot matrix MUST
            # have exactly one row per item. build_attr_multi_hot only checks item_id covers
            # 0..len(csv)-1 against its OWN row count, so a STALE item_features.csv (split
            # regenerated, item_ids.csv changed) would otherwise build silently and train/evaluate/
            # explain every item i with a DIFFERENT article's attributes (row i of the stale matrix).
            assert item_param['attr_multi_hot'].shape[0] == n_items, \
                (f"attr_multi_hot has {item_param['attr_multi_hot'].shape[0]} rows but the dataset has "
                 f"{n_items} items — stale/mismatched item_features.csv for this split")

            attr_lookup = AttributeLookup(item_param['attr_multi_hot'])
            item_proto = AttributePrototypeEmbedding(
                n_items, attr_lookup,
                n_prototypes=item_param['n_prototypes'],
                field_offsets=item_param['field_offsets'],
                sim_proto_weight=item_param['sim_proto_weight'] if 'sim_proto_weight' in item_param else 1.,
                sim_batch_weight=item_param['sim_batch_weight'] if 'sim_batch_weight' in item_param else 1.,
                reg_proto_type=item_param['reg_proto_type'] if 'reg_proto_type' in item_param else 'soft',
                reg_batch_type=item_param['reg_batch_type'] if 'reg_batch_type' in item_param else 'soft',
                cosine_type=item_param['cosine_type'] if 'cosine_type' in item_param else 'shifted',
                nonneg_prototypes=item_param['nonneg_prototypes'] if 'nonneg_prototypes' in item_param else False,
                crisp_weight=item_param['crisp_weight'] if 'crisp_weight' in item_param else 0.,
                sep_weight=item_param['sep_weight'] if 'sep_weight' in item_param else 0.,
                sep_margin=item_param['sep_margin'] if 'sep_margin' in item_param else 0.5,
                push_weight=item_param['push_weight'] if 'push_weight' in item_param else 0.,
                push_n_items=item_param['push_n_items'] if 'push_n_items' in item_param else 256,
                push_seed=item_param['push_seed'] if 'push_seed' in item_param else 98765)
            item_proj = AttributeProjection(attr_lookup, out_dimension=user_n_prototypes)

            # No shared learnable table here (the lookup is parameter-free): the prototypes keep
            # their construction randn (host semantics) and AttributeProjection inits only its own
            # linear via init_parameters — nothing for the factory to single-owner-init.

            user_feature_extractor = ConcatenateFeatureExtractors(user_proto, user_embed, invert=False)
            item_feature_extractor = ConcatenateFeatureExtractors(item_proto, item_proj, invert=True)

            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'lightfm':
            # S0.4 — LightFM-style CBF baseline (B5, Kula 2015): rows `lightfm_tags` (feature-only,
            # use_id_feature=False) and `lightfm_tags_ids` (features + ID, B5's best variant).
            # `detached`-style: user = plain CF Embedding (S0.2 side disposition; B5's users are
            # indicator-only in these variants), item = the dc01-tested FeatureEmbedding
            # (q_i = Σ_f e_f over the S0.5 canonical field set, optional per-item ID row).
            # Plain dot-product score, NO prototype machinery; bias-free per fleet parity
            # (use_bias=0 in base_param — declared deviation from B5's σ(q_u·p_i + b_u + b_i)).
            # With use_id_feature=True and zero fields this reduces bit-identically to `mf`
            # (keystone dc_checks/s0/t03). `feature_ids`/`n_features` are injected into
            # item_ft_ext_param by feature_ids.inject_feature_ids — never serialized into the config.
            user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users,
                                                                          embedding_dim)

            item_param = ft_ext_param['item_ft_ext_param']
            assert 'feature_ids' in item_param and 'n_features' in item_param, \
                "feature_ids/n_features not injected — call feature_ids.inject_feature_ids in _build_model first"
            item_max_norm = item_param['max_norm'] if 'max_norm' in item_param else None
            use_id_feature = item_param['use_id_feature'] if 'use_id_feature' in item_param else True
            # bag layout (S0-build extension): weights injected alongside feature_ids when the
            # dataset's canonical layout is 'bags'; absent = the original fixed path.
            feature_weights = item_param['feature_weights'] if 'feature_weights' in item_param else None

            item_feature_extractor = FeatureEmbedding(n_items, item_param['feature_ids'], item_param['n_features'],
                                                      embedding_dim, use_id_feature=use_id_feature,
                                                      max_norm=item_max_norm, feature_weights=feature_weights)
            # No sharing here (single consumer): RecSys.init_parameters initializes both extractors
            # directly — no factory-side single-owner init needed (unlike the dc01 branch).
            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'lightfm_hist':
            # dc05 S1 decoupled control (F-DC05-21, dc05 SC.8 gate 2026-08-20): the fU user
            # representation WITHOUT the prototype layer — the user-side mirror of the 'lightfm'
            # rows above. User = HistoryFeatureEmbedding (q_u = Σ_f w̄_{u,f}·e_f over the train
            # basket's attribute words, optional per-user ID row — byte-identical machinery to the
            # fU branch), item = plain CF Embedding in R^d, plain dot-product score, NO prototype
            # machinery. Rows: `lightfm_hist` (history-only, use_id_feature=False) and
            # `lightfm_hist_ids` (+ per-user ID row) — naming mirrors lightfm_tags/_ids. With zero
            # words and ID on this reduces to a per-user embedding lookup (= `mf`'s user branch;
            # keystone pinned in dc_checks/dc05/t13). Bias-free fleet parity (use_bias=0).
            # `hist_value_ids`/`hist_weights`/`n_features` are injected into user_ft_ext_param by
            # feature_ids.inject_feature_ids — never serialized into the Ray/JSON config.
            user_param = ft_ext_param['user_ft_ext_param']
            assert ft_ext_param['item_ft_ext_param']['ft_type'] == 'embedding', \
                "lightfm_hist expects a plain 'embedding' item branch " \
                f"(got {ft_ext_param['item_ft_ext_param']['ft_type']!r})"
            assert 'hist_value_ids' in user_param and 'hist_weights' in user_param \
                   and 'n_features' in user_param, \
                "hist_value_ids/hist_weights/n_features not injected — call " \
                "feature_ids.inject_feature_ids in _build_model first"

            user_max_norm = user_param['max_norm'] if 'max_norm' in user_param else None
            use_id_feature = user_param['use_id_feature'] if 'use_id_feature' in user_param else True

            user_feature_extractor = HistoryFeatureEmbedding(n_users, user_param['hist_value_ids'],
                                                             user_param['hist_weights'],
                                                             user_param['n_features'], embedding_dim,
                                                             use_id_feature=use_id_feature,
                                                             max_norm=user_max_norm,
                                                             **_loo_kwargs(user_param))
            item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'],
                                                                          n_items, embedding_dim)
            # No wrapper here (single consumer): RecSys.init_parameters initializes both extractors
            # directly — no factory-side single-owner init needed (unlike the fU branch, where
            # PrototypeEmbedding wraps the history embedding).
            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'acf':
            # Anchor-based collaborative filtering

            n_anchors = ft_ext_param['n_anchors']
            delta_exc = ft_ext_param['delta_exc']
            delta_inc = ft_ext_param['delta_inc']
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            # Create shared parameters
            anchors = nn.Parameter(torch.randn(n_anchors, embedding_dim))

            user_feature_extractor = AnchorBasedCollaborativeFiltering(n_users, embedding_dim, anchors,
                                                                       max_norm=max_norm)
            item_feature_extractor = AnchorBasedCollaborativeFiltering(n_items, embedding_dim, anchors, delta_exc,
                                                                       delta_inc, max_norm=max_norm)
            return user_feature_extractor, item_feature_extractor
        else:
            raise ValueError(f'FeatureExtractor <{ft_type}> Not Implemented..yet')

    @staticmethod
    def create_model(ft_ext_param: dict, n_objects: int, embedding_dim: int) -> FeatureExtractor:
        """
        Creates the specified FeatureExtractor model by reading the ft_ext_param. Currently available:
        - Embedding: represents objects by learning an embedding, A.K.A. Collaborative Filtering.
        - EmbeddingW: As Embedding but followed by a linear layer.
        - PrototypeEmbedding: represents an object by the similarity to the prototypes.

        :param ft_ext_param: parameters specific for the model type. ft_ext_param.ft_type is used for switching between
                models.
        :param embedding_dim: dimension of the final embeddings
        :param n_objects: number of objects in the system
        """

        ft_type = ft_ext_param["ft_type"]

        print('--- Building FeatureExtractor model ---')
        if ft_type == 'embedding':
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            only_positive = ft_ext_param['only_positive'] if 'only_positive' in ft_ext_param else False
            model = Embedding(n_objects, embedding_dim, max_norm, only_positive)
        elif ft_type == 'embedding_w':
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            out_dimension = ft_ext_param['out_dimension'] if 'out_dimension' in ft_ext_param else None
            use_bias = ft_ext_param['use_bias'] if 'use_bias' in ft_ext_param else False
            model = EmbeddingW(n_objects, embedding_dim, max_norm, out_dimension, use_bias)
        elif ft_type == 'prototypes':
            n_prototypes = ft_ext_param['n_prototypes'] if 'n_prototypes' in ft_ext_param else None
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            sim_proto_weight = ft_ext_param['sim_proto_weight'] if 'sim_proto_weight' in ft_ext_param else 1.
            sim_batch_weight = ft_ext_param['sim_batch_weight'] if 'sim_batch_weight' in ft_ext_param else 1.
            reg_proto_type = ft_ext_param['reg_proto_type'] if 'reg_proto_type' in ft_ext_param else 'soft'
            reg_batch_type = ft_ext_param['reg_batch_type'] if 'reg_batch_type' in ft_ext_param else 'soft'
            cosine_type = ft_ext_param['cosine_type'] if 'cosine_type' in ft_ext_param else 'shifted'
            use_weight_matrix = ft_ext_param['use_weight_matrix'] if 'use_weight_matrix' in ft_ext_param else False

            model = PrototypeEmbedding(n_objects, embedding_dim, n_prototypes, use_weight_matrix, sim_proto_weight,
                                       sim_batch_weight, reg_proto_type, reg_batch_type, cosine_type, max_norm)

        elif ft_type == 'acf':
            n_anchors = ft_ext_param['n_anchors']
            delta_exc = ft_ext_param['delta_exc']
            delta_inc = ft_ext_param['delta_inc']
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None

            anchors = nn.Parameter(torch.randn(n_anchors, embedding_dim))

            model = AnchorBasedCollaborativeFiltering(n_objects, embedding_dim, anchors, delta_exc, delta_inc,
                                                      max_norm=max_norm)

        else:
            raise ValueError(f'FeatureExtractor <{ft_type}> Not Implemented..yet')

        print('--- Finished building FeatureExtractor model ---\n')
        return model
