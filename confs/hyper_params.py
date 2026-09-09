import copy

import torch
from ray import tune

base_param = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 100,
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256,
    'rec_sys_param': {'use_bias': 0},
}

base_hyper_params = {
    **base_param,
    'neg_train': tune.randint(1, 50),
    'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'loss_func_name': tune.choice(['bce', 'bpr', 'sampled_softmax']),
    'batch_size': tune.choice([64, 128, 256, 512]),
    'optim_param': {
        'optim': tune.choice(['adam', 'adagrad']),
        'wd': tune.loguniform(1e-4, 1e-2),
        'lr': tune.loguniform(1e-4, 1e-1)
    },
}

mf_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "detached",
        'embedding_dim': tune.randint(10, 100),
        'user_ft_ext_param': {
            "ft_type": "embedding",
        },
        'item_ft_ext_param': {
            "ft_type": "embedding",
        }
    },
}
anchor_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'sum',
    'ft_ext_param': {
        "ft_type": "acf",
        'embedding_dim': tune.randint(10, 100),
        'n_anchors': tune.randint(10, 100),
        'delta_exc': tune.loguniform(1e-2, 10),
        'delta_inc': tune.loguniform(1e-2, 10),
    },
}

user_proto_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes",
        'embedding_dim': tune.randint(10, 100),
        'user_ft_ext_param': {
            "ft_type": "prototypes",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
        },
        'item_ft_ext_param': {
            "ft_type": "embedding",
        }
    },
}

item_proto_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes",
        'embedding_dim': tune.randint(10, 100),
        'item_ft_ext_param': {
            "ft_type": "prototypes",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max'
        },
        'user_ft_ext_param': {
            "ft_type": "embedding",
        }
    },
}
# Debug config: fixed tiny values for fast local CPU smoke tests.
# Use with: python start.py -m debug -d ml-1m
# Runs a single trial with no hyperparameter search.
debug_hyper_params = {
    **base_param,
    'num_samples': 1,
    'n_epochs': 3,
    'neg_train': 5,
    'train_neg_strategy': 'uniform',
    'loss_func_name': 'bce',
    'loss_func_aggr': 'mean',
    'batch_size': 128,
    'optim_param': {
        'optim': 'adam',
        'wd': 1e-3,
        'lr': 1e-3,
    },
    'ft_ext_param': {
        "ft_type": "prototypes_double_tie",
        'embedding_dim': 8,
        'item_ft_ext_param': {
            "ft_type": "prototypes_double_tie",
            'sim_proto_weight': 1.0,
            'sim_batch_weight': 1.0,
            'use_weight_matrix': False,
            'n_prototypes': 3,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
        },
        'user_ft_ext_param': {
            "ft_type": "prototypes_double_tie",
            'sim_proto_weight': 1.0,
            'sim_batch_weight': 1.0,
            'use_weight_matrix': False,
            'n_prototypes': 3,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
        },
    },
}

# W&B integration test: 10 trials, 15 epochs, item_proto with small search space.
# Use with: python start.py -m wandb_test -d ml-1m
wandb_test_hyper_params = {
    **base_param,
    'num_samples': 10,
    'n_epochs': 15,
    'neg_train': tune.randint(5, 20),
    'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'loss_func_name': tune.choice(['bce', 'bpr']),
    'loss_func_aggr': 'mean',
    'batch_size': tune.choice([128, 256]),
    'optim_param': {
        'optim': 'adam',
        'wd': tune.loguniform(1e-4, 1e-2),
        'lr': tune.loguniform(1e-4, 1e-2),
    },
    'ft_ext_param': {
        "ft_type": "prototypes",
        'embedding_dim': tune.randint(16, 64),
        'item_ft_ext_param': {
            "ft_type": "prototypes",
            'sim_proto_weight': tune.loguniform(1e-2, 5),
            'sim_batch_weight': tune.loguniform(1e-2, 5),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 50),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
        },
        'user_ft_ext_param': {
            "ft_type": "embedding",
        }
    },
}

proto_double_tie_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes_double_tie",
        'embedding_dim': tune.randint(10, 100),
        'item_ft_ext_param': {
            "ft_type": "prototypes_double_tie",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max'
        },
        'user_ft_ext_param': {
            "ft_type": "prototypes_double_tie",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max'
        },
    },
}

# dc01 — feature_item_proto (fI-ProtoMF; re-hosted 2026-07-12 from the UI double-tie to the
# I-ProtoMF host). Mirrors item_proto_chose_original_hyper_params EXACTLY so that it is directly
# comparable to the item_proto baseline (same search space + dev profile), with two additions on
# the ITEM side only: `use_id_feature` (LightFM "tags + ids") and the lightweight `feature_fields`
# spec (the actual feature_ids tensor is built in _build_model, never serialized into the config).
# `feature_fields: 'canonical'` resolves per dataset (charter C5 as amended — H&M: the S0.5 five,
# V=426; ml-1m: genres+tags@0.8 bags); price_band is built but deferred
# per the S0.5 gate decision (F-DC01-01). User side = plain CF embedding (host property: the free
# user vector lives in R^{K_t}, sized by the factory from the item side's n_prototypes).
feature_item_proto_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "feature_item_proto",
        'embedding_dim': tune.randint(10, 100),
        'item_ft_ext_param': {
            "ft_type": "feature_item_proto",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
            'use_id_feature': True,
            # per-dataset canonical field set (charter C5 as amended 2026-07-12): resolved by
            # experiment_helper.start_hyper against feature_ids.CANONICAL_FEATURE_FIELDS
            # (hm_* -> the S0.5 five, layout 'fixed'; ml-1m* -> genres+tags@0.8, layout 'bags');
            # the RESOLVED list + layout land in every trial config / C8 manifest.
            'feature_fields': 'canonical',
        },
        'user_ft_ext_param': {
            "ft_type": "embedding",
        },
    },
}

# dc01 ablations (§3.6). Each is feature_item_proto with exactly ONE item-side knob changed, so the
# delta vs the default isolates that knob. Same search space + dev profile → directly comparable.

# Ablation A — no ID feature: the item factor is the feature composition ALONE (q_i = Σ_f e_f, no
# per-item row). Maximizes the feature-explained fraction (R4) but caps capacity at the no-ID
# equivalence-class ceiling (hm_1_month: 6,517 distinct 5-field tuples among 13,651 items).
feature_item_proto_noid_hyper_params = copy.deepcopy(feature_item_proto_hyper_params)
feature_item_proto_noid_hyper_params['ft_ext_param']['item_ft_ext_param']['use_id_feature'] = False

# Ablation B — F=0 reduction: ID feature ON, ZERO metadata fields → architecturally reduces to
# item_proto (I-ProtoMF; proven bit-identical, keystone i02 / fI re-check C5′). Any delta vs the
# item_proto baseline is then attributable to the metadata, not the wrapper — the exact isolating
# control.
feature_item_proto_f0_hyper_params = copy.deepcopy(feature_item_proto_hyper_params)
feature_item_proto_f0_hyper_params['ft_ext_param']['item_ft_ext_param']['feature_fields'] = []

# ── dc06 fI′ — field-mean-centred composition (experimental wave, ml-1m only; design doc
# `Master/docs/design_candidates/dc06_fi_prime_field_mean_centering.md` §6 run set). Each arm
# is feature_item_proto with the centring knobs changed and NOTHING else (same search space →
# fleet-comparable under its own dev-profile hyperopt, user decision 2026-08-24). The A11
# gauge snap rides `center_fields` automatically (Trainer hook). `center_mode` is never
# searched — fixed per arm (dc06 §3.1).

# fI′-ids: centred composition, ID row kept (weighted = catalog-frequency field means).
feature_item_proto_c_hyper_params = copy.deepcopy(feature_item_proto_hyper_params)
feature_item_proto_c_hyper_params['ft_ext_param']['item_ft_ext_param']['center_fields'] = True
feature_item_proto_c_hyper_params['ft_ext_param']['item_ft_ext_param']['center_mode'] = 'weighted'

# fI′-noid: the A3 PRIMARY-endpoint arm (vs feature_item_proto_noid).
feature_item_proto_c_noid_hyper_params = copy.deepcopy(feature_item_proto_c_hyper_params)
feature_item_proto_c_noid_hyper_params['ft_ext_param']['item_ft_ext_param']['use_id_feature'] = False

# fI′-noid-iw: interaction-weighted field means (A7 flavour arm, ml-1m only).
feature_item_proto_c_noid_iw_hyper_params = copy.deepcopy(feature_item_proto_c_noid_hyper_params)
feature_item_proto_c_noid_iw_hyper_params['ft_ext_param']['item_ft_ext_param']['center_mode'] = 'interaction'

# α-control (A1): NO centring — constant down-weight of the metadata sum to match fI′'s
# trained metadata/ID balance. metadata_scale is a PLACEHOLDER here: the real α is measured
# from the finished fI′-ids checkpoint (post-centring mean metadata norm ratio) and set at
# queue time; queuing this arm with α=1.0 is a plain fI re-run and therefore a mistake.
feature_item_proto_alpha_hyper_params = copy.deepcopy(feature_item_proto_hyper_params)
feature_item_proto_alpha_hyper_params['ft_ext_param']['item_ft_ext_param']['metadata_scale'] = 1.0  # SET AT QUEUE TIME

# dc05 — feature_user_proto (fU-ProtoMF: history-composed user factors on the U-ProtoMF host).
# Mirrors user_proto_chose_original_hyper_params EXACTLY so it is directly comparable to the
# user_proto baseline (same search space + dev profile — the M1 stage bar, like-for-like), with
# two additions on the USER side only: `use_id_feature` (the per-user ID row e_ID(u)) and the
# lightweight `feature_fields` spec (dc01 key names on purpose — the actual hist_value_ids/
# hist_weights tensors are built in _build_model from the split's OWN train file via
# inject_feature_ids, never serialized into the config). `feature_fields: 'canonical'` resolves
# per dataset (charter C5 as amended), applied to user purchase histories (design doc §3.5) —
# on ml-1m via the bags layout (every token of every rated movie). Item side = plain CF
# embedding (host property: the free item vector lives in R^{K_u}, sized by the factory from the
# user side's n_prototypes).
feature_user_proto_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "feature_user_proto",
        'embedding_dim': tune.randint(10, 100),
        'user_ft_ext_param': {
            "ft_type": "feature_user_proto",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
            'use_id_feature': True,
            'loo_pooling': True,  # train-time leave-one-out pooling (P5 hygiene, 2026-09-08)
            # per-dataset canonical field set (charter C5 as amended 2026-07-12): resolved by
            # experiment_helper.start_hyper against feature_ids.CANONICAL_FEATURE_FIELDS
            # (hm_* -> the S0.5 five, layout 'fixed'; ml-1m* -> genres+tags@0.8, layout 'bags');
            # the RESOLVED list + layout land in every trial config / C8 manifest.
            'feature_fields': 'canonical',
        },
        'item_ft_ext_param': {
            "ft_type": "embedding",
        },
    },
}

# dc05 isolating ablation (§3.6, charter C7 / F-DC01-01: never a searched flag): no ID row — the
# user factor is the basket composition ALONE (q_u = Σ_f w̄_{u,f}·e_f). Representational ceiling
# is the basket-signature twin rate (hm_1_month: 0.61% of users share a normalized basket vector).
feature_user_proto_noid_hyper_params = copy.deepcopy(feature_user_proto_hyper_params)
feature_user_proto_noid_hyper_params['ft_ext_param']['user_ft_ext_param']['use_id_feature'] = False

# ---------------------------------------------------------------------------------------------
# dc08 — item-identity history rows in the fU composition (SVD++/FISM y_j as a summand of the
# purchase mean). Knobs on `feature_user_proto`, NOT a new ft_type (dc06/dc07 precedent):
#   history_id_rows    — add (id = n_features + j, weight = λ_y/|H_u|) per purchase
#   history_pooling    — 'auto': flat embedding_bag when the bag is wide (5b A10). The identity
#                        slots widen the bag by max_u |H_u|, which is ~6.5 on hm (padded stays)
#                        but in the low thousands on ml-1m, where the padded gather would retain
#                        a ~0.5-1 GB (B, D_max, d) autograd intermediate that is ~97% padding.
#   history_row_weight — λ_y (1.0 = the literal SVD++ form under the mean; 0.0 ≡ dc05 exactly)
#   freeze_history_rows— 5b A2 capacity control: block present, frozen at random init
# Never searched flags (charter C7 / F-DC01-01): each arm is its own registered row.
#
# The four-arm design (candidate §3.6 as amended):
#   A  feature_user_proto_noid        words only                      (exists, dc05)
#   B  feature_user_proto             words + user row                (exists, dc05)
#   C  feature_user_proto_noid_y      words + item identity rows      (below) ← reporting arm
#   D  feature_user_proto_y           words + user row + identity     (below)
#   A' feature_user_proto_noid_yfrozen  capacity control for C        (below, 5b A2)
# Reporting-arm note (5b A1 / 4b C9): the three-way named/identity/personal read-out is
# identified only WITHOUT the user-ID row, i.e. in arm C; arm D quotes named-vs-unnamed only.
feature_user_proto_y_hyper_params = copy.deepcopy(feature_user_proto_hyper_params)
feature_user_proto_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_id_rows'] = True
feature_user_proto_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_row_weight'] = 1.0
feature_user_proto_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_pooling'] = 'auto'

feature_user_proto_noid_y_hyper_params = copy.deepcopy(feature_user_proto_noid_hyper_params)
feature_user_proto_noid_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_id_rows'] = True
feature_user_proto_noid_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_row_weight'] = 1.0
feature_user_proto_noid_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_pooling'] = 'auto'

# 5b A2 — capacity control: identical parameter count and identical composition arithmetic to
# arm C, zero learned identity signal (gradient masked on the identity block). C − A' isolates
# the mechanism from the capacity the mechanism brings with it.
feature_user_proto_noid_yfrozen_hyper_params = copy.deepcopy(feature_user_proto_noid_y_hyper_params)
feature_user_proto_noid_yfrozen_hyper_params['ft_ext_param']['user_ft_ext_param']['freeze_history_rows'] = True

# dc05 local CPU smoke: single trial, fixed tiny values (dc02 debug convention — NOT a fleet row).
# Use with: python Master/scripts/run_combo.py -m feature_user_proto_debug -d hm_1_month --skip-explanations
feature_user_proto_debug_hyper_params = {
    **base_param,
    'num_samples': 1,
    'n_epochs': 2,
    'neg_train': 5,
    'train_neg_strategy': 'uniform',
    'loss_func_name': 'bce',
    'loss_func_aggr': 'mean',
    'batch_size': 512,
    'optim_param': {
        'optim': 'adam',
        'wd': 1e-3,
        'lr': 1e-3,
    },
    'ft_ext_param': {
        "ft_type": "feature_user_proto",
        'embedding_dim': 8,
        'user_ft_ext_param': {
            "ft_type": "feature_user_proto",
            'sim_proto_weight': 1.0,
            'sim_batch_weight': 1.0,
            'use_weight_matrix': False,
            'n_prototypes': 4,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
            'use_id_feature': True,
            'loo_pooling': True,  # train-time leave-one-out pooling (P5 hygiene, 2026-09-08)
            # per-dataset canonical field set (charter C5 as amended 2026-07-12): resolved by
            # experiment_helper.start_hyper against feature_ids.CANONICAL_FEATURE_FIELDS
            # (hm_* -> the S0.5 five, layout 'fixed'; ml-1m* -> genres+tags@0.8, layout 'bags');
            # the RESOLVED list + layout land in every trial config / C8 manifest.
            'feature_fields': 'canonical',
        },
        'item_ft_ext_param': {
            "ft_type": "embedding",
        },
    },
}

# S0.4 — LightFM-style CBF baselines (`ft_type` 'lightfm'; B5 = Kula 2015). Search space mirrors
# mf_hyper_params EXACTLY (same emb-dim range, loss, optimizer, negatives) — the row isolates
# "features (± ID) without prototypes." Item side adds only the lightweight spec: per-dataset
# canonical fields (`'canonical'`, resolved by start_hyper) + `use_id_feature` (the actual
# feature_ids tensor is built in _build_model via inject_feature_ids, never serialized into
# the config).
# - lightfm_tags_ids (use_id_feature=True): the strong feature-aware baseline (B5's best variant).
# - lightfm_tags (use_id_feature=False): the feature-only baseline, natively cold-capable.
# Bias-free fleet parity (base_param use_bias=0) is a declared deviation from B5's published form.
lightfm_tags_ids_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "lightfm",
        'embedding_dim': tune.randint(10, 100),
        'user_ft_ext_param': {
            "ft_type": "embedding",
        },
        'item_ft_ext_param': {
            "ft_type": "lightfm",
            'use_id_feature': True,
            # per-dataset canonical field set (charter C5 as amended 2026-07-12): resolved by
            # experiment_helper.start_hyper against feature_ids.CANONICAL_FEATURE_FIELDS
            # (hm_* -> the S0.5 five, layout 'fixed'; ml-1m* -> genres+tags@0.8, layout 'bags');
            # the RESOLVED list + layout land in every trial config / C8 manifest.
            'feature_fields': 'canonical',
        },
    },
}

lightfm_tags_hyper_params = copy.deepcopy(lightfm_tags_ids_hyper_params)
lightfm_tags_hyper_params['ft_ext_param']['item_ft_ext_param']['use_id_feature'] = False

# dc05 S1 decoupled control (`ft_type` 'lightfm_hist'; F-DC05-21, dc05 SC.8 gate 2026-08-20):
# the fU user representation WITHOUT the prototype layer — user-side mirror of the lightfm rows
# above. User = history-composed HistoryFeatureEmbedding (identical builder/injection to
# feature_user_proto), item = plain CF Embedding, plain dot score. Search space mirrors
# mf_hyper_params/lightfm EXACTLY (same emb-dim range, loss, optimizer, negatives — and thereby
# fU's own space minus the prototype knobs): the row isolates "history composition (± ID)
# without prototypes." Rows (naming mirrors lightfm_tags/_ids):
# - lightfm_hist (use_id_feature=False): pure basket composition, the S1 spectrum mid-point.
# - lightfm_hist_ids (use_id_feature=True): composition + per-user ID row (fU-headline mirror).
lightfm_hist_ids_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "lightfm_hist",
        'embedding_dim': tune.randint(10, 100),
        'user_ft_ext_param': {
            "ft_type": "lightfm_hist",
            'use_id_feature': True,
            'loo_pooling': True,  # train-time leave-one-out pooling (P5 hygiene, 2026-09-08)
            # per-dataset canonical field set (charter C5 as amended 2026-07-12): resolved by
            # experiment_helper.start_hyper against feature_ids.CANONICAL_FEATURE_FIELDS
            # (hm_* -> the S0.5 five, layout 'fixed'; ml-1m* -> genres+tags@0.8, layout 'bags'),
            # applied to user purchase histories; RESOLVED list + layout land in every C8 manifest.
            'feature_fields': 'canonical',
        },
        'item_ft_ext_param': {
            "ft_type": "embedding",
        },
    },
}

lightfm_hist_hyper_params = copy.deepcopy(lightfm_hist_ids_hyper_params)
lightfm_hist_hyper_params['ft_ext_param']['user_ft_ext_param']['use_id_feature'] = False

# dc08 dot-product twins (5b A6): the paired 2x2 head x identity design. `lightfm_hist(_ids)`
# already gives the {dot} x {no identity} cells; these give {dot} x {identity}. Running all four
# separates the HOST penalty (dot vs prototype on the same composition) from the CHANNEL
# (identity rows) and their interaction — the one-way "cheap gate" reading was retired at 5b.
lightfm_hist_y_hyper_params = copy.deepcopy(lightfm_hist_hyper_params)
lightfm_hist_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_id_rows'] = True
lightfm_hist_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_row_weight'] = 1.0
lightfm_hist_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_pooling'] = 'auto'

lightfm_hist_ids_y_hyper_params = copy.deepcopy(lightfm_hist_ids_hyper_params)
lightfm_hist_ids_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_id_rows'] = True
lightfm_hist_ids_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_row_weight'] = 1.0
lightfm_hist_ids_y_hyper_params['ft_ext_param']['user_ft_ext_param']['history_pooling'] = 'auto'

# dc02 — attr_item_proto (attribute-space item prototypes, concept-bottleneck-anchored).
# User side byte-identical to proto_double_tie_chose_original_hyper_params (same search space →
# directly comparable to the user_item_proto baseline). Item side: prototypes live in attribute
# space, so there is NO item-side embedding_dim (its "dimension" is V, data-determined from
# `attr_fields`) and NO max_norm (inert — there is no item embedding). `attr_fields` = all 9
# categorical columns of item_features.csv (design doc §3.1; hm_3_month → V=534). The four §3.5
# constraint knobs are config-gated and DEFAULT OFF (fixed values, not tune objects — Rashomon
# tuning is a later, deliberate step). NOTE: dc02 requires use_bias=0 (bottleneck side channel,
# §3.5(e)) — already the base_param default.
attr_item_proto_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "attr_item_proto",
        'embedding_dim': tune.randint(10, 100),          # user-side CF dim d only
        'item_ft_ext_param': {
            "ft_type": "attr_item_proto",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),       # K — same range as host (§3.2)
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
            'attr_fields': [                              # all 9, item_features.csv column order
                'product_group_name',
                'product_type_name',
                'graphical_appearance_name',
                'colour_group_name',
                'perceived_colour_master_name',
                'index_group_name',
                'garment_group_name',
                'section_name',
                'department_name',
            ],
            # --- dc02 §3.5 constraint knobs — DEFAULT OFF ---
            'nonneg_prototypes': False,   # K1 softplus reparam (binary mechanism knob)
            'crisp_weight': 0.0,          # K2 field-crispness entropy
            'sep_weight': 0.0,            # K3 separation hinge
            'sep_margin': 0.5,            # K3 margin m ∈ [0,1)
            'push_weight': 0.0,           # K4 data pull
            'push_n_items': 256,          # K4 |S|, resampled every step
            'push_seed': 98765,           # K4 dedicated-generator seed (RNG isolation)
        },
        'user_ft_ext_param': {
            "ft_type": "attr_item_proto",
            'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': tune.loguniform(1e-3, 10),
            'use_weight_matrix': False,
            'n_prototypes': tune.randint(10, 100),
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max'
        },
    },
}

# dc02 local CPU smoke: single trial, fixed tiny values (the full search space samples
# unpredictably large dims for a laptop). Use with:
#   python Master/scripts/run_combo.py -m attr_item_proto_debug -d hm_3_month --skip-explanations
attr_item_proto_debug_hyper_params = {
    **base_param,
    'num_samples': 1,
    'n_epochs': 2,
    'neg_train': 5,
    'train_neg_strategy': 'uniform',
    'loss_func_name': 'bce',
    'loss_func_aggr': 'mean',
    'batch_size': 512,
    'optim_param': {
        'optim': 'adam',
        'wd': 1e-3,
        'lr': 1e-3,
    },
    'ft_ext_param': {
        "ft_type": "attr_item_proto",
        'embedding_dim': 8,
        'item_ft_ext_param': {
            "ft_type": "attr_item_proto",
            'sim_proto_weight': 1.0,
            'sim_batch_weight': 1.0,
            'use_weight_matrix': False,
            'n_prototypes': 4,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
            'attr_fields': [
                'product_group_name',
                'product_type_name',
                'graphical_appearance_name',
                'colour_group_name',
                'perceived_colour_master_name',
                'index_group_name',
                'garment_group_name',
                'section_name',
                'department_name',
            ],
            'nonneg_prototypes': False,
            'crisp_weight': 0.0,
            'sep_weight': 0.0,
            'sep_margin': 0.5,
            'push_weight': 0.0,
            'push_n_items': 256,
            'push_seed': 98765,
        },
        'user_ft_ext_param': {
            "ft_type": "attr_item_proto",
            'sim_proto_weight': 1.0,
            'sim_batch_weight': 1.0,
            'use_weight_matrix': False,
            'n_prototypes': 4,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_batch_type': 'max',
        },
    },
}

# dc02 knobs-on smoke: all four constraint knobs active at small weights — proves the constraint
# loss path trains end-to-end (NOT a tuned configuration).
attr_item_proto_debug_knobs_hyper_params = copy.deepcopy(attr_item_proto_debug_hyper_params)
attr_item_proto_debug_knobs_hyper_params['ft_ext_param']['item_ft_ext_param'].update({
    'nonneg_prototypes': True,
    'crisp_weight': 1e-2,
    'sep_weight': 1e-2,
    'push_weight': 1e-2,
    'push_n_items': 64,
})

# ══════════════════════════════════════════════════════════════════════════════════════════
# dc07 — membership similarity for the prototype layer (requirements memo
# Master/temp/dc07_membership_similarity_requirements_2026-09-08.md). Each arm is its cosine
# parent with EXACTLY two prototype-side keys changed: cosine_type ∈ {softmax, sigmoid} and the
# new temperature τ (D2: a searched hyperparameter, fixed per trial, never learned). Everything
# else — host, K/d spaces, both 'max' regularisers, loss, sampling, use_bias=0 — is inherited.
# τ range (pinned by dc_checks/dc07/t03 on 2026-09-08, stock init, d,K ∈ {10,30,60,100}): the
# uniform regime (mean H > 0.9·log K ⇒ score ≈ per-item mean of t, the popularity channel rebuilt)
# begins at τ ≈ 3.2 for the smallest host shape, so τ_max = 2.0 stays below it; the one-hot regime
# (H < 0.1·log K, saturated softmax, dead gradients) ends at τ ≈ 0.28–1.0 for the host shapes
# (≈ ‖q‖/10 with ‖q‖ ≈ √d), so τ_min = 0.1 sits one decade below the largest host hard-exit —
# the memo's 0.05 was deep in saturation for every shape. τ and ‖q‖ are jointly redundant during
# training; the range therefore fixes the INIT-time sharpness the search can start from. The
# composed fI shape has ≈2.5× the norm (hard-exit up to 2.5, uniform ≥ 7.9) — a stage-2 range
# shift is a recorded option, not a stage-1 concern.
# Suffixes (D4): _sm = softmax, _sg = sigmoid. Stage 1 queues the two _sm hosts only; the _sg
# twins and the composed fI/fU _sm arms are registered but not queued.
DC07_TAU_MIN = 0.1
DC07_TAU_MAX = 2.0


def _dc07_arm(parent: dict, side: str, mode: str) -> dict:
    arm = copy.deepcopy(parent)
    proto_side = arm['ft_ext_param'][f'{side}_ft_ext_param']
    proto_side['cosine_type'] = mode
    proto_side['temperature'] = tune.loguniform(DC07_TAU_MIN, DC07_TAU_MAX)
    return arm


# stage-1 hosts
user_proto_sm_hyper_params = _dc07_arm(user_proto_chose_original_hyper_params, 'user', 'softmax')
item_proto_sm_hyper_params = _dc07_arm(item_proto_chose_original_hyper_params, 'item', 'softmax')
# sigmoid twins (built, not queued)
user_proto_sg_hyper_params = _dc07_arm(user_proto_chose_original_hyper_params, 'user', 'sigmoid')
item_proto_sg_hyper_params = _dc07_arm(item_proto_chose_original_hyper_params, 'item', 'sigmoid')
# stage-2 composed arms (registered, queued only on a stage-1 C verdict)
feature_item_proto_sm_hyper_params = _dc07_arm(feature_item_proto_hyper_params, 'item', 'softmax')
feature_item_proto_noid_sm_hyper_params = _dc07_arm(feature_item_proto_noid_hyper_params, 'item', 'softmax')
feature_user_proto_sm_hyper_params = _dc07_arm(feature_user_proto_hyper_params, 'user', 'softmax')
feature_user_proto_noid_sm_hyper_params = _dc07_arm(feature_user_proto_noid_hyper_params, 'user', 'softmax')
