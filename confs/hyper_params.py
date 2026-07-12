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
# The 5 fields below are the S0.5 canonical field set (V=426); price_band is built but deferred
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
            'feature_fields': [
                'department_name',
                'product_type_name',
                'section_name',
                'colour_group_name',
                'graphical_appearance_name',
            ],
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

# S0.4 — LightFM-style CBF baselines (`ft_type` 'lightfm'; B5 = Kula 2015). Search space mirrors
# mf_hyper_params EXACTLY (same emb-dim range, loss, optimizer, negatives) — the row isolates
# "features (± ID) without prototypes." Item side adds only the lightweight spec: the S0.5
# canonical 5-field set + `use_id_feature` (the actual feature_ids tensor is built in
# _build_model via inject_feature_ids, never serialized into the config).
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
            'feature_fields': [
                'department_name',
                'product_type_name',
                'section_name',
                'colour_group_name',
                'graphical_appearance_name',
            ],
        },
    },
}

lightfm_tags_hyper_params = copy.deepcopy(lightfm_tags_ids_hyper_params)
lightfm_tags_hyper_params['ft_ext_param']['item_ft_ext_param']['use_id_feature'] = False

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
