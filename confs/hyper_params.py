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

# dc01 — feature_item_proto. Mirrors proto_double_tie_chose_original_hyper_params exactly so that
# it is directly comparable to the user_item_proto baseline (same search space + dev profile),
# with two additions on the ITEM side only: `use_id_feature` (LightFM "tags + ids") and the
# lightweight `feature_fields` spec (the actual feature_ids tensor is built in _build_model, never
# serialized into the config). The 5 fields below are the categorical columns present in the H&M
# item_features.csv; price-band (dc01 §3.5, F=6) is added by P7 later and is optional for this model.
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
            "ft_type": "feature_item_proto",
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
