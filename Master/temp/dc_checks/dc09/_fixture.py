# dc09 test fixture — one consistent toy world + the golden recipes. Imported by t00..t09.
# Not a test itself.
#
# The toy world mirrors the real payloads: F fields with disjoint vocabularies (global
# field-offset codes), a fixed one-word-per-field item table, an item bag table, user baskets,
# the user history built from those baskets (w = n_{u,f}/|H_u|), and the leave-one-out payload
# (the item word table + |H_u|). Training batches put a basket item in column 0, as the
# dataloader does.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_extractors import (FeatureEmbedding, HistoryFeatureEmbedding,
                                                   PrototypeEmbedding)
from rec_sys.rec_sys import RecSys
from utilities.utils import reproducible

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden_pre_sc3.pt')

VOCAB = (3, 4, 2)          # per-field vocabulary sizes -> V = 9
N_ITEMS, N_USERS = 12, 8
D, K_U, K_T = 6, 3, 4
B, N_NEG = 5, 3


def make_check():
    fails = []

    def check(name, cond, evidence=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
        if not cond:
            fails.append(name)

    return check, fails


def toy_world(seed=0, n_items=N_ITEMS, n_users=N_USERS, vocab=VOCAB, empty_users=()):
    g = torch.Generator().manual_seed(seed)
    offsets = [sum(vocab[:k]) for k in range(len(vocab))]
    V = sum(vocab)
    feature_ids = torch.stack([torch.randint(0, v, (n_items,), generator=g) + o
                               for v, o in zip(vocab, offsets)], dim=1)          # (M, F)
    # bag layout: 1..4 distinct words per item, padded with id 0 / weight 0.0
    bag_ids = torch.zeros((n_items, 4), dtype=torch.long)
    bag_w = torch.zeros((n_items, 4))
    for i in range(n_items):
        r = int(torch.randint(1, 5, (1,), generator=g))
        bag_ids[i, :r] = torch.randperm(V, generator=g)[:r]
        bag_w[i, :r] = 1.0
    baskets = []
    for u in range(n_users):
        if u in empty_users:
            baskets.append([])
            continue
        r = int(torch.randint(2, 6, (1,), generator=g))
        baskets.append(sorted(torch.randperm(n_items, generator=g)[:r].tolist()))
    hist_ids, hist_w = history_from_baskets(baskets, feature_ids, V)
    return dict(V=V, feature_ids=feature_ids, bag_ids=bag_ids, bag_w=bag_w, baskets=baskets,
                hist_value_ids=hist_ids, hist_weights=hist_w,
                item_token_ids=feature_ids.clone(), item_token_w=torch.ones(feature_ids.shape),
                hist_sizes=torch.tensor([float(len(b)) for b in baskets]))


def history_from_baskets(baskets, feature_ids, V):
    """w_{u,f} = n_{u,f}/|H_u| over the basket's item words (fixed item layout), padded."""
    rows = []
    for b in baskets:
        counts = torch.zeros(V)
        for i in b:
            counts.index_add_(0, feature_ids[i], torch.ones(feature_ids.shape[1]))
        nz = counts.nonzero().flatten()
        rows.append((nz, counts[nz] / max(len(b), 1)))
    d_max = max(1, max(len(r[0]) for r in rows))
    ids = torch.zeros((len(baskets), d_max), dtype=torch.long)
    w = torch.zeros((len(baskets), d_max))
    for u, (nz, ww) in enumerate(rows):
        ids[u, :len(nz)] = nz
        w[u, :len(nz)] = ww
    return ids, w


def train_batch(world, seed=1, b=B, n_neg=N_NEG):
    """Users with non-empty baskets; column 0 = a basket item (dataloader contract)."""
    g = torch.Generator().manual_seed(seed)
    cands = [u for u, bk in enumerate(world['baskets']) if bk]
    u = torch.tensor([cands[int(torch.randint(0, len(cands), (1,), generator=g))] for _ in range(b)])
    pos = torch.tensor([world['baskets'][int(x)][int(torch.randint(0, len(world['baskets'][int(x)]),
                                                                    (1,), generator=g))] for x in u])
    neg = torch.randint(0, world['feature_ids'].shape[0], (b, n_neg), generator=g)
    i = torch.cat([pos.unsqueeze(1), neg], dim=1)
    labels = torch.zeros(b, 1 + n_neg)
    labels[:, 0] = 1.0
    return u, i, labels


def proto_block(k):
    return dict(sim_proto_weight=0.7, sim_batch_weight=0.3, use_weight_matrix=False,
                n_prototypes=k, cosine_type='shifted', reg_proto_type='max', reg_batch_type='max')


def item_payload(world, use_id_feature=True, layout='fixed'):
    p = {'use_id_feature': use_id_feature, 'n_features': world['V']}
    if layout == 'bags':
        p.update(feature_ids=world['bag_ids'], feature_weights=world['bag_w'])
    else:
        p.update(feature_ids=world['feature_ids'])
    return p


def user_payload(world, use_id_feature=True):
    return {'use_id_feature': use_id_feature, 'n_features': world['V'],
            'hist_value_ids': world['hist_value_ids'], 'hist_weights': world['hist_weights'],
            'hist_item_token_ids': world['item_token_ids'],
            'hist_item_token_w': world['item_token_w'],
            'hist_sizes': world['hist_sizes'], 'loo_pooling': True}


# --- parameter dicts of the existing factory branches (fresh dicts: the host branch mutates) ---

def param_fI(world):
    return {'ft_type': 'feature_item_proto', 'embedding_dim': D,
            'item_ft_ext_param': {'ft_type': 'feature_item_proto', **proto_block(K_T), **item_payload(world)},
            'user_ft_ext_param': {'ft_type': 'embedding'}}


def param_fU(world):
    return {'ft_type': 'feature_user_proto', 'embedding_dim': D,
            'user_ft_ext_param': {'ft_type': 'feature_user_proto', **proto_block(K_U), **user_payload(world)},
            'item_ft_ext_param': {'ft_type': 'embedding'}}


def param_UI():
    return {'ft_type': 'prototypes_double_tie', 'embedding_dim': D,
            'item_ft_ext_param': {'ft_type': 'prototypes_double_tie', **proto_block(K_T)},
            'user_ft_ext_param': {'ft_type': 'prototypes_double_tie', **proto_block(K_U)}}


def param_fUfI(world, share=True, item_id=True, user_id=True, layout='fixed'):
    """The dc09 branch with both payloads injected (what inject_feature_ids hands the factory)."""
    return {'ft_type': 'feature_user_item_proto', 'embedding_dim': D, 'share_word_table': share,
            'item_ft_ext_param': {'ft_type': 'feature_user_item_proto', **proto_block(K_T),
                                  **item_payload(world, item_id, layout)},
            'user_ft_ext_param': {'ft_type': 'feature_user_item_proto', **proto_block(K_U),
                                  **user_payload(world, user_id)}}


def zero_vocab_world(n_items=N_ITEMS, n_users=N_USERS):
    """The keystone payload (V = 0): no words on either side, so q_i = e_ID(i), q_u = e_ID(u)."""
    w = toy_world(n_items=n_items, n_users=n_users)
    return dict(w, V=0, feature_ids=torch.zeros((n_items, 0), dtype=torch.long),
                hist_value_ids=torch.zeros((n_users, 1), dtype=torch.long),
                hist_weights=torch.zeros((n_users, 1)),
                item_token_ids=torch.zeros((n_items, 0), dtype=torch.long),
                item_token_w=torch.zeros((n_items, 0)))


def build(param, loss='sampled_softmax', n_users=N_USERS, n_items=N_ITEMS):
    u_fe, i_fe = FeatureExtractorFactory.create_models(param, n_users, n_items)
    rec = RecSys(n_users, n_items, {'use_bias': 0}, u_fe, i_fe, loss)
    rec.init_parameters()
    return rec


def train_step_record(rec, u, i, labels):
    """One training-mode forward/backward: scores, loss, every parameter's grad (by name)."""
    rec.train()
    rec.zero_grad()
    logits = rec(u, i)
    loss = rec.loss_func(logits, labels)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in rec.named_parameters() if p.grad is not None}
    return {'logits': logits.detach().clone(), 'loss': loss.detach().clone(), 'grads': grads}


# --- golden recipes (run on the PRE-change code by t00, recomputed after the change by t03) ---

def recipe_proto():
    """dc01 t03 recipe: PrototypeEmbedding with its own base Embedding."""
    reproducible(42)
    m = PrototypeEmbedding(20, 8, 4, use_weight_matrix=False, cosine_type='shifted',
                           reg_proto_type='max', reg_batch_type='max')
    m.init_parameters()
    x = torch.tensor([[0, 3, 7], [19, 2, 2]])
    with torch.no_grad():
        out = m(x)
    return {'state_dict': {k: v.clone() for k, v in m.state_dict().items()}, 'out': out,
            'reg': torch.as_tensor(m.get_and_reset_loss()).clone()}


def recipe_fe(world, layout, use_id):
    reproducible(7)
    p = item_payload(world, use_id, layout)
    fe = FeatureEmbedding(N_ITEMS, p['feature_ids'], p['n_features'], D, use_id_feature=use_id,
                          feature_weights=p.get('feature_weights'))
    fe.init_parameters()
    idx = torch.tensor([[0, 5, 11], [3, 3, 7]])
    with torch.no_grad():
        out = fe(idx)
    return {'weight': fe.embedding_layer.weight.detach().clone(), 'out': out}


def recipe_hfe(world, pooling, use_id):
    reproducible(11)
    p = user_payload(world, use_id)
    fe = HistoryFeatureEmbedding(N_USERS, p['hist_value_ids'], p['hist_weights'], p['n_features'], D,
                                 use_id_feature=use_id, item_token_ids=p['hist_item_token_ids'],
                                 item_token_w=p['hist_item_token_w'], hist_sizes=p['hist_sizes'],
                                 loo_pooling=True, pooling=pooling)
    fe.init_parameters()
    u, i, _ = train_batch(world)
    fe.train()
    with torch.no_grad():
        loo = fe(u, pos_i_idxs=i[:, 0])
    fe.eval()
    with torch.no_grad():
        full = fe(torch.arange(N_USERS))
    return {'weight': fe.embedding_layer.weight.detach().clone(), 'loo': loo, 'full': full}


def recipe_model(world, which):
    reproducible(23)
    param = {'fI': lambda: param_fI(world), 'fU': lambda: param_fU(world), 'UI': param_UI}[which]()
    rec = build(param)
    sd = {k: v.detach().clone() for k, v in rec.state_dict().items()}
    u, i, labels = train_batch(world)
    rec_out = train_step_record(rec, u, i, labels)
    return {'state_dict': sd, **rec_out}


def all_recipes():
    w = toy_world()
    return {
        'proto': recipe_proto(),
        'fe_fixed_id': recipe_fe(w, 'fixed', True),
        'fe_fixed_noid': recipe_fe(w, 'fixed', False),
        'fe_bags_id': recipe_fe(w, 'bags', True),
        'hfe_padded_id': recipe_hfe(w, 'padded', True),
        'hfe_padded_noid': recipe_hfe(w, 'padded', False),
        'hfe_bag_id': recipe_hfe(w, 'bag', True),
        'model_fI': recipe_model(w, 'fI'),
        'model_fU': recipe_model(w, 'fU'),
        'model_UI': recipe_model(w, 'UI'),
    }


def flatten(d, prefix=''):
    """Nested dict of tensors -> {dotted.name: tensor}."""
    out = {}
    for k, v in d.items():
        name = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, name + '.'))
        else:
            out[name] = v
    return out
