# dc09 t03 — the single-table path is untouched by the SC.3 build (REGRESSION-CRITICAL).
#
# (a) Every surface the build touched — PrototypeEmbedding, FeatureEmbedding (fixed/bags,
#     ±ID), HistoryFeatureEmbedding (padded/bag, ±ID, leave-one-out) and the fI / fU / UI
#     factory branches through one training step (state_dict, logits, loss, every grad) — is
#     BITWISE equal to golden_pre_sc3.pt, captured on the unmodified code (t00, 27dc914).
# (b) A borrowed table holding the same weights gives the same outputs as an owned one.
# Run: python Master/temp/dc_checks/dc09/t03_single_table_path.py
import sys

import torch
from torch import nn

from _fixture import (D, GOLDEN, N_ITEMS, N_USERS, all_recipes, flatten, item_payload, make_check,
                      toy_world, train_batch, user_payload)
from feature_extraction.feature_extractors import FeatureEmbedding, HistoryFeatureEmbedding

check, FAILS = make_check()

golden = torch.load(GOLDEN, weights_only=False)
meta = golden.pop('meta')
if meta['torch'] != torch.__version__:
    print(f"FAIL: golden captured under torch {meta['torch']}, running {torch.__version__} — "
          f"bitwise comparison is only meaningful in the capture environment.")
    sys.exit(1)

# --- (a) bitwise against the pre-change golden ---
g, now = flatten(golden), flatten(all_recipes())
check('t03a.same_tensor_set', g.keys() == now.keys(),
      f'golden-only {sorted(g.keys() - now.keys())[:3]}, new-only {sorted(now.keys() - g.keys())[:3]}')
differ = [k for k in g if k in now and not torch.equal(g[k], now[k])]
check('t03a.all_bitwise_equal', not differ, f'{len(g)} tensors, differing: {differ[:5]}')

# --- (b) borrowed table == owned table, same weights ---
w = toy_world()
idx = torch.tensor([[0, 5, 11], [3, 3, 7]])
for layout in ('fixed', 'bags'):
    p = item_payload(w, True, layout)
    own = FeatureEmbedding(N_ITEMS, p['feature_ids'], p['n_features'], D,
                           feature_weights=p.get('feature_weights'))
    own.init_parameters()
    table = nn.Embedding(own.embedding_layer.num_embeddings + 4, D)  # + rows it never indexes
    with torch.no_grad():
        table.weight[:own.embedding_layer.num_embeddings] = own.embedding_layer.weight
    before = table.weight.detach().clone()
    bor = FeatureEmbedding(N_ITEMS, p['feature_ids'], p['n_features'], D,
                           feature_weights=p.get('feature_weights'), embedding_layer=table)
    bor.init_parameters()  # must NOT touch a borrowed table
    check(f't03b.fe_{layout}_borrowed_not_reinitialised', torch.equal(table.weight, before))
    with torch.no_grad():
        check(f't03b.fe_{layout}_borrowed_equals_owned', torch.equal(own(idx), bor(idx)))

p = user_payload(w, True)
kw = dict(item_token_ids=p['hist_item_token_ids'], item_token_w=p['hist_item_token_w'],
          hist_sizes=p['hist_sizes'], loo_pooling=True)
own = HistoryFeatureEmbedding(N_USERS, p['hist_value_ids'], p['hist_weights'], p['n_features'], D, **kw)
own.init_parameters()
M_PAD = 5  # a shared table puts another block between the words and this side's IDs
V = p['n_features']
table = nn.Embedding(V + M_PAD + N_USERS, D)
with torch.no_grad():
    table.weight[:V] = own.embedding_layer.weight[:V]
    table.weight[V + M_PAD:] = own.embedding_layer.weight[V:]
bor = HistoryFeatureEmbedding(N_USERS, p['hist_value_ids'], p['hist_weights'], V, D,
                              embedding_layer=table, id_block_offset=M_PAD, **kw)
check('t03b.hfe_id_row_offset', bor.id_row_offset == V + M_PAD, f'{bor.id_row_offset}')
u, i, _ = train_batch(w)
for mode in ('train', 'eval'):
    own.train(mode == 'train'); bor.train(mode == 'train')
    with torch.no_grad():
        a = own(u, pos_i_idxs=i[:, 0]) if mode == 'train' else own(torch.arange(N_USERS))
        b = bor(u, pos_i_idxs=i[:, 0]) if mode == 'train' else bor(torch.arange(N_USERS))
    check(f't03b.hfe_borrowed_equals_owned_{mode}', torch.equal(a, b), f'max {(a - b).abs().max():.1e}')

print("\nt03: ALL PASS" if not FAILS else f"\nt03: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
