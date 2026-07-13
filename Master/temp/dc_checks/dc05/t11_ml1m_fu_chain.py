# dc05 t11 — fU-on-ml-1m end-to-end (SC.3, dataset-scope charter amendment 2026-07-12).
# Pins the REAL second-testbed chain the S0-build extension enabled: canonical resolution
# ('ml-1m' -> genres+tags@0.8, layout='bags') -> build_user_history_weights on the real split
# -> injection seam (payload lands on the USER side, caller untouched) -> factory -> forward
# -> one training step with gradients reaching every parameter surface.
# Independent recompute: per-user vote mass Σ_f w̄_{u,f} must equal the mean token count over
# the user's dedup basket (raw bags, S0.5-addendum decision A), recomputed from the raw CSVs
# by a separate pandas path; 3 sampled users' q_u recomposed by hand from the raw CSVs vs the
# module forward.
# Skips (with a loud line) if data/ml-1m is not staged locally (gitignored artifacts).
# Run: python Master/temp/dc_checks/dc05/t11_ml1m_fu_chain.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import torch

from _harness import make_check, build_recsys, feature_user_proto_param

from feature_extraction.feature_ids import (resolve_canonical_fields,
                                            build_user_history_weights,
                                            inject_feature_ids)

check, FAILS = make_check()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
ML = os.path.join(REPO_ROOT, 'data', 'ml-1m')

if not os.path.exists(os.path.join(ML, 'item_features.csv')):
    print("[SKIP] data/ml-1m not staged locally (item_features.csv missing) — t11 needs the "
          "real split; nothing checked")
    sys.exit(0)

# --- 1. canonical resolution (charter C5 as amended 2026-07-12) ---
fields, layout = resolve_canonical_fields('ml-1m')
check("t11.canonical_fields", fields == ['genres', 'tags'] and layout == 'bags',
      f"{fields} / {layout}")
fields_cold, layout_cold = resolve_canonical_fields('ml-1m_cold')
check("t11.cold_variant_resolves_same", fields_cold == fields and layout_cold == layout)

# --- 2. real builder run ---
ids, w, nf = build_user_history_weights(ML, fields, layout=layout)
n_users_csv = len(pd.read_csv(os.path.join(ML, 'user_ids.csv')))
n_items_csv = len(pd.read_csv(os.path.join(ML, 'item_ids.csv')))
check("t11.row_per_user", ids.shape[0] == n_users_csv == w.shape[0],
      f"{ids.shape[0]} vs {n_users_csv}")
# vocab: 18 genres + 1,043 tags@0.8 (t10 pins the item builder; this pins the fU consumption)
check("t11.n_features_1061", nf == 1061, f"nf={nf}")
check("t11.padding_discipline", bool(((w > 0) | (ids == 0)).all()),
      "padding slots must hold id 0")

# --- 3. independent vote-mass recompute (raw bags: Σ_f w̄ == mean tokens/purchase) ---
feats = pd.read_csv(os.path.join(ML, 'item_features.csv'), keep_default_na=False, dtype=str)
feats['item_id'] = feats['item_id'].astype(int)
feats = feats.sort_values('item_id').reset_index(drop=True)


def n_tokens(cell):
    return 0 if cell == '' else len(cell.split('|'))


tok_count = (feats['genres'].map(n_tokens) + feats['tags'].map(n_tokens)).to_numpy()
pairs = pd.read_csv(os.path.join(ML, 'listening_history_train.csv'),
                    usecols=['user_id', 'item_id'])
pairs = pairs.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
expected_mass = pd.Series(tok_count[pairs['item_id'].to_numpy()],
                          index=pairs.index).groupby(pairs['user_id']).mean()
got_mass = w.sum(dim=1)
mass_vec = torch.zeros(n_users_csv)
mass_vec[torch.as_tensor(expected_mass.index.to_numpy(), dtype=torch.long)] = \
    torch.as_tensor(expected_mass.to_numpy(), dtype=torch.float32)
check("t11.vote_mass_equals_mean_tokens_per_purchase",
      bool(torch.allclose(got_mass, mass_vec, atol=1e-4)),
      f"maxΔ={float((got_mass - mass_vec).abs().max()):.2e}")
check("t11.every_user_has_history", bool((got_mass > 0).all()),
      "canonical ml-1m: no user without train rows")

# --- 4. injection seam (payload on the USER side; caller dict untouched) ---
param = {
    'ft_type': 'feature_user_proto', 'embedding_dim': 8,
    'user_ft_ext_param': {
        'ft_type': 'feature_user_proto', 'sim_proto_weight': 1.0, 'sim_batch_weight': 1.0,
        'use_weight_matrix': False, 'n_prototypes': 4, 'cosine_type': 'shifted',
        'reg_proto_type': 'max', 'reg_batch_type': 'max', 'use_id_feature': True,
        'feature_fields': fields, 'feature_layout': layout,
    },
    'item_ft_ext_param': {'ft_type': 'embedding'},
}
injected = inject_feature_ids(param, ML)
check("t11.inject_payload_present",
      'hist_value_ids' in injected['user_ft_ext_param']
      and 'hist_weights' in injected['user_ft_ext_param']
      and injected['user_ft_ext_param']['n_features'] == 1061, "")
check("t11.inject_caller_untouched", 'hist_value_ids' not in param['user_ft_ext_param'],
      "tensor must never enter the Ray/JSON config dict")
check("t11.inject_tensors_match_builder",
      torch.equal(injected['user_ft_ext_param']['hist_value_ids'], ids)
      and torch.equal(injected['user_ft_ext_param']['hist_weights'], w), "")

# --- 5. factory + forward + hand-recomposed q_u for 3 sampled users ---
K_u, d = 4, 8
model = build_recsys(feature_user_proto_param(d, K_u, ids, w, nf, use_id_feature=True,
                                              feature_fields=fields),
                     n_users_csv, n_items_csv)
hfe = model.user_feature_extractor.embedding_ext
E = hfe.embedding_layer.weight.detach()

g = torch.Generator().manual_seed(7)
sample_users = torch.randint(0, n_users_csv, (3,), generator=g)
for u in sample_users.tolist():
    basket = pairs[pairs['user_id'] == u]['item_id'].to_numpy()
    # hand recompose: for every purchased movie, every token weighs 1; w̄ = n/|H|; + ID row
    from collections import Counter
    cnt = Counter()
    genre_vocab = sorted({t for cell in feats['genres'] if cell != ''
                          for t in cell.split('|')})
    tag_vocab = sorted({t for cell in feats['tags'] if cell != '' for t in cell.split('|')})
    g_code = {v: i for i, v in enumerate(genre_vocab)}
    t_code = {v: i + len(genre_vocab) for i, v in enumerate(tag_vocab)}
    for it in basket:
        gc, tc = feats.at[it, 'genres'], feats.at[it, 'tags']
        for t in (gc.split('|') if gc != '' else []):
            cnt[g_code[t]] += 1
        for t in (tc.split('|') if tc != '' else []):
            cnt[t_code[t]] += 1
    q_hand = torch.zeros(d)
    for tok, n in cnt.items():
        q_hand += (n / len(basket)) * E[tok]
    q_hand += E[nf + u]
    q_mod = hfe(torch.tensor([u]))[0].detach()
    check(f"t11.hand_recomposed_q_u_user{u}",
          bool(torch.allclose(q_hand, q_mod, atol=1e-4)),
          f"maxΔ={float((q_hand - q_mod).abs().max()):.2e}")

# --- 6. one training step: loss finite, grads reach all four parameter surfaces ---
gb = torch.Generator().manual_seed(11)
u_idx = torch.randint(0, n_users_csv, (16,), generator=gb)
i_idx = torch.randint(0, n_items_csv, (16, 3), generator=gb)
labels = torch.zeros(16, 3)
labels[:, 0] = 1.0
out = model(u_idx, i_idx)
check("t11.forward_finite", bool(torch.isfinite(out).all()), f"shape {tuple(out.shape)}")
loss = model.loss_func(out, labels)
loss.backward()
check("t11.loss_finite", bool(torch.isfinite(loss)), f"{float(loss):.4f}")
word_grad = hfe.embedding_layer.weight.grad
proto_grad = model.user_feature_extractor.prototypes.grad
item_grad = model.item_feature_extractor.embedding_layer.weight.grad
check("t11.grads_reach_word_table", word_grad is not None
      and bool((word_grad[:nf].abs().sum() > 0)), "")
check("t11.grads_reach_id_rows", word_grad is not None
      and bool((word_grad[nf:].abs().sum() > 0)), "")
check("t11.grads_reach_prototypes", proto_grad is not None
      and bool(proto_grad.abs().sum() > 0), "")
check("t11.grads_reach_item_table", item_grad is not None
      and bool(item_grad.abs().sum() > 0), "")
check("t11.buffers_no_grad", hfe.hist_weights.grad is None and hfe.hist_value_ids.grad is None,
      "")

print(f"\nt11: {'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
sys.exit(0 if not FAILS else 1)
