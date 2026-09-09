# dc05 t14 — train-time leave-one-out pooling (P5 hygiene, 2026-09-08).
# Pins: (1) builder returns the LOO payload (item token table + |H_u|) consistent with the
# history tables; (2) HistoryFeatureEmbedding in train mode with pos_i_idxs equals a brute-force
# mean over the OTHER purchases (fixed AND bags layout, ids on/off); (3) eval mode and calls
# without pos_i_idxs are bit-identical to the pre-fix composition; (4) a single-purchase user
# composes to exactly zero metadata mass (ID row only / zero); (5) the full chain
# inject_feature_ids -> factory -> RecSys applies LOO in train mode and not in eval mode, for BOTH
# feature_user_proto (wrapped in PrototypeEmbedding) and lightfm_hist; (6) loo_pooling=False
# restores the pre-fix path; (7) F=0 keystone is untouched by LOO.
# Run: python Master/temp/dc_checks/dc05/t14_leave_one_out_pooling.py
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check, build_recsys

from feature_extraction.feature_ids import build_user_history_weights, inject_feature_ids
from feature_extraction.feature_extractors import HistoryFeatureEmbedding

check, FAILS = make_check()
torch.manual_seed(0)

tmp = tempfile.mkdtemp(prefix='t14_dc05_')
toy = os.path.join(tmp, 'toy')
os.makedirs(toy)
# 5 users, 5 items, 2 fields; 'col' is multi-valued for the bags layout ('|' separated).
with open(os.path.join(toy, 'user_ids.csv'), 'w') as f:
    f.write('user_id,user\n' + ''.join(f'{u},u{u}\n' for u in range(5)))
with open(os.path.join(toy, 'item_ids.csv'), 'w') as f:
    f.write('item_id,item\n' + ''.join(f'{i},a{i}\n' for i in range(5)))
with open(os.path.join(toy, 'item_features.csv'), 'w') as f:
    f.write('item_id,item,dep,col\n'
            '0,a0,Blouse,Black\n'
            '1,a1,Trouser,Black|White\n'
            '2,a2,Blouse,White\n'
            '3,a3,Trouser,Red\n'
            '4,a4,Dress,Red|Black|Green\n')
# user0: {0,1}  user1: {0,2,3} (+dup)  user2: {3} (single purchase)  user3: none  user4: {1,2,4}
PAIRS = [(0, 0), (0, 1), (1, 0), (1, 0), (1, 2), (1, 3), (2, 3), (4, 1), (4, 2), (4, 4)]
with open(os.path.join(toy, 'listening_history_train.csv'), 'w') as f:
    f.write('t_dat,customer_id,article_id,user_id,user,item_id,item\n')
    for u, i in PAIRS:
        f.write(f'2020-09-01,u{u},a{i},{u},u{u},{i},a{i}\n')
HIST = {}
for u, i in PAIRS:
    HIST.setdefault(u, set()).add(i)
N_USERS, N_ITEMS, D = 5, 5, 6

for layout in ('fixed', 'bags'):
    fields = ['dep', 'col'] if layout == 'bags' else ['dep']  # 'col' is multi-valued: bags only
    out = build_user_history_weights(toy, fields, layout=layout, return_item_tokens=True)
    # arity became 7 with dc08 (n_history_rows appended; 0 when identity rows are off)
    check(f"t14.{layout}.builder_returns_seven", len(out) == 7, f"len={len(out)}")
    ids, w, nf, tok_ids, tok_w, sizes, n_hist = out
    check(f"t14.{layout}.n_history_rows_zero_by_default", n_hist == 0, f"n_hist={n_hist}")
    ids3, w3, nf3 = build_user_history_weights(toy, fields, layout=layout)
    check(f"t14.{layout}.default_return_unchanged",
          torch.equal(ids, ids3) and torch.equal(w, w3) and nf == nf3)
    exp_sizes = torch.tensor([float(len(HIST.get(u, ()))) for u in range(N_USERS)])
    check(f"t14.{layout}.hist_sizes", torch.equal(sizes, exp_sizes), f"{sizes.tolist()}")
    check(f"t14.{layout}.token_w_binary", set(tok_w.unique().tolist()) <= {0.0, 1.0})
    # consistency: |H|*w̄ row (raw counts) == Σ over purchased items of the item token indicators
    ok = True
    for u in range(N_USERS):
        cnt = torch.zeros(nf)
        for i in HIST.get(u, ()):
            cnt.index_add_(0, tok_ids[i][tok_w[i] > 0], torch.ones(int((tok_w[i] > 0).sum())))
        row = torch.zeros(nf)
        row.index_add_(0, ids[u][w[u] > 0], w[u][w[u] > 0] * sizes[u])
        ok &= torch.allclose(row, cnt, atol=1e-6)
    check(f"t14.{layout}.raw_counts_equal_sum_of_item_bags", ok)

    for use_id in (True, False):
        fe = HistoryFeatureEmbedding(N_USERS, ids, w, nf, D, use_id_feature=use_id,
                                     item_token_ids=tok_ids, item_token_w=tok_w, hist_sizes=sizes)
        fe.init_parameters()
        E = fe.embedding_layer.weight.detach()
        tag = f"t14.{layout}.{'ids' if use_id else 'noid'}"

        def brute(u, held_out):
            others = [i for i in HIST.get(u, ()) if i != held_out]
            q = torch.zeros(D)
            for i in others:
                q += E[tok_ids[i][tok_w[i] > 0]].sum(0)
            q = q / len(others) if others else q
            return q + (E[nf + u] if use_id else 0)

        us = torch.tensor([u for u, _ in PAIRS])
        ps = torch.tensor([i for _, i in PAIRS])
        # (2) train mode + pos -> brute-force LOO mean
        fe.train()
        with torch.no_grad():
            q_loo = fe(us, pos_i_idxs=ps)
        exp = torch.stack([brute(u, i) for u, i in PAIRS])
        check(f"{tag}.train_loo_equals_bruteforce", torch.allclose(q_loo, exp, atol=1e-5)
              and fe.last_forward_loo, f"max abs diff {(q_loo - exp).abs().max():.3e}")
        # (3) no pos / eval mode -> pre-fix composition (full mean)
        with torch.no_grad():
            q_full_train = fe(us)
        fe.eval()
        with torch.no_grad():
            q_eval_pos = fe(us, pos_i_idxs=ps)
            q_eval = fe(us)
        full_exp = torch.stack([brute(u, -1) for u, _ in PAIRS])
        check(f"{tag}.no_pos_is_full_mean", torch.allclose(q_full_train, full_exp, atol=1e-5)
              and not fe.last_forward_loo)
        check(f"{tag}.eval_ignores_pos", torch.equal(q_eval_pos, q_eval)
              and torch.allclose(q_eval, full_exp, atol=1e-5))
        # LOO actually changes the vector where the user has >1 purchase
        check(f"{tag}.loo_differs_from_full", (q_loo - q_full_train).abs().max() > 1e-3)
        # (4) single-purchase user 2 holding out item 3 -> zero metadata mass
        fe.train()
        with torch.no_grad():
            q2 = fe(torch.tensor([2]), pos_i_idxs=torch.tensor([3]))[0]
        exp2 = E[nf + 2] if use_id else torch.zeros(D)
        check(f"{tag}.single_purchase_zero_meta", torch.allclose(q2, exp2, atol=1e-6),
              f"max abs diff {(q2 - exp2).abs().max():.3e}")
        # (6) loo_pooling=False -> supports_loo False, pos ignored even in train mode
        fe_off = HistoryFeatureEmbedding(N_USERS, ids, w, nf, D, use_id_feature=use_id,
                                         item_token_ids=tok_ids, item_token_w=tok_w,
                                         hist_sizes=sizes, loo_pooling=False)
        fe_off.embedding_layer.weight.data.copy_(E)
        fe_off.train()
        with torch.no_grad():
            q_off = fe_off(us, pos_i_idxs=ps)
        check(f"{tag}.loo_off_restores_prefix", not fe_off.supports_loo
              and torch.allclose(q_off, full_exp, atol=1e-5) and not fe_off.last_forward_loo)
        # payload-less construction (older callers) -> supports_loo False
        fe_bare = HistoryFeatureEmbedding(N_USERS, ids, w, nf, D, use_id_feature=use_id)
        check(f"{tag}.bare_ctor_no_loo", not fe_bare.supports_loo)

# (5) full chain: inject -> factory -> RecSys, train vs eval, both ft_types
def hist_ext(rec):
    fe = rec.user_feature_extractor
    return getattr(fe, 'embedding_ext', fe)

for ft_type, extra in (('feature_user_proto', dict(n_prototypes=3, sim_proto_weight=1.0,
                                                    sim_batch_weight=1.0, use_weight_matrix=False,
                                                    cosine_type='shifted', reg_proto_type='max',
                                                    reg_batch_type='max')),
                       ('lightfm_hist', {})):
    for use_id in (True, False):
        param = {'ft_type': ft_type, 'embedding_dim': D,
                 'user_ft_ext_param': {'ft_type': ft_type, 'use_id_feature': use_id,
                                       'feature_fields': ['dep', 'col'], 'feature_layout': 'bags',
                                       **extra},
                 'item_ft_ext_param': {'ft_type': 'embedding'}}
        inj = inject_feature_ids(param, toy)
        tag = f"t14.chain.{ft_type}.{'ids' if use_id else 'noid'}"
        check(f"{tag}.payload_injected",
              all(k in inj['user_ft_ext_param'] for k in ('hist_item_token_ids', 'hist_item_token_w', 'hist_sizes'))
              and 'hist_sizes' not in param['user_ft_ext_param'])
        rec = build_recsys(inj, N_USERS, N_ITEMS)
        check(f"{tag}.supports_loo", bool(rec.user_feature_extractor.supports_loo))
        us = torch.tensor([u for u, _ in PAIRS])
        i_idxs = torch.stack([torch.tensor([i, (i + 1) % N_ITEMS, (i + 2) % N_ITEMS]) for _, i in PAIRS])
        rec.train()
        with torch.no_grad():
            out_train = rec(us, i_idxs)
        loo_train = hist_ext(rec).last_forward_loo
        rec.user_feature_extractor.get_and_reset_loss(); rec.item_feature_extractor.get_and_reset_loss()
        rec.eval()
        with torch.no_grad():
            out_eval = rec(us, i_idxs)
        loo_eval = hist_ext(rec).last_forward_loo
        check(f"{tag}.train_applies_loo_eval_does_not", loo_train and not loo_eval)
        check(f"{tag}.train_vs_eval_logits_differ", (out_train - out_eval).abs().max() > 1e-4,
              f"max abs diff {(out_train - out_eval).abs().max():.3e}")
        # loo_pooling=False knob through the config path
        param_off = {**param, 'user_ft_ext_param': {**param['user_ft_ext_param'], 'loo_pooling': False}}
        rec_off = build_recsys(inject_feature_ids(param_off, toy), N_USERS, N_ITEMS)
        rec_off.train()
        with torch.no_grad():
            rec_off(us, i_idxs)
        check(f"{tag}.config_knob_off", not rec_off.user_feature_extractor.supports_loo
              and not hist_ext(rec_off).last_forward_loo)
        # gradient still flows through the LOO branch to the word table
        rec.train()
        out = rec(us, i_idxs)
        labels = torch.zeros_like(out); labels[:, 0] = 1
        rec.loss_func(out, labels).backward()
        g = hist_ext(rec).embedding_layer.weight.grad
        check(f"{tag}.grad_reaches_word_table", g is not None and g[:inj['user_ft_ext_param']['n_features']].abs().sum() > 0)

# (7) F=0 keystone untouched: no fields -> LOO subtracts nothing, composition == ID row
param0 = {'ft_type': 'lightfm_hist', 'embedding_dim': D,
          'user_ft_ext_param': {'ft_type': 'lightfm_hist', 'use_id_feature': True,
                                'feature_fields': [], 'feature_layout': 'bags'},
          'item_ft_ext_param': {'ft_type': 'embedding'}}
rec0 = build_recsys(inject_feature_ids(param0, toy), N_USERS, N_ITEMS)
E0 = rec0.user_feature_extractor.embedding_layer.weight.detach()
us = torch.tensor([u for u, _ in PAIRS]); ps = torch.tensor([i for _, i in PAIRS])
rec0.train()
with torch.no_grad():
    q0 = rec0.user_feature_extractor(us, pos_i_idxs=ps)
check("t14.keystone_F0_loo_is_identity", torch.equal(q0, E0[us]))

print("t14: ALL PASS" if not FAILS else f"t14: {len(FAILS)} FAIL -> {FAILS}")
sys.exit(1 if FAILS else 0)
