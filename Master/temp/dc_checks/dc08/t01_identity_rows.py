# dc08 t01 — item-identity history rows in the fU composition (SVD++/FISM y_j).
# Pins, on a toy split, for BOTH layouts (fixed / bags) and BOTH id arms:
#   1. builder: identity slots land in the bag at (n_features + j, lambda_y/|H_u|), and the
#      leave-one-out payload gains the matching (n_features + j, lambda_y) slot;
#   2. KEYSTONE lambda_y = 0 reduces the composition to dc05 (4b C2) — bit-identically in the
#      'fixed' layout, to float round-off in 'bags' (F-DC08-05: the wider padded bag changes
#      torch's reduction blocking, not the composition);
#   3. KEYSTONE history_id_rows=False is byte-identical to today's builder output;
#   4. forward == the mathematical form q_u = mean_j(sum_f e_f(j) + lambda_y*y_j) + e_ID(u) (4b C1);
#   5. the EXISTING algebraic leave-one-out removes the positive's identity row exactly, i.e.
#      equals a brute-force mean over H_u \ {i} (4b C3) — the finding that superseded the seed
#      entry's "mask-then-pool" requirement;
#   6. the user-ID row moves to n_features + n_history_rows (id_row_offset) and the F=0 keystone
#      (fU(F=0,+ID) == a plain per-user Embedding) still holds with identity rows present;
#   7. gradient reaches every purchased item's y_j and is <= round-off on a LOO-removed positive
#      (F-DC08-01: "algebraically zero", NOT ==0 — fp32 leaves ~1e-8 at odd |H_u|);
#   8. freeze_history_rows (5b A2 capacity control) leaves the identity block at its init value
#      after an optimizer step while words/user rows still move;
#   9. full chain inject_feature_ids -> factory -> RecSys for feature_user_proto and lightfm_hist.
# Run: conda run -n protomf python Master/temp/dc_checks/dc08/t01_identity_rows.py
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dc05')))
import torch

from _harness import make_check, build_recsys, feature_user_proto_param

from feature_extraction.feature_ids import build_user_history_weights, inject_feature_ids
from feature_extraction.feature_extractors import HistoryFeatureEmbedding

check, FAILS = make_check()
torch.manual_seed(0)

tmp = tempfile.mkdtemp(prefix='t01_dc08_')
toy = os.path.join(tmp, 'toy')
os.makedirs(toy)
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
# user0: {0,1} (|H|=2, even)  user1: {0,2,3} (|H|=3, odd)  user2: {3} (single)  user3: none
# user4: {1,2,4}
PAIRS = [(0, 0), (0, 1), (1, 0), (1, 0), (1, 2), (1, 3), (2, 3), (4, 1), (4, 2), (4, 4)]
with open(os.path.join(toy, 'listening_history_train.csv'), 'w') as f:
    f.write('t_dat,customer_id,article_id,user_id,user,item_id,item\n')
    for u, i in PAIRS:
        f.write(f'2020-09-01,u{u},a{i},{u},u{u},{i},a{i}\n')
HIST = {}
for u, i in PAIRS:
    HIST.setdefault(u, set()).add(i)
N_USERS, N_ITEMS, D = 5, 5, 6
LAM = 0.75  # a non-1 lambda_y, so a wrong weight cannot pass by coincidence


def item_token_rows(tok_ids, tok_w, emb, item):
    return (emb(tok_ids[item]) * tok_w[item].unsqueeze(-1)).sum(dim=0)


for layout in ('fixed', 'bags'):
    fields = ['dep', 'col'] if layout == 'bags' else ['dep']
    L = layout

    # --- builder: off is byte-identical to dc05 (keystone 3) ---
    base = build_user_history_weights(toy, fields, layout=layout, return_item_tokens=True)
    off = build_user_history_weights(toy, fields, layout=layout, return_item_tokens=True,
                                     history_id_rows=False, history_row_weight=LAM)
    check(f"t01.{L}.off_is_dc05_byte_identical",
          all(torch.equal(a, b) if torch.is_tensor(a) else a == b for a, b in zip(base, off)))

    ids, w, nf, tok_ids, tok_w, sizes, n_hist = build_user_history_weights(
        toy, fields, layout=layout, return_item_tokens=True,
        history_id_rows=True, history_row_weight=LAM)
    check(f"t01.{L}.n_history_rows_is_n_items", n_hist == N_ITEMS, f"{n_hist}")

    # identity slots present at the right id/weight, one per purchase
    ok_slots = True
    for u in range(N_USERS):
        want = {nf + j: LAM / len(HIST[u]) for j in HIST.get(u, ())}
        got = {int(c): float(x) for c, x in zip(ids[u], w[u])
               if float(x) != 0.0 and int(c) >= nf}
        if want.keys() != got.keys() or any(abs(want[k] - got[k]) > 1e-6 for k in want):
            ok_slots = False
    check(f"t01.{L}.identity_slots_are_lambda_over_H", ok_slots)

    # LOO payload gained the item's own identity slot at weight lambda_y
    own_ok = all(any(int(c) == nf + j and abs(float(x) - LAM) < 1e-6
                     for c, x in zip(tok_ids[j], tok_w[j])) for j in range(N_ITEMS))
    check(f"t01.{L}.loo_payload_has_own_identity_slot", own_ok)

    for use_id in (True, False):
        tag = f"t01.{L}.ids{int(use_id)}"
        emb = HistoryFeatureEmbedding(N_USERS, ids, w, nf, D, use_id_feature=use_id,
                                      item_token_ids=tok_ids, item_token_w=tok_w,
                                      hist_sizes=sizes, loo_pooling=True,
                                      n_history_rows=n_hist, history_row_weight=LAM)
        emb.init_parameters()
        E = emb.embedding_layer

        # --- id_row_offset (keystone 6) ---
        check(f"{tag}.id_row_offset", emb.id_row_offset == nf + N_ITEMS)
        check(f"{tag}.table_rows",
              E.weight.shape[0] == nf + N_ITEMS + (N_USERS if use_id else 0))

        # --- 4b C1: forward == the mathematical form ---
        emb.eval()
        with torch.no_grad():
            q = emb(torch.arange(N_USERS))
        want = torch.zeros(N_USERS, D)
        for u in range(N_USERS):
            H = sorted(HIST.get(u, ()))
            if H:
                acc = torch.zeros(D)
                for j in H:
                    acc = acc + item_token_rows(tok_ids, tok_w, E, j)  # words + lambda*y_j
                want[u] = acc / len(H)
            if use_id:
                want[u] = want[u] + E.weight[emb.id_row_offset + u]
        err = float((q - want).abs().max())
        check(f"{tag}.C1_bag_equals_math_form", err < 1e-6, f"max|err|={err:.2e}")

        # --- 4b C3: algebraic LOO removes the positive's words AND its identity row ---
        emb.train()
        us, pos = [], []
        for u in range(N_USERS):
            for j in sorted(HIST.get(u, ())):
                us.append(u); pos.append(j)
        us_t, pos_t = torch.tensor(us), torch.tensor(pos)
        with torch.no_grad():
            q_loo = emb(us_t, pos_i_idxs=pos_t)
        want_loo = torch.zeros(len(us), D)
        for r, (u, i) in enumerate(zip(us, pos)):
            rest = sorted(HIST[u] - {i})
            if rest:
                acc = torch.zeros(D)
                for j in rest:
                    acc = acc + item_token_rows(tok_ids, tok_w, E, j)
                want_loo[r] = acc / len(rest)
            if use_id:
                want_loo[r] = want_loo[r] + E.weight[emb.id_row_offset + u]
        err = float((q_loo - want_loo).abs().max())
        check(f"{tag}.C3_loo_equals_bruteforce_minus_i", err < 1e-6, f"max|err|={err:.2e}")

        # --- 4b C2 keystone: lambda_y = 0 is bit-identical to dc05 ---
        ids0, w0, nf0, tok0, tokw0, sz0, nh0 = build_user_history_weights(
            toy, fields, layout=layout, return_item_tokens=True,
            history_id_rows=True, history_row_weight=0.0)
        emb0 = HistoryFeatureEmbedding(N_USERS, ids0, w0, nf0, D, use_id_feature=use_id,
                                       item_token_ids=tok0, item_token_w=tokw0,
                                       hist_sizes=sz0, loo_pooling=True,
                                       n_history_rows=nh0, history_row_weight=0.0)
        b_ids, b_w, b_nf, b_tok, b_tokw, b_sz, _ = base
        emb_dc05 = HistoryFeatureEmbedding(N_USERS, b_ids, b_w, b_nf, D, use_id_feature=use_id,
                                           item_token_ids=b_tok, item_token_w=b_tokw,
                                           hist_sizes=b_sz, loo_pooling=True)
        with torch.no_grad():   # same words, same user rows, identity block irrelevant at lam=0
            emb0.embedding_layer.weight[:nf0] = emb_dc05.embedding_layer.weight[:b_nf]
            if use_id:
                emb0.embedding_layer.weight[emb0.id_row_offset:] = \
                    emb_dc05.embedding_layer.weight[emb_dc05.id_row_offset:]
        emb0.eval(); emb_dc05.eval()
        with torch.no_grad():
            d0 = (emb0(torch.arange(N_USERS)) - emb_dc05(torch.arange(N_USERS))).abs().max()
        # F-DC08-05 (found by this check, 2026-09-09): bit-identity at lambda_y = 0 holds in the
        # FIXED layout but not in BAGS (~6e-8). The identity slots contribute exact +0.0, but
        # they widen the padded bag, and torch's blocked/pairwise `sum(dim=-2)` associates the
        # SAME nonzero terms differently at a different width — so the difference is reduction
        # order, not a changed composition. The keystone that actually protects existing configs
        # is `off_is_dc05_byte_identical` above (history_id_rows=False, unchanged tensors).
        check(f"{tag}.C2_lambda0_matches_dc05", float(d0) < 1e-6, f"max|diff|={float(d0):.2e}")
        check(f"{tag}.C2_lambda0_bit_identical_fixed_layout",
              float(d0) == 0.0 or layout == 'bags',
              f"max|diff|={float(d0)} (bit-identity expected in 'fixed' only)")

        # --- 4b C8 / F-DC08-01: gradient reaches purchased y_j; LOO-removed positive <= round-off
        emb.train()
        u, i = 1, 0                      # |H_1| = 3 (odd — the fp32 residual case)
        emb.zero_grad()
        emb(torch.tensor([u]), pos_i_idxs=torch.tensor([i])).sum().backward()
        g = E.weight.grad[nf:nf + N_ITEMS].abs().sum(dim=1)
        others = sorted(HIST[u] - {i})
        check(f"{tag}.C8_grad_reaches_remaining_history",
              all(float(g[j]) > 1e-8 for j in others), f"g={[round(float(g[j]),5) for j in others]}")
        check(f"{tag}.C8_loo_positive_grad_is_roundoff",
              float(g[i]) < 1e-5, f"|grad(y_pos)|={float(g[i]):.2e} (algebraically 0)")
        check(f"{tag}.C8_unpurchased_grad_exactly_zero",
              all(float(g[j]) == 0.0 for j in range(N_ITEMS) if j not in HIST[u]))

    # --- 5b A2: freeze_history_rows keeps the identity block at init ---
    frz = HistoryFeatureEmbedding(N_USERS, ids, w, nf, D, use_id_feature=False,
                                  item_token_ids=tok_ids, item_token_w=tok_w,
                                  hist_sizes=sizes, loo_pooling=True,
                                  n_history_rows=n_hist, history_row_weight=LAM,
                                  freeze_history_rows=True)
    frz.init_parameters()
    y0 = frz.embedding_layer.weight[nf:nf + N_ITEMS].detach().clone()
    e0 = frz.embedding_layer.weight[:nf].detach().clone()
    opt = torch.optim.SGD(frz.parameters(), lr=1.0)
    frz.train()
    opt.zero_grad()
    frz(torch.tensor([0, 1, 4])).pow(2).sum().backward()
    opt.step()
    y1 = frz.embedding_layer.weight[nf:nf + N_ITEMS].detach()
    e1 = frz.embedding_layer.weight[:nf].detach()
    check(f"t01.{L}.A2_frozen_identity_block_unchanged",
          torch.equal(y0, y1), f"max|dy|={float((y1 - y0).abs().max())}")
    check(f"t01.{L}.A2_words_still_train", float((e1 - e0).abs().max()) > 0)

# --- full chain: inject_feature_ids -> factory -> RecSys, both ft_types ---
for ft in ('feature_user_proto', 'lightfm_hist'):
    param = feature_user_proto_param(D, 4, None, None, None, use_id_feature=False,
                                     feature_fields=['dep'])
    param['ft_type'] = ft
    param['user_ft_ext_param']['ft_type'] = ft
    for k in ('hist_value_ids', 'hist_weights', 'n_features'):
        param['user_ft_ext_param'].pop(k)
    param['user_ft_ext_param']['feature_layout'] = 'fixed'
    param['user_ft_ext_param']['history_id_rows'] = True
    param['user_ft_ext_param']['history_row_weight'] = LAM
    injected = inject_feature_ids(param, toy)
    check(f"t01.chain.{ft}.injector_sets_n_history_rows",
          injected['user_ft_ext_param']['n_history_rows'] == N_ITEMS)
    model = build_recsys(injected, N_USERS, N_ITEMS)
    hist = (model.user_feature_extractor.embedding_ext if ft == 'feature_user_proto'
            else model.user_feature_extractor)
    check(f"t01.chain.{ft}.extractor_has_identity_block",
          hist.n_history_rows == N_ITEMS and hist.id_row_offset == hist.n_features + N_ITEMS)
    check(f"t01.chain.{ft}.lambda_recorded", abs(hist.history_row_weight - LAM) < 1e-12)
    model.train()
    out = model(torch.tensor([0, 1]), torch.tensor([[0, 2], [2, 3]]))
    check(f"t01.chain.{ft}.forward_shape", tuple(out.shape) == (2, 2), f"{tuple(out.shape)}")
    check(f"t01.chain.{ft}.loo_applied_in_train", hist.last_forward_loo is True)

print("\nt01: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: {FAILS}"))
sys.exit(1 if FAILS else 0)
