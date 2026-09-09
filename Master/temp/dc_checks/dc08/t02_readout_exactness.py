# dc08 t02 — the read-outs stay EXACT once the composition carries item-identity rows.
# Pins (fU host, both id arms, both layouts):
#   1. user_history_rows: the weighted slot rows still sum to q_u exactly (identity rows are
#      just more rows) and the user-ID row is read from id_row_offset, not n_features;
#   2. per_purchase_rows: each purchase row now carries its own lambda_y*y_i, so the
#      per-purchase regrouping still sums to q_u — i.e. the word-level and purchase-level
#      readings remain two exact regroupings of ONE sum (4b C5) rather than two different sums;
#   3. per_feature_shares over either grouping sums to cos(q_u, p_l) for every prototype (4b C4);
#   4. the three-way coverage split (named / item-identity / user-ID) partitions the personalized
#      mass exactly, and the identity block is disjoint from the nameable code range.
# Run: conda run -n protomf python Master/temp/dc_checks/dc08/t02_readout_exactness.py
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dc05')))
import torch

from _harness import make_check, build_recsys, feature_user_proto_param

from feature_extraction.feature_ids import build_feature_bags, build_feature_ids, inject_feature_ids
from utilities.explanations.feature_readout import per_feature_shares
from utilities.explanations.history_readout import (per_purchase_rows, user_history_rows,
                                                    user_train_items)

check, FAILS = make_check()
torch.manual_seed(3)

tmp = tempfile.mkdtemp(prefix='t02_dc08_')
toy = os.path.join(tmp, 'toy')
os.makedirs(toy)
with open(os.path.join(toy, 'user_ids.csv'), 'w') as f:
    f.write('user_id,user\n' + ''.join(f'{u},u{u}\n' for u in range(4)))
with open(os.path.join(toy, 'item_ids.csv'), 'w') as f:
    f.write('item_id,item\n' + ''.join(f'{i},a{i}\n' for i in range(5)))
with open(os.path.join(toy, 'item_features.csv'), 'w') as f:
    f.write('item_id,item,dep,col\n'
            '0,a0,Blouse,Black\n1,a1,Trouser,Black|White\n2,a2,Blouse,White\n'
            '3,a3,Trouser,Red\n4,a4,Dress,Red|Black|Green\n')
PAIRS = [(0, 0), (0, 1), (0, 4), (1, 2), (1, 3), (2, 1), (3, 0), (3, 2), (3, 3), (3, 4)]
with open(os.path.join(toy, 'listening_history_train.csv'), 'w') as f:
    f.write('t_dat,customer_id,article_id,user_id,user,item_id,item\n')
    for u, i in PAIRS:
        f.write(f'2020-09-01,u{u},a{i},{u},u{u},{i},a{i}\n')
N_USERS, N_ITEMS, D, KU = 4, 5, 6, 4
LAM = 0.75

for layout in ('fixed', 'bags'):
    fields = ['dep', 'col'] if layout == 'bags' else ['dep']
    for use_id in (True, False):
        tag = f"t02.{layout}.ids{int(use_id)}"
        param = feature_user_proto_param(D, KU, None, None, None, use_id_feature=use_id,
                                         feature_fields=fields)
        for k in ('hist_value_ids', 'hist_weights', 'n_features'):
            param['user_ft_ext_param'].pop(k)
        param['user_ft_ext_param']['feature_layout'] = layout
        param['user_ft_ext_param']['history_id_rows'] = True
        param['user_ft_ext_param']['history_row_weight'] = LAM
        model = build_recsys(inject_feature_ids(param, toy), N_USERS, N_ITEMS)
        model.eval()
        proto_fe = model.user_feature_extractor
        hist = proto_fe.embedding_ext

        if layout == 'bags':
            fids, bag_w, _ = build_feature_bags(toy, fields)
        else:
            fids, _ = build_feature_ids(toy, fields)
            bag_w = None

        for u in range(N_USERS):
            with torch.no_grad():
                q = hist(torch.tensor([u])).squeeze(0)

            # (1) word-level rows sum to q_u
            rows_w, codes, has_id = user_history_rows(hist, u)
            e1 = float((rows_w.sum(dim=0) - q).abs().max())

            # (2) per-purchase rows sum to the SAME q_u (identity mass included)
            purchases = user_train_items(toy, u)
            rows_p, _ = per_purchase_rows(hist, u, purchases, fids, feature_weights=bag_w)
            e2 = float((rows_p.sum(dim=0) - q).abs().max())
            check(f"{tag}.u{u}.word_rows_sum_to_q", e1 < 1e-6, f"{e1:.2e}")
            check(f"{tag}.u{u}.purchase_rows_sum_to_q", e2 < 1e-6, f"{e2:.2e}")

            # (3) shares of either grouping sum to cos(q_u, p_l)
            with torch.no_grad():
                cos = torch.nn.functional.cosine_similarity(
                    q.unsqueeze(0), proto_fe.prototypes, dim=-1).squeeze(0)
            sw = per_feature_shares(rows_w, proto_fe.prototypes).sum(dim=0)
            sp = per_feature_shares(rows_p, proto_fe.prototypes).sum(dim=0)
            check(f"{tag}.u{u}.C4_word_shares_sum_to_cos",
                  float((sw - cos).abs().max()) < 1e-5, f"{float((sw - cos).abs().max()):.2e}")
            check(f"{tag}.u{u}.C5_purchase_shares_sum_to_cos",
                  float((sp - cos).abs().max()) < 1e-5, f"{float((sp - cos).abs().max()):.2e}")

            # (4) coverage split partitions the mass exactly; identity codes are outside the
            #     nameable range and every purchase contributes exactly one identity code
            is_y = [hist.is_history_row(c) for c in codes]      # one flag per SLOT code
            allsh = per_feature_shares(rows_w, proto_fe.prototypes)   # (len(codes) + has_id, K)
            slot_sh = allsh[:len(codes)]
            id_sh = allsh[-1] if has_id else torch.zeros(KU)
            y_sh = slot_sh[torch.tensor(is_y, dtype=torch.bool)].sum(dim=0)
            w_sh = slot_sh[~torch.tensor(is_y, dtype=torch.bool)].sum(dim=0)
            split = w_sh + y_sh + id_sh
            check(f"{tag}.u{u}.A1_three_way_split_is_exact",
                  float((split - cos).abs().max()) < 1e-5,
                  f"{float((split - cos).abs().max()):.2e}")
            check(f"{tag}.u{u}.identity_codes_outside_vocab",
                  sum(is_y) == len(purchases)
                  and all(c >= hist.n_features for c, b in zip(codes, is_y) if b),
                  f"{sum(is_y)} identity codes vs |H_u|={len(purchases)}")

print("\nt02: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: {FAILS}"))
sys.exit(1 if FAILS else 0)
