# dc08 t03 — the flat embedding_bag pooling path (5b A10, the ml-1m enabler).
# Pins:
#   1. 'bag' and 'padded' compose the SAME q_u (to float round-off) for every user, both id arms,
#      with and without identity rows, including empty-history and single-purchase users;
#   2. the leave-one-out path agrees between the two poolings (the algebraic subtraction runs on
#      the pooled vector, so it must not care how the pooling was done);
#   3. gradients agree between the two poolings (embedding_bag's backward reaches the same rows
#      with the same values);
#   4. 'auto' picks padded on a narrow bag and bag on a wide one, at the configured threshold;
#   5. the bag path does NOT keep the padded tensors (memory is the whole point) and its flat
#      buffers hold exactly the real slots;
#   6. read-outs work identically under both layouts via user_slots().
# Run: conda run -n protomf python Master/temp/dc_checks/dc08/t03_bag_pooling.py
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dc05')))
import torch

from _harness import make_check

from feature_extraction.feature_ids import build_user_history_weights
from feature_extraction.feature_extractors import HistoryFeatureEmbedding
from utilities.explanations.history_readout import user_history_rows

check, FAILS = make_check()
torch.manual_seed(11)

tmp = tempfile.mkdtemp(prefix='t03_dc08_')
toy = os.path.join(tmp, 'toy')
os.makedirs(toy)
N_USERS, N_ITEMS, D = 6, 7, 5
with open(os.path.join(toy, 'user_ids.csv'), 'w') as f:
    f.write('user_id,user\n' + ''.join(f'{u},u{u}\n' for u in range(N_USERS)))
with open(os.path.join(toy, 'item_ids.csv'), 'w') as f:
    f.write('item_id,item\n' + ''.join(f'{i},a{i}\n' for i in range(N_ITEMS)))
with open(os.path.join(toy, 'item_features.csv'), 'w') as f:
    f.write('item_id,item,dep,col\n'
            '0,a0,Blouse,Black\n1,a1,Trouser,Black|White\n2,a2,Blouse,White\n'
            '3,a3,Trouser,Red\n4,a4,Dress,Red|Black|Green\n5,a5,Dress,Blue\n'
            '6,a6,Blouse,Blue|Red\n')
# u0: 4 purchases | u1: 1 (single) | u2: none (empty history) | u3: 6 | u4: 2 | u5: 3
HIST = {0: [0, 1, 2, 3], 1: [4], 3: [0, 1, 2, 3, 4, 5], 4: [5, 6], 5: [1, 3, 6]}
with open(os.path.join(toy, 'listening_history_train.csv'), 'w') as f:
    f.write('t_dat,customer_id,article_id,user_id,user,item_id,item\n')
    for u, items in HIST.items():
        for i in items:
            f.write(f'2020-09-01,u{u},a{i},{u},u{u},{i},a{i}\n')
LAM = 0.6


def make(ids, w, nf, tok, tokw, sz, nh, use_id, pooling, thr=256):
    m = HistoryFeatureEmbedding(N_USERS, ids, w, nf, D, use_id_feature=use_id,
                                item_token_ids=tok, item_token_w=tokw, hist_sizes=sz,
                                loo_pooling=True, n_history_rows=nh, history_row_weight=LAM,
                                pooling=pooling, pooling_auto_threshold=thr)
    return m


for layout in ('fixed', 'bags'):
    fields = ['dep', 'col'] if layout == 'bags' else ['dep']
    for id_rows in (False, True):
        payload = build_user_history_weights(toy, fields, layout=layout, return_item_tokens=True,
                                             history_id_rows=id_rows, history_row_weight=LAM)
        for use_id in (True, False):
            tag = f"t03.{layout}.y{int(id_rows)}.ids{int(use_id)}"
            pad = make(*payload, use_id, 'padded')
            bag = make(*payload, use_id, 'bag')
            bag.load_state_dict(pad.state_dict())
            check(f"{tag}.modes_selected",
                  pad.pooling == 'padded' and bag.pooling == 'bag')
            check(f"{tag}.bag_drops_padded_tensors",
                  not hasattr(bag, 'hist_value_ids') and hasattr(bag, 'flat_ids'))
            check(f"{tag}.flat_holds_exactly_real_slots",
                  int(bag.flat_ids.numel()) == int((payload[1] != 0).sum()),
                  f"{int(bag.flat_ids.numel())} vs {int((payload[1] != 0).sum())}")

            # (1) same composition
            pad.eval(); bag.eval()
            u_all = torch.arange(N_USERS)
            with torch.no_grad():
                e = float((pad(u_all) - bag(u_all)).abs().max())
            check(f"{tag}.compositions_agree", e < 1e-6, f"max|diff|={e:.2e}")

            # 2-D index shape is supported too
            with torch.no_grad():
                q2 = bag(u_all.reshape(3, 2))
            check(f"{tag}.bag_supports_2d_idx", tuple(q2.shape) == (3, 2, D), f"{tuple(q2.shape)}")

            # (2) leave-one-out agrees
            us, pos = [], []
            for u, items in HIST.items():
                for i in items:
                    us.append(u); pos.append(i)
            us_t, pos_t = torch.tensor(us), torch.tensor(pos)
            pad.train(); bag.train()
            with torch.no_grad():
                e = float((pad(us_t, pos_i_idxs=pos_t) - bag(us_t, pos_i_idxs=pos_t)).abs().max())
            check(f"{tag}.loo_agrees", e < 1e-6, f"max|diff|={e:.2e}")

            # (3) gradients agree
            for m in (pad, bag):
                m.zero_grad()
                m(us_t, pos_i_idxs=pos_t).pow(2).sum().backward()
            gp = pad.embedding_layer.weight.grad
            gb = bag.embedding_layer.weight.grad
            e = float((gp - gb).abs().max())
            scale = float(gp.abs().max())
            # Relative tolerance: the two paths accumulate the SAME terms in different orders
            # (padded gather-then-sum vs embedding_bag's fused scatter-add), so agreement is to
            # float32 round-off relative to the gradient's own magnitude, not absolute.
            check(f"{tag}.gradients_agree", e <= 1e-5 * max(scale, 1.0),
                  f"max|diff|={e:.2e} rel={e / max(scale, 1e-12):.2e} (max|grad|={scale:.1f})")

            # (6) read-outs agree through user_slots()
            same = True
            for u in range(N_USERS):
                cp, _, _ = user_history_rows(pad, u)
                cb, _, _ = user_history_rows(bag, u)
                if cp.shape != cb.shape:
                    same = False
                elif cp.numel() and float((cp - cb).abs().max()) > 1e-6:
                    same = False
            check(f"{tag}.readout_rows_agree", same)

# (4) 'auto' threshold behaviour
payload = build_user_history_weights(toy, ['dep'], layout='fixed', return_item_tokens=True,
                                     history_id_rows=True, history_row_weight=LAM)
d_max = payload[0].shape[1]
lo = make(*payload, False, 'auto', thr=d_max)          # not greater than threshold -> padded
hi = make(*payload, False, 'auto', thr=d_max - 1)      # greater -> bag
check("t03.auto.picks_padded_at_or_below_threshold", lo.pooling == 'padded', f"D_max={d_max}")
check("t03.auto.picks_bag_above_threshold", hi.pooling == 'bag', f"D_max={d_max}")

print("\nt03: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: {FAILS}"))
sys.exit(1 if FAILS else 0)
