# dc09 t07 — sharing keystone (design doc §3.2 SC.2 amendment, SC.2 toy D1): the shared model IS
# the `_sep` model with E^U_words := E^I_words tied. Weights of a shared model are copied into a
# `_sep` model (word rows into BOTH tables, item IDs into the item table, user IDs into the user
# table); then in training mode (leave-one-out on), under sampled softmax and BCE, float64,
# fixed and bags item layouts, ids and noid:
#   - logits and loss BITWISE equal (both arms run the same op sequence — plan G1),
#   - ∇ shared word rows = ∇ item-table word rows + ∇ user-table word rows (chain rule),
#   - ID-row and prototype / linear gradients equal.
import sys

import torch

from _fixture import N_ITEMS, build, make_check, param_fUfI, toy_world, train_batch
from utilities.utils import reproducible

check, FAILS = make_check()
w = toy_world()
V = w['V']
u, i, labels = train_batch(w)

for loss in ('sampled_softmax', 'bce'):
    for ids in (True, False):
        for layout in ('fixed', 'bags'):
            tag = f'{loss}.{"ids" if ids else "noid"}.{layout}'
            reproducible(5)
            sh = build(param_fUfI(w, share=True, item_id=ids, user_id=ids, layout=layout), loss).double()
            sp = build(param_fUfI(w, share=False, item_id=ids, user_id=ids, layout=layout), loss).double()
            su, si, pu, pi = (sh.user_feature_extractor, sh.item_feature_extractor,
                              sp.user_feature_extractor, sp.item_feature_extractor)
            T = si.base.embedding_layer.weight
            check(f't07.{tag}.one_table_both_sides', T is su.base.embedding_layer.weight
                  and pi.base.embedding_layer.weight is not pu.base.embedding_layer.weight)
            with torch.no_grad():
                pi.base.embedding_layer.weight[:V] = T[:V]
                pu.base.embedding_layer.weight[:V] = T[:V]
                if ids:
                    pi.base.embedding_layer.weight[V:] = T[V:V + N_ITEMS]
                    pu.base.embedding_layer.weight[V:] = T[V + N_ITEMS:]
                for a, b in ((pu, su), (pi, si)):
                    a.proto.prototypes.copy_(b.proto.prototypes)
                    a.linear_layer.weight.copy_(b.linear_layer.weight)
            res = []
            for m in (sh, sp):
                m.train()
                m.zero_grad()
                logits = m(u, i)
                lv = m.loss_func(logits, labels.double())
                lv.backward()
                res.append((logits.detach(), lv.detach()))
            check(f't07.{tag}.logits_bitwise', torch.equal(res[0][0], res[1][0]),
                  f'max {(res[0][0] - res[1][0]).abs().max():.1e}')
            check(f't07.{tag}.loss_bitwise', torch.equal(res[0][1], res[1][1]))
            G = T.grad
            summed = pi.base.embedding_layer.weight.grad[:V] + pu.base.embedding_layer.weight.grad[:V]
            rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
            check(f't07.{tag}.word_grad_is_sum_of_both_sides', rel(G[:V], summed) < 1e-13,
                  f'max rel {rel(G[:V], summed):.1e}; user-side share of |grad| '
                  f'{pu.base.embedding_layer.weight.grad[:V].norm() / summed.norm():.2f}')
            others = [(su.proto.prototypes.grad, pu.proto.prototypes.grad),
                      (si.proto.prototypes.grad, pi.proto.prototypes.grad),
                      (su.linear_layer.weight.grad, pu.linear_layer.weight.grad),
                      (si.linear_layer.weight.grad, pi.linear_layer.weight.grad)]
            if ids:
                others += [(G[V:V + N_ITEMS], pi.base.embedding_layer.weight.grad[V:]),
                           (G[V + N_ITEMS:], pu.base.embedding_layer.weight.grad[V:])]
            worst = max(rel(a, b) for a, b in others)
            check(f't07.{tag}.other_grads_equal', worst < 1e-13, f'max rel {worst:.1e}')

print("\nt07: ALL PASS" if not FAILS else f"\nt07: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
