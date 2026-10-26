# dc09 t01 — keystone: with ZERO words (V = 0) and ID rows on both sides, fUfI-ProtoMF IS the
# UI-ProtoMF host (design doc 4b C4). Weights are copied from a built `user_item_proto`
# (prototypes_double_tie) model into the fUfI model; then, in float32 and float64, under
# sampled softmax and BCE, in training mode:
#   - logits are BITWISE equal (same values through the same ops; q = the ID row exactly),
#   - the loss (rec + all four inclusion regularisers) is bitwise equal,
#   - every gradient matches to machine precision (not bitwise: the host accumulates two
#     lookups of the tied table, fUfI back-propagates one composed q read by two roles).
# Covers both table layouts (shared [item IDs ; user IDs] and `_sep`).
import sys

import torch

from _fixture import (N_ITEMS, N_USERS, build, make_check, param_UI, param_fUfI, train_batch,
                      zero_vocab_world)
from utilities.utils import reproducible

check, FAILS = make_check()
w = zero_vocab_world()


def copy_host_into(host, ufi, share):
    """host: Concatenate(PrototypeEmbedding, EmbeddingW) per side; ufi: TwoRoleComposedExtractor."""
    hu, hi = host.user_feature_extractor, host.item_feature_extractor
    uu, ui = ufi.user_feature_extractor, ufi.item_feature_extractor
    with torch.no_grad():
        u_tab = hu.model_1.embedding_ext.embedding_layer.weight
        i_tab = hi.model_1.embedding_ext.embedding_layer.weight
        if share:  # one table [item IDs ; user IDs] (V = 0)
            t = ui.base.embedding_layer.weight
            assert t is uu.base.embedding_layer.weight
            t[:N_ITEMS] = i_tab
            t[N_ITEMS:] = u_tab
        else:
            ui.base.embedding_layer.weight.copy_(i_tab)
            uu.base.embedding_layer.weight.copy_(u_tab)
        uu.proto.prototypes.copy_(hu.model_1.prototypes)
        ui.proto.prototypes.copy_(hi.model_1.prototypes)
        uu.linear_layer.weight.copy_(hu.model_2.linear_layer.weight)
        ui.linear_layer.weight.copy_(hi.model_2.linear_layer.weight)


def grad_pairs(host, ufi, share):
    hu, hi = host.user_feature_extractor, host.item_feature_extractor
    uu, ui = ufi.user_feature_extractor, ufi.item_feature_extractor
    hug = hu.model_1.embedding_ext.embedding_layer.weight.grad
    hig = hi.model_1.embedding_ext.embedding_layer.weight.grad
    if share:
        tg = ui.base.embedding_layer.weight.grad
        tabs = [('item_table', hig, tg[:N_ITEMS]), ('user_table', hug, tg[N_ITEMS:])]
    else:
        tabs = [('item_table', hig, ui.base.embedding_layer.weight.grad),
                ('user_table', hug, uu.base.embedding_layer.weight.grad)]
    return tabs + [('user_protos', hu.model_1.prototypes.grad, uu.proto.prototypes.grad),
                   ('item_protos', hi.model_1.prototypes.grad, ui.proto.prototypes.grad),
                   ('user_linear', hu.model_2.linear_layer.weight.grad, uu.linear_layer.weight.grad),
                   ('item_linear', hi.model_2.linear_layer.weight.grad, ui.linear_layer.weight.grad)]


for dtype, tol in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
    for loss in ('sampled_softmax', 'bce'):
        for share in (True, False):
            tag = f'{str(dtype)[6:]}.{loss}.{"shared" if share else "sep"}'
            reproducible(3)
            host = build(param_UI(), loss).to(dtype)
            ufi = build(param_fUfI(w, share=share), loss).to(dtype)
            copy_host_into(host, ufi, share)
            u, i, labels = train_batch(w)
            out = []
            for m in (host, ufi):
                m.train()
                m.zero_grad()
                logits = m(u, i)
                lv = m.loss_func(logits, labels.to(dtype))
                lv.backward()
                out.append((logits.detach(), lv.detach()))
            check(f't01.{tag}.logits_bitwise', torch.equal(out[0][0], out[1][0]),
                  f'max {(out[0][0] - out[1][0]).abs().max():.1e}')
            check(f't01.{tag}.loss_bitwise', torch.equal(out[0][1], out[1][1]),
                  f'{out[0][1].item():.9g} vs {out[1][1].item():.9g}')
            worst = max(((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()
                        for _, a, b in grad_pairs(host, ufi, share))
            check(f't01.{tag}.grads_machine_precision', worst < tol, f'max rel {worst:.1e} (tol {tol:.0e})')

print("\nt01: ALL PASS" if not FAILS else f"\nt01: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
