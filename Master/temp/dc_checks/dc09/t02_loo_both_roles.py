# dc09 t02 — leave-one-out reaches BOTH user roles (design doc 4b C8, SC.2 D2).
# In training, RecSys hands the positive to the user wrapper, which composes ONE q_u^{LOO} and
# feeds it to the cosine role (u*) and the linear role (û): the wrapper output must equal
# [sim(q_LOO), W^u q_LOO] bitwise, differ from the full-basket composition, and eval mode must
# never apply leave-one-out. The item side never receives the positive. All four arms.
import sys

import torch

from _fixture import build, make_check, param_fUfI, toy_world, train_batch

check, FAILS = make_check()
w = toy_world()
u, i, labels = train_batch(w)
pos = i[:, 0]

for share in (True, False):
    for ids in (True, False):
        tag = f'{"shared" if share else "sep"}.{"ids" if ids else "noid"}'
        rec = build(param_fUfI(w, share=share, item_id=ids, user_id=ids))
        ufe, ife = rec.user_feature_extractor, rec.item_feature_extractor
        check(f't02.{tag}.wrapper_supports_loo', ufe.supports_loo and not ife.supports_loo)
        rec.train()
        with torch.no_grad():
            got = ufe(u, pos_i_idxs=pos)
            q_loo = ufe.base(u, pos_i_idxs=pos)
            want = torch.cat([ufe.proto.sim_from_embed(q_loo), ufe.linear_layer(q_loo)], dim=-1)
            q_full = ufe.base(u)
        ufe.get_and_reset_loss()
        check(f't02.{tag}.both_roles_read_q_loo', torch.equal(got, want), f'max {(got - want).abs().max():.1e}')
        check(f't02.{tag}.loo_changes_q', bool((q_loo - q_full).abs().max() > 1e-6),
              f'max |q_loo - q_full| {(q_loo - q_full).abs().max():.2e}')
        with torch.no_grad():
            rec(u, i)
        check(f't02.{tag}.recsys_train_applies_loo', ufe.base.last_forward_loo is True)
        rec.eval()
        with torch.no_grad():
            rec(u, i)
        check(f't02.{tag}.eval_never_applies_loo', ufe.base.last_forward_loo is False)
        ufe.get_and_reset_loss(); ife.get_and_reset_loss()

print("\nt02: ALL PASS" if not FAILS else f"\nt02: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
