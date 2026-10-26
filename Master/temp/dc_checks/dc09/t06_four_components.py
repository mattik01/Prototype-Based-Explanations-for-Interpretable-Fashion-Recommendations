# dc09 t06 — the score is the four-component sum (design doc §3.1, 4b C1) and the role pairing
# is the host's: user output [u*, û], item output [t̂, t*], so the dot pairs u* with t̂ (both
# K_u wide) and û with t* (both K_t wide). K_u != K_t here, so a swapped layout could still
# broadcast — the pairing is therefore checked against the modules' own roles, not shapes:
#   S = 1ᵀt̂ + (u*−1)ᵀt̂ + 1ᵀû + ûᵀ(t*−1)   (B_item + U-half + B_user + I-half), float64.
import sys

import torch

from _fixture import K_U, build, make_check, param_fUfI, toy_world, train_batch

check, FAILS = make_check()
w = toy_world()
u, i, _ = train_batch(w)

for share in (True, False):
    for ids in (True, False):
        for layout in ('fixed', 'bags'):
            tag = f'{"shared" if share else "sep"}.{"ids" if ids else "noid"}.{layout}'
            rec = build(param_fUfI(w, share=share, item_id=ids, user_id=ids, layout=layout)).double()
            rec.eval()
            ufe, ife = rec.user_feature_extractor, rec.item_feature_extractor
            with torch.no_grad():
                S = rec(u, i)
                uo, io = ufe(u), ife(i)
                q_u, q_i = ufe.base(u), ife.base(i)
            u_star, u_hat = uo[:, :K_U], uo[:, K_U:]
            t_hat, t_star = io[..., :K_U], io[..., K_U:]
            check(f't06.{tag}.user_linear_role_is_W_u_q_u',
                  torch.equal(u_hat, ufe.linear_layer(q_u).detach()))
            check(f't06.{tag}.item_linear_role_is_W_t_q_i',
                  torch.equal(t_hat, ife.linear_layer(q_i).detach()))
            parts = (t_hat.sum(-1) + ((u_star - 1).unsqueeze(1) * t_hat).sum(-1)
                     + u_hat.sum(-1, keepdim=True) + (u_hat.unsqueeze(1) * (t_star - 1)).sum(-1))
            err = ((S - parts).abs().max() / S.abs().max()).item()
            check(f't06.{tag}.four_components_sum_to_S', err < 1e-13, f'max rel {err:.1e}')
            ufe.get_and_reset_loss(); ife.get_and_reset_loss()

print("\nt06: ALL PASS" if not FAILS else f"\nt06: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
