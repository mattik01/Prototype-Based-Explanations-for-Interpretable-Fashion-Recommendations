# dc09 t09 — checkpoint round-trip keeps the shared table SHARED.
# A trained-state model is saved (state_dict, as the Trainer does) and loaded into a freshly
# built model with a different init: eval logits must be bitwise equal, the item and user
# compositions must still borrow ONE Parameter after loading (load_state_dict copies in place),
# and parameters() / an optimizer must see that table once. `_sep` arms: two tables.
import io
import sys

import torch

from _fixture import build, make_check, param_fUfI, toy_world, train_batch
from utilities.utils import reproducible

check, FAILS = make_check()
w = toy_world()
u, i, labels = train_batch(w)

for share in (True, False):
    for ids in (True, False):
        tag = f'{"shared" if share else "sep"}.{"ids" if ids else "noid"}'
        reproducible(1)
        a = build(param_fUfI(w, share=share, item_id=ids, user_id=ids))
        opt = torch.optim.Adam(a.parameters(), lr=0.05)
        a.train()
        for _ in range(3):  # move away from init so the round-trip is not trivially init == init
            opt.zero_grad()
            a.loss_func(a(u, i), labels).backward()
            opt.step()
        buf = io.BytesIO()
        torch.save(a.state_dict(), buf)
        buf.seek(0)
        reproducible(2)
        b = build(param_fUfI(w, share=share, item_id=ids, user_id=ids))
        b.load_state_dict(torch.load(buf))
        a.eval(); b.eval()
        with torch.no_grad():
            la, lb = a(u, i), b(u, i)
        check(f't09.{tag}.logits_bitwise_after_load', torch.equal(la, lb), f'max {(la - lb).abs().max():.1e}')
        ti = b.item_feature_extractor.base.embedding_layer.weight
        tu = b.user_feature_extractor.base.embedding_layer.weight
        check(f't09.{tag}.table_sharing_survives_load', (ti is tu) == share)
        n_unique = len({id(p) for p in b.parameters()})
        n_opt = sum(len(g['params']) for g in torch.optim.Adam(b.parameters()).param_groups)
        expected = 5 + (0 if share else 1)  # table(s) + 2 prototype sets + 2 linear maps
        check(f't09.{tag}.parameters_seen_once', n_unique == n_opt == expected,
              f'unique {n_unique}, optimizer {n_opt}, expected {expected}')
        a.item_feature_extractor.get_and_reset_loss(); a.user_feature_extractor.get_and_reset_loss()
        b.item_feature_extractor.get_and_reset_loss(); b.user_feature_extractor.get_and_reset_loss()

print("\nt09: ALL PASS" if not FAILS else f"\nt09: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
