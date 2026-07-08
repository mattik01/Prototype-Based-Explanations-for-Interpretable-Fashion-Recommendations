# i05 — claims 2 + 6 through the factory-built model (dc02 P8):
#  - COLD ITEM: item M-1 (X-row twin of item 0) is NEVER sampled during a 200-step toy training
#    run; afterwards its t* is finite, non-constant across k, and BITWISE equal to its warm twin's.
#  - EXACT TIES: same-signature items receive identical full scores for EVERY user (no biases).
# (The "a per-item bias destroys the tie" half of claim 6 was verified in the frozen toy check;
#  under the dc02 config use_bias=0 is mandatory, asserted here.)
import sys

import torch

from _harness import make_check, toy_attr_data, attr_item_proto_param, build_recsys

torch.manual_seed(31)
check, fails = make_check()

N_u, M, d, Lu, K = 9, 13, 8, 4, 5
X, offs = toy_attr_data(M=M, V_fs=(3, 4, 5), seed=0)
COLD = M - 1
check("toy data: cold twin shares item 0's X row (setup)", torch.equal(X[COLD], X[0]))

rec = build_recsys(attr_item_proto_param(d, Lu, K, X, offs), N_u, M, use_bias=0)
check("use_bias is False under the dc02 config (§3.5(e))", rec.use_bias is False)

opt = torch.optim.Adam(rec.parameters(), lr=1e-2)
g = torch.Generator().manual_seed(99)
first_loss = last_loss = None
for step in range(200):
    u = torch.randint(0, N_u, (8,), generator=g)
    i = torch.randint(0, M - 1, (8, 3), generator=g)      # NEVER samples item M-1
    labels = torch.zeros(8, 3)
    labels[:, 0] = 1.0
    out = rec(u, i)
    loss = rec.loss_func(out, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step == 0:
        first_loss = float(loss)
    last_loss = float(loss)

check("toy training loss decreased", last_loss < first_loss,
      f"{first_loss:.4f} -> {last_loss:.4f}")

item_proto = rec.item_feature_extractor.model_1
with torch.no_grad():
    t_pair = item_proto(torch.tensor([0, COLD]))          # (2, K)
item_proto.get_and_reset_loss()

check("claim 2: cold item t* finite", bool(torch.isfinite(t_pair[1]).all()))
check("claim 2: cold item t* non-constant across k", float(t_pair[1].std()) > 1e-4,
      f"std {float(t_pair[1].std()):.4f}")
check("claim 2: cold t* BITWISE equal to warm twin's", torch.equal(t_pair[0], t_pair[1]))

# claim 6: exact score tie for EVERY user
with torch.no_grad():
    u_all = torch.arange(N_u)
    i_pair = torch.tensor([[0, COLD]]).expand(N_u, 2)
    scores = rec(u_all, i_pair)                           # (N_u, 2)
rec.item_feature_extractor.get_and_reset_loss()
rec.user_feature_extractor.get_and_reset_loss()
gap = (scores[:, 0] - scores[:, 1]).abs().max()
check("claim 6: exact tie for every user", float(gap) == 0.0, f"max gap {float(gap):.2e}")
check("claim 6: ties are non-trivial (scores vary across users)",
      float(scores[:, 0].std()) > 1e-4)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
