# dc01 I3′ — gradient reaches the feature rows through the SINGLE score path (u·t*).
# Re-pointed at the SC.3 re-run (2026-07-12): under the fI host there is exactly one score
# surface (S = u·t*, t* = 1 + cos(q_i, P^t)); this is the integration analogue of the fI
# re-check C3′ (was "both UI halves" / C3 under the UI build — in git history).
# Run: python Master/temp/dc_checks/dc01/i03_gradient_single_path.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check, feature_item_proto_param, build_recsys, toy_batch

check, FAILS = make_check()

torch.manual_seed(0)
N_USERS, N_ITEMS, D, KT, F, NF = 5, 6, 8, 4, 2, 7
feature_ids = torch.randint(0, NF, (N_ITEMS, F))

rec = build_recsys(feature_item_proto_param(D, KT, feature_ids, NF, use_id_feature=True),
                   N_USERS, N_ITEMS)
item_proto = rec.item_feature_extractor
item_feat_embed = item_proto.embedding_ext

u, i, _ = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2, seed=5)
# a metadata feature row used by the first positive item, and that item's ID row
scored_item = int(i[0, 0])
used_row = int(feature_ids[scored_item, 0])
id_row = item_feat_embed.n_features + scored_item

rec.zero_grad(set_to_none=True)
score = rec(u, i).sum()
score.backward()
g = item_feat_embed.embedding_layer.weight.grad
check("i03.grad_table_populated", g is not None, "")
check("i03.grad_on_used_metadata_row", g[used_row].norm().item() > 1e-8,
      f"||∂S/∂e_row|| = {g[used_row].norm().item():.3e}")
check("i03.grad_on_scored_item_id_row", g[id_row].norm().item() > 1e-8,
      f"||∂S/∂e_id|| = {g[id_row].norm().item():.3e}")

# an unused metadata row gets NO gradient (indexed-lookup discipline, no full-vocab pass)
unused_rows = sorted(set(range(NF)) - set(feature_ids[i.flatten()].flatten().tolist()))
if unused_rows:
    check("i03.no_grad_on_unused_row", g[unused_rows[0]].norm().item() == 0.0,
          f"row {unused_rows[0]}: {g[unused_rows[0]].norm().item():.3e}")

# prototypes receive gradient through the same single path
check("i03.grad_on_prototypes", item_proto.prototypes.grad is not None
      and item_proto.prototypes.grad.abs().sum().item() > 0, "")

print("\ni03: ALL PASS" if not FAILS else f"\ni03: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
