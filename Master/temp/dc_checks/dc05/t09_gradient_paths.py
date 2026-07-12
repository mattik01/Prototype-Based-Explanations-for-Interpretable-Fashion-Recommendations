# dc05 t09 — gradient paths (dc01 i03 pattern). Pins: gradients flow to the used word rows,
# the batch users' ID rows, the user prototypes and the item table; the history buffers carry
# no grad; rows only ever referenced through zero-weight padding slots receive EXACTLY zero
# gradient (padding is a true no-op in the backward pass too).
# Run: python Master/temp/dc_checks/dc05/t09_gradient_paths.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check, feature_user_proto_param, build_recsys

check, FAILS = make_check()
torch.manual_seed(0)
N_USERS, N_ITEMS, V, D, KU = 4, 6, 5, 4, 3

# Hand-built baskets that NEVER use word 0 with nonzero weight — word 0 appears only as the
# padding id. Words 3, 4 are never used either (never referenced at all).
ids = torch.tensor([[1, 2, 0],
                    [2, 0, 0],
                    [1, 0, 0],
                    [2, 1, 0]])
w = torch.tensor([[0.5, 0.5, 0.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.7, 0.3, 0.0]])

model = build_recsys(feature_user_proto_param(D, KU, ids, w, V), N_USERS, N_ITEMS)
hist = model.user_feature_extractor.embedding_ext

u = torch.tensor([0, 1, 3])                      # user 2 NOT in the batch
i = torch.tensor([[0, 1], [2, 3], [4, 5]])
labels = torch.zeros(3, 2); labels[:, 0] = 1.0

loss = model.loss_func(model(u, i), labels) + model.user_feature_extractor.get_and_reset_loss()
loss.backward()

W = hist.embedding_layer.weight
g = W.grad
check("t09.grad_exists_on_word_table", g is not None)
check("t09.used_word_rows_get_grad",
      float(g[1].abs().sum()) > 0 and float(g[2].abs().sum()) > 0,
      f"|g1|={float(g[1].abs().sum()):.2e}, |g2|={float(g[2].abs().sum()):.2e}")
check("t09.padding_only_row0_zero_grad", float(g[0].abs().sum()) == 0.0,
      f"|g0|={float(g[0].abs().sum()):.3e} (word 0 referenced ONLY via weight-0 padding)")
check("t09.never_referenced_rows_zero_grad",
      float(g[3].abs().sum()) == 0.0 and float(g[4].abs().sum()) == 0.0)
# ID rows: batch users get grad, absent user 2 does not
check("t09.batch_user_id_rows_get_grad",
      all(float(g[V + uu].abs().sum()) > 0 for uu in (0, 1, 3)))
check("t09.absent_user_id_row_zero_grad", float(g[V + 2].abs().sum()) == 0.0)

check("t09.prototypes_get_grad",
      model.user_feature_extractor.prototypes.grad is not None
      and float(model.user_feature_extractor.prototypes.grad.abs().sum()) > 0)
check("t09.item_table_gets_grad",
      model.item_feature_extractor.embedding_layer.weight.grad is not None
      and float(model.item_feature_extractor.embedding_layer.weight.grad.abs().sum()) > 0)
check("t09.buffers_carry_no_grad",
      hist.hist_value_ids.grad is None and hist.hist_weights.grad is None
      and not hist.hist_weights.requires_grad)

print("\nt09: ALL PASS" if not FAILS else f"\nt09: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
