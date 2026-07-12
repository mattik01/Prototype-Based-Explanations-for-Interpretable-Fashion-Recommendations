# dc05 t08 — checkpoint roundtrip (dc01 i04 pattern). The history buffers are NON-persistent:
# they never enter the checkpoint, and a freshly rebuilt model (new RNG, same injected buffers)
# restored with strict load_state_dict must produce bit-identical scores.
# Run: python Master/temp/dc_checks/dc05/t08_checkpoint_roundtrip.py
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import (make_check, feature_user_proto_param, build_recsys, toy_batch,
                      make_toy_history)

check, FAILS = make_check()
torch.manual_seed(0)
N_USERS, N_ITEMS, V, D, KU = 8, 10, 6, 5, 3

ids, w = make_toy_history(N_USERS, V, d_max=3, seed=4)
A = build_recsys(feature_user_proto_param(D, KU, ids, w, V), N_USERS, N_ITEMS)

# one gradient step so the checkpoint is not the init state
u, i, labels = toy_batch(N_USERS, N_ITEMS, seed=7)
opt = torch.optim.SGD(A.parameters(), lr=0.1)
loss = A.loss_func(A(u, i), labels) + A.user_feature_extractor.get_and_reset_loss()
loss.backward()
opt.step()

with torch.no_grad():
    ref = A(u, i)
A.user_feature_extractor.get_and_reset_loss()

tmp = tempfile.mkdtemp(prefix='t08_dc05_')
ckpt = os.path.join(tmp, 'best_model.pth')
torch.save(A.state_dict(), ckpt)

state = torch.load(ckpt)
check("t08.no_buffers_in_checkpoint", not any('hist_' in k for k in state),
      f"{[k for k in state if 'hist_' in k]}")

# fresh rebuild (different RNG draws), same injected buffers, STRICT load
torch.manual_seed(999)
B = build_recsys(feature_user_proto_param(D, KU, ids, w, V), N_USERS, N_ITEMS)
missing_unexpected = B.load_state_dict(state)  # strict=True default: raises on mismatch
with torch.no_grad():
    out = B(u, i)
B.user_feature_extractor.get_and_reset_loss()
check("t08.strict_load_clean",
      not missing_unexpected.missing_keys and not missing_unexpected.unexpected_keys)
check("t08.scores_bit_identical_after_roundtrip", torch.equal(out, ref),
      f"max |Δ| {(out - ref).abs().max().item():.3e}")

print("\nt08: ALL PASS" if not FAILS else f"\nt08: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
