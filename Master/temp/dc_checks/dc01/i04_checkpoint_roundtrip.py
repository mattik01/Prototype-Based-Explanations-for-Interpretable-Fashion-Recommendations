# dc01 I4 — checkpoint round-trip (tester path). feature_ids (persistent=False) absent from the
# saved keys; learned weights round-trip; strict load has no missing/unexpected keys; forward
# reproduces pre-save scores exactly. feature_ids is reconstructed at build (emulating P6).
# Run: python Master/temp/dc_checks/dc01/i04_checkpoint_roundtrip.py
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import make_check, feature_item_proto_param, build_recsys, toy_batch

check, FAILS = make_check()

torch.manual_seed(0)
N_USERS, N_ITEMS, D, KU, KT, F, NF = 5, 6, 8, 3, 4, 2, 7
feature_ids = torch.randint(0, NF, (N_ITEMS, F))

A = build_recsys(feature_item_proto_param(D, KU, KT, feature_ids, NF, use_id_feature=True),
                 N_USERS, N_ITEMS)
u, i, _ = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2, seed=7)
with torch.no_grad():
    logits_pre = A(u, i).clone()

sd = A.state_dict()
keys = list(sd.keys())
check("i04.feature_ids_absent_from_checkpoint", not any('feature_ids' in k for k in keys),
      "feature_ids is a non-persistent buffer")
check("i04.embedding_layer_weight_present", any('embedding_layer.weight' in k for k in keys), "")

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'best_model.pth')
    torch.save(sd, path)
    try:
        params = torch.load(path, weights_only=True)
    except TypeError:
        params = torch.load(path)

    # Rebuild fresh (feature_ids reconstructed deterministically from the same spec).
    C = build_recsys(feature_item_proto_param(D, KU, KT, feature_ids, NF, use_id_feature=True),
                     N_USERS, N_ITEMS)
    try:
        res = C.load_state_dict(params, strict=True)
        missing = list(getattr(res, 'missing_keys', []))
        unexpected = list(getattr(res, 'unexpected_keys', []))
        loaded_ok = not missing and not unexpected
        msg = f"missing={missing}, unexpected={unexpected}"
    except RuntimeError as e:
        loaded_ok = False
        msg = f"RuntimeError: {str(e)[:120]}"
    check("i04.strict_load_no_missing_unexpected", loaded_ok, msg)

    with torch.no_grad():
        logits_post = C(u, i).clone()
    err = (logits_pre - logits_post).abs().max().item()
    check("i04.forward_reproduces_after_load", err < 1e-6, f"max abs score diff {err:.3e}")

print("\ni04: ALL PASS" if not FAILS else f"\ni04: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
