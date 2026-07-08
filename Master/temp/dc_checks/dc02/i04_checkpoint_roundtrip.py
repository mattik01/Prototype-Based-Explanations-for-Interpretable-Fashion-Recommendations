# i04 — checkpoint roundtrip in BOTH modes (dc02 P8), emulating the tester path:
# save state_dict -> fresh factory rebuild (fresh injection) -> load_state_dict(strict=True)
# -> forward reproduces exactly. X (non-persistent buffer) never round-trips.
#
# Documented hazard (design decision, not a bug): the state_dict key is 'prototypes' in BOTH
# modes — it stores A in base mode and Ã (pre-softplus) in nonneg mode. A base checkpoint would
# strict-load into a nonneg build without error but with changed semantics. This is safe in
# practice because tester/loader always rebuild the model from the checkpoint's OWN config.json
# (which carries `nonneg_prototypes`), so the interpreting flag travels with the weights.
import os
import sys
import tempfile

import torch

from _harness import make_check, toy_attr_data, attr_item_proto_param, build_recsys, toy_batch

torch.manual_seed(29)
check, fails = make_check()

N_u, M, d, Lu, K = 9, 13, 8, 4, 5
X, offs = toy_attr_data(M=M, V_fs=(3, 4, 5), seed=0)
u, i, _ = toy_batch(N_u, M, B=4, n_neg=2, seed=3)

for mode, knobs in (('base', {}), ('nonneg', {'nonneg_prototypes': True})):
    rec = build_recsys(attr_item_proto_param(d, Lu, K, X, offs, **knobs), N_u, M, use_bias=0)
    sd = rec.state_dict()

    check(f"[{mode}] no buffer key in state_dict",
          not any('attr_multi_hot' in k for k in sd))
    item_keys = sorted(k for k in sd if k.startswith('item_feature_extractor'))
    check(f"[{mode}] item branch keys exactly {{prototypes, linear weight}}",
          item_keys == ['item_feature_extractor.model_1.prototypes',
                        'item_feature_extractor.model_2.linear_layer.weight'], f"{item_keys}")

    out_before = rec(u, i).detach()
    rec.item_feature_extractor.get_and_reset_loss()
    rec.user_feature_extractor.get_and_reset_loss()

    with tempfile.TemporaryDirectory(prefix='dc02_i04_') as tmp:
        path = os.path.join(tmp, 'best_model.pth')
        torch.save(sd, path)

        # fresh rebuild: new random init + fresh injection-shaped param dict (the tester path)
        rec2 = build_recsys(attr_item_proto_param(d, Lu, K, X, offs, **knobs), N_u, M, use_bias=0)
        out_fresh = rec2(u, i).detach()
        rec2.item_feature_extractor.get_and_reset_loss()
        rec2.user_feature_extractor.get_and_reset_loss()
        check(f"[{mode}] fresh build differs before load (sanity)",
              not torch.equal(out_fresh, out_before))

        params = torch.load(path, map_location='cpu')
        missing, unexpected = rec2.load_state_dict(params, strict=True), None
        check(f"[{mode}] strict load: no missing/unexpected keys",
              missing.missing_keys == [] and missing.unexpected_keys == [])

        out_after = rec2(u, i).detach()
        rec2.item_feature_extractor.get_and_reset_loss()
        rec2.user_feature_extractor.get_and_reset_loss()
        check(f"[{mode}] forward reproduces exactly after reload",
              torch.equal(out_after, out_before))

    if mode == 'nonneg':
        stored = sd['item_feature_extractor.model_1.prototypes']
        eff = rec.item_feature_extractor.model_1.effective_prototypes().detach()
        check("[nonneg] stored tensor is raw Ã (has negative entries)",
              bool((stored < 0).any()))
        check("[nonneg] softplus(stored) == effective prototypes",
              torch.equal(torch.nn.functional.softplus(stored), eff))

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
