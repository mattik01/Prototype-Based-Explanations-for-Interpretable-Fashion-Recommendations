# i06 — regression build-check: existing models still assemble/forward/backward after the dc02
# additions (which touch no existing class body; the full dc01 suite rerun is the deeper gate).
import sys

import torch

from _harness import make_check, user_item_proto_param, mf_param, build_recsys, toy_batch

torch.manual_seed(37)
check, fails = make_check()

N_u, M, d = 9, 13, 8


def exercise(tag, param):
    rec = build_recsys(param, N_u, M, use_bias=0)
    u, i, labels = toy_batch(N_u, M, B=4, n_neg=2, seed=4)
    out = rec(u, i)
    loss = rec.loss_func(out, labels)
    loss.backward()
    check(f"{tag}: builds, forwards (B,1+n_neg), backward finite",
          out.shape == (4, 3) and bool(torch.isfinite(loss).all()))


exercise("user_item_proto", user_item_proto_param(d, 4, 5))
exercise("mf", mf_param(d))

# feature_item_proto (dc01, fI host since the 2026-07-12 re-host): tiny synthetic feature_ids
# (2 fields, vocab 2+2, injected form); user side = plain embedding (I-ProtoMF shape)
feature_ids = torch.stack([torch.randint(0, 2, (M,)), 2 + torch.randint(0, 2, (M,))], dim=1)
fip = {
    'ft_type': 'feature_item_proto', 'embedding_dim': d,
    'user_ft_ext_param': {'ft_type': 'embedding'},
    'item_ft_ext_param': {'ft_type': 'feature_item_proto', 'sim_proto_weight': 1.0,
                          'sim_batch_weight': 1.0, 'use_weight_matrix': False, 'n_prototypes': 5,
                          'cosine_type': 'shifted', 'reg_proto_type': 'max', 'reg_batch_type': 'max',
                          'use_id_feature': True, 'feature_fields': ['f0', 'f1'],
                          'feature_ids': feature_ids, 'n_features': 4},
}
exercise("feature_item_proto", fip)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
