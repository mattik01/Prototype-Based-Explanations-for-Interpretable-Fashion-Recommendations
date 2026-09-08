# dc07 t06 — the three cosine modes are untouched: (a) the dc01 RNG-order golden (t03) still
# passes bitwise; (b) 'shifted' / 'standard' / 'shifted_and_div' forwards equal the manual
# a·cos + c formula in float64; (c) a cosine-mode module has no membership_bias parameter and
# ignores `temperature`; (d) the factory's generic positional call still builds the stock module.
# Run: python Master/temp/dc_checks/dc07/t06_cosine_modes_untouched.py
import subprocess
from _common import *  # noqa: F401,F403

check, FAILS = make_check('t06')
r = subprocess.run([sys.executable, os.path.join(REPO, 'Master', 'temp', 'dc_checks', 'dc01', 't03_prototype_injection.py')],
                   capture_output=True, text=True)
check("dc01_t03_golden_still_bitwise", r.returncode == 0 and 'ALL PASS' in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])

torch.manual_seed(6)
N, D, K = 40, 12, 7
AFF = {'shifted': (1.0, 1.0), 'standard': (1.0, 0.0), 'shifted_and_div': (0.5, 0.5)}
for ct, (a, c) in AFF.items():
    for tau in (1.0, 0.1):
        m = proto(N, D, K, ct, tau)
        with torch.no_grad():
            idx = torch.arange(N); out = m(idx); q = m.embedding_ext(idx)
            cos = torch.nn.functional.normalize(q, dim=1) @ torch.nn.functional.normalize(m.prototypes, dim=1).T
            ref = c + a * cos
        check(f"{ct}.tau{tau}.equals_affine_cos", torch.allclose(out, ref, atol=1e-12), f"{(out - ref).abs().max():.2e}")
        check(f"{ct}.tau{tau}.no_membership_params", not hasattr(m, 'membership_bias') and not m.membership_mode)
        m.get_and_reset_loss()
    m1, m2 = proto(N, D, K, ct, 1.0), None
    torch.manual_seed(6); m1 = proto(N, D, K, ct, 1.0)
    torch.manual_seed(6); m2 = proto(N, D, K, ct, 0.01)
    with torch.no_grad():
        same = torch.equal(m1(torch.arange(N)), m2(torch.arange(N)))
    check(f"{ct}.temperature_ignored", same)

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory  # noqa: E402
cfg = {'ft_type': 'prototypes', 'embedding_dim': D,
       'user_ft_ext_param': {'ft_type': 'prototypes', 'n_prototypes': K, 'cosine_type': 'shifted',
                             'reg_proto_type': 'max', 'reg_batch_type': 'max', 'use_weight_matrix': False,
                             'sim_proto_weight': 1.0, 'sim_batch_weight': 1.0},
       'item_ft_ext_param': {'ft_type': 'embedding'}}
u, i = FeatureExtractorFactory.create_models(cfg, N, 20)
check("factory_generic_path_stock_module", isinstance(u, PrototypeEmbedding) and u.cosine_type == 'shifted'
      and u.temperature == 1.0 and not u.membership_mode and u.cosine_sim_func is not None)
finish('t06', FAILS)
