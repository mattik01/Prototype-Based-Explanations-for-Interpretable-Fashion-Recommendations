# dc07 t05 — the eight arms are registered in BOTH registries (start.py choices + elif chain,
# run_combo.MODEL_CONFIGS — source scan, as dc01/dc02 t05 do), each config differs from its
# cosine parent by EXACTLY the two prototype-side keys {cosine_type, temperature}, τ is a
# loguniform sampler over [DC07_TAU_MIN, DC07_TAU_MAX], none is a headline (EXPLAINABLE) model,
# and every arm BUILDS through the factory on a toy dataset shape (sampled search space).
# Run: python Master/temp/dc_checks/dc07/t05_config_registration.py
import copy
import re
from _common import *  # noqa: F401,F403

from ray import tune  # noqa: E402
try:  # Ray ≥ 2.x (LEO5 venv) moved the sampler classes
    from ray.tune.search import sample as _sample  # noqa: E402
except ImportError:  # Ray 1.6.0 (laptop)
    from ray.tune import sample as _sample  # noqa: E402
import confs.hyper_params as hp  # noqa: E402

check, FAILS = make_check('t05')

ARMS = {  # name: (parent, side, mode)
    'user_proto_sm': ('user_proto_chose_original_hyper_params', 'user', 'softmax'),
    'item_proto_sm': ('item_proto_chose_original_hyper_params', 'item', 'softmax'),
    'user_proto_sg': ('user_proto_chose_original_hyper_params', 'user', 'sigmoid'),
    'item_proto_sg': ('item_proto_chose_original_hyper_params', 'item', 'sigmoid'),
    'feature_item_proto_sm': ('feature_item_proto_hyper_params', 'item', 'softmax'),
    'feature_item_proto_noid_sm': ('feature_item_proto_noid_hyper_params', 'item', 'softmax'),
    'feature_user_proto_sm': ('feature_user_proto_hyper_params', 'user', 'softmax'),
    'feature_user_proto_noid_sm': ('feature_user_proto_noid_hyper_params', 'user', 'softmax'),
}


def strip_samplers(d):
    """Replace tune samplers by a marker so dict equality ignores object identity."""
    if isinstance(d, dict):
        return {k: strip_samplers(v) for k, v in d.items()}
    if isinstance(d, (_sample.Domain,)):
        return ('SAMPLER', type(d).__name__, type(getattr(d, 'sampler', None)).__name__,
                getattr(d, 'lower', None), getattr(d, 'upper', None))
    return d


with open(os.path.join(REPO, 'start.py')) as f:
    start_src = f.read()
with open(os.path.join(REPO, 'Master', 'scripts', 'run_combo.py')) as f:
    rc_src = f.read()
mc = re.search(r"MODEL_CONFIGS\s*=\s*\{(.*?)\n\}", rc_src, re.DOTALL).group(1)
em = re.search(r"EXPLAINABLE_MODELS\s*=\s*(.*)", rc_src).group(1)

for name, (parent_name, side, mode) in ARMS.items():
    cfg = getattr(hp, f'{name}_hyper_params')
    parent = getattr(hp, parent_name)
    ps = cfg['ft_ext_param'][f'{side}_ft_ext_param']
    check(f"{name}.cosine_type", ps.get('cosine_type') == mode, f"{ps.get('cosine_type')}")
    t = ps.get('temperature')
    is_logu = isinstance(t, _sample.Float) and 'LogUniform' in type(t.sampler).__name__
    check(f"{name}.temperature_loguniform", is_logu and t.lower == hp.DC07_TAU_MIN and t.upper == hp.DC07_TAU_MAX,
          f"{type(t).__name__}[{getattr(t, 'lower', None)}, {getattr(t, 'upper', None)}]")
    # exactly two keys differ from the parent
    a, b = strip_samplers(copy.deepcopy(cfg)), strip_samplers(copy.deepcopy(parent))
    bp = b['ft_ext_param'][f'{side}_ft_ext_param']
    ap = a['ft_ext_param'][f'{side}_ft_ext_param']
    diff_keys = {k for k in set(ap) | set(bp) if ap.get(k) != bp.get(k)}
    a['ft_ext_param'][f'{side}_ft_ext_param'] = b['ft_ext_param'][f'{side}_ft_ext_param'] = None
    check(f"{name}.deepcopy_diff_is_exactly_two_keys", diff_keys == {'cosine_type', 'temperature'} and a == b,
          f"diff keys {sorted(diff_keys)}; rest equal: {a == b}")
    check(f"{name}.parent_untouched", parent['ft_ext_param'][f'{side}_ft_ext_param']['cosine_type'] == 'shifted'
          and 'temperature' not in parent['ft_ext_param'][f'{side}_ft_ext_param'])
    # registries
    check(f"{name}.start_choice", f"'{name}'" in start_src)
    check(f"{name}.start_elif", f"elif model == '{name}':\n    conf_dict = {name}_hyper_params" in start_src)
    check(f"{name}.run_combo_model_configs", f"'{name}': {name}_hyper_params" in mc)
    check(f"{name}.not_explainable", name not in em, f"EXPLAINABLE_MODELS = {em.strip()}")

# builds through the factory (hosts: generic 'prototypes' path with the τ keyword)
from feature_extraction.feature_extractor_factories import FeatureExtractorFactory  # noqa: E402


def sample(d):
    if isinstance(d, dict):
        return {k: sample(v) for k, v in d.items()}
    if isinstance(d, _sample.Domain):
        return d.sample()
    return d


torch.manual_seed(5)
for name in ('user_proto_sm', 'item_proto_sm', 'user_proto_sg', 'item_proto_sg'):
    cfg = sample(getattr(hp, f'{name}_hyper_params')['ft_ext_param'])
    u, i = FeatureExtractorFactory.create_models(cfg, 20, 30)
    side = u if name.startswith('user') else i
    check(f"{name}.factory_builds_membership_mode",
          isinstance(side, PrototypeEmbedding) and side.membership_mode and side.temperature == cfg[f"{'user' if name.startswith('user') else 'item'}_ft_ext_param"]['temperature'],
          f"τ={side.temperature:.4g}, mode={side.cosine_type}")
    check(f"{name}.sigmoid_bias_iff_sg", hasattr(side, 'membership_bias') == name.endswith('_sg'))
finish('t05', FAILS)
