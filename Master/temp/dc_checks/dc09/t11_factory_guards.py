# dc09 t11 — the factory refuses configs it would otherwise build wrongly (SC.4 findings 1 + 6):
#   - `normalize_composed` (the design doc's §3.7 length-ablation control) is not built, so a config
#     carrying it must fail instead of silently training the unablated model;
#   - one vocabulary index means the same fields in the same layout on both sides, not only equal
#     counts (two equal-size field lists would misalign the shared words);
#   - the default payload (no field lists, or identical ones) still builds.
import sys

from _fixture import build, make_check, param_fUfI, toy_world

check, FAILS = make_check()
w = toy_world()


def raises(mutate):
    p = param_fUfI(w, share=True)
    mutate(p)
    try:
        build(p)
        return False, 'built'
    except AssertionError as e:
        return True, str(e)[:80]


for where in ('ft', 'user', 'item'):
    def set_norm(p, where=where):
        {'ft': p, 'user': p['user_ft_ext_param'], 'item': p['item_ft_ext_param']}[where]['normalize_composed'] = True
    ok, msg = raises(set_norm)
    check(f't11.normalize_composed_rejected[{where}]', ok, msg)


def mismatch_fields(p):
    p['item_ft_ext_param']['feature_fields'] = ['colour', 'section']
    p['user_ft_ext_param']['feature_fields'] = ['colour', 'department']


def mismatch_layout(p):
    p['item_ft_ext_param']['feature_layout'] = 'fixed'
    p['user_ft_ext_param']['feature_layout'] = 'bags'


for name, mutate in (('fields', mismatch_fields), ('layout', mismatch_layout)):
    ok, msg = raises(mutate)
    check(f't11.vocabulary_identity_enforced[{name}]', ok, msg)


def same_fields(p):
    for side in ('item_ft_ext_param', 'user_ft_ext_param'):
        p[side].update(feature_fields=['colour', 'section'], feature_layout='fixed')


ok, msg = raises(same_fields)
check('t11.identical_vocabulary_still_builds', not ok, msg)
ok, msg = raises(lambda p: None)
check('t11.default_payload_still_builds', not ok, msg)

print("\nt11: ALL PASS" if not FAILS else f"\nt11: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
