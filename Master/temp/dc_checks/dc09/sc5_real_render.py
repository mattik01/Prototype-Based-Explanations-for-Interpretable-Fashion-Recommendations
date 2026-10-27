# dc09 SC.5 — the new breakdown slots on REAL data (login node, CPU, read-only on checkpoints).
# For each dataset: the fUfI smoke checkpoints of all four arms (SC.3 stage D, 3 epochs — content is
# NOT read; this checks the slots' exactness self-checks and layout at real scale: hm fixed fields,
# ml-1m bags with long baskets and the 1,043-tag what-if) and the UI host's 100-trial checkpoint,
# rendered side by side for the same user and the same item (the fUfI arm's top-1). Every artifact
# goes to OUT, never into a results folder. Usage:
#   python sc5_real_render.py <smoke_results_root> <n100_results_root> <data_root> <out_dir>
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO)

from utilities.explanations.breakdown import render_for_results_dir  # noqa: E402

smoke, n100, data, out = sys.argv[1:5]
USERS = {'hm_1_month': (0, 42, 4242), 'ml-1m': (0, 42, 3000)}
ARMS = ('feature_user_item_proto', 'feature_user_item_proto_noid', 'feature_user_item_proto_sep',
        'feature_user_item_proto_noid_sep')
fails = 0
for ds, users in USERS.items():
    host = os.path.join(n100, f'user_item_proto_{ds}_s38210573_n100')
    for arm in ARMS:
        rd = os.path.join(smoke, f'{arm}_{ds}_s38210573')
        for u in users:
            t0 = time.time()
            try:
                stem = render_for_results_dir(rd, u, out_dir=os.path.join(out, ds, arm),
                                              data_dir=os.path.join(data, ds))
                item = int(stem.rsplit('_item', 1)[1])
                print(f'OK {ds} {arm} user {u} item {item} ({time.time() - t0:.1f}s)', flush=True)
                if arm == 'feature_user_item_proto':
                    render_for_results_dir(host, u, item_id=item,
                                           out_dir=os.path.join(out, ds, 'user_item_proto_n100'),
                                           data_dir=os.path.join(data, ds))
                    print(f'OK {ds} host user {u} item {item}', flush=True)
            except Exception as e:
                fails += 1
                print(f'FAIL {ds} {arm} user {u}: {e!r}', flush=True)
# F-DC09-19 (SC.5 gate): the canonical read-out and the common-to-all-users line in the earlier
# slots — rendered on the wave-1 100-trial checkpoints (read only; artifacts to OUT/<ds>/lineage/).
LINEAGE = ('feature_item_proto', 'feature_item_proto_noid', 'item_proto', 'feature_user_proto',
           'feature_user_proto_noid', 'user_proto', 'lightfm_tags', 'lightfm_tags_ids')
for ds, users in USERS.items():
    for m in LINEAGE:
        rd = os.path.join(n100, f'{m}_{ds}_s38210573_n100')
        if not os.path.exists(os.path.join(rd, 'best_model.pth')):
            print(f'SKIP {ds} {m} (no 100-trial checkpoint)', flush=True)
            continue
        for u in users[:2]:
            try:
                render_for_results_dir(rd, u, out_dir=os.path.join(out, ds, 'lineage', m),
                                       data_dir=os.path.join(data, ds))
                print(f'OK {ds} {m} user {u}', flush=True)
            except Exception as e:
                fails += 1
                print(f'FAIL {ds} {m} user {u}: {e!r}', flush=True)

# weight_viz now joins fUfI: the explanations pipeline on one fUfI smoke checkpoint (its own
# results folder is a smoke folder; artifacts under explanations_sc5/)
from utilities.explanations.pipeline import run_explanations_pipeline  # noqa: E402
try:
    d = run_explanations_pipeline(os.path.join(smoke, 'feature_user_item_proto_hm_1_month_s38210573'),
                                  output_subdir='explanations_sc5', sample_users_for_weight_viz=[42])
    files = os.listdir(d)
    ok = {'weight_viz_user42_user_side.png', 'weight_viz_user42_item_side.png'} <= set(files)
    print(('OK' if ok else 'FAIL') + f' pipeline fUfI hm: {len(files)} files', flush=True)
    fails += 0 if ok else 1
except Exception as e:
    fails += 1
    print(f'FAIL pipeline fUfI hm: {e!r}', flush=True)
print('SC5_RENDER', 'ALL OK' if not fails else f'{fails} FAIL')
