# dc09 SC.8.4 — rendered-explanation spot-check on the 100-trial checkpoints (login node, CPU,
# read-only on checkpoints). Per dataset: the four fUfI arms' breakdowns for three users (each arm's
# own top-1 item), the UI host side by side for the same user and the same item as the ids arm's
# top-1; then the explanations pipeline (cards, naming both routes, t-SNE, top-k, weight_viz) on the
# ids arm and the host. Every artifact goes to OUT (absolute pipeline output dirs), never into a
# results folder. Usage:
#   python sc8_real_render.py <n100_results_root> <data_root> <out_dir>
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO)

from utilities.explanations.breakdown import render_for_results_dir  # noqa: E402
from utilities.explanations.pipeline import run_explanations_pipeline  # noqa: E402

n100, data, out = sys.argv[1:4]
USERS = {'hm_1_month': (0, 42, 4242), 'ml-1m': (0, 42, 3000)}
ARMS = ('feature_user_item_proto', 'feature_user_item_proto_noid', 'feature_user_item_proto_sep',
        'feature_user_item_proto_noid_sep')
fails = 0
for ds, users in USERS.items():
    host = os.path.join(n100, f'user_item_proto_{ds}_s38210573_n100')
    for arm in ARMS:
        rd = os.path.join(n100, f'{arm}_{ds}_s38210573_n100')
        for u in users:
            t0 = time.time()
            try:
                stem = render_for_results_dir(rd, u, out_dir=os.path.join(out, ds, arm),
                                              data_dir=os.path.join(data, ds))
                item = int(stem.rsplit('_item', 1)[1])
                print(f'OK {ds} {arm} user {u} item {item} ({time.time() - t0:.1f}s)', flush=True)
                if arm == 'feature_user_item_proto':
                    render_for_results_dir(host, u, item_id=item,
                                           out_dir=os.path.join(out, ds, 'user_item_proto'),
                                           data_dir=os.path.join(data, ds))
                    print(f'OK {ds} host user {u} item {item}', flush=True)
            except Exception as e:
                fails += 1
                print(f'FAIL {ds} {arm} user {u}: {e!r}', flush=True)
    for m in ('feature_user_item_proto', 'user_item_proto'):
        rd = os.path.join(n100, f'{m}_{ds}_s38210573_n100')
        try:
            d = run_explanations_pipeline(rd, output_subdir=os.path.join(out, ds, 'pipeline', m),
                                          sample_users_for_weight_viz=[42])
            print(f'OK pipeline {ds} {m}: {d} ({len(os.listdir(d))} files)', flush=True)
        except Exception as e:
            fails += 1
            print(f'FAIL pipeline {ds} {m}: {e!r}', flush=True)
print('SC8_RENDER', 'ALL OK' if not fails else f'{fails} FAIL')
