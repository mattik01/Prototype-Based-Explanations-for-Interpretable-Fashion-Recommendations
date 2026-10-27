# S0 pin — one results folder per run, and no silent overwrite (wave 2, 2026-10-27).
#   1. hyperopts: production (100 trials) -> `_n100`, dev (30) -> bare (unchanged convention);
#   2. a retrain inherits its SOURCE run's tier: from `<...>_n100/config.json` -> `_n100` (via
#      metadata.json, and via the folder name when metadata is missing); from a bare dev-tier
#      folder -> bare (the 2026-08 dev-tier cold retrains keep their names);
#   3. the success check in main() and the save step use the same function (no drift: before the
#      fix a production-profile retrain saved to the bare folder and was checked at `_n100`);
#   4. run_single_combo refuses to start when the target already holds best_model.pth, before
#      any training; --overwrite-results lifts it.
import json
import os
import sys
import tempfile

ROOT = tempfile.mkdtemp()
os.environ['PROTOMF_RESULTS_PATH'] = os.path.join(ROOT, 'results')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts')))

import run_combo  # noqa: E402

fails = []


def check(name, cond, evidence=''):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f' | {evidence}' if evidence else ''))
    if not cond:
        fails.append(name)


def name(**kw):
    return os.path.basename(run_combo.target_results_dir(**kw))


S = 38210573
check('hyperopt production -> _n100',
      name(model='mf', dataset='hm_1_month', seed=S, resource_cfg={'num_samples': 100}) == 'mf_hm_1_month_s38210573_n100')
check('hyperopt dev -> bare',
      name(model='mf', dataset='hm_1_month', seed=S, resource_cfg={'num_samples': 30}) == 'mf_hm_1_month_s38210573')

src_meta = os.path.join(ROOT, 'src', 'renamed_folder')
os.makedirs(src_meta)
json.dump({'num_samples': 100}, open(os.path.join(src_meta, 'metadata.json'), 'w'))
json.dump({}, open(os.path.join(src_meta, 'config.json'), 'w'))
src_name = os.path.join(ROOT, 'src', 'feature_item_proto_hm_1_month_s38210573_n100')
os.makedirs(src_name)
src_bare = os.path.join(ROOT, 'src', 'feature_item_proto_hm_1_month_s38210573')
os.makedirs(src_bare)
for kind, src, want in (('metadata', src_meta, 'feature_item_proto_hm_1_month_cold_s38210573_n100'),
                        ('folder name', src_name, 'feature_item_proto_hm_1_month_cold_s38210573_n100'),
                        ('bare source', src_bare, 'feature_item_proto_hm_1_month_cold_s38210573')):
    got = name(model='feature_item_proto', dataset='hm_1_month_cold', seed=S,
               resource_cfg={'num_samples': 100}, retrain_config=os.path.join(src, 'config.json'))
    check(f'retrain inherits the source tier ({kind})', got == want, got)

target = run_combo.target_results_dir('feature_item_proto', 'hm_1_month_cold', S, {'num_samples': 100},
                                      os.path.join(src_name, 'config.json'))
os.makedirs(target)
open(os.path.join(target, 'best_model.pth'), 'w').close()
try:
    run_combo.run_single_combo('feature_item_proto', 'hm_1_month_cold', S, resource_cfg={'num_samples': 100},
                               retrain_config=os.path.join(src_name, 'config.json'))
    check('guard refuses a finished folder', False, 'no exception')
except FileExistsError as e:
    check('guard refuses a finished folder', True, str(e)[:80])
try:
    run_combo.run_single_combo('feature_item_proto', 'hm_1_month_cold', S, resource_cfg={'num_samples': 100},
                               retrain_config=os.path.join(src_name, 'config.json'), overwrite_results=True)
except FileExistsError:
    check('--overwrite-results lifts the guard', False)
except Exception as e:  # past the guard: fails later on the empty config, as expected
    check('--overwrite-results lifts the guard', True, f'{type(e).__name__} after the guard')

print('\nALL PASS' if not fails else f'\nFAILURES: {fails}')
sys.exit(1 if fails else 0)
