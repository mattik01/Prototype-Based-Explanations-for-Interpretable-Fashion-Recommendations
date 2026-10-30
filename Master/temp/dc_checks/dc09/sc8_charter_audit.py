"""dc09 SC.8.1/8.2 — charter-compliance audit + anomaly triage of the actual runs (2026-10-30).

Walks every dc09-scope run dir on LEO5 scratch: the 8 fUfI hyperopts, the 6 two-sided control
hyperopts (incl. the ml-1m `_sep` pair), and the cold retrains of all of them. Per run:

- C3/C4/C5/C8 (charter, as amended by the dossier header's 100-trial deviation): seed, profile
  epochs/patience, selection metric, 1 DataLoader worker (F-S0-02), uniform eval negatives, explicit
  `use_bias = 0`, canonical feature fields + layout per dataset, the arm's ID / sharing flags on both
  sides, user-side LOO pooling; trials completed (100 warm, 1 retrain); concurrency from summary.md.
- Retrains: config identical to the source hyperopt's apart from data_path.
- Anomaly triage from the Ray dirs: errored trials, NaN/inf in any progress.csv, the winner trial's
  epochs (matched to config.json by lr + embedding_dim), its best epoch, and whether the best epoch
  sits at the 100-epoch cap (a cut-off winner); retrain epochs vs the source winner's.

C1 (one commit) is not recorded per run; it is read from the fleet checkout's reflog (separately).
Stdlib only. Run on the LEO5 login node: python3 sc8_charter_audit.py > audit.md
"""
import csv
import glob
import json
import math
import os

R = '/scratch/c7031336/protomf_results'
RAY = '/scratch/c7031336/ray_results'
S = '_s38210573_n100'
ARMS = {  # model -> (use_id_feature, share_word_table)
    'feature_user_item_proto': (True, True), 'feature_user_item_proto_noid': (False, True),
    'feature_user_item_proto_sep': (True, False), 'feature_user_item_proto_noid_sep': (False, False),
    'lightfm_tags_hist_ids': (True, True), 'lightfm_tags_hist': (False, True),
    'lightfm_tags_hist_ids_sep': (True, False), 'lightfm_tags_hist_sep': (False, False),
}
FIELDS = {'hm_1_month': (['department_name', 'product_type_name', 'section_name', 'colour_group_name',
                          'graphical_appearance_name'], 'fixed'),
          'ml-1m': (['genres', 'tags'], 'bags')}


def ray_trials(run_name):
    out = []
    for p in glob.glob(os.path.join(RAY, run_name, '*', 'progress.csv')):
        rows = list(csv.DictReader(open(p)))
        tdir = os.path.dirname(p)
        params = json.load(open(os.path.join(tdir, 'params.json'))) if os.path.exists(os.path.join(tdir, 'params.json')) else {}
        bad = any(not math.isfinite(float(v)) for r in rows for k, v in r.items()
                  if k in ('hit_ratio@10', 'ndcg@10', 'train_loss', 'val_loss') and v not in ('', None))
        hr = [float(r['hit_ratio@10']) for r in rows if r.get('hit_ratio@10') not in ('', None)]
        out.append(dict(rows=len(rows), bad=bad, hr=hr, params=params))
    err = len(glob.glob(os.path.join(RAY, run_name, '*', 'error.txt')))
    return out, err


def winner(trials, conf):
    lr = conf['optim_param']['lr']; d = conf['ft_ext_param'].get('embedding_dim')
    for t in trials:
        p = t['params']
        if p and abs(p.get('optim_param', {}).get('lr', -1) - lr) < 1e-12 \
                and p.get('ft_ext_param', {}).get('embedding_dim') == d:
            return t
    return None


def check(model, ds, cold):
    tag = f'{ds}_cold' if cold else ds
    d = os.path.join(R, f'{model}_{tag}{S}')
    if not os.path.isdir(d):
        return None
    conf = json.load(open(os.path.join(d, 'config.json')))
    meta = json.load(open(os.path.join(d, 'metadata.json')))
    issues = []
    want_n = 1 if cold else 100
    if meta.get('num_completed') != want_n:
        issues.append(f"num_completed {meta.get('num_completed')} != {want_n}")
    for k, v in (('seed', 38210573), ('_num_workers', 1), ('eval_neg_strategy', 'uniform'),
                 ('n_epochs', 100), ('_max_patience', 10), ('_optimizing_metric', 'hit_ratio@10')):
        if conf.get(k) != v:
            issues.append(f'{k}={conf.get(k)!r}')
    if conf.get('rec_sys_param', {}).get('use_bias', 'ABSENT') != 0:
        issues.append(f"use_bias={conf.get('rec_sys_param', {}).get('use_bias', 'ABSENT')}")
    fe = conf['ft_ext_param']
    ids, share = ARMS[model]
    if fe.get('share_word_table') is not share:
        issues.append(f"share_word_table={fe.get('share_word_table')}")
    fields, layout = FIELDS[ds]
    for side in ('user_ft_ext_param', 'item_ft_ext_param'):
        s = fe[side]
        if s.get('use_id_feature') is not ids:
            issues.append(f'{side}.use_id_feature={s.get("use_id_feature")}')
        if list(s.get('feature_fields', [])) != fields or s.get('feature_layout') != layout:
            issues.append(f'{side} fields/layout {s.get("feature_fields")}/{s.get("feature_layout")}')
    if fe['user_ft_ext_param'].get('loo_pooling') is not True:
        issues.append('user loo_pooling not True')
    if not conf['data_path'].endswith('/' + tag):
        issues.append(f"data_path {conf['data_path']}")
    if cold:
        src = json.load(open(os.path.join(R, f'{model}_{ds}{S}', 'config.json')))
        a = {k: v for k, v in conf.items() if k != 'data_path'}
        b = {k: v for k, v in src.items() if k != 'data_path'}
        if a != b:
            issues.append('config != source config (beyond data_path)')
    conc = '?'
    sm = os.path.join(d, 'summary.md')
    if os.path.exists(sm):
        for line in open(sm):
            if line.startswith('- **Concurrency:**'):
                conc = line.split('**')[2].strip()
    trials, err = ray_trials(meta['run_name'])
    nan = sum(t['bad'] for t in trials)
    w = winner(trials, conf)
    if w is None:
        wtxt = 'winner not matched'
        issues.append('winner not matched')
    else:
        best = max(range(len(w['hr'])), key=lambda i: w['hr'][i]) + 1 if w['hr'] else 0
        wtxt = f"{w['rows']} ep, best @{best}" + (' **AT CAP**' if best >= 100 else '')
    if err:
        issues.append(f'{err} errored trial(s)')
    if nan:
        issues.append(f'{nan} trial(s) with NaN/inf')
    return dict(run=f'{model} × {tag}', n=meta.get('num_completed'), trials=len(trials), conc=conc,
                win=wtxt, issues=issues, w=w)


print('# dc09 SC.8 — charter audit + anomaly triage (generator: Master/temp/dc_checks/dc09/sc8_charter_audit.py)\n')
print('| run | completed | Ray trials | concurrency | winner (epochs, best epoch) | issues |')
print('|---|---|---|---|---|---|')
src_epochs = {}
for cold in (False, True):
    for ds in ('hm_1_month', 'ml-1m'):
        for m in ARMS:
            r = check(m, ds, cold)
            if r is None:
                continue
            note = ''
            if cold and (m, ds) in src_epochs and r['w']:
                note = f" (source {src_epochs[(m, ds)]} ep → ×{r['w']['rows'] / src_epochs[(m, ds)]:.2f})"
            if not cold and r['w']:
                src_epochs[(m, ds)] = r['w']['rows']
            print(f"| {r['run']} | {r['n']} | {r['trials']} | {r['conc']} | {r['win']}{note} | "
                  f"{'; '.join(r['issues']) or 'none'} |")
