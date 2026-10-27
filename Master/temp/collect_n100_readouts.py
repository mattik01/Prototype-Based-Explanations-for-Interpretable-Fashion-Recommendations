# Collect the 100-trial (`_n100`) rows into one table: identical-negatives readout HR@10 / N@10
# (the comparison regime) beside the summary test number (a different negative draw — never mixed),
# the winning trial's loss / d / K / batch, and the readout's item-popularity buckets.
# Run on LEO5 (plain python3, no torch):  python3 collect_n100_readouts.py [results_root]
import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/scratch/c7031336/protomf_results'
SKIP = ('feature_item_proto_c_', 'feature_user_proto_itemid_')  # dc06 / dc08 — out of the Nov 3 scope

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, '*_s38210573_n100'))):
    name = os.path.basename(d)
    if name.startswith(SKIP):
        continue
    dataset ='hm_1_month' if '_hm_1_month_' in name else 'ml-1m'
    model = name.replace(f'_{dataset}_s38210573_n100', '')
    r = {'model': model, 'dataset': dataset}
    try:
        ro = json.load(open(os.path.join(d, 'stratified_readout', 'readout.json')))
        r['ro_hr'], r['ro_n'] = ro['overall']['hit_ratio@10'], ro['overall']['ndcg@10']
        r['buckets'] = {b['bucket']: b['hit_ratio@10'] for b in ro['by_item_pop_bucket']}
    except FileNotFoundError:
        r['ro_hr'] = r['ro_n'] = None
        r['buckets'] = {}
    tm = json.load(open(os.path.join(d, 'test_metrics.json')))
    r['te_hr'], r['te_n'] = tm['hit_ratio@10'], tm['ndcg@10']
    c = json.load(open(os.path.join(d, 'config.json')))
    f = c.get('ft_ext_param', {})
    ks = [f.get(s, {}).get('n_prototypes') for s in ('user_ft_ext_param', 'item_ft_ext_param')]
    r['cfg'] = (f"{c.get('loss_func_name')}, d={f.get('embedding_dim')}, "
                f"K={'/'.join(str(k) for k in ks if k)}, B={c.get('batch_size')}")
    rows.append(r)

fmt = lambda x: '—' if x is None else f'{x:.4f}'
for ds in ('hm_1_month', 'ml-1m'):
    print(f'\n## {ds}\n')
    print('| model | readout HR@10 / N@10 | summary test HR@10 / N@10 | pop 1-5 / 6-20 / 21-100 / >100 (readout HR@10) | winner |')
    print('|---|---|---|---|---|')
    for r in sorted((r for r in rows if r['dataset'] == ds), key=lambda r: -(r['ro_hr'] or 0)):
        b = r['buckets']
        bs = ' / '.join(f"{b[k]:.3f}" if k in b else '—' for k in ('1-5', '6-20', '21-100', '101-inf'))
        print(f"| {r['model']} | {fmt(r['ro_hr'])} / {fmt(r['ro_n'])} | {fmt(r['te_hr'])} / {fmt(r['te_n'])} | {bs} | {r['cfg']} |")
