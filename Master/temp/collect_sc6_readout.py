# Warm 100-trial table incl. the SC.6 rows + the pre-registered 2x2 (dc09 dossier SC.6 §4).
import glob, json, os
ROOT = '/scratch/c7031336/protomf_results'
SKIP = ('feature_item_proto_c_', 'feature_user_proto_itemid_')
rows = {}
for d in sorted(glob.glob(os.path.join(ROOT, '*_s38210573_n100'))):
    name = os.path.basename(d)
    if name.startswith(SKIP) or '_cold_' in name:
        continue
    ds = 'hm_1_month' if '_hm_1_month_' in name else 'ml-1m'
    model = name.replace(f'_{ds}_s38210573_n100', '')
    try:
        ro = json.load(open(os.path.join(d, 'stratified_readout', 'readout.json')))
    except FileNotFoundError:
        continue
    rows[(model, ds)] = ro
fmt = lambda x: f'{x:.4f}'
for ds in ('hm_1_month', 'ml-1m'):
    print(f'\n## {ds}')
    print('| model | HR@10 | N@10 | pop 1-5 / 6-20 / 21-100 / >100 | user-hist Q1..Q4 HR@10 |')
    print('|---|---|---|---|---|')
    sel = sorted(((m, r) for (m, d), r in rows.items() if d == ds), key=lambda t: -t[1]['overall']['hit_ratio@10'])
    for m, r in sel:
        b = {e['bucket']: e['hit_ratio@10'] for e in r['by_item_pop_bucket']}
        bs = ' / '.join(f"{b[k]:.3f}" if k in b else '-' for k in ('1-5', '6-20', '21-100', '101-inf'))
        qs = ' / '.join(f"{e['hit_ratio@10']:.3f}" for e in r['by_user_history_quartile'])
        print(f"| {m} | {fmt(r['overall']['hit_ratio@10'])} | {fmt(r['overall']['ndcg@10'])} | {bs} | {qs} |")
    g = lambda m, k='hit_ratio@10': rows[(m, ds)]['overall'][k]
    for k in ('hit_ratio@10', 'ndcg@10'):
        print(f'\n2x2 ({k}) ids pairing: fUfI <-> lightfm_tags_hist_ids <-> mf, host UI')
        f, c, mf, ui = g('feature_user_item_proto', k), g('lightfm_tags_hist_ids', k), g('mf', k), g('user_item_proto', k)
        print(f'  fUfI-UI {f-ui:+.4f} = head|comp {f-c:+.4f} + comp|dot {c-mf:+.4f} - head|free {ui-mf:+.4f}; interaction {(f-c)-(ui-mf):+.4f}')
        fn, cn = g('feature_user_item_proto_noid', k), g('lightfm_tags_hist', k)
        print(f'  noid: head|comp {fn-cn:+.4f}  (fUfI-noid {fn:.4f}, control {cn:.4f}); noid-UI {fn-ui:+.4f}')
        for s, base in (('feature_user_item_proto_sep', 'feature_user_item_proto'), ('feature_user_item_proto_noid_sep', 'feature_user_item_proto_noid')):
            print(f'  sharing: {s} - {base} = {g(s, k)-g(base, k):+.4f}')
    # `_sep` controls (ml-1m only, warm readout 2026-10-30): head term for the `_sep` arms, paired by ID setting,
    # and the sharing effect on the control side.
    if ('lightfm_tags_hist_ids_sep', ds) in rows:
        for k in ('hit_ratio@10', 'ndcg@10'):
            print(f'\n_sep pairing ({k}):')
            for arm, ctl in (('feature_user_item_proto_sep', 'lightfm_tags_hist_ids_sep'),
                             ('feature_user_item_proto_noid_sep', 'lightfm_tags_hist_sep')):
                print(f'  head|comp_sep: {arm} - {ctl} = {g(arm, k)-g(ctl, k):+.4f}  ({g(arm, k):.4f} vs {g(ctl, k):.4f})')
            for ctl, base in (('lightfm_tags_hist_ids_sep', 'lightfm_tags_hist_ids'), ('lightfm_tags_hist_sep', 'lightfm_tags_hist')):
                print(f'  control sharing: {ctl} - {base} = {g(ctl, k)-g(base, k):+.4f}')
