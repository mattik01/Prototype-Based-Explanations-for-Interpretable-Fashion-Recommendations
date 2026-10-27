"""Collector for the 100-trial cold rows (SC.6 wave group C, cold evals 2026-10-27).

Reads <model>_<ds>_cold_s38210573_n100/cold_eval/report.json (generator: Master/temp/run_cold_evals.sh
`<model> <ds> _n100`) and prints one markdown table per dataset. Quoting rules applied mechanically:
- F-S0-14: a cold number is a tie bracket @10 (worst / expected / best); quoted as a point only when
  the bracket is tight (best − worst ≤ 0.002), otherwise as "expected [worst, best]";
- F-S0-15: a degenerate no-signal block (bracket spanning ≥ 0.9) is quoted as "chance", never 0.00;
- F-S0-17: the popularity row on cold-vs-cold is chance by construction (one per-item tie-break draw) —
  quoted as 0.10.
The dev-tier hm cold rows (2026-07/08, folders without `_n100`) are printed beside the hm table for
reference only: a different tier, never a comparison row (two-tier numbers policy).
Run on the LEO5 login node: python3 collect_cold_n100.py > cold_readout.md  (stdlib only)
"""
import json
import os

ROOT = '/scratch/c7031336/protomf_results'
MODELS = ['mf', 'user_proto', 'item_proto', 'user_item_proto', 'feature_item_proto',
          'feature_item_proto_noid', 'feature_user_proto', 'feature_user_proto_noid',
          'lightfm_tags', 'lightfm_tags_ids']


def q(block, metric='hit_ratio'):
    if not block or 'tie_bracket@10' not in block:
        return '—'
    tb = block['tie_bracket@10'][metric]
    w, e, b = tb['worst'], tb['expected'], tb['best']
    if b - w >= 0.9:
        return 'chance'
    if b - w <= 0.002:
        return f'{e:.4f}'
    return f'{e:.4f} [{w:.4f}, {b:.4f}]'


def load(path):
    p = os.path.join(path, 'cold_eval', 'report.json')
    return json.load(open(p)) if os.path.exists(p) else None


def table(ds):
    rows = [f'## {ds}', '',
            '| model | warm ref (n100 test) | warm on variant | native cold-vs-cold | native cold-vs-all | '
            'attr-kNN cold-vs-cold | attr-kNN cold-vs-all | cold-vs-cold N@10 (native / kNN) |',
            '|---|---|---|---|---|---|---|---|']
    pop = None
    for m in MODELS:
        r = load(os.path.join(ROOT, f'{m}_{ds}_cold_s38210573_n100'))
        if r is None:
            rows.append(f'| {m} | (missing) | | | | | | |')
            continue
        knn = r.get('attr_knn') or {}
        ref = r.get('canonical_warm_reference') or {}
        rows.append(f"| {m} | {ref.get('hit_ratio@10', float('nan')):.4f} | {r['warm_test']['hit_ratio@10']:.4f} | "
                    f"{q(r['cold_vs_cold'])} | {q(r['cold_vs_all'])} | {q(knn.get('cold_vs_cold'))} | "
                    f"{q(knn.get('cold_vs_all'))} | {q(r['cold_vs_cold'], 'ndcg')} / {q(knn.get('cold_vs_cold'), 'ndcg')} |")
        pop = pop or r.get('popularity_reference')
    if pop:
        rows.append(f"| popularity | | | 0.10 (chance by construction, F-S0-17) | {q(pop['cold_vs_all'])} | | | |")
    rows.append('')
    if ds == 'hm_1_month':
        rows += ['**Dev-tier hm cold rows (2026-07/08; reference only, other tier):**', '',
                 '| model | native cold-vs-cold | native cold-vs-all | attr-kNN cold-vs-cold | attr-kNN cold-vs-all | convention |',
                 '|---|---|---|---|---|---|']
        for m in MODELS:
            r = load(os.path.join(ROOT, f'{m}_{ds}_cold_s38210573'))
            if r is None:
                continue
            knn = r.get('attr_knn') or {}
            rows.append(f"| {m} | {q(r['cold_vs_cold'])} | {q(r['cold_vs_all'])} | {q(knn.get('cold_vs_cold'))} | "
                        f"{q(knn.get('cold_vs_all'))} | {r.get('cold_id_convention', 'drop (pre-2026-10-26)')} |")
        rows.append('')
    return rows


out = ['# 100-trial cold rows — collected', '',
       'Single seed 38210573; single-config cold retrains of the 100-trial winners (S0.3 §6); 99 sampled '
       'negatives; mean-ID imputation for cold ID rows. Quoting: F-S0-14 / -15 / -17 (see the collector docstring).', '']
for ds in ('hm_1_month', 'ml-1m'):
    out += table(ds)
print('\n'.join(out))
