"""D4 stratified readout (Scrutiny S0.3 §7) — per-row test metrics of an EXISTING canonical run,
sliced by user train-history length (facet-B warm-thin claim) and by positive-item train
popularity (stratum TW / ProtoCF-style tail readout). Zero training cost: one test pass on the
saved checkpoint.

Output (into <results_dir>/stratified_readout/): readout.json + readout.md with
HR@10 / NDCG@10 by user-history quartile and by item-popularity bucket (1-5, 6-20, 21-100, >100;
bucket 0 = the F-S0-04 train-unseen slice, reported separately when present).

Usage:
    python Master/scripts/stratified_readout.py --results-dir Master/experiments/results/mf_hm_1_month_s38210573
    (dataset dir resolved from config.json's data_path basename against the local DATA_PATH;
     override with --dataset)
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils import data

from rec_sys.protomf_dataset import ProtoRecDataset
from utilities.cold_eval import build_model, load_checkpoint, evaluate_rows
from utilities.consts import DATA_PATH, NEG_VAL, SINGLE_SEED
from utilities.utils import reproducible

ITEM_BUCKETS = [(1, 5), (6, 20), (21, 100), (101, np.inf)]
METRICS = ('hit_ratio@10', 'ndcg@10')


def _bucket_label(lo, hi):
    return f'{lo}-{hi if np.isfinite(hi) else "inf"}'


def run_readout(results_dir: str, dataset: str = None, device: str = 'cpu',
                batch_size: int = 256, n_neg: int = NEG_VAL, seed: int = SINGLE_SEED) -> dict:
    with open(os.path.join(results_dir, 'config.json')) as f:
        conf = json.load(f)
    dataset = dataset or os.path.basename(conf['data_path'].rstrip('/'))
    data_path = os.path.join(DATA_PATH, dataset)

    model, _, _ = build_model(conf, data_path, device)
    load_checkpoint(model, results_dir, device)

    reproducible(seed)
    ds = ProtoRecDataset(data_path, 'test', n_neg, 'uniform')
    agg, per_row, _ties = evaluate_rows(model, ds, batch_size, device)

    # row order == COO order (sorted by user, item)
    row_users, row_items = ds.coo_matrix.row, ds.coo_matrix.col

    train = pd.read_csv(os.path.join(data_path, 'listening_history_train.csv'),
                        usecols=['user_id', 'item_id'])
    hist = train.groupby('user_id').size()
    user_hist = np.zeros(ds.n_users, dtype=int)
    user_hist[hist.index.to_numpy()] = hist.to_numpy()
    pop = train.groupby('item_id').size()
    item_pop = np.zeros(ds.n_items, dtype=int)
    item_pop[pop.index.to_numpy()] = pop.to_numpy()

    report = {'results_dir': os.path.abspath(results_dir), 'dataset': dataset,
              'n_rows': int(len(row_users)), 'n_neg': n_neg, 'seed': seed,
              'overall': {k: agg[k] for k in METRICS}}

    # --- user-history quartiles (over evaluated rows; one row per user on canonical splits) ---
    row_hist = user_hist[row_users]
    qs = np.quantile(row_hist, [0.25, 0.5, 0.75])
    edges = [-np.inf, *qs, np.inf]
    report['user_history_quartile_edges'] = [float(q) for q in qs]
    report['by_user_history_quartile'] = []
    for qi in range(4):
        m = (row_hist > edges[qi]) & (row_hist <= edges[qi + 1])
        entry = {'quartile': qi + 1, 'n_rows': int(m.sum()),
                 'hist_range': [int(row_hist[m].min()), int(row_hist[m].max())] if m.any() else None}
        entry.update({k: float(per_row[k][m].mean()) for k in METRICS} if m.any() else {})
        report['by_user_history_quartile'].append(entry)

    # --- item-popularity buckets (stratum TW; pop 0 = F-S0-04 train-unseen slice) ---
    row_pop = item_pop[row_items]
    report['by_item_pop_bucket'] = []
    zero = row_pop == 0
    if zero.any():
        report['by_item_pop_bucket'].append(
            {'bucket': '0 (train-unseen, F-S0-04)', 'n_rows': int(zero.sum()),
             **{k: float(per_row[k][zero].mean()) for k in METRICS}})
    for lo, hi in ITEM_BUCKETS:
        m = (row_pop >= lo) & (row_pop <= hi)
        if not m.any():
            continue
        report['by_item_pop_bucket'].append(
            {'bucket': _bucket_label(lo, hi), 'n_rows': int(m.sum()),
             **{k: float(per_row[k][m].mean()) for k in METRICS}})

    out_dir = os.path.join(results_dir, 'stratified_readout')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'readout.json'), 'w') as f:
        json.dump(report, f, indent=2)

    lines = [f"# Stratified readout — {os.path.basename(results_dir)}", '',
             f"dataset: {dataset} | rows: {report['n_rows']} | "
             f"overall HR@10 {report['overall']['hit_ratio@10']:.4f} / "
             f"NDCG@10 {report['overall']['ndcg@10']:.4f}", '',
             '## By user train-history quartile (D4, facet B warm-thin)', '',
             '| quartile | history range | rows | HR@10 | NDCG@10 |', '|---|---|---:|---:|---:|']
    for e in report['by_user_history_quartile']:
        lines.append(f"| Q{e['quartile']} | {e['hist_range']} | {e['n_rows']} | "
                     f"{e.get('hit_ratio@10', float('nan')):.4f} | {e.get('ndcg@10', float('nan')):.4f} |")
    lines += ['', '## By positive-item train popularity (stratum TW)', '',
              '| bucket | rows | HR@10 | NDCG@10 |', '|---|---:|---:|---:|']
    for e in report['by_item_pop_bucket']:
        lines.append(f"| {e['bucket']} | {e['n_rows']} | {e['hit_ratio@10']:.4f} | {e['ndcg@10']:.4f} |")
    with open(os.path.join(out_dir, 'readout.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'stratified readout written to {out_dir}')
    return report


def main():
    ap = argparse.ArgumentParser(description='D4 stratified test readout (S0.3 §7)')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--dataset', default=None,
                    help='dataset dir name (default: basename of config.json data_path)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n-neg', type=int, default=NEG_VAL)
    ap.add_argument('--seed', type=int, default=SINGLE_SEED)
    args = ap.parse_args()
    run_readout(args.results_dir, args.dataset, args.device, args.batch_size, args.n_neg, args.seed)


if __name__ == '__main__':
    main()
