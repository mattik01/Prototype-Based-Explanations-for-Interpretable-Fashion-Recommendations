"""D3 popularity-negatives secondary readout (Scrutiny S0.1 D3) — one test-only pass of an
EXISTING canonical run with eval negatives drawn popularity-proportionally (pop^0.75,
`_neg_sample_popular`) instead of the headline uniform draw. Zero training cost; addresses the
popularity-blindness caveat of uniform-99 sampled evaluation. Secondary table only — the
headline eval stays uniform-99 (S0.1 D2).

Output (into <results_dir>/popneg_readout/): readout.json + readout.md with the full metric set
under popular negatives, next to the stored uniform-99 test metrics for delta reading.

Comparability note: `evaluate_rows` iterates a num_workers=0 DataLoader after `reproducible(seed)`,
so the popular-negative draws are identical across model rows evaluated with the same seed
(the t05-pinned sequential-access convention).

Usage:
    python Master/scripts/popneg_readout.py --results-dir /scratch/.../mf_hm_1_month_s38210573
    (dataset dir resolved from config.json's data_path basename against DATA_PATH;
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

from rec_sys.protomf_dataset import ProtoRecDataset
from utilities.cold_eval import build_model, load_checkpoint, evaluate_rows
from utilities.consts import DATA_PATH, NEG_VAL, SINGLE_SEED
from utilities.utils import reproducible


def run_readout(results_dir: str, dataset: str = None, device: str = 'cpu',
                batch_size: int = 256, n_neg: int = NEG_VAL, seed: int = SINGLE_SEED) -> dict:
    with open(os.path.join(results_dir, 'config.json')) as f:
        conf = json.load(f)
    dataset = dataset or os.path.basename(conf['data_path'].rstrip('/'))
    data_path = os.path.join(DATA_PATH, dataset)

    model, _, _ = build_model(conf, data_path, device)
    load_checkpoint(model, results_dir, device)

    reproducible(seed)
    ds = ProtoRecDataset(data_path, 'test', n_neg, 'popular')
    agg, _per_row, _ties = evaluate_rows(model, ds, batch_size, device)

    uniform = {}
    tm_path = os.path.join(results_dir, 'test_metrics.json')
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            uniform = json.load(f)

    out = {
        'results_dir': results_dir,
        'dataset': dataset,
        'seed': seed,
        'n_neg': n_neg,
        'eval_neg_strategy': 'popular',
        'metrics_popular': agg,
        'metrics_uniform_stored': uniform,
        'delta_popular_minus_uniform': {k: agg[k] - uniform[k]
                                        for k in agg if k in uniform},
    }

    out_dir = os.path.join(results_dir, 'popneg_readout')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'readout.json'), 'w') as f:
        json.dump(out, f, indent=2)

    keys = sorted(agg, key=lambda k: (k.split('@')[0], int(k.split('@')[1])))
    lines = [f'# D3 popularity-negatives readout — {os.path.basename(results_dir)}', '',
             f'dataset `{dataset}` | seed {seed} | {n_neg} popular negatives (pop^0.75)', '',
             '| metric | popular | uniform (stored) | Δ pop−uni |', '|---|---|---|---|']
    for k in keys:
        u = uniform.get(k)
        lines.append(f'| {k} | {agg[k]:.4f} | ' +
                     (f'{u:.4f} | {agg[k] - u:+.4f} |' if u is not None else '— | — |'))
    with open(os.path.join(out_dir, 'readout.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'  D3 readout saved to: {out_dir}')
    for k in ('hit_ratio@10', 'ndcg@10'):
        u = uniform.get(k)
        print(f'    {k}: popular {agg[k]:.4f}' +
              (f' vs uniform {u:.4f} (Δ {agg[k] - u:+.4f})' if u is not None else ''))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=SINGLE_SEED)
    args = parser.parse_args()
    run_readout(args.results_dir, dataset=args.dataset, device=args.device,
                batch_size=args.batch_size, seed=args.seed)
