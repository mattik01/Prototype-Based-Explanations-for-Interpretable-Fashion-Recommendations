"""dc02 LOCAL smoke harness — ray-free end-to-end train/val/test on the laptop.

Why this exists: the repo's orchestrated entrypoints (start.py / run_combo.py) import
experiment_helper, which targets the Ray 2.x API used on LEO5 (`ray.tune.search.*`); the local
laptop env has Ray 1.6.0, so those entrypoints cannot run here (pre-existing env/codebase
mismatch — NOT a dc02 issue; dc01's end-to-end smoke ran on the cluster). This harness drives
the SAME repo components directly — ProtoRecDataset dataloaders, Trainer(use_ray=False),
Tester — and assembles the results dir the explanations pipeline and readout_demo consume:
best_model.pth / config.json / metadata.json / test_metrics.json (the subset run_combo produces
that the pipeline actually reads; val_metrics.json + hardware logs are run_combo extras not
needed here). Results are tagged `_localsmoke` — pipeline sanity only, never reportable.

Usage (repo root, conda env protomf):
  python Master/temp/dc_checks/dc02/smoke_train.py -m attr_item_proto_debug -d hm_3_month
  python Master/temp/dc_checks/dc02/smoke_train.py -m attr_item_proto_debug_knobs -d hm_3_month
"""
import argparse
import copy
import json
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from confs.hyper_params import (attr_item_proto_debug_hyper_params,
                                attr_item_proto_debug_knobs_hyper_params)
from rec_sys.protomf_dataset import get_protorecdataset_dataloader
from rec_sys.tester import Tester
from rec_sys.trainer import Trainer
from utilities.consts import DATA_PATH, NEG_VAL, SINGLE_SEED, EXPERIMENT_RESULTS_PATH
from utilities.utils import reproducible

CONFS = {
    'attr_item_proto_debug': attr_item_proto_debug_hyper_params,
    'attr_item_proto_debug_knobs': attr_item_proto_debug_knobs_hyper_params,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', '-m', choices=list(CONFS.keys()), required=True)
    ap.add_argument('--dataset', '-d', default='hm_3_month')
    ap.add_argument('--seed', '-s', type=int, default=SINGLE_SEED)
    ap.add_argument('--num-workers', type=int, default=2)
    args = ap.parse_args()

    conf = copy.deepcopy(CONFS[args.model])
    conf['data_path'] = os.path.join(DATA_PATH, args.dataset)
    conf['seed'] = args.seed
    ns = argparse.Namespace(**conf)

    results_dir = os.path.join(EXPERIMENT_RESULTS_PATH,
                               f'{args.model}_{args.dataset}_s{args.seed}_localsmoke')
    os.makedirs(results_dir, exist_ok=True)

    # torch 1.9.1 rejects prefetch_factor when num_workers == 0 (even prefetch_factor=None), so
    # only pass it in the multi-worker case.
    train_extra = {'prefetch_factor': 5} if args.num_workers > 0 else {}
    train_loader = get_protorecdataset_dataloader(
        data_path=ns.data_path, split_set='train', n_neg=ns.neg_train,
        neg_strategy=ns.train_neg_strategy, batch_size=ns.batch_size, shuffle=True,
        num_workers=args.num_workers, **train_extra)
    val_loader = get_protorecdataset_dataloader(
        data_path=ns.data_path, split_set='val', n_neg=NEG_VAL,
        neg_strategy=ns.eval_neg_strategy, batch_size=ns.val_batch_size,
        num_workers=args.num_workers)

    reproducible(args.seed)
    t0 = time.time()
    trainer = Trainer(train_loader, val_loader, ns, use_ray=False)
    trainer.run(checkpoint_dir=results_dir)
    wall_train = time.time() - t0

    ckpt = os.path.join(results_dir, 'best_model.pth')
    assert os.path.exists(ckpt), 'no best_model.pth saved — validation never improved?'

    # run_combo-compatible artifacts (config carries only the lightweight spec — no tensors)
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(conf, f, indent=2, default=str)
    # metadata['model'] MUST be the canonical model TYPE ('attr_item_proto'), not the CLI config
    # name ('attr_item_proto_debug[_knobs]'): the explanations pipeline gates on
    # metadata['model'] ∈ EXPLAINABLE_MODELS and get_accessor() resolves by that key, both of
    # which know only 'attr_item_proto'. The debug config name is preserved separately for
    # provenance. (config.json's ft_type is already 'attr_item_proto', so the rebuild is correct.)
    with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
        json.dump({'model': 'attr_item_proto', 'config_name': args.model,
                   'dataset': args.dataset, 'seed': args.seed,
                   'num_samples': 1, 'num_completed': 1, 'num_total': 1,
                   'run_name': os.path.basename(results_dir), 'profile': 'local-smoke',
                   'wall_seconds': round(wall_train, 1), 'machine': socket.gethostname(),
                   'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

    test_loader = get_protorecdataset_dataloader(
        data_path=ns.data_path, split_set='test', n_neg=NEG_VAL,
        neg_strategy=ns.eval_neg_strategy, batch_size=ns.val_batch_size,
        num_workers=args.num_workers)
    reproducible(args.seed)
    tester = Tester(test_loader, ns, ckpt)
    metrics = tester.test()
    with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    finite = all(math.isfinite(float(v)) for v in metrics.values())
    print(f'\n===== SMOKE {args.model} × {args.dataset} =====')
    print(f'results dir : {results_dir}')
    print(f'train wall  : {wall_train:.0f}s | test metrics finite: {finite}')
    print(f'hit_ratio@10: {float(metrics.get("hit_ratio@10", float("nan"))):.4f} | '
          f'ndcg@10: {float(metrics.get("ndcg@10", float("nan"))):.4f}')
    print(f'SMOKE {"PASS" if finite else "FAIL"}')
    sys.exit(0 if finite else 1)


if __name__ == '__main__':
    main()
