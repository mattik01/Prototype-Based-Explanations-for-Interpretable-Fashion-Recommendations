"""
Regularizer sensitivity sweep: rerun a hyperopt-optimized user_item_proto config
with the prototype regularizer weights (sim_proto_weight, sim_batch_weight, both
user and item side) jointly scaled by a set of multipliers — no hyperopt, no Ray.

Motivation (protocol 2026-06-11_1144): the two regularizers are interpretability
penalties, yet their weights were selected by hyperopt against hit_ratio@10. This
sweep measures how accuracy AND explanation quality respond when they are turned
far down / far up around the accuracy-chosen operating point.

Each variant trains once with the baseline's exact hyperparameters and budget
(n_epochs / patience from the baseline config), evaluates on val + test, and runs
the full explanations pipeline. Results land in one folder per variant under
PROTOMF_REGSWEEP_PATH (default: Master/experiments/reg_sensitivity/).

Usage:
    python Master/scripts/run_reg_sweep.py -d ml-1m
    python Master/scripts/run_reg_sweep.py -d hm_1_month
    python Master/scripts/run_reg_sweep.py -d ml-1m --multipliers 0 0.1 10
    python Master/scripts/run_reg_sweep.py -d ml-1m --smoke   # 2-epoch pipeline check
"""

import argparse
import copy
import csv
import json
import os
import platform
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('WANDB_START_METHOD', 'thread')

# Windows consoles default to cp1252, which chokes on unicode in upstream prints
# (e.g. the 'Δ' in trainer.py). Harmless no-op on Linux/UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

import torch

# NOTE: deliberately NOT importing experiment_helper — it imports Ray Tune at module
# level (new-API path, breaks on the laptop's Ray 1.6). This sweep is Ray-free.
from rec_sys.protomf_dataset import get_protorecdataset_dataloader
from rec_sys.tester import Tester
from rec_sys.trainer import Trainer
from utilities.consts import NEG_VAL, NUM_WORKERS, PROJECT_NAME, SINGLE_SEED, \
    WANDB_API_KEY, DATA_PATH
from utilities.utils import reproducible, safe_wandb_init, safe_wandb_finish, safe_wandb_login

MODEL = 'user_item_proto'
DEFAULT_MULTIPLIERS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
REGSWEEP_ROOT = os.environ.get(
    'PROTOMF_REGSWEEP_PATH',
    os.path.join(REPO_ROOT, 'Master', 'experiments', 'reg_sensitivity'))


def variant_name(mult):
    if mult == 0:
        return 'off'
    return f"x{mult:g}"


def load_baseline_config(path):
    with open(path) as f:
        config = json.load(f)
    return {k: v for k, v in config.items() if not k.startswith('_wandb_')}


def scale_reg_weights(config, mult):
    """Return a copy of config with all four reg weights scaled by mult."""
    conf = copy.deepcopy(config)
    for side in ('user_ft_ext_param', 'item_ft_ext_param'):
        ft = conf['ft_ext_param'][side]
        ft['sim_proto_weight'] = ft['sim_proto_weight'] * mult
        ft['sim_batch_weight'] = ft['sim_batch_weight'] * mult
    return conf


def flat_reg_weights(config):
    user_ft = config['ft_ext_param']['user_ft_ext_param']
    item_ft = config['ft_ext_param']['item_ft_ext_param']
    return {
        'user_sim_proto_weight': user_ft['sim_proto_weight'],
        'user_sim_batch_weight': user_ft['sim_batch_weight'],
        'item_sim_proto_weight': item_ft['sim_proto_weight'],
        'item_sim_batch_weight': item_ft['sim_batch_weight'],
    }


def _make_loader(config, split_set, num_workers):
    is_train = split_set == 'train'
    kwargs = dict(
        data_path=config['data_path'],
        split_set=split_set,
        n_neg=config['neg_train'] if is_train else NEG_VAL,
        neg_strategy=config['train_neg_strategy'] if is_train else config['eval_neg_strategy'],
        batch_size=config['batch_size'] if is_train else config['val_batch_size'],
        shuffle=is_train,
        num_workers=num_workers,
    )
    # torch 1.9 rejects prefetch_factor (even None) when num_workers == 0
    if is_train and num_workers > 0:
        kwargs['prefetch_factor'] = 5
    return get_protorecdataset_dataloader(**kwargs)


def run_variant(config, dataset, mult, run_dir, num_workers, wandb_tags):
    """Train + test one fixed config. Returns (val_metrics, test_metrics, wall_sec)."""
    os.makedirs(run_dir, exist_ok=True)
    seed = config['seed']
    name = variant_name(mult)

    safe_wandb_login()
    safe_wandb_init(
        project=PROJECT_NAME,
        group=f'regsweep_{MODEL}_{dataset}',
        name=f'{MODEL}_{dataset}_s{seed}_reg_{name}',
        job_type='train/val',
        tags=[MODEL, dataset, f'seed:{seed}', 'reg_sweep', f'reg:{name}'] + wandb_tags,
        config={**{k: v for k, v in config.items() if not k.startswith('_')},
                'reg_multiplier': mult, **flat_reg_weights(config)},
        force=True,
        reinit=True,
    )

    wall_start = time.monotonic()
    try:
        conf_ns = argparse.Namespace(**config)
        train_loader = _make_loader(config, 'train', num_workers)
        val_loader = _make_loader(config, 'val', num_workers)
        reproducible(seed)
        trainer = Trainer(train_loader, val_loader, conf_ns, use_ray=False)
        trainer.run(checkpoint_dir=run_dir)

        ckpt_path = os.path.join(run_dir, 'best_model.pth')
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"no checkpoint written for variant {name} "
                               f"(training never improved over init?)")

        # Val metrics at the best checkpoint (run() keeps the last, not best, weights)
        try:
            state = torch.load(ckpt_path, map_location=config['device'], weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=config['device'])
        trainer.model.module.load_state_dict(state)
        val_metrics = trainer.val()
    finally:
        safe_wandb_finish()

    test_loader = _make_loader(config, 'test', num_workers)
    reproducible(seed)
    tester = Tester(test_loader, argparse.Namespace(**config), ckpt_path)
    test_metrics = tester.test()
    wall_sec = time.monotonic() - wall_start
    return val_metrics, test_metrics, wall_sec


def save_variant_results(run_dir, config, dataset, mult, val_metrics, test_metrics,
                         wall_sec, profile):
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    with open(os.path.join(run_dir, 'val_metrics.json'), 'w') as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    meta = {
        'model': MODEL,
        'dataset': dataset,
        'seed': config['seed'],
        'experiment': 'reg_sensitivity',
        'reg_multiplier': mult,
        'variant': variant_name(mult),
        **flat_reg_weights(config),
        'profile': profile,
        'wall_seconds': wall_sec,
        'machine': platform.node(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(run_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def write_sweep_summary(out_root, dataset, seed, rows, profile):
    csv_path = os.path.join(out_root, f'sweep_results_{dataset}_s{seed}.csv')
    fieldnames = ['dataset', 'variant', 'reg_multiplier',
                  'user_sim_proto_weight', 'user_sim_batch_weight',
                  'item_sim_proto_weight', 'item_sim_batch_weight',
                  'val_hit_ratio@10', 'val_ndcg@10',
                  'test_hit_ratio@10', 'test_ndcg@10',
                  'wall_seconds', 'profile', 'status']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSweep summary written to: {csv_path}")

    md_path = os.path.join(out_root, f'sweep_summary_{dataset}_s{seed}.md')
    lines = [
        f"# Regularizer sensitivity sweep: {MODEL} x {dataset} (seed={seed}, profile={profile})",
        "",
        "All hyperparameters fixed at the hyperopt-selected baseline; only the four",
        "prototype regularizer weights are jointly scaled by `reg_multiplier`.",
        "",
        "| Variant | Multiplier | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 | Wall | Status |",
        "|---------|:----------:|:---------:|:-----------:|:----------:|:------------:|:----:|:------:|",
    ]
    for r in rows:
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else (v if v != '' else '—')
        wall = f"{int(r['wall_seconds'] // 60)}m" if r['wall_seconds'] != '' else '—'
        lines.append(
            f"| {r['variant']} | {r['reg_multiplier']:g} | {_fmt(r['val_hit_ratio@10'])} | "
            f"{_fmt(r['val_ndcg@10'])} | {_fmt(r['test_hit_ratio@10'])} | "
            f"{_fmt(r['test_ndcg@10'])} | {wall} | {r['status']} |")
    lines.append("")
    with open(md_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Sweep summary written to: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fixed-config regularizer sensitivity sweep for user_item_proto')
    parser.add_argument('--dataset', '-d', required=True,
                        choices=['ml-1m', 'hm_1_month', 'hm_3_month', 'hm_full',
                                 'amazon2014', 'lfm2b-1mon'])
    parser.add_argument('--seed', '-s', type=int, default=SINGLE_SEED)
    parser.add_argument('--baseline-config', type=str, default=None,
                        help='Path to the optimized baseline config.json '
                             '(default: Master/experiments/results/'
                             f'{MODEL}_<dataset>_s<seed>/config.json)')
    parser.add_argument('--multipliers', type=float, nargs='+',
                        default=DEFAULT_MULTIPLIERS,
                        help=f'Joint scale factors for all four reg weights '
                             f'(default: {DEFAULT_MULTIPLIERS})')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help=f'DataLoader workers (default: {NUM_WORKERS})')
    parser.add_argument('--out-root', type=str, default=REGSWEEP_ROOT,
                        help=f'Results root (default: {REGSWEEP_ROOT})')
    parser.add_argument('--wandb-mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--skip-explanations', action='store_true', default=False)
    parser.add_argument('--smoke', action='store_true', default=False,
                        help='2-epoch pipeline sanity check (NOT for reporting)')
    args = parser.parse_args()

    os.environ['WANDB_MODE'] = args.wandb_mode
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY

    baseline_path = args.baseline_config or os.path.join(
        REPO_ROOT, 'Master', 'experiments', 'results',
        f'{MODEL}_{args.dataset}_s{args.seed}', 'config.json')
    if not os.path.exists(baseline_path):
        sys.exit(f"Baseline config not found: {baseline_path}")

    base_config = load_baseline_config(baseline_path)
    base_config['data_path'] = os.path.join(DATA_PATH, args.dataset)
    base_config['seed'] = args.seed
    base_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_config['_num_workers'] = args.num_workers

    # The variants inherit the baseline's training budget so the comparison is
    # like-for-like with the run the weights were optimized in.
    profile = f"fixed-rerun (n_epochs={base_config['n_epochs']}, " \
              f"patience={base_config.get('_max_patience')})"
    wandb_tags = []
    if args.smoke:
        base_config['n_epochs'] = 2
        base_config['_max_patience'] = 2
        profile = 'smoke'
        wandb_tags = ['smoke']

    print(f"{'=' * 60}\n  reg sweep: {MODEL} x {args.dataset} (seed={args.seed})\n"
          f"  baseline:    {baseline_path}\n"
          f"  multipliers: {args.multipliers}\n"
          f"  out root:    {args.out_root}\n"
          f"  device:      {base_config['device']}\n"
          f"  profile:     {profile}\n{'=' * 60}")

    os.makedirs(args.out_root, exist_ok=True)
    status_file = os.environ.get('RUN_STATUS_FILE')
    rows, failures = [], []

    for mult in args.multipliers:
        name = variant_name(mult)
        run_dir = os.path.join(
            args.out_root, f'{MODEL}_{args.dataset}_s{args.seed}_reg_{name}')
        config = scale_reg_weights(base_config, mult)
        print(f"\n{'-' * 60}\n  variant reg_{name} (x{mult:g}) -> {run_dir}\n"
              f"  weights: {flat_reg_weights(config)}\n{'-' * 60}")

        row = {'dataset': args.dataset, 'variant': name, 'reg_multiplier': mult,
               **flat_reg_weights(config), 'val_hit_ratio@10': '', 'val_ndcg@10': '',
               'test_hit_ratio@10': '', 'test_ndcg@10': '', 'wall_seconds': '',
               'profile': profile, 'status': 'FAIL'}
        try:
            val_metrics, test_metrics, wall_sec = run_variant(
                config, args.dataset, mult, run_dir, args.num_workers, wandb_tags)
            save_variant_results(run_dir, config, args.dataset, mult,
                                 val_metrics, test_metrics, wall_sec, profile)
            row.update({'val_hit_ratio@10': val_metrics.get('hit_ratio@10'),
                        'val_ndcg@10': val_metrics.get('ndcg@10'),
                        'test_hit_ratio@10': test_metrics.get('hit_ratio@10'),
                        'test_ndcg@10': test_metrics.get('ndcg@10'),
                        'wall_seconds': wall_sec, 'status': 'OK'})
            print(f"  reg_{name}: val HR@10={val_metrics.get('hit_ratio@10'):.4f}, "
                  f"test HR@10={test_metrics.get('hit_ratio@10'):.4f} "
                  f"({wall_sec / 60:.0f} min)")

            if not args.skip_explanations:
                try:
                    from utilities.explanations.pipeline import run_explanations_pipeline
                    exp_dir = run_explanations_pipeline(run_dir)
                    if exp_dir:
                        print(f"  explanations -> {exp_dir}")
                except Exception as e:
                    print(f"  ! explanations failed for reg_{name} (non-fatal): {e!r}")
        except Exception as e:
            failures.append(name)
            print(f"  X variant reg_{name} FAILED: {type(e).__name__}: {e}")
        rows.append(row)

    write_sweep_summary(args.out_root, args.dataset, args.seed, rows, profile)

    if failures:
        msg = f"FAIL: variants failed: {', '.join(failures)}"
    else:
        msg = "OK"
    if status_file:
        try:
            with open(status_file, 'w') as f:
                f.write(msg + "\n")
        except OSError as e:
            print(f"  ! could not write status file {status_file}: {e!r}")

    if failures:
        print(f"\nSWEEP_FAILED: {msg}")
        sys.exit(1)
    print(f"\nSWEEP_OK: {len(rows)} variants completed for {args.dataset}")


if __name__ == '__main__':
    main()
