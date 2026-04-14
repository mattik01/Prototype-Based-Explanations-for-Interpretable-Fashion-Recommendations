"""
Orchestrator: run a single dataset×model replication combo with GPU resource logging.

Usage:
    python Master/scripts/run_combo.py -m item_proto -d ml-1m
    python Master/scripts/run_combo.py -m mf -d amazon2014 -s 9491758

Delayed start (wait N hours before starting):
    python Master/scripts/run_combo.py -m item_proto -d ml-1m --delay 0.5
"""

import argparse
import copy
import json
import os
import platform
import shutil
import sys
import time

# Ensure repo root and script dir are on sys.path so imports work from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

# Set env vars before any other imports (Ray, W&B, PyTorch)
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

from confs.hyper_params import (
    mf_hyper_params,
    anchor_hyper_params,
    user_proto_chose_original_hyper_params,
    item_proto_chose_original_hyper_params,
    proto_double_tie_chose_original_hyper_params,
)
from experiment_helper import start_hyper
from utilities.consts import SINGLE_SEED, EXPERIMENT_RESULTS_PATH
from gpu_sampler import GpuSampler

MODEL_CONFIGS = {
    'mf': mf_hyper_params,
    'acf': anchor_hyper_params,
    'user_proto': user_proto_chose_original_hyper_params,
    'item_proto': item_proto_chose_original_hyper_params,
    'user_item_proto': proto_double_tie_chose_original_hyper_params,
}

VALID_DATASETS = ['amazon2014', 'ml-1m', 'lfm2b-1mon', 'hm_full', 'hm_3_month']

LOG_DIR = os.path.join(REPO_ROOT, "Master", "temp", "gpu_logs")


def _format_duration(seconds):
    """Format seconds into a human-readable duration string."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
    elif s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    else:
        return f"{seconds:.0f}s"


def _format_eta(seconds):
    """Format seconds into HH:MM for ETA display."""
    s = int(seconds)
    h, m = divmod(s, 3600)
    m //= 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _extract_hyperparams(config):
    """Extract flat hyperparameters from nested config dict for the summary."""
    ft = config.get('ft_ext_param', {})
    item_ft = ft.get('item_ft_ext_param', {})
    user_ft = ft.get('user_ft_ext_param', {})
    optim = config.get('optim_param', {})

    loss = config.get('loss_func_name', '')
    if loss == 'sampled_softmax':
        loss = 's_sm'

    hp = {
        'emb_dim': ft.get('embedding_dim', ''),
        'batch': config.get('batch_size', ''),
        'loss': loss,
        'optim': optim.get('optim', ''),
        'lr': optim.get('lr', ''),
        'wd': optim.get('wd', ''),
        'neg_train': config.get('neg_train', ''),
    }

    # Prototype-specific
    u_protos = user_ft.get('n_prototypes', '')
    i_protos = item_ft.get('n_prototypes', '')
    n_anchors = ft.get('n_anchors', '')

    if u_protos and i_protos:
        hp['n_protos'] = f"{u_protos}(u)/{i_protos}(i)"
        hp['sim_proto_w'] = f"{user_ft.get('sim_proto_weight', ''):.4g}(u)/{item_ft.get('sim_proto_weight', ''):.4g}(i)"
        hp['sim_batch_w'] = f"{user_ft.get('sim_batch_weight', ''):.4g}(u)/{item_ft.get('sim_batch_weight', ''):.4g}(i)"
    elif u_protos:
        hp['n_protos'] = f"{u_protos} (u)"
        hp['sim_proto_w'] = f"{user_ft.get('sim_proto_weight', ''):.4g}"
        hp['sim_batch_w'] = f"{user_ft.get('sim_batch_weight', ''):.4g}"
    elif i_protos:
        hp['n_protos'] = f"{i_protos} (i)"
        hp['sim_proto_w'] = f"{item_ft.get('sim_proto_weight', ''):.4g}"
        hp['sim_batch_w'] = f"{item_ft.get('sim_batch_weight', ''):.4g}"
    elif n_anchors:
        hp['n_protos'] = f"{n_anchors} (n_a)"
        hp['sim_proto_w'] = ''
        hp['sim_batch_w'] = ''
    else:
        hp['n_protos'] = ''
        hp['sim_proto_w'] = ''
        hp['sim_batch_w'] = ''

    return hp


def _generate_summary(result, hw_summary, wall_sec, gpu_per_trial=0.0625):
    """Generate summary.md with pre-formatted replication report rows."""
    model = result['model']
    dataset = result['dataset']
    seed = result['seed']
    test = result['test_metrics']
    val = result['val_metrics']
    hp = _extract_hyperparams(result['best_config'])
    concurrency = int(1 / gpu_per_trial)
    wall_str = _format_duration(wall_sec) if wall_sec else 'N/A'
    host = platform.node()

    lines = [
        f"# Experiment Summary: {model} x {dataset} (seed={seed})",
        "",
        f"- **Machine:** {host}",
        f"- **Completed:** {result['num_completed']}/{result['num_total']} trials (configured: {result['num_samples']})",
        f"- **ASHA:** yes",
        f"- **Concurrency:** {concurrency}",
        f"- **Wall time:** {wall_str}",
        f"- **Run name:** {result['run_name']}",
        "",
    ]

    # --- GPU Benchmark row ---
    if hw_summary and hw_summary.get('total_samples', 0) > 0:
        lines += [
            "## GPU Benchmark Row",
            "",
            "| Dataset | Model | Concurrency | ASHA | Peak VRAM (MB) | Avg GPU% | Avg CPU% | Avg RAM (MB) | Wall Time |",
            "|---------|-------|:-----------:|:----:|:--------------:|:--------:|:--------:|:------------:|:---------:|",
            f"| {dataset} | {model} | {concurrency} | yes | {hw_summary['max_vram_mib']:,.0f} | {hw_summary['avg_gpu_util']:.1f} | {hw_summary['avg_cpu_pct']:.1f} | {hw_summary['avg_ram_mib']:,.0f} | {wall_str} |",
            "",
        ]

    # --- Results row ---
    val_hr10 = val.get('hit_ratio@10', '')
    val_ndcg10 = val.get('ndcg@10', '')
    test_hr10 = test.get('hit_ratio@10', '')
    test_ndcg10 = test.get('ndcg@10', '')

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else ''

    lines += [
        "## Results Row",
        "",
        "| Dataset | Model | Seeds | Val HR@10 | Val NDCG@10 | Test HR@10 | Test NDCG@10 |",
        "|---------|-------|:-----:|:---------:|:-----------:|:----------:|:------------:|",
        f"| {dataset} | {model} | single | {_fmt(val_hr10)} | {_fmt(val_ndcg10)} | **{_fmt(test_hr10)}** | {_fmt(test_ndcg10)} |",
        "",
    ]

    # --- Hyperparameters row ---
    def _fmt_sci(v):
        if isinstance(v, float):
            return f"{v:.2e}" if v < 0.01 else f"{v:.4g}"
        return str(v)

    lines += [
        "## Hyperparameters Row",
        "",
        "| Dataset | Model | emb_dim | batch | n_protos (u/i) | sim_proto_w (u/i) | sim_batch_w (u/i) | loss | optim | lr | wd | neg_train |",
        "|---------|-------|:-------:|:-----:|:--------------:|:------------------:|:------------------:|:----:|:-----:|:--:|:--:|:---------:|",
        f"| {dataset} | {model} | {hp['emb_dim']} | {hp['batch']} | {hp['n_protos']} | {hp['sim_proto_w']} | {hp['sim_batch_w']} | {hp['loss']} | {hp['optim']} | {_fmt_sci(hp['lr'])} | {_fmt_sci(hp['wd'])} | {hp['neg_train']} |",
        "",
    ]

    # --- Full metrics dump ---
    lines += ["## All Test Metrics", ""]
    for k, v in sorted(test.items()):
        lines.append(f"- **{k}:** {v:.4f}")
    lines += ["", "## All Validation Metrics (at best checkpoint)", ""]
    for k, v in sorted(val.items()):
        lines.append(f"- **{k}:** {v:.4f}")
    lines.append("")

    return "\n".join(lines)


def _save_combo_results(result, hw_summary=None, hw_csv_path=None, wall_sec=None,
                        gpu_per_trial=0.0625):
    """Save all experiment artifacts to a structured results folder."""
    folder_name = f"{result['model']}_{result['dataset']}_s{result['seed']}"
    results_dir = os.path.join(EXPERIMENT_RESULTS_PATH, folder_name)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Copy best model checkpoint
    ckpt_src = result['best_checkpoint_path']
    if os.path.exists(ckpt_src):
        shutil.copy2(ckpt_src, os.path.join(results_dir, 'best_model.pth'))

    # 2. Config (strip wandb internals)
    config_clean = {k: v for k, v in result['best_config'].items()
                    if not k.startswith('_wandb_')}
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config_clean, f, indent=2, default=str)

    # 3. Test metrics
    with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
        json.dump(result['test_metrics'], f, indent=2)

    # 4. Val metrics
    with open(os.path.join(results_dir, 'val_metrics.json'), 'w') as f:
        json.dump(result['val_metrics'], f, indent=2)

    # 5. Hardware log CSV
    if hw_csv_path and os.path.exists(hw_csv_path):
        shutil.copy2(hw_csv_path, os.path.join(results_dir, 'hardware_log.csv'))

    # 6. Hardware summary
    if hw_summary:
        with open(os.path.join(results_dir, 'hardware_summary.json'), 'w') as f:
            json.dump(hw_summary, f, indent=2)

    # 7. Metadata
    meta = {
        'model': result['model'],
        'dataset': result['dataset'],
        'seed': result['seed'],
        'num_samples': result['num_samples'],
        'num_completed': result['num_completed'],
        'num_total': result['num_total'],
        'run_name': result['run_name'],
        'wall_seconds': wall_sec,
        'machine': platform.node(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 8. Summary markdown
    summary = _generate_summary(result, hw_summary, wall_sec, gpu_per_trial=gpu_per_trial)
    with open(os.path.join(results_dir, 'summary.md'), 'w') as f:
        f.write(summary)

    print(f"\n  Results saved to: {results_dir}")
    return results_dir


def run_single_combo(model, dataset, seed, resource_cfg=None):
    """Run one combo with GPU logging. Returns (summary_dict, csv_path, wall_seconds)."""
    conf = copy.deepcopy(MODEL_CONFIGS[model])

    print(f"\n{'=' * 60}")
    print(f"  run_combo: {model} × {dataset} (seed={seed})")
    if resource_cfg:
        print(f"  resource_cfg: {resource_cfg}")
    print(f"{'=' * 60}")

    sampler = GpuSampler(log_dir=LOG_DIR, interval_sec=10)
    sampler.start()
    wall_start = time.monotonic()

    try:
        result = start_hyper(conf, model, dataset, seed, resource_cfg=resource_cfg)
    finally:
        sampler.stop()
        new_name = f"gpu_{model}_{dataset}_s{seed}.csv"
        csv_path = sampler.rename_log(new_name)

    wall_sec = time.monotonic() - wall_start
    hw_summary = sampler.get_summary()

    print(f"\n{'=' * 60}")
    print(f"  Resource Utilization Summary — {model} × {dataset}")
    print(f"{'=' * 60}")
    print(f"  Peak VRAM:       {hw_summary['max_vram_mib']:.0f} MiB")
    print(f"  Avg VRAM:        {hw_summary['avg_vram_mib']:.0f} MiB")
    print(f"  Avg GPU util:    {hw_summary['avg_gpu_util']:.1f}%")
    print(f"  Peak GPU temp:   {hw_summary['max_gpu_temp_c']:.0f} °C")
    print(f"  Avg GPU temp:    {hw_summary['avg_gpu_temp_c']:.0f} °C")
    print(f"  Peak RAM:        {hw_summary['max_ram_mib']:.0f} MiB")
    print(f"  Avg RAM:         {hw_summary['avg_ram_mib']:.0f} MiB")
    print(f"  Avg CPU:         {hw_summary['avg_cpu_pct']:.1f}%")
    print(f"  Total runtime:   {_format_duration(wall_sec)}")
    print(f"  Samples:         {hw_summary['total_samples']}")
    print(f"  CSV log:         {csv_path}")
    print(f"{'=' * 60}")

    # Save structured results
    gpu_per_trial = resource_cfg.get('gpu_per_trial', 0.0625) if resource_cfg else 0.0625
    _save_combo_results(result, hw_summary=hw_summary, hw_csv_path=csv_path, wall_sec=wall_sec,
                        gpu_per_trial=gpu_per_trial)

    return result, hw_summary, csv_path, wall_sec


def wait_with_countdown(delay_hours):
    """Sleep for delay_hours with periodic status prints."""
    total_sec = delay_hours * 3600
    start = time.monotonic()
    target = time.strftime('%H:%M', time.localtime(time.time() + total_sec))

    print(f"\n{'=' * 60}")
    print(f"  DELAYED START: waiting {delay_hours:.2g}h (until ~{target})")
    print(f"{'=' * 60}")

    while True:
        elapsed = time.monotonic() - start
        remaining = total_sec - elapsed
        if remaining <= 0:
            break
        sleep_chunk = min(600, remaining)
        time.sleep(sleep_chunk)
        remaining = total_sec - (time.monotonic() - start)
        if remaining > 0:
            print(f"  ... {_format_eta(remaining)} remaining")

    print(f"  Delay complete — starting now.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run a single replication combo with GPU logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -m item_proto -d ml-1m\n"
            "  %(prog)s -m mf -d amazon2014 -s 9491758\n"
            "  %(prog)s -m item_proto -d ml-1m --delay 0.5\n"
        ),
    )

    parser.add_argument('--model', '-m', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model to run')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        choices=VALID_DATASETS,
                        help='Dataset to use')
    parser.add_argument('--seed', '-s', type=int, default=SINGLE_SEED,
                        help=f'Random seed (default: {SINGLE_SEED})')
    parser.add_argument('--delay', type=float, default=0,
                        help='Hours to wait before starting (e.g. 2.5 for 2h30m)')

    # --- Execution-level resource configuration ---
    res = parser.add_argument_group('resource configuration',
                                    'Per-execution overrides (defaults match current global consts)')
    res.add_argument('--num-samples', type=int, default=100,
                     help='Total hyperparameter trials (default: 100)')
    res.add_argument('--gpu-per-trial', type=float, default=0.0625,
                     help='Fractional GPU per trial — concurrency = 1/value (default: 0.0625 → 16 trials)')
    res.add_argument('--cpu-per-trial', type=int, default=1,
                     help='CPU cores per trial (default: 1)')
    res.add_argument('--num-workers', type=int, default=2,
                     help='DataLoader workers per trial (default: 2)')
    res.add_argument('--n-epochs', type=int, default=100,
                     help='Max epochs per trial (default: 100)')
    res.add_argument('--grace-period', type=int, default=4,
                     help='ASHA min epochs before early termination (default: 4)')
    res.add_argument('--patience', type=int, default=10,
                     help='Epochs without improvement before stopping a trial (default: 10)')
    res.add_argument('--wandb-mode', type=str, default='online',
                     choices=['online', 'offline', 'disabled'],
                     help='W&B logging mode (default: online)')
    res.add_argument('--optimizing-metric', type=str, default='hit_ratio@10',
                     help='Metric for model selection (default: hit_ratio@10)')
    res.add_argument('--no-asha', dest='asha', action='store_false', default=True,
                     help='Disable ASHA early stopping scheduler (default: enabled)')
    res.add_argument('--wandb-tag', action='append', default=[],
                     help='Extra W&B tag (repeatable, e.g. --wandb-tag smoke --wandb-tag test)')

    args = parser.parse_args()

    # Set W&B mode via env var (respected by wandb.init)
    os.environ['WANDB_MODE'] = args.wandb_mode

    # Build resource_cfg dict from CLI args
    resource_cfg = {
        'num_samples': args.num_samples,
        'gpu_per_trial': args.gpu_per_trial,
        'cpu_per_trial': args.cpu_per_trial,
        'num_workers': args.num_workers,
        'n_epochs': args.n_epochs,
        'grace_period': args.grace_period,
        'patience': args.patience,
        'optimizing_metric': args.optimizing_metric,
        'asha': args.asha,
        'wandb_tags': args.wandb_tag,
    }

    if args.delay > 0:
        wait_with_countdown(args.delay)

    run_single_combo(args.model, args.dataset, args.seed, resource_cfg=resource_cfg)


if __name__ == "__main__":
    main()
