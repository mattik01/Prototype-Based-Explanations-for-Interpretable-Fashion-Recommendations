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
import os
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
from utilities.consts import SINGLE_SEED
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


def run_single_combo(model, dataset, seed):
    """Run one combo with GPU logging. Returns (summary_dict, csv_path, wall_seconds)."""
    conf = copy.deepcopy(MODEL_CONFIGS[model])

    print(f"\n{'=' * 60}")
    print(f"  run_combo: {model} × {dataset} (seed={seed})")
    print(f"{'=' * 60}")

    sampler = GpuSampler(log_dir=LOG_DIR, interval_sec=10)
    sampler.start()
    wall_start = time.monotonic()

    try:
        start_hyper(conf, model, dataset, seed)
    finally:
        sampler.stop()
        new_name = f"gpu_{model}_{dataset}_s{seed}.csv"
        csv_path = sampler.rename_log(new_name)

    wall_sec = time.monotonic() - wall_start
    summary = sampler.get_summary()

    print(f"\n{'=' * 60}")
    print(f"  Resource Utilization Summary — {model} × {dataset}")
    print(f"{'=' * 60}")
    print(f"  Peak VRAM:       {summary['max_vram_mib']:.0f} MiB")
    print(f"  Avg VRAM:        {summary['avg_vram_mib']:.0f} MiB")
    print(f"  Avg GPU util:    {summary['avg_gpu_util']:.1f}%")
    print(f"  Peak GPU temp:   {summary['max_gpu_temp_c']:.0f} °C")
    print(f"  Avg GPU temp:    {summary['avg_gpu_temp_c']:.0f} °C")
    print(f"  Peak RAM:        {summary['max_ram_mib']:.0f} MiB")
    print(f"  Avg RAM:         {summary['avg_ram_mib']:.0f} MiB")
    print(f"  Avg CPU:         {summary['avg_cpu_pct']:.1f}%")
    print(f"  Total runtime:   {_format_duration(wall_sec)}")
    print(f"  Samples:         {summary['total_samples']}")
    print(f"  CSV log:         {csv_path}")
    print(f"{'=' * 60}")

    return summary, csv_path, wall_sec


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

    args = parser.parse_args()

    if args.delay > 0:
        wait_with_countdown(args.delay)

    run_single_combo(args.model, args.dataset, args.seed)


if __name__ == "__main__":
    main()
