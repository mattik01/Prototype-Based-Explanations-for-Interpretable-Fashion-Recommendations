"""
GPU VRAM & utilization benchmark for every dataset × model combination.
Two run types per combo:
  - "worstcase": 1 trial, maxed hyperparams (already done — pre-seeded in CSV)
  - "normal":   10 trials, real hyperopt search spaces, 3 epochs

Results written to CSV incrementally after each combo.

Usage:
    conda activate protomf
    python Master/temp/gpu_benchmark.py
"""
import copy
import csv
import os
import subprocess
import sys
import threading

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from experiment_helper import start_hyper
import experiment_helper
# Force 1 trial at a time so GPU monitor measures per-trial VRAM accurately
experiment_helper.GPU_PER_TRIAL = 1.0
from confs.hyper_params import (
    mf_hyper_params,
    anchor_hyper_params,
    user_proto_chose_original_hyper_params,
    item_proto_chose_original_hyper_params,
    proto_double_tie_chose_original_hyper_params,
)

# ---------------------------------------------------------------------------
# CSV setup
# ---------------------------------------------------------------------------
FIELDNAMES = ['dataset', 'model', 'run_type', 'max_vram_mb', 'avg_gpu_util_pct',
              'n_samples', 'status']

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'gpu_benchmark_results.csv')

DATASETS = ['amazon2014', 'ml-1m']
MODELS = ['mf', 'acf', 'user_proto', 'item_proto', 'user_item_proto']

# ---------------------------------------------------------------------------
# Previous worst-case results (already measured, 512 MB baseline NOT subtracted)
# ---------------------------------------------------------------------------
WORSTCASE_RESULTS = [
    {'dataset': 'amazon2014', 'model': 'mf',              'run_type': 'worstcase', 'max_vram_mb': 1062.0, 'avg_gpu_util_pct': 3.6,  'n_samples': 279, 'status': 'ok', },
    {'dataset': 'amazon2014', 'model': 'acf',             'run_type': 'worstcase', 'max_vram_mb': 1198.0, 'avg_gpu_util_pct': 10.0, 'n_samples': 243, 'status': 'ok', },
    {'dataset': 'amazon2014', 'model': 'user_proto',      'run_type': 'worstcase', 'max_vram_mb': 1266.0, 'avg_gpu_util_pct': 26.8, 'n_samples': 234, 'status': 'ok', },
    {'dataset': 'amazon2014', 'model': 'item_proto',      'run_type': 'worstcase', 'max_vram_mb': 10072.0,'avg_gpu_util_pct': 35.7, 'n_samples': 352, 'status': 'ok', },
    {'dataset': 'amazon2014', 'model': 'user_item_proto', 'run_type': 'worstcase', 'max_vram_mb': 13024.0,'avg_gpu_util_pct': 37.3, 'n_samples': 348, 'status': 'ok', },
    {'dataset': 'ml-1m',      'model': 'mf',              'run_type': 'worstcase', 'max_vram_mb': 4080.0, 'avg_gpu_util_pct': 6.5,  'n_samples': 368, 'status': 'ok', },
    {'dataset': 'ml-1m',      'model': 'acf',             'run_type': 'worstcase', 'max_vram_mb': 4138.0, 'avg_gpu_util_pct': 21.8, 'n_samples': 459, 'status': 'ok', },
    {'dataset': 'ml-1m',      'model': 'user_proto',      'run_type': 'worstcase', 'max_vram_mb': 4180.0, 'avg_gpu_util_pct': 32.9, 'n_samples': 382, 'status': 'ok', },
    {'dataset': 'ml-1m',      'model': 'item_proto',      'run_type': 'worstcase', 'max_vram_mb': 13002.0,'avg_gpu_util_pct': 61.6, 'n_samples': 928, 'status': 'ok', },
    {'dataset': 'ml-1m',      'model': 'user_item_proto', 'run_type': 'worstcase', 'max_vram_mb': 13020.0,'avg_gpu_util_pct': 62.2, 'n_samples': 935, 'status': 'ok', },
]

# ---------------------------------------------------------------------------
# Normal configs — real search spaces, overridden to 10 trials / 3 epochs
# ---------------------------------------------------------------------------
TRIALS_PER_COMBO = 10

def _make_normal_config(base_conf):
    """Deep-copy a hyper_params config: 1 trial, 3 epochs (called repeatedly)."""
    conf = copy.deepcopy(base_conf)
    conf['num_samples'] = 1
    conf['n_epochs'] = 3
    return conf

NORMAL_CONFIGS = {
    'mf':              _make_normal_config(mf_hyper_params),
    'acf':             _make_normal_config(anchor_hyper_params),
    'user_proto':      _make_normal_config(user_proto_chose_original_hyper_params),
    'item_proto':      _make_normal_config(item_proto_chose_original_hyper_params),
    'user_item_proto': _make_normal_config(proto_double_tie_chose_original_hyper_params),
}

# ---------------------------------------------------------------------------
# GPU monitor — polls nvidia-smi in a background thread
# ---------------------------------------------------------------------------
class GPUMonitor:
    def __init__(self, poll_interval=0.5):
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = None
        self.max_vram_mb = 0.0
        self.utilization_samples = []

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ['nvidia-smi',
                     '--query-gpu=memory.used,utilization.gpu',
                     '--format=csv,noheader,nounits'],
                    text=True, timeout=5,
                )
                for line in out.strip().split('\n'):
                    parts = line.split(',')
                    vram_mb = float(parts[0].strip())
                    util_pct = float(parts[1].strip())
                    self.max_vram_mb = max(self.max_vram_mb, vram_mb)
                    self.utilization_samples.append(util_pct)
            except Exception:
                pass
            self._stop.wait(self.poll_interval)

    def start(self):
        self.max_vram_mb = 0.0
        self.utilization_samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def avg_utilization(self):
        if not self.utilization_samples:
            return 0.0
        return sum(self.utilization_samples) / len(self.utilization_samples)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def write_csv(rows):
    """(Over)write the full CSV with current rows."""
    with open(RESULTS_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print(f"\n{'dataset':<14} {'model':<18} {'type':<10} {'max_vram':>10} {'avg_gpu%':>10} {'status'}")
    print('-' * 78)
    for r in rows:
        print(f"{r['dataset']:<14} {r['model']:<18} {r['run_type']:<10} "
              f"{r['max_vram_mb']:>10} {r['avg_gpu_util_pct']:>9}% {r['status']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Start with pre-existing worst-case results
    all_rows = list(WORSTCASE_RESULTS)
    write_csv(all_rows)
    print(f"Pre-seeded {len(WORSTCASE_RESULTS)} worst-case results into CSV.")

    total_trials = len(DATASETS) * len(MODELS) * TRIALS_PER_COMBO
    trial_num = 0

    for dataset in DATASETS:
        for model in MODELS:
            for t in range(1, TRIALS_PER_COMBO + 1):
                trial_num += 1
                tag = f"[{trial_num}/{total_trials}] {model} × {dataset} (trial {t}/{TRIALS_PER_COMBO})"
                print(f"\n{'=' * 60}")
                print(f"  {tag}")
                print(f"{'=' * 60}\n")

                config = copy.deepcopy(NORMAL_CONFIGS[model])
                monitor = GPUMonitor(poll_interval=0.5)

                try:
                    monitor.start()
                    start_hyper(config, f'bench_{model}', dataset)
                    monitor.stop()
                    status = 'ok'
                except Exception as e:
                    monitor.stop()
                    status = f'error: {e}'
                    print(f"  !! FAILED: {e}")

                row = {
                    'dataset': dataset,
                    'model': model,
                    'run_type': f'normal_t{t}',
                    'max_vram_mb': round(monitor.max_vram_mb, 1),
                    'avg_gpu_util_pct': round(monitor.avg_utilization, 1),
                    'n_samples': len(monitor.utilization_samples),
                    'status': status,
                }
                all_rows.append(row)

                # Write incrementally after each trial
                write_csv(all_rows)

                print(f"  >> max VRAM: {row['max_vram_mb']} MB | "
                      f"avg util: {row['avg_gpu_util_pct']}% | "
                      f"status: {status}")
                print(f"  >> CSV updated ({len(all_rows)} rows)")

    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK COMPLETE — {len(all_rows)} total rows")
    print(f"  Results: {RESULTS_PATH}")
    print(f"{'=' * 60}")
    print_summary(all_rows)


if __name__ == '__main__':
    main()
