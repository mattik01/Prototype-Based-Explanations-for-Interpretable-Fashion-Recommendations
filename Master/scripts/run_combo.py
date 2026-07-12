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
    feature_item_proto_hyper_params,
    feature_item_proto_noid_hyper_params,
    feature_item_proto_f0_hyper_params,
    attr_item_proto_hyper_params,
    attr_item_proto_debug_hyper_params,
    attr_item_proto_debug_knobs_hyper_params,
    lightfm_tags_hyper_params,
    lightfm_tags_ids_hyper_params,
    feature_user_proto_hyper_params,
    feature_user_proto_noid_hyper_params,
    feature_user_proto_debug_hyper_params,
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
    'feature_item_proto': feature_item_proto_hyper_params,
    'feature_item_proto_noid': feature_item_proto_noid_hyper_params,
    'feature_item_proto_f0': feature_item_proto_f0_hyper_params,
    'attr_item_proto': attr_item_proto_hyper_params,
    'attr_item_proto_debug': attr_item_proto_debug_hyper_params,
    'attr_item_proto_debug_knobs': attr_item_proto_debug_knobs_hyper_params,
    'lightfm_tags': lightfm_tags_hyper_params,
    'lightfm_tags_ids': lightfm_tags_ids_hyper_params,
    'feature_user_proto': feature_user_proto_hyper_params,
    'feature_user_proto_noid': feature_user_proto_noid_hyper_params,
    'feature_user_proto_debug': feature_user_proto_debug_hyper_params,
}

VALID_DATASETS = ['amazon2014', 'ml-1m', 'lfm2b-1mon', 'hm_full', 'hm_3_month', 'hm_1_month',
                  'hm_1_month_cold']

# Keys of a saved config.json that constitute the model/training spec (used by --retrain-config).
# Execution-level keys (data_path, seed, _num_workers, ...) are re-derived by start_hyper.
RETRAIN_CONFIG_KEYS = ('n_epochs', 'eval_neg_strategy', 'val_batch_size', 'rec_sys_param',
                       'neg_train', 'train_neg_strategy', 'loss_func_name', 'loss_func_aggr',
                       'batch_size', 'optim_param', 'ft_ext_param', 'device')

# feature_item_proto wired at dc01 SC.5 (F-DC01-11, 2026-07-12): the pipeline carries its
# intrinsic naming route, the dual post-hoc route (F-DC01-13), the shared breakdown renderer
# and prototype cards — auto-explanations now run for the fI headline model. The _noid/_f0
# ablation arms are separate model keys and stay out of auto-explanations (host convention:
# only headline models auto-explain); reach them via the standalone pipeline/breakdown CLI.
# attr_item_proto (dc02) stays opt-in until ITS SC.5: its pipeline route exists
# (utilities/explanations), but auto-explanations after training remain off — run with
# --skip-explanations and invoke the pipeline standalone on the results dir.
# feature_user_proto (dc05) follows the same convention: renderer/cards/naming slots are wired
# at build, but auto-explanations stay opt-in until SC(dc05)'s explanation gate.
EXPLAINABLE_MODELS = {'item_proto', 'user_proto', 'user_item_proto', 'feature_item_proto'}

# Preset bundles for the hyperopt search budget. Each profile sets defaults for
# num_samples / n_epochs / patience / grace_period; any explicit CLI flag still wins.
#
# - production: paper-comparable full sweep (the original global defaults). Keep for
#   final, reportable numbers.
# - dev: lighter sweep for fast iteration. ~4x less runtime, lands ~90% of the
#   full-100 best val on average. Derived by back-testing 729 recorded trials
#   (best-of-first-K curve, winner peak-epoch distribution, ASHA rung@4 percentiles);
#   see Master/temp/analyze_trial_history.py. Caveat: slow-converging combos
#   (user_proto / item_proto on amazon2014) are under-served — bump samples/epochs
#   if iterating specifically on those. NOT for final reporting.
# - smoke: minimal sanity check that the pipeline runs end-to-end.
PROFILES = {
    'production': dict(num_samples=100, n_epochs=100, patience=10, grace_period=4),
    'dev':        dict(num_samples=30,  n_epochs=60,  patience=7,  grace_period=4),
    'smoke':      dict(num_samples=5,   n_epochs=15,  patience=10, grace_period=4),
}
PROFILE_KEYS = ('num_samples', 'n_epochs', 'patience', 'grace_period')

# GPU telemetry staging dir. Overridable so the cluster can point it at a per-job
# scratch path (avoids cross-job collisions on shared networked storage).
LOG_DIR = os.environ.get("GPU_LOG_DIR", os.path.join(REPO_ROOT, "Master", "temp", "gpu_logs"))


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


def load_retrain_config(config_path):
    """S0.3 §6 single-config retrain: take a completed run's config.json (the best config the
    canonical search selected) and rebuild a FIXED conf dict from it — num_samples=1, no search
    space. Execution-level keys are dropped (start_hyper re-derives data_path/seed/etc.), so the
    same config retrains cleanly on another dataset (the cold variant)."""
    with open(config_path) as f:
        saved = json.load(f)
    conf = {k: saved[k] for k in RETRAIN_CONFIG_KEYS if k in saved}
    conf['num_samples'] = 1
    # device is a machine property, not part of the selected config — re-detect on this host
    import torch
    conf['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[retrain-config] fixed config loaded from {config_path} "
          f"(ft_type={conf.get('ft_ext_param', {}).get('ft_type')})")
    return conf


def run_single_combo(model, dataset, seed, resource_cfg=None, retrain_config=None):
    """Run one combo with GPU logging. Returns (summary_dict, csv_path, wall_seconds)."""
    if retrain_config:
        conf = load_retrain_config(retrain_config)
    else:
        conf = copy.deepcopy(MODEL_CONFIGS[model])

    print(f"\n{'=' * 60}")
    print(f"  run_combo: {model} × {dataset} (seed={seed})")
    if resource_cfg:
        print(f"  resource_cfg: {resource_cfg}")
    print(f"{'=' * 60}")

    sampler = GpuSampler(log_dir=LOG_DIR, interval_sec=10)
    sampler.start()
    wall_start = time.monotonic()

    csv_path = None
    try:
        result = start_hyper(conf, model, dataset, seed, resource_cfg=resource_cfg)
    finally:
        # Telemetry teardown is best-effort and must NEVER abort the run or prevent
        # result-saving (a failure here previously discarded a successful run).
        try:
            sampler.stop()
            new_name = f"gpu_{model}_{dataset}_s{seed}.csv"
            csv_path = sampler.rename_log(new_name)
        except Exception as e:
            print(f"  ⚠ GPU sampler teardown failed (non-fatal): {e!r}")

    wall_sec = time.monotonic() - wall_start
    try:
        hw_summary = sampler.get_summary()
    except Exception as e:
        print(f"  ⚠ GPU sampler summary failed (non-fatal): {e!r}")
        hw_summary = None

    print(f"\n{'=' * 60}")
    print(f"  Resource Utilization Summary — {model} × {dataset}")
    print(f"{'=' * 60}")
    if hw_summary:
        print(f"  Peak VRAM:       {hw_summary['max_vram_mib']:.0f} MiB")
        print(f"  Avg VRAM:        {hw_summary['avg_vram_mib']:.0f} MiB")
        print(f"  Avg GPU util:    {hw_summary['avg_gpu_util']:.1f}%")
        print(f"  Peak GPU temp:   {hw_summary['max_gpu_temp_c']:.0f} °C")
        print(f"  Avg GPU temp:    {hw_summary['avg_gpu_temp_c']:.0f} °C")
        print(f"  Peak RAM:        {hw_summary['max_ram_mib']:.0f} MiB")
        print(f"  Avg RAM:         {hw_summary['avg_ram_mib']:.0f} MiB")
        print(f"  Avg CPU:         {hw_summary['avg_cpu_pct']:.1f}%")
        print(f"  Samples:         {hw_summary['total_samples']}")
    else:
        print(f"  (GPU telemetry unavailable — results unaffected)")
    print(f"  Total runtime:   {_format_duration(wall_sec)}")
    print(f"  CSV log:         {csv_path}")
    print(f"{'=' * 60}")

    # Save structured results
    gpu_per_trial = resource_cfg.get('gpu_per_trial', 0.0625) if resource_cfg else 0.0625
    results_dir = _save_combo_results(result, hw_summary=hw_summary, hw_csv_path=csv_path,
                                      wall_sec=wall_sec, gpu_per_trial=gpu_per_trial)

    # Auto-invoke explanations pipeline for explainable models
    skip_explanations = resource_cfg.get('skip_explanations', False) if resource_cfg else False
    if model in EXPLAINABLE_MODELS and not skip_explanations:
        # Generate both naming scorings: 'lift' (distinctive descriptors) and
        # 'raw' (dominant/frequent descriptors). Each lands in its own
        # explanations/<scoring>/ folder; geometry plots are duplicated but cheap.
        from utilities.explanations.pipeline import run_explanations_pipeline
        for scoring in ("lift", "raw"):
            try:
                exp_dir = run_explanations_pipeline(
                    results_dir, naming_overrides={"scoring": scoring})
                if exp_dir:
                    print(f"  Explanations ({scoring}) saved to: {exp_dir}")
            except Exception as e:
                # Explanations are non-critical; a failure must not invalidate a completed run.
                print(f"  ⚠ Explanations pipeline ({scoring}) failed (non-fatal): {e!r}")

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
    parser.add_argument('--retrain-config', type=str, default=None,
                        help='Path to a completed run\'s config.json: retrain that FIXED config '
                             '(num_samples=1, no search) on the given dataset — the S0.3 §6 '
                             'cold-variant convention. -m still names the model for the results dir.')

    # --- Execution-level resource configuration ---
    res = parser.add_argument_group('resource configuration',
                                    'Per-execution overrides (defaults match current global consts)')
    res.add_argument('--profile', choices=list(PROFILES.keys()), default='production',
                     help='Search-budget preset for num-samples/n-epochs/patience/grace-period. '
                          'production=100/100/10/4 (paper-comparable, default); '
                          'dev=30/60/7/4 (~4x faster, ~90%% of best val — for iteration, not reporting); '
                          'smoke=5/15/10/4 (pipeline sanity check). '
                          'Any explicit flag below overrides the profile value.')
    res.add_argument('--num-samples', type=int, default=None,
                     help='Total hyperparameter trials (overrides --profile; production default: 100)')
    res.add_argument('--gpu-per-trial', type=float, default=0.0625,
                     help='Fractional GPU per trial — concurrency = 1/value (default: 0.0625 → 16 trials)')
    res.add_argument('--cpu-per-trial', type=int, default=1,
                     help='CPU cores per trial (default: 1)')
    res.add_argument('--num-workers', type=int, default=2,
                     help='DataLoader workers per trial (default: 2)')
    res.add_argument('--n-epochs', type=int, default=None,
                     help='Max epochs per trial (overrides --profile; production default: 100)')
    res.add_argument('--grace-period', type=int, default=None,
                     help='ASHA min epochs before early termination (overrides --profile; production default: 4)')
    res.add_argument('--patience', type=int, default=None,
                     help='Epochs without improvement before stopping a trial (overrides --profile; production default: 10)')
    res.add_argument('--min-delta', type=float, default=None,
                     help='Min improvement on the optimizing metric required to reset patience '
                          '(default: per-metric value from MIN_DELTA_DEFAULTS, e.g. 1e-4 for hit_ratio@10/ndcg@10)')
    res.add_argument('--wandb-mode', type=str, default='online',
                     choices=['online', 'offline', 'disabled'],
                     help='W&B logging mode (default: online — live dashboard; safe because '
                          'safe_wandb_* makes W&B non-fatal in any mode. Use offline to fully '
                          'decouple training from the network, synced after the run).')
    res.add_argument('--optimizing-metric', type=str, default='hit_ratio@10',
                     help='Metric for model selection (default: hit_ratio@10)')
    res.add_argument('--no-asha', dest='asha', action='store_false', default=True,
                     help='Disable ASHA early stopping scheduler (default: enabled)')
    res.add_argument('--wandb-tag', action='append', default=[],
                     help='Extra W&B tag (repeatable, e.g. --wandb-tag smoke --wandb-tag test)')
    res.add_argument('--skip-explanations', action='store_true', default=False,
                     help='Disable auto-invocation of the explanations pipeline after training')

    args = parser.parse_args()

    # Resolve search-budget profile: fill any arg left unset (None) from the
    # chosen profile; explicit CLI flags always win.
    profile = PROFILES[args.profile]
    resolved = []
    for k in PROFILE_KEYS:
        if getattr(args, k) is None:
            setattr(args, k, profile[k])
        else:
            resolved.append(k)
    print(f"[profile] {args.profile}: " +
          ", ".join(f"{k}={getattr(args, k)}" for k in PROFILE_KEYS) +
          (f"  (overridden: {', '.join(resolved)})" if resolved else ""))
    # Tag non-production runs so they stay filterable / out of paper-comparable analysis.
    if args.profile != 'production' and args.profile not in args.wandb_tag:
        args.wandb_tag.append(args.profile)

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
        'min_delta': args.min_delta,
        'optimizing_metric': args.optimizing_metric,
        'asha': args.asha,
        'wandb_tags': args.wandb_tag,
        'skip_explanations': args.skip_explanations,
    }

    if args.delay > 0:
        wait_with_countdown(args.delay)

    # ── Honest success/failure reporting ─────────────────────────────────────
    # Ray/W&B teardown can force a 0 exit code even when a run failed, so we do not
    # rely on the process exit code alone. We write an explicit status sentinel
    # (RUN_STATUS_FILE) that the SLURM wrapper checks, AND sys.exit(1) on failure.
    # A run counts as successful ONLY if it completed >0 trials AND persisted
    # test_metrics.json — anything less must fail loudly, never masquerade as success.
    status_file = os.environ.get('RUN_STATUS_FILE')

    def _write_status(text):
        if status_file:
            try:
                with open(status_file, 'w') as f:
                    f.write(text + "\n")
            except OSError as e:
                print(f"  ⚠ could not write status file {status_file}: {e!r}")

    try:
        result, _hw, _csv, _wall = run_single_combo(
            args.model, args.dataset, args.seed, resource_cfg=resource_cfg,
            retrain_config=args.retrain_config)
    except Exception as e:
        _write_status(f"FAIL: {type(e).__name__}: {e}")
        print(f"\nRUN_FAILED: {type(e).__name__}: {e}")
        raise

    results_dir = os.path.join(EXPERIMENT_RESULTS_PATH,
                               f"{args.model}_{args.dataset}_s{args.seed}")
    metrics_path = os.path.join(results_dir, 'test_metrics.json')
    completed = bool(result) and result.get('num_completed', 0) > 0
    persisted = os.path.exists(metrics_path)

    if completed and persisted:
        _write_status("OK")
        print(f"\nRUN_OK: {result['num_completed']}/{result['num_total']} trials, "
              f"results persisted to {results_dir}")
    else:
        reason = (f"completed_trials={result.get('num_completed', 0) if result else 0}, "
                  f"test_metrics.json_persisted={persisted}")
        _write_status(f"FAIL: {reason}")
        print(f"\nRUN_FAILED: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
