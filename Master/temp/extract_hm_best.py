"""Extract best trial from user_item_proto hm_3_month runs and run test."""
import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, REPO_ROOT)

from ray.tune import ExperimentAnalysis

# Check both run directories
dirs = [
    os.path.expanduser('~/ray_results/user_item_proto_hm_3_month_DE_38210573_2026-4-1_11-18-46.129340'),
    os.path.expanduser('~/ray_results/user_item_proto_hm_3_month_DE_38210573_2026-3-31_23-44-47.921663'),
]

metric = 'hit_ratio@10'

for d in dirs:
    print(f'\n=== {os.path.basename(d)} ===')
    try:
        analysis = ExperimentAnalysis(d)
        completed = [t for t in analysis.trials if t.status == 'TERMINATED']
        errored = [t for t in analysis.trials if t.status == 'ERROR']
        print(f'Completed: {len(completed)}, Errored: {len(errored)}, Total: {len(analysis.trials)}')

        if completed:
            best_trial = analysis.get_best_trial(metric, 'max', scope='all')
            best_result = best_trial.last_result
            print(f'Best val HR@10: {best_result.get("hit_ratio@10", "N/A")}')
            print(f'Best val NDCG@10: {best_result.get("ndcg@10", "N/A")}')

            best_checkpoint = analysis.get_best_checkpoint(best_trial, metric, 'max')
            if best_checkpoint:
                cp_dir = best_checkpoint.to_directory()
                model_path = os.path.join(cp_dir, 'best_model.pth')
                print(f'Checkpoint exists: {os.path.exists(model_path)}')
                print(f'Checkpoint path: {model_path}')
            else:
                print('No checkpoint found')
    except Exception as e:
        print(f'Error: {e}')
