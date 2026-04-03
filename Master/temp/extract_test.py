"""Extract test results for user_item_proto on amazon2014."""
import os
from ray.tune import ExperimentAnalysis

result_dir = os.path.expanduser('~/ray_results/user_item_proto_amazon2014_DE_38210573_2026-3-31_13-15-21.491510')
analysis = ExperimentAnalysis(result_dir)

metric = 'hit_ratio@10'
best_trial = analysis.get_best_trial(metric, 'max', scope='all')
best_config = best_trial.config

# Find and load the best checkpoint
best_checkpoint = analysis.get_best_checkpoint(best_trial, metric, 'max')
if best_checkpoint:
    checkpoint_dir = best_checkpoint.to_directory()
    model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    print(f'Best checkpoint: {model_path}')
    print(f'Exists: {os.path.exists(model_path)}')
else:
    print('No checkpoint found')

# Check if test was already run by looking at W&B or the trial results
# The test is run automatically by start_hyper after finding best trial
# Let's check the last result more carefully
print()
print('=== All keys in best trial last_result ===')
for k in sorted(best_trial.last_result.keys()):
    print(f'  {k}: {best_trial.last_result[k]}')
