"""Run test evaluation on best user_item_proto hm_3_month trial."""
import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, REPO_ROOT)

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

from ray.tune import ExperimentAnalysis
from experiment_helper import start_testing
from utilities.consts import WANDB_API_KEY
import wandb

result_dir = os.path.expanduser('~/ray_results/user_item_proto_hm_3_month_DE_38210573_2026-3-31_23-44-47.921663')
analysis = ExperimentAnalysis(result_dir)

metric = 'hit_ratio@10'
best_trial = analysis.get_best_trial(metric, 'max', scope='all')
best_config = best_trial.config
best_checkpoint = analysis.get_best_checkpoint(best_trial, metric, 'max')
checkpoint_path = os.path.join(best_checkpoint.to_directory(), 'best_model.pth')

# Print best config summary
ft = best_config.get('ft_ext_param', {})
item_ft = ft.get('item_ft_ext_param', {})
user_ft = ft.get('user_ft_ext_param', {})
print(f'Best val HR@10: {best_trial.last_result["hit_ratio@10"]:.4f}')
print(f'Best val NDCG@10: {best_trial.last_result["ndcg@10"]:.4f}')
print(f'embedding_dim: {ft.get("embedding_dim")}')
print(f'batch_size: {best_config.get("batch_size")}')
print(f'loss: {best_config.get("loss_func_name")}')
print(f'optim: {best_config.get("optim_param", {}).get("optim")}')
print(f'lr: {best_config.get("optim_param", {}).get("lr")}')
print(f'wd: {best_config.get("optim_param", {}).get("wd")}')
print(f'neg_train: {best_config.get("neg_train")}')
print(f'item n_prototypes: {item_ft.get("n_prototypes")}')
print(f'item sim_proto_weight: {item_ft.get("sim_proto_weight")}')
print(f'item sim_batch_weight: {item_ft.get("sim_batch_weight")}')
print(f'user n_prototypes: {user_ft.get("n_prototypes")}')
print(f'user sim_proto_weight: {user_ft.get("sim_proto_weight")}')
print(f'user sim_batch_weight: {user_ft.get("sim_batch_weight")}')
print(f'\nCheckpoint: {checkpoint_path}')
print(f'\nRunning test...')

# Log to W&B and run test
model = 'user_item_proto'
dataset = 'hm_3_month'
seed = best_config.get('seed', 38210573)

wandb.login(key=WANDB_API_KEY)
wandb.init(project='protomf', group=f'{model}_{dataset}', config=best_config,
           name=f'{model}_{dataset}_s{seed}_test', force=True,
           job_type='test', tags=[model, dataset, f'seed:{seed}', 'hm_baseline'])

metric_values = start_testing(best_config, checkpoint_path)

print(f'\n=== Test Results ===')
for k, v in sorted(metric_values.items()):
    print(f'{k}: {v:.4f}')

wandb.finish()
