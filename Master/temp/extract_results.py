"""Extract best trial results for user_item_proto on amazon2014."""
import os
from ray.tune import ExperimentAnalysis

result_dir = os.path.expanduser('~/ray_results/user_item_proto_amazon2014_DE_38210573_2026-3-31_13-15-21.491510')
analysis = ExperimentAnalysis(result_dir)

metric = 'hit_ratio@10'
best_trial = analysis.get_best_trial(metric, 'max', scope='all')
best_config = best_trial.config

print('=== Best Trial Config ===')
ft = best_config.get('ft_ext_param', {})
item_ft = ft.get('item_ft_ext_param', {})
user_ft = ft.get('user_ft_ext_param', {})

print(f'embedding_dim: {ft.get("embedding_dim")}')
print(f'batch_size: {best_config.get("batch_size")}')
print(f'loss: {best_config.get("loss_func_name")}')
print(f'optim: {best_config.get("optim_param", {}).get("optim")}')
print(f'lr: {best_config.get("optim_param", {}).get("lr")}')
print(f'wd: {best_config.get("optim_param", {}).get("wd")}')
print(f'neg_train: {best_config.get("neg_train")}')
print(f'train_neg_strategy: {best_config.get("train_neg_strategy")}')
print(f'item n_prototypes: {item_ft.get("n_prototypes")}')
print(f'item sim_proto_weight: {item_ft.get("sim_proto_weight")}')
print(f'item sim_batch_weight: {item_ft.get("sim_batch_weight")}')
print(f'user n_prototypes: {user_ft.get("n_prototypes")}')
print(f'user sim_proto_weight: {user_ft.get("sim_proto_weight")}')
print(f'user sim_batch_weight: {user_ft.get("sim_batch_weight")}')

print()
print('=== Best Trial Val Metrics ===')
best_result = best_trial.last_result
for k, v in sorted(best_result.items()):
    if 'hit_ratio' in k or 'ndcg' in k:
        print(f'{k}: {v:.4f}')

print()
print(f'Completed trials: {len([t for t in analysis.trials if t.status == "TERMINATED"])}')
print(f'Total trials: {len(analysis.trials)}')
