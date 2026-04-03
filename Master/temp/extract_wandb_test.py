"""Fetch test metrics for user_item_proto amazon2014 from W&B."""
import wandb
api = wandb.Api()

# List runs in group
runs = api.runs('protomf', filters={
    'group': 'user_item_proto_amazon2014',
    'jobType': 'test'
})

for r in runs:
    print(f'Run: {r.name}, job_type: {r.job_type}, state: {r.state}')
    summary = r.summary
    for k in sorted(summary.keys()):
        if 'hit_ratio' in k or 'ndcg' in k:
            print(f'  {k}: {summary[k]:.4f}')
