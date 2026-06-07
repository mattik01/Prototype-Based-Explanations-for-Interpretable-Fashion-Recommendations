"""Pull per-trial val-metric history from W&B for the fully-swept groups, to
back-test lighter ASHA/patience/num_samples configs. Writes a tidy CSV.

For each finished trial we record:
  group, name, created (TPE order), best_val, best_epoch, last_epoch,
  val_at_grace4 (val metric at epoch index 4 = ASHA first rung),
  n_logged.
"""
import os, csv, sys, time
import wandb

METRIC = 'hit_ratio@10'
TARGET_GROUPS = [
    'mf_ml-1m', 'acf_ml-1m', 'user_proto_ml-1m', 'item_proto_ml-1m', 'user_item_proto_ml-1m',
    'mf_amazon2014', 'acf_amazon2014', 'user_proto_amazon2014', 'item_proto_amazon2014', 'user_item_proto_amazon2014',
]
OUT = os.path.join(os.path.dirname(__file__), 'trial_history.csv')

api = wandb.Api(timeout=60)
ent = api.default_entity
rows = []
for g in TARGET_GROUPS:
    runs = api.runs(f'{ent}/protomf', filters={'group': g, 'state': 'finished'}, per_page=200)
    cnt = 0
    for r in runs:
        try:
            hist = r.history(keys=[METRIC, 'epoch'], samples=500, pandas=False)
        except Exception as e:
            continue
        vals = [(h.get('epoch'), h.get(METRIC)) for h in hist if h.get(METRIC) is not None]
        if not vals:
            continue
        best_epoch, best_val = max(vals, key=lambda x: x[1])
        last_epoch = max(e for e, _ in vals if e is not None) if any(e is not None for e, _ in vals) else None
        # val at first ASHA rung (epoch index 4); fall back to closest epoch >=4
        rung = None
        for e, v in sorted(vals, key=lambda x: (x[0] is None, x[0])):
            if e is not None and e >= 4:
                rung = v
                break
        rows.append({
            'group': g, 'name': r.name, 'created': r.created_at,
            'best_val': round(best_val, 5), 'best_epoch': best_epoch,
            'last_epoch': last_epoch, 'val_at_grace4': (round(rung,5) if rung is not None else ''),
            'n_logged': len(vals),
        })
        cnt += 1
    print(f'{g}: {cnt} finished trials pulled', flush=True)

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['group','name','created','best_val','best_epoch','last_epoch','val_at_grace4','n_logged'])
    w.writeheader()
    w.writerows(rows)
print(f'WROTE {len(rows)} rows -> {OUT}', flush=True)
