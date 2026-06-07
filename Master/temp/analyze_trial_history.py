"""Back-test lighter hyperopt configs against the known 100-sample winners.

Reads Master/temp/trial_history.csv. Answers:
  1) num_samples: best-of-first-K curve -> how few trials still land near-best.
  2) n_epochs / patience: epoch at which the WINNING trials peaked.
  3) ASHA: would harsher pruning (rung@4) have killed the eventual winner?
"""
import os, csv, statistics
from collections import defaultdict

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, 'trial_history.csv')

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        for k in ('best_val','val_at_grace4'):
            r[k] = float(r[k]) if r[k] not in ('','None') else None
        for k in ('best_epoch','last_epoch','n_logged'):
            r[k] = int(float(r[k])) if r[k] not in ('','None') else None
        rows.append(r)

by_group = defaultdict(list)
for r in rows:
    by_group[r['group']].append(r)

print('='*78)
print('1) num_samples back-test: best-of-first-K vs full 100-sample best')
print('='*78)
ks = [5,10,15,20,25,30,40,50,75,100]
# fraction of full-best reached, averaged across groups
agg = defaultdict(list)
for g, rs in sorted(by_group.items()):
    rs_ord = sorted(rs, key=lambda x: x['created'])  # TPE order
    full_best = max(x['best_val'] for x in rs_ord)
    n = len(rs_ord)
    line = [f'{g:28s} (n={n:3d}, best={full_best:.4f})']
    for k in ks:
        sub = rs_ord[:k]
        if not sub:
            continue
        bk = max(x['best_val'] for x in sub)
        frac = bk/full_best
        agg[k].append(frac)
        line.append(f'K{k}:{frac*100:5.1f}%')
    print('  '.join(line))
print('-'*78)
print('AVG % of full-best reached by first K trials:')
for k in ks:
    if agg[k]:
        print(f'  K={k:3d}: {statistics.mean(agg[k])*100:5.2f}%  (worst group {min(agg[k])*100:5.2f}%)')

print()
print('='*78)
print('2) epoch where WINNING trials peaked (informs n_epochs cap & patience)')
print('='*78)
# Per group: best_epoch of the top trial, and of top-10% trials
all_top_epochs = []
for g, rs in sorted(by_group.items()):
    rs_sorted = sorted(rs, key=lambda x: -x['best_val'])
    top = rs_sorted[0]
    topk = rs_sorted[:max(1,len(rs_sorted)//10)]
    eps = [x['best_epoch'] for x in topk]
    all_top_epochs.extend(eps)
    print(f'{g:28s} winner peak@{top["best_epoch"]:>3} (last {top["last_epoch"]:>3}) | top10% peak: med {statistics.median(eps):.0f} max {max(eps)}')
print('-'*78)
ate = sorted(all_top_epochs)
print(f'Across all groups, top-10% trials peak epoch: med {statistics.median(ate):.0f}, '
      f'p90 {ate[int(0.9*len(ate))-1]}, max {max(ate)}')

print()
print('='*78)
print('3) ASHA aggressiveness: rung@4 percentile of the eventual winner')
print('='*78)
for g, rs in sorted(by_group.items()):
    rsv = [x for x in rs if x['val_at_grace4'] is not None]
    if not rsv:
        print(f'{g:28s} no rung data'); continue
    winner = max(rsv, key=lambda x: x['best_val'])
    # winner's percentile rank by val_at_grace4 (higher=safer from pruning)
    worse = sum(1 for x in rsv if x['val_at_grace4'] < winner['val_at_grace4'])
    pct = worse/len(rsv)*100
    print(f'{g:28s} winner rung@4 percentile: {pct:5.1f}%  '
          f'(survives top-1/{max(1,round(1/((1-pct/100)+1e-9)))} cut)')
print('-'*78)
print('If winner sits high (>50%) at rung@4, aggressive ASHA (reduction_factor up,')
print('grace down) is SAFE. If low, harsh pruning risks killing the winner.')
