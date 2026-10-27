# dc09 SC.6 sizing — read the sc6_speed_ratio.sh cases: per-epoch wall (training + validation, Ray's
# time_this_iter_s for epochs 0 and 1; the epoch -1 initial-validation report carries set-up and is
# skipped), median over the concurrent trials; then old -> new ratios per model x dataset.
import csv
import glob
import os
import statistics

B = '/scratch/c7031336/smoke_runs/dc09/sc6_speed/ray_results'
rows = {}
for case in sorted(os.listdir(B)):
    times = []
    for p in glob.glob(os.path.join(B, case, '*', '*', 'progress.csv')):
        for r in csv.DictReader(open(p)):
            if r.get('epoch') not in (None, '') and int(float(r['epoch'])) >= 0:
                times.append(float(r['time_this_iter_s']))
    rows[case] = (statistics.median(times), len(times)) if times else (None, 0)
    print('%-48s median epoch %s s over %d trial-epochs' % (case, None if not times else round(rows[case][0]), len(times)))
print()
for case, (t_new, _) in rows.items():
    if case.startswith('new_'):
        old = 'old_' + case[4:]
        if old in rows and rows[old][0] and t_new:
            print('%-44s new/old = %.2f' % (case[4:], t_new / rows[old][0]))
