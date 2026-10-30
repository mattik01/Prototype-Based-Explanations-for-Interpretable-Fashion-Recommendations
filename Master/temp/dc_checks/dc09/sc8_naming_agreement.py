# dc09 SC.8.4 — intrinsic vs post-hoc naming: agreement and catalogue frequency of the picked words (2026-10-30).
# Usage: python3 sc8_naming_agreement.py <dir> — <dir> holds hm_item_features.csv, ml_item_features.csv (copies of the
# scratch item_features.csv) and out/<ds>/pipeline/<model>/{cosine,lift}/prototype_names_{item,user}.csv from sc8_real_render.py.
import csv, sys, os, collections, statistics
SP = sys.argv[1]
def item_freq(ds):
    p = f'{SP}/hm_item_features.csv' if ds == 'hm_1_month' else f'{SP}/ml_item_features.csv'
    rows = list(csv.DictReader(open(p)))
    c = collections.Counter()
    for r in rows:
        if ds == 'hm_1_month':
            for f in ('department_name', 'product_type_name', 'section_name', 'colour_group_name', 'graphical_appearance_name'):
                c[r[f]] += 1
        else:
            for g in (r['genres'] or '').split('|') + (r['tags'] or '').split('|'):
                if g: c[g] += 1
    return c, len(rows)
def names(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[r['prototype']] = [d.split('=', 1)[1].rsplit('(score', 1)[0] for d in r['descriptors'].split('; ') if '=' in d]
    return out
for ds in ('hm_1_month', 'ml-1m'):
    c, n = item_freq(ds)
    allf = sorted(c.values())
    pct = lambda v: sum(x <= v for x in allf) / len(allf)
    base = f'{SP}/out/{ds}/pipeline'
    for m, intr, post in (('feature_user_item_proto', 'cosine', 'lift'),):
        for side in ('item', 'user'):
            pi, pp = f'{base}/{m}/{intr}/prototype_names_{side}.csv', f'{base}/{m}/{post}/prototype_names_{side}.csv'
            if not (os.path.exists(pi) and os.path.exists(pp)):
                print(ds, m, side, 'missing'); continue
            I, P = names(pi), names(pp)
            common = [k for k in I if k in P]
            ov = [len(set(I[k]) & set(P[k])) for k in common]
            fi = [c.get(w, 0) / n for k in common for w in I[k]]
            fp = [c.get(w, 0) / n for k in common for w in P[k]]
            distinct_i = len({tuple(sorted(I[k])) for k in common})
            print(f"{ds} {side}: K={len(common)}; ≥1 shared word {sum(o > 0 for o in ov)}/{len(common)}; mean shared {statistics.mean(ov):.2f}/3; "
                  f"distinct names intrinsic {distinct_i}, post-hoc {len({tuple(sorted(P[k])) for k in common})}; "
                  f"median item-share of picked words: intrinsic {statistics.median(fi):.4f} (pctile {pct(statistics.median(fi)*n):.2f}), post-hoc {statistics.median(fp):.4f} (pctile {pct(statistics.median(fp)*n):.2f}); "
                  f"vocab median {statistics.median(allf)/n:.4f}")
