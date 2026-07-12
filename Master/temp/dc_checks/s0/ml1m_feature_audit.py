"""ml-1m feature audit — S0.5 addendum measurement script (2026-07-12).

Regenerates every number quoted in the S0.5 ml-1m addendum / the ml-1m
expansion scoping discussion, from the raw artifacts:

  - data/ml-1m/item_ids.csv           (the split's item -> movieId map)
  - data/ml-1m/movies.dat             (GroupLens ml-1m catalog: genres)
  - data/ml-1m/tag-genome/            (2014 Tag Genome: tag_relevance.dat,
                                       tags.dat, movies.dat)
  - data/ml-1m/listening_history_train.csv

Field-set decision under audit (charter C5 amendment 2026-07-12): B5's
published MovieLens recipe — genres + Tag-Genome tags at relevance >= 0.8,
indicator bags, no year.

Scrutiny working basis only — no thesis quotes (F-DC01-06 restriction).
Run: conda activate protomf && python ml1m_feature_audit.py
"""
import collections
import csv
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
ML = os.path.join(REPO, 'data', 'ml-1m')
GENOME = os.path.join(ML, 'tag-genome')
THRESHOLD = 0.8  # B5 §4.1's published relevance threshold — theirs, not ours


def main():
    ids = [r['item'] for r in csv.DictReader(open(os.path.join(ML, 'item_ids.csv')))]
    split_mids = set(ids)
    print(f"split items: {len(ids)} (unique movieIds: {len(split_mids)})")

    genres = {}
    for line in open(os.path.join(ML, 'movies.dat'), encoding='latin-1'):
        mid, _title, g = line.rstrip('\n').split('::')
        genres[mid] = g.split('|') if g else []
    no_match = [m for m in split_mids if m not in genres]
    gtok = [len(genres.get(m, [])) for m in split_mids]
    print(f"items with NO ml-1m movies.dat match: {len(no_match)}")
    print(f"genre tokens/item: mean {np.mean(gtok):.2f} median {np.median(gtok):.0f} "
          f"min {min(gtok)} max {max(gtok)}; zero-genre items: {sum(1 for x in gtok if x == 0)}")

    genome_mids = set()
    for line in open(os.path.join(GENOME, 'movies.dat'), encoding='latin-1'):
        genome_mids.add(line.split('\t')[0])
    covered = split_mids & genome_mids
    print(f"\ngenome movie list: {len(genome_mids)}; split coverage: "
          f"{len(covered)}/{len(split_mids)} = {100 * len(covered) / len(split_mids):.1f}%")

    tags08 = collections.Counter()
    vocab = set()
    for line in open(os.path.join(GENOME, 'tag_relevance.dat')):
        mid, tid, rel = line.rstrip('\n').split('\t')
        if mid in split_mids and float(rel) >= THRESHOLD:
            tags08[mid] += 1
            vocab.add(tid)
    counts = [tags08.get(m, 0) for m in covered]
    zero_in = sum(1 for c in counts if c == 0)
    print(f"unique tag vocab @{THRESHOLD}: {len(vocab)}")
    print(f"tags@{THRESHOLD} per covered movie: mean {np.mean(counts):.1f} "
          f"median {np.median(counts):.0f} p90 {np.percentile(counts, 90):.0f} max {max(counts)}")
    print(f"covered-but-ZERO-tags@{THRESHOLD}: {zero_in}/{len(covered)}")

    eff_poor = (len(split_mids) - len(covered)) + zero_in
    print(f"\nEFFECTIVE: full rows (genres + >=1 tag): {len(split_mids) - eff_poor} "
          f"({100 * (len(split_mids) - eff_poor) / len(split_mids):.1f}%); "
          f"genres-only rows: {eff_poor} ({100 * eff_poor / len(split_mids):.1f}%)")

    tot = [len(genres.get(m, [])) + tags08.get(m, 0) for m in split_mids]
    print(f"TOTAL tokens/item: mean {np.mean(tot):.1f} median {np.median(tot):.0f} "
          f"p10 {np.percentile(tot, 10):.0f} p90 {np.percentile(tot, 90):.0f} max {max(tot)}")

    item_by_iid = {r['item_id']: r['item']
                   for r in csv.DictReader(open(os.path.join(ML, 'item_ids.csv')))}
    n_int = n_poor = 0
    with open(os.path.join(ML, 'listening_history_train.csv')) as f:
        rd = csv.DictReader(f)
        icol = 'item_id' if 'item_id' in rd.fieldnames else rd.fieldnames[1]
        for r in rd:
            n_int += 1
            mid = item_by_iid.get(r[icol])
            if mid is None or mid not in covered or tags08.get(mid, 0) == 0:
                n_poor += 1
    print(f"\ntrain interactions: {n_int}; on genres-only items: "
          f"{n_poor} ({100 * n_poor / n_int:.2f}%)")


if __name__ == '__main__':
    main()
