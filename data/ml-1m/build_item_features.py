"""Build the MODEL-GRADE `item_features.csv` for ml-1m: the processed `item_ids.csv` joined
with the raw MovieLens-1M `movies.dat` catalog (genres) and the Tag-Genome relevance table
(tags at relevance >= 0.8 — B5 §4.1's published threshold, charter C5 as amended 2026-07-12).

Output columns:

    item_id, item, title, year, genres, tags

- `genres` / `tags` are the ml-1m canonical feature fields (indicator bags, pipe-joined —
  the explanations naming layer's multi_value_sep convention), consumed by
  `feature_extraction.feature_ids.build_feature_bags`. A movie without genome coverage or
  without tags at the threshold gets an EMPTY `tags` cell — a legal short bag (the ratified
  S0.5-addendum missingness policy: ~3% genres-only tail, disclosed, no imputation).
  Tag names within a cell are sorted (deterministic artifact).
- `title` / `year` are display-grade extras (never model fields — `title` is identity-like,
  the product_code analogue; `year` excluded per C5, LightFM used none).
  `utilities.explanations.items_info.load_items_info` auto-prefers `item_features.csv`, so
  they flow into the per-prototype top-K CSVs via `get_top_k_items`.

The printed report re-states the S0.5-addendum audit facts (coverage, vocab, genres-only
count) so a rebuild is verifiable against `Master/temp/dc_checks/s0/ml1m_feature_audit.py`.

Usage (stdlib only, no pandas needed):
    python build_item_features.py            # ./item_ids.csv, ./movies.dat, ./tag-genome
    python build_item_features.py --ids ./item_ids.csv --movies ./movies.dat \
        --genome ./tag-genome --threshold 0.8 --out ./item_features.csv
"""
import argparse
import csv
import os
import re

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _load_movies(movies_path: str) -> dict:
    """movieId (str) -> (title, genres). MovieLens-1M movies.dat is latin-1."""
    catalog = {}
    with open(movies_path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            movie_id, title, genres = line.split("::")
            catalog[movie_id] = (title, genres)
    return catalog


def _load_genome_tags(genome_dir: str, movie_ids: set, threshold: float) -> dict:
    """movieId (str) -> sorted list of tag NAMES with relevance >= threshold.

    Streams `tag_relevance.dat` (movieId \\t tagId \\t relevance, ~11M rows); names from
    `tags.dat` (tagId \\t tag \\t count). Only movies in `movie_ids` are collected.
    """
    tags_path = os.path.join(genome_dir, "tags.dat")
    rel_path = os.path.join(genome_dir, "tag_relevance.dat")
    for p in (tags_path, rel_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Tag-Genome file missing: {p} — the model-grade ml-1m item_features.csv "
                f"requires the 2014 Tag-Genome download (S0-build extension precondition)")

    tag_name = {}
    with open(tags_path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tag_id, name, _count = line.split("\t")
            tag_name[tag_id] = name

    by_movie = {}
    with open(rel_path, encoding="latin-1") as f:
        for line in f:
            mid, tid, rel = line.rstrip("\n").split("\t")
            if mid in movie_ids and float(rel) >= threshold:
                by_movie.setdefault(mid, []).append(tag_name[tid])
    return {mid: sorted(names) for mid, names in by_movie.items()}


def build(ids_path: str, movies_path: str, genome_dir: str, threshold: float,
          out_path: str) -> int:
    catalog = _load_movies(movies_path)

    with open(ids_path, encoding="utf-8", newline="") as f:
        id_rows = list(csv.DictReader(f))
    movie_ids = {r["item"] for r in id_rows}

    genome = _load_genome_tags(genome_dir, movie_ids, threshold)

    rows = []
    missing = 0
    genres_only = 0
    vocab = set()
    max_tokens = 0
    for r in id_rows:
        movie_id = r["item"]
        title, genres = catalog.get(movie_id, ("", ""))
        if title == "":
            missing += 1
        tags = genome.get(movie_id, [])
        if not tags:
            genres_only += 1
        vocab.update(tags)
        n_tok = (len(genres.split("|")) if genres else 0) + len(tags)
        max_tokens = max(max_tokens, n_tok)
        rows.append({
            "item_id": r["item_id"],
            "item": movie_id,
            "title": title,
            "year": _extract_year(title),
            "genres": genres,
            "tags": "|".join(tags),
        })

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["item_id", "item", "title", "year", "genres", "tags"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  genome movies covered: {len(genome)}/{len(rows)}; "
          f"genres-only rows (0 tags @{threshold}): {genres_only} "
          f"({100 * genres_only / len(rows):.1f}%)")
    print(f"  tag vocab @{threshold}: {len(vocab)}; max total tokens/item: {max_tokens}")
    if missing:
        print(f"  warning: {missing} item_id(s) had no movies.dat match (title/genres left blank)")
    return len(rows)


def _extract_year(title: str) -> str:
    m = _YEAR_RE.search(title)
    return m.group(1) if m else ""


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", default=os.path.join(here, "item_ids.csv"))
    p.add_argument("--movies", default=os.path.join(here, "movies.dat"))
    p.add_argument("--genome", default=os.path.join(here, "tag-genome"))
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Tag-Genome relevance threshold (B5 §4.1's published 0.8)")
    p.add_argument("--out", default=os.path.join(here, "item_features.csv"))
    args = p.parse_args()

    n = build(args.ids, args.movies, args.genome, args.threshold, args.out)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
