"""Build `item_features.csv` for ml-1m by joining the processed `item_ids.csv`
with the raw MovieLens-1M `movies.dat` catalog.

`item_ids.csv` (produced by `movielens_splitter.py`) maps the remapped 0-based
`item_id` to the original MovieLens `item` (movieId). `movies.dat` holds
`movieId::title::genres` (latin-1). We left-join on movieId so every item_id is
retained, and emit `item_features.csv` with the columns:

    item_id, item, title, year, genres

`utilities.explanations.items_info.load_items_info` auto-prefers `item_features.csv`
over the minimal `item_ids.csv`, so the extra columns flow straight into the
per-prototype top-K CSVs via `get_top_k_items` (which keeps every non-id column).
Purely a readability/interpretation aid — the CF model itself ignores these features.

Usage (stdlib only, no pandas needed):
    python build_item_features.py            # uses ./item_ids.csv and ./movies.dat
    python build_item_features.py --ids ./item_ids.csv --movies ./movies.dat --out ./item_features.csv
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


def _extract_year(title: str) -> str:
    m = _YEAR_RE.search(title)
    return m.group(1) if m else ""


def build(ids_path: str, movies_path: str, out_path: str) -> int:
    catalog = _load_movies(movies_path)

    rows = []
    missing = 0
    with open(ids_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item_id = r["item_id"]
            movie_id = r["item"]
            title, genres = catalog.get(movie_id, ("", ""))
            if title == "":
                missing += 1
            rows.append({
                "item_id": item_id,
                "item": movie_id,
                "title": title,
                "year": _extract_year(title),
                "genres": genres,
            })

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "item", "title", "year", "genres"])
        writer.writeheader()
        writer.writerows(rows)

    if missing:
        print(f"  warning: {missing} item_id(s) had no movies.dat match (title/genres left blank)")
    return len(rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", default=os.path.join(here, "item_ids.csv"))
    p.add_argument("--movies", default=os.path.join(here, "movies.dat"))
    p.add_argument("--out", default=os.path.join(here, "item_features.csv"))
    args = p.parse_args()

    n = build(args.ids, args.movies, args.out)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
