"""Phase 4.0 feature study, part E: engineered-feature probes on the V1 window.

From raw transactions (filtered to the hm_1_month window + V1 item set):
  E1: per-item price stats; price-decile homogeneity lift + popularity eta^2
      (same probes as part D, so price is comparable to the categorical features)
  E2: repeat-purchase concentration (article-level and product_code/variant-level)
  E3: recency structure (per-week item activity within the window)

Outputs CSVs to Master/experiments/hm_features/feature_analysis/ and prints a summary.
Chunk-reads the 31M-row raw file; local, a few minutes.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "hm_1_month"
RAW = REPO / "data" / "hm" / "raw"
OUT = REPO / "Master" / "experiments" / "hm_features" / "feature_analysis"
os.makedirs(OUT, exist_ok=True)


def load_window_transactions():
    items = pd.read_csv(DATA / "item_features.csv", dtype={"item": str}, usecols=["item_id", "item"])
    train_dates = pd.read_csv(DATA / "listening_history_train.csv", usecols=["t_dat"])["t_dat"]
    start = train_dates.min()
    keep = set(items["item"])
    chunks = []
    for chunk in pd.read_csv(
        RAW / "transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id", "price"],
        dtype={"article_id": str},
        chunksize=2_000_000,
    ):
        chunk = chunk[chunk["t_dat"] >= start]
        if chunk.empty:
            continue
        chunk["article_id"] = chunk["article_id"].str.zfill(10)
        chunks.append(chunk[chunk["article_id"].isin(keep)])
    tx = pd.concat(chunks, ignore_index=True)
    arts = pd.read_csv(RAW / "articles.csv", dtype={"article_id": str, "product_code": str},
                       usecols=["article_id", "product_code"])
    arts["article_id"] = arts["article_id"].str.zfill(10)
    tx = tx.merge(arts, on="article_id", how="left")
    print(f"window start: {start}, raw window transactions on V1 items: {len(tx):,}")
    return tx


def homogeneity_lift(tx, col):
    cnt = tx.groupby(["customer_id", col], observed=True).size()
    per_user = cnt.groupby(level=0).sum()
    same = (cnt * (cnt - 1)).groupby(level=0).sum()
    mask = per_user >= 2
    within = (same[mask] / (per_user[mask] * (per_user[mask] - 1))).mean()
    p = tx[col].value_counts(normalize=True)
    return within, (p ** 2).sum()


def part_e1(tx):
    stats = tx.groupby("article_id")["price"].agg(["mean", "std", "min", "max", "count"])
    stats.to_csv(OUT / "price_stats.csv")
    decile = pd.qcut(stats["mean"], 10, labels=False, duplicates="drop")
    tx = tx.merge(decile.rename("price_decile"), left_on="article_id", right_index=True)
    within, base = homogeneity_lift(tx, "price_decile")
    pop = np.log1p(stats["count"])
    grp = pop.groupby(decile)
    eta2 = ((grp.size() * (grp.mean() - pop.mean()) ** 2).sum()
            / ((pop - pop.mean()) ** 2).sum())
    return {"feature": "price_decile(10)", "within_user_match": within,
            "baseline_match": base, "homogeneity_lift": within / base,
            "popularity_eta2": eta2}


def part_e2(tx):
    by_art = tx.groupby(["customer_id", "article_id"]).size()
    by_code = tx.groupby(["customer_id", "product_code"])["article_id"].agg(["size", "nunique"])
    return {
        "user_article_pairs": len(by_art),
        "repeat_article_pairs_pct": 100 * (by_art > 1).mean(),
        "tx_that_are_article_repeats_pct": 100 * (by_art - 1).sum() / len(tx),
        "user_code_pairs": len(by_code),
        "repeat_code_pairs_pct": 100 * (by_code["size"] > 1).mean(),
        "multi_variant_code_pairs_pct": 100 * (by_code["nunique"] > 1).mean(),
    }


def part_e3(tx):
    week = pd.to_datetime(tx["t_dat"]).dt.isocalendar().week.astype(int)
    active = tx.groupby([week, "article_id"], observed=True).size().rename("n").reset_index()
    weekly = active.groupby("week")["article_id"].nunique()
    weekly.to_csv(OUT / "weekly_active_items.csv")
    last_week = week.max()
    alive_last = set(active.loc[active["week"] == last_week, "article_id"])
    cov = tx.loc[week == last_week, "article_id"].isin(alive_last).mean()  # trivially 1; use prior week
    prior = set(active.loc[active["week"] == last_week - 1, "article_id"])
    cov_prior = tx.loc[week == last_week, "article_id"].isin(prior).mean()
    return {
        "weeks_in_window": int(week.nunique()),
        "items_active_per_week_mean": float(weekly.mean()),
        "items_total": int(tx["article_id"].nunique()),
        "last_week_tx_covered_by_prior_week_items_pct": 100 * cov_prior,
    }


def main():
    tx = load_window_transactions()
    e1 = part_e1(tx)
    probes = pd.DataFrame([e1])
    probes.to_csv(OUT / "engineered_probes.csv", index=False)
    print("\n== E1: price as a feature (compare to part D table) ==")
    print(probes.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    e2 = part_e2(tx)
    print("\n== E2: repeat-purchase structure (raw, pre-dedup) ==")
    for k, v in e2.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")
    e3 = part_e3(tx)
    print("\n== E3: recency / weekly item activity ==")
    for k, v in e3.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")
    pd.DataFrame([{**e2, **e3}]).to_csv(OUT / "repeat_recency_stats.csv", index=False)
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
