"""Phase 4.0 feature study, parts B-D (see Master/temp/feature_study_4_0_plan.md).

Analyzes the hm_1_month (V1) item set:
  B: per-feature stats (cardinality, entropy, tail shape)
  C: pairwise NMI + functional-dependency (hierarchy) structure
  D: within-user homogeneity lift + popularity eta^2 per feature

Outputs CSVs to Master/experiments/hm_features/feature_analysis/ and prints a summary.
Runs locally in minutes (13.7K items / 623K train rows).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "hm_1_month"
RAW = REPO / "data" / "hm" / "raw"
OUT = REPO / "Master" / "experiments" / "hm_features" / "feature_analysis"
os.makedirs(OUT, exist_ok=True)

EXTRACTED = [
    "product_group_name", "product_type_name", "graphical_appearance_name",
    "colour_group_name", "perceived_colour_master_name", "index_group_name",
    "garment_group_name", "section_name", "department_name",
]
EXTRA = ["product_code", "perceived_colour_value_name"]  # not in item_features.csv yet


def load_items():
    feats = pd.read_csv(DATA / "item_features.csv", dtype={"item": str})
    arts = pd.read_csv(
        RAW / "articles.csv",
        dtype={"article_id": str, "product_code": str},
        usecols=["article_id"] + EXTRA,
    )
    arts["article_id"] = arts["article_id"].str.zfill(10)
    feats = feats.merge(arts, left_on="item", right_on="article_id", how="left")
    cols = EXTRACTED + EXTRA
    feats[cols] = feats[cols].fillna("__NA__")
    return feats.set_index("item_id"), cols


def part_b(items, cols):
    rows = []
    n = len(items)
    for c in cols:
        vc = items[c].value_counts()
        p = vc / n
        entropy = -(p * np.log(p)).sum()
        rows.append({
            "feature": c,
            "n_unique": len(vc),
            "missing_pct": 100 * (items[c] == "__NA__").mean(),
            "effective_n_values": np.exp(entropy),
            "top1_share_pct": 100 * p.iloc[0],
            "top10_share_pct": 100 * p.iloc[:10].sum(),
            "values_under_5_items_pct": 100 * (vc < 5).mean(),
        })
    df = pd.DataFrame(rows).sort_values("n_unique")
    df.to_csv(OUT / "feature_stats.csv", index=False)
    return df


def part_c(items, cols):
    nmi = pd.DataFrame(index=cols, columns=cols, dtype=float)
    fd = pd.DataFrame(index=cols, columns=cols, dtype=float)  # fd[a][b] = P(predict b from a)
    n = len(items)
    for a in cols:
        for b in cols:
            if a == b:
                nmi.loc[a, b] = 1.0
                fd.loc[a, b] = 1.0
                continue
            if pd.isna(nmi.loc[a, b]):
                v = normalized_mutual_info_score(items[a], items[b])
                nmi.loc[a, b] = nmi.loc[b, a] = v
            fd.loc[a, b] = items.groupby(a, observed=True)[b].agg(
                lambda s: s.value_counts().iloc[0]
            ).sum() / n
    nmi.to_csv(OUT / "nmi_matrix.csv")
    fd.to_csv(OUT / "functional_dependency.csv")
    return nmi, fd


def part_d(items, cols):
    train = pd.read_csv(DATA / "listening_history_train.csv", usecols=["user_id", "item_id"])
    pop = train["item_id"].value_counts()
    rows = []
    for c in cols:
        f = train["item_id"].map(items[c])
        # D1: within-user same-value match prob vs interaction-weighted baseline
        cnt = pd.DataFrame({"u": train["user_id"], "v": f}).groupby(["u", "v"], observed=True).size()
        per_user_n = cnt.groupby(level=0).sum()
        pair_same = (cnt * (cnt - 1)).groupby(level=0).sum()
        pair_all = per_user_n * (per_user_n - 1)
        mask = per_user_n >= 2
        within = (pair_same[mask] / pair_all[mask]).mean()
        p = f.value_counts(normalize=True)
        baseline = (p ** 2).sum()
        # D2: eta^2 of log-popularity across feature values (item-level)
        ipop = np.log1p(pop.reindex(items.index).fillna(0))
        grp = ipop.groupby(items[c], observed=True)
        ss_between = (grp.size() * (grp.mean() - ipop.mean()) ** 2).sum()
        ss_total = ((ipop - ipop.mean()) ** 2).sum()
        rows.append({
            "feature": c,
            "within_user_match": within,
            "baseline_match": baseline,
            "homogeneity_lift": within / baseline,
            "popularity_eta2": ss_between / ss_total,
        })
    df = pd.DataFrame(rows).sort_values("homogeneity_lift", ascending=False)
    df.to_csv(OUT / "discriminative_power.csv", index=False)
    return df


def main():
    items, cols = load_items()
    print(f"V1 items: {len(items)}, features analyzed: {len(cols)}")
    b = part_b(items, cols)
    print("\n== B: per-feature stats ==")
    print(b.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    nmi, fd = part_c(items, cols)
    print("\n== C: strongest redundancies (NMI > 0.5, off-diagonal) ==")
    pairs = nmi.where(np.triu(np.ones(nmi.shape, bool), 1)).stack()
    print(pairs[pairs > 0.5].sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n== C: near-functional dependencies (FD > 0.95, a->b) ==")
    fpairs = fd.stack()
    fpairs = fpairs[(fpairs > 0.95) & (fpairs < 1.0)]
    print(fpairs.sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
    d = part_d(items, cols)
    print("\n== D: discriminative power (within-user homogeneity lift, popularity eta^2) ==")
    print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
