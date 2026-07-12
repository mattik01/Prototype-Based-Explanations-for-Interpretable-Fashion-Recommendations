# t10 — model-grade ml-1m item_features.csv (S0-build extension component d).
# Pins the built artifact against the S0.5-addendum audit facts (ml1m_feature_audit.py,
# gate-ratified 2026-07-12) END-TO-END through the new builders:
#   CSV shape/missingness -> build_feature_bags vocab/tokens/signatures -> fU bags builder
#   -> injection with the resolved canonical fields.
# Requires data/ml-1m/item_features.csv (gitignored; rebuild: python data/ml-1m/build_item_features.py).
import csv
import os
import sys
from collections import Counter

import numpy as np
import torch

from _harness import make_check, REPO_ROOT

from feature_extraction.feature_ids import (build_feature_bags, build_user_history_weights,
                                            resolve_canonical_fields, inject_feature_ids)

check, fails = make_check()

ML = os.path.join(REPO_ROOT, 'data', 'ml-1m')
if not os.path.exists(os.path.join(ML, 'item_features.csv')):
    print("[SKIP] data/ml-1m/item_features.csv not present — build it first "
          "(python data/ml-1m/build_item_features.py)")
    sys.exit(0)

# --- CSV shape + ratified missingness facts (S0.5 addendum §1/§3) ---
with open(os.path.join(ML, 'item_features.csv'), encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
check("csv: 3,125 rows (split items)", len(rows) == 3125, f"{len(rows)}")
check("csv: model-grade columns present",
      set(rows[0].keys()) == {'item_id', 'item', 'title', 'year', 'genres', 'tags'},
      f"{sorted(rows[0].keys())}")
genres_only = sum(1 for r in rows if r['tags'] == '')
check("csv: 94 genres-only rows (3.0% tail, ratified missingness policy)",
      genres_only == 94, f"{genres_only}")
check("csv: no empty-genres row (min 1 token/item holds)",
      all(r['genres'] != '' for r in rows), "")
check("csv: no movies.dat miss (audit superseded the '4 missing' display-build note)",
      all(r['title'] != '' for r in rows), "")

# --- bag builder on the real artifact: vocab + token stats (audit §1) ---
fields, layout = resolve_canonical_fields('ml-1m')
check("canonical fields for ml-1m = genres+tags, bags", fields == ['genres', 'tags'] and layout == 'bags')
bag_ids, bag_w, nf = build_feature_bags(ML, fields)
check("bags: V = 18 genres + 1,043 tags = 1,061", nf == 1061, f"{nf}")
tok = bag_w.sum(dim=1).numpy()
check("bags: max total tokens/item = 72 (== D_max, audit)",
      int(tok.max()) == 72 and bag_ids.shape == (3125, 72), f"max={int(tok.max())}, {tuple(bag_ids.shape)}")
check("bags: token stats match audit (mean 12.3, median 10, p10 3, p90 24, min 1)",
      abs(float(tok.mean()) - 12.3) < 0.05 and int(np.median(tok)) == 10
      and int(np.percentile(tok, 10)) == 3 and int(np.percentile(tok, 90)) == 24
      and int(tok.min()) == 1,
      f"mean={tok.mean():.2f} med={np.median(tok):.0f} p10={np.percentile(tok, 10):.0f} "
      f"p90={np.percentile(tok, 90):.0f} min={tok.min():.0f}")

# --- signature collisions (fI twin counterpart — audit §4, dc01's ml-1m inheritance) ---
sigs = Counter(frozenset(bag_ids[i, bag_w[i] > 0].tolist()) for i in range(bag_ids.shape[0]))
twins = sum(c for c in sigs.values() if c > 1)
check("signatures: 3,027 distinct / 118 twins (3.8%) / max class 32",
      len(sigs) == 3027 and twins == 118 and max(sigs.values()) == 32,
      f"distinct={len(sigs)}, twins={twins}, max={max(sigs.values())}")

# --- fU bags builder on the real train file (audit §5 basket facts) ---
hist_ids, hist_w, nf_u = build_user_history_weights(ML, fields, layout='bags')
check("fU bags: 6,034 users, shared vocab V=1,061",
      hist_ids.shape[0] == 6034 and nf_u == 1061, f"{tuple(hist_ids.shape)}, nf={nf_u}")
mass = hist_w.sum(dim=1)
# NOTE the mean is ~24, NOT the catalog's 12.3: baskets are interaction-weighted, and token
# count correlates with train popularity (rho=0.672, the S0.5-addendum must-mention channel) —
# users rate popular movies, popular movies are tag-richer. This check deliberately pins the
# gap as the channel's first live measurement (catalog-uniform 12.3 vs basket-weighted ~24).
check("fU bags: per-user vote mass = mean tokens/rated movie; basket-weighted mean ~24 "
      ">> catalog mean 12.3 (popularity coupling, rho=0.672)",
      bool((mass >= 1).all()) and bool((mass <= 72).all()) and 20.0 < float(mass.mean()) < 28.0,
      f"mass mean={float(mass.mean()):.2f}, min={float(mass.min()):.2f}, max={float(mass.max()):.2f}")

# --- injection e2e with the resolved spec (the exact run-time path) ---
spec = inject_feature_ids({'ft_type': 'lightfm',
                           'item_ft_ext_param': {'ft_type': 'lightfm',
                                                 'feature_fields': fields,
                                                 'feature_layout': layout,
                                                 'use_id_feature': False},
                           'user_ft_ext_param': {'ft_type': 'embedding'}}, ML)
it = spec['item_ft_ext_param']
check("inject e2e: lightfm_tags payload on real ml-1m (ids+weights (3125,72), nf=1061)",
      it['feature_ids'].shape == (3125, 72) and it['feature_weights'].shape == (3125, 72)
      and it['n_features'] == 1061, "")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
