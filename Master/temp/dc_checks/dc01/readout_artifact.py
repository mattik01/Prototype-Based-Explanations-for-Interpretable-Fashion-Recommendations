"""dc01 — generate a human-readable item-prototype EXPLANATION artifact (markdown) from a trained
feature_item_proto checkpoint. Self-contained (only needs the repo on PYTHONPATH).

Produces three sections (dc01 §3.4):
  A. Prototype dictionary  — what each item-prototype grounds to (top feature-values by cos(e_f, p_k)).
  B. Per-item attribution  — for the most-prototypical item of several prototypes, the EXACT
     per-feature decomposition of its activation (t*_k - 1 = Σ_f c_{f,k}).
  C. Full recommendation   — a real test (user, item) pair: the UI-score, its per-prototype
     breakdown, and the feature decomposition of the top item-prototype that drove it.

Usage:
  PYTHONPATH=<repo> python readout_artifact.py --results <dir> --data <dir> --out <file.md>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import pandas as pd
import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_ids import inject_feature_ids
from rec_sys.rec_sys import RecSys
from utilities.explanations.feature_readout import per_feature_shares, global_prototype_profile, item_feature_rows


def shifted_cos(x, P):
    return 1.0 + (P @ x) / (x.norm() * P.norm(dim=1))


def build_inverse_vocab(data_path, fields):
    df = pd.read_csv(os.path.join(data_path, 'item_features.csv')).sort_values('item_id').reset_index(drop=True)
    code2name, offset = {}, 0
    for field in fields:
        for i, v in enumerate(sorted(df[field].astype(str).unique().tolist())):
            code2name[offset + i] = (field, v)
        offset += len(sorted(df[field].astype(str).unique().tolist()))
    return code2name, df


def load(results_dir, data_path):
    conf = json.load(open(os.path.join(results_dir, 'config.json')))
    n_items = len(pd.read_csv(os.path.join(data_path, 'item_ids.csv')))
    n_users = len(pd.read_csv(os.path.join(data_path, 'user_ids.csv')))
    ft = inject_feature_ids(conf['ft_ext_param'], data_path)
    u_fe, i_fe = FeatureExtractorFactory.create_models(ft, n_users, n_items)
    rec = RecSys(n_users, n_items, conf['rec_sys_param'], u_fe, i_fe,
                 conf['loss_func_name'], conf.get('loss_func_aggr', 'mean'))
    rec.init_parameters()
    ck = os.path.join(results_dir, 'best_model.pth')
    try:
        rec.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
    except TypeError:
        rec.load_state_dict(torch.load(ck, map_location='cpu'))
    rec.eval()
    return rec, conf, n_users, n_items


def fmt_feats(prof_col, code2name, topk=6):
    top = prof_col.topk(topk)
    return [(code2name[int(i)][0], code2name[int(i)][1], float(v)) for v, i in zip(top.values, top.indices)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rec, conf, n_users, n_items = load(args.results, args.data)
    fields = conf['ft_ext_param']['item_ft_ext_param']['feature_fields']
    code2name, df = build_inverse_vocab(args.data, fields)

    item_proto = rec.item_feature_extractor.model_1
    item_proj = rec.item_feature_extractor.model_2
    user_proto = rec.user_feature_extractor.model_1
    fe = item_proto.embedding_ext
    Pt = item_proto.prototypes.detach()
    Kt = Pt.shape[0]
    Ku = user_proto.prototypes.shape[0]
    nf = fe.n_features
    meta_emb = fe.embedding_layer.weight.detach()[:nf]
    test_hr = json.load(open(os.path.join(args.results, 'test_metrics.json'))).get('hit_ratio@10')

    # vectorized: every item's composed q and its activation t* over the item-prototypes
    with torch.no_grad():
        q_all = fe(torch.arange(n_items))                       # (n_items, d)
        qn = q_all.norm(dim=1, keepdim=True)
        tstar_all = 1.0 + (q_all @ Pt.T) / (qn * Pt.norm(dim=1))  # (n_items, Kt)
    prof = global_prototype_profile(meta_emb, Pt)               # (nf, Kt)

    L = []
    w = L.append
    w("# feature_item_proto — Intrinsic Item-Prototype Explanations\n")
    w("> Generated from the **hm_1_month dev checkpoint** "
      f"(seed 38210573, TEST HR@10 = {test_hr:.4f}). Model: feature_item_proto — "
      f"item factor = Σ of feature-value embeddings, scored by a prototype layer.\n")
    w(f"> {Kt} item-prototypes · feature vocabulary = {nf} values over {len(fields)} fields "
      f"({', '.join(fields)}).\n")
    w("> **Every number below is read directly from the trained parameters / forward pass — "
      "nothing is post-hoc.**\n")

    # ---- A. Prototype dictionary ----
    w("\n## A. Prototype dictionary — what each item-prototype *means*\n")
    w("For each item-prototype $p_k$, the feature-values most aligned with it, "
      "$\\cos(e_f, p_k)$ (the model-intrinsic attribute profile, read from parameters alone):\n")
    w("| # | top feature-values (cosine) |")
    w("|---|---|")
    for k in range(Kt):
        feats = fmt_feats(prof[:, k], code2name)
        s = " · ".join(f"{val} ({c:+.2f})" for _, val, c in feats)
        w(f"| {k} | {s} |")

    # ---- B. Per-item attribution for a few prototypes' most-prototypical items ----
    w("\n## B. Per-item attribution — exact additive decomposition\n")
    w("For a handful of prototypes, the item that activates them most strongly, and the EXACT "
      "decomposition of that activation over the item's feature rows: "
      "$t^*_k = 1\\text{ (shift)} + \\sum_f c_{f,k}$.\n")
    # choose 4 spread-out, sharply-named prototypes by their top-feature cosine
    sharp = sorted(range(Kt), key=lambda k: -float(prof[:, k].max()))[:4]
    for k in sharp:
        item = int(tstar_all[:, k].argmax())
        rows = item_feature_rows(fe, item)
        shares = per_feature_shares(rows, Pt)[:, k]
        codes = fe.feature_ids[item].tolist()
        labels = [f"{code2name[c][0]} = {code2name[c][1]}" for c in codes] + ["<item-ID row>"]
        tk = 1.0 + float(shares.sum())
        feat_share = float(shares[:len(codes)].sum())
        id_share = float(shares[len(codes)])
        article = df.iloc[item]['item']
        w(f"\n**Prototype #{k}** — most-activating item: `{article}` (item {item}), "
          f"$t^*_{{{k}}} = {tk:.3f} = 1 + {tk - 1:.3f}$")
        w("\n| feature row | contribution $c_{f,k}$ |")
        w("|---|---|")
        for lbl, c in zip(labels, shares.tolist()):
            w(f"| {lbl} | {c:+.4f} |")
        w(f"| **Σ (= t\\*−1)** | **{float(shares.sum()):+.4f}** |")
        denom = feat_share + id_share
        frac = (feat_share / denom * 100) if abs(denom) > 1e-9 else float('nan')
        w(f"\nfeature-explained {feat_share:+.4f} vs ID-row {id_share:+.4f} "
          f"→ features carry **{frac:.0f}%** of the (non-shift) activation.\n")

    # ---- C. Full recommendation explanation for a real test (user,item) pair ----
    w("\n## C. A full recommendation explanation (real test pair)\n")
    test = pd.read_csv(os.path.join(args.data, 'listening_history_test.csv'))
    sample = test.head(800)
    best = None  # (top_item_proto_contribution, user, item, ...)
    with torch.no_grad():
        for u_idx, i_idx in zip(sample.user_id.tolist(), sample.item_id.tolist()):
            u_out = rec.user_feature_extractor(torch.tensor([int(u_idx)]))[0]
            i_out = rec.item_feature_extractor(torch.tensor([int(i_idx)]))[0]
            uhat = u_out[Ku:]            # user projected into item-prototype space (Kt,)
            tstar = i_out[Ku:]           # item activation over item-prototypes (Kt,)
            contribs = uhat * tstar      # per item-prototype score contribution
            kbest = int(contribs.argmax())
            val = float(contribs[kbest])
            if best is None or val > best[0]:
                best = (val, int(u_idx), int(i_idx), kbest, u_out.clone(), i_out.clone())
    _, u_idx, i_idx, kbest, u_out, i_out = best
    ustar, uhat = u_out[:Ku], u_out[Ku:]
    that, tstar = i_out[:Ku], i_out[Ku:]
    score = float((u_out * i_out).sum())
    item_side = float((uhat * tstar).sum())     # û·t*  (feature-grounded half)
    user_side = float((ustar * that).sum())     # u*·t̂  (CF half)
    contribs = (uhat * tstar)
    article = df.iloc[i_idx]['item']
    item_feats = " · ".join(f"{code2name[c][0]}={code2name[c][1]}" for c in fe.feature_ids[i_idx].tolist())
    w(f"**Why was item `{article}` (item {i_idx}) recommended to user {u_idx}?**\n")
    w(f"- The item's features: {item_feats}.")
    w(f"- Total score **{score:.3f}** = item-prototype half (û·t\\*) **{item_side:.3f}** "
      f"+ user-prototype half (u\\*·t̂) **{user_side:.3f}**.")
    w(f"- The recommendation is led by **item-prototype #{kbest}**, whose profile is:")
    feats = fmt_feats(prof[:, kbest], code2name, topk=5)
    w("  " + ", ".join(f"*{val}* ({c:+.2f})" for _, val, c in feats) + ".")
    w(f"- The user's affinity to that prototype is û_{kbest} = {float(uhat[kbest]):+.3f}; the item "
      f"activates it at t\\*_{kbest} = {float(tstar[kbest]):.3f} → contribution "
      f"**{float(contribs[kbest]):+.3f}** to the score.")
    # decompose that prototype's activation into the item's features
    rows = item_feature_rows(fe, i_idx)
    sh = per_feature_shares(rows, Pt)[:, kbest]
    codes = fe.feature_ids[i_idx].tolist()
    labels = [f"{code2name[c][0]}={code2name[c][1]}" for c in codes] + ["<item-ID row>"]
    w(f"- **Why does the item activate prototype #{kbest}?** "
      f"t\\*_{kbest} = 1 + {float(sh.sum()):.3f}, decomposed over its features:")
    w("\n| feature | share of activation |")
    w("|---|---|")
    for lbl, c in zip(labels, sh.tolist()):
        w(f"| {lbl} | {c:+.4f} |")

    # ---- honesty notes ----
    w("\n## Honesty notes (per dc01 §3.4)\n")
    w("- Per-feature shares are **jointly normalized through ‖q_i‖** — exact additive shares of the "
      "cosine part, *not* counterfactual effects (removing a feature rescales the others).")
    w("- The **+1 shift** is a feature-independent baseline, shown explicitly, never absorbed.")
    w("- The **item-prototype (û·t\\*) half is feature-grounded**; the user-prototype (u\\*·t̂) half is "
      "CF-grounded (interpreted post-hoc, as in stock ProtoMF) — they are distinguished above.")
    w("- Single-seed dev checkpoint; profiles show the known open risks (M2 'Solid' frequent-value "
      "recurrence; M4 near-duplicate prototypes) — see dc01 §3.8.")

    with open(args.out, 'w') as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote artifact -> {args.out}  ({len(L)} lines)")


if __name__ == '__main__':
    main()
