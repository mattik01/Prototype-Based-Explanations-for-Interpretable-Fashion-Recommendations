"""dc02 intrinsic read-out DEMO on a trained attr_item_proto checkpoint.

Loads a trained model from a run's results dir and renders the dc02 §3.4 read-outs:
  1. item-prototype attribute profiles — top-weight values per field, read DIRECTLY off the
     effective A (grounding by construction; no naming procedure needed),
  2. per-field weight-mass table — the free single-field-collapse diagnostic (§3.5 mode (c)),
  3. user-prototype attribute-response profiles — read off W_t rows (§3.4 pt. 4),
  4. one (user, item) EXACT score decomposition: u*_l·t̂_l terms + item-discriminating
     û_k·(t*_k−1) terms (each with its exact per-attribute shares) + the disclosed
     user-constant Σû_k (§3.4 amendment), with a sum check against the model's own logit.

Usage (local smoke checkpoint):
  python Master/temp/dc_checks/dc02/readout_demo.py \
    --results Master/experiments/results/attr_item_proto_debug_hm_3_month_s38210573 \
    --data    data/hm_3_month [--user 42] [--item 0] [--top-m 3]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import pandas as pd
import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_ids import build_attr_multi_hot, inject_feature_ids
from rec_sys.rec_sys import RecSys
from utilities.explanations.attr_readout import (per_attribute_shares, per_field_mass,
                                                 prototype_attr_profile, score_decomposition)

COLLAPSE_WARN = 0.90   # per-field mass share above which a prototype counts as single-field


def load_trained(results_dir, data_path):
    conf = json.load(open(os.path.join(results_dir, 'config.json')))
    n_items = len(pd.read_csv(os.path.join(data_path, 'item_ids.csv')))
    n_users = len(pd.read_csv(os.path.join(data_path, 'user_ids.csv')))
    ft = inject_feature_ids(conf['ft_ext_param'], data_path)   # rebuild X (non-mutating)
    u_fe, i_fe = FeatureExtractorFactory.create_models(ft, n_users, n_items)
    rec = RecSys(n_users, n_items, conf['rec_sys_param'], u_fe, i_fe,
                 conf['loss_func_name'], conf.get('loss_func_aggr', 'mean'))
    rec.init_parameters()
    ckpt = os.path.join(results_dir, 'best_model.pth')
    try:
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    except TypeError:
        sd = torch.load(ckpt, map_location='cpu')
    rec.load_state_dict(sd)
    rec.eval()
    return rec, conf


def gather(rec, conf, data_path):
    """Everything the renderers need, computed once."""
    fields = conf['ft_ext_param']['item_ft_ext_param']['attr_fields']
    X, field_offsets, code_to_cv = build_attr_multi_hot(data_path, fields)
    item_proto = rec.item_feature_extractor.model_1     # AttributePrototypeEmbedding
    item_proj = rec.item_feature_extractor.model_2      # AttributeProjection
    A = item_proto.effective_prototypes().detach()      # (K, V) — ALWAYS the effective matrix
    W_t = item_proj.linear_layer.weight.detach()        # (L_u, V)
    return dict(fields=fields, X=X, offs=field_offsets, code_to_cv=code_to_cv,
                A=A, W_t=W_t, item_proto=item_proto, item_proj=item_proj,
                user_proto=rec.user_feature_extractor.model_1,
                user_proj=rec.user_feature_extractor.model_2)


def fmt_profile(profiles, tag, top_m):
    lines = []
    for k, fields in enumerate(profiles):
        lines.append(f"\n{tag} prototype P{k}:")
        for f in fields:
            tops = " · ".join(f"{v} ({w:+.3f})" for v, w in f['top'][:top_m])
            lines.append(f"    {f['field']:<32} {tops}")
    return "\n".join(lines)


def fmt_mass(mass, fields):
    header = "proto | " + " | ".join(f"{f[:14]:>14}" for f in fields)
    lines = [header, "-" * len(header)]
    warned = []
    for k in range(mass.shape[0]):
        row = mass[k]
        lines.append(f"P{k:<4} | " + " | ".join(f"{float(v):>14.3f}" for v in row))
        if float(row.max()) > COLLAPSE_WARN:
            warned.append((k, fields[int(row.argmax())], float(row.max())))
    for k, f, v in warned:
        lines.append(f"⚠ P{k}: {v:.1%} of weight mass in single field '{f}' — "
                     f"single-field collapse (§3.5 mode (c))")
    if not warned:
        lines.append(f"(no prototype above the {COLLAPSE_WARN:.0%} single-field threshold)")
    return "\n".join(lines)


def demo_decomposition(rec, g, user_id, item_id, top_m):
    u_idx = torch.tensor([user_id])
    i_idx = torch.tensor([[item_id]])
    with torch.no_grad():
        logit = float(rec(u_idx, i_idx)[0, 0])
        u_star = g['user_proto'](u_idx)[0]
        u_hat = g['user_proj'](u_idx)[0]
        t_hat = g['item_proj'](i_idx)[0, 0]
        t_star = g['item_proto'](i_idx)[0, 0]
    dec = score_decomposition(u_star, u_hat, t_hat, t_star)
    shares = per_attribute_shares(g['X'][item_id], g['A'])           # (F, K)
    v_idx = (g['X'][item_id] > 0).nonzero(as_tuple=False).squeeze(1)
    value_names = [f"{g['code_to_cv'][int(v)][0]}={g['code_to_cv'][int(v)][1]}" for v in v_idx]

    lines = [f"\n=== EXACT decomposition: user {user_id} × item {item_id} "
             f"(logit {logit:+.4f}) ==="]
    proj = dec['proj_terms']
    order = torch.argsort(proj.abs(), descending=True)[:top_m]
    lines.append("u*ᵀt̂ half (user-prototype × item's learned affinity):")
    for l in order:
        lines.append(f"    UP{int(l)}: u*={float(u_star[l]):+.3f} × t̂={float(t_hat[l]):+.3f}"
                     f" = {float(proj[l]):+.4f}")
    proto = dec['proto_terms_discriminating']
    order_k = torch.argsort(proto.abs(), descending=True)[:top_m]
    lines.append("ûᵀt* half — ITEM-DISCRIMINATING û_k·(t*_k−1) (§3.4 amendment):")
    for k in order_k:
        lines.append(f"    P{int(k)}: û={float(u_hat[k]):+.3f} × (t*−1)={float(t_star[k]-1):+.3f}"
                     f" = {float(proto[k]):+.4f}")
        share_strs = sorted(zip(value_names, shares[:, k].tolist()),
                            key=lambda t: -abs(t[1]))[:top_m]
        for name, s in share_strs:
            lines.append(f"        {name}: {s:+.4f}")
    lines.append(f"user-constant baseline (disclosed once, ranking-irrelevant): "
                 f"Σû = {float(dec['user_constant']):+.4f}")
    err = abs(float(dec['total']) - logit)
    lines.append(f"SUM CHECK: Σ addends = {float(dec['total']):+.4f} vs logit {logit:+.4f} "
                 f"→ |err| = {err:.2e} {'✔ exact' if err < 1e-4 else '✘ MISMATCH'}")
    return "\n".join(lines), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--user', type=int, default=42)
    ap.add_argument('--item', type=int, default=0)
    ap.add_argument('--top-m', type=int, default=3)
    args = ap.parse_args()

    rec, conf = load_trained(args.results, args.data)
    g = gather(rec, conf, args.data)
    nonneg = conf['ft_ext_param']['item_ft_ext_param'].get('nonneg_prototypes', False)

    print(f"\n########## dc02 read-out — {os.path.basename(args.results)} "
          f"(K={g['A'].shape[0]}, V={g['A'].shape[1]}, nonneg={nonneg}) ##########")

    print("\n===== 1. Item-prototype attribute profiles (read off effective A) =====")
    print(fmt_profile(prototype_attr_profile(g['A'], g['offs'], g['code_to_cv'], args.top_m),
                      "item", args.top_m))

    print("\n===== 2. Per-field weight mass (collapse diagnostic) =====")
    print(fmt_mass(per_field_mass(g['A'], g['offs']), g['fields']))

    print("\n===== 3. User-prototype attribute-response profiles (read off W_t rows) =====")
    print(fmt_profile(prototype_attr_profile(g['W_t'], g['offs'], g['code_to_cv'], args.top_m),
                      "user", args.top_m))

    text, err = demo_decomposition(rec, g, args.user, args.item, args.top_m)
    print(text)
    sys.exit(0 if err < 1e-4 else 1)


if __name__ == '__main__':
    main()
