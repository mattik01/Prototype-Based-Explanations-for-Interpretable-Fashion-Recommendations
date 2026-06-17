"""dc01 intrinsic read-out DEMO on a trained checkpoint.

Loads a trained feature_item_proto model from a run's results dir and renders the two intrinsic
read-outs (dc01 §3.4) in human-readable form:
  1. global prototype attribute profile — top feature-values per item-prototype (cos(e_f, p_k)),
  2. exact per-feature decomposition of a single item's prototype activation (t*_k − 1 = Σ_f c_{f,k}),
     with the +1 shift shown and the feature-explained vs ID-row split (the R4↔R6 fidelity gauge).

Usage (on the cluster, in the protomf venv):
  python Master/temp/dc_checks/dc01/readout_demo.py \
    --results /scratch/c7031336/protomf_results/feature_item_proto_hm_1_month_s38210573 \
    --data    /scratch/c7031336/protomf_data/hm_1_month
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
from utilities.explanations.feature_readout import (
    per_feature_shares, global_prototype_profile, item_feature_rows)


def build_inverse_vocab(data_path, fields):
    """Reconstruct the exact field-offset vocab build_feature_ids uses, keeping value NAMES."""
    df = pd.read_csv(os.path.join(data_path, 'item_features.csv')).sort_values('item_id').reset_index(drop=True)
    code2name, offset = {}, 0
    for field in fields:
        vocab = sorted(df[field].astype(str).unique().tolist())
        for i, v in enumerate(vocab):
            code2name[offset + i] = f"{field}={v}"
        offset += len(vocab)
    return code2name, df


def load_trained(results_dir, data_path):
    conf = json.load(open(os.path.join(results_dir, 'config.json')))
    n_items = len(pd.read_csv(os.path.join(data_path, 'item_ids.csv')))
    n_users = len(pd.read_csv(os.path.join(data_path, 'user_ids.csv')))
    ft = inject_feature_ids(conf['ft_ext_param'], data_path)  # rebuild feature_ids (non-mutating)
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
    return rec, conf, n_items


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--n-protos', type=int, default=6, help='how many item-prototypes to profile')
    p.add_argument('--item', type=int, default=0, help='example item index for the decomposition')
    args = p.parse_args()

    rec, conf, n_items = load_trained(args.results, args.data)
    fields = conf['ft_ext_param']['item_ft_ext_param']['feature_fields']
    code2name, df = build_inverse_vocab(args.data, fields)

    item_proto = rec.item_feature_extractor.model_1            # PrototypeEmbedding
    fe = item_proto.embedding_ext                              # FeatureEmbedding
    Pt = item_proto.prototypes.detach()                        # (Kt, d)
    Kt, d = Pt.shape
    n_features = fe.n_features
    meta_emb = fe.embedding_layer.weight.detach()[:n_features]  # (n_features, d), metadata rows only

    print(f"\n=== model: feature_item_proto | d={d}, item-prototypes={Kt}, "
          f"feature-vocab={n_features}, fields={len(fields)} ===")
    print(f"(checkpoint: {os.path.basename(args.results)})\n")

    # --- (1) global prototype attribute profiles ---
    print(f"--- Global prototype attribute profiles: top feature-values by cos(e_f, p_k) ---")
    prof = global_prototype_profile(meta_emb, Pt)              # (n_features, Kt)
    for k in range(min(args.n_protos, Kt)):
        top = prof[:, k].topk(6)
        parts = [f"{code2name[int(i)]} ({float(v):+.2f})" for v, i in zip(top.values, top.indices)]
        print(f"  item-proto #{k:>2}: " + " | ".join(parts))

    # --- (2) exact per-feature decomposition for one item ---
    item = args.item
    rows = item_feature_rows(fe, item)                         # (F+1, d) incl ID row
    shares = per_feature_shares(rows, Pt)                      # (F+1, Kt)
    tstar = 1.0 + shares.sum(dim=0)                            # (Kt,)
    kbest = int(tstar.argmax())
    item_codes = fe.feature_ids[item].tolist()
    row_labels = [code2name[c] for c in item_codes] + ['<item-ID row>']

    print(f"\n--- Per-feature decomposition for item {item} "
          f"(article {df.iloc[item]['item']}), its top-activated prototype #{kbest} ---")
    print(f"  t*_{kbest} = {float(tstar[kbest]):.4f} = 1 (shift) + {float(tstar[kbest] - 1):.4f} (feature match)")
    F = len(item_codes)
    for lbl, c in zip(row_labels, shares[:, kbest].tolist()):
        print(f"    {lbl:<42} {c:+.4f}")
    feat_share = float(shares[:F, kbest].sum())
    id_share = float(shares[F, kbest])
    s = float(shares[:, kbest].sum())
    print(f"    {'sum of shares':<42} {s:+.4f}   (== t*-1 = {float(tstar[kbest] - 1):+.4f})")
    print(f"  feature-explained: {feat_share:+.4f}   ID-row: {id_share:+.4f}   "
          f"(feature fraction = {feat_share / (feat_share + id_share) * 100:+.0f}% of the non-shift activation)")
    print("\n  NOTE: shares are jointly normalized through ||q_i|| (exact additive shares of the "
          "cosine part, not counterfactual); the +1 shift is feature-independent.")


if __name__ == '__main__':
    main()
