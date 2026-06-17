"""dc01 — render a post-hoc vs intrinsic item-prototype interpretation figure (PNG).

For a trained feature_item_proto checkpoint, for each of a few item-prototypes p_k, show two
side-by-side explanations of the SAME prototype:
  LEFT  — POST-HOC (ProtoMF's original method): the prototype's nearest items (top by t*_k);
          you must read the item cloud and infer the concept.
  RIGHT — INTRINSIC (dc01): the prototype's feature-value profile, cos(e_f, p_k), read directly
          from the learned parameters — the concept is named, not inferred.

Usage: PYTHONPATH=<repo> python make_proto_figure.py --results <dir> --data <dir> --out fig.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_ids import inject_feature_ids
from rec_sys.rec_sys import RecSys
from utilities.explanations.feature_readout import global_prototype_profile


def build_inverse_vocab(data_path, fields):
    df = pd.read_csv(os.path.join(data_path, 'item_features.csv')).sort_values('item_id').reset_index(drop=True)
    code2name, offset = {}, 0
    for field in fields:
        vocab = sorted(df[field].astype(str).unique().tolist())
        for i, v in enumerate(vocab):
            code2name[offset + i] = v
        offset += len(vocab)
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
    return rec, conf, n_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--protos', default='2,7,6', help='comma-separated item-prototype indices to show')
    args = ap.parse_args()

    rec, conf, n_items = load(args.results, args.data)
    fields = conf['ft_ext_param']['item_ft_ext_param']['feature_fields']
    code2name, df = build_inverse_vocab(args.data, fields)
    test_hr = json.load(open(os.path.join(args.results, 'test_metrics.json'))).get('hit_ratio@10')

    item_proto = rec.item_feature_extractor.model_1
    fe = item_proto.embedding_ext
    Pt = item_proto.prototypes.detach()
    nf = fe.n_features
    meta_emb = fe.embedding_layer.weight.detach()[:nf]

    with torch.no_grad():
        q_all = fe(torch.arange(n_items))
        qn = q_all.norm(dim=1, keepdim=True)
        tstar_all = 1.0 + (q_all @ Pt.T) / (qn * Pt.norm(dim=1))   # (n_items, Kt)
    prof = global_prototype_profile(meta_emb, Pt)                  # (nf, Kt)

    protos = [int(x) for x in args.protos.split(',')]
    nrow = len(protos)
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 3.2 * nrow),
                             gridspec_kw={'width_ratios': [1.05, 1.0]})
    if nrow == 1:
        axes = axes.reshape(1, 2)

    TOPK = 8
    for r, k in enumerate(protos):
        axL, axR = axes[r, 0], axes[r, 1]

        # LEFT — post-hoc: nearest items by activation t*_k
        top_items = tstar_all[:, k].topk(TOPK).indices.tolist()
        axL.axis('off')
        lines = []
        for it in top_items:
            pt = str(df.iloc[it]['product_type_name'])[:22]
            cg = str(df.iloc[it]['colour_group_name'])[:16]
            lines.append(f"• {pt} · {cg}")
        axL.text(0.02, 0.90, f"Item-prototype #{k}\nPOST-HOC: top nearest items",
                 fontsize=12, fontweight='bold', va='top', family='DejaVu Sans')
        axL.text(0.02, 0.72, "\n".join(lines), fontsize=10.5, va='top', family='DejaVu Sans Mono')
        axL.text(0.02, 0.02, "(read the item cloud → infer the concept yourself)",
                 fontsize=8.5, style='italic', color='dimgray', va='bottom')

        # RIGHT — intrinsic: feature-value profile cos(e_f, p_k)
        top = prof[:, k].topk(TOPK)
        vals = top.values.tolist()[::-1]
        labs = [code2name[int(i)][:26] for i in top.indices.tolist()][::-1]
        axR.barh(range(TOPK), vals, color='#3b7dd8', edgecolor='none')
        axR.set_yticks(range(TOPK))
        axR.set_yticklabels(labs, fontsize=10)
        axR.set_xlim(0, 1)
        axR.set_xlabel('cos(e_f, p_k)', fontsize=9)
        axR.set_title("INTRINSIC: feature-value profile (dc01)", fontsize=12, fontweight='bold', loc='left')
        axR.grid(axis='x', alpha=0.25)
        for spine in ('top', 'right'):
            axR.spines[spine].set_visible(False)

    fig.suptitle(
        "Item-prototype interpretation — POST-HOC (ProtoMF, nearest items) vs "
        "INTRINSIC feature grounding (dc01)\n"
        f"feature_item_proto · hm_1_month · TEST HR@10 = {test_hr:.4f} · "
        "every value read from the trained model",
        fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=170, bbox_inches='tight')
    print(f"wrote figure -> {args.out}")


if __name__ == '__main__':
    main()
