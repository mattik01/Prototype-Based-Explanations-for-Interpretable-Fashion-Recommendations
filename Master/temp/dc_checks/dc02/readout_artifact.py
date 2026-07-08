"""dc02 read-out ARTIFACT: renders the same §3.4 read-outs as readout_demo.py into a
markdown file (default: <results>/explanations/attr/readout_artifact.md) for the thesis
record. Reuses readout_demo's loaders/renderers — keep the two in sync via that import.

Usage:
  python Master/temp/dc_checks/dc02/readout_artifact.py \
    --results Master/experiments/results/attr_item_proto_debug_hm_3_month_s38210573 \
    --data    data/hm_3_month [--user 42] [--item 0] [--top-m 3] [--out <path.md>]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from readout_demo import (COLLAPSE_WARN, demo_decomposition, fmt_mass, fmt_profile, gather,
                          load_trained)
from utilities.explanations.attr_readout import per_field_mass, prototype_attr_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--user', type=int, default=42)
    ap.add_argument('--item', type=int, default=0)
    ap.add_argument('--top-m', type=int, default=3)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rec, conf = load_trained(args.results, args.data)
    g = gather(rec, conf, args.data)
    nonneg = conf['ft_ext_param']['item_ft_ext_param'].get('nonneg_prototypes', False)
    dec_text, err = demo_decomposition(rec, g, args.user, args.item, args.top_m)

    out = args.out or os.path.join(args.results, 'explanations', 'attr', 'readout_artifact.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(out, 'w') as f:
        f.write(f"# dc02 intrinsic read-out — `{os.path.basename(args.results)}`\n\n")
        f.write(f"K={g['A'].shape[0]}, V={g['A'].shape[1]}, fields={len(g['fields'])}, "
                f"nonneg_prototypes={nonneg}. All quantities are model parameters read directly "
                f"or exact forward-pass arithmetic (design doc §3.4); contributions are rendered "
                f"item-discriminating (û·(t*−1)) with the user constant disclosed (§3.4 amendment).\n\n")
        f.write("## 1. Item-prototype attribute profiles (effective A)\n\n```\n")
        f.write(fmt_profile(prototype_attr_profile(g['A'], g['offs'], g['code_to_cv'],
                                                   args.top_m), "item", args.top_m))
        f.write("\n```\n\n")
        f.write(f"## 2. Per-field weight mass (collapse diagnostic, warn >{COLLAPSE_WARN:.0%})\n\n```\n")
        f.write(fmt_mass(per_field_mass(g['A'], g['offs']), g['fields']))
        f.write("\n```\n\n")
        f.write("## 3. User-prototype attribute-response profiles (W_t rows)\n\n```\n")
        f.write(fmt_profile(prototype_attr_profile(g['W_t'], g['offs'], g['code_to_cv'],
                                                   args.top_m), "user", args.top_m))
        f.write("\n```\n\n")
        f.write("## 4. Worked example — exact score decomposition\n\n```\n")
        f.write(dec_text)
        f.write("\n```\n")

    print(f"artifact written → {out} (sum-check err {err:.2e})")
    sys.exit(0 if err < 1e-4 else 1)


if __name__ == '__main__':
    main()
