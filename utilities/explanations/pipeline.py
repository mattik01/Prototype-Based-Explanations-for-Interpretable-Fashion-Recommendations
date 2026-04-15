"""Single entrypoint for the explanations pipeline.

Usable both as a library (called from run_combo.py) and as a standalone CLI:

    python -m utilities.explanations.pipeline \\
        --results-dir Master/experiments/results/<combo_folder>
"""
import argparse
import json
import os
import sys
from typing import List, Optional

# Ensure repo root is on sys.path when invoked as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")  # headless backend for auto-invocation

from utilities.explanations.accessor import get_accessor
from utilities.explanations.explainers import ExplainCtx, REGISTERED_EXPLAINERS
from utilities.explanations.items_info import load_items_info
from utilities.explanations.loader import load_recsys_from_results_dir

EXPLAINABLE_MODELS = {"item_proto", "user_proto", "user_item_proto"}


def run_explanations_pipeline(
    results_dir: str,
    output_subdir: str = "explanations",
    sample_users_for_weight_viz: Optional[List[int]] = None,
    tsne_sample_size: int = 2000,
    top_k: int = 10,
) -> Optional[str]:
    """
    Generate explanations for a completed combo.

    :param results_dir: folder containing best_model.pth, config.json, metadata.json
    :param output_subdir: subfolder under results_dir where artifacts are written
    :param sample_users_for_weight_viz: users for WeightVizExplainer (user_item_proto only).
        Defaults to [42].
    :param tsne_sample_size: how many objects to sample per TSNE plot
    :param top_k: how many items to return per prototype in top-k CSVs
    :return: absolute path to output dir, or None if the model is not explainable
    """
    if sample_users_for_weight_viz is None:
        sample_users_for_weight_viz = [42]

    # Read metadata first so we can cheap-skip non-explainable models without loading the model.
    metadata_path = os.path.join(results_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {results_dir}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    model_type = metadata["model"]
    dataset = metadata["dataset"]

    if model_type not in EXPLAINABLE_MODELS:
        print(f"[explanations] model '{model_type}' is not explainable — skipping.")
        return None

    print(f"[explanations] starting for {model_type} × {dataset} ({results_dir})")

    model, _config, _metadata = load_recsys_from_results_dir(results_dir)
    accessor = get_accessor(model_type, model)
    items_info = load_items_info(dataset)

    output_dir = os.path.join(results_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    ctx = ExplainCtx(
        accessor=accessor,
        items_info=items_info,
        output_dir=output_dir,
        tsne_sample_size=tsne_sample_size,
        top_k=top_k,
        sample_users_for_weight_viz=sample_users_for_weight_viz,
    )

    for explainer in REGISTERED_EXPLAINERS:
        if not explainer.supports(model_type):
            continue
        print(f"[explanations]   running '{explainer.name}' ...")
        try:
            explainer.run(ctx)
        except Exception as e:
            # One broken explainer must not kill the rest.
            print(f"[explanations]   ⚠ '{explainer.name}' failed: {e!r}")

    print(f"[explanations] done → {output_dir}")
    return output_dir


def _parse_args():
    p = argparse.ArgumentParser(description="Generate explanations for a completed combo.")
    p.add_argument("--results-dir", required=True, help="Combo results folder")
    p.add_argument("--output-subdir", default="explanations")
    p.add_argument("--tsne-sample-size", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--weight-viz-users", type=int, nargs="*", default=None,
        help="User ids for weight_viz (user_item_proto only). Default: [42]",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    run_explanations_pipeline(
        results_dir=args.results_dir,
        output_subdir=args.output_subdir,
        sample_users_for_weight_viz=args.weight_viz_users,
        tsne_sample_size=args.tsne_sample_size,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
