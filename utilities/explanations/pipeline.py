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
from utilities.explanations.naming import (
    get_naming_config,
    intrinsic_naming_config,
    name_prototypes_from_weights,
    name_prototypes_intrinsic,
)

EXPLAINABLE_MODELS = {"item_proto", "user_proto", "user_item_proto", "feature_item_proto"}


def run_explanations_pipeline(
    results_dir: str,
    output_subdir: str = "explanations",
    sample_users_for_weight_viz: Optional[List[int]] = None,
    tsne_sample_size: int = 2000,
    top_k: int = 10,
    naming_overrides: Optional[dict] = None,
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

    model, config, _metadata = load_recsys_from_results_dir(results_dir)
    accessor = get_accessor(model_type, model)
    items_info = load_items_info(dataset)

    # Resolve naming config first so artifacts nest under the scoring used to build them
    # (e.g. .../explanations/lift/), keeping lift vs. raw vs. cosine runs cleanly separated.
    base_cfg = get_naming_config(dataset, items_info, overrides=naming_overrides)
    # feature_item_proto grounds item prototypes INTRINSICALLY (cos(e_f, p_k)); the panels and
    # descriptors then come from the model's own feature fields, and outputs nest under cosine/.
    intrinsic_item = getattr(accessor, "has_intrinsic_item_grounding", False)
    feature_fields = None
    if intrinsic_item:
        feature_fields = config["ft_ext_param"]["item_ft_ext_param"]["feature_fields"]
        min_score = float((naming_overrides or {}).get("min_score", 0.30))
        naming_cfg = intrinsic_naming_config(base_cfg, feature_fields, min_score=min_score)
    else:
        naming_cfg = base_cfg

    output_dir = os.path.join(results_dir, output_subdir, naming_cfg.scoring)
    os.makedirs(output_dir, exist_ok=True)

    # Derive prototype names once; every explainer reads them off the context.
    naming_item = naming_user = None
    try:
        if accessor.has_item_prototypes:
            if intrinsic_item:
                # Item prototypes share the feature-embedding space -> intrinsic cos route.
                naming_item = name_prototypes_intrinsic(
                    accessor.feature_value_embeddings(), accessor.item_prototypes(),
                    feature_fields, items_info, naming_cfg, side="item")
            else:
                sim = accessor.item_to_item_proto_sim()
                if sim is not None:
                    # Item prototypes share the item space -> post-hoc closeness route.
                    naming_item = name_prototypes_from_weights(sim, items_info, naming_cfg, side="item")
        if accessor.has_user_prototypes:
            ups = accessor.items_in_user_proto_space()
            if ups is not None:
                # User prototypes are CF (no features) -> post-hoc activation route, base cfg.
                naming_user = name_prototypes_from_weights(ups, items_info, base_cfg, side="user")
    except Exception as e:
        print(f"[explanations] ⚠ prototype naming failed: {e!r}")

    ctx = ExplainCtx(
        accessor=accessor,
        items_info=items_info,
        output_dir=output_dir,
        tsne_sample_size=tsne_sample_size,
        top_k=top_k,
        sample_users_for_weight_viz=sample_users_for_weight_viz,
        naming_cfg=naming_cfg,
        naming_item=naming_item,
        naming_user=naming_user,
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
    # --- prototype naming knobs (override NamingConfig defaults) ---
    p.add_argument("--naming-top-k", type=int, default=None,
                   help="How many top items define a prototype's name (NamingConfig.top_k_for_naming)")
    p.add_argument("--naming-max-descriptors", type=int, default=None,
                   help="Max feature descriptors per prototype name (NamingConfig.max_descriptors)")
    p.add_argument("--naming-scoring", choices=["lift", "raw"], default=None,
                   help="Descriptor selection metric (NamingConfig.scoring)")
    p.add_argument("--naming-min-count", type=int, default=None,
                   help="Min supporting top-k items for a descriptor (NamingConfig.min_count)")
    return p.parse_args()


def _naming_overrides_from_args(args) -> Optional[dict]:
    mapping = {
        "top_k_for_naming": args.naming_top_k,
        "max_descriptors": args.naming_max_descriptors,
        "scoring": args.naming_scoring,
        "min_count": args.naming_min_count,
    }
    overrides = {k: v for k, v in mapping.items() if v is not None}
    return overrides or None


def main():
    args = _parse_args()
    run_explanations_pipeline(
        results_dir=args.results_dir,
        output_subdir=args.output_subdir,
        sample_users_for_weight_viz=args.weight_viz_users,
        tsne_sample_size=args.tsne_sample_size,
        top_k=args.top_k,
        naming_overrides=_naming_overrides_from_args(args),
    )


if __name__ == "__main__":
    main()
