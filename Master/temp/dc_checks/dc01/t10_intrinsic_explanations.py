# dc01 P9 — intrinsic explanations wiring: accessor registered, pipeline explainable,
# and name_prototypes_intrinsic maps cos(e_f, p_k) alignment to the correct feature-value names.
# (Pure-logic checks; avoids importing the matplotlib/sklearn explainer pipeline so it runs anywhere.)
# Extended at the SC.3 re-run (2026-07-12): fI-host accessor shape (no user prototypes, no
# projections) + weight_viz gating consistency (source checks).
# Run: python Master/temp/dc_checks/dc01/t10_intrinsic_explanations.py
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

FAILS = []


def check(name, cond, evidence=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
    if not cond:
        FAILS.append(name)


# accessor registered (no matplotlib)
from utilities.explanations.accessor import _ACCESSORS, FeatureItemProtoAccessor
check("t10.accessor_registered",
      _ACCESSORS.get('feature_item_proto') is FeatureItemProtoAccessor, "")
check("t10.accessor_intrinsic_flag",
      getattr(FeatureItemProtoAccessor, 'has_intrinsic_item_grounding', False) is True, "")

# fI host shape (SC.3 re-run): item prototypes only — no user prototypes, no projections
check("t10.accessor_fI_flags",
      FeatureItemProtoAccessor.has_item_prototypes is True
      and FeatureItemProtoAccessor.has_user_prototypes is False
      and FeatureItemProtoAccessor.has_projections is False,
      "has_item_prototypes=True, has_user_prototypes=False, has_projections=False")

# weight_viz must NOT claim feature_item_proto anymore (no projection branch on the fI host);
# source check to avoid the matplotlib import
with open(os.path.join(REPO, 'utilities', 'explanations', 'explainers', 'weight_viz.py')) as f:
    wv_src = f.read()
import re as _re
_ret = _re.search(r"return model_type in \{([^}]*)\}", wv_src)
check("t10.weight_viz_excludes_fI",
      _ret is not None and 'feature_item_proto' not in _ret.group(1),
      f"weight_viz supports {{{_ret.group(1) if _ret else '?'}}}")

# pipeline marks it explainable (source check — importing pipeline pulls matplotlib/sklearn)
with open(os.path.join(REPO, 'utilities', 'explanations', 'pipeline.py')) as f:
    src = f.read()
import re
m = re.search(r"EXPLAINABLE_MODELS\s*=\s*\{([^}]*)\}", src)
check("t10.pipeline_explainable", m is not None and 'feature_item_proto' in m.group(1),
      "feature_item_proto in pipeline EXPLAINABLE_MODELS")
check("t10.pipeline_intrinsic_route", 'name_prototypes_intrinsic' in src, "intrinsic naming wired in pipeline")

# dual naming route wired (F-DC01-13, SC.5): the post-hoc pass runs for intrinsic models
check("t10.pipeline_dual_route", 'dual naming route' in src.lower(),
      "post-hoc pass alongside intrinsic (F-DC01-13)")

# run_combo auto-explanations wiring (F-DC01-11, SC.5): fI headline model included;
# attr_item_proto still excluded until dc02's SC.5 (source check, same regex style)
with open(os.path.join(REPO, 'Master', 'scripts', 'run_combo.py')) as f:
    rc_src = f.read()
m_rc = re.search(r"EXPLAINABLE_MODELS\s*=\s*\{([^}]*)\}", rc_src)
check("t10.run_combo_explainable_fI",
      m_rc is not None and 'feature_item_proto' in m_rc.group(1),
      "feature_item_proto in run_combo EXPLAINABLE_MODELS (F-DC01-11)")
check("t10.run_combo_attr_still_excluded",
      m_rc is not None and 'attr_item_proto' not in m_rc.group(1),
      "attr_item_proto excluded until dc02 SC.5")

# intrinsic naming maps cos alignment -> correct names + matches build_feature_ids vocab order
from utilities.explanations.naming import intrinsic_naming_config, get_naming_config, name_prototypes_intrinsic

items = pd.DataFrame({'item_id': [0, 1, 2, 3],
                      'f1': ['a', 'b', 'a', 'b'], 'f2': ['x', 'y', 'z', 'x']})
fields = ['f1', 'f2']  # build_feature_ids order: f1[a,b]->0,1 ; f2[x,y,z]->2,3,4
emb = np.eye(5, 6, dtype=np.float32)
protos = np.zeros((3, 6), dtype=np.float32)
protos[0, 0] = 1.0           # -> f1=a (code 0)
protos[1, 3] = 1.0           # -> f2=y (code 3)
protos[2] = emb[0] + emb[2]  # -> f1=a + f2=x
cfg = intrinsic_naming_config(get_naming_config('hm_1_month', items), fields, min_score=0.3)
res = name_prototypes_intrinsic(emb, protos, fields, items, cfg, side='item')

check("t10.cfg_scoring_cosine", cfg.scoring == 'cosine', f"{cfg.scoring}")
check("t10.proto0_named_a", res.names[0].name == 'a', f"{res.names[0].name}")
check("t10.proto1_named_y", res.names[1].name == 'y', f"{res.names[1].name}")
check("t10.proto2_two_descriptors", res.names[2].name in ('a, x', 'x, a'),
      f"{res.names[2].name}")
# vocab-size mismatch must raise (alignment guard)
try:
    name_prototypes_intrinsic(np.eye(4, 6, dtype=np.float32), protos, fields, items, cfg)
    check("t10.vocab_mismatch_raises", False, "no error")
except ValueError:
    check("t10.vocab_mismatch_raises", True, "raised ValueError on size mismatch")

print("\nt10: ALL PASS" if not FAILS else f"\nt10: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
