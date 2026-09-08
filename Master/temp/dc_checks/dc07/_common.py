# dc07 check harness (shared): repo path, check(), float64 toy builders for the three input
# shapes of the prototype layer — free Embedding (host), FeatureEmbedding (fI-shaped),
# HistoryFeatureEmbedding (fU-shaped).
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from feature_extraction.feature_extractors import (FeatureEmbedding, HistoryFeatureEmbedding,  # noqa: E402
                                                   PrototypeEmbedding)

torch.set_default_dtype(torch.float64)
DT = torch.float64


def make_check(tag):
    fails = []

    def check(name, cond, evidence=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {tag}.{name}" + (f" | {evidence}" if evidence else ""))
        if not cond:
            fails.append(name)
    return check, fails


def finish(tag, fails):
    print(f"\n{tag}: ALL PASS" if not fails else f"\n{tag}: {len(fails)} FAILED -> {fails}")
    sys.exit(1 if fails else 0)


def proto(n, d, k, cosine_type, tau=1.0, ext=None, reg=('max', 'max')):
    m = PrototypeEmbedding(n, d, k, use_weight_matrix=False, cosine_type=cosine_type,
                           reg_proto_type=reg[0], reg_batch_type=reg[1], embedding_ext=ext, temperature=tau)
    m.init_parameters()
    if ext is None:
        m.embedding_ext.init_parameters()
    else:
        ext.init_parameters()
    return m.double()


def fi_ext(n, d, n_feat=11, F=3, use_id=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, n_feat, (n, F), generator=g)
    return FeatureEmbedding(n, ids, n_feat, d, use_id_feature=use_id), ids


def fu_ext(n, d, n_feat=11, d_max=5, use_id=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, n_feat, (n, d_max), generator=g)
    w = torch.rand(n, d_max, generator=g)
    mask = torch.rand(n, d_max, generator=g) < 0.3
    w[mask] = 0.0
    ids[mask] = 0
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    return HistoryFeatureEmbedding(n, ids, w, n_feat, d, use_id_feature=use_id), ids, w
