# t03 — lightfm CBF baseline (S0.4): factory branch, injection seam, shapes, KEYSTONE
# (use_id_feature=True + zero fields reduces bit-identically to mf), cold path
# (tags-only scores unseen items finitely/non-constantly, signature twins tie exactly),
# no extractor reg loss, checkpoint roundtrip, gradient flow to the feature table.
import os
import shutil
import sys
import tempfile

import torch

from _harness import make_check

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_extractors import Embedding, FeatureEmbedding
from feature_extraction.feature_ids import build_feature_ids, inject_feature_ids
from rec_sys.rec_sys import RecSys

check, fails = make_check()

tmp = tempfile.mkdtemp(prefix='s0_t03_')

N_USERS, N_ITEMS, D = 10, 6, 8

# toy catalog: item 5 duplicates item 0's signature (the twin); others distinct
rows = [
    (0, 'blue', 'square'),
    (1, 'green', 'circle'),
    (2, 'red', 'stripe'),
    (3, 'blue', 'circle'),
    (4, 'green', 'stripe'),
    (5, 'blue', 'square'),   # twin of 0
]
with open(os.path.join(tmp, 'item_features.csv'), 'w') as f:
    f.write('item_id,color,shape\n')
    for r in rows:
        f.write(','.join(str(x) for x in r) + '\n')


def lightfm_spec(use_id, fields):
    return {
        'ft_type': 'lightfm', 'embedding_dim': D,
        'user_ft_ext_param': {'ft_type': 'embedding'},
        'item_ft_ext_param': {'ft_type': 'lightfm', 'use_id_feature': use_id,
                              'feature_fields': list(fields)},
    }


def build(param, seed=7, use_bias=0):
    torch.manual_seed(seed)
    u_fe, i_fe = FeatureExtractorFactory.create_models(param, N_USERS, N_ITEMS)
    rec = RecSys(N_USERS, N_ITEMS, {'use_bias': use_bias}, u_fe, i_fe, 'bce')
    rec.init_parameters()
    return rec


def toy_batch(B=4, n_neg=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randint(0, N_USERS, (B,), generator=g)
    i = torch.randint(0, N_ITEMS, (B, 1 + n_neg), generator=g)
    return u, i


# --- 1. injection seam: lightfm branch, non-mutating copy discipline ---
spec = lightfm_spec(True, ['color', 'shape'])
out = inject_feature_ids(spec, tmp)
check("inject returns a new object", out is not spec)
check("inject adds feature_ids + n_features",
      'feature_ids' in out['item_ft_ext_param'] and out['item_ft_ext_param']['n_features'] == 6,
      f"n_features={out['item_ft_ext_param'].get('n_features')}")   # colors{blue,green,red}=3 + shapes{circle,square,stripe}=3
check("caller item dict NOT mutated (no tensor leak into config)",
      'feature_ids' not in spec['item_ft_ext_param'] and 'n_features' not in spec['item_ft_ext_param'])
check("user sub-dict copied", out['user_ft_ext_param'] is not spec['user_ft_ext_param'])

# --- 2. factory build + shapes ---
model = build(out)
check("user extractor is plain Embedding", isinstance(model.user_feature_extractor, Embedding),
      model.user_feature_extractor.name)
check("item extractor is FeatureEmbedding", isinstance(model.item_feature_extractor, FeatureEmbedding),
      model.item_feature_extractor.name)
u, i = toy_batch()
logits = model(u, i)
check("forward logits shape (B, 1+n_neg)", logits.shape == (4, 3), f"{tuple(logits.shape)}")
check("logits finite", bool(torch.isfinite(logits).all()))

# --- 3. KEYSTONE: use_id_feature=True + F=0 fields ≡ mf, bit-identical ---
mf_param = {'ft_type': 'detached', 'embedding_dim': D,
            'user_ft_ext_param': {'ft_type': 'embedding'},
            'item_ft_ext_param': {'ft_type': 'embedding'}}
f0 = inject_feature_ids(lightfm_spec(True, []), tmp)
check("F=0 injection: n_features == 0 and empty tensor",
      f0['item_ft_ext_param']['n_features'] == 0
      and f0['item_ft_ext_param']['feature_ids'].shape == (N_ITEMS, 0))

mf_model = build(mf_param, seed=13)
f0_model = build(f0, seed=13)
sd_mf, sd_f0 = mf_model.state_dict(), f0_model.state_dict()
check("keystone: identical state_dict keys", set(sd_mf.keys()) == set(sd_f0.keys()),
      f"mf-only: {set(sd_mf) - set(sd_f0)}, f0-only: {set(sd_f0) - set(sd_mf)}")
check("keystone: identical parameter tensors",
      all(torch.equal(sd_mf[k], sd_f0[k]) for k in sd_mf))
u, i = toy_batch(seed=3)
check("keystone: bit-identical logits", torch.equal(mf_model(u, i), f0_model(u, i)))

# --- 4. cold path: tags-only (use_id_feature=False) ---
tags = build(inject_feature_ids(lightfm_spec(False, ['color', 'shape']), tmp), seed=5)
all_items = torch.arange(N_ITEMS)
emb = tags.item_feature_extractor(all_items)                     # (M, D)
check("tags-only: finite embeddings for every item (incl. never-trained)",
      bool(torch.isfinite(emb).all()))
check("tags-only: signature twins identical (items 0 and 5)", torch.equal(emb[0], emb[5]))
check("tags-only: distinct signatures differ (items 0 and 1)", not torch.equal(emb[0], emb[1]))
check("tags-only: NO per-item rows in the table",
      tags.item_feature_extractor.embedding_layer.num_embeddings == 6,
      f"{tags.item_feature_extractor.embedding_layer.num_embeddings}")

# with ID rows: table has n_features + n_items rows; dropping the ID column at cold time is
# the dc01 convention (S0.3 §5) — here we just pin the row layout that convention relies on
ids_model = build(inject_feature_ids(lightfm_spec(True, ['color', 'shape']), tmp), seed=5)
check("tags+ids: table rows == n_features + n_items",
      ids_model.item_feature_extractor.embedding_layer.num_embeddings == 6 + N_ITEMS)

# --- 5. no extractor reg loss (plain dot-product model) ---
_ = tags(u, i)
check("get_and_reset_loss == 0 (user + item)",
      tags.user_feature_extractor.get_and_reset_loss() == 0.
      and tags.item_feature_extractor.get_and_reset_loss() == 0.)

# --- 6. checkpoint roundtrip: save → fresh rebuild → strict load → identical outputs ---
pth = os.path.join(tmp, 'ck.pth')
torch.save(tags.state_dict(), pth)
tags2 = build(inject_feature_ids(lightfm_spec(False, ['color', 'shape']), tmp), seed=99)
tags2.load_state_dict(torch.load(pth))
u, i = toy_batch(seed=11)
check("checkpoint roundtrip: identical logits after strict load",
      torch.equal(tags(u, i), tags2(u, i)))
check("feature_ids buffer NOT in checkpoint",
      not any('feature_ids' in k for k in torch.load(pth).keys()))

# --- 7. gradient flows to the feature table (tags-only) ---
u, i = toy_batch(seed=17)
out_ = tags(u, i)
labels = torch.zeros_like(out_)
labels[:, 0] = 1.
tags.loss_func(out_, labels).backward()
g = tags.item_feature_extractor.embedding_layer.weight.grad
check("gradient reaches the feature-value table", g is not None and bool((g != 0).any()))

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
