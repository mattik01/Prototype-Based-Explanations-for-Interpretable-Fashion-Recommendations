"""dc06 build verification (t02): the implemented FeatureEmbedding knobs against the design's
verified claims — synthetic tensors, no dataset, CPU, seconds.

Checks:
  V1  knobs-off identity: center_fields=False, metadata_scale=1.0 is bit-identical to the
      manual raw sum on BOTH layouts (dc01-behaviour preservation).
  V2  exact-C (4b C1): fixed layout, centred output == raw output − C, C = Σ_f μ_f, both
      'weighted' and 'unweighted' modes.
  V3  gauge invariance (4b C2): shifting one field's rows by a constant leaves centred
      outputs unchanged.
  V4  bag layout per-token form (4b C9): centred bag output == Σ w·(e−μ) manual; removed
      mass is item-dependent for unequal bag sizes.
  V5  gauge_snap function-inertness (4b C2/C7): outputs before/after snap identical to
      float tolerance; post-snap field means ~0.
  V6  metadata_scale: output == α·Σe_meta + e_ID exactly; ID untouched.
  V7  interaction mode: field_probs match hand-computed interaction-weighted frequencies;
      zero-count values get probability 0.
"""
import sys, os
sys.path.insert(0, '/home/mattik01/Desktop/githubs/ProtoMF')
import torch

from feature_extraction.feature_extractors import FeatureEmbedding

torch.manual_seed(0)
TOL = 1e-6
d = 8
n_items = 6
# 2 fields: field 0 vocab {0,1,2}, field 1 vocab {3,4} -> n_features = 5
n_features = 5
field_index = torch.tensor([0, 0, 0, 1, 1])
feature_ids_fixed = torch.tensor([[0, 3], [1, 3], [2, 4], [0, 4], [1, 3], [0, 3]])

def mk(center=False, mode='weighted', scale=1.0, ids=True, bag=False, iw=None, detach=False):
    if bag:
        # bag layout: variable token counts, padding id 0 with weight 0
        feat = torch.tensor([[0, 3, 0], [1, 3, 4], [2, 0, 0], [0, 4, 1], [1, 3, 0], [0, 3, 0]])
        w = torch.tensor([[1., 1., 0.], [1., 1., 1.], [1., 0., 0.], [1., 1., 1.], [1., 1., 0.], [1., 1., 0.]])
    else:
        feat, w = feature_ids_fixed, None
    m = FeatureEmbedding(n_items, feat, n_features, d, use_id_feature=ids,
                         feature_weights=w, center_fields=center, center_mode=mode,
                         center_detach=detach, metadata_scale=scale,
                         field_index=field_index if center else None, item_weights=iw)
    m.init_parameters()
    return m

def copy_table(src, dst):
    dst.embedding_layer.weight.data.copy_(src.embedding_layer.weight.data)

fails = []
def check(name, ok, detail=''):
    print(f'{name}: {"PASS" if ok else "FAIL"} {detail}')
    if not ok:
        fails.append(name)

o = torch.arange(n_items)

# V1 knobs-off identity (fixed + bag)
for bag in (False, True):
    m = mk(bag=bag)
    E = m.embedding_layer.weight
    if bag:
        feat, w = m.feature_ids, m.feature_weights
        manual = (E[feat] * w.unsqueeze(-1)).sum(-2) + E[n_features + o]
    else:
        manual = E[m.feature_ids].sum(-2) + E[n_features + o]
    check(f'V1 knobs-off identity ({"bag" if bag else "fixed"})',
          torch.equal(m(o), manual))

# V2 exact-C, both modes
for mode in ('weighted', 'unweighted'):
    raw = mk(); cen = mk(center=True, mode=mode); copy_table(raw, cen)
    E = cen.embedding_layer.weight
    if mode == 'weighted':
        counts = torch.bincount(feature_ids_fixed.reshape(-1), minlength=n_features).double()
    else:
        counts = torch.ones(n_features, dtype=torch.float64)
    mu0 = (counts[:3] / counts[:3].sum() @ E[:3].double())
    mu1 = (counts[3:] / counts[3:].sum() @ E[3:5].double())
    C = (mu0 + mu1).float()
    diff = (raw(o) - cen(o)) - C
    check(f'V2 exact-C ({mode})', diff.abs().max().item() < TOL,
          f'max dev {diff.abs().max().item():.2e}')

# V3 gauge invariance
cen = mk(center=True)
out0 = cen(o).detach().clone()
with torch.no_grad():
    cen.embedding_layer.weight[:3] += 3.7  # shift all of field 0's rows
out1 = cen(o)
check('V3 gauge invariance', (out0 - out1).abs().max().item() < 1e-5,
      f'max dev {(out0 - out1).abs().max().item():.2e}')

# V4 bag per-token form + item-dependent removed mass
raw = mk(bag=True); cen = mk(bag=True, center=True); copy_table(raw, cen)
E = cen.embedding_layer.weight
counts = torch.zeros(n_features, dtype=torch.float64)
counts.scatter_add_(0, cen.feature_ids.reshape(-1), cen.feature_weights.reshape(-1).double())
mu = torch.zeros(2, d, dtype=torch.float64)
mu[0] = counts[:3] / counts[:3].sum() @ E[:3].double()
mu[1] = counts[3:] / counts[3:].sum() @ E[3:5].double()
manual = ((E[cen.feature_ids].double() - mu[field_index[cen.feature_ids]])
          * cen.feature_weights.unsqueeze(-1).double()).sum(-2).float() + E[n_features + o]
check('V4 bag per-token form', (cen(o) - manual).abs().max().item() < TOL,
      f'max dev {(cen(o) - manual).abs().max().item():.2e}')
removed = raw(o) - cen(o)
check('V4 removed mass item-dependent', (removed[0] - removed[1]).abs().max().item() > 1e-3)

# V5 gauge_snap inertness + post-snap means ~0
cen = mk(center=True)
out0 = cen(o).detach().clone()
cen.gauge_snap()
out1 = cen(o)
check('V5 snap function-inert', (out0 - out1).abs().max().item() < 1e-5,
      f'max dev {(out0 - out1).abs().max().item():.2e}')
post_mu = cen.field_probs @ cen.embedding_layer.weight[:n_features]
check('V5 post-snap means ~0', post_mu.abs().max().item() < 1e-6,
      f'max |mu| {post_mu.abs().max().item():.2e}')

# V6 metadata_scale
raw = mk(); al = mk(scale=0.35); copy_table(raw, al)
E = al.embedding_layer.weight
manual = 0.35 * E[al.feature_ids].sum(-2) + E[n_features + o]
check('V6 metadata_scale exact', (al(o) - manual).abs().max().item() < TOL)

# V7 interaction-mode field_probs
iw = torch.tensor([5., 0., 2., 1., 0., 3.])  # per-item train counts (two zero-count items)
m = mk(center=True, mode='interaction', iw=iw)
exp = torch.zeros(n_features, dtype=torch.float64)
for i in range(n_items):
    for f in range(2):
        exp[feature_ids_fixed[i, f]] += float(iw[i])
p_exp = torch.zeros(2, n_features, dtype=torch.float64)
p_exp[0, :3] = exp[:3] / exp[:3].sum(); p_exp[1, 3:] = exp[3:] / exp[3:].sum()
check('V7 interaction field_probs', (m.field_probs.double() - p_exp).abs().max().item() < TOL,
      f'probs row0 {m.field_probs[0].tolist()}')

print('\n' + ('ALL PASS' if not fails else f'FAILURES: {fails}'))
sys.exit(1 if fails else 0)
