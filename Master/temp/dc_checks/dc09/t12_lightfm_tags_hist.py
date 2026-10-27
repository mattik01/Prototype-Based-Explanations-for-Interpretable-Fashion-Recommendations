# dc09 t12 — the two-sided decoupled control `lightfm_tags_hist(_ids)` (SC.3 re-run 2026-10-27,
# design doc §3.7 S1; charter C6 amendment). fUfI's two compositions under a plain dot product:
#   1. keystone — zero words, ID rows on both sides, weight-copied ≡ `mf`: scores bitwise;
#   2. the score is the plain dot q_u·q_i of the two compositions (float64 recomposition from the
#      table rows), and the model owns no prototype or projection parameters;
#   3. sharing keystone — shared ≡ `_sep` with the shared rows copied into both tables: scores
#      bitwise; the shared word-row gradient = the sum of the two sides' word-row gradients;
#   4. leave-one-out reaches the user composition in training (RecSys routes the positive) and is
#      off in eval;
#   5. registration — all four names (shared + `_sep`) in start.py / run_combo, configs mirror the lightfm rows plus the
#      declared keys, canonical fields resolve on both sides, injection fills both payloads, the
#      config is never mutated;
#   6. cold — both ID blocks are imputation targets at the right offsets (shared table: items at
#      V + i, users at V + M + u), `config_expects_*` know the ft_type, noid is a no-op;
#   7. checkpoint round-trip keeps the shared table one Parameter;
#   8. the factory guards (one vocabulary index, no dc08 rows) hold here too.
import copy
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from _fixture import (D, N_ITEMS, N_USERS, build, item_payload, make_check, toy_world, train_batch,
                      user_payload, zero_vocab_world)
from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_ids import resolve_feature_fields_in_conf
from rec_sys.rec_sys import RecSys
from utilities.cold_eval import apply_cold_id_convention, config_expects_id_drop, config_expects_user_id_drop

check, FAILS = make_check()
w = toy_world()
V = w['V']


def param_ctrl(world, share=True, ids=True, layout='fixed'):
    return {'ft_type': 'lightfm_tags_hist', 'embedding_dim': D, 'share_word_table': share,
            'item_ft_ext_param': {'ft_type': 'lightfm_tags_hist', **item_payload(world, ids, layout)},
            'user_ft_ext_param': {'ft_type': 'lightfm_tags_hist', **user_payload(world, ids)}}


# 1. keystone vs mf
z = zero_vocab_world()
ctrl = build(param_ctrl(z, share=True, ids=True)).eval()
mf = build({'ft_type': 'detached', 'embedding_dim': D, 'user_ft_ext_param': {'ft_type': 'embedding'},
            'item_ft_ext_param': {'ft_type': 'embedding'}}).eval()
T = ctrl.item_feature_extractor.embedding_layer.weight.data  # [item IDs ; user IDs] (V = 0)
mf.item_feature_extractor.embedding_layer.weight.data.copy_(T[:N_ITEMS])
mf.user_feature_extractor.embedding_layer.weight.data.copy_(T[N_ITEMS:])
u, i, lab = train_batch(w)
with torch.no_grad():
    check('t12.keystone_zero_words_ids_equals_mf', torch.equal(ctrl(u, i), mf(u, i)))

# 2. plain dot of the two compositions, no prototype machinery
for share in (True, False):
    tag = 'shared' if share else 'sep'
    rec = build(param_ctrl(w, share=share)).eval()
    ufe, ife = rec.user_feature_extractor, rec.item_feature_extractor
    Ti, Tu = ife.embedding_layer.weight.detach().double(), ufe.embedding_layer.weight.detach().double()
    q_i = Ti[w['feature_ids'][i]].sum(-2) + Ti[V + i]
    hv, hw = w['hist_value_ids'][u], w['hist_weights'][u].double()
    q_u = (Tu[hv] * hw.unsqueeze(-1)).sum(-2) + Tu[ufe.id_row_offset + u]
    with torch.no_grad():
        s = rec(u, i).double()
    err = ((s - (q_u.unsqueeze(1) * q_i).sum(-1)).abs().max() / s.abs().max()).item()
    check(f't12.{tag}.plain_dot_of_compositions', err < 1e-6, f'max rel {err:.1e}')
    names = [n for n, _ in rec.named_parameters()]
    check(f't12.{tag}.no_prototype_or_projection_parameters',
          not any('proto' in n or 'linear' in n or 'weight_matrix' in n for n in names), str(names))
    check(f't12.{tag}.user_id_offset', ufe.id_row_offset == V + (N_ITEMS if share else 0),
          f'{ufe.id_row_offset}')

# 3. sharing keystone
sh = build(param_ctrl(w, share=True))
sp = build(param_ctrl(w, share=False))
Tsh = sh.item_feature_extractor.embedding_layer.weight.data
sp.item_feature_extractor.embedding_layer.weight.data.copy_(Tsh[:V + N_ITEMS])
sp.user_feature_extractor.embedding_layer.weight.data.copy_(torch.cat([Tsh[:V], Tsh[V + N_ITEMS:]]))
for r in (sh, sp):
    r.train()
    r.zero_grad()
s_sh, s_sp = sh(u, i), sp(u, i)
check('t12.sharing_keystone_scores_bitwise', torch.equal(s_sh, s_sp))
sh.loss_func(s_sh, lab).backward()
sp.loss_func(s_sp, lab).backward()
g_sh = sh.item_feature_extractor.embedding_layer.weight.grad
g_i = sp.item_feature_extractor.embedding_layer.weight.grad
g_u = sp.user_feature_extractor.embedding_layer.weight.grad
err = (g_sh[:V] - (g_i[:V] + g_u[:V])).abs().max().item()
check('t12.sharing_keystone_word_grad_is_sum_of_sides', err < 1e-6
      and torch.allclose(g_sh[V:V + N_ITEMS], g_i[V:]) and torch.allclose(g_sh[V + N_ITEMS:], g_u[V:]),
      f'max abs {err:.1e}')

# 4. leave-one-out routing
rec = build(param_ctrl(w)).train()
ufe, ife = rec.user_feature_extractor, rec.item_feature_extractor
with torch.no_grad():
    s_train = rec(u, i)
    expect = (ufe(u, pos_i_idxs=i[:, 0]).unsqueeze(1) * ife(i)).sum(-1)
    rec.eval()
    s_eval = rec(u, i)
check('t12.loo_reaches_user_composition_in_training', torch.allclose(s_train, expect)
      and ufe.supports_loo and not torch.allclose(s_train, s_eval))

# 5. registration + configs
sys.argv = ['run_combo']
import Master.scripts.run_combo as rc  # noqa: E402
from confs import hyper_params as hp  # noqa: E402

def canon(x):
    """Comparable form of a config leaf (Ray Tune domains have no value equality; as in t04)."""
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    if hasattr(x, 'sampler') or hasattr(x, 'categories'):
        smp = getattr(x, 'sampler', None)
        return (type(x).__name__, getattr(x, 'lower', None), getattr(x, 'upper', None),
                tuple(getattr(x, 'categories', ()) or ()), type(smp).__name__, getattr(smp, 'base', None))
    return x


start_src = open(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'start.py')).read()
for name, ids, share in (('lightfm_tags_hist', False, True), ('lightfm_tags_hist_ids', True, True),
                         ('lightfm_tags_hist_sep', False, False), ('lightfm_tags_hist_ids_sep', True, False)):
    check(f't12.registered.{name}', name in rc.MODEL_CONFIGS
          and f"'{name}'" in start_src and f"model == '{name}'" in start_src)
    conf = rc.MODEL_CONFIGS[name]
    f = conf['ft_ext_param']
    ref = hp.lightfm_hist_ids_hyper_params  # the one-sided twin: same search space outside ft_ext_param
    same_base = set(conf) == set(ref) and all(canon(conf[k]) == canon(ref[k]) for k in ref if k != 'ft_ext_param')
    check(f't12.{name}.search_space_mirrors_lightfm_rows', same_base
          and canon(f['embedding_dim']) == canon(ref['ft_ext_param']['embedding_dim']))
    check(f't12.{name}.arm_flags', f['share_word_table'] is share
          and f['item_ft_ext_param']['use_id_feature'] is ids and f['user_ft_ext_param']['use_id_feature'] is ids
          and f['user_ft_ext_param']['loo_pooling'] is True
          and {f['ft_type'], f['item_ft_ext_param']['ft_type'], f['user_ft_ext_param']['ft_type']}
          == {'lightfm_tags_hist'})
    for ds in ('hm_1_month', 'ml-1m'):
        c = copy.deepcopy(conf)
        resolve_feature_fields_in_conf(c, ds)
        it, us = c['ft_ext_param']['item_ft_ext_param'], c['ft_ext_param']['user_ft_ext_param']
        check(f't12.{name}.{ds}.canonical_resolves_identically_both_sides',
              it['feature_fields'] != 'canonical' and it['feature_fields'] == us['feature_fields']
              and it['feature_layout'] == us['feature_layout'])
    check(f't12.{name}.cold_expectations', config_expects_id_drop(conf) is ids
          and config_expects_user_id_drop(conf) is ids)
pc = param_ctrl(w)
before = copy.deepcopy({k: v for k, v in pc.items() if not isinstance(v, dict)})
before_sub = {s: {k: v for k, v in pc[s].items() if not torch.is_tensor(v)}
              for s in ('item_ft_ext_param', 'user_ft_ext_param')}
FeatureExtractorFactory.create_models(pc, N_USERS, N_ITEMS)
check('t12.factory_does_not_mutate_config',
      {k: v for k, v in pc.items() if not isinstance(v, dict)} == before
      and all({k: v for k, v in pc[s].items() if not torch.is_tensor(v)} == before_sub[s] for s in before_sub))

# 6. cold — imputation targets on both ID blocks of the shared table
tmp = tempfile.mkdtemp()
COLD_I, COLD_U = [1, 7], [2]
pd.DataFrame({'item_id': COLD_I}).to_csv(os.path.join(tmp, 'cold_items.csv'), index=False)
pd.DataFrame({'user_id': COLD_U}).to_csv(os.path.join(tmp, 'cold_users.csv'), index=False)
rec = build(param_ctrl(w)).eval()
W = rec.item_feature_extractor.embedding_layer.weight
before = W.detach().clone()
info_i, info_u = apply_cold_id_convention(rec, tmp, 'impute_mean')
warm_i = [x for x in range(N_ITEMS) if x not in COLD_I]
warm_u = [x for x in range(N_USERS) if x not in COLD_U]
uoff = rec.user_feature_extractor.id_row_offset
ok_items = all(torch.equal(W[V + c], before[torch.tensor(warm_i) + V].mean(0)) for c in COLD_I)
ok_users = all(torch.equal(W[uoff + c], before[torch.tensor(warm_u) + uoff].mean(0)) for c in COLD_U
               if w['baskets'][c])
changed = sorted((W != before).any(1).nonzero().flatten().tolist())
check('t12.cold_imputation_both_blocks', ok_items and ok_users and uoff == V + N_ITEMS
      and changed == sorted([V + c for c in COLD_I] + [uoff + c for c in COLD_U]),
      f'{info_i} / {info_u} / changed {changed}')
rec_noid = build(param_ctrl(w, ids=False)).eval()
check('t12.cold_noid_is_noop', apply_cold_id_convention(rec_noid, tmp, 'impute_mean') == (None, None))

# 7. checkpoint round-trip
rec = build(param_ctrl(w)).eval()
path = os.path.join(tmp, 'ckpt.pth')
torch.save(rec.state_dict(), path)
rec2 = build(param_ctrl(w)).eval()
rec2.load_state_dict(torch.load(path, weights_only=True))
with torch.no_grad():
    same_scores = torch.equal(rec(u, i), rec2(u, i))
check('t12.checkpoint_roundtrip', same_scores
      and rec2.item_feature_extractor.embedding_layer.weight is rec2.user_feature_extractor.embedding_layer.weight
      and len(list(rec2.parameters())) == 1)

# 8. factory guards
bad = param_ctrl(w)
bad['item_ft_ext_param']['feature_fields'], bad['user_ft_ext_param']['feature_fields'] = ['a'], ['b']
try:
    build(bad)
    check('t12.vocabulary_identity_enforced', False, 'built')
except AssertionError:
    check('t12.vocabulary_identity_enforced', True)
bad = param_ctrl(w)
bad['user_ft_ext_param']['n_history_rows'] = N_ITEMS
try:
    build(bad)
    check('t12.dc08_rows_rejected', False, 'built')
except AssertionError:
    check('t12.dc08_rows_rejected', True)

print("\nt12: ALL PASS" if not FAILS else f"\nt12: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
