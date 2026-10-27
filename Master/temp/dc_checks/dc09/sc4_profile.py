# dc09 SC.4 optimisation audit — one-off, descriptive: where a training trial spends its time.
# Login node, one GPU, hm_1_month + ml-1m, the matched bench configs (sc4_epoch_bench.py: d=64,
# K=50, B=256, 50 popular negatives, sampled softmax, Adam). Stages, each timed on its own:
#   A. the inherited negative sampler per sample (popular / uniform), in the main process;
#   B. DataLoader throughput at num_workers 0 / 1 / 2 (the fleet runs 1; upstream ran 2);
#   C. one optimiser step on the GPU from pre-generated batches (forward + loss + backward + step),
#      per model — the model's own cost, with the input pipeline removed;
#   D. trial set-up: train-dataset construction, feature injection, factory build;
#   E. one validation pass (99 uniform negatives per user, the eval regime).
import copy
import json
import os
import sys
import time

import numpy as np
import torch

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_ids import inject_feature_ids, resolve_canonical_fields
from rec_sys.protomf_dataset import ProtoRecDataset, get_protorecdataset_dataloader
from rec_sys.rec_sys import RecSys
from utilities.consts import DATA_PATH, NEG_VAL
from utilities.eval import Evaluator

BENCH = '/scratch/c7031336/smoke_runs/dc09/sc4_bench'
MODELS = ['user_item_proto', 'feature_user_proto', 'feature_item_proto', 'feature_user_item_proto',
          'feature_user_item_proto_sep']
DEV = 'cuda'
B, N_NEG = 256, 50


def conf_for(model, dataset):
    c = json.load(open(os.path.join(BENCH, model, 'config.json')))
    fields, layout = resolve_canonical_fields(dataset)
    for side in ('user_ft_ext_param', 'item_ft_ext_param'):
        if 'feature_fields' in c['ft_ext_param'].get(side, {}):
            c['ft_ext_param'][side].update(feature_fields=fields, feature_layout=layout)
    return c


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


report = {}
for dataset in ('hm_1_month', 'ml-1m'):
    dp = os.path.join(DATA_PATH, dataset)
    r = report[dataset] = {}
    np.random.seed(0)
    ds, t_ds = timed(ProtoRecDataset, dp, 'train', N_NEG, 'popular')
    r['n_train_rows'], r['n_items'], r['n_batches'] = len(ds), ds.n_items, int(np.ceil(len(ds) / B))
    r['D.train_dataset_build_s'] = t_ds
    # A. sampler per sample
    users = ds.coo_matrix.row[np.random.randint(0, len(ds), 2000)]
    for strat, fn in (('popular', ds._neg_sample_popular), ('uniform', ds._neg_sample_uniform)):
        t = time.perf_counter()
        for u in users:
            fn(u)
        r[f'A.sampler_{strat}_us_per_sample'] = (time.perf_counter() - t) / len(users) * 1e6
    r['A.popular_sampling_s_per_epoch_one_process'] = r['A.sampler_popular_us_per_sample'] * len(ds) / 1e6
    # B. loader throughput
    for nw in (0, 1, 2):
        dl = get_protorecdataset_dataloader(dp, 'train', N_NEG, 'popular', batch_size=B, shuffle=True,
                                            num_workers=nw, prefetch_factor=5 if nw > 0 else None)
        it = iter(dl)
        next(it)  # worker start-up excluded
        t = time.perf_counter()
        for _ in range(60):
            next(it)
        r[f'B.loader_s_per_batch[workers={nw}]'] = (time.perf_counter() - t) / 60
        r[f'B.loader_s_per_epoch[workers={nw}]'] = r[f'B.loader_s_per_batch[workers={nw}]'] * r['n_batches']
        del it, dl
    # pre-generated batches for C
    dl = get_protorecdataset_dataloader(dp, 'train', N_NEG, 'popular', batch_size=B, shuffle=True, num_workers=0)
    batches = [b for _, b in zip(range(40), dl)]
    for model in MODELS:
        c = conf_for(model, dataset)
        ft, t_inj = timed(inject_feature_ids, copy.deepcopy(c['ft_ext_param']), dp)
        (ufe, ife), t_fac = timed(FeatureExtractorFactory.create_models, ft, ds.n_users, ds.n_items)
        rec = RecSys(ds.n_users, ds.n_items, c['rec_sys_param'], ufe, ife, c['loss_func_name'],
                     c.get('loss_func_aggr', 'mean'))
        rec.init_parameters()
        rec = rec.to(DEV).train()
        opt = torch.optim.Adam(rec.parameters(), lr=1e-3, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        for k, (u, i, lab) in enumerate(batches):
            if k == 5:  # warm-up excluded
                torch.cuda.synchronize()
                t = time.perf_counter()
            u, i, lab = u.to(DEV), i.to(DEV), lab.to(DEV)
            loss = rec.loss_func(rec(u, i), lab)
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        step = (time.perf_counter() - t) / (len(batches) - 5)
        # E. one validation pass (fresh val loader, the trainer's settings)
        rec.eval()
        vl = get_protorecdataset_dataloader(dp, 'val', NEG_VAL, 'uniform', batch_size=256, num_workers=1)
        ev = Evaluator(vl.dataset.coo_matrix.nnz)
        t = time.perf_counter()
        with torch.no_grad():
            for u, i, _ in vl:
                ev.eval_batch(torch.sigmoid(rec(u.to(DEV), i.to(DEV))).cpu().numpy())
        t_val = time.perf_counter() - t
        r[model] = {'D.inject_s': t_inj, 'D.factory_s': t_fac,
                    'C.gpu_step_ms': step * 1e3, 'C.gpu_s_per_epoch': step * r['n_batches'],
                    'C.peak_vram_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
                    'E.val_pass_s': t_val,
                    'n_params': sum(p.numel() for p in rec.parameters())}
        print(dataset, model, {k: round(v, 3) if isinstance(v, float) else v for k, v in r[model].items()}, flush=True)
        del rec, opt, ufe, ife, ft
        torch.cuda.empty_cache()
    print(dataset, {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items() if not isinstance(v, dict)}, flush=True)

json.dump(report, open(os.path.join(BENCH, 'profile.json'), 'w'), indent=2)
print('written', os.path.join(BENCH, 'profile.json'))
