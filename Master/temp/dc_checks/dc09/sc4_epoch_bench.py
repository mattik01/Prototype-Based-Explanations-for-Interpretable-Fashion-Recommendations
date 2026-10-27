# dc09 SC.4 optimisation parity — one-off, descriptive: per-epoch training time and peak VRAM of
# fUfI against its hosts under ONE matched config (d=64, K=50 per prototype side, B=256, 50
# popular negatives, sampled softmax, Adam 1e-3 / wd 1e-4), lone trial, hm_1_month, same GPU,
# run one after another (trials are CPU-bound, so parallel runs would distort each other).
# Step 1 (this script, `make`): write one matched config.json per model from that model's own
# saved config (so each keeps its exact ft_ext_param structure). Step 2: run_combo
# --retrain-config <bench>/<model>/config.json --n-epochs 2 per model (sc4_epoch_bench.sh).
# Step 3 (`read`): epoch times from the logs, VRAM from hardware_summary.json.
import glob
import json
import os
import re
import sys

BENCH = '/scratch/c7031336/smoke_runs/dc09/sc4_bench'
SOURCES = {
    'user_item_proto': '/scratch/c7031336/protomf_results/user_item_proto_hm_1_month_s38210573_n100/config.json',
    'feature_user_proto': '/scratch/c7031336/protomf_results/feature_user_proto_hm_1_month_s38210573_n100/config.json',
    'feature_item_proto': '/scratch/c7031336/protomf_results/feature_item_proto_hm_1_month_s38210573_n100/config.json',
    'feature_user_item_proto': '/scratch/c7031336/smoke_runs/dc09/results/feature_user_item_proto_hm_1_month_s38210573/config.json',
    'feature_user_item_proto_sep': '/scratch/c7031336/smoke_runs/dc09/results/feature_user_item_proto_sep_hm_1_month_s38210573/config.json',
}


def make():
    for model, src in SOURCES.items():
        c = json.load(open(src))
        c.update(batch_size=256, neg_train=50, train_neg_strategy='popular', loss_func_name='sampled_softmax',
                 loss_func_aggr='mean', optim_param={'optim': 'adam', 'lr': 1e-3, 'wd': 1e-4})
        f = c['ft_ext_param']
        f['embedding_dim'] = 64
        for side in ('user_ft_ext_param', 'item_ft_ext_param'):
            p = f.get(side, {})
            if 'n_prototypes' in p:
                p.update(n_prototypes=50, sim_proto_weight=0.1, sim_batch_weight=0.1)
        os.makedirs(os.path.join(BENCH, model), exist_ok=True)
        json.dump(c, open(os.path.join(BENCH, model, 'config.json'), 'w'), indent=2)
        print(model, {s: f.get(s, {}).get('ft_type') for s in ('user_ft_ext_param', 'item_ft_ext_param')})


def read():
    for model in SOURCES:
        log = os.path.join(BENCH, 'logs', model + '.log')
        txt = open(log).read() if os.path.exists(log) else ''
        ep = [int(x) for x in re.findall(r'Epoch \d+ - Train Loss [-0-9.e]+ - (\d+)s', txt)]
        hw = glob.glob(os.path.join(BENCH, 'results', model + '_hm_1_month_s38210573', 'hardware_summary.json'))
        vram = json.load(open(hw[0])).get('max_vram_mib') if hw else None
        print('%-30s epoch s %s | peak VRAM %s MiB' % (model, ep, None if vram is None else round(vram)))


if __name__ == '__main__':
    {'make': make, 'read': read}[sys.argv[1]]()
