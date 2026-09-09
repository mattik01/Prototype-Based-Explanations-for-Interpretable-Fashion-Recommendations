# dc08 t04 — what the flat embedding_bag pooling actually buys at ml-1m scale (5b A10).
# NOT a correctness test (t03 pins equivalence); this is the evidence behind the claim that the
# padded gather is the wrong data structure once dc08's identity slots widen the bag.
# Synthetic ml-1m-shaped history: N=6034 users, mean |H_u| ~93 with a heavy tail (real ml-1m
# power users rate thousands of films), ~300 distinct word slots per user, M=3125 items,
# d=100 (the top of the searched range), B=512 (the top of the searched batch sizes).
# Run: PYTHONPATH=. conda run -n protomf python Master/temp/dc_checks/dc08/t04_scale_benchmark.py
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import torch

from feature_extraction.feature_extractors import HistoryFeatureEmbedding

torch.manual_seed(0)
N, V, M, d, B, W = 6034, 400, 3125, 100, 512, 300
# lognormal history lengths: median ~70, heavy tail to ~2000 (ml-1m's shape, not its exact values)
lens = torch.clamp((torch.distributions.LogNormal(4.25, 0.95).sample((N,))).long(), 5, 2000)
D_max = int(lens.max()) + W
ids = torch.zeros(N, D_max, dtype=torch.long)
w = torch.zeros(N, D_max)
for u in range(N):
    L = int(lens[u]) + W
    ids[u, :L] = torch.randint(0, V + M, (L,))
    w[u, :L] = 1.0 / float(lens[u])
tok, tokw, sz = torch.randint(0, V + M, (M, 25)), torch.ones(M, 25), lens.float()

real = int((w != 0).sum())
print(f"history lengths: mean {float(lens.float().mean()):.0f}, median {int(lens.median())}, "
      f"max {int(lens.max())}")
print(f"D_max = {D_max} padded columns | real slots = {real/1e6:.2f} M "
      f"({100*real/(N*D_max):.1f}% fill, {100-100*real/(N*D_max):.1f}% padding)")
print(f"padded buffers (ids+w) = {N*D_max*12/1e6:.0f} MB")

out, timing = {}, {}
ref_state = None
for mode in ('padded', 'bag'):
    m = HistoryFeatureEmbedding(N, ids, w, V, d, use_id_feature=False, item_token_ids=tok,
                                item_token_w=tokw, hist_sizes=sz, loo_pooling=True,
                                n_history_rows=M, history_row_weight=1.0, pooling=mode)
    if ref_state is None:
        ref_state = {k: v.clone() for k, v in m.state_dict().items()}
    else:
        m.load_state_dict(ref_state)          # identical weights, so the outputs are comparable
    m.train()
    u, p = torch.arange(B), torch.randint(0, M, (B,))
    m(u[:8], pos_i_idxs=p[:8])                # warm-up
    t0 = time.time()
    q = m(u, pos_i_idxs=p)
    q.pow(2).sum().backward()
    timing[mode] = time.time() - t0
    # Agreement is reported on the COMPOSITION (eval forward), not on the leave-one-out output:
    # this synthetic bag is not consistent with the synthetic item-token table, so the LOO step
    # ((|H|·q − e_pos)/(|H|−1), |H| up to 2000 here) is an arbitrary near-cancellation that
    # amplifies float noise by |H|. Equivalence on real builder output is pinned in t03.
    m.eval()
    with torch.no_grad():
        out[mode] = m(u).detach()
    peak = B * D_max * d * 4 / 1e6 if mode == 'padded' else 0.0
    print(f"{mode:7s}: fwd+bwd {timing[mode]:.3f}s | retained (B, D_max, d) fp32 intermediate "
          f"= {peak:,.0f} MB" + ("" if mode == 'padded' else "  (fused — none)"))

rel = float((out['padded'] - out['bag']).abs().max() / out['padded'].abs().max())
print(f"\nspeedup {timing['padded']/timing['bag']:.1f}x | outputs agree to {rel:.2e} relative "
      f"(same slots, different reduction order)")
print("NOTE: the retained-intermediate figure is the forward activation alone; autograd holds a "
      "same-shaped gradient during backward, so the padded path's true peak is roughly double.")
