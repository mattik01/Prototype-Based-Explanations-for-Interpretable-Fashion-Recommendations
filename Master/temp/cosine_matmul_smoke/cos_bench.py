"""Micro-benchmark: ProtoMF's broadcast cosine vs normalise-then-matmul, at the tuned
I-ProtoMF ML-1M shape. Standalone (torch only). Reports time, peak GPU memory, and the
largest difference in outputs and gradients."""
import time, torch
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| device', dev)
cos = torch.nn.CosineSimilarity(dim=-1)

def old(o, p):   # as in PrototypeEmbedding.sim_from_embed (cosine_type='shifted')
    return 1 + cos(o.unsqueeze(-2), p)

def new(o, p):
    return 1 + F.normalize(o, dim=-1) @ F.normalize(p, dim=-1).T

def run(fn, o, p, reps):
    if dev == 'cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(reps):
        o.grad = p.grad = None
        s = fn(o, p)
        # same shape of work as training: the two regularisers + a use of the scores
        loss = s.sum() - s.reshape(-1, s.shape[-1]).max(0).values.mean() \
               - s.reshape(-1, s.shape[-1]).max(1).values.mean()
        loss.backward()
    if dev == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev == 'cuda' else float('nan')
    return dt, mem, s.detach(), o.grad.clone(), p.grad.clone()

for label, (B, N, K, d) in {'tuned I-ProtoMF (B=64, 49 slots, K=99, d=99)': (64, 49, 99, 99),
                            'largest in search (B=512, 50 slots, K=99, d=99)': (512, 50, 99, 99)}.items():
    torch.manual_seed(0)
    o = torch.randn(B, N, d, device=dev, requires_grad=True)
    p = torch.randn(K, d, device=dev, requires_grad=True)
    reps = 20 if B == 64 else 5
    run(old, o, p, 2); run(new, o, p, 2)                      # warm-up
    t0, m0, s0, go0, gp0 = run(old, o, p, reps)
    t1, m1, s1, go1, gp1 = run(new, o, p, reps)
    print(f'\n{label}')
    print(f'  current : {t0*1e3:8.2f} ms / step   peak GPU mem {m0:8.1f} MB')
    print(f'  matmul  : {t1*1e3:8.2f} ms / step   peak GPU mem {m1:8.1f} MB')
    print(f'  speed-up x{t0/t1:.1f}   memory x{m0/m1:.1f}')
    print(f'  max |diff|: scores {float((s0-s1).abs().max()):.2e}, '
          f'grad objects {float((go0-go1).abs().max()):.2e}, grad prototypes {float((gp0-gp1).abs().max()):.2e}')
