# First-party check of the verifier's counter-map T_c (shared ids arm), independent code.
import numpy as np
rng = np.random.default_rng(7)
V, M, N, d, Ku, Kt, F = 12, 9, 5, 6, 4, 3, 3
fields = np.array_split(rng.permutation(V), F)
items = [[rng.choice(f) for f in fields] for _ in range(M)]          # fixed layout
baskets = [rng.choice(M, size=rng.integers(2, 6), replace=False) for _ in range(N)]
Ew, EI, EU = rng.normal(size=(V, d)), rng.normal(size=(M, d)), rng.normal(size=(N, d))
Pu, Pt = rng.normal(size=(Ku, d)), rng.normal(size=(Kt, d))
Wu, Wt = rng.normal(size=(Kt, d)), rng.normal(size=(Ku, d))
sim = lambda a, P: 1 + (P @ a) / (np.linalg.norm(P, axis=1) * np.linalg.norm(a))
def qi(Ew, EI, i): return Ew[items[i]].sum(0) + EI[i]
def qu(Ew, EU, u, drop=None):
    b = [j for j in baskets[u] if j != drop]
    return np.mean([Ew[items[j]].sum(0) for j in b], 0) + EU[u]
def S(p, u, i, drop=None):
    Ew, EI, EU, Wu, Wt = p
    a, b = qu(Ew, EU, u, drop), qi(Ew, EI, i)
    return sim(a, Pu) @ (Wt @ b) + (Wu @ a) @ sim(b, Pt)
p = (Ew, EI, EU, Wu, Wt)
for c in (0.2, 3.0):
    m = np.stack([np.mean([Ew[items[j]].sum(0) for j in baskets[u]], 0) for u in range(N)])
    pc = (Ew / c, EI / c, c * EU + (c - 1 / c) * m, Wu / c, c * Wt)
    full = max(abs(S(p, u, i) - S(pc, u, i)) for u in range(N) for i in range(M))
    loo = max(abs(S(p, u, i, drop=i) - S(pc, u, i, drop=i)) for u in range(N) for i in baskets[u])
    r0 = np.linalg.norm(Wt) / np.linalg.norm(Wu); r1 = np.linalg.norm(pc[4]) / np.linalg.norm(pc[3])
    # user-side word share of q_u (norm fraction of the word part) before/after
    u = 0; w0 = np.linalg.norm(m[u]) / np.linalg.norm(qu(Ew, EU, u)); w1 = np.linalg.norm(m[u] / c) / np.linalg.norm(qu(pc[0], pc[2], u))
    print(f"c={c}: max|dS| full-basket {full:.1e} | LOO {loo:.1e} | ratio x{r1/r0:.3f} (c^2={c*c:.3f}) | user0 word-part fraction {w0:.3f} -> {w1:.3f}")
