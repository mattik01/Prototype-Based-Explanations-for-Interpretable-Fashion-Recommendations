"""Follow-up probes on the edges of C7/C8/C9/C11 that the main battery
only touched at one configuration."""
import numpy as np, torch
V, M, N, d, K = 11, 9, 7, 5, 4
def wr(f): return f
def ir(j): return V + j
def ur(u): return V + M + u

# ---- C8: is the LOO gradient on the removed positive EXACTLY zero for all n,
#      in float64 and float32?
print("C8 edge: |grad y_pos| under LOO, by |H_u| and dtype")
for dtype in (torch.float64, torch.float32):
    row = []
    for n in range(2, 9):
        g = torch.Generator().manual_seed(n)
        E = torch.randn(V+M+N, d, generator=g, dtype=dtype, requires_grad=True)
        P = torch.randn(K, d, generator=g, dtype=dtype)
        t = torch.randn(K, generator=g, dtype=dtype)
        Hu = list(range(n)); pos = 0; lam = 1.0
        fb = {j: [(j*3) % V, (j*5+1) % V] for j in Hu}
        ids, ws = [], []
        cnt = {}
        for j in Hu:
            for f in fb[j]: cnt[f] = cnt.get(f, 0) + 1
        for f in sorted(cnt): ids.append(wr(f)); ws.append(cnt[f]/n)
        for j in Hu: ids.append(ir(j)); ws.append(lam/n)
        idst = torch.tensor(ids); wst = torch.tensor(ws, dtype=dtype)
        qm = (wst[:, None]*E[idst]).sum(0)
        e_pos = sum(E[wr(f)] for f in fb[pos]) + lam*E[ir(pos)]
        qm = (n*qm - e_pos)/(n-1)
        q = qm + E[ur(0)]
        cl = torch.nn.functional.cosine_similarity(q[None, :], P, dim=1)
        (t*(1+cl)).sum().backward()
        gp = E.grad[ir(pos)].abs().max().item()
        gw = E.grad[wr(fb[pos][0])].abs().max().item()  # word shared? check
        row.append(f"n={n}:{gp:.1e}")
    print("  ", dtype, " ".join(row))

# ---- C9: family at lambda_y = 0, and dimension of the family
print("\nC9 edge:")
r = np.random.default_rng(0)
E = r.normal(size=(V+M+N, d)); Hu = [0,1,2]; lam = 0.0
fb = {j: [(j*3) % V] for j in range(M)}
def q(EE, lam):
    acc = np.zeros(d)
    for j in Hu:
        for f in fb[j]: acc += EE[wr(f)]
        acc += lam*EE[ir(j)]
    return acc/len(Hu) + EE[ur(0)]
delta = r.normal(size=d)
E2 = E.copy(); E2[ur(0)] += delta
print(f"   lam=0: no finite compensation exists (1/lam undefined); "
      f"drift if user row alone is moved = {np.abs(q(E2,0.)-q(E,0.)).max():.3e}")
# dimension: delta ranges over R^d -> family dimension
print(f"   family dimension = d = {d} (delta is a free vector in R^d), "
      f"not 1 as the spec words it")

# ---- C11: float32 torch cosine with zero vector, and eps sensitivity
print("\nC11 edge: torch.cosine_similarity(0, P)")
for dtype in (torch.float64, torch.float32):
    P = torch.randn(K, d, dtype=dtype)
    z = torch.zeros(K, d, dtype=dtype)
    c = torch.nn.functional.cosine_similarity(z, P, dim=1)
    print(f"   {dtype}: {c.numpy()}  max|c|={c.abs().max().item():.3e}")
# tiny-but-nonzero q
for s in [1e-4, 1e-8, 1e-12, 1e-20, 1e-30]:
    P = torch.randn(K, d, dtype=torch.float32)
    qv = torch.full((K, d), s, dtype=torch.float32)
    c = torch.nn.functional.cosine_similarity(qv, P, dim=1)
    print(f"   q=const {s:g} (fp32): cos={c.numpy()}")

# ---- C7: |H_u| = 1 makes alpha inert even with use_id=True
print("\nC7 edge: alpha with |H_u|=1")
E = r.normal(size=(V+M+N, d)); P = r.normal(size=(K, d)); lam = 1.0
Hu1 = [4]
def qa(alpha, use_id):
    den = len(Hu1)**alpha
    acc = np.zeros(d)
    for j in Hu1:
        for f in fb[j]: acc += E[wr(f)]
        acc += lam*E[ir(j)]
    v = acc/den
    return v + (E[ur(0)] if use_id else 0)
dif = max(abs(float(qa(1.,True)@P[l]/(np.linalg.norm(qa(1.,True))*np.linalg.norm(P[l])))
             - float(qa(2.,True)@P[l]/(np.linalg.norm(qa(2.,True))*np.linalg.norm(P[l]))))
          for l in range(K))
print(f"   |H|=1, use_id=True, alpha 1 vs 2: max dcos = {dif:.3e} (1^alpha==1, so inert)")
