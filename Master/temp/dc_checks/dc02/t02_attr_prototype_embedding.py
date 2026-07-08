# t02 — AttributePrototypeEmbedding, BASE mode (dc02 P3):
#  - forward == manual 1+cos
#  - HOST-EQUIVALENCE KEYSTONE: weight-copied base-mode instance vs raw host
#    PrototypeEmbedding(embedding_dim=V, embedding_ext=AttributeLookup) -> bitwise-equal outputs
#    AND equal get_and_reset_loss (proves the forward mirror + reg transplant are host-faithful)
#  - claim 3 (exact per-attribute decomposition), claim 7 (regularizer directions)
import math
import sys

import torch

from _harness import make_check, toy_attr_data

from feature_extraction.feature_extractors import (AttributeLookup, AttributePrototypeEmbedding,
                                                   PrototypeEmbedding)

torch.manual_seed(7)
check, fails = make_check()

M, V_fs, K = 12, (3, 4, 5), 5
X, offs = toy_attr_data(M=M, V_fs=V_fs, seed=0)
V = X.shape[1]
F = len(V_fs)


def build(sim_proto_weight=1.0, sim_batch_weight=1.0):
    return AttributePrototypeEmbedding(
        M, AttributeLookup(X), K, offs,
        sim_proto_weight=sim_proto_weight, sim_batch_weight=sim_batch_weight,
        reg_proto_type='max', reg_batch_type='max', cosine_type='shifted')


m = build()

# --- shapes ---
out1 = m(torch.tensor([0, 4, 9]))
check("1-dim idxs -> (B, K)", out1.shape == (3, K), f"{tuple(out1.shape)}")
idx2 = torch.tensor([[0, 1, 2], [5, 6, 11]])
out2 = m(idx2)
check("2-dim idxs -> (B, n, K)", out2.shape == (2, 3, K), f"{tuple(out2.shape)}")
m.get_and_reset_loss()  # drain accumulators

# --- forward == manual 1 + cos ---
A = m.effective_prototypes()
x = X[idx2]                                                     # (2, 3, V)
manual = 1 + torch.nn.CosineSimilarity(dim=-1)(x.unsqueeze(-2), A)
check("forward == manual 1+cos", torch.equal(out2, manual))
m_std = AttributePrototypeEmbedding(M, AttributeLookup(X), K, offs,
                                    reg_proto_type='max', reg_batch_type='max',
                                    cosine_type='standard')
m_std.prototypes.data.copy_(m.prototypes.data)
check("cosine_type standard == manual cos",
      torch.equal(m_std(idx2), torch.nn.CosineSimilarity(dim=-1)(x.unsqueeze(-2), A)))
m_std.get_and_reset_loss()

# --- base-mode invariants ---
check("use_weight_matrix False, no weight_matrix module",
      m.use_weight_matrix is False and not hasattr(m, 'weight_matrix'))
check("prototypes shape (K, V)", tuple(m.prototypes.shape) == (K, V))
check("embedding_dim == V (prototype space IS attribute space)", m.embedding_dim == V)
proto_before = m.prototypes.detach().clone()
m.init_parameters()
check("init_parameters leaves prototypes untouched (host semantics)",
      torch.equal(m.prototypes.detach(), proto_before))
check("only parameter is the prototypes",
      [tuple(p.shape) for p in m.parameters()] == [(K, V)])

# --- HOST-EQUIVALENCE KEYSTONE ---
host = PrototypeEmbedding(M, embedding_dim=V, n_prototypes=K, use_weight_matrix=False,
                          sim_proto_weight=1.0, sim_batch_weight=1.0,
                          reg_proto_type='max', reg_batch_type='max', cosine_type='shifted',
                          embedding_ext=AttributeLookup(X))
host.prototypes.data.copy_(m.prototypes.data)
idxs = torch.tensor([[0, 3, 7, 11], [2, 5, 8, 1]])
out_sub, out_host = m(idxs), host(idxs)
check("KEYSTONE: outputs bitwise equal to host", torch.equal(out_sub, out_host))
l_sub, l_host = m.get_and_reset_loss(), host.get_and_reset_loss()
check("KEYSTONE: get_and_reset_loss equal to host", torch.equal(l_sub, l_host),
      f"sub={float(l_sub):.6f} host={float(l_host):.6f}")
check("accumulators reset after get_and_reset_loss",
      m._acc_r_proto == 0 and m._acc_r_batch == 0)

# --- claim 3: exact per-attribute decomposition t*-1 = (x·a_k)/(sqrt(F)·||a_k||) ---
t_star = m(torch.arange(M))                                     # (M, K)
m.get_and_reset_loss()
manual_dec = (X @ A.T) / (math.sqrt(F) * A.norm(dim=1))         # (M, K)
err = (t_star - 1 - manual_dec).abs().max()
check("claim 3: exact per-attribute decomposition", float(err) < 1e-5, f"max err {float(err):.2e}")

# --- claim 7(a): R_batch strictly decreases when a row's best prototype aligns more with it ---
batch_idx = torch.arange(M)


def losses(A_data, spw, sbw):
    mm = build(sim_proto_weight=spw, sim_batch_weight=sbw)
    mm.prototypes.data.copy_(A_data)
    mm(batch_idx)
    return mm.get_and_reset_loss()


A0 = m.prototypes.detach().clone()
with torch.no_grad():
    sim = 1 + torch.nn.CosineSimilarity(dim=-1)(X.unsqueeze(-2), A0)   # (M, K)
r = 3
k_star = int(sim[r].argmax())
A_nudged = A0.clone()
A_nudged[k_star] += 0.05 * X[r]
rb0, rb1 = losses(A0, 0.0, 1.0), losses(A_nudged, 0.0, 1.0)
check("claim 7a: R_batch strictly decreases on alignment nudge",
      float(rb1) < float(rb0), f"{float(rb0):.6f} -> {float(rb1):.6f}")

# --- claim 7(b): R_proto strictly decreases when worst-claimed prototype rotates toward an item ---
k_worst = int(sim.max(dim=0).values.argmin())
best_row = int(sim[:, k_worst].argmax())
A_rot = A0.clone()
alpha = 0.3
A_rot[k_worst] = (1 - alpha) * A0[k_worst] + alpha * X[best_row]
rp0, rp1 = losses(A0, 1.0, 0.0), losses(A_rot, 1.0, 0.0)
check("claim 7b: R_proto strictly decreases on rotation toward claiming item",
      float(rp1) < float(rp0), f"{float(rp0):.6f} -> {float(rp1):.6f}")

# --- gradient reaches A through forward ---
m.zero_grad()
out = m(idxs)
(out ** 2).sum().backward()
check("grad reaches prototypes via forward", m.prototypes.grad is not None
      and float(m.prototypes.grad.norm()) > 0)
m.get_and_reset_loss()

# --- field_offsets guard ---
try:
    AttributePrototypeEmbedding(M, AttributeLookup(X), K, [0, 3, 4])
    check("bad field_offsets raise", False)
except AssertionError:
    check("bad field_offsets raise", True)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
