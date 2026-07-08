# t03 — AttributeProjection: t̂ = W_t·x_i over the SHARED AttributeLookup (dc02 P4).
import sys

import torch

from _harness import make_check, toy_attr_data

from feature_extraction.feature_extractors import AttributeLookup, AttributeProjection

torch.manual_seed(11)
check, fails = make_check()

M, V_fs, Lu = 12, (3, 4, 5), 4
X, offs = toy_attr_data(M=M, V_fs=V_fs, seed=0)
V = X.shape[1]

lookup = AttributeLookup(X)
proj = AttributeProjection(lookup, out_dimension=Lu)

# --- shapes ---
out1 = proj(torch.tensor([0, 5, 11]))
check("1-dim idxs -> (B, Lu)", out1.shape == (3, Lu), f"{tuple(out1.shape)}")
idx2 = torch.tensor([[0, 1], [7, 9], [2, 4]])
out2 = proj(idx2)
check("2-dim idxs -> (B, n, Lu)", out2.shape == (3, 2, Lu), f"{tuple(out2.shape)}")

# --- forward == linear(lookup(idxs)) ---
check("forward == linear(lookup(idxs))",
      torch.equal(out2, proj.linear_layer(lookup(idx2))))

# --- bias-free, single (Lu, V) parameter ---
check("linear bias is None", proj.linear_layer.bias is None)
check("exactly one parameter of shape (Lu, V)",
      [tuple(p.shape) for p in proj.parameters()] == [(Lu, V)])

# --- shared-instance semantics ---
check("lookup instance shared (is)", proj.lookup is lookup)

# --- init touches only the linear; X bitwise unchanged; loss-free ---
X_before = lookup.attr_multi_hot.clone()
w_before = proj.linear_layer.weight.detach().clone()
proj.init_parameters()
check("init_parameters changes the linear", not torch.equal(proj.linear_layer.weight.detach(), w_before))
check("init_parameters leaves X untouched", torch.equal(lookup.attr_multi_hot, X_before))
check("get_and_reset_loss == 0.0", proj.get_and_reset_loss() == 0.0)

# --- gradient reaches the linear, not X ---
out = proj(idx2)
(out ** 2).sum().backward()
check("grad reaches linear", proj.linear_layer.weight.grad is not None
      and float(proj.linear_layer.weight.grad.norm()) > 0)
check("X receives no grad", lookup.attr_multi_hot.grad is None)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
