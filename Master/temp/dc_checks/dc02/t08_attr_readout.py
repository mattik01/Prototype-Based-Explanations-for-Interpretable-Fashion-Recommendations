# t08 — attr_readout: exact shares, exact score decomposition (claim 4 + §3.4 amendment),
# prototype profiles, per-field mass collapse detector; all against the REAL modules' outputs.
import math
import sys

import torch

from _harness import make_check, toy_attr_data, attr_item_proto_param, build_recsys

from utilities.explanations.attr_readout import (activation_from_shares,
                                                 item_discriminating_contributions,
                                                 per_attribute_shares, per_field_mass,
                                                 prototype_attr_profile, score_decomposition,
                                                 user_constant)

torch.manual_seed(41)
check, fails = make_check()

N_u, M, d, Lu, K = 9, 13, 8, 4, 5
X, offs = toy_attr_data(M=M, V_fs=(3, 4, 5), seed=0)
V = X.shape[1]
F = len(offs) - 1
code_to_cv = [(f'f{f}', f'v{j}') for f in range(F) for j in range(offs[f + 1] - offs[f])]

rec = build_recsys(attr_item_proto_param(d, Lu, K, X, offs), N_u, M, use_bias=0)
user_fe, item_fe = rec.user_feature_extractor, rec.item_feature_extractor
item_proto, item_proj = item_fe.model_1, item_fe.model_2
A = item_proto.effective_prototypes().detach()

# --- shares: (F, K), sum EXACTLY to t*-1 vs the module's own forward ---
with torch.no_grad():
    t_star_all = item_proto(torch.arange(M))          # (M, K)
item_proto.get_and_reset_loss()
max_err = 0.0
for i in range(M):
    shares = per_attribute_shares(X[i], A)            # (F, K)
    check_i = (activation_from_shares(shares) - t_star_all[i]).abs().max()
    max_err = max(max_err, float(check_i))
check("shares shape (F, K)", per_attribute_shares(X[0], A).shape == (F, K))
check("claim 3: shares sum EXACTLY to t*-1 (all items, vs module forward)",
      max_err < 1e-6, f"max err {max_err:.2e}")

# --- score decomposition reproduces the RecSys logit (claim 4 + §3.4 amendment) ---
u_idx, i_idx = torch.tensor([3]), torch.tensor([[5]])
with torch.no_grad():
    logit = rec(u_idx, i_idx)[0, 0]
    u_star = user_fe.model_1(u_idx)[0]                # (Lu,)
    u_hat = user_fe.model_2(u_idx)[0]                 # (K,)
    t_hat = item_proj(i_idx)[0, 0]                    # (Lu,)
    t_star = item_proto(i_idx)[0, 0]                  # (K,)
user_fe.get_and_reset_loss()
item_fe.get_and_reset_loss()

dec = score_decomposition(u_star, u_hat, t_hat, t_star)
check("decomposition total == RecSys logit", float((dec['total'] - logit).abs()) < 1e-5,
      f"err {float((dec['total'] - logit).abs()):.2e}")
check("decomposition has L_u + K + 1 addends",
      dec['proj_terms'].shape == (Lu,) and dec['proto_terms_discriminating'].shape == (K,)
      and dec['user_constant'].dim() == 0)
# §3.4 amendment — verify the RENDERED quantities INDEPENDENTLY (not by re-calling the same
# helpers score_decomposition uses internally, which would be tautological): the item-discriminating
# contribution must be exactly û_k·(t*_k−1) and the disclosed constant exactly Σ_k û_k.
expected_proto = u_hat * (t_star - 1.0)
expected_const = u_hat.sum()
check("proto terms are EXACTLY û·(t*-1) (independent recompute)",
      torch.allclose(dec['proto_terms_discriminating'], expected_proto, atol=1e-6),
      f"max dev {float((dec['proto_terms_discriminating'] - expected_proto).abs().max()):.2e}")
check("user constant is EXACTLY Σû (independent recompute)",
      float((dec['user_constant'] - expected_const).abs()) < 1e-6,
      f"dec={float(dec['user_constant']):.4f} vs Σû={float(expected_const):.4f}")
# cross-check the module helpers agree with the independent recompute too (catches drift in EITHER)
check("helpers match independent recompute",
      torch.allclose(item_discriminating_contributions(u_hat, t_star), expected_proto, atol=1e-6)
      and float((user_constant(u_hat) - expected_const).abs()) < 1e-6)
# the user constant is item-INDEPENDENT: same for another item
with torch.no_grad():
    t_star2 = item_proto(torch.tensor([[9]]))[0, 0]
item_proto.get_and_reset_loss()
dec2 = score_decomposition(u_star, u_hat, item_proj(torch.tensor([[9]]))[0, 0].detach(), t_star2)
check("user constant identical across items", torch.equal(dec['user_constant'], dec2['user_constant']))

# --- prototype profiles: top values per field map to the right (field, value) names ---
A_known = torch.zeros(K, V)
A_known[0, offs[0] + 1] = 3.0    # f0 -> v1 strongest
A_known[0, offs[1] + 2] = 2.0    # f1 -> v2 strongest
A_known[0, offs[2] + 0] = 1.0    # f2 -> v0 strongest
A_known[1:, :] = torch.rand(K - 1, V) * 0.1
profiles = prototype_attr_profile(A_known, offs, code_to_cv, top_m=2)
check("profile structure: K prototypes × F fields", len(profiles) == K and len(profiles[0]) == F)
p0 = profiles[0]
check("profile top values correct per field",
      p0[0]['field'] == 'f0' and p0[0]['top'][0] == ('v1', 3.0)
      and p0[1]['top'][0] == ('v2', 2.0) and p0[2]['top'][0] == ('v0', 1.0),
      f"{[(f['field'], f['top'][0]) for f in p0]}")

# --- per-field mass: rows sum to 1; detects constructed single-field collapse ---
mass = per_field_mass(A, offs)
check("per_field_mass shape (K, F), rows sum to 1",
      mass.shape == (K, F) and bool((mass.sum(dim=1) - 1).abs().max() < 1e-6))
A_collapse = torch.zeros(K, V)
A_collapse[:, offs[1]:offs[2]] = torch.rand(K, offs[2] - offs[1]) + 0.5   # all mass in field 1
mass_c = per_field_mass(A_collapse, offs)
check("collapse detector: constructed single-field A -> field share ~1",
      bool((mass_c[:, 1] > 0.999).all()), f"field-1 share min {float(mass_c[:, 1].min()):.4f}")

# --- nonneg mode: read-outs consume EFFECTIVE A ---
rec_n = build_recsys(attr_item_proto_param(d, Lu, K, X, offs, nonneg_prototypes=True), N_u, M)
proto_n = rec_n.item_feature_extractor.model_1
A_eff = proto_n.effective_prototypes().detach()
check("nonneg: effective A strictly positive", bool((A_eff > 0).all()))
with torch.no_grad():
    t_star_n = proto_n(torch.arange(M))
proto_n.get_and_reset_loss()
err_n = max(float((activation_from_shares(per_attribute_shares(X[i], A_eff)) - t_star_n[i]).abs().max())
            for i in range(M))
check("nonneg: shares (from effective A) still sum exactly to t*-1", err_n < 1e-6,
      f"max err {err_n:.2e}")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(0 if not fails else 1)
