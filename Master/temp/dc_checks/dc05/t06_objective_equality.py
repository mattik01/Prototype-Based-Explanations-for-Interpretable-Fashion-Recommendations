# dc05 t06 — A1 TERM-BY-TERM OBJECTIVE EQUALITY at the zero-word point (vault
# 2026-07-12_1739, check obligation 2 — the part of A1 that was argued, not yet pinned).
# The ids-arm dominance theorem's assumption A1 (ignorability) claims: the fU parameter space
# contains a point reproducing the host EXACTLY — all V word rows at ZERO, ID block = host user
# table — and at that point EVERY term of the training objective takes the host's value:
#   (1) recommendation loss (identical scores),
#   (2) BOTH inclusion regularizers (identical activation geometry),
#   (3) L2 mass (zero rows contribute exactly zero to Σ‖θ‖²),
#   (4) max_norm feasibility (zero rows trivially satisfy any cap; renorm is a no-op).
# Unlike t05 (V=0, rows REMOVED), here the full word vocabulary EXISTS and is set to zero —
# the actual A1 point of the shipped model class.
# Run: python Master/temp/dc_checks/dc05/t06_objective_equality.py
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from _harness import (make_check, feature_user_proto_param, user_proto_param, build_recsys,
                      toy_batch, make_toy_history)

check, FAILS = make_check()
torch.manual_seed(0)
N_USERS, N_ITEMS, V, D, KU = 6, 8, 9, 7, 4

ids, w = make_toy_history(N_USERS, V, d_max=4, seed=5)   # REAL baskets — weights nonzero

# A1 compares "same hyperparameters" — the cap (when set) applies to BOTH models. The cap is
# generous (feasible for the host's own rows), so the theorem's feasibility claim — the A1
# point violates no constraint the host satisfies — is what (4) pins.
for max_norm, tag in [(None, "no_cap"), (10.0, "max_norm")]:
    A = build_recsys(feature_user_proto_param(D, KU, ids, w, V, use_id_feature=True,
                                              max_norm=max_norm),
                     N_USERS, N_ITEMS, use_bias=0)
    B = build_recsys(user_proto_param(D, KU, max_norm=max_norm), N_USERS, N_ITEMS, use_bias=0)
    A_up, B_up = A.user_feature_extractor, B.user_feature_extractor

    # Map A to the A1 point: word rows := 0, ID block := host user table, rest copied.
    with torch.no_grad():
        A_up.embedding_ext.embedding_layer.weight[:V] = 0.0
        A_up.embedding_ext.embedding_layer.weight[V:].copy_(
            B_up.embedding_ext.embedding_layer.weight)
        A_up.prototypes.copy_(B_up.prototypes)
        A.item_feature_extractor.embedding_layer.weight.copy_(
            B.item_feature_extractor.embedding_layer.weight)

    u, i, labels = toy_batch(N_USERS, N_ITEMS, B=4, n_neg=2, seed=3)
    logits_A, logits_B = A(u, i), B(u, i)

    # (1) recommendation loss — scores first (composition adds exact zeros: w̄_j·0 = 0)
    check(f"t06[{tag}].scores_bit_identical", torch.equal(logits_A, logits_B),
          f"max |Δ| {(logits_A - logits_B).abs().max().item():.3e}")
    lA = A.loss_func(logits_A, labels).item()
    lB = B.loss_func(logits_B, labels).item()
    check(f"t06[{tag}].rec_loss_equal", lA == lB, f"|Δ| {abs(lA - lB):.3e}")

    # (2) both inclusion regularizers — read the RAW accumulators before reset so the two
    # terms (proto-side, batch-side) are pinned separately, not only their weighted sum
    rpA, rbA = float(A_up._acc_r_proto), float(A_up._acc_r_batch)
    rpB, rbB = float(B_up._acc_r_proto), float(B_up._acc_r_batch)
    check(f"t06[{tag}].reg_proto_term_equal", rpA == rpB, f"|Δ| {abs(rpA - rpB):.3e}")
    check(f"t06[{tag}].reg_batch_term_equal", rbA == rbB, f"|Δ| {abs(rbA - rbB):.3e}")
    A_up.get_and_reset_loss(); B_up.get_and_reset_loss()

    # (3) L2 mass: Σ‖θ‖² over ALL parameters — zero word rows contribute exactly zero
    l2A = sum(float((p ** 2).sum()) for p in A.parameters())
    l2B = sum(float((p ** 2).sum()) for p in B.parameters())
    check(f"t06[{tag}].l2_mass_equal", l2A == l2B, f"|Δ| {abs(l2A - l2B):.3e}")

    # (4) max_norm feasibility: zero rows have norm 0 ≤ cap — the renorm-on-lookup is a no-op
    if max_norm is not None:
        before = A_up.embedding_ext.embedding_layer.weight.detach().clone()
        _ = A(u, i)   # lookups trigger max_norm renorm in place
        A_up.get_and_reset_loss()
        after = A_up.embedding_ext.embedding_layer.weight.detach()
        check(f"t06[{tag}].max_norm_noop_at_A1_point", torch.equal(before, after),
              f"max |Δ| {(before - after).abs().max().item():.3e}")

print("\nt06: ALL PASS (A1 term-by-term)" if not FAILS else f"\nt06: {len(FAILS)} FAILED -> {FAILS}")
sys.exit(1 if FAILS else 0)
