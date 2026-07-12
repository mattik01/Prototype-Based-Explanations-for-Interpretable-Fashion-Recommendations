---
date: 2026-07-12
time: "17:39"
phase: 4
---
# The ids-arm dominance theorem — thesis placement + the meticulous-check directive

*(Verbatim-capture entry — dc05 concept-review sitting. User directive: "this
mathematical argumentation should find its way into the thesis and be
meticulously checked." Placement per user: (a) the ID-vs-no-ID discussion,
and (b) the framing "what does the extra interpretability cost us, if
anything?" — companion to [[2026-07-12_1208_giving-up-predictive-power-sparsity-flips-the-trade]].
The user also fixed how the assumptions must be stated; formulated properly
below.)*

## The theorem (informal statement, to be re-derived rigorously at writing time)

Fix a host (U-ProtoMF or I-ProtoMF) and its ids-arm feature variant (fU or fI
with the ID row kept), with the same hyperparameters and the same full
training objective (recommendation loss + L2 + inclusion regularizers, under
the same max_norm caps). Two assumptions, exactly as the user framed them:

- **(A1) Ignorability.** The variant's parameter space contains a point that
  reproduces the host exactly — all word vectors at zero — and at that point
  **every term of the objective takes the host's value**: zero rows contribute
  zero L2, the activation geometry (hence both inclusion regularizers) is
  identical, and the max_norm caps are trivially satisfied. (The function part
  is the verified keystone reduction; the term-by-term objective equality is a
  cheap explicit check owed at build time.)
- **(A2) Training adequacy.** "Training is good" in precisely this sense: the
  optimizer finds a solution at least as good (in objective value) as the
  ignore-all-new-information point. Global optimality is NOT needed — it
  suffices that training would not do worse than the solution where all new
  info is ignored.

**Conclusion:** the trained ids variant's training-objective value is ≤ the
host's optimum. *Imposing the interpretable structure with the ID row kept
costs nothing in expressiveness and provably nothing on the training
objective.*

**The honest boundary (must always accompany the theorem):** nothing
transfers to held-out ranking metrics. Generalization is outside the
theorem's reach — worse test performance stays mathematically feasible for
every arm; that is exactly why R6 remains an empirical bet and why the
like-for-like host gaps are *measured*, never assumed. A large ids-below-host
gap is however diagnostically loud: the model had the host solution available
at no objective cost, so a regression must come from generalization through
the new parameters (or finite model-selection noise) — a huntable mechanism.

**No-ID corollary (reversed inequality):** the noid arm's function class is
genuinely restricted (achievable user/item embedding matrices confined to
{W̄·E}, ≤ V·d degrees of freedom; basket/signature twins forced identical), so
on the training objective the HOST weakly dominates the noid arm. On held-out
metrics the sign is open in both directions — the sparsity-flip
([[2026-07-12_1208_giving-up-predictive-power-sparsity-flips-the-trade]]) is
the empirical possibility, never a guarantee.

## Check obligations (the "meticulously checked" directive, itemized)

1. Keystone bit-identity (function part of A1): verified at toy scale for both
   dc01 (C5/i02′) and dc05 (4b C5′); re-verified on the real build.
2. **Term-by-term objective equality at the zero-word point** (L2, both
   inclusion regularizers, max_norm feasibility): explicit dc_checks test at
   the dc05 build — this is the part of A1 currently argued, not yet pinned.
3. The theorem statement itself is re-derived with full rigor at thesis
   writing time (standing rule: scrutiny reasoning is working evidence; thesis
   claims get dedicated treatment).

[[phase-4]] [[decisions]] [[explanations]] [[evaluation]]
