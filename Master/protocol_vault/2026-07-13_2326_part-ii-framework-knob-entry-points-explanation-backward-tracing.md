---
date: 2026-07-13
time: "23:26"
phase: 4
---
# Part II framework (tailored): knob taxonomy by entry point + the explanation-backward tracing procedure

*(Verbatim-capture entry — Matteo thinking the Part II framework out loud during the outline session, protocolled after discussion. Status per Matteo: **"a decent starting point, far from worked out properly."** Tailored to our model by deliberate choice — the generalized "Towards..." framing is "a shoe too big for me"; generality goes to Future Work as "transfers in principle" at most.)*

## Knob taxonomy by entry point (Matteo, near-literal core)

"There is something about the nature of the knob — depending on when they enter the model, or if they are a property of the model itself — that defines how they can be processed."

1. **Outside the model** (architecture/hyperparameters): number of prototypes (dimensionality: probably eliminated as a knob). Integer, non-differentiable, no surrogate → tunable ONLY through hyperopt within an acceptable accuracy window. Maps verbatim onto no-differentiable-surrogate ⇒ Route-1-only ([[2026-07-12_1410_knob-metric-axis-and-principledness-spectrum]]) and the accuracy-tolerance window = Rashomon search ([[2026-06-11_1230_regularizer-tuning-as-rashomon-search]]). Open empirical sub-question (Matteo): is "more prototypes" a *consistent positive direction* for interpretability? The framework should test, not assume.
2. **Measurable properties, affected but not trained for**: prototype sharpness and distinctiveness are metrics; the existing regularizers move them as a side effect while being tuned on accuracy (the blind-tuning mismatch, now as a taxonomy class).
3. **Loss-enterable**: "what has even the slightest hope of entering the training loss — well, it has to be differentiable." The differentiability gate; only what passes becomes Route 2.

## The explanation-backward tracing procedure (Matteo, near-literal, lightly ordered)

Core insight: **"the explanation is a formula"** — the score decomposition — so interpretability constraints have a precise object to attach to.

1. Fix the explanation object (the formula).
2. Define interpretability functions on it — sparsity first: sharpness/distinctiveness improve the explanation *because* "for all intents and purposes it makes it sparser (less relevant components)" — "that is specifically something Rudin mentions". Equivalence (Matteo): sparse neuron activations — the prototype-affinity profile is the activation vector being sparsified. Other constraints from the taxonomy or newly invented join later.
3. **Trace back through the model every component that would increase this** interpretability function.
4. For each identified component: evaluate whether it can be regularized so that it improves the desired function — **"either guaranteed or probably"** (= the principledness spectrum doing real work).
5. If yes: test whether it can enter the training objective (Route 2); else it stays a Route-1 selection criterion.

## Reality checks recorded

- **Compound loss is not a leap — it is the ground:** the host already trains rec-loss + λ₁·reg_proto + λ₂·reg_batch (`sim_proto_weight`/`sim_batch_weight` exist in the codebase). Adding an interpretability term is mechanically routine.
- The genuinely nontrivial parts (Matteo: "I don't know how easy dual-objective/compound training loss really is" — the honest unknowns): (a) weight selection → the Rashomon-ε protocol; (b) gradient conflict between objectives; (c) the surrogate gap — a differentiable proxy may move without the true metric moving. All three are tractable as experiments where the theory is hard.

## Relation to the promised deliverable

Refines the three-step Part II framework (define interpretable output → identify impacting knobs → how/to what extent they tune) into an executable pipeline; this is the pre-supervisor-meeting milestone that defends Block II ([[2026-07-13_2134_thesis-sentence-interpretability-graded-quantify-optimize]] is its one-sentence licence).

[[phase-4]] [[explanations]] [[decisions]] [[thesis-writing]]
