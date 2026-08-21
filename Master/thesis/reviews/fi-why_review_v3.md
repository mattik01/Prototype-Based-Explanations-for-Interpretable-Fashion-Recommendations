# Review v3 — Section 5.1 `fi-why`

**Date:** 2026-08-21 · **Skill:** /thesis-review (third invocation; frame = Matteo's argument-mechanism directive of this session) · **Git state:** HEAD 9d8bdd4, draft uncommitted on `feat/scrutiny`
**New frame (Matteo, this session):** the grounding/vocabulary material must open the section, before the asymmetry — vocabulary = constraint/context on latent factors (that constraining IS grounding); prototypes = structural anchors of the embedding space (which is why THEY are what needs grounding); grounded anchors are self-explanatory, making the score inherently explainable through the two-ingredients lens; and the framing must be forcing, not offering (side information is not added as a benefit — the model is forced to use it, that is what makes it intrinsically explainable). Only then the asymmetry. Intro lead-in sentence: disliked, parked for later.
*LLM-usage-disclosure artifact — do not delete.*

## Phase 1 — mechanical fixes applied

1. "lense" → "lens" (regressed spelling)
2. double space "scale.  The" → single
3. Source rewrapped to the 78-char width (Matteo's standing request; layout-only, placeholders/blank lines preserved)

## Phase 2 — findings

### V3-1 — BLOCKING · red line: argument order inverted vs the agreed mechanism
The definitional block ("Let it again be made clear that when we talk about vocabulary… explain themselves.") sits at the END, but everything before it depends on it. Missing entirely: (a) the forcing move (constraint vs offered side-information → intrinsic explainability); (b) the payoff chain (grounded anchors self-explanatory → score inherently explainable through the two ingredients). "prototypes act as anchor points, which add some structure to the embedding spaces" is too weak to carry the payoff — the anchor role must be strong enough that self-explanatory anchors make the *score* explainable. **Fix direction — proposed five-move architecture for part one (before the asymmetry):**
  A. ProtoMF's structure: users/items represented via similarity to prototypes — prototypes are the structural anchors of the space; every score routes through them.
  B. The problem: all of it learned from the purely relational collaborative signal → no human-readable meaning anywhere, anchors included.
  C. Grounding, defined: constrain the latent factors to a chosen vocabulary — the model is FORCED to build its internals from interpretable ingredients, not handed side information as an optional extra.
  D. Payoff: so-constrained prototypes explain themselves; because they anchor the space and the score is taste·content, the score becomes inherently explainable through the two-ingredients lens.
  E. Hand-off: the vocabulary must come from somewhere → what do the two ingredients each offer? → asymmetry part (current second paragraph, largely reusable as-is).

### V3-2 — MAJOR · grammar/clarity: pivot sentence broken mid-comparison
> "The two ingredients lens in contrast, is far more intuitive than, and will therefore inform how the prototypes should be grounded."
Comparison object missing after "than" (agreed referent: the collaborative signal); comma placement off; "lens" still coined cold. Under the V3-1 restructure this sentence likely becomes the E hand-off and gets rewritten anyway.

### V3-3 — MAJOR (c/f) · grammar: "but it as an unlabeled vector" still missing its verb
Matteo's word choice pending (drop "it" / "does so as" / "arrives as").

### V3-4 — PARKED · chapter lead-in sentence disliked (Matteo) — redo later, after the section's body stabilizes.

### V3-5 — OPEN DECISION · "anchor" terminology (c/f V2-6, updated)
Matteo now uses "structural anchors" as a load-bearing concept. The collision with the ACF baseline's "anchors" (Barkan et al.) remains real since the thesis teaches and runs ACF. Options: keep "structural anchors" + one disambiguating clause where ACF is taught; or a synonym here. Needs Matteo's call.

### Minors (all c/f): "User Attributes" capitals + might/often double hedge · missing koren/samuelson cites · "Item-Characteristics-Groups"/"User-Taste-Groups" coinages · transition sentence granularity + missing \ref{ch:fu} · "adapted as interpretable vocabulary" wording · closing definitional block will dissolve into move C/D under V3-1.

## Phase 3 — discussion outcomes
- Five-move architecture proposed (above); sample formulation of part one to be produced on explicit request.
