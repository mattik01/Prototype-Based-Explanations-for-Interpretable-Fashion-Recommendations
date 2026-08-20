---
date: 2026-07-12
time: "18:08"
phase: 4
tags: [thesis/hidden-effects, thesis/fu-protomf]
placed: 2026-08-20
---
# B(t): the shifted cosine's implicit per-item intercept — mechanism, host asymmetry, verification, and the three thesis directives

*(Verbatim-capture entry — dc05 concept-review sitting, the B(t) deep-dive.
The user drilled to the core ("I want to understand this to the very core"),
verified everything independently, and issued binding directives. This entry
must let a thesis-drafting session regenerate the full argument alone.)*

## The mechanism (the "+1 → intercept" argument, as accepted)

ProtoMF's shifted cosine u* = **1** + c(u) inserts a **constant coordinate
into every user's representation** — an always-on input, exactly like the
intercept column in linear regression. The score is linear in that input:

> S(u,t) = tᵀ(**1** + c(u)) = **tᵀ1** + tᵀc(u) = **B(t)** + personalized.

A free vector's response to a constant input is an intercept — so B(t) = 1ᵀt
is a learned, **implicit per-item intercept (an item-bias channel)**; the
model tunes it by moving t along the all-ones direction, raising an item's
score for every user identically, without engaging the taste communities.
One-liner: **the shift converts an invisible direction into a universal one**
(without the +1, a t-direction unseen by all users' cosine vectors contributes
nothing to any score; with it, the same direction pays 1ᵀv to everyone).

**The asymmetry (the original core — the paper treats U/I as mirrors; under
the shift they are not):** the intercept always belongs to the FREE side's
entity. I-ProtoMF/fI → per-USER intercept (u ᵀ1) → same number added to every
item that user ranks → rank-inert. U-ProtoMF/fU → per-ITEM intercept (tᵀ1) →
differs across the items being ranked → **rank-active**. UI-ProtoMF carries
one of each (live via the u*ᵀt̂ half, inert via ûᵀt*). Consequence for the
fleet: "bias-free" (use_bias=0, charter C3) removes only the explicit scalar;
the structural channel remains on the U and UI hosts — **explicitly bias-free
everywhere; structurally bias-free only on the I host** (per-host footnote
owed wherever the thesis says "bias-free").

## Verification status (unusually strong for a concept-stage claim)

- Toy-verified (dc05 4b C4): decomposition exact to 1e-15; ranking decided by
  B(t) for constructed taste-identical items; Var(B) ≈ K over random items.
- Hand-verified by Matteo: full algebra + every cell of the worked example
  (the pedagogical note: `Master/temp/bt_baseline_demo/bt_baseline_demo.pdf`
  — two users, two items differing only along an all-users-invisible
  direction v with 1ᵀv = 0.9; identical personalized parts; item B wins by
  exactly 0.9 for every user; deleting the +1 → exact ties. Worth preserving
  beyond temp — thesis-figure seed material).
- Independently cross-checked by a second LLM (ChatGPT) against the note AND
  the A1 paper: algebra/arithmetic/orthogonality/asymmetry all confirmed;
  paper sweep found NO indication of author awareness — the shift is
  motivated once ("guarantees that all the values of u* are positive in the
  range of 0 to 2", §3.1), the decomposition tᵀ1 + tᵀc never appears, §5.2
  narrates the score as prototype contributions only, "bias" appears only in
  the societal sense (§5.3).

**Wording rules adopted (reviewer-proofing, from the cross-check):**
(1) say "implicit per-item intercept (an item-bias channel)" — not bare
"bias" (which reviewers read as an explicit parameter); (2) the supportable
claim is *"the paper introduces the shifted cosine for positivity and does
not discuss the induced decomposition nor its intercept reading"* — "the
intercept is unintended" is an inference about intent and is NOT asserted;
(3) for broad audiences prefer "null/unidentifiable direction" over "gauge
direction".

## The three directives (binding)

1. **Thesis — describe:** full treatment in the planned hidden-effects
   section (algorithm first, then this as one of the hidden mechanisms:
   what/when/sign/how measured), including the asymmetry and the per-host
   bias-free footnote. Textbook specimen of the section's reader-challenge
   design ("+1 for positivity" quietly builds an item-bias dial into half the
   model family).
2. **Experiments — measure the magnitude:** B(t)'s variance share of the
   score AND its correlation with item train-popularity, on both fU and the
   host (primary scrutiny-stage metrics, already committed in the dc05 design
   doc; thesis numbers via dedicated experiments at writing time per the
   standing working-evidence rule). If popularity-correlation is high, the
   rendered baseline line can honestly be annotated "largely popularity" in
   thesis text (analogue of the fI ID-residual labeling idea).
3. **Renderer — display it:** B(t) is a deliberate, first-class line of the
   rendered score decomposition (user-confirmed at this sitting) — arithmetic
   honesty category (ON the figures per the SC.5 scope split), worded
   "item baseline — non-personalized, this item's own scalar, same for every
   user", identical for fU and the host, and included in the artifact's
   exactness self-check (B(t) + rendered parts == forward score).

[[phase-4]] [[decisions]] [[explanations]] [[experiments]] [[evaluation]]
