---
date: 2026-07-12
time: "18:15"
phase: 4
tags: [thesis/fufi-merge, thesis/hidden-effects]
placed: 2026-08-20
---
# The intercept on the UI host: present, one live + one inert — and in fUfI it becomes feature-attributable ("check which side is free")

*(dc05 review sitting, follow-up to [[2026-07-12_1808_bt-implicit-item-intercept-shift-asymmetry-directives]].
User question: "Is this effect present in UI-ProtoMF / is it a risk in any
merged candidate we might come up with?" Answer worked through and captured;
the fUfI-facing parts also live in the scratchpad merge-baseline entry.)*

## Present in UI-ProtoMF — one live, one inert

UI-score = u*ᵀt̂ + ûᵀt* (A1 Eq. 12). Each shifted vector contributes its
constant: the **U-half's 1ᵀt̂ is a per-ITEM scalar → live** (rank-active; the
model reaches it as (Wᵗᵀ1)·t, a learned linear functional of the item
embedding — it controls both the dial and the direction it reads from); the
**I-half's 1ᵀû is a per-USER scalar → rank-inert** for item ranking. So the
published UI model already carries exactly one popularity-capable implicit
intercept, through the half whose explanation machinery never separated it.
No flag removes it (it is structural, not the `use_bias` scalar).

## The fUfI twist: the live intercept decomposes over feature words

In the merge the projected item is feature-composed, t̂ = Wᵗ(Σ_f e_f + e_ID),
so the live intercept splits **exactly**:

> 1ᵀt̂ = Σ_f (Wᵗᵀ1)·e_f + (Wᵗᵀ1)·e_ID

— per-feature-word bias contributions plus an item-specific residual. Two
readings worth keeping verbatim: (a) this is **LightFM's composed bias
b_i = Σ_f b_f re-emerging implicitly** — the very channel dc01's deviation 4
dropped in its explicit form; (b) it is one concrete realization of the
length-refund prediction that "popularity can live in ID-row length through
the linear half" (the component along Wᵗᵀ1). Renderer consequence for the
merge stage: the baseline line becomes zoomable ("item baseline +1.9: of
which department-word +1.2, colour-word +0.3, item-specific +0.4") — class-
level popularity gets *named per feature word*, only the residual stays
opaque. The channel converts from pure risk into attributable content.

## Risk verdict + the general rule

Any merged candidate on the UI host inherits the live channel by
construction. By the merge stage we will have: the measured magnitude on both
single-sided hosts (dc05's primary B(t) instruments), the renderer's baseline
line as standing practice, and the per-word attribution above. **General
diagnostic rule for every future candidate's hidden-effects pre-audit:
wherever a shifted-cosine side is scored against free per-item coefficients,
a live item intercept exists — check which side is free.**

[[phase-4]] [[explanations]] [[decisions]] [[experiments]]
