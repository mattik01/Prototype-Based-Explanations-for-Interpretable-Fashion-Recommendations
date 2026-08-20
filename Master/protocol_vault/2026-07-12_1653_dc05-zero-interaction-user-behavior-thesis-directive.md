---
date: 2026-07-12
time: "16:53"
phase: 4
tags: [thesis/fu-protomf]
placed: 2026-08-20
---
# dc05: zero-interaction users — exact behavior, and the directive to explain it fully in the thesis

*User directive at the dc05 concept review (first review sitting, companion doc + step-by-step walkthrough): the zero-interaction-user case must be **exactly explained** — in the design record and especially, clearly, in the thesis — even though the current split contains no such user. Rationale: this is the user side, where the cold question is structurally hardest and where hand-waving would be most tempting; the honest degenerate behavior is itself thesis content. Captured close to the walkthrough per the verbatim rule.*

## The exact behavior, as worked through

A dc05/fU user's representation is q_u = (average of the attribute words of the
items they bought in the train window) + e_ID(u), their personal ID row.
For a user with **zero interactions** the basket term is empty, and the chain
is:

1. **With the ID row (ids arm):** q_u = e_ID(u) and nothing else. But a user
   with zero train interactions never received a single gradient step — their
   ID row is **untouched random initialization**. "The user is just his ID"
   therefore means "the user is just noise": personalizing on it would be
   meaningless. This is why the design mirrors dc01's cold convention one
   level up: at cold-user scoring the ID row is **dropped** rather than scored
   on init noise (the exact analogue of fI's cold-item ID drop, F-S0-07
   lineage).
2. **Without the ID row (noid arm, or after that drop):** q_u is the zero
   vector → the eps-guarded cosine returns 0 for every prototype → every
   community activation is exactly 1 (u* ≡ 1) → the score collapses to
   S = B(t) = Σ_l t_l, the **item baseline alone**. The model degrades to a
   non-personalized, popularity-shaped ranking: "we know nothing about you,
   here is what is generally scored high." This is not a hand-waved fallback —
   it was verified literally at toy scale (dc05 4b C3a: u* all-ones exact,
   S = Σ t_l to 1e-16).
3. **One purchase later:** the basket exists, q_u is a real composition, and
   no retraining is needed (LightFM's fold-in property, B5 §7.1) — the user
   converts from cold to warm-thin in a single transaction.

Footnote that must accompany any thesis statement: in the current evaluation
split this user **cannot occur** (k-core guarantees ≥3 train purchases per
evaluated user), so the behavior is a verified *property* of the design, not a
measured result; the cold-user testbed remains future scrutiny-stage work.

## The directive (binding for the build + thesis writing)

- The dc05 build documents the zero-interaction path explicitly (drop
  convention + zero-vector fallback), and the thesis text presenting fU
  explains it **exactly and prominently on the user side** — the three-step
  chain above (noise ID → drop → baseline fallback → fold-in recovery), not a
  one-line "handles cold users gracefully" claim.
- The reason it deserves prominence despite being unmeasurable today: it is
  the honest boundary of the user-side sparsity claim (R5) — stating precisely
  where personalization ends and the popularity-shaped baseline begins is part
  of the transparency story, and the B(t) fallback connects it to the U-host's
  baseline-channel finding (same sitting's review discussion).

[[phase-4]] [[decisions]] [[explanations]]
