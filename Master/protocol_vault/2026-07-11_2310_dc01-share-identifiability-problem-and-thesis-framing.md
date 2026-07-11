---
date: 2026-07-11
time: "23:10"
phase: 4
---
# dc01: when the split between correlated features stops meaning anything — the problem, two remedies, and the delegated math problem

*(Written under the verbatim-capture rule — carries the SC.1b Point-3 gate discussion close to literally; the finding is F-DC01-06, this entry is its plain-language source of truth for the SC.2 amendment and the thesis.)*

## The problem, built up slowly

Every activation itemizes as t*_k − 1 = c_dept + c_ptype + c_section + c_colour + c_pattern (+ c_ID). Each share is exact algebra, verified to machine precision. The problem: **exact arithmetic can still be arbitrary arithmetic.**

The model never sees feature arrows individually — only their **sum** q_i. Now take two feature values that *always occur together* (retail hierarchies nearly do this constantly: department "Trousers Denim" and product type "Jeans" ride the same items almost every time). Since the loss touches them only through their sum, **their gradients are identical in every training step** — both arrows get pushed by exactly the same force, always. Consequence: their *sum* is carefully trained (it is the "denim-trousers-ness" direction the model genuinely uses), but their *difference* never receives a training signal. Under plain SGD the difference stays frozen at whatever random initialization happened to produce (weight decay slowly shrinks it toward zero, adding nothing).

Now look at what the rendered explanation prints: "department +0.45, product type +0.01." That split is determined by… the difference of the two arrows — which is **initialization noise**. The flagship explanation, for perfectly co-occurring values, performs precision about a quantity the model never learned. Run a different seed, get "+0.23 / +0.23." Both exact. Both meaningless *as a split*.

Reality is a softened version: our fields co-occur strongly but not strictly, so the difference does get trained — but only by the rare items where the fields decouple. Weak signal → weakly determined split, high variance. The strength of the problem tracks the strength of the correlation. What is NOT threatened: the activation totals t*_k (the sum is trained), the per-prototype contributions, and the splits *between* nearly-independent families (product-kind vs colour vs pattern) — which is where R7's question ("did the user-visible features earn any of the score?") actually lives.

**Recorded exhibit (committed with the amendment, slots into SC.2's toy re-check):** synthetic data with one perfectly lockstep pair, several inits — the sum is stable across runs while the printed split wanders.

## Remedy A — the read-out quotient (grouped shares), and why it stays intrinsic

Nothing is adjusted, corrected, or recomputed. What changes is **which level of the breakdown is presented as trustworthy** — a receipt with line items and subtotals:

> Before (fake precision): department +0.45 · type +0.01 · section +0.07 · colour +0.15 · pattern +0.04
> After (honest granularity): **product-kind +0.53** · **colour +0.15** · **pattern +0.04** *(per-field detail shown, labeled "split within product-kind is indicative only")*

The subtotal is a **trained, identified quantity** — an exact summand of the actually-computed score, just read at coarser resolution. No reconstruction, no estimation → still intrinsic in the R2 sense. The same discipline applies to prototype *naming*: for lockstep values each arrow is (trained sum ± frozen noise)/2, so within-group profile *ordering* is noise — naming treats group-mates as one concept; cross-group ordering stays meaningful.

## Remedy B — determinacy annotation (threshold-free, the research-side answer)

User principle, adopted into the learnings ledger: **"production decides; research characterizes"** — thresholds reek of arbitrariness and belong to engineering, not research. So instead of deciding groupings by a cutoff:

- **Name the exact object:** for any pair, the split's determinacy is the *loss curvature along the redistribution direction* (shift credit e_a→e_b keeping the sum fixed — would the model's predictions change? If yes, the split is real; if no, it is a gauge freedom, an untrained dial).
- **Measure its cheap proxy honestly:** curvature is generated only by items where the pair decouples, so the pair's **decoupling mass** in the catalog is the first-order proxy — one committed script, counted at field level (presentation) and value level (where the mechanism truly lives; two fields can be weakly associated overall while one specific value pair is in perfect lockstep).
- **Report the continuum, never a verdict:** every rendered split carries its determinacy annotation ("split weakly/well determined"); grouped views are summary visualizations only — no scientific claim ever rests on where a line was drawn, because no line is drawn.
- Seed-to-seed split stability stays the deferred ground-truth measurement (full-profile stage); the annotations are the single-seed substitute.

Empirical correction recorded: whether *section* clusters with department+product_type is an **output of the counting**, not an assumption (the 4.0 analysis placed section in its own context group; the "dept+ptype+section clique" wording in the adversary's critique was its presumption, not our measurement).

## The delegated math problem (user framing for the thesis)

The thesis presents the problem convincingly (mechanism + recorded exhibit), presents both remedies, and states the general resolution **as a math problem for production to solve**: *find a transformation of the value set into a reduced space in which all splits are identifiable up to tolerance, with the map fixed, data-derived, and human-performable.* This is the standard quotient construction — merge the tokens the data cannot tell apart ("Jeans/Trousers-Denim" becomes one token), dividing out precisely the arbitrary directions. Why it does not become post-hoc: applied at read-out it aggregates exact trained summands (Remedy A); applied at build time (strongest variant, noted but not adopted in dc01 scrutiny) the gauge freedom never exists at all. The **human-performability condition** carries the interpretability responsibility: each merged group must admit one coherent name — "Jeans/Trousers-Denim" is legal, "Jeans/Red" is forbidden — so any human could roughly perform the transformation and end-user interpretability (R7) survives the map. The tolerance choice is explicitly production's.

Connection worth keeping: this is the same reading-vs-splitting distinction as the ID-residual discussion ([[2026-07-11_2132_dc01-id-residual-three-roles]]) — the group sum is a *reading*; the within-group split is an unconstrained *attribution of identity*. And the geometric root is in [[2026-07-11_2201_latent-directions-meaning-prototype-recoordinatization]]: correlated fields are *forced* to share directions because only d independent directions fit in the room.

[[decisions]] [[explanations]] [[evaluation]] [[phase-4]]
