---
date: 2026-08-20
time: "12:18"
phase: 4
---
# Prototype collapse tracks per-item mobility, not feature presence — the ID row as the engine of the self-built bias column

Worked through at the dc01 SC.8 gate sitting (verbatim-capture entry — gate
discussion the user drove to resolution). Context: the SC.8 geometry
instruments had just landed (`sc8_geometry_readout.py`, all four hm arms),
showing fI (ids) collapsed (effective rank 1.22/76, 6 distinct prototype
names of 76, 54× "Swimwear…") while fI-noid stayed differentiated
(effective rank 1.93/99, near-orthogonal prototypes, 92 distinct names of
99). Two user realizations set it up, then an apparent paradox forced the
mechanism into the open.

## Step 1 — the ID row is absolutely necessary (the "cool print" realization)

The canonical 5-field vocabulary is coarse: 67.5% of the catalog shares its
complete attribute signature with at least one other item. Two "Black ·
Solid · Sweater · Knitwear · Womens Everyday" items are identical to the
feature channel — but one has the print/cut/drape people love and the other
doesn't. The actual design is visible in the photo, not in the five fields.
The ID row is the only parameter where the model can store it. Drop it
(noid) and signature twins are structurally indistinguishable forever (twin
q-cosine exactly 1.0 — measured, pinned). So the ID row is the **residual
channel**: everything real about the item that the vocabulary misses.
Popularity is a large component of what it learns (D3/D4 instruments), but
not the only one — and the two are entangled by causation: a print becomes
popular partly *because* people like it. Within-signature demand
differences are simultaneously "popularity" (exposure, feedback loops) and
"genuine appeal the features cannot see" — the §3.6 rule's component (2),
the honest grounding cost, asserting itself. D4 separates head from tail;
it cannot fully disentangle within-class appeal from within-class exposure.
Forward pointer: the "cool print" is literally visible in the garment photo
— an image channel (S3, dc03 salvage) would move part of the ID row's
content into a grounded feature. The residual channel is exactly what
images would shrink.

## Step 2 — two channels hide inside "item bias", and this fleet has only one

- The literal item-bias **scalar** — item-specific, user-UNIVERSAL (same
  additive amount for every user) — is switched OFF fleet-wide
  (`use_bias=0`, charter C3).
- What fI has is the ID **row** — item-specific but user-DIFFERENTIATED.
  It cannot add anything to the score directly; the only thing it can do is
  **steer the item's composed vector inside prototype space**. "Cool print"
  gets encoded as extra affinity toward certain prototypes that the
  features alone would not justify — the item is nudged toward whatever
  direction the people-who-buy-cool-prints cluster loads on. The invisible
  attribute is expressed *through the prototype vocabulary*, because that
  vocabulary is the only language the score speaks. Different users weight
  those prototypes differently (u_k), so this channel is personalized.
- And when an item has **universal** appeal (raw popularity) with the bias
  scalar off? There is no dedicated column for it — so the architecture
  manufactures one: one dominant direction in prototype space that ALL
  users load on positively. Alignment with that direction × everyone's
  positive weight ≈ a per-item scalar added to every score. **The collapsed
  direction IS a self-built item-bias column**, reconstructed inside the
  geometry because the honest one was disabled. This makes the S0-era
  finding ("prototype collapse is a self-built popularity bias", vault
  2026-07-15_1112) mechanistic and quantitative: fI users put 62.5% of
  their energy on direction 0; the personalized taste structure lives in
  directions 1–3.

## Step 3 — the paradox: if the collapse is the self-built bias column, why does noid stay clean?

noid still encodes class-level popularity (its D4 still reads head 0.55 vs
tail 0.24) — so why is ITS space differentiated? Resolution: **the collapse
is not caused by demand signal existing — it is caused by items being
MOBILE enough to chase it.**

- For the item cloud to contract onto a universal-appeal direction, each
  item must be able to move individually: popularity is a per-item quantity
  (item A popular, its signature-twin B not), so encoding it through
  direction-only geometry means A rotates toward the demand direction and B
  does not. That requires a per-item degree of freedom. **The ID row IS
  that degree of freedom.** Give every item its own vector and the
  optimizer rotates each toward/away from the demand direction in
  proportion to its popularity — the cloud contracts into a cone, and once
  the cloud is a cone the coverage regularizer has nothing to spread
  prototypes over (the F-DC01-05 spread-mediation mechanism), so the
  prototypes pile onto the cone too. Collapse.
- **noid items are pinned.** q_i is a sum of five shared class vectors — an
  item sits wherever its attribute combination puts it. Popularity can only
  be encoded at class level (a popular section's embedding grows toward the
  demand direction a bit — hence the D4 head advantage), but item A cannot
  leave its twin B behind, and the combinatorics of thousands of distinct
  signatures keep the cloud spread. Measured: random-pair direction cosine
  0.40 for fI (concentrated cone) vs 0.11 for noid (spread cloud). With a
  spread cloud the coverage force works as designed → near-orthogonal
  prototypes, 92 names.
- **The clinching cross-check: the host collapses too, with no features at
  all.** `item_proto` and f0 — free per-item embeddings, maximum mobility —
  are the MOST collapsed (effective rank 1.14, prototype pairwise cos 0.46,
  "Shirt, Shirts, Contemporary Casual" wallpapered across the cards). So
  across all four arms, collapse severity tracks **per-item mobility, not
  feature presence**: free-ID ≈ features+ID collapsed; features-only
  differentiated. It was never "features save the space" — it is
  "anchoring prevents the demand-chase."

## Step 4 — the honest flip side, and where it points

noid's pretty geometry is **incapacity, not virtue**: the space stays
differentiated because the accuracy-winning move — concentrating on
per-item demand — is inexpressible in it. That is the same 0.076 warm HR@10
gap seen from the geometry side. Which is why this all points at the parked
**C3 bias-on ablation**: the interesting configuration gives per-item
demand a *dedicated* channel (the bias scalar) so the ID row and the
prototype space might stop being colonized by it — accuracy of ids,
geometry closer to noid. Today's spectrum numbers are the sharpest trigger
evidence yet for that ablation, and it is precisely the bias-on × cosine
2×2 the August sessions instrumented (vault 2026-08-17_1937: base settings
stay (shifted, bias-off) for comparability; the ablation remains a parked
user decision, now with mechanism-level motivation).

Working evidence tier throughout (dev profile, single seed, scrutiny
working basis — no thesis quotes; thesis treatment gets dedicated
experiments designed at writing time).

[[phase-4]] [[explanations]] [[experiments]]
