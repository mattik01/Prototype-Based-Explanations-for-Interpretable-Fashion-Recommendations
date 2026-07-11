---
date: 2026-07-11
time: "22:01"
phase: 4
---
# Latent axes are meaningless, directions are meaningful — prototypes as a learned re-coordinatization

*(Rewritten same day under the verbatim-capture rule — carries the gate explanation close to literally so the argument regenerates without the conversation in context. From the dc01 SC.1b gate discussion; user question: do the embedding space's directions carry meaning? Answer: half right — and the half that's wrong explains why prototypes exist at all.)*

## The axes themselves: provably meaningless

Thought experiment: take the trained model — every feature arrow, every ID arrow, every user arrow, every prototype — and rotate *everything* by the same rotation, like turning a globe. What happens to the model's behavior? **Nothing. Literally nothing.** Every quantity the model ever computes is a dot product or cosine between two arrows, and rotating both arrows together preserves all angles and lengths: cos(Ra, Rb) = cos(a, b); the projection matrices absorb the rotation. Every score, every loss, every one of our explanation read-outs comes out identical.

But the *coordinates* — the 64 numbers in each vector — change completely. So the training objective cannot care which numbers land where: a solution and its rotated twin are indistinguishable, and which one you get is decided by random initialization. Consequence: **"dimension 17 means sportiness" is never a claim we can make.** Axis 17 is scaffolding, an accident; re-run with a different seed and "sportiness" — if encoded at all — lies along a completely different diagonal. (This is why the disentanglement literature exists: making individual axes meaningful requires *extra machinery* vanilla MF doesn't have — and even then, disentangled ≠ interpretable, the F2 caveat.) Honesty footnote: the invariance is exact for the *objective*; coordinate-wise optimizers (Adam/Adagrad) break the symmetry a tiny bit in the training dynamics — far too weak to hang meaning on, but a sharp examiner could ask.

## The directions relative to each other: exactly as meaningful as intuition demands

The behavior — 623,000 purchase decisions — absolutely does organize the space. It writes its meaning into the **configuration of the arrows relative to one another**, not into the coordinate grid. Training pins down which arrows cluster, which are far apart, and along which *contrasts* items differ in ways that matter to users. If menswear-vs-womenswear is the strongest behavioral divide in the data (in fashion it is), some direction encodes that contrast strongly — a real, meaningful direction. It's just not an axis; it's a diagonal, discoverable only by comparing arrows to arrows. Same phenomenon as word2vec's king − man + woman ≈ queen: the "gender direction" is real, but it is a *relation between vectors*, not coordinate #23.

Precise statement: **the space is full of meaningful directions, but the model's coordinate system doesn't label any of them. Meaning lives in relations; the grid is arbitrary.**

## Directions vs independent directions (didactic step — seemed obvious, wasn't)

Dimension d limits the number of **mutually independent** directions — at most d perpendicular ones fit. But the number of **distinct directions** is a continuum: in the 2D plane, two perpendicular directions fit at once, yet every angle (0°, 1°, 17.3°, …) is a direction — a full circle of them; in d dimensions, a whole sphere shell. Meaningful contrasts are diagonals, need not be perpendicular to each other, and cannot exceed d in the *independent* sense. Two consequences: (a) 426 feature arrows in ~64 dimensions are *necessarily* massively redundant among themselves (why "fits some combination of features" is vacuous); (b) correlated fields are *forced* to share directions — the geometric root of the per-feature share non-identifiability (F-DC01-06).

## Why this is secretly the design rationale of the whole thesis

Two things fall out, both load-bearing:

1. **Every one of our read-outs is a cosine between arrows** — feature shares, prototype profiles, the ID-residual probe. None quotes a coordinate. Not an aesthetic choice: a read-out that quoted coordinates would be reading initialization noise. All our explanation devices are rotation-invariant — they read exactly the part of the space that carries meaning and are blind to the part that doesn't.

2. **The prototype layer is a learned re-coordinatization.** The activation vector t* = (t*_1, …, t*_K) is the item *re-expressed in a new coordinate system* — one coordinate per prototype — and unlike the original axes, **these axes are meaningful by construction**: each is a learned landmark arrow sitting somewhere deliberate in the behavioral geometry. ProtoMF's whole trick, seen this way: *replace the meaningless latent grid with a learned grid whose axes are things.* And dc01's contribution in the same language: the host makes the new axes *exist* but leaves them unlabeled (post-hoc top-k naming); dc01 makes each axis **nameable from the model's own parameters**, because the landmark arrows now live among vocabulary arrows they can be read against.

**One-sentence thesis framing:** *behavior fills the space with meaning; prototypes give that meaning addresses; features give the addresses names.*

[[paper-understanding]] [[explanations]] [[decisions]] [[phase-4]]
