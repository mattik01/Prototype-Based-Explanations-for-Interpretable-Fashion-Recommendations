---
date: 2026-07-12
time: "12:00"
phase: 4
---
# Why linearity ↔ interpretability (and nonlinearity works against it) — the additive-attribution argument

*(Verbatim-capture entry — worked through interactively, flagged by the user as **needing to become a short paragraph in the thesis**. Written so a reader without the conversation can reconstruct the full argument and compress it themselves. The one-paragraph thesis version has NOT been written yet — this is the source material for it.)*

**The question that prompted it:** why, in this project specifically, does linearity relate so much to interpretability, and nonlinearity broadly work directly against it? The answer is more precise than "linear = simple = readable" — it is about what an *explanation* is here.

## The core chain (five steps, plain build-up)

1. **In this project an explanation *is* a decomposition into named parts.** When we say "this recommendation happened because the item is dark-menswear-denim and the user leans that way," we are claiming the score is literally *made of* those pieces — that you can point at a part and say "this much came from here." That sentence is only *true* (not metaphor) if the score is a **sum**: `score = part₁ + part₂ + … + part_K`. A dot product `u·q = Σ u_k q_k` is exactly a sum of terms, and the prototype layer's job is to make each term a *named* term ("similarity to prototype k"). So linearity is not a nice-to-have — it is the one algebraic property that makes "the score has parts" a true statement.

2. **Nonlinearity destroys the very notion of "a part's contribution."** The moment you write `f(a, b)` with `f` nonlinear, the contribution of `a` depends on the current value of `b` — interaction terms appear, and there is no context-free, basis-free answer to "how much did `a` contribute." You can still *force* an answer (Shapley values, gradients, LIME surrogates), but that attribution is now a **choice imposed from outside the model**, and different reasonable choices give different, equally-defensible stories about the same number. That is exactly the pathology the thesis runs from — post-hoc explanation with researcher degrees of freedom, bolted on after the fact. Additivity removes the choice: the decomposition isn't *picked* by an analyst, it **is** the computation.

3. **Linearity keeps a unit's meaning constant across the whole space; nonlinearity makes meaning point-dependent.** Grounding a prototype = reading off a coefficient ("feature word *denim* contributes +0.7 to this axis"). A linear coefficient is **one number, everywhere** — that is why it can carry a stable name you can print, check, and validate. If the feature→factor map were nonlinear, the same feature would mean different things at different points, there'd be no single label to attach and nothing fixed to validate against. *A name is only a name if it doesn't drift.*

4. **Additivity buys modularity, and modularity is what makes the design extend and degrade gracefully.** The whole `user = Σ purchased-item feature-words + Σ attribute rows + ID row` construction only works because addends don't entangle. That single property lets us: attribute **per purchase** (each purchase is its own summand); handle a **cold user** with no special-casing (their history sum is simply empty — "LightFM's cold story one level up"); and drop **images** in later as *one more row type* without disturbing the rest. A nonlinear mixer (e.g. an MLP over concatenated features) would couple all of it — you couldn't zero out one contribution cleanly, couldn't read a per-purchase share, couldn't add a modality without retraining the meaning of everything else. Linearity is why the representation is a set of independent, individually-meaningful pieces rather than a blended soup.

5. **The tension is real and directional.** Nonlinearity is the *engine of accuracy* precisely because it lets features interact — and interaction is the exact thing that makes "which part caused this" unanswerable. So the two genuinely pull in opposite directions; this is not a coincidence to be optimized away but a structural trade the architecture has to navigate.

## Two caveats that must survive into the thesis prose (this is where fluent versions go wrong)

- **(a) Linearity is necessary but NOT sufficient.** A raw dot product can be split along *any* basis — infinitely many "contribution stories," none canonical (the same non-identifiability that plagues correlated features; cf. the re-coordinatization entry [[2026-07-11_2214_thesis-recoordinatization-narrative-and-3d-figure]]). So additivity alone gives you *a* decomposition, not *the* decomposition. What makes it canonical is that **ProtoMF commits to K specific landmarks *during training*** — the model picks the basis for you, in-loop, instead of an analyst picking it afterward. Linearity gives you parts; the designated prototypes give you *these* parts, non-negotiably; grounding (dc01 / fI-ProtoMF) then makes those parts nameable from parameters. All three are required — do not let the thesis imply "linear ⇒ interpretable."

- **(b) Nonlinearity is not banned everywhere — it is *quarantined* off the attribution path.** The rule is not "no nonlinear functions." It is "**the last mile into the score must be additive.**" A CNN reading an image is wildly nonlinear, yet fine *because its output enters the sum as one more row* — the nonlinearity is sealed inside a feature extractor and never touches how contributions combine. Even the cosine similarity ProtoMF already uses is a tolerated nonlinearity, and it was chosen carefully: many-to-one, but what it discards is vector **magnitude, not meaning** (cf. the cosine caveat noted with the 3D figure).

## The discipline in one line (candidate thesis sentence)
> Push nonlinearity **upstream** of the additive combination, keep the combination itself **linear**, and **disclose** whatever the nonlinear step drops.

## Framing anchor (why this matters for the whole thesis)
This is the operational form of the project's Rudin-flavored thesis that *interpretability = the explanation IS the computation, not something imposed on it*. Only additive computation literally has readable parts; nonlinear black boxes force post-hoc explanation with its known pathologies. The architecture's craft is deciding **where** to spend nonlinearity (inside encapsulated modules) so it never contaminates the additive, named surface where the explanation is read. Natural placement: alongside/after the re-coordinatization argument, before or within the fI-ProtoMF grounding chapter.

[[decisions]] [[explanations]] [[phase-4]]
