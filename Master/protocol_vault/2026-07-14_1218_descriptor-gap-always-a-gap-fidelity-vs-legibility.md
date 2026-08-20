---
date: 2026-07-14
time: "12:18"
phase: 4
tags: [thesis/knobs-metrics, thesis/interpretability-tuning, thesis/background]
placed: 2026-08-20
---
# The descriptor gap: there is always a gap between descriptor and element — grounding shrinks and audits it, never closes it (+ the fidelity/legibility split of the metrics)

*(Verbatim-capture entry — Matteo's conceptual claim from the 2026-07-14 outline session, rigorously double-checked in discussion before protocoling (his request). Slots into: the naming problem (§1.2), The Gap (Ch 2), the knobs-and-metrics taxonomy, and the tuning method. Working term: **"descriptor gap"** — naming still open; must NOT collide with "The Gap" (the literature gap) in prose.)*

## The thought (Matteo, near-literal)

"I've come to understand that there will always be a gap, in a way. Take our backtracing method: it decomposes mathematics — for now sticking to linear parts, for simplicity — down to atomic components (vectors or values) that are not decomposable anymore. They can be from the dataset, or somehow latent through training. These components then have to be assigned a **descriptor**, so each is usable as a measurable contribution that can be explained. (This slots into a *textual* explanation taxonomy — there are others, like visual explanations (PPMs); we should make that clear and cite S2 for it honestly.) **The gap is always between descriptor and element** — especially if the element is latent — and the work is about making it as small as possible. Our learned word embeddings for every feature value are also latent, right — but we need to argue that *that* gap is smaller than a latent prototype vector with post-hoc labeling."

Included in this: part of the motivation for **not claiming generality of the method** — the backtracing part is quite tailored to linear models and textual explanations; we should admit that, make it clear, and leave the task of generalizing it to somebody else.

## The check (worked through; three sharpenings that must survive into prose)

1. **Scope to learned components.** In a hand-built symbolic rule ("score += 2 if red") descriptor and element coincide — gap ≈ 0. The claim holds *for learned components*: a descriptor is static human vocabulary; a learned element is a point in a training-shaped geometry whose content includes everything that correlates with its role. Since the whole model family is learned, the scoped claim covers everything in the thesis — but state the scoping, or a reviewer produces the symbolic counterexample in one line. Evidence that even *anchored* rows gap: e_red absorbs popularity and correlated-attribute signal (token-mass coupling, share identifiability — the hidden-effects material). **Anchored identity ≠ controlled content.**
2. **Atomicity = descriptor-atomicity, not mathematical atomicity.** Math decomposes deeper than description: u·q splits per coordinate, but no coordinate can bear a descriptor ("axes meaningless, directions meaningful"). The method's atoms are the smallest units that still carry a separable contribution AND can bear a descriptor. That mismatch — decomposition bottoms out below description — is *why* the gap exists, so the definition makes the claim self-supporting.
3. **The dominance argument (why our words beat post-hoc-labeled prototypes), rigorous form — direction of fit:**
   - e_red's descriptor is assigned *before training, by construction*: its **extension** (where it participates) is guaranteed by the forward pass — the row fires exactly for red items. Only the **semantic content** can drift. Anchored gap = semantic drift only.
   - A post-hoc label is inferred *after training from usage statistics*: extension AND semantics are estimates, and different reasonable summarization choices give different names (researcher degrees of freedom — the same pathology as post-hoc attribution, [[2026-07-12_1200_linearity-interpretability-additive-attribution]]). Post-hoc gap = semantic drift + extensional uncertainty + naming-procedure variance.
   - **Auditability is the categorical difference:** the anchored gap can be probed against ground truth (catalogue says which items are red — mis-tag probe; ID residual as honesty meter); the post-hoc gap has no fixed point to audit against. A measurable gap and an unmeasurable one differ in kind, not just size.
   - Honest thesis claim: **we do not close the gap — we shrink it, bound what kinds of drift it can contain, and make it measurable.** This is also the inoculation against "but e_red is just a latent vector too."

## Matteo's pushback and the resolution: NOT all metrics are gap measurements

Claude initially claimed all Part II metrics are "gap measurements at different loci"; Matteo didn't buy it, and the strong claim **fails**: profile sharpness/entropy is orthogonal to the gap by construction (a 100-term perfectly-descriptored explanation has zero gap and is unusable; a sharp 3-term explanation can be mislabeled throughout). Same for separation (mostly legibility; gap-adjacent only via descriptor distinguishability) and model-size sparsity. What survives is a **two-family split of the metric taxonomy**:

- **Fidelity metrics** — measure the descriptor gap: *is the explanation true?* Feature-explained fraction (coverage: ID mass = content with no descriptor at all), purity (extension fit: residual heterogeneity = irreducible mismatch), determinacy/share identifiability (is the descriptor's claimed contribution even well-posed?), cosine-readout reliability (Steck), seed-stability (descriptor drift variance).
- **Legibility metrics** — measure explanation complexity: *is it simple enough to hold?* Sharpness/entropy, separation, model-size sparsity.

"Explained well" = small gap AND legible — the window/room image predicted the split (an honestly-windowed room can be messy; a tidy room can have a lying window). Matches the faithfulness-vs-comprehensibility axis in the XAI literature (citable, not invented). **Consequence for the taxonomy: every metric row gets a fidelity/legibility tag** (some rows both).

## Modality and generality boundaries (deliberate)

- Descriptor assignment as framed here is specific to **textual** explanation; PPMs use *ostensive* descriptors (pointing at an exemplar patch — "this looks like that") rather than symbolic naming; post-hoc centroid naming is an ostensive-to-symbolic conversion, part of why it is lossy. Cite S2 for the explanation-modality taxonomy, honestly.
- The backtracing method's linearity restriction is **forced, not lazy**: atoms carrying separable contributions only exist under additivity (nonlinearity destroys "a part's contribution" as a concept — the five-step chain). Phrase the non-generality as a theorem-shaped boundary: the method's central object does not exist elsewhere without imported approximations (Shapley et al. — the machinery the thesis runs from). Generalizing it is explicitly left to future work / somebody else.

[[explanations]] [[decisions]] [[thesis-writing]] [[phase-4]]
