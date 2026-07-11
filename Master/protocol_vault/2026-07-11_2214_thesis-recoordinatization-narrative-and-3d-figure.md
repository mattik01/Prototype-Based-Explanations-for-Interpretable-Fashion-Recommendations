---
date: 2026-07-11
time: "22:14"
phase: 4
---
# Thesis narrative + figure: the re-coordinatization argument, queued after grounding, with a 3D emergence visualization

*(Rewritten same day under the verbatim-capture rule — carries the gate explanations close to literally. Builds on [[2026-07-11_2201_latent-directions-meaning-prototype-recoordinatization]]. From the dc01 SC.1b gate discussion.)*

## Narrative placement (user insight — it sharpens the argument)

The re-coordinatization reasoning is motivation for prototypes, but queue it in the thesis AFTER introducing ProtoMF and AFTER showing how our approach grounds prototypes. Why (user's words, kept): before grounding, prototype coordinates are just another grid — "prototype 1, prototype 2, prototype 3" is exactly as arbitrary as x, y, z. The re-coordinatization alone doesn't finish the job; **only once prototypes have meaning does the transformation truly transform** — ProtoMF builds the new coordinate system, grounding turns its axes from labels into meanings. Candidate chapter/section: "Prototypes — a deeper understanding" (or similar).

## The committed figure (d=3, K=3 — learned or hand-constructed, doesn't matter)

- **Left panel:** ~a dozen items as points in a 3D latent space, colored by category, with the three prototype arrows among them. Axis labels deliberately bland ("latent dim 1/2/3") — visibly arbitrary scaffolding.
- **Right panel:** *the same items, same colors*, re-plotted at coordinates (sim to p₁, sim to p₂, sim to p₃) — and now each axis carries a name ("dark menswear denim…"). The cluster structure visibly survives the mapping, but is now readable off the axes.
- ~~Rotated-copy detail (same left panel rotated → identical right panel)~~ — *discarded same gate (user: rotation invariance is obvious, don't spend figure space on it).*
- Disclosed caveat: the map is many-to-one — cosine drops vector length; what's dropped is magnitude, not meaning.
- v2 for the dc01 chapter: draw the feature arrows into the left panel near the prototypes, showing where the axis *names* come from.
- Build at thesis-writing/SC.9 stage, not scrutiny scope.

## Caption argument — why NOT explain the first space's directions directly (three stacked reasons, plain form)

1. **The model designates nothing.** Meaningful directions exist in the first space — but there are infinitely many directions, and the model never commits to *which* ones matter. To name any of them, an analyst must bring an external procedure (PCA, probing classifiers, contrast sets). Whatever you find is a product of *that procedure's* choices, not of the model — post-hoc in the precise sense: the explanation apparatus is bolted on after, with researcher degrees of freedom.
2. **The score doesn't factor through any particular directions (the deepest one).** A raw dot product u·q can be split into "contributions along directions" using *any* basis whatsoever, and every basis gives a *different, equally valid* story about the same score — infinitely many decompositions, none canonical (the same non-identifiability disease as correlated features, afflicting the entire space). The prototype layer repairs exactly this: the architecture itself computes the score as a sum of K designated components (verified claim C4). The decomposition isn't chosen by the analyst — it *is* the computation. That is the real content of R2: **explanation = the computation, versus an explanation imposed *on* the computation.**
3. **No validation surface.** Analyst-found directions shift with the seed, the sample, the probe design — and there is nothing to check the naming against. The prototype decomposition is at least pinned by the model, and with dc01's grounding its names are read from parameters and *validatable* (SC.8 profile-validity check).

**One-line caption:** *the left space contains meaning but offers no canonical way to read it — every reading is imposed from outside; the right space is the model's own reading.*

## "Could we explain d independent directions instead?" (user question — the program exists, and why we don't take that road)

The idea "pick a basis of d independent directions, interpret each, and you've explained the space" is a real research program: PCA (max-variance directions), ICA, factor rotation in psychometrics, disentangled representation learning — and most recently sparse dictionary learning on neural activations, the current mainstream of LLM interpretability. Walking through what it takes exposes four problems, in ascending depth — each answered by prototypes by construction:

1. **Which 64?** Infinitely many valid bases, so you must add a selection criterion (variance, independence, sparsity) — and each criterion is *your* commitment, producing a *different* explanation of the same model. The arbitrariness isn't removed; it's relocated into the method choice.
2. **Orthogonality is a mathematical convenience, not a semantic truth.** A basis forces the directions to be mutually perpendicular; real behavioral contrasts aren't (menswear↔womenswear correlates with department, price). Forcing orthogonality smears each found direction into a mix of several semantic contrasts — the classic factor-analysis complaint: components interpretable only after creative squinting.
3. **Nothing trained those directions to be clean (the deep one).** During training, no force ever encouraged *any* particular direction to correspond to one concept — so why would the directions your procedure finds afterwards happen to be clean? The disentanglement literature's sobering result (the F2 caveat): even *adding* training pressure for axis-alignment doesn't guarantee interpretability. Without it, you're panning for gold in a river no one salted.
4. **Naming and validating still need a second post-hoc step.** Suppose PCA hands you direction #7 — what is it? You correlate it with features (exactly the post-hoc lift-naming machinery we're moving beyond), with no ground truth to validate against, and the direction itself shifts with seed and sample.

**Punchline:** the prototype layer is the same idea — "designate K special directions and explain those" — **moved into training**: the model commits to its K landmarks while it learns; the score is computed *through* them (canonical decomposition); the inclusion regularizers anchor them onto real data; and dc01 makes them nameable from parameters rather than by a bolt-on procedure. The Rudin argument in miniature: rather than interrogating a black box's geometry after the fact, build the commitment in. *Explaining latent directions post-hoc is a real research program with known pathologies; ProtoMF is that program's intrinsic counterpart — and grounding completes it.*

**Fool's-errand verdict:** not a fool's errand in general — when the model is fixed and you cannot retrain (LLM interpretability's situation), post-hoc direction-finding is the *only* option, which is why the field lives with all four pathologies. It becomes a strategic error **when you own the training loop** — doing archaeology on a building you are still constructing. Reviewer inoculation ("why not just run PCA/SAE on the embeddings?"): *we didn't need to dig for directions post-hoc; the architecture designates and names them intrinsically.*

## Wording guards (self-audit — quantifier slippage is where fluent synthesis hides errors)

- "Counterpart" is an **analogy, not an equivalence**: SAE dictionaries are overcomplete and reconstruction-based; prototypes are K score-bearing landmarks. Thesis writes "counterpart," never "the same thing done right."
- "Canonical decomposition" is canonical **given the trained model**, not across retrainings — which K landmarks the model learned is seed-contingent (Rashomon residue, F-DC01-02). The conditional must survive into thesis prose.
- "426 arrows span the entire room" is a genericity claim ("generically/almost surely after noisy training"), not a theorem.

[[decisions]] [[explanations]] [[phase-4]]
