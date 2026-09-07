# L2 — Grounding the item PROTOTYPES directly, item embeddings stay free CF

Locus: the dual of dc01 / dc04's region. Constraint to satisfy before anything else: the
explanation object must be **identifiable by construction** (dc04's death: the profile
a_k is an arbitrary preimage of p_k = A·C, ~V−d−1 ≈ 469-dim null space per prototype).

## 0. The identifiability diagnosis (why dc04 failed, what any fix must do)

Non-identifiability arose because the *rendered* object (a_k ∈ Δ^{V−1}) and the *scored*
object (p_k ∈ R^d) were linked by a non-injective map. Two ways out, and only two:
(i) make the rendered object **discrete** (a choice from a finite, named set — a
class/word/lattice node), or (ii) make the rendered object **be** the scored object
(no profile at all: the prototype *is* a named point). Everything continuous and
soft over V anchors in d<V dims stays dead.

## 1. Chosen candidate: **Word-Anchored Prototypes (wI-ProtoMF)** — "the vocabulary IS the prototype set"

**One idea.** Delete the free prototype table. Take the user-perceivable attribute
values (R7 fields only: colour, pattern/graphical appearance, garment/product group,
index/section group — call the value set 𝒲, K := |𝒲| ≈ 60–150) and let **prototype
w be the centred centroid of the items that carry word w**, computed from the *live,
free* item table:

    c_w   = (1/n_w) Σ_{i : w ∈ x_i} t_i                         (centroid of w-items)
    μ     = (1/M) Σ_i t_i                                        (global mean)
    p_w   = c_w − μ                                              (L2.1, centred anchor = prototype)
    t*_iw = 1 + cos(t_i − μ, p_w)                                (host Eq. 7 form; shifted cosine)
    ŷ_ui  = Σ_w u_w · t*_iw                                      (I-ProtoMF score, u ∈ R^K free)

Learned: free item table T ∈ R^{M×d} (host-identical, full capacity), free user table
U ∈ R^{N×K}. **No** prototype parameters, **no** profile, **no** new loss. Buffer vs
live: compute c_w, μ *differentiably* every step — it is one sparse (K×M)·(M×d) product
(M≈27k, F≈4 nnz/row → ~10^5 nnz), negligible (S5: the only new per-step cost; the
in-batch regularisers R_proto/R_batch act on the same K-column sim matrix, unchanged).
A detached, epoch-refreshed buffer (dc04-style) is the cheaper fallback — the
live form is preferred because it *is* the captured-dc04 REGULARIZE route with a
read-out: gradient flows score → t* → p_w → every same-word item, pulling
attribute-similar items toward similar prototype profiles (S7 §4.1.3 "gravity"),
and the read-out exists by definition.

**Cold item** (no trained row): t_cold := μ + (1/F) Σ_{w∈x_cold} p_w (dc04.4 reading:
"where an item with these words sits"); its t* follows; R5 with zero extra parameters.
Biases: host `use_bias` → cold bias = mean bias of its words (dc04 amendment A3).

### Intrinsic read-out (all exact, all identifiable)
1. **Prototype meaning: definitional.** Prototype w *is* "the typical Dark-Blue item's
   collaborative position." Nothing to name, nothing to fit. Seed-stability of the
   *meaning* is trivial (it is a label); what varies across seeds is geometry, which is
   exactly what is measured, not read.
2. **Per-prototype contribution:** u_w·(t*_iw − 1); Σ_w u_w is rank-inert on the I host
   (fI finding carries over verbatim). Rendered: "you like Dark-Blue (+0.4); this item
   resembles the typical Dark-Blue item (cos 0.61) → +0.24."
3. **Per-feature share — deliberately absent and honest.** t*_iw is a cosine between a
   free vector and one word-anchor; there is nothing below it to decompose. The
   behaviour-vs-catalogue tension is rendered as a *two-column* line per word:
   `tagged: ✓/✗ | behaves-like: 0.61` — the discrepancy is a feature (mis-tag /
   latent-property probe, scratchpad 2026-07-11), not a contradiction: "behaves like"
   is precisely what a ProtoMF prototype ever meant (resemblance to a typical item).
4. **User profile:** u ∈ R^K is a *taste vector over words* — the user-side
   explanation comes free and in the item vocabulary (fU's goal, cheaper; but additive:
   "dark AND denim" conjunction is expressed only through items that are high on both).

### Hidden effects (named, checkable)
- **Centring is load-bearing.** Toy (numpy, d=64, M=5000, K=30, 80% init-noise rows;
  *toy*): raw centroids share the global mean → cos(t_i,c_w) ≈ 0.66 for *every* word
  (std 0.06, own-word 0.68 — no discrimination); centred → mean 0.00, std 0.13, own-word
  0.18 at moderate structure, 0.01 at *no* attribute structure. Hence (L2.1) centres;
  and the null case is visibly null rather than faked — an honesty property.
- **Popularity in centroids.** Unweighted mean over a table where most rows are
  near-init: noise rows average to ≈0 direction, so p_w's *direction* is set by the
  trained (popular) w-items — "typical" = "typical among the bought". State it; the
  count-weighted vs interaction-weighted centroid is a Rashomon-tunable knob.
- **Baseline mass** Σ_w u_w: rank-inert (I host). No live item intercept (no free
  per-item coefficient meets a shifted cosine — the "check which side is free" rule).
- **Coupled dynamics (EM-like).** Items chase centroids that are their own means. With
  live centroids this is a single differentiable objective (no refresh discontinuity —
  dc04's pt.7 oscillation risk disappears); the remaining risk is *over-clustering*
  (same-word items collapsing onto p_w) — counter-forces are the rec loss and the
  host's batch entropy regulariser; diagnostic: within-word vs between-word spread.
- **Anchor confounding** (dept ↔ garment group near-identical centroids) is harmless
  here: confounded prototypes are *redundant*, not *arbitrary* (coordinates, not a fit).
- **Low-support words** (n_w < ~20): single-item pointers → exemplar prototypes by
  accident; merge into an "other" word per field (build-time, human-performable).

### How aggressive vs dc01 — and does "attribute group" richness survive?
Item side: **zero constraint** — the table is host-verbatim; the grounding lives
entirely in what a prototype is. dc01 constrains the item *representation* (sum of
words); wI constrains the item *coordinate system*. Param delta: −K·d (prototypes),
user table K instead of host K. Richness: a prototype is **one word, not a group** —
the conjunction ("dark · denim · menswear") is pushed up to the user vector and to the
item's t* *pattern* across words. That is a real loss of ProtoMF's "taste-group"
prototype and the candidate's honest price; the richer sibling that keeps groups is
the lattice variant below.

### R1–R8 (no cross-candidate ranking)
- R1 tied/not bolt-on — **strong**: prototypes are the score's coordinate axes; no parallel path.
- R2 intrinsic — **strong**: every rendered number is a forward-pass quantity; meaning is definitional.
- R3 prototype paradigm — **partial**: prototype-shaped score and read-out intact; prototype = word, not learned group.
- R4 feature-grounded, *learned* association — **partial/strong**: association is by construction (explicit, identifiable, no preimage); what is learned is each item's alignment to it (dc04-A1's critique "explicit but not learned" applies — and is the price of identifiability).
- R5 cold — **strong**: word-centroid fallback, zero params, native to the same reading.
- R6 dense accuracy — **open bet**: full item capacity kept; expressivity lost = prototype freedom; the K=|𝒲| restriction vs host's tuned K is the risk; live-centroid gravity may help or hurt.
- R7 perceivable vocabulary — **strong**: 𝒲 is *chosen* perceivable; internal taxonomy simply not a prototype.
- R8 parsimony — **strong**: one substitution (prototype table → word centroids), no new loss, one knob (field set), optional weighting.

## 2. Rejected siblings (one line each)
- **(b) Signature-push** (free prototypes, ProtoPNet-style periodic push onto the nearest
  restricted-signature centroid; C1 §"projection of prototypes", verified in PDF): keeps
  group richness and is discrete/identifiable, but CF clusters ≠ attribute classes → push
  is a big jump (unlike ProtoPNet's same-class patch), needs refit of u after push, and
  the artefact is ≈ post-hoc nearest-class naming of the host (dc03's kill) — kept as
  the *second* candidate if richness is judged non-negotiable.
- **(a′) Lattice-node prototypes** (prototypes = centroids of attribute-conjunction
  nodes with support ≥ n_min, chosen by support): restores groups, still definitional,
  but node *selection* reintroduces an unlearned, arbitrary design choice and K grows
  combinatorially — variant of the chosen one, not a separate idea.
- **(c) Facet-restricted profiles** (softmax within one field per prototype): identifiable
  only if V_f ≤ d and field centroids full-rank (colour ~50 in d=64: ill-conditioned);
  semantics "0.4 Dark-Blue + 0.3 Black" is a soft word, weaker than a word.
- **(d) dc04 with V ≤ d, centred full-rank anchors**: identifiable in theory, unstable in
  practice (near-collinear correlated values → huge coefficient swings); rejected.
- **(e) Straight-through Gumbel per-field selection**: discrete → identifiable, prototype =
  synthesized conjunction centroid, but ST estimator + temperature schedule + K·F
  categoricals is three tricks (R8), and its fixed point is (a′) by other means.

## 3. Relation to the thesis text
"Grounding = constraining latent factors to a vocabulary" must widen to "constraining
the *coordinate system* (prototypes) to a vocabulary" — items stay free; what is
grounded is the space they are measured in. This is the "prototypes as
re-coordinatization" section made literal, and it closes the dc02 ↔ wI pair neatly:
dc02 = the item in catalogue words (what is *said*), wI = the item in the same words by
*behaviour* (where it sits relative to the typical word-item), cold items degrading to
the former.

Literature verified locally: ProtoPNet push (C1 PDF); ACF anchors = entities as convex
combinations over a spanning anchor set (A2 PDF abstract — the *reverse* direction:
entities from anchors; wI keeps entities free and anchors from entities). SAGE / MMF /
Kim 2014 / TCAV: not in local PDFs — [from memory — verify] before citing.
