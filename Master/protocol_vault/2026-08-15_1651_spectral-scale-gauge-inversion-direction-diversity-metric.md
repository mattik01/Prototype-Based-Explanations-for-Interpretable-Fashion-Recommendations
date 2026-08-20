---
date: 2026-08-15
time: "16:51"
phase: 4
---
# F5's spectral-scale metric doesn't port — instrument invariances must match readout invariances; norm becomes meaningful again at the double tie

*(Verbatim-capture entry from the F5 reading session, sibling of [[2026-08-15_1639_pm-purity-port-groundtruth-target-discussion]]. Status per Matteo: **awareness-level, not a commitment** — we are not 100% sure we will use this metric or something like it, but "we possibly need a metric capturing direction diversity at that level" is now a recorded need. The double-tie norm observation below is flagged by Matteo as extremely interesting.)*

## The question

F5 introduces σ_geo (geometric mean of the singular values of the centered embedding matrix) to measure whether a representation has expanded beyond a compressed init ball, explicitly *criticizing effective rank for being scale-invariant* — a tiny isotropic noise ball can look structurally rich to effective rank while being too small to move any logit. Matteo asked: is this relevant in our space at all? We only read out cosine similarities — every prototype could be normalized onto the unit sphere; wouldn't "spread around the ball" be the right measurement instead?

## Answer: correct — and F5's own critique inverts in our space

**The gauge argument (confirmed, not just theoretical).** Our prototypes are read through shifted cosine and nothing else: scaling a prototype changes no output. Norm is pure gauge; the only force on it is weight decay — and the S0.7 collapse measurements showed exactly that (all prototype norms decayed to near-identical values, 0.123–0.135, cv 1.6%; see [[2026-07-15_1112_prototype-collapse-is-a-self-built-popularity-bias]]). Any norm-sensitive metric on our prototypes measures optimizer dynamics, not model geometry.

**The inversion.** F5's effective-rank critique is *space-relative*, not absolute:
- In THEIR space (raw inner-product readout ⟨h, x⟩) absolute scale is behaviorally meaningful → scale-invariant metrics mislead → σ_geo is the fix.
- In OUR space (cosine readout) scale is gauge → scale-*sensitive* metrics (σ_geo) mislead — they would report weight-decay behavior — → scale-invariant spectral measures are exactly right.

**General lesson (the transferable part): a geometry instrument must match the invariances of the model's readout.** Quoting F5's σ_geo argument in our setting would import a correction for a disease we don't have; conversely their critique of effective rank must never be repeated against our normalized-spectrum measurements.

## The direction-diversity instrument menu (candidates, not commitments)

Row-normalize the prototype matrix first (the gauge fix), then:
1. **Effective direction count** — pairwise cosine with duplicate threshold (cos > 0.999). Already the committed SC.8 instrument from the collapse finding; bluntest, most legible ("76 nominal, ~2 effective").
2. **Effective rank / spectral entropy of the normalized matrix** — the smooth, threshold-free version (5 tight clusters → spectrum ≈ rank 5). After normalization the Frobenius norm is pinned at √K, so scale-blindness is no longer a bug. This is the honest σ_geo replacement if a spectral number is ever wanted.
3. **Mean resultant length** (directional statistics) — norm of the average unit vector: ≈1 concentrated, ≈0 spread. Cheapest one-number concentration summary; the uncentered top singular vector alongside it names the shared direction (in the collapse case: the popularity axis).

## The extremely interesting boundary case: at the double tie, norm stops being gauge — and the answer to "which norm?" is ALL of them

In the single-sided hosts (fI on the I host, fU on the U host) the gauge argument is exact: prototypes and the prototype-side entity embedding are cosine-only (norms gauge; for composed embeddings the *summand* row norms still matter for direction, but the total norm doesn't); only the linear-side entity embedding has a meaningful norm.

At the **double tie** (`prototypes_double_tie`, the fUfI merge target) every vector picks up an inner-product job, so **no norm is pure gauge anymore**:
- **Prototype row norms** become meaningful via the second job — each side's prototype matrix is reused as the projection basis of the *other* side's linear half, and in that role prototype row k's norm directly scales coordinate k of the opposite side's linear representation.
- **Entity embedding norms** become meaningful because each side's embedding also feeds its linear half (inner-product read), not only the cosine branch.

Consequences: (a) the instrument choice is **host-dependent** — spread-on-the-ball for single-sided models, and at the merge a scale-aware spectral measure regains behavioral meaning (σ_geo-like measurements stop being nonsense exactly there); (b) this is the same "second job" mechanism the collapse entry hypothesized as the reason the double-tie stayed healthiest — duplicating a prototype there costs representational rank that accuracy *does* see (coheres with the F-DC01-07 host-specificity insight); (c) the host-dependence itself is a finding-shaped observation for the merge chapter (`fufi-merge`): the merge changes not just the model but *which geometry measurements are meaningful on it*.

Scrutiny-evidence-not-thesis applies to anything measured under this entry.

[[explanations]] [[evaluation]] [[decisions]] [[paper-understanding]] [[phase-4]]

> ⚠ **CORRECTION 2026-08-17:** parts of this entry are corrected in [[2026-08-17_1823_correction-double-tie-projection-basis-and-ui-effective-count]] — the double-tie "prototypes as projection basis" mechanism is code- and paper-refuted (prototype norms stay pure gauge at UI; only entity embeddings hold a second job), and the UI-user "~64 effective" figure is unreproducible (measured: 10 canonical / 8 old run).
